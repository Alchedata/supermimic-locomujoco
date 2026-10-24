"""Paper D.1 return-based reset sampling over one-second motion bins."""
import numpy as np

class ReturnCurriculum:
    def __init__(self,ad,n):
        self.frames=ad.bin_frames
        counts=(np.asarray(ad.lengths)+self.frames-1)//self.frames
        self.offset=np.r_[0,np.cumsum(counts[:-1])]
        self.size=int(counts.sum());self.returns=np.zeros(self.size)
        self.visits=np.zeros(self.size,dtype=np.int64)
        self.history=np.ones(self.size)/self.size
        self.probs=np.ones(self.size)/self.size
        # Native dense MimicReward maximum and gamma=.99.
        rf=ad.rf
        self.gmax=float(rf._qpos_w_sum+rf._qvel_w_sum+rf._rpos_w_sum+rf._rquat_w_sum+2*rf._rvel_w_sum)/.01
        self.total=np.zeros(n);self.discount=np.ones(n);self.valid=np.ones(n,bool)

    def begin(self,tid,frame):
        self.start_bin=self.offset[tid]+frame//self.frames

    def observe(self,reward,done,valid,tid,frame):
        self.total+=self.discount*reward;self.discount*=.99;self.valid&=valid
        eligible=done & self.valid
        ids=self.start_bin[eligible]
        sums=np.bincount(ids,weights=self.total[eligible],minlength=self.size)
        counts=np.bincount(ids,minlength=self.size)
        seen=counts>0
        # The paper specifies a running estimate, without an EMA coefficient.
        # Use 0.1 per completed-episode batch; first observations initialize bins.
        means=sums/np.maximum(counts,1)
        self.returns[seen]=np.where(self.visits[seen]==0,means[seen],.9*self.returns[seen]+.1*means[seen])
        self.visits+=counts
        self.total[done]=0.;self.discount[done]=1.;self.valid[done]=True
        self.start_bin[done]=(self.offset[tid]+frame//self.frames)[done]

    def refresh(self):
        logits=-25*self.returns/self.gmax
        current=np.exp(logits-logits.max());current/=current.sum()
        self.history+=current
        self.probs=.5/self.size+.4*current+.1*self.history/self.history.sum()

    def snapshot(self):
        return {k:getattr(self,k) for k in ['returns','visits','history','probs','total','discount','start_bin','valid','gmax']}
