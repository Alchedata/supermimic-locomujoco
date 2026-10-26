"""Bounded SuperMimic-style MLP / BPTT baseline port to LocoMuJoCo.

Explicit port differences are in README.md. Real simulator transitions alone fit
the surrogate; policy gradients never differentiate a physics step.
"""
import argparse
import os
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE','0')
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import json
import pickle
import hashlib
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from adapter import Adapter,make_config,make_env,write_json
from models import World,Policy,Critic
from ode_world import AccelerationWorld,StructuredODE
from dynamics_diagnostics import diagnose
from evaluate import evaluate
from numerical_safety import state_valid,history_valid,LIMITS
from curriculum import ReturnCurriculum

OBJECTIVE_VERSION='loco_native_aligned_v0'


class Replay:
    diagnostic_dir=None
    rejected_transitions=0
    sample_rejections=0
    velocity_indexes=np.arange(24,49)
    def __init__(self,capacity,n):
        self.capacity,self.n=capacity,n;self.ptr=0;self.filled=0;self.arr={}
    def add(self,**items):
        host={k:np.array(jax.device_get(v),copy=True) for k,v in items.items()}
        valid=(np.asarray(host.get('valid',np.ones(self.n,dtype=bool)))
               & state_valid(host['x']) & state_valid(host['x1'])
               & history_valid(host['h'],Replay.velocity_indexes))
        valid &= np.isfinite(host['a']).all(axis=-1) & np.isfinite(host['r'])
        rejected=~valid
        if rejected.any():
            Replay.rejected_transitions+=int(rejected.sum())
            path=Path(self.diagnostic_dir)/f'rejected_simulator_transition_{Replay.rejected_transitions}.npz'
            np.savez(path,**{k:v[rejected] for k,v in host.items()})
            with (Path(self.diagnostic_dir)/'numerical_events.jsonl').open('a') as fp:
                fp.write(json.dumps({'rejected_total':Replay.rejected_transitions,
                    'motion_ids':host['tid'][rejected].tolist(),'frames':host['f'][rejected].tolist(),
                    'raw_transition_file':path.name})+'\n')
            # Raw values above remain recoverable. Safe placeholders never become
            # sampled starts or regression targets, even for a finite outlier.
            host['x'][rejected]=0.;host['x'][rejected,3]=1.
            host['x1'][rejected]=host['x'][rejected]
            host['h'][rejected]=0.;host['a'][rejected]=0.;host['r'][rejected]=0.
            host['done'][rejected]=True;host['absorbing'][rejected]=True
        host['valid']=valid
        bad=np.zeros(self.n,dtype=bool)
        for v in host.values():
            if np.issubdtype(v.dtype,np.floating):
                bad|=~np.isfinite(v).reshape(self.n,-1).all(axis=1)
        if bad.any():
            path=Path(self.diagnostic_dir)/'nonfinite_simulator_transition.npz'
            np.savez(path,**{k:v[bad] for k,v in host.items()})
            raise FloatingPointError(f'Nonfinite real transition in {int(bad.sum())} environments; {path}')
        for k,v in host.items():
            if k not in self.arr:self.arr[k]=np.empty((self.capacity,*v.shape),dtype=v.dtype)
            self.arr[k][self.ptr]=v
        self.ptr=(self.ptr+1)%self.capacity;self.filled=min(self.filled+1,self.capacity)
    def sample(self,rng,b,window=8):
        oldest=0 if self.filled<self.capacity else self.ptr
        t=rng.integers(0,self.filled-window+1,size=b)
        e=rng.integers(0,self.n,size=b)
        for attempt in range(100):
            ok=self.arr['valid'][(oldest+t)%self.capacity,e]
            if ok.all():break
            bad=~ok;Replay.sample_rejections+=int(bad.sum())
            t[bad]=rng.integers(0,self.filled-window+1,size=int(bad.sum()))
            e[bad]=rng.integers(0,self.n,size=int(bad.sum()))
        else:raise RuntimeError('Could not sample a valid real-state start; no invalid batch was used')
        idx=(oldest+t[:,None]+np.arange(window)[None])%self.capacity
        result={k:jnp.asarray(v[idx,e[:,None]]) for k,v in self.arr.items() if k!='h'}
        result['h']=jnp.asarray(self.arr['h'][(oldest+t)%self.capacity,e])
        d=np.asarray(result['done'])|~np.asarray(result['valid'])
        result['mask']=jnp.asarray(np.concatenate([np.ones((b,1)),np.cumprod(1-d[:,:-1],axis=1)],axis=1))*result['valid']
        assert np.asarray(result['mask'][:,0]).all()
        assert state_valid(np.asarray(result['x'][:,0])).all()
        assert history_valid(np.asarray(result['h']),Replay.velocity_indexes).all()
        return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--upstream',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--num-envs',type=int,default=256)
    p.add_argument('--iterations',type=int,default=8000);p.add_argument('--max-hours',type=float,default=1.5)
    p.add_argument('--batch',type=int,default=512);p.add_argument('--buffer-steps',type=int,default=512)
    p.add_argument('--wm-batches',type=int,default=10);p.add_argument('--warmup-iters',type=int,default=0)
    p.add_argument('--eval-every',type=int,default=1000);p.add_argument('--eval-horizon',type=int,default=1000)
    p.add_argument('--eval-envs',type=int,default=32);p.add_argument('--smoke',action='store_true')
    p.add_argument('--imagined-horizon',type=int,default=32)
    p.add_argument('--adapt-iters',type=int,default=0,
                   help='WM and critic adaptation before committing any actor update')
    p.add_argument('--wm-rehearsal-fraction',type=float,default=0.,
                   help='Fraction of each WM batch drawn from independent reset/random training episodes')
    p.add_argument('--resume',help='Model/optimizer continuation with a fresh simulator and replay')
    p.add_argument('--capture-gradient-above',type=float,default=0.,
                   help='Opt-in pre-update actor/critic inputs capture; zero disables')
    p.add_argument('--max-gradient-captures',type=int,default=4)
    p.add_argument('--retain-eval-checkpoints',action='store_true',
                   help='Keep immutable optimizer checkpoints at each evaluation')
    p.add_argument('--retain-every',type=int,default=100,
                   help='Keep immutable optimizer checkpoints at this iteration interval')
    p.add_argument('--actor-batch',type=int,default=4096)
    p.add_argument('--actor-microbatch',type=int,default=1024)
    p.add_argument('--solver',choices=['euler','midpoint'],required=True)
    p.add_argument('--world-width',type=int,default=2158)
    p.add_argument('--saturation-weight',type=float,default=0.)
    p.add_argument('--saturation-threshold',type=float,default=1.)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert a.wm_rehearsal_fraction==0
    assert a.actor_batch%a.actor_microbatch==0
    if (out/'latest.pkl').exists():raise RuntimeError('Existing run: explicit resume implementation is required')
    write_json(out/'args.json',vars(a))
    Replay.diagnostic_dir=out
    previous=None
    if a.resume:
        source=Path(a.resume);payload=source.read_bytes();previous=pickle.loads(payload)
        required={'wp','pp','vp','ow','op','ov','norm','args','key','numpy_rng','controls','policy_updates','iteration'}
        assert required<=previous.keys(),f'Not an optimizer-resume checkpoint: missing {required-previous.keys()}'
        assert previous.get('action_transform')=='tanh_native_bounds','Incompatible action interface'
        assert previous.get('objective_version')==OBJECTIVE_VERSION,'Convert old weights with explicit new-objective metadata'
        assert previous['args']['num_envs']==a.num_envs,'Keep collection batch size unchanged'
        assert previous['args'].get('solver')==a.solver
        assert previous['args'].get('world_width')==a.world_width
        assert a.iterations>previous['iteration'],'iterations is an absolute target'
        write_json(out/'continuation.json',{'parent_checkpoint':str(source),'parent_sha256':hashlib.sha256(payload).hexdigest(),
                   'parent_iteration':previous['iteration'],'parent_controls':previous['controls'],
                   'type':'model_optimizer_rng_continuation_fresh_simulator_and_replay',
                   'refill_vector_steps':64,'normalization':'restored, not recalibrated'})
    cfg=make_config(a.upstream,a.num_envs)
    ad=Adapter(make_env(cfg))
    ode=StructuredODE(ad,a.solver)
    Replay.velocity_indexes=np.flatnonzero((np.asarray(ad.obs_state_indexes)>=ad.nq)
                                          & (np.asarray(ad.obs_state_indexes)<ad.nq+ad.nv))
    write_json(out/'numerical_safety.json',{'limits':LIMITS,'njmax':256,'nconmax':32768,
        'valid_start_sampling':True,'sequence_mask_cuts_on_done_or_invalid':True,
        'invalid_real_transitions':'raw logged, numerical failure, reset, excluded from supervision and starts',
        'imagined_rollouts':'original objective retained; nonfinite updates fail before commit'})
    names=list(cfg.experiment.task_factory.params.lafan1_dataset_conf.dataset_name)
    assert len(names)==len(ad.lengths)==40
    write_json(out/'motion_manifest.json',{'robot':'MjxUnitreeH1','dataset':'LAFAN1',
        'motion_names':names,'frames_per_motion':np.asarray(ad.lengths).tolist(),
        'total_frames':int(np.asarray(ad.lengths).sum()),'control_dt':float(ad.env.dt),
        'sampling':'1s bins: 50% uniform, 40% current return difficulty, 10% historical difficulty',
        'initialization':'from scratch' if not previous else 'continuation',
        'evaluation':'4 fixed random starts per motion; 160 total episodes'})
    ad.motion_names=names
    action_lo=jnp.asarray(ad.env.info.action_space.low)
    action_hi=jnp.asarray(ad.env.info.action_space.high)
    def bounded(raw):return (action_lo+action_hi)/2+(action_hi-action_lo)/2*jnp.tanh(raw)
    rng=np.random.default_rng(a.seed);key=jax.random.PRNGKey(a.seed)
    if previous:
        rng.bit_generator.state=previous['numpy_rng'];key=jnp.asarray(previous['key'])
    def nextkey():
        nonlocal key
        key,k=jax.random.split(key);return k
    curriculum=ReturnCurriculum(ad,a.num_envs)
    if previous:
        for name in ['returns','visits','history','probs']:
            setattr(curriculum,name,np.array(previous['curriculum'][name],copy=True))
    state=ad.train_reset(jax.random.split(nextkey(),a.num_envs),jnp.asarray(curriculum.probs))
    x,tid,f=ad.extract(state)
    hist,K,H,W=10,10,a.imagined_horizon,8
    assert H>0
    h=jnp.repeat(ad.obs(x)[:,None],hist,axis=1)
    replay=Replay(a.buffer_steps,a.num_envs)
    start=time.perf_counter();controls=previous['controls'] if previous else 0
    policy_updates=previous['policy_updates'] if previous else 0
    start_iteration=previous['iteration'] if previous else 0
    write_json(out/'status.json',{'stage':'warmup_collection','iteration':start_iteration,'policy_updates':policy_updates})
    # Continuation refills from the saved policy without changing its parameters.
    refill_pi=Policy(ad.na)
    for _ in range(64):
        center=0.
        if previous:
            pm,ps=map(jnp.asarray,(previous['norm']['mean'],previous['norm']['std']))
            gh=ad.goal(tid,f,K)
            ng=((gh.reshape(a.num_envs,K,ad.gd)-pm[2:ad.observation_state_dim])/ps[2:ad.observation_state_dim]).reshape(gh.shape)
            center=bounded(refill_pi.apply(previous['pp'],(h-ad.obs(pm))/ad.obs(ps),ng))
        action=jnp.clip(center+.1*jax.random.normal(nextkey(),(a.num_envs,ad.na)),action_lo,action_hi)
        ready,tr=ad.train_step(state,action,jnp.asarray(curriculum.probs)) if previous else ad.step(state,action)
        x1,t1,f1,r,absorbing,done,valid=tr
        replay.add(x=x,x1=x1,a=action,tid=tid,f=f,r=r,absorbing=absorbing,done=done,h=h,valid=valid,bad_count=state.bad_count)
        controls+=a.num_envs
        state=ready;x,tid,f=ad.extract(state)
        hn=jnp.concatenate([h[:,1:],ad.obs(x)[:,None]],axis=1)
        h=jnp.where(done[:,None,None],jnp.repeat(ad.obs(x)[:,None],hist,axis=1),hn)
    curriculum.begin(np.asarray(tid),np.asarray(f))
    # Frozen normalization calibrated on real warmup transitions. Avoid moving
    # coordinate systems while validating the first port.
    norm_valid=replay.arr['valid'][:replay.filled].reshape(-1)
    xx=replay.arr['x'][:replay.filled].reshape(-1,ad.sd)[norm_valid]
    dx=(replay.arr['x1'][:replay.filled]-replay.arr['x'][:replay.filled]).reshape(-1,ad.sd)[norm_valid]
    mean=jnp.asarray(xx.mean(0));std=jnp.asarray(np.maximum(xx.std(0),.1))
    delta_mean=jnp.asarray(dx.mean(0));delta_std=jnp.asarray(np.maximum(dx.std(0),.01))
    norm={'mean':mean,'std':std,'delta_mean':delta_mean,'delta_std':delta_std}
    if previous:
        norm=jax.tree.map(jnp.asarray,previous['norm'])
        mean,std,delta_mean,delta_std=(norm[k] for k in ['mean','std','delta_mean','delta_std'])
    def normx(z):return (z-mean)/std
    def normh(z):return (z-ad.obs(mean))/ad.obs(std)
    def normg(g):return ((g.reshape(g.shape[0],K,ad.gd)-mean[2:ad.observation_state_dim])/std[2:ad.observation_state_dim]).reshape(g.shape)
    wm,pi,critic=AccelerationWorld(ad.nv,a.world_width),Policy(ad.na),Critic()
    if previous:wp,pp,vp=(jax.tree.map(jnp.asarray,previous[k]) for k in ['wp','pp','vp'])
    else:
        wp=wm.init(nextkey(),normx(ode.features(x[:1,:ode.physical_dim])),jnp.zeros((1,ad.na)))
        pp=pi.init(nextkey(),normh(h[:1]),normg(ad.goal(tid[:1],f[:1],K)))
        vp=critic.init(nextkey(),normx(x[:1]),normg(ad.goal(tid[:1],f[:1],K)))
    schedule=lambda count,per_iteration:2e-4*(.03+.97*.5*(1+jnp.cos(jnp.pi*jnp.minimum(count//per_iteration,250000)/250000)))
    txw=optax.chain(optax.clip_by_global_norm(1),optax.adam(lambda count:schedule(count,a.wm_batches)))
    txp=optax.chain(optax.clip_by_global_norm(1),optax.clip(.1),optax.adam(lambda count:schedule(count,1),b1=.9,b2=.95))
    txv=optax.chain(optax.clip_by_global_norm(1),optax.adam(lambda count:schedule(count,20)))
    ow,op,ov=txw.init(wp),txp.init(pp),txv.init(vp)
    if previous:ow,op,ov=(jax.tree.map(jnp.asarray,previous[k]) for k in ['ow','op','ov'])
    # Preserve the exact restored inputs for a read-only continuation audit.
    if previous:
        def identical(left,right):
            la,ls=jax.tree.flatten(left);ra,rs=jax.tree.flatten(right)
            return ls==rs and all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(la,ra))
        checks={k:identical(previous[k],v) for k,v in {'wp':wp,'pp':pp,'vp':vp,'ow':ow,'op':op,'ov':ov,'norm':norm}.items()}
        checks['curriculum']=all(np.array_equal(previous['curriculum'][k],getattr(curriculum,k)) for k in ['returns','visits','history','probs'])
        write_json(out/'restore_check.json',checks)
        assert all(checks.values()),checks
    counts={name:sum(z.size for z in jax.tree.leaves(params)) for name,params in [('world',wp),('policy',pp),('critic',vp)]}
    write_json(out/'model.json',{'parameters':counts,'world_hidden':[a.world_width]*3,'solver':a.solver,'physical_state_dim':ode.physical_dim,'world_loss_space':'physical_qpos_qvel_51','analytic_observation_head':True,'policy_hidden':[2048,1024,512,256],
                               'critic_hidden':[256,128],'history':hist,'goal_frames':K,'BPTT':H,'world_window':W,
                               'action_transform':'tanh_native_bounds',
                               'world_input_state_dim':ad.sd,'world_input_action_dim':ad.na,
                               'world_input_features':'canonical differentiable FK, successor reused at next field; midpoint intermediate fresh',
                               'fk_reuse':True,'actor_microbatch':a.actor_microbatch,'gradient_accumulation':'python',
                               'world_output_acceleration_dim':ad.nv})
    def world_step(w,x,a):
        return ode.step(wm,w,x,a,norm)
    def world_loss(w,b):
        def one(x,j):
            active=b['mask'][:,j]>0
            x=jnp.where(active[:,None],x,jax.lax.stop_gradient(b['x'][:,j,:ode.physical_dim]))
            pred,logit=ode.physical_step(wm,w,x,b['a'][:,j],norm)
            err=jnp.mean(((pred-b['x1'][:,j,:ode.physical_dim])/delta_std[:ode.physical_dim])**2,axis=-1)
            term=optax.sigmoid_binary_cross_entropy(logit,b['absorbing'][:,j].astype(jnp.float32))
            return pred,(err,term)
        _,(err,term)=jax.lax.scan(one,b['x'][:,0,:ode.physical_dim],jnp.arange(W))
        mask=b['mask'].T
        return ((err+term)*mask).sum()/jnp.maximum(mask.sum(),1)
    def finite_tree(tree):
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(z)) for z in jax.tree.leaves(tree)]))
    @jax.jit
    def update_world(w,o,b):
        loss,grad=jax.value_and_grad(world_loss)(w,b)
        upd,o=txw.update(grad,o,w)
        candidate=optax.apply_updates(w,upd)
        return candidate,o,loss,optax.global_norm(grad),finite_tree((candidate,o,loss,grad)),optax.global_norm(upd)
    @jax.jit
    def validate_world(w,b):
        weight=b['mask'][:,0]
        def average(error):return (jnp.mean(error,axis=-1)*weight).sum()/jnp.maximum(weight.sum(),1)
        current=b['x'][:,0,:ode.physical_dim];target=b['x1'][:,0,:ode.physical_dim]
        pred,_=ode.physical_step(wm,w,current,b['a'][:,0],norm)
        mse=average(((pred-target)/std[:ode.physical_dim])**2)
        persistence=average(((current-target)/std[:ode.physical_dim])**2)
        swapped,_=ode.physical_step(wm,w,current,jnp.roll(b['a'][:,0],1,axis=0),norm)
        return mse,persistence,average(((swapped-target)/std[:ode.physical_dim])**2)
    @jax.jit
    def actor(p,hh,tt,ff):return bounded(pi.apply(p,normh(hh),normg(ad.goal(tt,ff,K))))
    def value(v,xx,tt,ff,bc):
        cx=normx(xx)
        return critic.apply(v,cx,normg(ad.goal(tt,ff,K)))
    def policy_loss(p,w,v,b):
        xx=b['x'][:,0];tt=b['tid'][:,0];ff=b['f'][:,0];hh=b['h'];bc=b['bad_count'][:,0]
        def one(carry,t):
            xx,hh,ff,survival,total,bc,full=carry
            active=survival>0
            gg=ad.goal(tt,ff,K)
            act=bounded(pi.apply(p,normh(hh),normg(gg)))
            xn,logit=ode.step_reuse(wm,w,full,act,norm)
            fn=ff+1;ref=ad.reference(tt,fn)
            old_bc=bc
            rr=ad.reward(xn,ref,act)
            valid=(fn<ad.lengths[tt]).astype(jnp.float32)
            cont=(fn<ad.lengths[tt]-1).astype(jnp.float32)*jax.nn.sigmoid(-logit)
            vv=value(v,xx,tt,ff,old_bc)
            total=total+.99**t*survival*rr*valid
            hn=jnp.concatenate([hh[:,1:],ad.obs(xn)[:,None]],axis=1)
            return (xn,hn,fn,survival*cont,total,bc,xn),(xx,gg,rr*valid,cont,vv,(ff<ad.lengths[tt]),old_bc)
        (endx,_,endf,survival,total,end_bc,_),tr=jax.lax.scan(one,(xx,hh,ff,jnp.ones(xx.shape[0]),jnp.zeros(xx.shape[0]),bc,ode.features(xx[...,:ode.physical_dim])),jnp.arange(H))
        bootstrap=value(v,endx,tt,endf,end_bc)
        total=total+.99**H*survival*bootstrap
        # Match the MLP Anti-Saturation treatment: raw actions at real replay starts.
        # This term leaves imagined rewards, survival weights and critic targets unchanged.
        raw=pi.apply(p,normh(b['h']),normg(ad.goal(tt,b['f'][:,0],K)))
        penalty=jnp.mean(jax.nn.relu(jnp.abs(raw)-a.saturation_threshold)**2)
        return -total.mean()+a.saturation_weight*penalty,(tr,bootstrap)
    @jax.jit
    def policy_grad(p,w,v,b):
        return jax.value_and_grad(policy_loss,has_aux=True)(p,w,v,b)
    @jax.jit
    def apply_policy(p,o,grad):
        upd,o=txp.update(grad,o,p)
        candidate=optax.apply_updates(p,upd)
        return candidate,o,optax.global_norm(grad),optax.global_norm(upd),finite_tree((candidate,o,grad))
    def update_policy(p,o,w,v,b):
        chunks=a.actor_batch//a.actor_microbatch
        grads=None;loss=0.;traces=[];boots=[]
        for i in range(chunks):
            micro=jax.tree.map(lambda z:z[i*a.actor_microbatch:(i+1)*a.actor_microbatch],b)
            (ll,(tr,boot)),grad=policy_grad(p,w,v,micro)
            grads=grad if grads is None else jax.tree.map(jnp.add,grads,grad)
            loss=loss+ll/chunks;traces.append(tr);boots.append(boot)
        grads=jax.tree.map(lambda g:g/chunks,grads)
        aux=(jax.tree.map(lambda *z:jnp.concatenate(z,axis=1),*traces),jnp.concatenate(boots))
        candidate,o,gn,un,finite=apply_policy(p,o,grads)
        return candidate,o,loss,aux,gn,un,finite & finite_tree((loss,aux))
    @jax.jit
    def critic_targets(aux):
        (xs,gs,rs,cs,vs,mask,bcs),bootstrap=jax.tree.map(jax.lax.stop_gradient,aux)
        nextvs=jnp.concatenate([vs[1:],bootstrap[None]],axis=0)
        def backward(ret,inp):
            r,c,nv=inp;ret=r+.99*c*((1-.95)*nv+.95*ret);return ret,ret
        _,target=jax.lax.scan(backward,bootstrap,(rs,cs,nextvs),reverse=True)
        return normx(xs.reshape(-1,ad.sd)),normg(gs.reshape(-1,K*ad.gd)),target.reshape(-1),mask.reshape(-1).astype(jnp.float32)
    @jax.jit
    def update_critic(v,o,data,index):
        cx,gg,yy,weight=(z[index] for z in data)
        def loss(v):
            residual=jnp.where(weight>0,critic.apply(v,cx,gg)-yy,0.)
            return jnp.sum(residual**2)/jnp.maximum(weight.sum(),1.)
        ll,grad=jax.value_and_grad(loss)(v);upd,o=txv.update(grad,o,v)
        candidate=optax.apply_updates(v,upd)
        return candidate,o,ll,optax.global_norm(grad),optax.global_norm(upd),finite_tree((candidate,o,ll,grad,yy)),jnp.mean(yy),jnp.max(jnp.abs(yy))
    def save(it,stage):
        ck={'wp':wp,'pp':pp,'vp':vp,'ow':ow,'op':op,'ov':ov,'norm':norm,'iteration':it,
            'controls':controls,'policy_updates':policy_updates,'args':vars(a),'key':key,'numpy_rng':rng.bit_generator.state,
            'action_transform':'tanh_native_bounds','objective_version':OBJECTIVE_VERSION,
            'policy_updates_scope':'one actor update per iteration', 'curriculum':curriculum.snapshot()}
        tmp=out/'latest.tmp';tmp.write_bytes(pickle.dumps(jax.device_get(ck)));tmp.replace(out/'latest.pkl')
        if a.retain_every>0 and it%a.retain_every==0:
            dest=out/'checkpoints';dest.mkdir(exist_ok=True)
            target=dest/f'iter_{it}.pkl'
            if not target.exists():os.link(out/'latest.pkl',target)
        write_json(out/'status.json',{'stage':stage,'iteration':it,'controls':controls,'policy_updates':policy_updates,
                                     'elapsed_seconds':time.perf_counter()-start,'scope':'H1 all40 LAFAN1 joint training'})
    anchor_rng=np.random.default_rng(800044)
    anchor_batch_size=0
    # Episode-separated world-model validation is collected by independent resets,
    # not sampled from the training replay. Never enters any optimizer minibatch.
    vs=ad.reset(jax.random.split(jax.random.PRNGKey(900042),a.num_envs))
    vx,vt,vf=ad.extract(vs);vh=jnp.repeat(ad.obs(vx)[:,None],hist,axis=1)
    val=Replay(16,a.num_envs)
    # Reproduce the original fresh-run random-action validation stream even on
    # continuation: 1 reset key + 64 warmup keys + 3 network initialization keys.
    # Its RNG is separate from all training and anchor draws.
    validation_key=jax.random.PRNGKey(a.seed)
    for _ in range(1+64+3):validation_key,_=jax.random.split(validation_key)
    for _ in range(16):
        validation_key,vk=jax.random.split(validation_key)
        aa=jnp.clip(.1*jax.random.normal(vk,(a.num_envs,ad.na)),action_lo,action_hi);bc=vs.bad_count;vs,tr=ad.step(vs,aa)
        xn,tn,fn,rr,ab,dd,valid=tr
        val.add(x=vx,x1=xn,a=aa,tid=vt,f=vf,r=rr,absorbing=ab,done=dd,h=vh,valid=valid,bad_count=bc)
        vx,vt,vf=ad.extract(vs);vh=jnp.where(dd[:,None,None],jnp.repeat(ad.obs(vx)[:,None],hist,axis=1),jnp.concatenate([vh[:,1:],ad.obs(vx)[:,None]],1))
    validation=val.sample(np.random.default_rng(900043),a.batch,W)
    np.savez_compressed(out/'wm_validation_batch.npz',**jax.device_get(validation))
    initial_mse,initial_persist,initial_swapped=map(float,validate_world(wp,validation))
    write_json(out/'wm_validation_initial.json',{'mse':initial_mse,'persistence_mse':initial_persist,
        'swapped_action_mse':initial_swapped,'gate':'diagnostic only; never blocks actor',
        'validation_reset_seed':900042,'training_reset_seed':800042,
        'validation_enters_optimizer':False,'wm_rehearsal_fraction':anchor_batch_size/a.batch,
        'validation_rng':'original fresh-run stream; independent of continuation RNG',
        'training_reset_controls':0})
    with (out/'wm_validation_events.jsonl').open('a') as fp:
        fp.write(json.dumps({'iteration':start_iteration,'stage':'before_repair',
            'mse':initial_mse,'persistence_mse':initial_persist,'swapped_action_mse':initial_swapped})+'\n')
    eval_act=lambda hh,gg,obs,xx:bounded(pi.apply(pp,normh(hh),normg(gg)))
    best_return=-float('inf')
    def remember_best(report,it):
        nonlocal best_return
        if report['mean_return']>best_return:
            best_return=report['mean_return']
            best={'wp':wp,'pp':pp,'vp':vp,'norm':norm,'iteration':it,'controls':controls,
                  'action_transform':'tanh_native_bounds','args':vars(a),'evaluation':report,
                  'scope':'development-selected inference checkpoint, not optimizer resume','objective_version':OBJECTIVE_VERSION}
            tmp=out/'best.tmp';tmp.write_bytes(pickle.dumps(jax.device_get(best)));tmp.replace(out/'best.pkl')
            write_json(out/'best.json',{'iteration':it,'mean_return':best_return,'evaluation_tag':report['tag']})
    save(start_iteration,'initialized')
    if a.retain_eval_checkpoints:
        checkpoints=out/'checkpoints';checkpoints.mkdir(exist_ok=True)
        target=checkpoints/f'iter_{start_iteration}.pkl'
        if not target.exists():os.link(out/'latest.pkl',target)
    if not a.smoke:
        remember_best(evaluate(ad,jax.jit(eval_act),out/'evaluation','initial',n=a.eval_envs,horizon=a.eval_horizon),start_iteration)
    metrics=out/'metrics.jsonl'
    world_ready=False
    gate_blocked_checks=0
    gate_validation_checks=0
    gradient_captures=0
    def capture(it,reason,batch,aux=None,periodic=False,**metadata):
        nonlocal gradient_captures
        keep=periodic or gradient_captures<a.max_gradient_captures
        tag=f'{it}_{reason}'
        path=out/'diagnostics'/tag
        if keep:
            path.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(path/'actor_batch.npz',**jax.device_get(batch))
            snapshot={'wp':wp,'pp':pp,'vp':vp,'op':op,'ov':ov,'ow':ow,'norm':norm,
                      'iteration':it,'args':vars(a),'key':key,'numpy_rng':rng.bit_generator.state,
                      'scope':'exact diagnostic inputs, replay/simulator not serialized'}
            if aux is not None:snapshot['aux']=aux
            (path/'inputs.pkl').write_bytes(pickle.dumps(jax.device_get(snapshot)))
            if not periodic:gradient_captures+=1
        record={'iteration':it,'controls':controls,'reason':reason,'snapshot_saved':keep,
                'path':str(path) if keep else None,**metadata}
        record={k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in record.items()}
        if keep:write_json(path/'metadata.json',record)
        with (out/'diagnostic_events.jsonl').open('a') as fp:fp.write(json.dumps(record)+'\n')
    def fail(it,phase):
        save(it,'failed_nonfinite_'+phase)
        write_json(out/'numerical_failure.json',{'iteration':it,'phase':phase,'candidate_committed':False})
        raise FloatingPointError(f'Nonfinite {phase} update rejected before commit')
    @jax.jit
    def action_metrics(p,b):
        raw=pi.apply(p,normh(b['h']),normg(ad.goal(b['tid'][:,0],b['f'][:,0],K)))
        unit=jnp.tanh(raw)
        derivative=1-unit**2
        penalty=jnp.mean(jax.nn.relu(jnp.abs(raw)-a.saturation_threshold)**2)
        return dict(action_saturation_095=jnp.mean(jnp.abs(unit)>.95),
                    action_saturation_099=jnp.mean(jnp.abs(unit)>.99),
                    raw_action_absmax=jnp.max(jnp.abs(raw)),raw_action_absmean=jnp.mean(jnp.abs(raw)),
                    action_output_derivative_mean=jnp.mean(derivative),
                    action_output_derivative_below_001=jnp.mean(derivative<.01),
                    action_out_of_bounds_fraction=jnp.mean((bounded(raw)<action_lo)|(bounded(raw)>action_hi)),
                    replay_action_out_of_bounds_fraction=jnp.mean((b['a']<action_lo)|(b['a']>action_hi)),
                    saturation_penalty=penalty,saturation_loss=a.saturation_weight*penalty)
    for it in range(start_iteration,a.iterations):
        iteration_start=time.perf_counter()
        controls_before=controls
        # UTD 0.4: alternate 3 and 2 vectorized simulator steps per iteration.
        for _ in range(3 if it%2==0 else 2):
            action=jnp.clip(actor(pp,h,tid,f)+.1*jax.random.normal(nextkey(),(a.num_envs,ad.na)),action_lo,action_hi)
            bc=state.bad_count;state,tr=ad.train_step(state,action,jnp.asarray(curriculum.probs))
            xn,tn,fn,rr,ab,dd,valid=tr
            replay.add(x=x,x1=xn,a=action,tid=tid,f=f,r=rr,absorbing=ab,done=dd,h=h,valid=valid,bad_count=bc)
            controls+=a.num_envs;x,tid,f=ad.extract(state)
            curriculum.observe(np.asarray(rr),np.asarray(dd),np.asarray(valid),np.asarray(tid),np.asarray(f))
            hn=jnp.concatenate([h[:,1:],ad.obs(x)[:,None]],axis=1)
            h=jnp.where(dd[:,None,None],jnp.repeat(ad.obs(x)[:,None],hist,axis=1),hn)
        curriculum.refresh()
        collection_seconds=time.perf_counter()-iteration_start
        for _ in range(a.wm_batches):
            batch=replay.sample(rng,a.batch,W)
            candidate,optimizer_candidate,wl,wgn,world_finite,wun=update_world(wp,ow,batch)
            if not bool(world_finite):
                np.savez(out/'nonfinite_world_batch.npz',**jax.device_get(batch))
                (out/'nonfinite_world_inputs.pkl').write_bytes(pickle.dumps(jax.device_get({'wp':wp,'ow':ow,'norm':norm})))
                write_json(out/'numerical_failure.json',{'iteration':it,'controls':controls,
                    'loss_finite':bool(np.isfinite(float(wl))),'gradient_finite':bool(np.isfinite(float(wgn))),
                    'batch_all_finite':all(np.isfinite(np.asarray(v)).all() for v in batch.values())})
                fail(it,'world')
            wp,ow=candidate,optimizer_candidate
        if it%50==0 or it==start_iteration or it==a.iterations-1:
            mse,persist,swapped=map(float,validate_world(wp,validation))
            world_ready=(mse<persist and mse<swapped)
            anchor_mse=0.
            policy_probe=replay.sample(anchor_rng,a.batch,W)
            policy_mse,policy_persist,policy_swapped=map(float,validate_world(wp,policy_probe))
            gate_validation_checks+=1
            gate_blocked_checks=0 if world_ready else gate_blocked_checks+1
            with (out/'wm_validation_events.jsonl').open('a') as fp:
                fp.write(json.dumps({'iteration':it,'stage':'training','mse':mse,
                    'persistence_mse':persist,'swapped_action_mse':swapped,'world_ready':world_ready,
                    'training_anchor_mse':anchor_mse,'policy_replay_mse':policy_mse,
                    'policy_replay_persistence_mse':policy_persist,
                    'policy_replay_swapped_action_mse':policy_swapped,
                    'consecutive_blocked_checks':gate_blocked_checks})+'\n')
        pl=cl=pgn=pure_imagined=critic_tail=pun=cgn=cun=target_mean=target_absmax=0.
        bootstrap_mean=bootstrap_absmax=imagined_qvel_absmax=imagined_feature_absmax=0.
        if True:
            actor_batch=replay.sample(rng,a.actor_batch,W)
            periodic=(it+1)%a.eval_every==0
            if periodic:capture(it,'pre_actor_periodic',actor_batch,periodic=True)
            actor_committed=True
            if actor_committed:
                candidate_pp,candidate_op,pl,aux,pgn,pun,actor_finite=update_policy(pp,op,wp,vp,actor_batch)
            else:
                pl,aux=frozen_imagine(pp,wp,vp,actor_batch)
                candidate_pp,candidate_op,actor_finite=pp,op,finite_tree((pl,aux))
            if not bool(actor_finite):
                capture(it,'nonfinite_actor',actor_batch,aux=aux,periodic=True)
                fail(it,'actor')
            if a.capture_gradient_above>0 and float(pgn)>=a.capture_gradient_above:
                capture(it,'actor_gradient',actor_batch,aux=aux,raw_gradient_norm=float(pgn),update_l2=float(pun))
            pp,op=candidate_pp,candidate_op
            if it%10==0 or it==a.iterations-1:
                (_,_,rewards,continuations,_,_,_),bootstrap=aux
                survival=jnp.concatenate([jnp.ones_like(continuations[:1]),
                                          jnp.cumprod(continuations,axis=0)[:-1]],axis=0)
                pure_imagined=float(jnp.mean(jnp.sum((.99**jnp.arange(H))[:,None]*survival*rewards,axis=0)))
                critic_tail=float(jnp.mean(.99**H*jnp.prod(continuations,axis=0)*bootstrap))
                bootstrap_mean=float(jnp.mean(bootstrap));bootstrap_absmax=float(jnp.max(jnp.abs(bootstrap)))
                xs=aux[0][0]
                imagined_qvel_absmax=float(jnp.max(jnp.abs(xs[...,ad.nq:ad.nq+ad.nv])))
                imagined_feature_absmax=float(jnp.max(jnp.abs(normx(xs))))
            data=critic_targets(aux)
            critic_steps=0
            for epoch in range(5):
                order=rng.permutation(H*a.actor_batch).reshape(4,-1)
                for index in order:
                    cv,co,cl,gn,un,critic_finite,tm,ta=update_critic(vp,ov,data,jnp.asarray(index))
                    if not bool(critic_finite):
                        capture(it,'nonfinite_critic',actor_batch,aux=aux,periodic=True)
                        fail(it,'critic')
                    vp,ov=cv,co;critic_steps+=1
                    cgn=max(cgn,float(gn));cun=max(cun,float(un))
                    target_mean=float(tm);target_absmax=float(ta)
            assert critic_steps==20
            policy_updates+=int(actor_committed)
        if it%10==0 or it==a.iterations-1:
            row={'iteration':it,'controls':controls,'wm_loss':float(wl),'wm_gradient_norm':float(wgn),
                 'policy_loss':float(pl),'policy_gradient_norm':float(pgn),'critic_loss':float(cl),
                 'wm_validation_mse':mse,'persistence_mse':persist,'swapped_action_mse':swapped,
                 'world_ready':world_ready,'policy_updates':policy_updates,
                 'rejected_real_transitions':Replay.rejected_transitions,'elapsed_seconds':time.perf_counter()-start}
            row.update(learning_rate=float(schedule(it,1)),critic_optimizer_steps=20*(it+1),wm_optimizer_steps=10*(it+1),termination_learning=True,actor_gate_enabled=False,curriculum_min_probability=float(curriculum.probs.min()),curriculum_max_probability=float(curriculum.probs.max()))
            row.update(imagined_reward=pure_imagined,imagined_horizon=H,critic_bootstrap_tail=critic_tail)
            row.update(wm_rehearsal_fraction=anchor_batch_size/a.batch,
                       wm_training_anchor_mse=anchor_mse,wm_policy_replay_mse=policy_mse,
                       gate_consecutive_blocked_checks=gate_blocked_checks)
            row.update(objective_version=OBJECTIVE_VERSION,actor_adapting=it<start_iteration+a.adapt_iters)
            row.update(collection_seconds=collection_seconds,
                       collection_controls=controls-controls_before,
                       collection_controls_per_second=(controls-controls_before)/collection_seconds,
                       iteration_seconds=time.perf_counter()-iteration_start)
            row.update(world_update_l2=float(wun),actor_update_l2=float(pun),
                       actor_global_clip_scale=min(1.,1./max(float(pgn),1e-30)),
                       critic_gradient_norm_max=cgn,critic_update_l2_max=cun,
                       critic_target_mean=target_mean,critic_target_absmax=target_absmax,
                       bootstrap_value_mean=bootstrap_mean,bootstrap_value_absmax=bootstrap_absmax,
                       imagined_qvel_absmax=imagined_qvel_absmax,imagined_normalized_feature_absmax=imagined_feature_absmax,
                       rejected_start_draws=Replay.sample_rejections,
                       parameters_and_optimizer_finite=True)
            row.update({k:float(v) for k,v in action_metrics(pp,actor_batch).items()})
            if not all(np.isfinite(v) for v in row.values() if isinstance(v,float)):raise FloatingPointError(row)
            with metrics.open('a') as fp:fp.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if (it+1)%100==0:save(it+1,'training')
        if not a.smoke and (it+1)%a.eval_every==0:
            if a.retain_eval_checkpoints:
                # save() has just written the matching optimizer checkpoint.
                save(it+1,'training')
                checkpoints=out/'checkpoints';checkpoints.mkdir(exist_ok=True)
                (checkpoints/f'iter_{it+1}.pkl').write_bytes((out/'latest.pkl').read_bytes())
            # Fresh closure avoids a stale cached actor at later checkpoints.
            ep=pp
            act=jax.jit(lambda hh,gg,obs,xx:bounded(pi.apply(ep,normh(hh),normg(gg))))
            remember_best(evaluate(ad,act,out/'evaluation',f'iter_{it+1}',n=a.eval_envs,horizon=a.eval_horizon),it+1)
            diagnose(ad,ode,world_step,wp,act,norm,out/'dynamics_diagnostics',it+1)
        if a.max_hours>0 and time.perf_counter()-start>a.max_hours*3600:
            save(it+1,'time_limit_incomplete')
            raise TimeoutError('Wall-clock limit reached before iteration target; not marked complete')
    save(it+1,'training_complete')
    if a.smoke:
        assert policy_updates>0,'Smoke must exercise actual BPTT and critic updates'
        assert float(pgn)>0,'Actor gradient must be nonzero'
        ep=pp
        probe_act=jax.jit(lambda hh,gg,obs,xx:bounded(pi.apply(ep,normh(hh),normg(gg))))
        diagnose(ad,ode,world_step,wp,probe_act,norm,out/'dynamics_diagnostics',it+1)
    else:
        ep=pp
        remember_best(evaluate(ad,jax.jit(lambda hh,gg,obs,xx:bounded(pi.apply(ep,normh(hh),normg(gg)))),out/'evaluation','final',n=a.eval_envs,horizon=a.eval_horizon),it+1)
    save(it+1,'complete')


if __name__=='__main__':main()
