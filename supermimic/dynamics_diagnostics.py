"""Held-out open-loop dynamics and simulator directional-derivative probes.

Forty fixed starts (one/motion). Future actions held fixed for perturbations.
No simulator derivatives; all real derivatives are central finite differences.
Raw probes are saved so the discrete v0 can later be tested on exactly this data.
"""
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
from evaluate import stratified_keys
from adapter import write_json


def diagnose(ad,ode,world_step,params,act,norm,out,iteration):
    started=time.perf_counter();out=Path(out);out.mkdir(exist_ok=True,parents=True)
    keys=stratified_keys(ad,160)[::4];initial=ad.reset(keys)
    x0,_,_=ad.extract(initial);state=initial;x=x0
    history=jnp.repeat(ad.obs(x)[:,None],10,axis=1)
    actions=[];targets=[];references=[];masks=[];alive=np.ones(40,bool)
    for step in range(32):
        x,tid,frame=ad.extract(state)
        action=act(history,ad.goal(tid,frame,10),state.observation,x)
        state,tr=ad.step(state,action);xn,tn,fn,_,_,done,valid=tr
        actions.append(action);targets.append(xn);references.append(ad.reference(tn,fn))
        masks.append(alive&np.asarray(valid));alive&=~np.asarray(done)
        rx,_,_=ad.extract(state)
        history=jnp.where(done[:,None,None],jnp.repeat(ad.obs(rx)[:,None],10,axis=1),
            jnp.concatenate([history[:,1:],ad.obs(rx)[:,None]],axis=1))
    aa=jnp.stack(actions);yy=jnp.stack(targets);rr=jnp.stack(references);mask=np.stack(masks)
    folder=out/f'iter_{iteration}';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'probe.npz',initial_state=np.asarray(x0),actions=np.asarray(aa),
        targets=np.asarray(yy),references=np.asarray(rr),valid_mask=mask,reset_keys=np.asarray(keys))
    def prediction(w):
        def step(x,a):
            z,_=world_step(w,x,a)
            return z,z
        return jax.lax.scan(step,x0,aa)[1]
    pred=jax.jit(prediction)(params)
    report={'iteration':iteration,'motions':40,'source':'real simulator, fixed development resets',
        'supervision':'diagnostic only; never enters optimizer','prediction':{},'action_derivative':{},
        'gradient_objective':'discounted native reward; fixed future actions; no survival weighting; segments ending within probe excluded'}
    for horizon in [1,8,32]:
        k=horizon-1;px=np.asarray(pred[k]);tx=np.asarray(yy[k]);valid=mask[k]
        finite=np.isfinite(px).all(-1)
        row={'valid_real_trajectories':int(valid.sum()),'nonfinite_predictions':int((valid&~finite).sum())}
        if valid.any() and finite[valid].all():
            err=(px[valid,:ode.physical_dim]-tx[valid,:ode.physical_dim])/np.asarray(norm['std'][:ode.physical_dim])
            canonical=np.asarray(ode.features(jnp.asarray(tx[valid,:ode.physical_dim])))
            n=(len(ad.ids)-1)*3;s=ad.nq+ad.nv
            row.update(physical_normalized_rmse=float(np.sqrt(np.mean(err**2))),
                site_position_error_m=float(np.linalg.norm((px[valid,s:s+n]-canonical[:,s:s+n]).reshape(-1,len(ad.ids)-1,3),axis=-1).mean()))
        report['prediction'][str(horizon)]=row
    lo,hi=map(np.asarray,(ad.env.info.action_space.low,ad.env.info.action_space.high))
    direction=np.random.default_rng(20261024).normal(size=aa[0].shape).astype(np.float32)
    direction/=np.linalg.norm(direction,axis=-1,keepdims=True)
    margin=np.minimum(np.asarray(aa[0])-lo,hi-np.asarray(aa[0]))
    direction=np.where(margin>=.01*np.abs(direction),direction,0.)
    d=jnp.asarray(direction)
    def model_return(w,first,horizon):
        def step(x,k):
            action=jnp.where(k==0,first,aa[k]);x,_=world_step(w,x,action)
            return x,(.99**k)*ad.reward(x,rr[k],action)
        return jax.lax.scan(step,x0,jnp.arange(horizon))[1].sum()
    for horizon in [1,8]:
        gradient=jax.jit(jax.grad(lambda w,first:model_return(w,first,horizon),argnums=1))(params,aa[0])
        predicted=np.asarray(jnp.sum(gradient*d,axis=-1))
        for eps in [.001,.003,.01]:
            values=[];survived=[]
            for sign in [-1,1]:
                state=initial;total=np.zeros(40);alive=np.ones(40,bool)
                for k in range(horizon):
                    action=aa[k]+sign*eps*d if k==0 else aa[k]
                    state,tr=ad.step(state,action);_,_,_,reward,_,done,valid=tr
                    total+=(.99**k)*np.asarray(reward)*alive
                    alive&=~np.asarray(done)&np.asarray(valid)
                values.append(total);survived.append(alive)
            measured=(values[1]-values[0])/(2*eps)
            keep=survived[0]&survived[1]&mask[horizon-1]&(np.linalg.norm(direction,axis=-1)>0)
            finite=np.isfinite(predicted)&np.isfinite(measured)
            row={'surviving_segments':int(keep.sum()),'nonfinite_model_derivatives':int((keep&~finite).sum())}
            if keep.any() and finite[keep].all():
                p=predicted[keep];s=measured[keep];den=np.linalg.norm(p)*np.linalg.norm(s)
                active=np.abs(s)>1e-5
                row.update(directional_cosine=float(p@s/den) if den>1e-12 else None,
                    relative_mae=float(np.mean(np.abs(p-s))/(np.mean(np.abs(s))+1e-6)),
                    sign_agreement=float(np.mean(np.sign(p[active])==np.sign(s[active]))) if active.any() else None)
            report['action_derivative'][f'h{horizon}_eps{eps}']=row
            np.savez_compressed(folder/f'derivative_h{horizon}_eps{eps}.npz',direction=direction,
                model_directional=predicted,simulator_directional=measured,valid=keep)
    report['seconds']=time.perf_counter()-started
    write_json(folder/'report.json',report)
    print('DYNAMICS_DIAGNOSTICS',iteration,report['prediction'],flush=True)
    return report
