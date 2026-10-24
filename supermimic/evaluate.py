"""Fixed-reset deterministic real-simulator evaluation for both learners."""
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from adapter import write_json


def stratified_keys(ad, n):
    """Keep four deterministic random starts from every loaded motion."""
    motions=len(ad.lengths)
    assert n == motions*4
    if hasattr(ad,'evaluation_keys'):
        return ad.evaluation_keys
    picked=[[] for _ in range(motions)]
    base=jax.random.PRNGKey(20261022)
    for batch in range(64):
        keys=jax.random.split(jax.random.fold_in(base,batch),256)
        state=ad.reset(keys)
        ids=np.asarray(state.additional_carry.traj_state.traj_no)
        for k,tid in zip(np.asarray(keys),ids):
            if len(picked[int(tid)])<4:picked[int(tid)].append(k)
        if all(len(v)==4 for v in picked):break
    assert all(len(v)==4 for v in picked), [len(v) for v in picked]
    ad.evaluation_keys=jnp.asarray(np.stack([k for group in picked for k in group]))
    return ad.evaluation_keys


def evaluate(ad, act, out, tag, n=32, horizon=1000, hist=10, goal_k=10):
    start=time.perf_counter()
    # These development reset seeds are independent of collection, but references
    # remain training references. This is coverage, not unseen-motion generalization.
    keys=stratified_keys(ad,n)
    state=ad.reset(keys)
    x,tid,f=ad.extract(state)
    initial_tid=np.asarray(tid)
    assert np.array_equal(np.bincount(initial_tid,minlength=len(ad.lengths)),np.full(len(ad.lengths),4))
    initial_qpos=np.asarray(x[...,:ad.nq]);initial_frames=np.asarray(f)
    initial_ref=np.asarray(ad.reference(tid,f))
    last_xy=initial_qpos[:,:2].copy();last_ref_xy=initial_ref[:,:2].copy()
    h=jnp.repeat(ad.obs(x)[:,None],hist,axis=1)
    alive=np.ones(n,bool);returns=np.zeros(n);lengths=np.zeros(n,int)
    valid_lengths=np.zeros(n,int);numerical_failures=np.zeros(n,bool)
    falls=np.zeros(n,bool);native_done=np.zeros(n,bool);poserr=np.zeros(n);velerr=np.zeros(n)
    posture_bad=np.zeros(n);height_error=np.zeros(n);angle_error=np.zeros(n);root_velocity_sq=np.zeros(n)
    last_goal=ad.goal(tid,f,goal_k)
    for step in range(horizon):
        g=ad.goal(tid,f,goal_k)
        actions=act(h,g,state.observation,x)
        state,tr=ad.step(state,actions)
        x1,t1,f1,r,absorbing,done,valid=tr
        numerical_failures|=alive&~np.asarray(valid)
        valid_lengths+=alive&np.asarray(valid)
        x1=jnp.where(valid[:,None],x1,x)
        ref=ad.reference(t1,f1)
        s0=ad.nq+ad.nv;nn=(len(ad.ids)-1)*3
        pe=jnp.linalg.norm((x1[...,s0:s0+nn]-ref[...,s0:s0+nn]).reshape(n,-1,3),axis=-1).mean(-1)
        ve=jnp.mean(jnp.abs(x1[...,ad.nq:ad.nq+ad.nv]-ref[...,ad.nq:ad.nq+ad.nv]),axis=-1)
        task=ad.objective(x1,ref,actions)
        pb,he,ae,rv,px,rx=jax.device_get((task['bad_posture'],task['height_error_m'],task['root_angle_error_rad'],
            jnp.sum((x1[...,ad.nq:ad.nq+2]-ref[...,ad.nq:ad.nq+2])**2,-1),x1[:,:2],ref[:,:2]))
        weight=alive&np.asarray(valid)
        posture_bad+=weight*pb;height_error+=weight*he;angle_error+=weight*ae;root_velocity_sq+=weight*rv
        last_xy=np.where(weight[:,None],px,last_xy);last_ref_xy=np.where(weight[:,None],rx,last_ref_xy)
        rr,dd,aa,pp,vv=jax.device_get((r,done,absorbing,pe,ve))
        returns+=alive*rr;lengths+=alive;poserr+=alive*np.asarray(valid)*pp;velerr+=alive*np.asarray(valid)*vv
        falls|=alive&aa;native_done|=alive&dd;alive&=~dd
        if not alive.any():break
        x,tid,f=ad.extract(state)
        hn=jnp.concatenate([h[:,1:],ad.obs(x)[:,None]],axis=1)
        h=jnp.where(jnp.asarray(dd)[:,None,None],jnp.repeat(ad.obs(x)[:,None],hist,axis=1),hn)
    den=np.maximum(valid_lengths,1)
    task_success=(lengths==horizon)&~falls
    displacement_error=np.linalg.norm((last_xy-initial_qpos[:,:2])-(last_ref_xy-initial_ref[:,:2]),axis=-1)
    rows=[{'episode':i,'motion_id':int(initial_tid[i]),'motion_name':ad.motion_names[int(initial_tid[i])],
           'return':float(returns[i]),'length':int(lengths[i]),'absorbing_failure':bool(falls[i]),
           'numerical_failure':bool(numerical_failures[i]),'valid_error_controls':int(valid_lengths[i]),
           'native_done':bool(native_done[i]),'horizon_censored':bool(alive[i]),
           'mean_relative_site_position_error_m':float(poserr[i]/max(valid_lengths[i],1)),
           'mean_qvel_absolute_error':float(velerr[i]/max(valid_lengths[i],1)),
           'task_success':bool(task_success[i]),'posture_violation_fraction':float(posture_bad[i]/den[i]),
           'mean_root_height_error_m':float(height_error[i]/den[i]),
           'mean_root_angle_error_degrees':float(np.degrees(angle_error[i]/den[i])),
           'root_planar_velocity_rmse_mps':float(np.sqrt(root_velocity_sq[i]/den[i])),
           'root_displacement_error_m':float(displacement_error[i])} for i in range(n)]
    per_motion=[]
    for motion_id,name in enumerate(ad.motion_names):
        take=initial_tid==motion_id
        per_motion.append({'motion_id':motion_id,'motion_name':name,'episodes':int(take.sum()),
            'mean_return':float(returns[take].mean()),'absorbing_failures':int(falls[take].sum()),
            'numerical_failures':int(numerical_failures[take].sum()),
            'completed_horizon_without_failure':int(((lengths[take]==horizon)&~falls[take]).sum()),
            'relative_site_position_error_m':float((poserr[take]/np.maximum(valid_lengths[take],1)).mean()),
            'task_successes':int(task_success[take].sum())})
    summary={'tag':tag,'episodes':n,'motions':len(ad.motion_names),'horizon':horizon,'mean_return':float(returns.mean()),
             'mean_length':float(lengths.mean()),'absorbing_failures':int(falls.sum()),
             'numerical_failures':int(numerical_failures.sum()),
             'reward_per_control':float(returns.sum()/max(lengths.sum(),1)),
             'completed_horizon_without_failure':int(((lengths==horizon)&~falls).sum()),
             'mean_qvel_absolute_error':float(np.mean(velerr/np.maximum(valid_lengths,1))),
             'native_nonabsorbing_ends':int((native_done&~falls).sum()),'horizon_censored':int(alive.sum()),
             'relative_site_position_error_m':float(np.mean(poserr/np.maximum(valid_lengths,1))),
             'task_successes':int(task_success.sum()),'objective_version':'loco_native_aligned_v0',
             'success_definition':'full evaluation horizon without native or numerical termination',
             'mean_root_height_error_m':float((height_error/den).mean()),
             'mean_root_angle_error_degrees':float(np.degrees(angle_error/den).mean()),
             'root_planar_velocity_rmse_mps':float(np.sqrt(root_velocity_sq/den).mean()),
             'root_displacement_error_m':float(displacement_error.mean()),
             'seconds':time.perf_counter()-start,
             'protocol':'all40 training motions, 4 fixed random starts per motion; deterministic policy; first episode only',
             'per_motion':per_motion,'rows':rows}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez(out/f'{tag}_reset_manifest.npz',qpos=initial_qpos,frame=initial_frames,motion_id=initial_tid,keys=np.asarray(keys))
    write_json(out/f'{tag}.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','per_motion')}),flush=True)
    return summary
