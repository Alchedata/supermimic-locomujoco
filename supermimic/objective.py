"""Reference-relative posture diagnostics.

The candidate reward is retained for diagnostic inspection. The aligned trainer
uses the native reward and termination, not this candidate reward or posture gate.
"""
import numpy as np


def candidate_objective(x, ref, native_reward, nq=26, nv=25, sites=14, xp=np):
    q=x[...,3:7];qr=ref[...,3:7]
    q=q/xp.maximum(xp.linalg.norm(q,axis=-1,keepdims=True),1e-8)
    qr=qr/xp.maximum(xp.linalg.norm(qr,axis=-1,keepdims=True),1e-8)
    rw=xp.sum(q*qr,-1)
    rv=q[...,:1]*qr[...,1:]-qr[...,:1]*q[...,1:]-xp.cross(q[...,1:],qr[...,1:])
    angle=2*xp.arctan2(xp.sqrt(xp.sum(rv*rv,-1)+1e-16),xp.abs(rw))
    height=xp.sqrt((x[...,2]-ref[...,2])**2+1e-8)-1e-4
    start=nq+nv
    dp=(x[...,start:start+sites*3]-ref[...,start:start+sites*3]).reshape((*x.shape[:-1],sites,3))
    distance=xp.mean(xp.sqrt(xp.sum(dp*dp,-1)+1e-8)-1e-4,-1)
    # Global XY is excluded from pose: resets translate the robot to origin,
    # whereas native reference qpos still carries its dataset-global XY.
    nonquat=xp.asarray([0,1,2]+list(range(7,nq)))
    local_nonquat=xp.asarray([2]+list(range(7,nq)))
    raw_pose=.4*xp.exp(-10*(xp.mean((x[...,nonquat]-ref[...,nonquat])**2,-1)+angle))
    local_pose=.4*xp.exp(-10*(xp.mean((x[...,local_nonquat]-ref[...,local_nonquat])**2,-1)+angle))
    velocity_error=xp.mean((x[...,nq:nq+2]-ref[...,nq:nq+2])**2,-1)
    root_velocity_reward=.2*xp.exp(-velocity_error/.5**2)
    gate=xp.exp(-(height/.25)**2-(angle/.7)**2)
    reward=gate*(native_reward-raw_pose+local_pose+root_velocity_reward)-.5*distance-.5*height-.1*angle
    # Do not clip negative rewards. They preserve correction gradients far away.
    bad=(height>.25)|(angle>xp.pi/3)
    return {'reward':reward,'posture_gate':gate,'height_error_m':height,
            'root_angle_error_rad':angle,'site_error_m':distance,
            'bad_posture':bad,'corrected_pose_reward':local_pose,
            'root_velocity_reward':root_velocity_reward}


def reference_failure_step(count, bad_posture, hold_controls=10, xp=np):
    count=xp.where(bad_posture,count+1,0)
    return count,count>=hold_controls
