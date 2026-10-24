"""LocoMuJoCo H1 interface. Upstream files remain unmodified."""
from pathlib import Path
import json
import os
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from typing import Any
from objective import candidate_objective,reference_failure_step

from omegaconf import OmegaConf
from loco_mujoco import TaskFactory
import loco_mujoco
from numerical_safety import state_valid, LIMITS
from jax_constants_compat import install
if os.environ.get('JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS','').lower()=='true':
    install()
from loco_mujoco.core.utils.math import calculate_relative_site_quatities, quaternion_angular_distance

SITES = ['upper_body_mimic', 'left_hand_mimic', 'left_foot_mimic', 'right_hand_mimic', 'right_foot_mimic']


def make_config(upstream, num_envs=256, steps=32, total=8_388_608):
    # Local-only cache configuration; never mutate the upstream installation.
    variables = os.environ.get('LOCO_VARIABLES_FILE')
    if variables:
        loco_mujoco.PATH_TO_VARIABLES = Path(variables)
    cfg = OmegaConf.load(Path(upstream) / 'examples/training_examples/jax_rl_mimic/conf.yaml')
    e = cfg.experiment
    e.env_params.goal_params.visualize_goal = False
    e.env_params.use_mjwarp = True
    e.env_params.nconmax = 32768
    e.env_params.njmax = 256
    e.num_envs, e.num_steps, e.total_timesteps = num_envs, steps, total
    e.validation.num = 1
    e.validation.active = False
    e.debug = False
    manifest = os.environ.get('LOCO_MOTION_MANIFEST')
    if manifest:
        names = json.loads(Path(manifest).read_text())
        assert len(names) == 40 and len(set(names)) == 40
        e.task_factory.params.lafan1_dataset_conf.dataset_name = names
    return cfg


def make_env(cfg):
    e = cfg.experiment
    factory = TaskFactory.get_factory_cls(e.task_factory.name)
    return factory.make(**e.env_params, **e.task_factory.params)


@struct.dataclass
class TrackingState:
    native: Any
    bad_count: Any
    @property
    def data(self):return self.native.data
    @property
    def additional_carry(self):return self.native.additional_carry
    @property
    def observation(self):return self.native.observation
    @property
    def info(self):return self.native.info


class Adapter:
    """SuperMimic-style joint state/observation surrogate with analytic task reward.

    State includes full qpos/qvel and native relative-site observations. Learning
    those extra observations replaces analytic FK in this initial port; this is
    an explicit difference from the G1 reproduction. No simulator derivatives.
    """
    def __init__(self, env):
        self.env = env
        self.nq, self.nv, self.na = env._model.nq, env._model.nv, env.info.action_space.shape[0]
        self.rf = env._reward_function
        self.ids = env._goal._rel_site_ids
        self.body_ids = env._model.site_bodyid[self.ids]
        self.reward_site_indexes = jnp.asarray([list(self.ids).index(i)-1 for i in self.rf._rel_site_ids[1:]])
        assert self.ids[0] == self.rf._rel_site_ids[0]
        self.root_ids = env._model.body_rootid
        self.splits = jnp.asarray(env.th.traj.data.split_points)
        self.lengths = self.splits[1:] - self.splits[:-1]
        self.traj = env.th.traj.data
        # Current native observation is physical+site features with root XY omitted.
        self.od = (self.nq-2) + self.nv + (len(self.ids)-1)*12
        self.observation_state_dim = self.od + 2
        self.reward_feature_dim = (len(self.rf._rel_site_ids)-1)*12
        self.sd = self.observation_state_dim + self.reward_feature_dim
        self.gd = self.od
        base_dim=(self.nq-2)+self.nv
        self.obs_state_indexes=np.full(base_dim,-1,dtype=np.int32)
        for item in env.obs_container.values():
            if hasattr(item,'data_type') and item.data_type() in ('qpos','qvel'):
                ids=np.asarray(item.obs_ind)
                if ids.size and ids.max()<base_dim:
                    self.obs_state_indexes[ids]=np.asarray(item.data_type_ind)+(self.nq if item.data_type()=='qvel' else 0)
        assert np.all(self.obs_state_indexes>=0),self.obs_state_indexes
        self.obs_state_indexes=jnp.asarray(self.obs_state_indexes)
        assert env.info.observation_space.shape == (self.od+self.gd,), {
            'native_shape':env.info.observation_space.shape,'expected':self.od+self.gd,
            'nq':self.nq,'nv':self.nv,'reward_sites':self.ids.tolist(),
            'goal_sites':env._goal._rel_site_ids.tolist()}
        assert np.array_equal(self.rf._qpos_ind, np.arange(self.nq))
        assert np.array_equal(self.rf._qvel_ind, np.arange(self.nv))
        assert self.nq == self.nv+1
        assert np.array_equal(np.flatnonzero(self.rf._quat_in_qpos), np.arange(3,7))
        for name in ['_joint_acc_coeff','_joint_torque_coeff','_action_rate_coeff']:
            assert getattr(self.rf, name) == 0, name
        # Preserve actual terminal transitions before the native automatic reset.
        self._native_reset_done = env._mjx_reset_in_step
        env._mjx_reset_in_step = lambda state: state
        # Preserve the native benchmark absorbing test and reference/horizon ends.
        self.bin_frames=max(1,round(1./float(env.dt)))
        lengths=np.asarray(self.lengths)
        bins=[(i,f) for i,n in enumerate(lengths) for f in range(0,int(n),self.bin_frames)]
        self.bin_motion=jnp.asarray([b[0] for b in bins]);self.bin_start=jnp.asarray([b[1] for b in bins])
        self.bin_width=jnp.minimum(self.bin_frames,self.lengths[self.bin_motion]-self.bin_start)
        self.train_step=jax.jit(jax.vmap(self._step,in_axes=(0,0,None)))
        self.train_reset=jax.jit(jax.vmap(lambda k,p:self._with_reset_probs(lambda:self._reset(k),p),in_axes=(0,None)))
        self.reset = jax.jit(jax.vmap(self._reset))
        self.step = jax.jit(jax.vmap(self._step))
        self.extract = jax.jit(jax.vmap(self._extract))
        self.reference = jax.jit(jax.vmap(self._reference))

    def _with_reset_probs(self, call, probs):
        # Install only while tracing this reset call; weights are explicit JIT inputs.
        native=self.env.th.reset_state
        def sample(env,model,data,carry,backend):
            key,k1,k2=jax.random.split(carry.key,3)
            b=jax.random.categorical(k1,jnp.log(probs))
            tid=self.bin_motion[b]
            frame=self.bin_start[b]+jax.random.randint(k2,(),0,self.bin_width[b])
            ts=carry.traj_state.replace(traj_no=tid,subtraj_step_no=frame,subtraj_step_no_init=frame)
            return data,carry.replace(key=key,traj_state=ts)
        self.env.th.reset_state=sample
        try:return call()
        finally:self.env.th.reset_state=native

    def _reset(self,key):return TrackingState(self.env.mjx_reset(key),jnp.int32(0))

    def objective(self,x,ref,action):
        return candidate_objective(x,ref,self.native_reward(x,ref,action),self.nq,self.nv,len(self.ids)-1,jnp)

    def _features(self, data):
        p, rot, vel = calculate_relative_site_quatities(data, self.ids, self.body_ids, self.root_ids, jnp)
        # Computing the native 5 reward sites separately avoids differences in
        # batched rotation conversion compared with the 15-site observation path.
        rp,rr,rv=calculate_relative_site_quatities(data,self.rf._rel_site_ids,self.rf._rel_body_ids,self.root_ids,jnp)
        return jnp.concatenate([data.qpos, data.qvel, p.ravel(), rot.ravel(), vel.ravel(),
                                rp.ravel(),rr.ravel(),rv.ravel()])

    def _extract(self, state):
        t = state.additional_carry.traj_state
        return self._features(state.data), t.traj_no, t.subtraj_step_no

    def _reference(self, tid, frame):
        frame = jnp.clip(frame, 0, self.lengths[tid]-1)
        data = self.traj.get(tid, frame, jnp)
        return self._features(data)

    def obs(self, x):
        return jnp.concatenate([x[...,self.obs_state_indexes],x[...,self.nq+self.nv:self.observation_state_dim]],axis=-1)

    def goal(self, tid, frame, k=10):
        frames = frame[..., None] + jnp.arange(k)
        frames = jnp.minimum(frames, self.lengths[tid][..., None]-1)
        tids = jnp.broadcast_to(tid[..., None], frames.shape)
        refs = self.reference(tids.reshape(-1), frames.reshape(-1))
        return refs[:, 2:self.observation_state_dim].reshape((*tid.shape, k*self.gd))

    def _step(self, state, action, reset_probs=None):
        terminal = self.env.mjx_step(state.native, action)
        x1, tid1, frame1 = self._extract(terminal)
        ref=self._reference(tid1,frame1)
        task=self.objective(x1,ref,action)
        count=jnp.int32(0);task_failure=terminal.absorbing
        valid=(state_valid(x1,jnp,self.nq,self.nv) & jnp.isfinite(terminal.reward)
               & jnp.all(jnp.isfinite(action)) & jnp.all(jnp.isfinite(terminal.observation))
               & (jnp.max(jnp.abs(terminal.observation))<=LIMITS['feature_absmax'])
               & jnp.all(jnp.isfinite(terminal.data.qacc))
               & (jnp.max(jnp.abs(terminal.data.qacc))<=LIMITS['qacc_absmax']))
        # Keep exactly one Warp reset call site: tracing separate normal and
        # numerical reset branches leaks mutable backend tracers upstream.
        # Leave Warp's internal contact buffers untouched; its reset/forward
        # owns those buffers. Only clear physical force/warm-start fields.
        data=terminal.data.replace(**{name:jnp.where(valid,getattr(terminal.data,name),
            jnp.zeros_like(getattr(terminal.data,name))) for name in
            ['qacc_warmstart','qacc','ctrl','qfrc_applied','xfrc_applied']})
        info=jax.tree.map(lambda cur,prev:jnp.where(valid,cur,prev),terminal.info,state.info)
        sanitized=terminal.replace(data=data,observation=jnp.where(valid,terminal.observation,state.observation),
            info=info,done=terminal.done|task_failure|~valid,absorbing=task_failure|~valid,
            reward=jnp.where(valid,terminal.reward,0.),additional_carry=terminal.additional_carry.replace(
                key=jnp.where(valid,terminal.additional_carry.key,
                    jax.random.fold_in(terminal.additional_carry.key,7919))))
        reset_fn=self._native_reset_done if reset_probs is None else lambda s:self._with_reset_probs(lambda:self._native_reset_done(s),reset_probs)
        ready=jax.lax.cond(sanitized.done,reset_fn,lambda s:s,sanitized)
        # Preserve the raw invalid successor for diagnostics. Replay masks it,
        # while evaluation counts this numerical termination as a failed episode.
        return TrackingState(ready,jnp.where(sanitized.done,0,count)), (x1, tid1, frame1,sanitized.reward,
                       sanitized.absorbing,sanitized.done,valid)

    def project(self, x):
        q = x[..., 3:7]
        q = q/jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
        return x.at[...,3:7].set(q)

    def reward(self,x,ref,action,failure=False):
        return self.native_reward(x,ref,action)

    def native_reward(self, x, ref, action):
        rf = self.rf
        nonquat = jnp.array([0,1,2]+list(range(7,self.nq)))
        q = x[...,3:7]/jnp.maximum(jnp.linalg.norm(x[...,3:7],axis=-1,keepdims=True),1e-8)
        qr = ref[...,3:7]/jnp.maximum(jnp.linalg.norm(ref[...,3:7],axis=-1,keepdims=True),1e-8)
        rw=jnp.sum(q*qr,axis=-1)
        rv=q[...,:1]*qr[...,1:]-qr[...,:1]*q[...,1:]-jnp.cross(q[...,1:],qr[...,1:])
        angle=2*jnp.arctan2(jnp.sqrt(jnp.sum(rv*rv,axis=-1)+1e-16),jnp.abs(rw))
        # Match native float32 rotation arithmetic; use a finite subgradient at
        # identical rotations, where the unsmoothed norm has no unique gradient.
        native_angle=quaternion_angular_distance(x[...,3:7],ref[...,3:7],jnp)
        angle=angle+jax.lax.stop_gradient(native_angle-angle)
        ep = jnp.mean((x[...,nonquat]-ref[...,nonquat])**2,axis=-1) + angle
        ev = jnp.mean((x[...,self.nq:self.nq+self.nv]-ref[...,self.nq:self.nq+self.nv])**2,axis=-1)
        s0 = self.observation_state_dim
        n = (len(rf._rel_site_ids)-1)*3
        epos = jnp.mean((x[...,s0:s0+n]-ref[...,s0:s0+n])**2,axis=-1)
        erot = jnp.mean((x[...,s0+n:s0+2*n]-ref[...,s0+n:s0+2*n])**2,axis=-1)
        vel = (x[...,s0+2*n:]-ref[...,s0+2*n:]).reshape((*x.shape[:-1],len(rf._rel_site_ids)-1,6))
        eva = jnp.mean(vel[...,:3]**2,axis=(-2,-1))
        evl = jnp.mean(vel[...,3:]**2,axis=(-2,-1))
        r = (rf._qpos_w_sum*jnp.exp(-rf._qpos_w_exp*ep)
             +rf._qvel_w_sum*jnp.exp(-rf._qvel_w_exp*ev)
             +rf._rpos_w_sum*jnp.exp(-rf._rpos_w_exp*epos)
             +rf._rquat_w_sum*jnp.exp(-rf._rquat_w_exp*erot)
             +rf._rvel_w_sum*(jnp.exp(-rf._rvel_w_exp*eva)+jnp.exp(-rf._rvel_w_exp*evl)))
        lo,hi = jnp.asarray(self.env.info.action_space.low),jnp.asarray(self.env.info.action_space.high)
        penalty = -rf._action_out_of_bounds_coeff*jnp.mean((jnp.maximum(lo-action,0)+jnp.maximum(action-hi,0))**2,axis=-1)
        return jnp.maximum(r+jnp.maximum(penalty,-1),0)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
