"""Full-state acceleration MLP, Euler/midpoint, differentiable H1 kinematics.

MuJoCo's JAX kinematics/com transforms are used, never its simulation step.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.scipy.spatial.transform import Rotation
from mujoco import mjx
from mujoco.mjx._src import smooth
from loco_mujoco.core.utils.math import calculate_relative_site_quatities
from models import MLP


class AccelerationWorld(nn.Module):
    nv: int
    width: int = 2158

    @nn.compact
    def __call__(self,x,a):
        y=MLP((self.width,)*3,self.nv+1,norm=True,silu=True,zero=True)(jnp.concatenate([x,a],axis=-1))
        return y[...,:self.nv],y[...,self.nv]-5.


def quat_multiply(q,r):
    w=q[...,0]*r[...,0]-jnp.sum(q[...,1:]*r[...,1:],axis=-1)
    v=q[...,:1]*r[...,1:]+r[...,:1]*q[...,1:]+jnp.cross(q[...,1:],r[...,1:])
    return jnp.concatenate([w[...,None],v],axis=-1)


def integrate_position(q,v,dt):
    """MuJoCo free joint: world translation, local angular velocity, hinge joints."""
    rotation=v[...,3:6]*dt
    angle2=jnp.sum(rotation**2,axis=-1,keepdims=True)
    angle=jnp.sqrt(angle2+1e-16)
    factor=jnp.where(angle2<1e-6,.5-angle2/48+angle2**2/3840,jnp.sin(.5*angle)/angle)
    dq=jnp.concatenate([jnp.cos(.5*angle),rotation*factor],axis=-1)
    orientation=quat_multiply(q[...,3:7],dq)
    orientation=orientation/jnp.maximum(jnp.linalg.norm(orientation,axis=-1,keepdims=True),1e-8)
    return jnp.concatenate([q[...,:3]+dt*v[...,:3],orientation,q[...,7:]+dt*v[...,6:]],axis=-1)


def stable_rotvec(matrix):
    """Same rotation-log values as native helper, finite derivatives at identity."""
    q=Rotation.from_matrix(matrix).as_quat()  # scalar last
    q=jnp.where(q[...,-1:]<0,-q,q)
    vector=q[...,:3];r2=jnp.sum(vector**2,axis=-1,keepdims=True)
    r=jnp.sqrt(r2+1e-16)
    scale=jnp.where(r2<1e-6,2+r2/3+3*r2*r2/20,2*jnp.arctan2(r,q[...,-1:])/r)
    return vector*scale


class StructuredODE:
    def __init__(self,ad,solver='midpoint'):
        assert solver in ('euler','midpoint')
        self.solver=solver
        self.ad=ad;self.nq=ad.nq;self.nv=ad.nv;self.physical_dim=ad.nq+ad.nv
        assert self.nq==26 and self.nv==25,'v1 implements H1 free root +19 hinges'
        self.model=mjx.put_model(ad.env._model,impl='jax')
        self.template=mjx.make_data(self.model)
        self.features=jax.jit(jax.vmap(self._features))

    def _relative(self,data,ids,bodyids):
        pos,_,vel=calculate_relative_site_quatities(data,ids,bodyids,self.ad.root_ids,jnp)
        mats=data.site_xmat[ids].reshape(-1,3,3)
        relative=jnp.einsum('ik,nkj->nij',mats[0].T,mats[1:])
        rot=stable_rotvec(relative)
        return jnp.concatenate([pos.ravel(),rot.ravel(),vel.ravel()])

    def _features(self,physical):
        data=self.template.replace(qpos=physical[:self.nq],qvel=physical[self.nq:])
        data=smooth.kinematics(self.model,data)
        data=smooth.com_pos(self.model,data)
        data=smooth.com_vel(self.model,data)
        obs=self._relative(data,self.ad.ids,self.ad.body_ids)
        rew=self._relative(data,self.ad.rf._rel_site_ids,self.ad.rf._rel_body_ids)
        return jnp.concatenate([data.qpos,data.qvel,obs,rew])

    def physical_step(self,network,parameters,physical,action,norm,current_full=None):
        mean,std=norm['mean'],norm['std']
        dt=self.ad.env.dt
        acc_mean=norm['delta_mean'][self.nq:self.physical_dim]/dt
        acc_std=norm['delta_std'][self.nq:self.physical_dim]/dt
        def field(physical,full=None):
            # Rebuild the complete 267-D state at every vector-field evaluation,
            # including Midpoint's intermediate state. No stale site features
            # are carried between integration stages.
            full=self.features(physical) if full is None else full
            y,logit=network.apply(parameters,(full-mean)/std,action)
            return acc_mean+acc_std*y,logit
        q,v=physical[...,:self.nq],physical[...,self.nq:]
        a0,logit=field(physical,current_full)
        if self.solver=='euler':
            q1=integrate_position(q,v,dt);v1=v+dt*a0
            return jnp.concatenate([q1,v1],axis=-1),logit
        qm=integrate_position(q,v,.5*dt);vm=v+.5*dt*a0
        am,_=field(jnp.concatenate([qm,vm],axis=-1))
        q1=integrate_position(q,vm,dt);v1=v+dt*am
        # One termination probability per control interval, not an integrated state.
        return jnp.concatenate([q1,v1],axis=-1),logit

    def step(self,network,parameters,x,action,norm):
        physical,logit=self.physical_step(network,parameters,x[...,:self.physical_dim],action,norm)
        return self.features(physical),logit

    def step_reuse(self,network,parameters,full,action,norm):
        """full must be canonical differentiable FK at this physical state.

        Reuse the previous successor, never stale simulator features. Midpoint's
        intermediate field still reconstructs its own FK in physical_step.
        """
        physical,logit=self.physical_step(network,parameters,full[...,:self.physical_dim],
                                         action,norm,current_full=full)
        return self.features(physical),logit
