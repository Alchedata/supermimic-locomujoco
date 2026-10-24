"""Plain MLP baseline only: SuperMimic Table A.4 layer sizes, no ODE/PDE."""
import jax
import jax.numpy as jnp
import flax.linen as nn


class MLP(nn.Module):
    hidden: tuple
    out: int
    norm: bool = False
    silu: bool = False
    zero: bool = False
    small: bool = False

    @nn.compact
    def __call__(self,x):
        for width in self.hidden:
            x=nn.Dense(width)(x)
            if self.norm:x=nn.LayerNorm()(x)
            x=jax.nn.silu(x) if self.silu else jax.nn.elu(x)
        init=nn.initializers.zeros if self.zero else nn.initializers.variance_scaling(.0001 if self.small else 1.,'fan_in','truncated_normal')
        return nn.Dense(self.out,kernel_init=init)(x)


class World(nn.Module):
    sd: int

    @nn.compact
    def __call__(self,x,a):
        h=MLP((2048,2048,2048),self.sd+1,norm=True,silu=True,zero=True)(jnp.concatenate([x,a],axis=-1))
        # Initial termination prior is small, rather than sigmoid(0)=0.5.
        return h[...,:self.sd],h[...,self.sd]-5.


class Policy(nn.Module):
    na: int

    @nn.compact
    def __call__(self,h,g):
        z=jnp.tanh(MLP((2048,1024,512,256),32)(g))
        return MLP((2048,1024,512,256),self.na,small=True)(jnp.concatenate([h.reshape((h.shape[0],-1)),z],axis=-1))


class Critic(nn.Module):
    @nn.compact
    def __call__(self,x,g):
        return MLP((256,128),1)(jnp.concatenate([x,g],axis=-1))[...,0]
