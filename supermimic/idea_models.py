"""Two independently testable, approximately size-matched WM variants."""
import jax
import jax.numpy as jnp
import flax.linen as nn
from models import MLP


class TrustWorld(nn.Module):
    sd: int

    def setup(self):
        self.layers = [nn.Dense(1986) for _ in range(3)]
        self.norms = [nn.LayerNorm() for _ in range(3)]
        self.heads = [nn.Dense(self.sd + 1, kernel_init=nn.initializers.zeros)
                      for _ in range(2)]

    def all_heads(self, x, a):
        z = jnp.concatenate([x, a], axis=-1)
        for layer, norm in zip(self.layers, self.norms):
            z = jax.nn.silu(norm(layer(z)))
        outputs = jnp.stack([head(z) for head in self.heads], axis=0)
        return outputs[..., :self.sd], outputs[..., self.sd] - 5.

    def __call__(self, x, a):
        delta, logit = self.all_heads(x, a)
        return delta.mean(0), logit.mean(0)


class ExpertWorld(nn.Module):
    sd: int

    def setup(self):
        self.shared = MLP((2007, 2007, 2007), self.sd + 1,
                          norm=True, silu=True, zero=True)
        self.experts = [MLP((128, 128), self.sd, silu=True, zero=True)
                        for _ in range(4)]
        self.router = MLP((32,), 4, silu=True)

    def routing(self, x):
        return jax.nn.softmax(self.router(x), axis=-1)

    def __call__(self, x, a):
        z = jnp.concatenate([x, a], axis=-1)
        main = self.shared(z)
        residuals = jnp.stack([expert(z) for expert in self.experts], axis=-2)
        residual = (self.routing(x)[..., None] * residuals).sum(-2)
        return main[..., :self.sd] + residual, main[..., self.sd] - 5.
