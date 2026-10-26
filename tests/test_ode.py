"""Check integrator/FK reuse values and action gradients on a synthetic field."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'supermimic'))
import jax
import jax.numpy as jnp
import numpy as np
from ode_world import StructuredODE


class Field:
    def apply(self, parameters, full, action):
        # Depend on reconstructed features, so stale midpoint FK is detectable.
        return .1 * full[..., -25:] + action[..., :1], jnp.zeros(full.shape[:-1])


class ODETest(unittest.TestCase):
    def test_reuse_preserves_multistep_values_and_gradients(self):
        for solver in ['euler', 'midpoint']:
            ode = StructuredODE.__new__(StructuredODE)
            ode.solver = solver
            ode.nq, ode.nv, ode.physical_dim = 26, 25, 51
            ode.ad = SimpleNamespace(env=SimpleNamespace(dt=.02))
            ode.features = lambda x: jnp.concatenate([x, jnp.tile(x, (1, 5))[..., :216] ** 2], -1)
            x = jnp.linspace(-.1, .2, 51)[None, :].at[:, 3:7].set(jnp.array([1., 0., 0., 0.]))
            norm = dict(mean=jnp.zeros(267), std=jnp.ones(267),
                        delta_mean=jnp.zeros(267), delta_std=jnp.ones(267)*.02)
            def rollout(action, reuse):
                full = ode.features(x)
                for _ in range(4):
                    fn = ode.step_reuse if reuse else ode.step
                    full, _ = fn(Field(), None, full, action, norm)
                return full
            action = jnp.ones((1, 19)) * .13
            np.testing.assert_allclose(rollout(action, True), rollout(action, False), atol=1e-6)
            grad = lambda reuse: jax.grad(lambda a: rollout(a, reuse).sum())(action)
            np.testing.assert_allclose(grad(True), grad(False), rtol=1e-5, atol=1e-6)

    def test_midpoint_recomputes_intermediate_features(self):
        ode = StructuredODE.__new__(StructuredODE)
        ode.solver = 'midpoint'
        ode.nq, ode.nv, ode.physical_dim = 26, 25, 51
        ode.ad = SimpleNamespace(env=SimpleNamespace(dt=.02))
        seen = []
        def features(x):
            seen.append(np.asarray(x).copy())
            return jnp.concatenate([x, jnp.zeros((1, 216))], -1)
        ode.features = features
        x = jnp.zeros((1, 51)).at[:, 3].set(1.).at[:, 26].set(.5)
        full = features(x); seen.clear()
        norm = dict(mean=jnp.zeros(267), std=jnp.ones(267),
                    delta_mean=jnp.zeros(267), delta_std=jnp.ones(267)*.02)
        ode.step_reuse(Field(), None, full, jnp.ones((1, 19)), norm)
        self.assertEqual(len(seen), 2)  # intermediate field and successor
        self.assertFalse(np.array_equal(seen[0], np.asarray(x)))


if __name__ == '__main__':
    unittest.main()
