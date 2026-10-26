"""Regression checks for the real/replay/imagined action interface (CPU)."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'supermimic'))
import jax
import jax.numpy as jnp
import numpy as np
from actions import executed_action, policy_action, policy_command


class ActionInterfaceTest(unittest.TestCase):
    def test_exploration_precedes_execution_clamp(self):
        raw=jnp.array([2., -2., .2])
        noise=jnp.array([-.3, .3, .1])
        command=policy_command(raw, -1., 1., 'linear')
        action=executed_action(command, -1., 1., noise)
        np.testing.assert_allclose(action, [1., -1., .3], atol=1e-7)
        self.assertFalse(np.allclose(action, executed_action(executed_action(raw, -1., 1.), -1., 1., noise)))

    def test_real_replay_and_imagination_use_same_action(self):
        raw=jnp.array([[-10., -.6, 0., .8, 10.]])
        low=jnp.array([-1., -.5, -2., -.4, -1.])
        high=jnp.array([1., .5, 2., .6, 1.])
        real=executed_action(policy_command(raw, low, high, 'linear'), low, high)
        replay=np.asarray(real).copy()
        imagined=policy_action(raw, low, high, 'linear_native_clip_all')
        np.testing.assert_array_equal(real, imagined)
        np.testing.assert_array_equal(replay, imagined)

    def test_gradient_matches_bounded_actuator(self):
        raw=jnp.array([-2., -.5, 0., .5, 2.])
        grad=jax.grad(lambda x: policy_action(x, -1., 1., 'linear_native_clip_all').sum())(raw)
        np.testing.assert_array_equal(grad, [0., 1., 1., 1., 0.])
        # The auxiliary penalty supplies a restoring gradient beyond the bounds.
        penalty=jax.grad(lambda x: .01*jnp.mean(jax.nn.relu(jnp.abs(x)-1.)**2))(raw)
        self.assertLess(float(penalty[0]), 0.)
        self.assertGreater(float(penalty[-1]), 0.)

    def test_checkpoint_modes_preserve_historical_behavior(self):
        raw=jnp.array([-4., 0., 4.])
        np.testing.assert_array_equal(policy_action(raw, -1., 1., 'linear_native_clip_collection'), raw)
        np.testing.assert_allclose(policy_action(raw, -1., 1., 'tanh_native_bounds'), jnp.tanh(raw))


if __name__ == '__main__':
    unittest.main()
