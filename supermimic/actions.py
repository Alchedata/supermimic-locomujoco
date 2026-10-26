"""Policy commands and the normalized actions actually executed by H1."""
import jax.numpy as jnp


TRANSFORMS = ('tanh_native_bounds', 'linear_native_clip_all',
              'linear_native_clip_collection')


def policy_command(raw, low, high, output):
    if output == 'linear':
        return raw
    if output == 'tanh':
        return (low + high) / 2 + (high - low) / 2 * jnp.tanh(raw)
    raise ValueError(output)


def executed_action(command, low, high, noise=0.):
    # Exploration precedes the single execution clamp. Replay stores this result.
    return jnp.clip(command + noise, low, high)


def policy_action(raw, low, high, transform):
    if transform == 'linear_native_clip_collection':
        return raw  # Historical checkpoints used raw actions in imagination/eval.
    if transform == 'linear_native_clip_all':
        return executed_action(raw, low, high)
    if transform == 'tanh_native_bounds':
        return executed_action(policy_command(raw, low, high, 'tanh'), low, high)
    raise ValueError(transform)
