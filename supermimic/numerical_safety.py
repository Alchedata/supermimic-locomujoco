"""Shared numerical-failure bounds for actual H1 simulator transitions.

These are conservative numerical bounds, not task success/termination criteria.
"""
import numpy as np

LIMITS = dict(qvel_absmax=1000., qpos_absmax=1e4, feature_absmax=1e6, qacc_absmax=1e10)


def state_valid(x, xp=np, nq=26, nv=25):
    return (xp.all(xp.isfinite(x), axis=-1)
            & (xp.max(xp.abs(x[..., :nq]), axis=-1) <= LIMITS['qpos_absmax'])
            & (xp.max(xp.abs(x[..., nq:nq+nv]), axis=-1) <= LIMITS['qvel_absmax'])
            & (xp.max(xp.abs(x), axis=-1) <= LIMITS['feature_absmax']))


def history_valid(h, velocity_indexes, xp=np):
    return (xp.all(xp.isfinite(h), axis=(-2, -1))
            & (xp.max(xp.abs(h), axis=(-2, -1)) <= LIMITS['feature_absmax'])
            & (xp.max(xp.abs(h[..., velocity_indexes]), axis=(-2, -1)) <= LIMITS['qvel_absmax']))
