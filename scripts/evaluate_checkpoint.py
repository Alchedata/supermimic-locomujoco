"""Evaluate a trusted optimizer checkpoint on the fixed 160-episode protocol."""
import argparse
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('checkpoint', type=Path)
p.add_argument('--out', required=True)
p.add_argument('--gpu', default='0')
p.add_argument('--upstream', default=str(ROOT / 'external/loco-mujoco'))
p.add_argument('--variables')
a = p.parse_args()
os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, MUJOCO_GL='egl',
                  XLA_PYTHON_CLIENT_PREALLOCATE='false', NUMPY_MADVISE_HUGEPAGE='0',
                  JAX_DEFAULT_MATMUL_PRECISION='highest', JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS='true',
                  LOCO_MOTION_MANIFEST=str(ROOT / 'configs/lafan1_40.json'))
if a.variables:
    os.environ['LOCO_VARIABLES_FILE'] = str(Path(a.variables).resolve())
sys.path.insert(0, str(ROOT / 'supermimic'))
import jax
import jax.numpy as jnp
from adapter import Adapter, make_config, make_env
from evaluate import evaluate
from models import Policy
from actions import TRANSFORMS, policy_action

ck = pickle.loads(a.checkpoint.read_bytes())
cfg = make_config(a.upstream, 160)
ad = Adapter(make_env(cfg))
ad.motion_names = list(cfg.experiment.task_factory.params.lafan1_dataset_conf.dataset_name)
pi = Policy(ad.na)
mean, std = (jnp.asarray(ck['norm'][k]) for k in ['mean', 'std'])
lo, hi = (jnp.asarray(v) for v in [ad.env.info.action_space.low, ad.env.info.action_space.high])
transform = ck['action_transform']
assert transform in TRANSFORMS, transform

@jax.jit
def act(h, g, obs, x):
    ng = ((g.reshape(g.shape[0], 10, ad.gd) - mean[2:ad.observation_state_dim]) /
          std[2:ad.observation_state_dim]).reshape(g.shape)
    raw = pi.apply(ck['pp'], (h - ad.obs(mean)) / ad.obs(std), ng)
    return policy_action(raw, lo, hi, transform)

evaluate(ad, act, a.out, f'iter_{ck["iteration"]}', n=160, horizon=1000)
