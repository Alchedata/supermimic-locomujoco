"""Launch the common 5k protocol. Extra arguments go to the trainer."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--method', choices=['baseline', 'ode', 'experts', 'sensitivity', 'trust'], default='baseline')
p.add_argument('--solver', choices=['euler', 'midpoint'], default='euler')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--gpu', default='0')
p.add_argument('--out', required=True)
p.add_argument('--upstream', default=str(ROOT / 'external/loco-mujoco'))
p.add_argument('--variables', help='Optional local LocoMuJoCo cache configuration')
p.add_argument('--action-output', choices=['tanh', 'linear'], default='tanh')
p.add_argument('--iterations', type=int, default=5000)
p.add_argument('--smoke', action='store_true', help='Six full-size updates in a separate output directory')
a, extra = p.parse_known_args()
if a.method == 'ode' and a.action_output != 'tanh':
    p.error('The ODE experiments use tanh action output.')
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=a.gpu, MUJOCO_GL='egl',
           XLA_PYTHON_CLIENT_PREALLOCATE='false', NUMPY_MADVISE_HUGEPAGE='0',
           JAX_DEFAULT_MATMUL_PRECISION='highest', JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS='true',
           WANDB_MODE='disabled', LOCO_MOTION_MANIFEST=str(ROOT / 'configs/lafan1_40.json'))
if a.variables:
    env['LOCO_VARIABLES_FILE'] = str(Path(a.variables).resolve())
trainer = 'train_ode.py' if a.method == 'ode' else 'train.py'
method_args = ['--solver', a.solver] if a.method == 'ode' else ['--idea', a.method, '--action-output', a.action_output]
cmd = [sys.executable, '-u', str(ROOT / 'supermimic' / trainer),
       '--upstream', str(Path(a.upstream).resolve()), '--out', a.out,
       *method_args, '--seed', str(a.seed),
       '--num-envs', '4096', '--batch', '2048', '--actor-batch', '4096',
       '--actor-microbatch', '1024' if a.method == 'ode' else '512', '--buffer-steps', '1000',
       '--iterations', '6' if a.smoke else str(a.iterations), '--max-hours', '0',
       '--imagined-horizon', '32', '--wm-batches', '10', '--wm-rehearsal-fraction', '0',
       '--warmup-iters', '0', '--adapt-iters', '0', '--eval-every', '1000',
       '--eval-envs', '160', '--eval-horizon', '1000', '--retain-eval-checkpoints',
       '--retain-every', '100', '--capture-gradient-above', '10000',
       '--max-gradient-captures', '8']
if a.smoke:
    cmd.append('--smoke')
subprocess.run(cmd + extra, env=env, check=True)

