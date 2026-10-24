# SuperMimic on LocoMuJoCo

A JAX implementation of SuperMimic-style model-based control for the Unitree H1, trained jointly on 40 LAFAN1 motions. The world model learns from simulator transitions. The actor learns by backpropagating through 32-step imagined rollouts, with a critic for the remaining return.

This is an independent LocoMuJoCo port. The original [SuperMimic](https://supermimic.github.io/) uses the G1 and BONES-SEED. This repository contains our H1 baseline and world-model experiments.

## Setup

Tested with Python 3.12, Linux, CUDA 12 and an NVIDIA L40S. Replay lives in host memory; the 4096-environment setup needs substantially more RAM than GPU memory.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/robfiras/loco-mujoco.git external/loco-mujoco
git -C external/loco-mujoco checkout 3921fedbca93876c0595c90c535b24ec08ff623d
pip install -e external/loco-mujoco
```

The motion list is in `configs/lafan1_40.json`. LocoMuJoCo downloads the H1 motion data from its Hugging Face dataset on first use. To reuse converted data, create a local `configs/variables.yaml`:

```yaml
LOCOMUJOCO_CONVERTED_LAFAN1_PATH: /absolute/path/to/converted_lafan1
LOCOMUJOCO_MODELS_PATH: /absolute/path/to/robot_models
```

Pass it with `--variables configs/variables.yaml`. The datasets and robot assets remain under their upstream licenses.

## Train

```bash
# Baseline MLP world model
python scripts/train.py --method baseline --seed 43 --gpu 0 --out runs/v0_s43

# Residual-Expert world model
python scripts/train.py --method experts --seed 44 --gpu 1 --out runs/expert_s44

# Same Expert setup, with a linear action output instead of tanh
python scripts/train.py --method experts --seed 44 --gpu 1 \
  --action-output linear --out runs/expert_s44_linear
```

Each command runs 5,000 iterations. For a six-update smoke run, add `--smoke` and use a separate output directory. Other method choices are `sensitivity` and `trust`; their definitions are in [docs/methods.md](docs/methods.md).

| Setting | Value |
|---|---:|
| Parallel environments | 4096 |
| Real steps per iteration, per environment | Alternating 3 / 2 |
| Replay steps per environment | 1000 |
| WM batch size / updates per iteration | 2048 / 10 |
| WM training window | 8 |
| Imagined rollout batch / microbatch | 4096 / 512 |
| Imagined horizon | 32 |
| Critic updates per iteration | 20 |
| Evaluation interval | 1000 |
| Checkpoint interval | 100 |

Logs, evaluation episodes and checkpoints are written to `--out`. Checkpoints include model weights, optimizers, normalization, curriculum and RNG state. `--resume` restores those fields and rebuilds the simulator and replay buffer.

To evaluate a saved training checkpoint:

```bash
python scripts/evaluate_checkpoint.py runs/v0_s43/checkpoints/iter_5000.pkl \
  --gpu 0 --out runs/v0_s43/reevaluation
python scripts/summarize.py runs/v0_s43/reevaluation/iter_5000.json
```

The actor's 32-dimensional goal latent always uses `tanh`. `--action-output` controls the separate action output transform. The historical runs below use `tanh`; the linear-output experiment is in progress. Collection still clips noisy actions to the native bounds in both modes. See [docs/protocol.md](docs/protocol.md) for the action interface and evaluation details.

## Results at 5k

Return is the mean undiscounted episode sum. Error is mean relative-site position error in meters. Failures count native absorbing or numerical terminations across 160 evaluation episodes. These are fixed checkpoints, not best-of-run scores.

| Method | Seed | Return | Error (m) | Failures / 160 |
|---|---:|---:|---:|---:|
| v0 | 42 | 818.45 | 0.1358 | 2 |
| Expert | 42 | 860.06 | 0.1087 | 0 |
| v0 | 43 | 858.83 | 0.0932 | 2 |
| Expert | 43 | 863.23 | 0.0949 | 2 |
| v0 | 44 | 811.37 | 0.0937 | 17 |
| Expert | 44 | 149.72 | 0.5328 | 35 |
| Expert + head reset | 44 | 812.63 | 0.1131 | 2 |
| Expert + head reset + anti-saturation | 44 | 797.17 | 0.1202 | 0 |
| Action-Sensitivity | 42 | 813.50 | 0.1283 | 1 |
| Trust-Aware | 42 | 814.25 | 0.0923 | 15 |

Expert seed 44 became stuck with saturated action outputs. The two recovery runs branch from its 1k checkpoint and continue to total iteration 5k. A continuation with no weight changes and a fresh linear-output run are being tested separately. The seed 42 v0 run resumed at iteration 300 with restored weights and optimizers but rebuilt replay; seeds 43 and 44 ran from scratch.

The full episode records, parameter counts and run settings are in [results/](results/). All listed 5k evaluations had zero numerical failures; training-time numerical events are a separate log.

## Recovery experiments

```bash
python scripts/reset_head.py runs/expert_s44/checkpoints/iter_1000.pkl \
  runs/forks/expert44_head_reset.pkl

python scripts/train.py --method experts --seed 44 --gpu 0 \
  --resume runs/forks/expert44_head_reset.pkl --out runs/head_reset

python scripts/train.py --method experts --seed 44 --gpu 1 \
  --resume runs/forks/expert44_head_reset.pkl --saturation-weight 0.01 \
  --saturation-threshold 2 --out runs/head_reset_antisat
```

Head reset multiplies the final decoder layer's weights and bias by 0.1 and clears that layer's Adam moments. It preserves the shared Adam step counter. Anti-saturation adds a penalty on raw decoder outputs at real replay states; it does not change environment rewards.

For the no-change control, resume directly from the original `iter_1000.pkl` without running `reset_head.py`.

## Code

```text
supermimic/    trainer, models, simulator adapter and evaluation
scripts/      training launcher, checkpoint intervention and result reader
configs/      the 40-motion manifest
results/      5k episode records and run provenance
docs/         methods and evaluation protocol
```

## References

- [SuperMimic paper](https://arxiv.org/abs/2610.28480), Liao et al., 2026.
- [LocoMuJoCo](https://github.com/robfiras/loco-mujoco), the simulator, motion data interface and native reward used here.
- [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) and [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced), independent implementations consulted when checking the paper's architecture and training details.
