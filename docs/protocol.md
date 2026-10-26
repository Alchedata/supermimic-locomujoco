# Training and evaluation

## Environment

- LocoMuJoCo revision `3921fedbca93876c0595c90c535b24ec08ff623d`.
- `MjxUnitreeH1`, 19 actions, native torque control (`DefaultControl`), MuJoCo Warp.
- All 40 motion names in `configs/lafan1_40.json`, loaded into one environment and one shared policy.
- Native `MimicReward`, native absorbing conditions, horizon 1000 controls (10 seconds).
- Contact capacity `nconmax=32768`, constraint capacity `njmax=256`.

The adapter returns terminal transitions before automatic reset. Invalid numerical transitions are saved, counted and excluded from WM supervision. Evaluation counts them as failures. Quaternion normalization and finite reward subgradients keep the differentiable surrogate well-defined.

`objective.py` computes extra posture diagnostics. Its candidate reward is not used for training or evaluation returns: `Adapter.reward` uses `native_reward`, and real steps keep the simulator's reward. The reference-posture diagnostic does not override the native termination condition.

## World model and actor

State contains full joint positions and velocities plus relative-site features. The network predicts a normalized state increment and a termination logit. It learns autoregressively over eight real transitions. The successor quaternion is normalized after each predicted step. Reward is computed analytically from predicted state and reference features.

The actor receives ten proprioceptive frames and ten future reference frames. The motion encoder produces a 32-dimensional `tanh` latent. The decoder produces 19 raw actions. The critic takes normalized state and the reference goal. Learned termination probabilities weight survival through the imagined rollout; the WM is fixed during each actor gradient calculation.

Training uses one actor update, ten WM updates and twenty critic updates per iteration. The starting learning rate is 2e-4, with a cosine schedule over 250k iterations and a floor of 3%. Discount is 0.99 and GAE lambda is 0.95. Actor Adam betas are (0.9, 0.95); the other optimizers use Optax Adam defaults. Gradient norm clipping is 1; the actor also clips gradient elements to 0.1.

| Module | v0 parameters | Expert parameters |
|---|---:|---:|
| World model | 9,541,900 | 9,545,779 |
| Policy | 14,479,667 | 14,479,667 |
| Critic | 657,153 | 657,153 |

## Action output experiment

Let `u` be the last linear decoder output and `[low, high]` the native action bounds.

- Historical tanh mode: `command = (low+high)/2 + (high-low)/2 * tanh(u)`.
- Linear mode: `command = u`.

Real collection executes `clip(command + Gaussian(0, 0.1), low, high)` and replay stores that exact action. Imagined rollouts and deterministic evaluation both use `clip(command, low, high)`. `supermimic/actions.py` defines the shared mapping. H1 uses normalized torque commands with bounds [-1, 1]; the simulator converts them to actuator units. Native reward code and exploration noise are unchanged.

The first linear experiment used raw commands in imagination and evaluation despite clipped collection. It failed at iteration 140. That historical interface remains available with `--action-interface legacy`; checkpoint metadata distinguishes `linear_native_clip_collection` from the corrected `linear_native_clip_all`. The evaluation script follows the saved metadata.

Hard clipping has zero action gradient outside the bounds. A separate fresh Expert ablation adds `0.01 * mean(relu(abs(u) - 1)**2)` at real replay starts to supply a restoring gradient. The primary aligned Expert run has zero auxiliary weight. A fresh aligned MLP run controls for the Expert structure. All three use seed 44 and the same 5k training budget, reward and evaluation protocol.

The goal-latent `tanh` is unchanged. SuperMimic Appendix C, Table A.4 explicitly bounds the latent with `tanh` and lists an MLP action decoder. It does not specify a final action activation. The two public reproductions cited in the README use a linear final decoder layer. Our historical action-output `tanh` is a port choice.

## Evaluation

Every 1000 iterations, evaluate four fixed random starts from each of the 40 training motions. Reset keys use seed 20261022. The policy is deterministic and each episode is scored only until its first termination or the 1000-control horizon. The evaluator saves the reset manifest and all 160 episode records.

- **Return:** mean undiscounted episode reward sum; not normalized to [0, 1].
- **Error:** mean Euclidean distance between simulated and reference relative-site positions, averaged over sites, valid controls and episodes, in meters.
- **Failures:** native absorbing terminations or numerical terminations.
- **Full horizon:** 1000 controls completed without failure. Natural reference endings can stop an episode earlier without a failure; in this fixed manifest four episodes do so.

This is our fixed evaluation protocol for the port, not a claim that LocoMuJoCo prescribes a universal 40-motion/160-episode standard. It measures coverage of training motions. All method comparisons use the same reset set.

## Checkpoint continuation

`--iterations` is an absolute target, so resuming at 1000 with `--iterations 5000` performs 4000 additional iterations. Parameters, Adam state, normalization, RNG and curriculum are restored. Replay and live simulator state are rebuilt, with 64 refill steps. The run writes `restore_check.json` and the parent checkpoint SHA-256.

Head-reset and no-change continuation controls use this same refill procedure. Their replay contents can differ because their restored policies differ.

## ODE result provenance and compute

The completed full-state Euler results cover seeds42,43,44 with AS weights0 and0.01. Completed Midpoint results currently cover seed42 for both AS settings. Each `results/ode_*_provenance.json` includes exact training arguments, source hashes, model dimensions, and the SHA256 of its evaluation file. The three-seed accelerated Midpoint campaign is still running and is excluded from completed tables.

All ODE runs start from scratch. Seed42 results use actor microbatch512 and the original FK implementation; Euler seeds43/44 use microbatch1024 and differentiable successor-FK reuse. All use actor batch4096, horizon32, the same WM/critic update counts, and the same real-data budget. The public launcher defaults to the accelerated implementation for new ODE runs. It does not claim to reproduce an old floating-point trajectory bit for bit.

At5k there are51,200,000 collection transitions plus262,144 warmup transitions. Reported Return is mean over160 episodes, not the sum. Its fixed-horizon theoretical ceiling is1600 (1000 controls times reward ceiling1.6). Dividing by1600 gives a normalized return, not a success rate. Do not divide by each episode's observed survival length when comparing total task performance.

Runtime figures exclude first-iteration JIT compilation and periodic evaluation. The published fixed-batch actor benchmark found FK reuse faster; JIT-scan gradient accumulation was about1% slower and is not used. Long-run accelerated Euler timings are about2.35–2.44 seconds/iteration on L40S. Early Midpoint timings are3.63–3.69 seconds/iteration. Full training wall time also includes initialization, evaluation, diagnostics and checkpoint writes.
