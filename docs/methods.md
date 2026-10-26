# World-model experiments

All four methods share the actor, critic, reward, termination targets and evaluation code. The `--method` flag in `scripts/train.py` selects the world model and its auxiliary objective.

| Method | Change |
|---|---|
| `baseline` | A three-layer, width-2048 MLP predicts state increments and termination. |
| `experts` | A width-2007 shared MLP plus four two-layer width-128 residual experts. A state-conditioned router mixes the residuals. |
| `sensitivity` | The baseline WM also fits finite-difference action responses from paired simulator branches. |
| `trust` | A shared width-1986 trunk has two independently fitted output heads. Disagreement shortens imagined continuation through an early critic bootstrap. |

Expert uses a load-balancing loss `0.01 * (4 * sum(mean_router_usage**2) - 1)`. The router is a width-32 MLP. Termination comes from the shared trunk. Its parameter count is within 0.05% of v0.

Action-Sensitivity collects plus/minus perturbation pairs every 80 iterations with epsilon 0.02. A separate 256-pair holdout measures response prediction. The sensitivity loss weight is 0.1; extra simulator controls are logged separately.

Trust-Aware uses Bernoulli bootstrap masks for the two heads. Confidence is a stop-gradient exponential of disagreement in state-standard-deviation units, with scale 0.05. The first eight imagined steps remain active; later steps may bootstrap earlier. This does not treat model disagreement as physical death.

## Saturation recovery

Expert seed 44's tanh-output run plateaued near return 150. At iteration 1000, a separate real-rollout diagnostic measured 78.3% of action components above 0.99 in magnitude. The recovery comparison uses three continuations from that checkpoint:

1. Restore the checkpoint unchanged.
2. Multiply the final decoder layer by 0.1 and zero its Adam first/second moments.
3. Apply the same head reset, plus `0.01 * mean(relu(abs(u) - 2)**2)` on raw decoder outputs at sampled real replay states.

The shared Adam counter is preserved. No other network weights are reset. The third treatment changes the actor objective only; environment reward, critic targets and WM loss are unchanged.

The fresh linear-output runs keep the latent tanh and use the aligned execution mapping described in [protocol.md](protocol.md). The three controls are Expert, Expert plus the raw-output penalty with threshold 1 and weight 0.01, and the baseline MLP. All start from scratch with seed 44 and a 5k budget. The first linear run, which fed unbounded commands into the WM, is retained as a failed interface experiment.

## Full-State ODE and anti-saturation

`--method ode --solver euler|midpoint` selects the structured world model. Both integrators receive all267 state features and19 actions, predict25 generalized accelerations, integrate26 qpos and25 qvel coordinates, and reconstruct body-site features through differentiable FK. The root quaternion is updated on the rotation manifold and normalized. The WM has10,006,672 parameters; actor and critic match the baseline.

The WM loss is autoregressive physical51 prediction MSE (normalized by delta standard deviation) plus termination BCE over8 steps. Body features are analytic, not independent neural outputs. The actor optimizes32-step imagined rewards plus critic bootstrap, weighted by learned survival probabilities.

For Euler, q_next integrates current velocity and v_next = v + dt * acceleration. Midpoint evaluates acceleration again at a half-step physical state, then integrates with midpoint velocity/acceleration. It uses two field evaluations per step.

The accelerated implementation reuses canonical successor FK at the next field input, while retaining fresh intermediate-state FK for Midpoint. Initial field features are rebuilt from the replay physical state; the initial critic input/history are unchanged. No stop-gradient is introduced. WM supervised training retains its original FK path.

AS adds `weight * mean(relu(abs(u) - threshold)^2)` at real replay starts. Current ODE experiments use weight0.01 and threshold1; the no-AS control uses weight0. This adds a restoring gradient directly on the raw output rather than relying on the derivative of tanh near saturation. The results establish a useful intervention on seed43; per-episode causal diagnosis of individual failures is separate from aggregate failure counts.
