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

The fresh linear-output run tests removal of the final action tanh while keeping the latent tanh. It uses the same seed and 5k budget. These experiments investigate a failure observed in this port; the README reports completed results without assigning it to the original SuperMimic implementation.
