# Action-Conditioned Latent World Model — 200-Episode Baseline

Frozen V-JEPA trunk, learned transition MLP plus a linear output head:

```text
context half of clip + target_xy  ->  predicted future V-JEPA latent  ->  success / target_miss / final block_xy
```

The future half of the clip is a target only. It is never an input to the predictor, at
training or inference time — the only inputs are the context-half latent and the skill params.

Reproduce with:

```bash
uv run python -m waddle_wm.train_action_conditioned_latent --max-episodes 200
```

Raw numbers: [`results/action_conditioned_latent.json`](../results/action_conditioned_latent.json)
(seed 0, 200 episodes: 140 train / 30 val / 30 test, 16 frames per window, 8000 full-batch
epochs, plan encoder 128-d, context dropout 0.5, AdamW lr 1e-3 wd 1e-4).

## Read the dataset audit first

The trainer prints a `dataset_audit` block, and on this dataset it reports three identities
that decide how the outcome numbers may be read:

- `distinct_target_sites: 1` — the green landing zone sits at `(0.5, 0.3)` in **every** episode.
- `success_equals_plan_within_target_radius: true` — success is exactly `‖target_xy − (0.5,0.3)‖ ≤ 0.105`.
- `every_episode_lifted: true` — no grasp ever fails, so `target_miss ≡ ¬success`. The two
  outcome heads are the same bit, which is why their accuracies are identical everywhere below.

So **success, target_miss, and the final block position are deterministic functions of the
action alone.** Any model that sees `target_xy` should score ~100% on them, and the fact that
it does is a sanity check, not evidence of a world model. The future-latent metrics are the
only ones on this dataset that test whether the visual context contributes anything.

## Held-out results (test split, n=30)

`lat_mse` is MSE against the true future latent in train-standardized units, so the
train-mean predictor scores ≈1.0. `lat_cos` is cosine similarity on the same standardized
vectors (0.0 = no better than the mean; raw-latent cosine is useless here — mean-pooled
V-JEPA latents share a large common component and every predictor reads ~0.996).

| predictor | lat_mse ↓ | lat_cos ↑ | success acc ↑ | target_miss acc ↑ | block_xy err ↓ |
|---|---|---|---|---|---|
| **context + action** | **0.703** | **0.513** | **1.000** | **1.000** | 4.8 mm |
| action only (no context) | 0.713 | 0.493 | 1.000 | 1.000 | **1.8 mm** |
| context only (no action) | 0.949 | 0.027 | 0.500 | 0.500 | 89.1 mm |
| constant (majority / train mean) | 0.932 | 0.000 | 0.433 | 0.433 | 86.1 mm |
| persistence (nothing moves) | 5.496 | 0.006 | 0.433 | 0.433 | 475.1 mm |

Val split (n=30) agrees: 0.757 / 0.814 / 1.038 / 1.014 / 5.899 lat_mse in the same row order.

`block_xy err` is mean Euclidean error; the target radius is 105 mm for scale.

## What this does and does not show

**The issue's bar is met.** Context+action beats every no-action baseline on the future
latent (0.703 vs 0.932 constant, 5.496 persistence) and on all three outcome metrics. The
persistence baseline is catastrophic on the latent precisely because these clips are mostly
motion — assuming nothing changes is the worst thing you can do.

**But the action is carrying almost all of it.** Action-only is within 0.011 lat_mse of the
full model and is *better* on block position (1.8 mm vs 4.8 mm — the context path overfits;
its train lat_mse is 0.002 against 0.703 on test). The honest claim is a small, consistent
context gain on the future latent, and none anywhere else:

| seed | context+action | action only | diff |
|---|---|---|---|
| 0 | 0.7026 | 0.7133 | −0.0107 |
| 1 | 0.6830 | 0.7131 | −0.0301 |
| 2 | 0.6892 | 0.7138 | −0.0247 |

Pooling per-episode test errors across the three seeds, the mean difference is −0.0218 with
a 95% paired-bootstrap CI of [−0.0425, −0.0004] — consistent in sign, but small and barely
excluding zero on 30 test episodes.

That is about what the scene allows. With the landing zone fixed and the block start jittered
over only an 8 cm × 8 cm box, the context has little to say about the future beyond what
`target_xy` already determines. **Getting a stronger read on the context requires varying the
scene, not a bigger model** — randomize the target site per episode, and success stops being a
function of the action alone.

## One real defect this run exposed

The first 200-episode run had context+action at *chance* — 0.957 lat_mse, 0.567 success
accuracy, no better than the constant predictor, with `best_epoch` of 7 against 986 for
action-only. Adding the context made the model strictly worse.

Cause: concatenating a raw 2-d plan onto a standardized 1024-d context makes the action about
23× smaller in norm (‖context‖ ≈ √1024 ≈ 32 vs ‖plan‖ ≈ 1.4). With 140 training samples the
net memorized the context within a handful of full-batch steps, val loss turned up at epoch 7,
and early stopping returned a barely-trained model that had never learned to use the action.

Fixed by widening the plan through a 128-d encoder before the concat and applying dropout 0.5
to the context, selected on validation loss only (test was not consulted). Both are now in
`LatentPredictor`. The epoch budget went to 8000 because the fixed model's val optimum sits
near 7400; 12000 buys a further 0.004 and is not worth the time.

## Next

Scale to all 1000 episodes (~17 min to embed, cache is incremental). Beyond that, the
fixed-target-site limitation above is the thing worth fixing in the data generator before
reading much more into the context contribution.
