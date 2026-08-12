# Measured Results — frozen V-JEPA latents + action-conditioned dynamics

Corpus `data/ur5e_wm_wide` (5000 episodes, 3500/750/750 split), checkpoint
`models/latent_dynamics_wide.pt`, seed 0. All numbers are the **test** split with
**compiled** (plan-time) action chunks unless stated — that is the verifier's real
operating condition. Exported metrics live in
[`results/latent_dynamics_wide.json`](../results/latent_dynamics_wide.json); the
narrow 1000-episode numbers they replace are kept inline for comparison.

Reproduce:

```bash
uv run python -m waddle_wm.sim.generate_dataset --episodes 5000 --out data/ur5e_wm_wide
```

```bash
uv run python -m waddle_wm.embed_windows --data data/ur5e_wm_wide
```

```bash
uv run python -m waddle_wm.train_latent_dynamics --data data/ur5e_wm_wide --out models/latent_dynamics_wide.pt
```

```bash
uv run python -m waddle_wm.report_metrics --data data/ur5e_wm_wide --checkpoint models/latent_dynamics_wide.pt --out results/latent_dynamics_wide.json
```

`models/` and `data/` are gitignored, so the checkpoint is not in the repo; the
four commands above regenerate it from scratch. The trainer defaults to
`--focus-step 2 --focus-weight 3.0`, which upweights the diagnosed `z_2 -> z_3`
close/lift transition. The old narrow corpus needed
`--block-spawn-low 0.34,-0.22 --block-spawn-high 0.42,-0.14`; that box is no
longer the generator default.

## 0. The evaluation corpus

Final verifier metrics are reported on `data/ur5e_wm_wide`: **5000 episodes**,
seed 0, generated with the current wide 16x16 cm block spawn box so the compiled
plan leaks less about the initial block pose. It is a fresh corpus, not the old
1000 episodes with 4000 appended.

```bash
uv run python -m waddle_wm.sim.generate_dataset --episodes 5000 --out data/ur5e_wm_wide
```

```bash
uv run python -m waddle_wm.sim.validate_dataset --data data/ur5e_wm_wide
```

| split | n | success | `missed` | `target_miss` |
| --- | --- | --- | --- | --- |
| train | 3500 | 0.373 | 0.313 | 0.314 |
| val | 750 | 0.363 | 0.313 | 0.324 |
| test | 750 | 0.327 | 0.332 | 0.341 |

Split by which faculty each decision needs (the §3 grouping), the wide corpus
multiplies the informative grasp-failure examples roughly 5x, as §6 predicted:

| group | wide train / val / test | old 1k train / val / test |
| --- | --- | --- |
| A — target on zone, grasp holds | 1345 / 276 / 251 | 281 / 59 / 56 |
| B — target off zone (the plan decides) | 1570 / 333 / 361 | 308 / 63 / 71 |
| C — target on zone, grasp misses (**vision decides**) | **585 / 141 / 138** | 111 / 28 / 23 |

`validate_dataset` checks schema version, frame grid, per-frame track lengths,
spawn-box conformance, split sizes and outcome balance, duplicate ids, duplicate
scenes, cross-split scene leakage, and decodes a sample of clips. Full output at
the time of recording — every check passed:

```text
PASS  schema_version: [3]
PASS  episodes: 5000 records, manifest says 5000
PASS  frame grid (total, prelude, window): [(48, 8, 8)]
PASS  per-frame track lengths: [48]
PASS  block spawn box [0.3, -0.26] .. [0.46, -0.1]: observed [0.3001, -0.26] .. [0.46, -0.1001]
PASS  spawn spread, first half [0.1599, 0.1599] vs second half [0.1599, 0.1599]
PASS  splits: {'train': 3500, 'val': 750, 'test': 750}
PASS    train: n=3500 success=0.373 missed=0.313 target_miss=0.314
PASS    val: n=750 success=0.363 missed=0.313 target_miss=0.324
PASS    test: n=750 success=0.327 missed=0.332 target_miss=0.341
PASS  duplicate episode ids: 0
PASS  duplicate scenes: 0
PASS    train n val shared scenes: 0
PASS    train n test shared scenes: 0
PASS    val n test shared scenes: 0
PASS  missing clips: 0
PASS  decoded 50 sampled clips, 0 malformed or blank
```

The spawn-spread check is the one that would catch an append: the first half of
an appended corpus would show the old narrow 8x8 cm range against the second
half's 16x16 cm. Both halves span 0.1599 m, so the whole corpus is wide.
`data/ur5e_wm` fails only the spawn check, because its manifest predates the
flags and records no box.

## 1. The latent dynamics model works

| metric | wide 5k | old narrow 1k |
| --- | --- | --- |
| one-step latent cosine | **0.797** | 0.787 |
| persistence baseline (`z_hat = z_k`) | −0.016 | −0.037 |
| rollout cosine, steps 1..5 | 0.885, 0.794, 0.813, 0.737, 0.763 | 0.876, 0.780, 0.794, 0.729, 0.754 |

Consecutive windows are nearly orthogonal after normalisation, so a persistence
baseline scores zero. The predictor recovers most of the change from the action
chunk, and free-running rollout degrades gently over 4.0 s.

Training on one-step teacher forcing alone and evaluating a 5-step rollout was a
real train/eval mismatch; adding the free-running rollout term moved Brier from
0.148 to 0.122 and roughly doubled the gap between ensemble disagreement on right
vs wrong verdicts.

## 2. The verifier runs end-to-end from pixels

`waddle_wm.verifier` encodes the 8-frame pre-execution window, compiles the
proposed trace, rolls the ensemble forward 5 steps, and decodes the imagined
terminal latent. No cached outcome, no label.

| metric | wide 5k | old narrow 1k |
| --- | --- | --- |
| success accuracy | 0.852 | 0.833 |
| `in_target` accuracy | 0.851 | 0.833 |
| `lifted` accuracy | 0.720 | 0.713 |
| Brier | 0.119 | 0.131 |
| false accepts (of true failures) | 0.176 | 0.223 |
| false rejects (of true successes) | 0.090 | 0.071 |
| ensemble disagreement, correct verdicts | 0.042 | 0.028 |
| ensemble disagreement, wrong verdicts | 0.159 | 0.183 |

Uncertainty is still informative: the ensemble disagrees ~3.8x more on the
verdicts it gets wrong, which is the signal a planner would gate on — though the
margin is narrower than the ~6.6x on the narrow corpus.

The table is computed from cached window embeddings, which are the same encoder
outputs `verifier.py` produces from pixels. Spot-checking the CLI against the new
checkpoint agrees:

```bash
uv run python -m waddle_wm.verifier --episode ur5e_4999 --data data/ur5e_wm_wide --checkpoint models/latent_dynamics_wide.pt
```

```text
approve: true  p(success) 0.725  uncertainty 0.337  lifted 0.997  in_target 0.726
actual outcome: success
```

## 3. But it is not yet using the scene

A random forest given **only the compiled plan** — no image, no latent — is the
control:

| decision | world model | plan-only control | majority class |
| --- | --- | --- | --- |
| `success` | 0.852 | **0.855** | 0.673 |
| `in_target` | 0.851 | 0.855 | 0.673 |
| `lifted` (grasp holds) | 0.720 | 0.717 | 0.668 |

Five times the data and the close/lift weighting did not break the tie: the world
model is 0.003 *behind* the plan-only control on `success` and 0.003 ahead on
`lifted`, both inside run-to-run noise. Splitting the test set by which faculty
each decision needs makes it unambiguous:

| test group | n | correct (wide 5k) | mean p(success) | correct (old 1k) |
| --- | --- | --- | --- | --- |
| A — plan on zone, grasp holds (approve) | 251 | 227 (0.904) | 0.859 | 52/56 (0.929) |
| B — plan off zone (the plan decides) | 361 | 361 (1.000) | 0.000 | 71/71 (1.000) |
| C — plan on zone, grasp misses (**vision decides**) | 138 | **51 (0.370)** | 0.610 | 2/23 (0.087) |

Group B is perfect and worthless: the place waypoint is in the action chunk, so
arithmetic suffices. Group C is the entire reason to have a world model. Every
episode in it fails, so a verifier that rejected everything would score 1.000
there; this one still approves 87 of 138 plans that will drop the block.

Group C did improve — 0.087 to 0.370 — which is the only place the wide corpus
and the focus weighting show up at all. It is real movement in the right
direction and still far short of useful, and it is invisible in the aggregate
because group C is 18% of the test split.

## 4. Where the information is lost

Not in the encoder, and not in long-horizon compounding:

| window | `lifted` from imagined latent | from real latent (oracle) | majority class |
| --- | --- | --- | --- |
| 1 | 1.000 | 1.000 | 1.000 |
| 2 | 1.000 | 1.000 | 1.000 |
| 3 | **0.717** | 0.995 | 0.668 |
| 4 | 0.712 | 0.988 | 0.668 |
| 5 | 0.720 | 0.991 | 0.668 |

The frozen latent carries the grasp outcome almost perfectly (0.991), and block
position decodes from a real latent to **30 mm** RMSE. The imagined latent
collapses at window 3 — the *first* window after `close`/`lift` — and never
recovers. So it is a **single transition** that fails: `z_2` (gripper descending,
block visible) + "close and lift" -> `z_3`. That step requires comparing the
observed block centre against the commanded grasp waypoint at ~1 cm resolution.
Rollout length is not the problem.

What the wide corpus changed here is the size of the gap over the base rate, not
its shape: on the narrow corpus window 3 landed exactly on the base rate (0.727
vs 0.727), and it now clears it by 5 points (0.717 vs 0.668). The oracle gap is
still ~0.27, so the information is present in the latent and lost in the
transition.

The imagined terminal block position (0.222 m RMSE against a 0.030 m oracle) is
the same failure seen through the regression head: the block ends either where it
started or on the target, ~0.48 m apart, and an MSE head that has not decided
which outputs the midpoint. Conditioning the metric on confident predictions does
not help (0.221 m), because the model is confidently wrong rather than unsure.

## 5. What was tried and did not work

**Grounding the rollout with the detached readout.** Latent regression optimises
the bulk of the latent — arm pose, which moves identically whether or not the
grasp caught the block — so the natural fix is to also penalise what the imagined
latent *decodes to*, with the readout's parameters detached so it cannot adapt to
the dynamics' errors (`--grounding-weight`, implemented via
`torch.func.functional_call`). It made things worse at every weight tried:

| grounding weight | success | `lifted` |
| --- | --- | --- |
| 0.0 | 0.860 | 0.733 |
| 0.2 | 0.773 | 0.567 |
| 1.0 | 0.747 | 0.607 |

(Sweep predates the `--seed` flag, so init differs per run; run-to-run spread on
`success` is about ±0.02, which is why §3 leans on the group decomposition rather
than on third-decimal comparisons.) The knob is kept, defaulted to 0.

**Richer pooling was not needed.** Before spending 20 minutes re-embedding with
4x4 grid pooling (`--pool grid`), the window-0 oracle readout was measured at
5.8 mm median error — mean pooling already localises the block far better than
the ~28 mm decision threshold. The bottleneck is the transition, not the encoding.

**Close/lift transition weighting.** Upweighting `z_2 -> z_3` with `--focus-step 2
--focus-weight 3.0` moved the needle only slightly on the old narrow-spawn data:

| metric, test compiled-plan chunks | unweighted | focus-weighted |
| --- | --- | --- |
| success accuracy | 0.833 | 0.847 |
| `lifted` accuracy | 0.713 | 0.720 |
| Brier | 0.131 | 0.142 |
| false accepts | 0.223 | 0.234 |

The fair test was the same weighted trainer on the wide-spawn corpus, since the
narrow corpus still let the plan-only control exploit the tight block
distribution. That run is what §1-§4 now report: the weighting plus 5x the
group-C examples lifts group C from 0.087 to 0.370 and window 3 off the base
rate, and still does not beat the plan-only control on any aggregate decision.
Sweeping `--focus-weight 1, 3, 5` is the remaining cheap variation, but the §4
diagnostic says the ceiling is the transition's form, not how hard it is
weighted.

## 6. What to try next, in order

1. ~~**More episodes.**~~ Done: `data/ur5e_wm_wide` in §0 is 5000 episodes and
   carries 585 informative group-C training examples, up from 111.
2. ~~**Re-run with the widened block spawn box.**~~ Done: the same corpus is
   generated with the wide box, freshly rather than appended, validated, and
   embedded (`window_embeddings.pt` covers all 5000, `[6, 1024]` per episode),
   trained, and re-measured — §1-§4 are that run.
3. ~~**Re-run with close/lift transition weighting.**~~ Done: the trainer default
   `--focus-step 2 --focus-weight 3.0` trained on the wide corpus is what §1-§4
   report. Group C moved 0.087 -> 0.370; the aggregate tie with the plan-only
   control did not break.
4. **Change the transition, not the data.** Three exhausted knobs (more episodes,
   a wider spawn box, focus weighting) all left the `z_2 -> z_3` gap at ~0.27
   against the oracle, so the next attempt should change what that step can
   express — a discrete lift/no-lift branch rather than one residual MLP asked to
   output a bimodal future through an MSE loss.
5. Only after that: revisit V-JEPA 2-AC per `backbone_decision.md`. The current
   evidence says the frozen trunk is *not* the limitation — the oracle readout
   still recovers `lifted` at 0.991.

## 7. Honest summary

The pivot is built and runs: the record shape, the action-conditioned predictor,
the rollout, the uncertainty, and the verifier interface all work end-to-end from
pixels, and the latent prediction itself is clearly real (0.797 vs −0.016).

Half the claim in [`project.md`](project.md) is now demonstrated and half is not.
[`demo.md`](demo.md) closes the loop end to end: an LLM agent using this verifier does
identify and repair bad plans before execution, turning 0/8 unverified failures into
6/8 successes on identical scenes. But a deterministic geometry rule with no image
access scores 7/8 on those same scenes, so *verifying* the plan is worth it while
*imagining it visually* is still not shown to be. On this corpus the verifier's
competence remains confined to the part the plan already determines.
