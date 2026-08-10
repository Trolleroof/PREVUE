# Measured Results — frozen V-JEPA latents + action-conditioned dynamics

Corpus `data/ur5e_wm` (1000 episodes, 700/150/150 split), checkpoint
`models/latent_dynamics.pt`, seed 0. All numbers are the **test** split with
**compiled** (plan-time) action chunks unless stated — that is the verifier's real
operating condition.

Reproduce:

```bash
uv run python -m waddle_wm.sim.generate_dataset --episodes 5000 --out data/ur5e_wm
```

```bash
uv run python -m waddle_wm.embed_windows --data data/ur5e_wm
```

```bash
uv run python -m waddle_wm.train_latent_dynamics --data data/ur5e_wm
```

New datasets now default to a wider 16x16 cm block spawn box
(`--block-spawn-low 0.30,-0.26 --block-spawn-high 0.46,-0.10`) so the compiled
plan no longer leaks as much about the initial block pose. The trainer also
defaults to `--focus-step 2 --focus-weight 3.0`, which upweights the diagnosed
`z_2 -> z_3` close/lift transition.

## 1. The latent dynamics model works

| metric | value |
| --- | --- |
| one-step latent cosine | **0.787** |
| persistence baseline (`z_hat = z_k`) | −0.037 |
| rollout cosine, steps 1..5 | 0.876, 0.780, 0.794, 0.729, 0.754 |

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

| metric | value |
| --- | --- |
| success accuracy | 0.833 |
| Brier | 0.131 |
| false accepts (of true failures) | 0.223 |
| false rejects (of true successes) | 0.071 |
| ensemble disagreement, correct verdicts | 0.028 |
| ensemble disagreement, wrong verdicts | 0.183 |

Uncertainty is informative: the ensemble disagrees ~6.6x more on the verdicts it
gets wrong, which is exactly the signal a planner would gate on.

## 3. But it is not yet using the scene

A random forest given **only the compiled plan** — no image, no latent — is the
control:

| decision | world model | plan-only control | majority class |
| --- | --- | --- | --- |
| `success` | 0.833 | **0.833** | 0.627 |
| `in_target` | 0.833 | 0.833 | 0.627 |
| `lifted` (grasp holds) | 0.713 | 0.707 | **0.727** |

The world model ties the plan-only control on `success`, and both sit *below* the
majority class on `lifted`. Splitting the test set by which faculty each decision
needs makes it unambiguous:

| test group | n | correct | mean p(success) |
| --- | --- | --- | --- |
| A — target on zone, grasp holds (approve) | 56 | 52 (0.929) | 0.919 |
| B — target off zone (the plan decides) | 71 | 71 (1.000) | 0.000 |
| C — target on zone, grasp misses (**vision decides**) | 23 | **2 (0.087)** | 0.848 |

Group B is perfect and worthless: the place waypoint is in the action chunk, so
arithmetic suffices. Group C is the entire reason to have a world model, and the
verifier confidently approves 21 of 23 plans that will drop the block.

## 4. Where the information is lost

Not in the encoder, and not in long-horizon compounding:

| window | `lifted` from imagined latent | from real latent (oracle) | base rate |
| --- | --- | --- | --- |
| 1 | 1.000 | 1.000 | 0.000 |
| 2 | 1.000 | 1.000 | 0.000 |
| 3 | **0.727** | 0.993 | 0.727 |
| 4 | 0.713 | 0.987 | 0.727 |
| 5 | 0.713 | 0.993 | 0.727 |

The frozen latent carries the grasp outcome almost perfectly (0.993), and block
position decodes from a real latent to **19 mm** RMSE (5.8 mm at window 0). The
imagined latent collapses to the base rate at window 3 — the *first* window after
`close`/`lift` — and never recovers. So it is a **single transition** that fails:
`z_2` (gripper descending, block visible) + "close and lift" -> `z_3`. That step
requires comparing the observed block centre against the commanded grasp waypoint
at ~1 cm resolution, from ~110 training episodes in which the grasp actually
misses. Rollout length is not the problem.

The imagined terminal block position (0.230 m RMSE against a 0.019 m oracle) is
the same failure seen through the regression head: the block ends either where it
started or on the target, ~0.48 m apart, and an MSE head that has not decided
which outputs the midpoint. Conditioning the metric on confident predictions does
not help (0.227 m), because the model is confidently wrong rather than unsure.

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

**Close/lift transition weighting on the old 1000-episode corpus.** Upweighting
`z_2 -> z_3` with `--focus-step 2 --focus-weight 3.0` ran end-to-end, but only
moved the needle slightly on the old narrow-spawn data:

| metric, test compiled-plan chunks | unweighted | focus-weighted |
| --- | --- | --- |
| success accuracy | 0.833 | 0.847 |
| `lifted` accuracy | 0.713 | 0.720 |
| Brier | 0.131 | 0.142 |
| false accepts | 0.223 | 0.234 |

That is not enough. The next fair test is the same weighted trainer on a
regenerated wide-spawn corpus, because the old corpus still lets the plan-only
control exploit the narrow block distribution.

## 6. What to try next, in order

1. **More episodes.** Generation is ~0.7 s/episode and embedding ~1.3 s/episode,
   so 5000 episodes is about 2.5 hours unattended and multiplies the ~110 informative
   grasp-failure examples by five. This is the cheapest lever by a wide margin.
2. **Re-run with the widened block spawn box.** This is now the generator default.
   Existing `data/ur5e_wm` corpora keep their old distribution, so regenerate
   rather than append when measuring the comparison again.
3. **Re-run with close/lift transition weighting.** This is now the trainer
   default via `--focus-step 2 --focus-weight 3.0`; sweep `1, 3, 5` if the 5000
   episode run still collapses to the lifted base rate.
4. Only after those: revisit V-JEPA 2-AC per `backbone_decision.md`. The current
   evidence says the frozen trunk is *not* the limitation.

## 7. Honest summary

The pivot is built and runs: the record shape, the action-conditioned predictor,
the rollout, the uncertainty, and the verifier interface all work end-to-end from
pixels, and the latent prediction itself is clearly real (0.787 vs −0.037). What
is not yet demonstrated is the claim in [`project.md`](project.md) — that an LLM
agent using this verifier can identify and repair bad plans before execution, and
that imagining a skill catches failures a planner could not compute from its own
plan alone. On this corpus the verifier's competence is confined to the part the
plan already determines; the repair loop and three-way comparison are still open.
