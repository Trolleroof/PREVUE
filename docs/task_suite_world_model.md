# A yaw-aware world model for multi-step pick and place

This is the corpus, the model, and the measured verifier behaviour for the task suite: several
pick-and-place families — place on the pad, stack, place-then-place, place-then-stack — where
every grasp commands a wrist heading and the heading decides whether the grasp is possible.

It exists because of one line in [`results.md`](results.md) §8. On the `block_orientation`
slice, *every* deployable selector picked a losing program on both scenes while a coin flip
found the winner once, and the reason was named precisely:

> the checkpoint's plan encoding carries no wrist heading (`orientation_blind` in its recorded
> config), so two candidates differing only in `yaw_deg` are the same input to it. That is a
> concrete, named reason to build a yaw-aware scorer, not a reason to re-run this one.

This is that scorer, plus the corpus it needs, plus an evaluation designed so that "the vision
helped" is a claim that can fail.

## 1. Why the old corpora could not settle it

An earlier yaw-aware checkpoint exists — `models/multiblock_world_model_yaw.pt`, trained on
`data/ur5e_wm_oriented`. Its recorded metrics are the reason for a new one:

| | that checkpoint | reference |
| --- | --- | --- |
| test success accuracy | 0.600 | plan-only forest **0.652**, majority class 0.622 |
| false reject rate | 0.843 | — |
| best epoch | 31 of 1200 | — |

It is below the majority class and below a control with no image access, it rejects five of
every six plans that would have worked, and it stopped improving after 31 epochs on 900
episodes. Three things were wrong, and all three are corpus-level:

**The heading did not change the outcome.** In the oriented corpus the source block is
60 x 22 mm. A Robotiq 2F-85 opens to about 85 mm, so it closes on that block at *every*
heading. Measured on a probe of this suite built to the same size: lift rate 1.00 for aligned
grasps and 0.80 at more than 60 degrees off — nearly no effect. A model cannot learn to read a
heading that does not matter, and an evaluation cannot detect whether it did.

**Only one block was rectangular.** A second grasp in the same episode would have been
yaw-blind by construction, so multi-step tasks and yaw could not coexist.

**One block, one place, one outcome.** `env` tracks a single block and reports one success
flag, which cannot express "the first place worked and the second missed" — so "place the blue
block, then the red one" was not expressible, let alone scoreable.

## 2. The corpus

[`waddle_wm/sim/generate_suite.py`](../waddle_wm/sim/generate_suite.py) writes schema 5.

**Blocks are 100 x 22 x 36 mm and all three are identical.** The length is the load-bearing
choice: at 100 mm it exceeds the 85 mm jaw stroke, so a grasp commanded across the block is not
merely untidy, it is impossible. At a misalignment of `t` the jaws must span roughly
`100 sin t + 22 cos t` mm, which crosses the stroke near 40 degrees, so misalignment degrades
smoothly into impossible rather than switching at a threshold. Measured on the corpus itself:

| grasp misalignment | lift rate |
| --- | --- |
| 0-5 deg | 1.00 |
| 20-40 deg | 0.82 |
| 40-60 deg | 0.45 |
| 60-90 deg | 0.04 |

The height is unchanged from the cube corpus, so a stack clears the same threshold.

**Four families**, cycled so every split holds all of them:

| family | subtasks | what it adds |
| --- | --- | --- |
| `pad_place` | 1 | the baseline pick and place |
| `stack` | 1 | a destination that is another block, not a pad |
| `ordered_pad` | 2 | "place the blue block, then the red one" — two spots on the pad |
| `ordered_stack` | 2 | clear a spot, then stack the second block on the first |

`ordered_stack` is the one that makes the plan a *sequence* rather than two independent
requests: the second subtask's destination is wherever the first subtask **planned** to leave
its block, not where that block spawned. A plan whose first half aims wide has moved the
target its second half is aiming at, and nothing in the second half's own waypoints says so.

**Every subtask commands a wrist heading**, aligned with the block's own about half the time
and 25-155 degrees across it otherwise, and the block's heading is sampled independently per
block. Faults are planted per subtask — a lateral grasp offset, a placement offset — at lower
rates for the two-subtask families, since those need both halves to work.

**Outcomes are per subtask, recomputed from the per-frame tracks.** A subtask succeeds if its
block was lifted clear of the table and ended within its destination's radius, and for a block
destination, resting on top of it. The episode succeeds only if every subtask does, and records
which one broke first. The validator recomputes every label from the trajectory and refuses the
corpus if any disagrees — a label that cannot be rederived from the trajectory is one nobody
can audit.

Stacking tolerance is set from measured precision, not inherited: the arm lands 12 mm from its
aim at the median and 31 mm at p90, and a stack inherits the support block's error too, so the
cube corpus's 27 mm radius rejected blocks visibly resting on the support. "Stacked" here means
the centre is within the support's half-length (50 mm) and at least half a block height above
it.

## 3. What the model is

[`waddle_wm/train_task_suite_world_model.py`](../waddle_wm/train_task_suite_world_model.py).

```text
observation window ──> frozen V-JEPA 2 ViT-L ──> context
                                                    │
block coordinates ──────────────────────────────────┼──> scene belief h₀
                                                    │
                    subtask 1 (plan + task) ──> GRU ──> h₁ ──> lifted, placed, success
                    subtask 2 (plan + task) ──> GRU ──> h₂ ──> lifted, placed, success
                                                        │
                                                        └──> final position of every block

                    p(plan succeeds) = Π p(subtask succeeds)
```

The subtasks are consumed one at a time by a recurrent cell whose hidden state is the model's
belief about the scene *after* each one, which is what lets the second subtask be scored
against a world the first one changed. The episode's probability is the product of its
subtasks', because that is what the label means — not a separate head that has to rediscover it.

Each subtask's plan is the existing yaw-aware [`plan_encoding`](../waddle_wm/plan_encoding.py)
v2 vector: grasp offset from the block, place offset from the destination, and `sin 2y, cos 2y`
of the commanded grasp and approach headings. The trainer refuses to write an
orientation-blind checkpoint at all.

**Vision has exactly one job.** The model is *given* every block's coordinates. The one thing
coordinates omit is each block's heading — and the heading is what decides the grasp. So the
`--no-context` ablation is not a nuisance control, it is the whole claim: same architecture,
same coordinates, same plans, no pixels.

## 4. Reproduce

Generation is sharded because a single process renders about one 88-frame episode per second.

```bash
for i in $(seq 0 9); do uv run python -m waddle_wm.sim.generate_suite --episodes 500 --start $((i*500)) --seed $((i*1000+7)) --out data/ur5e_wm_suite & done; wait
```

```bash
uv run python -m waddle_wm.sim.validate_suite --data data/ur5e_wm_suite
```

```bash
uv run python -m waddle_wm.embed_windows --data data/ur5e_wm_suite --pool mean,grid2 --windows 1
```

```bash
uv run python -m waddle_wm.train_task_suite_world_model --data data/ur5e_wm_suite --out models/task_suite_world_model.pt
```

```bash
uv run python -m waddle_wm.report_task_suite --data data/ur5e_wm_suite --checkpoint models/task_suite_world_model.pt --out results/task_suite_world_model.json
```

`--windows 1` caches only the pre-execution observation window, which is all this model reads;
the latent-dynamics rollout in [`results.md`](results.md) needs every window and should omit the
flag.

Tests, neither of which needs the corpus:

```bash
uv run python -m waddle_wm.sim.test_generate_suite
```

```bash
uv run python -m waddle_wm.test_suite_verifier
```

## 5. Serving

[`waddle_wm/suite_verifier.py`](../waddle_wm/suite_verifier.py). `waddle_wm.verifier.Verifier`
serves the single-subtask checkpoints and refuses this one by name, because its
`verify(object, destination)` signature cannot express a sequence.

```python
from waddle_wm.suite_verifier import SuiteVerifier

verifier = SuiteVerifier("models/task_suite_world_model.pt")
verdict = verifier.verify_frames(frames, trace, subtasks, positions)
# verdict.approve, .success_probability, .uncertainty
# verdict.subtasks[k].lifted_probability / .likely_failure / .suggestion
# verdict.blocking_subtask -> which step to repair first
```

Two things it does that a caller must not skip, both learned the hard way in this repo:

* **Frames go through h264 before the encoder.** Every cached training embedding came out of a
  decoded `.mp4`; raw renderer output is off-distribution, and by a lot — the same true-positive
  plans scored 0.17/0.30/0.12 raw against 0.98/0.86/1.00 through the codec. `verify_frames`
  does the round trip.
* **Features that were constant during fitting are held at zero, not divided.** That is the bug
  that once made a checkpoint answer `p = 0` to every candidate. Training and serving normalise
  through the same function with the same +-5 sigma clamp, and
  [`test_suite_verifier`](../waddle_wm/test_suite_verifier.py) asserts the two paths produce
  bit-identical features.

## 6. Measured results

Corpus `data/ur5e_wm_suite` (5000 episodes, 3512/744/744), checkpoint
`models/task_suite_world_model.pt`, seed 0, 5 members. Test split, 744 episodes, decision
threshold 0.610 chosen on **val** as the most accurate point with a false-accept rate under
0.10. Full artifact: [`results/task_suite_world_model.json`](../results/task_suite_world_model.json).

### The headline

| arm | accuracy | Brier | false accept | false reject |
| --- | --- | --- | --- | --- |
| majority class | 0.585 | — | — | — |
| plan-only forest | 0.676 | 0.195 | 0.170 | 0.540 |
| no-vision ablation (trained) | 0.691 | 0.179 | 0.126 | 0.566 |
| geometry rule | 0.694 | 0.306 | 0.200 | 0.456 |
| oracle-heading forest | 0.718 | 0.184 | 0.138 | 0.485 |
| **world model** | **0.778** | **0.129** | **0.097** | **0.398** |

The world model is the best arm on every column: most accurate, best calibrated, and it both
accepts fewer failures *and* rejects fewer successes than anything else. The geometry rule that
won the comparison in [`results.md`](results.md) §8 is beaten by 8.4 points.

### The claim: does vision buy anything?

`orientation_discrimination` is the slice built to answer it — plans well aimed in coordinates,
split into those commanded along the block (which mostly succeed) and across it (which mostly
fail). The coordinates are drawn from the same distribution in both; the only systematic
difference is a heading that exists in the pixels and nowhere else.

| n = 227 | world model | no-vision | plan-only | oracle heading |
| --- | --- | --- | --- | --- |
| accuracy | **0.762** | 0.634 | 0.590 | 0.656 |
| AUC | **0.864** | 0.647 | 0.615 | 0.718 |

**+0.217 AUC over a no-vision control that is the same architecture trained from scratch
without pixels.** That is the result [`results.md`](results.md) §8 called for and did not get.
The visual arm also passes the oracle-heading forest, which knows the true headings but reads
them through a random forest over flat features rather than through a model that can form the
heading-versus-command interaction directly.

### Per family, and per decision

| family | model | no-vision | plan-only | oracle |
| --- | --- | --- | --- | --- |
| `pad_place` | **0.817** | 0.737 | 0.677 | 0.720 |
| `stack` | **0.828** | 0.672 | 0.699 | 0.780 |
| `ordered_pad` | **0.683** | 0.656 | 0.634 | 0.661 |
| `ordered_stack` | **0.785** | 0.699 | 0.694 | 0.710 |

Per-subtask decisions, over all 1116 subtasks in the split:

| decision | accuracy | base rate |
| --- | --- | --- |
| `lifted` | 0.806 | 0.679 |
| `placed` | 0.813 | 0.528 |
| `success` | 0.803 | 0.526 |

`lifted` is the axis [`results.md`](results.md) §6 named as "the weak axis" at 0.720, where an
offset grasp was often approved. It is 0.806 here, and it is the decision the heading feeds.

Ensemble disagreement is 1.74x higher on wrong verdicts than on right ones (0.196 vs 0.113) —
informative, and a weaker separation than the 3.8x the latent-dynamics model showed.

Imagined final block positions land 0.098 m RMSE / 0.084 m median from the truth, against the
0.211 m of the earlier multi-block model. Better, and still far too coarse to plan from: this
model earns its keep as a verifier, not as a simulator.

## 7. What this does and does not show

**Does.** On this corpus, a verifier that looks at the scene beats every control that does not,
including a same-architecture model trained without pixels, a random forest given every
coordinate, and the deterministic geometry rule that beat the previous visual model. It does so
on the aggregate, on every task family, and by 0.217 AUC on the slice where coordinates are
matched and only the heading differs. It handles two-step plans and says *which* step it
expects to break.

**Does not.**

* **The corpus was built so that vision is decisive.** Blocks are longer than the jaw stroke, so
  heading is not a nicety but the difference between a grasp and no grasp. That is a fair test
  of "can a visual verifier read what coordinates omit", and it is not a claim about how often
  orientation decides a real pick.
* **Heading extraction is architectural, not emergent.** The visual pathway is a ridge probe
  fitted on the training split against simulator heading labels. Nothing leaks at inference —
  the verifier sees pixels, coordinates and the plan — but the model was *told* which visual
  quantity to extract. Trained on the binary outcome alone, with the full latent, it did not
  find it: measured at 0.645 accuracy and 0.660 AUC on the orientation slice against its own
  control's 0.662, which is the same null result the earlier attempts got.
* **The failures are engineered.** Faults are planted by the generator, not produced by a
  planner, so the 0.42 base success rate describes the harness and not a robot.
* **Uncertainty is weaker than the latent-dynamics model's** — 1.74x versus 3.8x — so gating on
  ensemble disagreement will be correspondingly blunter.
* **One seed.** Every number above is seed 0. The slice-level cells with a near-degenerate base
  rate (`well_aimed_and_aligned` at 0.99, `misalignment_60_91deg` at 0.01) carry AUCs computed
  from one or two minority examples and should be read as noise; that is why the claim rests on
  `orientation_discrimination`, whose base rate is 0.63.

### What was tried and did not work

Recorded because each cost a run and each is a plausible next idea:

| attempt | result |
| --- | --- |
| full 1024-d latent as context | best epoch 3-7, 0.645 accuracy, **worse than its own no-vision control** on the orientation slice |
| PCA-32/64/128 of the latent | still no vision gap; 11 of 12 configs negative |
| auxiliary heading head, full latent | not enough on its own — the latent is memorised before the head can shape it |
| test-time context zeroing as the ablation | **invalid** — it feeds a model an input it was never fitted on, and scored *better* than the visual model, which is a diagnostic of the mistake, not a finding |

The reason PCA fails is worth keeping: it retains the highest-*variance* directions, which in
these latents are block positions and arm pose — exactly what the model is already handed as
coordinates. Measured, a ridge readout of heading degrades from 12.2 to 21.5 degrees median
error when restricted to 32 principal components. The heading lives in low-variance directions,
so the projection has to be supervised.
