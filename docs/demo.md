# The demo, and what it does and does not prove

This is the results page for [issue #19](https://github.com/Trolleroof/skill-level-world-model/issues/19):
one command that runs the closed loop end to end, the saved traces it produced, and an honest
account of which part of the project's claim it supports.

The claim under test, from [`project.md`](project.md):

> Can an LLM use an action-conditioned visual world model to identify and repair robot plans
> before execution?

Short answer from this page: **the loop works and is worth having — a plan that fails in physics
is caught, repaired by Claude, and then succeeds — but nothing here shows that the *visual* world
model is what makes it work.** A deterministic geometry rule with no image access does the same job
slightly better on the same scenes. The demo supports "imagining the plan before running it helps";
it does not support "imagining it *visually* helps".

## Reproduce it

```bash
uv run python -m waddle_wm.demo
```

Three arms — no verifier, the deterministic rules, the learned visual world model — each handed the
**same** deliberately flawed opening plan on the **same** scene, each running whatever it decided to
run in MuJoCo. About 40 s and $0.19 on Opus 5. Traces, videos, and the generated table land in
`results/demo/`.

The `none` arm is what makes this a claim rather than an anecdote: it executes the flawed plan
unverified, so the outcome the other two arms avoided is *measured*, not asserted.

Claude is the only non-deterministic part, so every plan it returned is stored in the trace and can
be re-run without spending a token:

```bash
uv run python -m waddle_wm.demo --replay results/demo
```

That re-verifies and re-executes the recorded plans and prints how far the replayed verdicts drifted
from the recorded ones. On the traces in this repo the drift is **0.0000** and there are no outcome
mismatches — the verifier and the simulator are deterministic; only the planner is not.

The rate table below comes from the sweep, which is the same thing over more scenes (about 6 minutes
and $1.67):

```bash
uv run python -m waddle_wm.demo --sweep 8
```

Offline contract checks, including the regression in §5, need neither MuJoCo nor Claude:

```bash
uv run python -m waddle_wm.test_demo --checkpoint models/multiblock_world_model.pt --traces results/demo
```

Prerequisite: `models/multiblock_world_model.pt`, which is gitignored. Train it with the command in
[`llm_agent.md`](llm_agent.md#multi-block-verifier-status). The `none` and `rules` arms need no
checkpoint at all.

## 1. The headline run

Scene: red block at (0.420, −0.200), camera-detected at (0.415, −0.195). The opening plan aims the
grasp **6 cm past the block in +x** — the flaw the whole project is about, because it is invisible in
the plan itself and only a look at the scene distinguishes it from a good plan.

| verifier | opening verdict | final verdict | MuJoCo outcome |
| --- | --- | --- | --- |
| none | _not verified_ | _not verified_ | **failure (`missed`)** — block nudged to z=0.025, ended 0.510 m from the pad |
| rules | p=0.000 reject — "grasp misses the block" | p=1.000 approve | success, 0.023 m from the pad centre |
| world-model | **p=0.073, u=0.085 reject** — "grasp misses the block" | **p=0.947, u=0.020 approve** | success, 0.023 m from the pad centre |

The world-model arm is the loop issue #19 asked for, and every step of it is real:

1. **Imagined failure.** p(lifted) = 0.432, p(in landing zone) = 0.168, p(success) = 0.073 — rejected.
2. **Repair.** Claude gets that verdict and its own plan back, and changes one thing:
   *"re-aimed the grasp waypoints (approach, descend, lift) from x=0.475 to the observed red block
   centre x=0.415, y=−0.195"*.
3. **Approval.** The repaired plan imagines p(success) = 0.947 at ensemble uncertainty 0.020.
4. **Execution.** MuJoCo lifts the block to z=0.269 and lands it 0.023 m from the pad centre.

Uncertainty behaves the way the design wants: 0.085 and 0.232 on the two rejected plans, 0.020 on the
approved one. The ensemble is most divided exactly where the verdict is closest.

The second scenario, `place_miss`, releases 22 cm short of the pad. It is the *easy* axis — the place
waypoint is in the plan, so arithmetic suffices — and all three verdicts follow the same shape.
Full table: [`results/demo/report.md`](../results/demo/report.md).

## 2. Over 8 scenes, not one

```text
| scenario   | verifier    | caught | repaired to success | success rate |
| grasp_miss | none        | 0/8    | 0/8                 | 0.00         |
| grasp_miss | rules       | 8/8    | 7/8                 | 0.88         |
| grasp_miss | world-model | 8/8    | 6/8                 | 0.75         |
| place_miss | none        | 0/8    | 0/8                 | 0.00         |
| place_miss | rules       | 8/8    | 7/8                 | 0.88         |
| place_miss | world-model | 8/8    | 6/8                 | 0.75         |
```

Both verifiers caught the flawed opening plan in **every** scene. The gap is in what happens next,
and the four losses have exactly two causes:

- **Scene 1 costs every arm, including `rules`.** The red block sits at (0.597, −0.114) and the
  camera puts it at (0.581, −0.102) — a **19.4 mm** lateral error, against 3.9–8.5 mm in the other
  seven scenes. Claude repairs the grasp onto the estimate, both verifiers approve it (p=1.000 and
  p=0.950), and the fingers close on nothing. This is a **perception** failure that the verifiers
  ratify, not a verification failure — and it is the one case in the sweep where a verifier that
  really read the pixels could have known something the coordinates could not. It did not.
- **Scene 4 costs the world model alone.** It rejects the flawed plan, then rejects both of Claude's
  repairs (p=0.441, p=0.408) although the first repair is a good plan, and the loop halts with
  nothing executed. That is the documented conservatism — held-out false-reject rate **0.679** — and
  it is the price of the learned arm: it does not break things, it refuses to do them.

## 3. What this proves

- **The closed loop is real and end-to-end.** Camera → Claude → imagined future → repair → approval →
  physics, one command, no privileged simulator state anywhere in the decision path. The verifier
  never sees an outcome label; object positions come from the depth buffer.
- **Pre-execution verification converts guaranteed failures into successes.** 0/8 unverified versus
  6/8 and 7/8 verified, on identical scenes with an identical opening plan. This is the project's
  central mechanism working.
- **The repair step is doing real work.** Claude is not resampling until something passes: in every
  scene it changed exactly the waypoint the verdict implicated, and said which one and why.
- **Uncertainty is informative in the loop**, not just in the offline table — high on contested
  verdicts, near zero on confident approvals.

## 4. What this does not prove

- **Not that vision is what helps.** The rules verifier — pure geometry, no image, no learning — beat
  the world model on both scenarios (0.88 vs 0.75). Nothing on this page is evidence for the visual
  half of the claim. The aggregate offline picture agrees: [`results.md`](results.md) §3 has the
  world model tied with a plan-only control that never sees an image, and
  [`llm_agent.md`](llm_agent.md#multi-block-verifier-status) has rules at 0.933 held-out accuracy
  against the learned model's 0.674.
- **Not a general result.** Two flaw types, eight scenes, one seed, one block, one task shape, and
  flaws that were *inserted* rather than encountered. The locked benchmark in
  [issue #18](https://github.com/Trolleroof/skill-level-world-model/issues/18) is what settles
  whether visual context adds anything over an estimated-state heuristic; this page is a
  demonstration, not that measurement.
- **Not that the imagined future is quantitatively right.** The verdict is useful, the imagined
  *coordinate* is not: on the approved plan the model imagines the block finishing at
  (0.458, 0.082) and it actually finishes at (0.503, 0.278) — 0.2 m out, matching the 0.200 m
  held-out RMSE. Read the probability, not the position.
- **Not that grasp failure is solved.** The world model rejected all 8 offset grasps here, but these
  are 6 cm offsets. `lifted` accuracy is 0.720 offline and the group where vision has to decide
  still runs at 0.370. A 6 cm miss is a demonstration, not the hard case.

## 5. The bug this demo found

Before any of the above could run, every world-model verdict in the live loop came back
`p(success) = 0.000`, "grasp misses the block", with the imagined block position tens of metres off
the table — for correct plans exactly as readily as for flawed ones. The learned verifier was not
conservative, it was **dead**, and nothing in the offline metrics showed it because the offline path
never hit the trigger.

`train_multiblock_world_model` normalises features with `std.clamp_min(1e-6)`. One plan feature,
`grasp_z − block_z`, is *identical in all 900 training episodes*: every recorded trace descends to
`GRASP_Z`, and every recorded block height is read out of MuJoCo. Its stored "std" is therefore the
clamp floor rather than a scale. Offline that is harmless — the feature is constant there too. Live,
the block height comes from the depth buffer instead of the simulator, and that few-millimetre
disagreement divided by 1e-6 arrived at the ensemble as **~3500 sigma**. One saturated input pinned
every verdict.

`Verifier.constant_features` now feeds a feature that never varied in training the value training
actually saw, instead of an exploded one. `test_demo.check_constant_feature_guard` holds the fixture
and asserts the checkpoint still has such a feature.

Worth stating plainly: the offline numbers in [`results.md`](results.md) and
[`llm_agent.md`](llm_agent.md) were never affected, and the rules verifier never was either — but
for as long as the bug existed, `--verifier world-model` in the browser demo and the agent CLI
rejected everything, and the honest reading is that the live learned path had not been exercised
end-to-end since the multi-block checkpoint landed.

## 6. Where this leaves the project

The demo closes the loop the project set out to build and gives it a reproducible artifact. The open
question is unchanged and now sharper: **the loop's value so far comes from checking the plan, not
from imagining it visually.** The one scene in this sweep where pixels would have beaten coordinates
— the 19.4 mm perception error — is the case the visual verifier also got wrong, which is a cleaner
statement of the gap than any aggregate here.

Per issue #19, V-JEPA 2-AC stays out of scope: nothing in these traces implicates the frozen
backbone. [`results.md`](results.md) §4 still localises the loss to a single `z_2 → z_3` transition
whose oracle readout recovers `lifted` at 0.991, and §6 item 4 — give that transition a discrete
lift/no-lift branch instead of one MSE-trained residual MLP — remains the next thing to try.
