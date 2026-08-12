# Claude self-rank vs the estimated-state heuristic vs the visual world model

This is the comparison [issue #18](https://github.com/Trolleroof/skill-level-world-model/issues/18)
asked for: does raw visual context help pick a successful code-as-policy program, beyond
Claude's own judgement of its own programs and beyond a strong geometry rule fed by the real
image-to-coordinate perception pipeline?

The question only means something if the three arms are given identical work. They are:
the same frozen pools from [#17](program_schema.md), the same nested prefixes, the same
outcomes measured by [#23](counterfactual_execution.md) *before any selector ran*, and the
same artifact and validator from [#24](benchmark_protocol.md). No selector executes, repairs,
or regenerates anything.

Files: [`waddle_wm/selectors.py`](../waddle_wm/selectors.py) (the three arms and the
information boundary), [`waddle_wm/benchmark_selectors.py`](../waddle_wm/benchmark_selectors.py)
(the one command, the report, the plot), [`waddle_wm/test_selectors.py`](../waddle_wm/test_selectors.py)
(the contract checks).

```bash
uv run python -m waddle_wm.benchmark_selectors \
    --counterfactual data/counterfactual/test-natural.json --pools data/pools \
    --checkpoint models/multiblock_world_model.pt --out results/programs
```

One command runs all three selectors, writes an artifact `benchmark_record.check_run`
validates, and writes the report and the main plot beside it.

## The arms, and exactly what each one reads

| arm | reads | never reads |
| --- | --- | --- |
| `claude_self_rank` | the task, the observation text the pool was generated from, and the source programs — anonymised and shuffled | coordinates it did not derive itself, frames, outcomes |
| `estimated_state` | each fixed program and its grounded waypoints, plus the perception pipeline's output: detected centres, pixel boxes, apparent size, and the task-frame landing pad | raw frames, simulator state, outcomes |
| `visual_world_model` | everything the heuristic reads, **plus** the raw camera window and the frozen V-JEPA latent computed from it | simulator state, outcomes |

The MuJoCo oracle is the hidden upper bound, not a fourth arm. `first` (take Claude's earliest
sample) and `random` come from #23 and stay in the artifact as the floors any ranking has to
beat.

The boundary is a data structure, not a promise: `ScenarioContext` carries `frames=None` for
every arm that is not the visual model, and `rank` refuses to call an arm that was handed
frames it did not declare. The pool itself is #23's `selector_view`, which is reconstructed
field by field, so a new key in the pool artifact has to be added there on purpose before any
selector can see it.

## What a selector may not do

`rank` enforces the one-shot ranking test rather than trusting it:

- the arm is handed a **copy** of the pool, so a misbehaving selector cannot corrupt what the
  next arm reads;
- that copy is fingerprinted before and after the call — an attempted repair, rewrite, or
  in-place "fix" fails the run;
- the returned rows must be exactly the prefix, each candidate once: no dropping a candidate
  it does not want to judge, no adding one it wrote itself;
- scores must be finite, and the choice is made by #24's locked argmax with the pool-index
  tie-break, never by the selector's own notion of its pick.

Each of those has a negative fixture in `test_selectors.py`.

## Nested prefixes

Pool sizes `N = 1, 4, 16, 32, 64` are nested prefixes of one sample-ordered pool, so pool size
is a variable rather than a confound. A selector at `N` is shown the first `N` candidates and
nothing else — Claude self-rank makes one fresh call per prefix, and the deterministic arms
score candidates independently, which is the same thing. Selector inputs do not depend on the
physics seed (the selector chooses before anything is executed and is never told which
perturbation the executor will apply), so one block per `(pool, prefix)` is recorded and
reused across the paired physics seeds.

## The estimated-state heuristic is not a strawman

The fairness control that matters most here is *do not weaken the coordinate baseline*. So it
gets the perception pipeline's output in full, and a rule that reads every part of a program a
coordinate can reach:

```text
grasp_offset_mm        grasp waypoint vs the detected object centre
grasp_height_mm        descent height vs the height a centred grasp needs
place_offset_mm        release point vs the destination centre
place_margin_mm        how far outside the landing pad the release is aimed
release_height_mm      drop height above the destination surface
lift/carry_clearance   whether the carry clears the block top at all
approach_offset_mm     hover point vs the object
yaw_commanded          whether a wrist heading was pinned
elongated_object       the detection box's aspect ratio
malformed              grasp/release structure missing or out of order
declined / redetects / retry_attempts / object_undetected
```

They go through a logistic whose weights are predeclared geometry (a 10 mm grasp error is
worth about −2.5 logits, a release outside the pad about −6), with a hinge on carry clearance.
`waddle_wm.selectors --fit` refits the same functional form on executed outcomes and
**refuses** any artifact whose split is not `train` or `calibration`.

A candidate that declines to act (`abort`) never executes and is never a success, which is
part of #24's published success definition. Every arm floors it at the same
`DECLINED_PROBABILITY`, so no arm is advantaged by knowing it.

## The visual arm, and two limits recorded rather than hidden

The visual arm is the multi-block state world model: frozen V-JEPA latent of the observation
window + estimated block coordinates + the candidate's grasp and place offsets + the task, in;
predicted terminal state and success probability out, over an ensemble whose spread is the
reported uncertainty. The window is rendered from the pool's own scene and pushed through the
same h264 path the training clips took, and it is encoded once per scenario, not once per
candidate.

Two properties of the checkpoint are recorded in its config and reported rather than papered
over:

- **`orientation_blind`** — its plan encoding carries no wrist heading, so two candidates that
  differ only in `yaw_deg` are indistinguishable to it. On the orientation slice that is a
  real handicap, and it is the checkpoint's, not the protocol's.
- **degenerate normalisation** — the grasp height above the block was *constant* across the
  training set (every recorded episode descended to a fixed height above a block centre read
  from the simulator), so that feature's standard deviation collapsed to the clamp floor.
  Dividing a live estimate's millimetre of perception noise by `1e-6` hands the network a
  five-thousand-sigma input, and it saturated to `p = 0` for *every* candidate — not a
  judgement, an overflow. Dimensions that were constant during fitting are held at their
  training constant and the rest are clamped to the range the fit covered. With the guard the
  arm separates the canonical program from the planted faults; without it, it ranks nothing.
  `test_selectors --live-visual` is the regression test.

## What the report says

`aggregate` (from #24) gives per-prefix, per-selector: `selected_success`,
`selection_efficiency`, `oracle_gap`, `pool_has_success`, `missed_available_success`,
`target_error_gap_mm`, false accepts and rejects, Brier score, mean uncertainty, ranking
latency, and cost — with **paired per-scene differences and bootstrap confidence intervals**,
never two separate averages. Every attempted scene is in `excluded` with its reason.

The main plot is selected-program success against candidate count for every selector, with the
oracle's curve above them.

Results are split by failure slice, using #25's locked labels:

| group | slices | what it tests |
| --- | --- | --- |
| `plan_visible` | malformed sequence, unreachable waypoint, obvious target miss | decidable from the program text; a selector that only wins here has shown nothing about vision |
| `scene_dependent` | grasp misalignment, occlusion, block orientation, path obstruction, prior-action state change (`visible_omitted_by_coordinates`); slip/drop, release dynamics (`latent_physics`) | state a centre coordinate cannot carry |

The **intended slice** for the visual arm is `visible_omitted_by_coordinates`:
`latent_physics` is declared by #25 as not inferable from one initial frame, and
`plan_visible_control` is decidable without any perception at all. The verdict block states
the paired difference, its confidence interval, and the number of paired scenes, and says
"no" — in words — unless the visual arm beat the estimated-state heuristic on that slice with
an interval excluding zero. There is no ordering assumed anywhere in this document or in the
code: the benchmark reports what it measured, per slice and per pool size.
