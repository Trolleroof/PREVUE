# The program schema and the candidate pools

This is the contract [issue #17](https://github.com/Trolleroof/skill-level-world-model/issues/17)
asked for: what a *candidate* is, what Claude is allowed to choose, and how a pool of
candidates is generated once and then frozen so that every verifier in
[#18](https://github.com/Trolleroof/skill-level-world-model/issues/18),
[#23](https://github.com/Trolleroof/skill-level-world-model/issues/23) and
[#26](https://github.com/Trolleroof/skill-level-world-model/issues/26) ranks the same
thing.

```text
observation (camera only)
        |
        v
  Claude, N independent samples
        |
        v
  program            symbolic: detect("red block"), offsets in mm, no coordinates
        |  validate  schema, known API, bounded parameters, structure
        v
  ground(observation)   the one place coordinates appear
        |
        v
  Plan / SkillStep -> compile_plan -> actions        (verifier)
                   -> run_trace    -> MuJoCo         (execution)
```

Two files: [`waddle_wm/program.py`](../waddle_wm/program.py) is the schema, the grounder,
and the diagnostic suite; [`waddle_wm/pools.py`](../waddle_wm/pools.py) generates, caches,
and integrity-checks the pools.

```bash
uv run python -m waddle_wm.program --seed 0                    # ground the canonical program
uv run python -m waddle_wm.pools --split test --pool-size 64   # the locked pools
uv run python -m waddle_wm.pools --validate data/pools         # integrity only
uv run python -m waddle_wm.test_program --live 3               # contract + diagnostics in MuJoCo
```

## A program is symbolic

A candidate never contains a coordinate. It contains symbols bound by `detect` and offsets
in millimetres from them, so the same program means different waypoints in different
scenes and Claude cannot smuggle simulator truth into a candidate. `test_program` asserts
this directly: no scene coordinate may appear anywhere in the program source.

```json
{"schema_version": 1,
 "strategy": "straight_pick_place",
 "task": {"object": "red block", "destination": "green pad"},
 "ops": [{"op": "detect", "query": "red block", "as": "src"},
         {"op": "detect", "query": "green pad", "as": "dst"},
         {"op": "move_above", "ref": "src", "offset_mm": [0, 0], "height_mm": 240,
          "direction": "top", "standoff_mm": 0},
         {"op": "descend_to", "ref": "src", "offset_mm": [0, 0], "height_mm": 15},
         {"op": "grasp"},
         {"op": "lift", "height_mm": 240},
         {"op": "move_above", "ref": "dst", "offset_mm": [0, 0], "height_mm": 300,
          "direction": "top", "standoff_mm": 0},
         {"op": "descend_to", "ref": "dst", "offset_mm": [0, 0], "height_mm": 15},
         {"op": "release"},
         {"op": "retreat", "height_mm": 240}],
 "note": "why these numbers"}
```

## The API, and what Claude gets to choose

Eight operations, and nothing else compiles. Anything outside this list is a static reject
before the candidate reaches a verifier.

| op | what it does | bounded parameters |
| --- | --- | --- |
| `detect(query, as)` | camera lookup, binds a symbol; detecting again rebinds it | `query` ∈ the three blocks + the green pad |
| `move_above(ref, …)` | travel over a symbol at a chosen height, optionally leading the descent | `offset_mm` ±60, `height_mm` 40–400, `direction` ∈ top/±x/±y, `standoff_mm` 0–60, `yaw_deg` ±90 |
| `descend_to(ref, …)` | come down over a symbol | `offset_mm` ±60, `height_mm` 12–60 before the grasp, 12–140 after it, `yaw_deg` ±90 |
| `grasp()` / `release()` | close / open the gripper | — |
| `lift(height_mm)` / `retreat(height_mm)` | rise straight up | 40–400 mm |
| `on_failure(policy, max_attempts)` | last op only; what to do if the lift did not take | `abort` or `redetect_regrasp`, at most 2 attempts |
| `abort(reason)` | decline to act; only `detect` may precede it | reason ≤ 200 chars |

So the choices the issue asked for map onto: approach direction (`direction` + `standoff_mm`),
grasp offset (`descend_to.offset_mm`), lift height (`lift.height_mm`), target offset
(`descend_to.offset_mm` on the destination), release height (`descend_to.height_mm` after the
grasp), redetection (a second `detect` mid-program), retry policy (`on_failure`), grasp
orientation (`yaw_deg`), and declining outright (`abort`).

Everything below the waypoints is unchanged and identical for every candidate: damped
least-squares IK on the pinch site, the same controller, the same success test.

**Two of those needed the executor to grow.** `TabletopEnv._ik` took position and a
downward approach axis and left the rotation about that axis in the null space, so a
program could not choose how the jaws are oriented. It now accepts an optional `yaw`; with
`yaw=None` — every caller that predates these programs — the error term is unchanged, and
the existing suites confirm it. `planner.validate` carries an optional `yaw` through the
plan contract the same way. Verified directly: commanding 0°, 45° and 90° puts the pinch
site's heading at 0.0°, 45.0° and 90.0° with the approach axis still straight down, and the
grasp still succeeds.

**Declining is a candidate, not a non-answer.** `abort` grounds to no trace at all. A pool
where every acting candidate fails should be won by the candidate that stops, and a
selector cannot be credited for that unless it was allowed to choose it.

**What a program still cannot see.** `detect` returns a centre, a pixel box, and an
apparent size — no orientation. So an "orientation-aware" grasp can only *pin* a heading,
not *read* one. That is enough today, because blocks spawn axis-aligned, and it is not
enough for the rotated objects [#25](https://github.com/Trolleroof/skill-level-world-model/issues/25)
plans to introduce: those need `bounding_box`/`detect_in_base` to report a yaw first.

## Grounding is where coordinates appear

`ground(program, observation)` resolves each symbol against one `SceneObservation` — the
detections from [`waddle_wm/perception.py`](../waddle_wm/perception.py) plus the task-frame
pad, which is given rather than detected — and emits the same waypoint program the chat
planner emits. It then goes through `planner.validate`, so a candidate is held to exactly
the contract the rest of the repo already enforces: the nine known phases, the workspace
box, and the eight-phase budget. A program with nine motion ops is legal source and fails
to ground, with a message Claude can act on.

Which phase an op becomes depends only on whether the object is held: `move_above` is
`approach` before the grasp and `move` after it, `descend_to` is `descend` before and
`place` after. The canonical ten-op program therefore grounds to exactly the eight-phase
pick-and-place shape the world-model verifier was trained on.

## What counts as the same candidate

Two candidates are the same candidate when they would *behave* the same way: identical
grounded waypoints to a tenth of a millimetre, identical retry policy, and the same answer
to whether a symbol is rebound mid-program. Symbol names, `strategy`, and `note` are not
part of the key — Claude varies the wording freely for plans that mean the same thing.

Duplicates are **kept in the pool**, flagged with `duplicate_of`. A pass@N curve is a
statement about a sampling distribution, and dropping repeats would silently reweight it.
The `dedup_key` is there so #23 can execute each distinct behaviour once and reuse the
outcome for its repeats.

## Pools

One natural pool and one diagnostic pool per scene seed, cached under `data/pools/`.

**Natural.** Up to 64 independent `claude -p` calls, same model, same system prompt, same
prompt template, same observation, one stateless turn each, no tools. A sample that fails
to parse, to validate, to ground, or to reach is recorded in `rejected` with its stage and
reason and is **not** re-asked — every candidate gets the same generation budget. Sampling
continues in sample order until 64 candidates are accepted or the attempt budget
(`--oversample`, 1.5× by default) runs out.

**Diagnostic.** Scripted, deterministic, Claude-free, and labelled on both sides: seven
named *strategies* a scene might call for, and six named *faults* a verifier ought to
catch. Every entry carries `diagnostic` and `diagnostic_kind`, so a failure slice says
which behaviour was missed rather than just how many. Reported separately from the natural
pool, because a planted bug says nothing about what Claude proposes.

A planted fault is a bug in the program, not a guaranteed failure — physics decides. Over
three seeds, red block → green pad:

| kind | diagnostic | succeeded |
| --- | --- | --- |
| strategy | `correct` | 3/3 |
| strategy | `redetect_regrasp` | 3/3 |
| strategy | `orientation_aware_grasp` | 3/3 |
| strategy | `alternate_approach` | 3/3 |
| strategy | `offset_grasp` | 3/3 |
| strategy | `controlled_release` | 3/3 |
| strategy | `abort_on_uncertainty` | 0/3 — it declines, by construction |
| fault | `stale_coordinates` | 0/3 |
| fault | `bad_grasp` | 0/3 |
| fault | `missing_lift` | 0/3 |
| fault | `early_release` | **3/3** |
| fault | `high_release` | **3/3** |
| fault | `wrong_target` | 0/3 |

Two things to read off that table before the benchmark runs.

Releasing the block from transit height onto the pad, or from 140 mm up, works every time:
the pad is flat and 105 mm across and the block simply drops onto it. The same holds when
the destination is another block. A verifier that rejects those two is producing false
rejects, not catching a fault.

And on a plain scene, most of the strategies are indistinguishable from `correct` — an
alternate approach matters when something is in the way, an orientation-aware grasp when
the object is rotated, a controlled release when the destination is small. They are in the
suite so the pool can *express* the strategy, which is the half of the problem this issue
owns; making them matter is the scene suite's half. One of them is already load-bearing:
a 45° grasp of an axis-aligned cube pinches two corners and slips about half the time,
which is why `orientation_aware_grasp` aligns the jaws with the faces rather than guessing
a heading.

## What one real pool looks like

Sixteen independent Haiku 4.5 samples on scene seed 0, then every candidate executed from
the identical restored scene:

```text
accepted 16/16, rejected 0, unique programs 16, $0.77
7/16 candidates succeed; the pool contains a success
of 16 candidates: 10 pin a grasp yaw, 10 redetect mid-program, 2 use an approach standoff
```

Nothing was rejected and nothing was a duplicate — the schema is loose enough that Claude
writes valid programs first time, and tight enough that the programs differ. They vary in
lift height, transit height, release height, grasp height, grasp orientation, approach
standoff, and whether the destination is rebound before the place.

The interesting part is what separates the seven successes from the nine failures: **lift
height, and nothing else.** Every candidate that lifted to 150 mm or more succeeded and
every candidate that lifted to 60–100 mm failed with `missed`. The success test requires
the block to clear 90 mm, and the block hangs about 15 mm below the pinch point, so a lift
to 100 mm leaves it just under the bar.

That is a warning for #18 as much as a result. A natural pool on a plain tabletop scene
can be dominated by one plan-visible number, which any selector that reads the program can
learn — and which says nothing about whether raw visual context helps. Scenes where the
outcome depends on what the scene looks like rather than on what the program says are
exactly what [#25](https://github.com/Trolleroof/skill-level-world-model/issues/25) is for,
and the scene-dependent slice is where the visual verifier has to earn its claim.

## Nested prefixes

Candidates are ranked in generation order and `prefixes` lists the ids for
N = 1, 4, 16, 32, 64. Prefix N is the first N of prefix 2N by construction, and
`check_pool` refuses a pool where that is not true, so every selector in #18 sees
identical prefixes and success@N curves are comparable across selectors.

## What each artifact records

`data/pools/<kind>/<object>_to_<destination>/<split>-seed<NNNN>.json`:

| field | contents |
| --- | --- |
| `pool_id`, `kind`, `split` | identity, natural or diagnostic, which disjoint seed range |
| `protocol` | protocol and schema version, git SHA, generator settings and their hash, generation time |
| `task` | instruction, object, destination |
| `scene` | seed, observation id, the exact observation text Claude saw, detections, pad, and `hidden_truth` / `block_spawn` — the MuJoCo positions, recorded for #23's snapshot and for perception-error reporting, never shown to a selector |
| `candidates[]` | candidate id, rank index, sample index, complete program, grounded trace, dedup key and `duplicate_of`, validation result, retry policy, redetect ops, abort reason, diagnostic name and kind, generation cost/latency, and Claude's raw reply |
| `rejected[]` | every discarded sample with its stage and reason |
| `prefixes` | the nested id lists |
| `pool_has_success` | `null` here, filled in by #23 once every candidate has been executed |
| `summary` | accepted, rejected, attempted, unique programs, duplicate fraction, acting vs declining candidates, distinct strategies, cost, reject reasons by stage |

`pool_has_success` is in the artifact from generation onward and must survive downstream.
Whether a pool contained a winning program at all is a fact about generation: a selector
that picks the best of sixty-four failures has not failed, and one that picks a success
from a pool of sixty-four successes has not succeeded. The same goes for the duplicate
fraction and the reject counts — they are generation outcomes, reported as such, never
attributed to a verifier.

Scene seeds are split disjointly — `train` 0–39, `calibration` 40–59, `test` 100–159 — and
the split is stamped into every pool. Nothing tuned on `test` may have been fitted on the
others.

Program *templates* are deliberately not split. The natural pool has no templates: Claude
writes each candidate. The diagnostic pool is thirteen named behaviours that must appear in
every scene, in every split, or the failure slices stop being comparable — its whole job is
to be the same ruler everywhere. Nothing is fitted on those templates, so there is nothing
to leak; what must not cross splits is a *tuned threshold or checkpoint*, and that is
enforced by the seed split.

## Integrity

`uv run python -m waddle_wm.pools --validate data/pools` fails on duplicate candidate ids,
a ranking with a gap, candidates out of generation order, a short or non-nested prefix, a
missing or empty grounded trace, a declining candidate that somehow carries a trace, the
wrong schema version, a repeated or unlabelled diagnostic, a dropped `pool_has_success`, a
"natural" pool that was not generated by Claude, and any program whose source contains a
ground-truth coordinate. `test_program` carries a negative fixture for each of those, so a
check that stops firing fails the contract test rather than passing silently.

The cache keys on a hash of the generator settings — model, system prompt, prompt template,
schema version, instruction — so editing the prompt invalidates every pool built with the
old one rather than silently mixing generations. Pool size is excluded from that hash: a
larger pool of the same kind extends a smaller one. The generate command prints each pool as
`cached`, `stale`, or `small` before it decides to spend anything.

## Cost

Generation dominates. One sample is one `claude -p` call, measured at **$0.048 on
`claude-haiku-4-5-20251001`** with this system prompt — so a 64-candidate pool is about $3
per scene, and eight scenes about $25. `claude-opus-5` is the default and has not been
measured for this prompt; it is several times more per call, so price a full Opus run
before starting one. Pools are cached and never regenerated: `--regenerate` is the only way
to spend that twice.
