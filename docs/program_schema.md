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
and the diagnostic faults; [`waddle_wm/pools.py`](../waddle_wm/pools.py) generates, caches,
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
| `move_above(ref, …)` | travel over a symbol at a chosen height, optionally leading the descent | `offset_mm` ±60, `height_mm` 40–400, `direction` ∈ top/±x/±y, `standoff_mm` 0–60 |
| `descend_to(ref, …)` | come down over a symbol | `offset_mm` ±60, `height_mm` 12–60 before the grasp, 12–140 after it |
| `grasp()` / `release()` | close / open the gripper | — |
| `lift(height_mm)` / `retreat(height_mm)` | rise straight up | 40–400 mm |
| `on_failure(policy, max_attempts)` | last op only; what to do if the lift did not take | `abort` or `redetect_regrasp`, at most 2 attempts |

So the choices the issue asked for map onto: approach direction (`direction` + `standoff_mm`),
grasp offset (`descend_to.offset_mm`), lift height (`lift.height_mm`), target offset
(`descend_to.offset_mm` on the destination), release height (`descend_to.height_mm` after the
grasp), redetection (a second `detect` mid-program), and retry policy (`on_failure`).

Everything below the waypoints is unchanged and identical for every candidate: damped
least-squares IK on the pinch site, the same controller, the same success test.

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

**Diagnostic.** Scripted, deterministic, Claude-free: the canonical correct program, a
correct program that redetects and regrasps, and six planted faults — `stale_coordinates`,
`bad_grasp`, `missing_lift`, `early_release`, `high_release`, `wrong_target`. Reported
separately from the natural pool, because a planted bug says nothing about what Claude
proposes.

A planted fault is a bug in the program, not a guaranteed failure — physics decides. Over
three seeds, red block → green pad:

| diagnostic | succeeded |
| --- | --- |
| `correct` | 3/3 |
| `correct_redetect_regrasp` | 3/3 |
| `stale_coordinates` | 0/3 |
| `bad_grasp` | 0/3 |
| `missing_lift` | 0/3 |
| `early_release` | **3/3** |
| `high_release` | **3/3** |
| `wrong_target` | 0/3 |

Releasing the block from transit height onto the pad, or from 140 mm up, works every time:
the pad is flat and 105 mm across and the block simply drops onto it. The same holds when
the destination is another block. That is worth stating before the benchmark runs — a
verifier that rejects those two is producing false rejects, not catching a fault.

## What one real pool looks like

Sixteen independent Haiku 4.5 samples on scene seed 0, then every candidate executed from
the identical restored scene:

```text
accepted 16/16, rejected 0, unique programs 16, $0.62
6/16 candidates succeed; the pool contains a success
```

Nothing was rejected and nothing was a duplicate — the schema is loose enough that Claude
writes valid programs first time, and tight enough that the programs differ. The programs
vary in lift height, transit height, release height, grasp height, whether the destination
is redetected mid-program, and whether there is a retry.

The interesting part is what separates the six successes from the ten failures: **lift
height, and nothing else.** Every candidate that lifted to 150 mm succeeded and every
candidate that lifted to 80–100 mm failed with `missed`. The success test requires the
block to clear 90 mm, and the block hangs about 15 mm below the pinch point, so a lift to
100 mm leaves it just under the bar.

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
| `candidates[]` | candidate id, rank index, sample index, complete program, grounded trace, dedup key and `duplicate_of`, validation result, retry policy, redetect ops, generation cost/latency, and Claude's raw reply |
| `rejected[]` | every discarded sample with its stage and reason |
| `prefixes` | the nested id lists |
| `summary` | accepted, rejected, unique programs, duplicate fraction, cost, reject reasons by stage |

Scene seeds are split disjointly — `train` 0–39, `calibration` 40–59, `test` 100–159 — and
the split is stamped into every pool. Nothing tuned on `test` may have been fitted on the
others.

## Integrity

`uv run python -m waddle_wm.pools --validate data/pools` fails on duplicate candidate ids,
a ranking with a gap, candidates out of generation order, a short or non-nested prefix, a
missing or empty grounded trace, the wrong schema version, a repeated diagnostic fault, a
"natural" pool that was not generated by Claude, and any program whose source contains a
ground-truth coordinate. `test_program` carries a negative fixture for each of those, so a
check that stops firing fails the contract test rather than passing silently.

## Cost

Generation dominates. One sample is one `claude -p` call, measured at **$0.04–0.06 on
`claude-haiku-4-5-20251001`** with this system prompt — so a 64-candidate pool is $2.50–$4
per scene, and eight scenes is $20–$30. `claude-opus-5` is the default and has not been
measured for this prompt; it is several times more per call, so price a full Opus run
before starting one. Pools are cached and never regenerated: `--regenerate` is the only way
to spend that twice.
