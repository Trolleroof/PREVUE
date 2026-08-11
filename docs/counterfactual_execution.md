# Counterfactual execution: every candidate, from one identical state

This is the contract [issue #23](https://github.com/Trolleroof/skill-level-world-model/issues/23)
asked for: every candidate in a pool executed from *one identical simulator state*, so that
the question "was there something better in there?" has an answer.

Executing only the program a selector picked cannot answer it. A success tells you that
program worked, not that it was the best available; a failure tells you nothing at all about
whether the selector had a winner to find. Both of those are needed before
[#18](https://github.com/Trolleroof/skill-level-world-model/issues/18) can say a verifier
ranks well.

```text
cached pool (#17)                 snapshot: the complete MuJoCo integration state
        |                                   taken once, before anything runs
        v
   preflight  ---- restore twice, same bytes? same observation?
        |           same outcomes with the order reversed?
        v
   shuffle the candidates          identity travels with the candidate, not the position
        |
        v
   for each: restore -> execute -> record   the nine locked OUTCOME_FIELDS
        |
        v
   benchmark_record (#24)          the oracle ordering, the artifact schema, the validator
```

**This module owns execution and nothing else.** The oracle ordering, the artifact schema,
the selector tie-break and the validator all live in
[`benchmark_protocol.md`](benchmark_protocol.md) / `waddle_wm/benchmark_record.py`. The
records written here are exactly its `OUTCOME_FIELDS`, the artifact is what its `check_run`
validates, and its `aggregate` reports from that artifact unchanged. There is one definition
of "best" in this repo and it is not here — `test_counterfactual` greps this module's source
to keep it that way.

Files: [`waddle_wm/counterfactual.py`](../waddle_wm/counterfactual.py), the snapshot mechanics
in [`waddle_wm/sim/env.py`](../waddle_wm/sim/env.py), the checks in
[`waddle_wm/test_counterfactual.py`](../waddle_wm/test_counterfactual.py).

```bash
uv run python -m waddle_wm.counterfactual --split train --kind diagnostic
uv run python -m waddle_wm.counterfactual --split test --kind natural --physics-seeds 3 --perturbation-mm 5
uv run python -m waddle_wm.counterfactual --validate data/counterfactual/test-natural.json
uv run python -m waddle_wm.test_counterfactual --live 1
```

One artifact per `(split, kind)`: the two kinds have different generators, and a run's
metadata records one.

## The snapshot is the whole state, not the block positions

`TabletopEnv.reset(blocks=...)` re-runs the keyframe and re-places the blocks. That is close
to the original scene, not identical to it, and "close" is exactly what a counterfactual
cannot use: the residual differences between candidate *k* and candidate *k+1* would be
mixed into whatever the candidates themselves did.

`TabletopEnv.snapshot()` records `mjSTATE_INTEGRATION` — time, `qpos`, `qvel`, `act`, the
warmstart accelerations, plugin state, `ctrl`, applied forces, equality flags, mocap and
userdata — plus `model.site_pos`, which is where the landing pad lives and which `reset`
can move. `restore()` writes it back and clears the recording. `state_digest()` fingerprints
the result, so a restore that did not land where it claimed is *recorded* rather than
assumed: every execution record carries `restore_ok`, and `check_execution` fails the run if
any of them is false.

## The preflight

No outcome is trusted until three things hold. A scenario whose preflight fails is dropped
into `excluded` rather than quietly averaged in:

| check | what it proves |
| --- | --- |
| `restores_to_same_bytes` | restoring twice gives the same `state_digest` as the snapshot itself |
| `observation_reproduced` | the restored scene renders the observation id the pool was generated against |
| `order_mismatches` | executing the first few candidates in the reverse order gives the same success, failure mode, lift and target error, to 0.1 mm |

The last one is what makes a shuffled execution order safe. Order is randomized (seeded on
the scenario id and physics seed) precisely so that a systematic order effect would show up
as noise across scenarios rather than as a consistent advantage for whoever ran first — and
the preflight is the direct test that there is no order effect to begin with. Candidate
identity travels in the record (`candidate_id`, `candidate_index`, `execution_order`), so
shuffling costs nothing.

The 0.1 mm probe tolerance is deliberately tighter than the oracle's 0.5 mm bucket: the claim
being tested is that restoration is *exact*, not that it is close enough to sort by.

Measured on the diagnostic pools: zero order mismatches, and a candidate re-executed after
all twelve others have run reproduces its target error to under 0.1 mm.

## Paired physics seeds

`--physics-seeds N` runs the whole pool `N` times. Seed 0 is the snapshot itself. Every seed
above it jitters all three blocks by a normal draw of `--perturbation-mm`, seeded on
`(scene seed, physics seed)` — so the *same* perturbed scene is presented to every candidate
in the pool. That is the pairing: a difference between two candidates under seed 2 is a
difference between the candidates, never a difference between two rolls of the dice.

The candidates are not told about the perturbation. They were written against the
unperturbed observation and their symbols stay bound to it, which is the point — a program
that redetects mid-flight can recover from a scene that moved and one that binds once cannot.
`Scene.execute` therefore takes the pool's observation explicitly rather than looking again,
so a perturbed replay cannot silently re-ground onto the truth the candidate never saw.

`check_execution` fails if the pools are unequal across physics seeds: every candidate gets
exactly one execution record per seed, or the comparison is void.

## Where the executions live, and why not only in the scenes

#24's `scenes` are keyed by pool *prefix* — 1, 4, 16, 32, 64 — because that is the cell its
paired metrics are paired on. A pool whose size is not a prefix boundary therefore has
executions that no scene covers: the 13-candidate diagnostic pool has prefixes 1 and 4, so
nine of its thirteen executions appear in no scene at all.

"Every candidate has exactly one execution record" is a claim about all thirteen. So the
complete set is kept whole under `execution[scenario_id][]`, and the scenes are the
#24-facing slices of it. `check_execution` runs the fairness checks over the complete set —
one snapshot, one record each, an execution order that is a permutation of the pool — and
separately checks that every scene is a faithful slice: no scored candidate that was never
executed, and no outcome that was edited after execution.

## What a selector is allowed to see

`selector_view(pool)` is the only input #18 gets: the observation text, the detections, the
landing pad, the nested prefixes, and every candidate's program, grounded trace, dedup key,
retry policy, redetect ops and abort reason. It is written beside the artifact as
`<split>-<kind>-views.json`, and it is reconstructed field by field rather than
copied-and-deleted, so a new key in the pool artifact has to be added to the view *on
purpose* before any verifier can read it.

`check_execution` then walks the view's keys structurally and fails if any of
`HIDDEN_FIELDS` appears — #24's `LEAKED_FIELDS` plus the nine `OUTCOME_FIELDS` plus
`block_spawn`, `snapshot`, `restore_ok` and friends. The check is on keys, not on text:
Claude's `note` is free prose and may well contain the word "success", which is not a leak.
A *field* named after an outcome is.

Nothing in this module calls Claude, the estimated-state heuristic, or the visual verifier.
Selectors arrive from the outside, as rankings.

## Selectors

A selector is a ranking of candidate ids. Two reference rankings are built in so the artifact
is never vacuous — `first` (the earliest sample, what an agent that asks Claude once already
does) and `random` (the coin flip any ranking has to beat). Real verifiers arrive through
`--selections`:

```json
{"<pool_id>": {"<selector name>": ["<candidate_id>", "<candidate_id>", ...]}}
```

A ranking is turned into *scores*, one per candidate in the prefix, and the choice is then
made by #24's `selector_choice` — the locked argmax with a pool-index tie-break — rather than
by a second, private notion of "the selector's pick". Latency is measured across #24's
`observation_ready_at` → `chosen_at` boundary like any other selector's; for these two it is
the cost of an index lookup, which is the honest number. Everything downstream —
`selected_success`, `selection_efficiency`, `oracle_gap`, the paired differences and their
bootstrap CIs — comes from `benchmark_record.aggregate`.

## Attribution: whose fault is a failure

Kept apart in the artifact, because they fail for different reasons.

- **`pool_has_success`** is a fact about what Claude proposed. If it is false, *no* selector
  could have succeeded; that is candidate-generation coverage, not verifier failure. It is
  written back into the pool artifact, where #17 reserved room for it, from physics seed 0.
- **The oracle** is the ceiling: it takes a success whenever the prefix holds one.
  `check_execution` fails the run if the two ever disagree, which is a direct test that the
  ordering is still an answer key.
- **`selection_efficiency`** is `None`, not 0, on a scenario where the pool held nothing. A
  pool with no success in it cannot make a selector look bad.

## What the diagnostic pools say

Two scenes, thirteen scripted candidates each, three physics seeds (unperturbed plus two
5 mm paired perturbations), 78 executions:

| kind | diagnostic | unperturbed | perturbed | mean final error |
| --- | --- | --- | --- | --- |
| strategy | `correct` | 2/2 | 4/4 | 21.3 mm |
| strategy | `redetect_regrasp` | 2/2 | 4/4 | 21.3 mm |
| strategy | `orientation_aware_grasp` | 2/2 | 4/4 | **9.8 mm** |
| strategy | `alternate_approach` | 2/2 | 4/4 | 20.8 mm |
| strategy | `offset_grasp` | 2/2 | 4/4 | 24.3 mm |
| strategy | `controlled_release` | 2/2 | 4/4 | 24.7 mm |
| strategy | `abort_on_uncertainty` | 0/2 | 0/4 | — declines |
| fault | `stale_coordinates` | 0/2 | 0/4 | 456.5 mm |
| fault | `bad_grasp` | 0/2 | 0/4 | 460.4 mm |
| fault | `missing_lift` | 0/2 | 0/4 | 50.6 mm |
| fault | `early_release` | 2/2 | 4/4 | **5.8 mm** |
| fault | `high_release` | 2/2 | 4/4 | 18.7 mm |
| fault | `wrong_target` | 0/2 | 0/4 | 546.2 mm |

Three things to read off it.

**Over the whole pool the oracle picked a planted fault in five of six scenarios.**
`early_release` opens the gripper at transit height, and on a flat pad 105 mm across the block
drops nearly dead centre — 5.8 mm, better than any of the strategies. The answer key is what
physics did, not what the label says. A verifier that "correctly" rejects `early_release` on
this scene suite is producing a false reject, and the counterfactual is what makes that
visible instead of letting it hide inside an aggregate accuracy number. Making the release
height matter is the scene suite's job
([#25](https://github.com/Trolleroof/skill-level-world-model/issues/25)), not the oracle's.

**At prefix 4 the answer is different**, and legitimately so: `early_release` is candidate 10,
outside the prefix, and the reported oracle is `orientation_aware_grasp` in five of six
scenarios, decided on target error. This is the reason the oracle is computed per prefix
rather than once per pool — "the best available" is a statement about what was available.

**5 mm of block jitter changed no outcome.** The gripper's lateral tolerance swallows it, so
these runs are a check that the pairing machinery works, not yet a robustness result. A
perturbation that separates candidates has to be larger than the grasp tolerance or has to
move something the program bound early — which is the same scene-suite problem.

The selection numbers on these pools are deliberately not quoted. The diagnostic pool is
ordered strategies-first, faults-last, so its first four candidates are all successes and
every selector scores 1.0 at N=4. That ordering is a fixed ruler, not a sampling
distribution; `selected_success` and `selection_efficiency` only mean something on the
natural pools.

## What the artifact records

`data/counterfactual/<split>-<kind>.json` is a #24 benchmark artifact — `metadata`, `scenes`,
`excluded` — plus two keys this module adds:

| field | contents |
| --- | --- |
| `preflight` | per scenario: snapshot id, both restore checks, the order probes and any mismatch |
| `execution` | per scenario, per physics seed: snapshot id, perturbation, candidate/success/decline/error counts, and `outcomes` — the complete execution set, keyed by candidate id |

Each outcome record carries the nine locked `OUTCOME_FIELDS` — `success`, `failure_mode`,
`max_lift_mm`, `final_target_error_mm`, `failed_attempts`, `timed_out`, `error`,
`execution_seconds`, `execution_order` — plus this module's own evidence that the execution
was fair: `restore_ok`, `snapshot_id`, `declined`, `attempts`, `sim_seconds`, `frames`,
`candidate_index`, `diagnostic`.

Millimetres, not metres, because that is what the oracle quantises in; a metres field here
would become a silent 1000× error in the answer key rather than a type error.
`data/counterfactual/<split>-<kind>-views.json` holds the selector views.

## Integrity

`uv run python -m waddle_wm.counterfactual --validate <artifact>` runs everything
`benchmark_record.check_run` checks — schema, metadata, the definition hash, split hygiene,
prefix nesting, the recomputed oracle and selector choice, the timing boundary, the leak scan
inside selector blocks — and then the execution-side claims on top: a failed preflight, a
missing or incomplete outcome record, an execution order that is not a permutation of the
pool, a restore mismatch, candidates started from more than one snapshot, unequal pools
across physics seeds, a scene that scores a candidate never executed or whose outcomes were
edited after the fact, an oracle that missed an available success, and any hidden field in a
selector view.

A run from a dirty worktree fails validation by design — its git SHA describes nothing — so
`--allow-dirty` is for exploratory runs only, never the locked one.

`test_counterfactual` carries a negative fixture for each of those, so a check that stops
firing fails the contract test rather than passing silently. `--live` runs the real thing in
MuJoCo and asserts the fairness controls hold end to end.
