# Counterfactual execution and the hidden oracle

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
   for each: restore -> execute -> record   success, failure mode, lift, target error,
        |                                    failed attempts, timeout/error, time
        v
   oracle ordering (locked)        the hidden answer key
        |
        v
   pool_has_success / pass@N | oracle success@N | selected success@N | gap
```

One file: [`waddle_wm/counterfactual.py`](../waddle_wm/counterfactual.py), with the snapshot
mechanics in [`waddle_wm/sim/env.py`](../waddle_wm/sim/env.py) and the checks in
[`waddle_wm/test_counterfactual.py`](../waddle_wm/test_counterfactual.py).

```bash
uv run python -m waddle_wm.counterfactual --kind diagnostic          # execute the cached pools
uv run python -m waddle_wm.counterfactual --split test --physics-seeds 3 --perturbation-mm 5
uv run python -m waddle_wm.counterfactual --validate data/counterfactual   # integrity only
uv run python -m waddle_wm.test_counterfactual --live 1                    # contract + MuJoCo
```

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
assumed: every execution record carries `restore_ok`, and `check_run` fails the run if any
of them is false.

## The preflight

No outcome is trusted until three things hold, and the run refuses to be aggregated
otherwise (`preflight.ok`, checked by `check_run`):

| check | what it proves |
| --- | --- |
| `restores_to_same_bytes` | restoring twice gives the same `state_digest` as the snapshot itself |
| `observation_reproduced` | the restored scene renders the observation id the pool was generated against |
| `order_mismatches` | executing the first few candidates in the reverse order gives the same success, failure mode, lift and target error, to 0.1 mm |

The last one is what makes a shuffled execution order safe. Order is randomized (seeded on
the scenario id) precisely so that a systematic order effect would show up as noise across
scenarios rather than as a consistent advantage for whoever ran first — and the preflight is
the direct test that there is no order effect to begin with. Candidate identity travels in
the record (`candidate_id`, `candidate_index`, `execution_order`), so shuffling costs
nothing.

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

`check_run` fails if the pools are unequal across physics seeds: every candidate gets exactly
one execution record per seed, or the comparison is void.

## The oracle ordering, locked before anything runs

Defined in `ORACLE_ORDERING`, in this order, all ascending in badness:

1. **success** — a candidate that did the task outranks every candidate that did not.
2. **failed attempts** — a bounded retry that was needed is worse than one that was not.
3. **final target error**, rounded to 0.1 mm.
4. **execution time**, rounded to 0.1 s.
5. **pool index** — so the ordering is *total* and independent of execution order.

Keys 3 and 4 are rounded on purpose. Two placements 20 µm apart are the same placement, and
wall-clock time is noisy enough that it should decide only coarse differences; below those
granularities the pool index decides, which is reproducible. `tied_with` reports how many
candidates the ordering could not separate before that tie-break, and a selector that picks
one of them is scored as `within_oracle_tie` rather than as wrong.

A run that errored or timed out has no measured placement and sorts behind every measured
one. A candidate that *declined* (`abort`) is a real candidate: it loses to any success and
beats a crash, and its "final target error" is the untouched scene's, because it moved
nothing.

**The oracle is not a verifier.** It is obtained only by executing every candidate from the
restored state, which nothing deployable can do. It is an offline answer key, and the whole
point of the module is that it is *hidden*.

## What a selector is allowed to see

`selector_view(pool)` is the only input #18 gets: the observation text, the detections, the
landing pad, the nested prefixes, and every candidate's program, grounded trace, dedup key,
retry policy, redetect ops and abort reason. It is reconstructed field by field rather than
copied-and-deleted, so a new key in the pool artifact has to be added to the view *on
purpose* before any verifier can read it.

`check_run` then walks the view's keys structurally and fails if any of `HIDDEN_FIELDS`
appears — `hidden_truth`, `block_spawn`, `snapshot`, `success`, `failure_mode`,
`target_error_m`, `max_lift_m`, `failed_attempts`, `oracle`, `pool_has_success`, … The check
is on keys, not on text: Claude's `note` is free prose and may well contain the word
"success", which is not a leak. A *field* named after an outcome is.

Nothing in this module calls Claude, the estimated-state heuristic, or the visual verifier.
Selectors arrive from the outside, as rankings.

## Selectors and the gap

A selector is a ranking of candidate ids. Two reference rankings are built in so the report
is never vacuous — `first` (the earliest sample, what an agent that asks Claude once already
does) and `random` (the coin flip any ranking has to beat). Real verifiers arrive through
`--selections`:

```json
{"<pool_id>": {"<selector name>": ["<candidate_id>", "<candidate_id>", ...]}}
```

Selected-at-N is the top-ranked candidate that is inside prefix N, so one ranking scores at
every prefix and a selector can never reach past the prefix it was given (`check_run`
enforces it). Per pick:

| field | meaning |
| --- | --- |
| `success` | did the chosen program actually work |
| `oracle_rank` | where the pick sits in the oracle ordering, 0 being best |
| `target_error_gap_m` | chosen final error minus the oracle's |
| `agrees_with_oracle` / `within_oracle_tie` | exact id match, and the weaker claim that the ordering could not separate them |
| `missed_available_success` | the pool held a success and the selector chose a failure |

## Attribution: whose fault is a failure

The report keeps generation and selection apart, because they fail for different reasons.

- **`pool_has_success` / pass@N** is a fact about what Claude proposed. If it is false, *no*
  selector could have succeeded; that is candidate-generation coverage, not verifier failure.
- **oracle success@N** is the ceiling. By construction it equals pass@N — the oracle takes a
  success whenever one exists — and `check_run` fails the run if they ever disagree, which is
  a direct test that the ordering is still an answer key.
- **selected success@N** is what a selector actually reached.
- **selection efficiency** is selected over oracle, computed *only* over scenarios where the
  oracle found a success. A pool with nothing in it cannot make a selector look bad, and a
  pool of nothing but successes cannot make one look good.
- **missed available successes** is the count that matters: `pool_has_success` and the
  selector chose a failure anyway.

Exact candidate-id agreement is reported but never the score. Several candidates can be
genuinely good, and a selector that picks a different real success has not chosen worse.

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
| fault | `stale_coordinates` | 0/2 | 0/4 | 456 mm |
| fault | `bad_grasp` | 0/2 | 0/4 | 460 mm |
| fault | `missing_lift` | 0/2 | 0/4 | 50.6 mm |
| fault | `early_release` | 2/2 | 4/4 | **5.8 mm** |
| fault | `high_release` | 2/2 | 4/4 | 18.7 mm |
| fault | `wrong_target` | 0/2 | 0/4 | 546 mm |

Two things to read off it.

**The oracle picked a planted fault in five of six scenarios.** `early_release` opens the
gripper at transit height, and on a flat pad 105 mm across the block drops nearly dead
centre — 5.8 mm, better than any of the strategies. The answer key is what physics did, not
what the label says. A verifier that "correctly" rejects `early_release` on this scene suite
is producing a false reject, and the counterfactual is what makes that visible instead of
letting it hide inside an aggregate accuracy number. Making the release height matter is the
scene suite's job ([#25](https://github.com/Trolleroof/skill-level-world-model/issues/25)),
not the oracle's.

**5 mm of block jitter changed no outcome.** The gripper's lateral tolerance swallows it, so
these runs are a check that the pairing machinery works, not yet a robustness result. A
perturbation that separates candidates has to be larger than the grasp tolerance or has to
move something the program bound early — which is the same scene-suite problem.

The selection numbers on these pools are deliberately not quoted. The diagnostic pool is
ordered strategies-first, faults-last, so its first four candidates are all successes and
every selector scores 1.0 at N=4. That ordering is a fixed ruler, not a sampling
distribution; pass@N and selection efficiency only mean something on the natural pools.

## What each artifact records

`data/counterfactual/<kind>/<object>_to_<destination>/<split>-seed<NNNN>.json`, mirroring the
pool tree:

| field | contents |
| --- | --- |
| `protocol` | protocol and schema version, git SHA + dirty flag, the pool's generator hash, candidate timeout, physics seeds, perturbation, wall clock |
| `preflight` | snapshot id, both restore checks, the order probes and any mismatch |
| `selector_view` | the complete, outcome-free input #18 consumes |
| `oracle_ordering` | the locked keys, copied into the artifact so a later edit is detectable |
| `scenarios[]` | one per physics seed: scenario id, snapshot id, perturbation, `pool_has_success`, successes, declines, errors, the oracle with its full ranking and `tied_with`, the selector rankings, the per-prefix breakdown, and every execution |
| `scenarios[].executions[]` | scenario/snapshot/pool/candidate ids, candidate index, execution order, `restore_ok`, declined, success, failure mode, attempts and failed attempts, max lift, final target error, wall clock, simulated seconds, frames, timeout flag, error |

`summary.json` at the root holds the aggregate: generation coverage, the oracle ceiling, and
every selector's success@N, selection efficiency, missed available successes, mean target-error
gap and mean oracle rank — plus `integrity`, which is empty on a usable run.

`pool_has_success` for physics seed 0 is written back into the pool artifact, where
[#17](https://github.com/Trolleroof/skill-level-world-model/issues/17) reserved room for it.
The perturbed seeds are a robustness slice and do not restate what Claude proposed.

## Integrity

`uv run python -m waddle_wm.counterfactual --validate data/counterfactual` fails on a failed
preflight, a missing or duplicated candidate, an execution order that is not a permutation of
the pool, a restore mismatch, candidates started from more than one snapshot, unequal pools
across physics seeds, `pool_has_success` disagreeing with the executions, an oracle that
missed an available success, an edited oracle ordering, a selector that reached outside its
prefix, and any hidden field appearing in the selector view.

`test_counterfactual` carries a negative fixture for each of those plus the ordering
semantics, so a check that stops firing fails the contract test rather than passing silently.
`--live` runs the real thing in MuJoCo and asserts the fairness controls hold end to end.
