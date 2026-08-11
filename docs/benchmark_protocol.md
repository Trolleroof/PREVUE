# The benchmark artifact, the oracle, and the integrity checks

This is the contract [issue #24](https://github.com/Trolleroof/skill-level-world-model/issues/24)
asked for: what a locked benchmark run has to record, how the oracle orders candidates, where
the selector's clock starts and stops, and what a validator refuses. It sits between
[`program_schema.md`](program_schema.md) (what a candidate is, #17) and the counterfactual
execution in [#23](https://github.com/Trolleroof/skill-level-world-model/issues/23), and it is
what [#18](https://github.com/Trolleroof/skill-level-world-model/issues/18) and
[#26](https://github.com/Trolleroof/skill-level-world-model/issues/26) report from.

One file: [`waddle_wm/benchmark_record.py`](../waddle_wm/benchmark_record.py) holds the
schema, the two orderings, the validator, and the paired aggregation.
[`waddle_wm/test_benchmark_record.py`](../waddle_wm/test_benchmark_record.py) is the negative
fixture suite — one broken run per check.

```bash
uv run python -m waddle_wm.benchmark_record --contract              # the locked definitions
uv run python -m waddle_wm.benchmark_record --validate results/programs/*.json
uv run python -m waddle_wm.test_benchmark_record                    # 31 negative fixtures
```

## Why the artifact is this heavy

The number a selector scores moves when the pool changes, when the observation changes, when
the success threshold moves, when the oracle's tie-break flips, and when Claude's generation
time or MuJoCo's execution time is quietly folded into "selector latency". None of those are
selector improvements. So the run records all of them, and the validator refuses a run whose
records cannot rule them out. A saved artifact specifies the evaluation completely: no shell
history, no "we used the usual checkpoint".

## Shape

```text
artifact
├── metadata          protocol + program-schema versions, git SHA and dirtiness, split and
│                     seeds, pool ids and order, generator settings, perception and camera
│                     settings, physics and execution budget, the success and oracle
│                     definitions plus their hash, one config block per selector, and the
│                     predeclared primary outcomes
├── scenes[]          one per (scenario, physics seed, pool prefix)
│   ├── pool_prefix           the candidate ids, in pool order
│   ├── counterfactual        candidate id -> the #23 execution record
│   ├── pool_has_success      candidate-generation coverage, never selector quality
│   ├── oracle                the locked answer key: winner, full ranking, what decided it
│   ├── claude_generation_seconds, mujoco_execution_seconds
│   └── selectors{}           per selector: information boundary, per-candidate score /
│                             probability / uncertainty, the choice, its score margin, its
│                             tie-break, its latency and component breakdown, its cost
└── excluded[]        every attempted scene left out, with the reason
```

`scenes` is keyed by `(scenario_id, physics_seed, prefix)` — that triple is the cell every
paired metric is paired on, and two records for one cell is an error, not an average.

### The counterfactual record (#23 writes it, this module reads it)

`OUTCOME_FIELDS` is the whole contract, and every field is required — a partially recorded
execution is a hole in the answer key:

```text
success  failure_mode  max_lift_mm  final_target_error_mm  failed_attempts
timed_out  error  execution_seconds  execution_order
```

## The two orderings, locked before anyone looks at results

**The oracle** (`oracle_best`) is the hidden offline answer key: success first, then fewer
failed attempts, then lower target error, then shorter execution time, then pool index.
It is not a deployable verifier — it exists only after every candidate has been executed
from the restored snapshot.

Target error and execution time are compared in buckets (0.5 mm, 50 ms) so a re-run whose
physics differ in the last digit cannot promote a candidate. Bucketing, not a pairwise
tolerance: "within 0.5 mm of each other" is not transitive, and an intransitive comparison
makes the answer key depend on which pair was compared first. Candidates inside one bucket
tie and fall to the next key; the last key is pool index, so the ordering is always total.
`decided_by` records which key separated the winner from the runner-up, so a report can tell
a decisive oracle from one that came down to 50 ms.

**The selector's choice** (`selector_choice`) is the argmax of its own scores, ties broken by
pool index — never by outcome. The recorded `score_margin` is the distance to the runner-up.
The validator recomputes both from the recorded scores and rejects a run whose `chosen` block
disagrees, so an artifact cannot claim a choice the scores did not make.

## The timing boundary

Selector latency runs from `observation_ready_at` — the observation the selector consumes
exists — to `chosen_at` — the final candidate is picked. Component timings are retained and
must be non-negative and sum inside that window.

Candidate generation and MuJoCo execution are recorded per scene, in their own fields, and
`check_timing` fails if either name appears inside a selector block. Claude's sampling cost
belongs to the pool, which was frozen long before; the simulator's cost belongs to #23's
counterfactual sweep, which no deployed selector ever pays.

## What the validator refuses

`check_run` returns the list of problems; empty means usable. It fails on:

- a candidate with no execution record, an execution outside the prefix, a repeated candidate,
  two candidates claiming one execution order, or a partial outcome record;
- a prefix that is not the smaller prefix extended (nesting is what makes pool-size a variable
  rather than a confound);
- unequal execution budgets across the run, or one scenario scored against two pool or
  observation identities;
- seeds outside the declared split, or a test run reusing train or calibration seeds;
- a protocol, program-schema, or definition hash that is not this code's, and definitions
  edited after the run was recorded;
- a selector block containing simulator state or the answer key (`LEAKED_FIELDS`), or one
  declaring a forbidden information source (`hidden_oracle`, `mujoco_state`,
  `counterfactual_outcome`) or no boundary at all;
- missing, unordered, or incomplete timing, and generation or execution time inside the
  selector window;
- a candidate scored zero or twice, a choice that is not the locked argmax, a recorded oracle
  that is not the locked ordering's, coverage that disagrees with the executions;
- a missing arm on any scene, a duplicated cell, a dirty worktree, missing metadata, a swapped
  primary outcome, or a missing exclusion list.

Every one of those has a negative fixture in `test_benchmark_record.py` that breaks a clean
run exactly that way and requires the matching complaint. A validator nobody has watched
reject anything is a validator that passes everything.

## Reporting

`aggregate` raises `NotComparable` rather than averaging across cells that do not describe the
same evaluation: a missing arm, a duplicated cell, or a differing protocol or definition hash.
Records are only combined when they share scenario, observation, pool prefix, execution budget,
success and oracle definitions, and protocol version.

Predeclared primary outcomes, fixed before the locked run:

| outcome | meaning |
| --- | --- |
| `selected_success` | did the chosen candidate actually succeed |
| `selection_efficiency` | selected success among the pools that held a success — **undefined**, not zero, when the pool held none |
| `oracle_gap` | the chosen candidate's position in the oracle ranking; 0 is the answer key |

Reported alongside, per pool prefix: `pool_has_success`, `missed_available_success`,
`target_error_gap_mm`, false accepts and rejects, Brier score, mean uncertainty, latency, and
cost. Comparisons are per-scene paired differences with bootstrap confidence intervals over
scenes, not two separate averages. `false_accepts_by_prefix` answers the predeclared question
of whether a selector accepts more failures as the pool grows — reported whichever way it
comes out.

The attribution boundary holds throughout. `pool_has_success` is candidate-generation
coverage: if it is false, no selector could have succeeded and none is charged for it. If it
is true and a selector picks a failure, that is a missed available success. Natural and
diagnostic pools are reported separately — a planted bug says nothing about what Claude
proposes. And nothing here ranks a selector by predicted performance: only measured paired
results from the locked run.

## Still to wire

Everything above is the schema, the orderings, and the checks. Populating a real artifact
needs #23's counterfactual sweep to exist; the writer that fills `counterfactual` from it, and
the report that renders `aggregate` output, land once #23 merges.
