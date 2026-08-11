"""Checks for counterfactual execution, the hidden oracle, and the fairness controls.

    uv run python -m waddle_wm.test_counterfactual              # offline: ordering, gaps, integrity
    uv run python -m waddle_wm.test_counterfactual --live 1     # + execute a real pool in MuJoCo

The offline part needs no simulator: it is the contract #18 leans on — the oracle ordering
is fixed and total, a selector is scored only against what was available to it, the selector
view carries no outcome, and every integrity check actually fires.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy

from waddle_wm import counterfactual as cf
from waddle_wm import program as prog
from waddle_wm.test_program import SCENE


def execution(index: int, **overrides) -> dict:
    """One synthetic execution record, defaulting to a clean success."""
    record = {"scenario_id": "s0", "pool_id": "p0", "snapshot_id": "snap", "physics_seed": 0,
              "candidate_id": f"c{index:02d}", "candidate_index": index, "execution_order": index,
              "restore_ok": True, "declined": False, "success": True, "failure_mode": None,
              "attempts": 1, "failed_attempts": 0, "max_lift_m": 0.25, "target_error_m": 0.02,
              "execution_seconds": 1.0, "sim_seconds": 4.0, "frames": 40, "timed_out": False,
              "error": None, "diagnostic": None, "strategy": ""}
    record.update(overrides)
    return record


def check_ordering():
    """The oracle key is the one the issue locked, and it is total."""
    # Success outranks failure even when the failure got closer to the target by accident.
    lucky_failure = execution(0, success=False, failure_mode="missed", target_error_m=0.001)
    plain_success = execution(1, target_error_m=0.09)
    assert cf.oracle_of([lucky_failure, plain_success])["candidate_id"] == "c01"

    # Then fewer failed attempts, then lower target error, then shorter execution.
    retried = execution(0, failed_attempts=1, target_error_m=0.01)
    clean = execution(1, failed_attempts=0, target_error_m=0.05)
    assert cf.oracle_of([retried, clean])["candidate_id"] == "c01"
    near, far = execution(0, target_error_m=0.05), execution(1, target_error_m=0.01)
    assert cf.oracle_of([near, far])["candidate_id"] == "c01"
    slow = execution(0, target_error_m=0.02, execution_seconds=9.0)
    quick = execution(1, target_error_m=0.02, execution_seconds=1.0)
    assert cf.oracle_of([slow, quick])["candidate_id"] == "c01"

    # Differences below the documented granularity do not decide anything; pool order does,
    # so the ordering is total and does not depend on which candidate ran first.
    a = execution(0, target_error_m=0.02, execution_seconds=1.00)
    b = execution(1, target_error_m=0.02 + cf.ERROR_GRANULARITY_M / 4, execution_seconds=1.02)
    assert cf.oracle_of([a, b])["candidate_id"] == "c00"
    assert cf.oracle_of([b, a])["candidate_id"] == "c00", "the oracle must not depend on record order"
    assert cf.oracle_of([a, b])["tied_with"] == 1, "an unseparated candidate must be reported as tied"

    # An execution that errored has no measured placement and sorts behind every measured one.
    crashed = execution(0, success=False, failure_mode="error", target_error_m=None, error="IK failed")
    missed = execution(1, success=False, failure_mode="missed", target_error_m=0.5)
    assert cf.oracle_of([crashed, missed])["candidate_id"] == "c01"

    # A declining candidate is a candidate: it loses to a success and beats a crash.
    declined = execution(0, success=False, failure_mode="declined", declined=True, target_error_m=0.5,
                         execution_seconds=0.0)
    assert cf.oracle_of([declined, plain_success])["candidate_id"] == "c01"
    assert cf.oracle_of([declined, crashed])["candidate_id"] == "c00"
    assert cf.oracle_of([])["candidate_id"] is None
    print(f"oracle ordering passed: {len(cf.ORACLE_ORDERING)} locked keys, total and record-order free")


def check_selector_scoring():
    """A selector is credited for choosing a real success and charged for missing one."""
    records = [execution(0, success=False, failure_mode="missed", target_error_m=0.5),
               execution(1, target_error_m=0.03),
               execution(2, target_error_m=0.01)]
    oracle = cf.oracle_of(records)
    assert oracle["candidate_id"] == "c02"

    picked_best = cf.score_selector(["c02", "c01"], records, oracle)
    assert picked_best["agrees_with_oracle"] and picked_best["oracle_rank"] == 0
    assert picked_best["target_error_gap_m"] == 0.0
    assert not picked_best["missed_available_success"]

    # A different success is not a miss — it is a gap. Exact candidate-id agreement is
    # reported and never scored, because several candidates may be genuinely good.
    second_best = cf.score_selector(["c01"], records, oracle)
    assert second_best["success"] and not second_best["agrees_with_oracle"]
    assert abs(second_best["target_error_gap_m"] - 0.02) < 1e-9, second_best
    assert not second_best["missed_available_success"]

    missed = cf.score_selector(["c00"], records, oracle)
    assert missed["missed_available_success"] and missed["oracle_rank"] == 2

    # When nothing in the pool worked, no selector could have succeeded and none is blamed.
    hopeless = [execution(i, success=False, failure_mode="missed", target_error_m=0.5) for i in range(3)]
    assert not cf.oracle_of(hopeless)["success"]
    assert not cf.score_selector(["c00"], hopeless, cf.oracle_of(hopeless))["missed_available_success"]

    # A ranking that only names candidates outside the prefix selects nothing rather than
    # silently reaching past the prefix it was given.
    outside = cf.score_selector(["c09"], records, oracle)
    assert outside["ranked_nothing_available"] and outside["candidate_id"] is None
    print("selector scoring passed: gap, agreement, ties, and missed available successes")


def offline_pool() -> dict:
    """A pool artifact in the shape #17 writes, built without MuJoCo."""
    candidates = []
    for index, (name, kind, program) in enumerate(prog.diagnostic_programs()):
        grounded = prog.ground(program, SCENE)
        candidates.append({"candidate_id": f"c{index:02d}", "index": index, "sample_index": index,
                           "program": program.as_json(),
                           "grounded_trace": grounded.step.summary()["trace"] if grounded.step else [],
                           "dedup_key": grounded.dedup_key(), "duplicate_of": None,
                           "validation": {"ok": True, "stage": "accepted", "error": None},
                           "retry": program.retry, "redetect_ops": program.redetects,
                           "aborts": program.aborts, "diagnostic": name, "diagnostic_kind": kind,
                           "strategy": program.strategy, "note": program.note,
                           "generation": {"model": "scripted"}, "raw": ""})
    ids = [candidate["candidate_id"] for candidate in candidates]
    return {"pool_id": "p0", "kind": "diagnostic", "split": "train",
            "protocol": {"generator_hash": "abc"},
            "task": {"instruction": "put the red block on the green pad", "object": "red block",
                     "destination": "green pad"},
            "scene": {"seed": 0, "observation_id": SCENE.observation_id, "observation": "text",
                      "detections": [], "landing_pad": {"centre": [0.5, 0.3], "radius": 0.105},
                      "hidden_truth": {"red_block": [0.3801, -0.1799, 0.018]},
                      "block_spawn": {"red_block": [0.3801, -0.1799, 0.018]}},
            "candidates": candidates, "rejected": [],
            "prefixes": {str(n): ids[:n] for n in (1, 4) if n <= len(ids)},
            "pool_has_success": None, "summary": {}}


def offline_run(pool: dict, successes=(1, 2)) -> dict:
    """A complete counterfactual run over `pool`, with `successes` the winning indices."""
    records = [execution(index, candidate_id=candidate["candidate_id"],
                         success=index in successes,
                         failure_mode=None if index in successes else "missed",
                         target_error_m=0.01 * (index + 1) if index in successes else 0.5,
                         declined=bool(candidate["aborts"]))
               for index, candidate in enumerate(pool["candidates"])]
    order = list(range(len(records)))
    for record, position in zip(records, reversed(order)):
        record["execution_order"] = position

    rankings = cf.reference_rankings(pool, "s0")
    prefixes = {}
    for key in pool["prefixes"]:
        inside = [record for record in records if record["candidate_index"] < int(key)]
        oracle = cf.oracle_of(inside)
        prefixes[key] = {"candidates": len(inside),
                         "pool_has_success": any(r["success"] for r in inside),
                         "successes": sum(r["success"] for r in inside), "oracle": oracle,
                         "selectors": {name: cf.score_selector(ranking, inside, oracle)
                                       for name, ranking in rankings.items()}}
    scenario = {"scenario_id": "s0", "pool_id": pool["pool_id"], "kind": pool["kind"],
                "split": pool["split"], "scene_seed": 0,
                "observation_id": pool["scene"]["observation_id"], "snapshot_id": "snap",
                "physics_seed": 0, "perturbation_mm": 0.0, "candidates": len(records),
                "pool_has_success": any(r["success"] for r in records),
                "successes": sum(r["success"] for r in records),
                "declined": sum(r["declined"] for r in records), "errors": 0,
                "oracle": cf.oracle_of(records), "oracle_ordering": list(cf.ORACLE_ORDERING),
                "selector_rankings": rankings, "prefixes": prefixes, "executions": records}
    return {"pool_id": pool["pool_id"], "kind": pool["kind"], "split": pool["split"],
            "scene_seed": 0, "protocol": {"pool_generator_hash": "abc"},
            "oracle_ordering": list(cf.ORACLE_ORDERING),
            "preflight": {"ok": True, "restores_to_same_bytes": True, "observation_reproduced": True,
                          "order_mismatches": []},
            "selector_view": cf.selector_view(pool), "scenarios": [scenario]}


def check_selector_view():
    """A selector sees the observation and the programs. It never sees the answer key."""
    pool = offline_pool()
    view = cf.selector_view(pool)
    keys = set(cf._keys_in(view))
    for forbidden in cf.HIDDEN_FIELDS:
        assert forbidden not in keys, f"the selector view exposes {forbidden}"
    assert "hidden_truth" not in json.dumps(view)
    for name, point in pool["scene"]["hidden_truth"].items():
        for value in point:
            assert f"{value:.4f}" not in json.dumps(view), f"{name} ground truth reached the selector"
    assert [c["candidate_id"] for c in view["candidates"]] == \
        [c["candidate_id"] for c in pool["candidates"]], "every candidate must be rankable"
    assert view["scene"]["observation"] == "text" and view["prefixes"] == pool["prefixes"]
    print(f"selector view passed: {len(view['candidates'])} candidates, "
          f"{len(cf.HIDDEN_FIELDS)} hidden fields withheld")


def check_integrity():
    pool = offline_pool()
    run = offline_run(pool)
    assert cf.check_run(run) == [], cf.check_run(run)

    def drop_candidate(broken):
        broken["scenarios"][0]["executions"].pop()
        broken["scenarios"][0]["candidates"] -= 1

    def duplicate_candidate(broken):
        records = broken["scenarios"][0]["executions"]
        records[1] = {**records[1], "candidate_id": records[0]["candidate_id"]}

    def second_snapshot(broken):
        broken["scenarios"][0]["executions"][2]["snapshot_id"] = "other"

    def unequal_pools(broken):
        extra = deepcopy(broken["scenarios"][0])
        extra.update(scenario_id="s1", physics_seed=1)
        extra["executions"] = extra["executions"][:-2]
        broken["scenarios"].append(extra)

    def edited_ordering(broken):
        broken["scenarios"][0]["oracle_ordering"] = [{"key": "target_error_m"}]

    def oracle_misses_success(broken):
        broken["scenarios"][0]["oracle"]["success"] = False

    def selector_reaches_past_prefix(broken):
        broken["scenarios"][0]["prefixes"]["1"]["selectors"]["first"]["candidate_index"] = 7

    def leaked_outcome(broken):
        for candidate in broken["selector_view"]["candidates"]:
            candidate["success"] = True

    negatives = [
        ("missing candidate", drop_candidate, "candidates missing"),
        ("duplicate candidate", duplicate_candidate, "executed more than once"),
        ("execution order is not a permutation",
         lambda b: b["scenarios"][0]["executions"][0].update(execution_order=99), "not a permutation"),
        ("restore mismatch",
         lambda b: b["scenarios"][0]["executions"][0].update(restore_ok=False), "restore mismatch"),
        ("two snapshots", second_snapshot, "not all started from one snapshot"),
        ("unequal pools across physics seeds", unequal_pools, "unequal pools"),
        ("pool_has_success disagrees",
         lambda b: b["scenarios"][0].update(pool_has_success=False), "disagrees with the executions"),
        ("the oracle missed a success", oracle_misses_success, "missed an available success"),
        ("edited oracle ordering", edited_ordering, "not the locked one"),
        ("selector chose outside its prefix", selector_reaches_past_prefix, "outside prefix"),
        ("failed preflight", lambda b: b["preflight"].update(ok=False), "preflight did not pass"),
        ("leaked outcome in the selector view", leaked_outcome, "exposes success"),
        ("empty selector view", lambda b: b["selector_view"].update(candidates=[]), "no candidates"),
    ]
    for name, break_it, fragment in negatives:
        broken = deepcopy(run)
        break_it(broken)
        problems = cf.check_run(broken)
        assert any(fragment in problem for problem in problems), (name, problems)

    # A pool where nothing succeeded is a clean run, not an integrity failure: the oracle is
    # allowed to have no success in it, and must then say so.
    hopeless = offline_run(pool, successes=())
    assert cf.check_run(hopeless) == [], cf.check_run(hopeless)
    assert not hopeless["scenarios"][0]["pool_has_success"]
    print(f"integrity passed: clean run accepted, {len(negatives)} checks fire")


def check_aggregate():
    pool = offline_pool()
    winnable, hopeless = offline_run(pool), offline_run(pool, successes=())
    hopeless["scenarios"][0]["scenario_id"] = "s1"
    report = cf.aggregate([winnable, hopeless])

    assert report["scenarios"] == 2 and report["executions"] == 2 * len(pool["candidates"])
    assert report["generation"]["4"]["pass_at_n"] == 0.5, report["generation"]
    # The oracle's ceiling is exactly generation coverage: it takes a success whenever the
    # pool holds one, so a gap between these two would mean the ordering is not an answer key.
    for key in report["generation"]:
        assert report["oracle"][key]["success_at_n"] == report["generation"][key]["pass_at_n"], key

    first = report["selectors"]["first"]["4"]
    assert first["scenarios"] == 2 and first["winnable_scenarios"] == 1
    # Half of all scenarios, but the unwinnable one cannot count against the selector: the
    # efficiency is conditioned on the pool having held a success at all.
    assert first["success_at_n"] == 0.0 and first["selection_efficiency"] == 0.0
    assert first["missed_available_successes"] == 1

    lucky = deepcopy(winnable)
    lucky["scenarios"][0]["prefixes"]["4"]["selectors"]["first"] = cf.score_selector(
        ["c01"], [r for r in lucky["scenarios"][0]["executions"] if r["candidate_index"] < 4],
        lucky["scenarios"][0]["prefixes"]["4"]["oracle"])
    improved = cf.aggregate([lucky, hopeless])["selectors"]["first"]["4"]
    assert improved["selection_efficiency"] == 1.0 and improved["missed_available_successes"] == 0
    print("aggregate passed: pass@N, oracle ceiling, selection efficiency conditioned on winnable pools")


def check_live(scenes: int, physics_seeds: int, perturbation_mm: float):
    """Execute a real diagnostic pool in MuJoCo and check the fairness controls hold."""
    from waddle_wm.pools import Scene, build_pool

    for seed in range(scenes):
        scene_obj = Scene(seed)
        pool = build_pool(scene_obj, "diagnostic", "pick up the red block and put it on the green pad",
                          "red block", "green pad", 13, "scripted", "train", 1, 1.0, 60.0)
        scene_obj.close()

        run = cf.run_pool(pool, physics_seeds, perturbation_mm, cf.DEFAULT_TIMEOUT, probes=3)
        problems = cf.check_run(run)
        assert problems == [], problems
        assert run["preflight"]["ok"], run["preflight"]

        expected = {candidate["candidate_id"] for candidate in pool["candidates"]}
        for scenario in run["scenarios"]:
            records = scenario["executions"]
            assert {record["candidate_id"] for record in records} == expected
            assert len(records) == len(expected), "exactly one record per candidate per physics seed"
            assert all(record["restore_ok"] for record in records)
            print(f"  seed {seed} physics {scenario['physics_seed']}  "
                  f"{scenario['successes']}/{scenario['candidates']} succeed  "
                  f"oracle {scenario['oracle']['candidate_id']} "
                  f"({dict((r['candidate_id'], r['diagnostic']) for r in records)[scenario['oracle']['candidate_id']]})")

        # The one claim the shuffled execution order rests on: the same candidate, executed
        # again from the same snapshot after everything else has run, gives the same answer.
        replay_scene = Scene(seed)
        try:
            candidate = pool["candidates"][0]
            program = prog.validate_program(candidate["program"])
            again = cf.run_candidate(replay_scene, program, replay_scene.snapshot,
                                     replay_scene.observation, cf.DEFAULT_TIMEOUT)
            first = next(record for record in run["scenarios"][0]["executions"]
                         if record["candidate_id"] == candidate["candidate_id"])
            assert again["success"] == first["success"], (again, first)
            assert abs(again["target_error_m"] - first["target_error_m"]) <= cf.ERROR_GRANULARITY_M, \
                (again["target_error_m"], first["target_error_m"])
        finally:
            replay_scene.close()
        print(f"  seed {seed} replay of {pool['candidates'][0]['diagnostic']} reproduced its outcome")

    print(f"live counterfactual passed over {scenes} scene(s), {physics_seeds} physics seed(s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", type=int, default=0, help="execute a real diagnostic pool on N scenes")
    ap.add_argument("--physics-seeds", type=int, default=2)
    ap.add_argument("--perturbation-mm", type=float, default=cf.DEFAULT_PERTURBATION_MM)
    args = ap.parse_args()

    check_ordering()
    check_selector_scoring()
    check_selector_view()
    check_integrity()
    check_aggregate()
    if args.live:
        check_live(args.live, args.physics_seeds, args.perturbation_mm)


if __name__ == "__main__":
    main()
