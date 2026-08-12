"""Checks for counterfactual execution and the fairness controls behind the outcome records.

    uv run python -m waddle_wm.test_counterfactual              # offline: shape, view, integrity
    uv run python -m waddle_wm.test_counterfactual --live 1     # + execute a real pool in MuJoCo

The oracle ordering, the artifact schema and the selector tie-break belong to
`benchmark_record` (#24) and are tested by `test_benchmark_record`. What is tested here is
that this module *feeds* them honestly: records in the locked `OUTCOME_FIELDS` shape, one per
candidate per physics seed, all from one restored snapshot, with nothing about the answer key
reaching the selector view.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy

from waddle_wm import benchmark_record as br
from waddle_wm import counterfactual as cf
from waddle_wm import program as prog
from waddle_wm.test_program import SCENE

BUDGET = {"candidate_timeout_s": 60.0, "max_attempts": 2, "perturbation_mm": 3.0, "physics_seeds": 1}


def outcome(index: int, candidate_id: str, **overrides) -> dict:
    """One synthetic outcome record, in the shape #23 is contracted to write."""
    record = {"success": True, "failure_mode": None, "max_lift_mm": 250.0,
              "final_target_error_mm": 20.0, "failed_attempts": 0, "timed_out": False,
              "error": None, "execution_seconds": 1.0, "execution_order": index,
              "declined": False, "attempts": 1, "restore_ok": True, "snapshot_id": "snap",
              "sim_seconds": 4.0, "frames": 40, "candidate_index": index, "diagnostic": None}
    record.update(overrides)
    return record


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
                           "generation": {"model": "scripted", "seconds": 0.0}, "raw": ""})
    ids = [candidate["candidate_id"] for candidate in candidates]
    return {"pool_id": "p0", "kind": "diagnostic", "split": "train",
            "protocol": {"generator_hash": "abc", "generator": {"kind": "diagnostic",
                                                                "model": "scripted"}},
            "task": {"instruction": "put the red block on the green pad", "object": "red block",
                     "destination": "green pad"},
            "scene": {"seed": 0, "observation_id": SCENE.observation_id, "observation": "text",
                      "detections": [], "landing_pad": {"centre": [0.5, 0.3], "radius": 0.105},
                      "hidden_truth": {"red_block": [0.3801, -0.1799, 0.018]},
                      "block_spawn": {"red_block": [0.3801, -0.1799, 0.018]}},
            "candidates": candidates, "rejected": [],
            "prefixes": {str(n): ids[:n] for n in (1, 4, 16) if n <= len(ids)},
            "pool_has_success": None, "summary": {}}


def offline_artifact(pool: dict, successes=(1, 2)) -> tuple[dict, dict]:
    """A complete run over `pool`, assembled through the real code path."""
    outcomes = {}
    for index, candidate in enumerate(pool["candidates"]):
        won = index in successes
        # Executed back to front, so a run that quietly used pool order would be caught.
        outcomes[candidate["candidate_id"]] = outcome(
            len(pool["candidates"]) - 1 - index, candidate["candidate_id"], success=won,
            failure_mode=None if won else "missed",
            final_target_error_mm=10.0 * (index + 1) if won else 500.0,
            candidate_index=index, declined=bool(candidate["aborts"]),
            diagnostic=candidate["diagnostic"])
    scenes = cf.scenes_for(pool, outcomes, 0, BUDGET, None)
    names = sorted({name for scene in scenes for name in scene["selectors"]})
    metadata = br.run_metadata(
        split="train", scene_seeds=[0], physics_seeds=[0],
        pools={pool["pool_id"]: {"kind": pool["kind"], "seed": 0, "candidates": len(outcomes)}},
        generator=pool["protocol"]["generator"],
        perception={"detector_queries": ["red block"]},
        physics={"perturbation_mm": 3.0, "physics_seeds": 1},
        selectors={name: {"kind": "reference_ranking"} for name in names},
        git_sha="abc123", git_dirty=False, created_at="2026-01-01T00:00:00")
    artifact = {"artifact_version": br.ARTIFACT_VERSION, "metadata": metadata, "scenes": scenes,
                "excluded": [], "kind": pool["kind"],
                "preflight": {cf.scenario_id_of(pool): {"ok": True, "order_probes": 3,
                                                         "order_mismatches": []}},
                "execution": {cf.scenario_id_of(pool): [
                    {"snapshot_id": "snap", "physics_seed": 0, "candidates": len(outcomes),
                     "successes": len(successes), "declined": 1, "errors": 0,
                     "outcomes": outcomes}]}}
    return artifact, {pool["pool_id"]: cf.selector_view(pool)}


def check_outcome_shape():
    """The record this module writes is exactly the one #24 contracted for."""
    record = outcome(0, "c00")
    missing = [key for key in br.OUTCOME_FIELDS if key not in record]
    assert not missing, missing
    # Millimetres, because that is what the oracle quantises in. A metres field here would
    # silently become a 1000x error in the answer key rather than a type error.
    assert "final_target_error_mm" in record and "max_lift_mm" in record
    assert not any(key.endswith("_m") for key in record), record

    # There is one definition of "best" in the repo, and it is not in this module.
    source = open(cf.__file__).read()
    for banned in ("ORACLE_ORDERING", "def order_key", "def oracle_of"):
        assert banned not in source, f"{banned} is a second oracle; use benchmark_record"
    print(f"outcome shape passed: all {len(br.OUTCOME_FIELDS)} locked fields, no local ordering")


def check_selector_block():
    """A ranking becomes scores, and the *locked* argmax turns scores into the choice."""
    pool = offline_pool()
    prefix = pool["prefixes"]["4"]
    block = cf.selector_block("first", [c["candidate_id"] for c in pool["candidates"]], prefix)

    scored = [row["candidate_id"] for row in block["scores"]]
    assert sorted(scored) == sorted(prefix), "exactly one score per candidate in the prefix"
    assert block["chosen"] == br.selector_choice(block["scores"], prefix), \
        "the choice must be the locked tie-break's, not a private argmax"
    assert block["chosen"]["candidate_id"] == prefix[0], block["chosen"]
    assert block["information_sources"] and all(
        source in br.INFORMATION_SOURCES for source in block["information_sources"])
    assert not br._leaked_keys(block), br._leaked_keys(block)
    assert not br.check_timing({"scenario_id": "s"}, "first", block), "timing must close its boundary"

    reversed_ranking = list(reversed([c["candidate_id"] for c in pool["candidates"]]))
    other = cf.selector_block("last", reversed_ranking, prefix)
    assert other["chosen"]["candidate_id"] == prefix[-1], other["chosen"]
    print("selector block passed: one score per candidate, locked argmax, closed timing boundary")


def check_selector_view():
    """A selector sees the observation and the programs. It never sees the answer key."""
    pool = offline_pool()
    view = cf.selector_view(pool)
    keys = set(cf._keys_in(view))
    for forbidden in cf.HIDDEN_FIELDS:
        assert forbidden not in keys, f"the selector view exposes {forbidden}"
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
    artifact, views = offline_artifact(pool)
    assert cf.check_execution(artifact, views) == [], cf.check_execution(artifact, views)

    def widest(broken):
        return max(broken["scenes"], key=lambda scene: scene["prefix"])

    def executed(broken):
        return next(iter(broken["execution"].values()))[0]["outcomes"]

    def unexecuted_candidate(broken):
        executed(broken).pop(widest(broken)["pool_prefix"][-1])

    def unscored_candidate(broken):
        scene = widest(broken)
        scene["counterfactual"].pop(scene["pool_prefix"][-1])

    def repeat_execution_order(broken):
        records = list(executed(broken).values())
        records[1]["execution_order"] = records[0]["execution_order"]

    def unequal_pools(broken):
        cells = next(iter(broken["execution"].values()))
        short = deepcopy(cells[0])
        short["physics_seed"] = 1
        short["outcomes"].pop(next(iter(short["outcomes"])))
        cells.append(short)

    def rewritten_scene_outcome(broken):
        scene = widest(broken)
        first = scene["pool_prefix"][0]
        scene["counterfactual"][first] = {**scene["counterfactual"][first], "success": True,
                                          "final_target_error_mm": 0.0}

    negatives = [
        ("a scored candidate that was never executed", unexecuted_candidate, None,
         "never executed"),
        ("an executed candidate with no outcome in the scene", unscored_candidate, None,
         "no counterfactual execution record"),
        ("repeated execution order", repeat_execution_order, None, "not a permutation"),
        ("missing outcome field",
         lambda b: list(executed(b).values())[0].pop("failed_attempts"), None,
         "missing failed_attempts"),
        ("restore mismatch",
         lambda b: list(executed(b).values())[0].update(restore_ok=False), None, "restore mismatch"),
        ("two snapshots",
         lambda b: list(executed(b).values())[2].update(snapshot_id="other"), None,
         "not all started from one snapshot"),
        ("unequal pools across physics seeds", unequal_pools, None, "unequal pools"),
        ("a scene outcome that was edited after execution", rewritten_scene_outcome, None,
         "differ from the recorded executions"),
        ("pool_has_success disagrees",
         lambda b: widest(b).update(pool_has_success=False), None, "disagrees with the executions"),
        ("failed preflight",
         lambda b: b["preflight"][next(iter(b["preflight"]))].update(ok=False), None,
         "preflight did not pass"),
        ("preflight skipped order probes",
         lambda b: b["preflight"][next(iter(b["preflight"]))].update(order_probes=0), None,
         "did not execute an order probe"),
        ("dirty worktree", lambda b: b["metadata"].update(git_dirty=True), None, "dirty worktree"),
        ("an edited oracle definition",
         lambda b: b["metadata"]["oracle_definition"].update(final_tie_break="whatever wins"),
         None, "definitions were changed"),
        ("leaked outcome in the selector view", None,
         lambda b, v: [c.update(success=True) for c in v["p0"]["candidates"]], "exposes success"),
        ("empty selector view", None, lambda b, v: v["p0"].update(candidates=[]), "no candidates"),
    ]
    for name, break_artifact, break_views, fragment in negatives:
        broken, broken_views = deepcopy(artifact), deepcopy(views)
        if break_artifact:
            break_artifact(broken)
        if break_views:
            break_views(broken, broken_views)
        problems = cf.check_execution(broken, broken_views)
        assert any(fragment in problem for problem in problems), (name, problems)

    # A pool where nothing succeeded is a clean run, not an integrity failure: the answer key
    # is allowed to have no success in it, and must then say so.
    hopeless, hopeless_views = offline_artifact(pool, successes=())
    assert cf.check_execution(hopeless, hopeless_views) == [], cf.check_execution(hopeless, hopeless_views)
    assert not max(hopeless["scenes"], key=lambda s: s["prefix"])["pool_has_success"]
    print(f"integrity passed: clean run accepted, {len(negatives)} checks fire")


def check_aggregates():
    """#24 must be able to report from what this module writes, without adaptation."""
    pool = offline_pool()
    artifact, _ = offline_artifact(pool)
    report = br.aggregate(artifact, prefix=4)
    assert report["scenes"] == 1
    row = report["prefixes"]["4"]["selectors"]["first"]
    for metric in br.PRIMARY_OUTCOMES:
        assert metric in row, (metric, sorted(row))
    assert report["prefixes"]["4"]["pool_has_success"] == 1.0
    # `first` picks candidate 0, which is a failure here, so the pool held a success it missed.
    assert row["selected_success"] == 0.0 and row["missed_available_success"] == 1.0
    assert row["oracle_gap"] > 0

    hopeless, _ = offline_artifact(pool, successes=())
    empty = br.aggregate(hopeless, prefix=4)["prefixes"]["4"]["selectors"]["first"]
    # Undefined, not zero: no selector could have succeeded, so none is charged for it.
    assert empty["selection_efficiency"] is None, empty

    covered, _ = offline_artifact(pool, successes=(10,))
    assert any(scene["pool_has_success"] for scene in covered["scenes"])
    assert cf.pool_coverage(covered, [pool]) == {pool["pool_id"]: True}
    assert cf.pool_coverage(hopeless, [pool]) == {pool["pool_id"]: False}
    print("aggregation passed: benchmark_record reports from the artifact unchanged")


def check_live(scenes: int, physics_seeds: int, perturbation_mm: float):
    """Execute a real diagnostic pool in MuJoCo and check the fairness controls hold."""
    from waddle_wm.pools import Scene, build_pool

    for seed in range(scenes):
        scene_obj = Scene(seed)
        pool = build_pool(scene_obj, "diagnostic", "pick up the red block and put it on the green pad",
                          "red block", "green pad", len(prog.diagnostic_programs()),
                          "scripted", "train", 1, 1.0, 60.0)
        scene_obj.close()

        artifact, views = cf.run_pools([pool], "train", "diagnostic", physics_seeds,
                                       perturbation_mm, cf.DEFAULT_TIMEOUT, probes=3)
        artifact["metadata"]["git_dirty"] = False        # a dev worktree is dirty by construction
        problems = cf.check_execution(artifact, views)
        assert problems == [], problems
        assert all(checks["ok"] for checks in artifact["preflight"].values())
        assert not artifact["excluded"], artifact["excluded"]

        # Every candidate has one execution record, including the complete prefix of 16.
        expected = {candidate["candidate_id"] for candidate in pool["candidates"]}
        cells = artifact["execution"][cf.scenario_id_of(pool)]
        assert len(cells) == physics_seeds, cells
        for facts in cells:
            outcomes = facts["outcomes"]
            assert set(outcomes) == expected, "exactly one record per candidate per physics seed"
            assert all(record["restore_ok"] for record in outcomes.values())
            assert sorted(r["execution_order"] for r in outcomes.values()) == list(range(len(expected)))
            widest = max((s for s in artifact["scenes"]
                          if s["physics_seed"] == facts["physics_seed"]),
                         key=lambda s: s["prefix"])
            winner = outcomes[widest["oracle"]["candidate_id"]]
            print(f"  seed {seed} physics {facts['physics_seed']}  "
                  f"{facts['successes']}/{facts['candidates']} candidates succeed  "
                  f"prefix {widest['prefix']} oracle {winner['diagnostic']} "
                  f"(decided by {widest['oracle']['decided_by']})")

        # The one claim the shuffled execution order rests on: the same candidate, executed
        # again from the same snapshot after everything else has run, gives the same answer.
        replay_scene = Scene(seed)
        try:
            candidate = pool["candidates"][0]
            program = prog.validate_program(candidate["program"])
            again = cf.run_candidate(replay_scene, program, replay_scene.snapshot,
                                     replay_scene.observation, cf.DEFAULT_TIMEOUT)
            unperturbed = next(facts for facts in cells if facts["physics_seed"] == 0)
            before = unperturbed["outcomes"][candidate["candidate_id"]]
            assert again["success"] == before["success"], (again, before)
            assert abs(again["final_target_error_mm"] - before["final_target_error_mm"]) \
                <= cf.ORDER_TOLERANCE_MM, (again["final_target_error_mm"],
                                           before["final_target_error_mm"])
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

    check_outcome_shape()
    check_selector_block()
    check_selector_view()
    check_integrity()
    check_aggregates()
    if args.live:
        check_live(args.live, args.physics_seeds, args.perturbation_mm)


if __name__ == "__main__":
    main()
