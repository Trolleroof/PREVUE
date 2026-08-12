"""Checks for the code-as-policy program schema and the candidate pools built from it.

    uv run python -m waddle_wm.test_program                # offline: schema, grounding, pool integrity
    uv run python -m waddle_wm.test_program --live 3       # + run the diagnostic suite in MuJoCo
    uv run python -m waddle_wm.test_program --claude       # + one real Claude program sample

The offline part needs no simulator and no Claude: it is the contract every downstream
benchmark leans on — a program is symbolic, grounding is deterministic, and a pool's
prefixes are nested.
"""
from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from waddle_wm import plan_encoding, program as prog
from waddle_wm.program import ProgramError, SceneObservation

SCENE = SceneObservation(
    {"red block": [0.3812, -0.1804, 0.0180], "blue block": [0.5031, -0.1612, 0.0180],
     "yellow block": [0.4507, 0.0503, 0.0180], "green pad": [0.5000, 0.3000, 0.0],
     "gripper": [0.4, 0.0, 0.3]}, pad_radius=0.105, seed=0)


def check_plan_encoding():
    trace = [{"phase": "approach", "yaw": 0.1},
             {"phase": "descend", "yaw": 0.2},
             {"phase": "descend", "yaw": 0.7},
             {"phase": "close"},
             {"phase": "descend", "yaw": 1.0}]
    assert plan_encoding.trace_yaws(trace) == (0.7, 0.1)

    plan = np.zeros((4, len(plan_encoding.fields(2))))
    plan[:, 7:9] = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    grasp_only = plan_encoding.yaw_informative(plan, np.ones(4, dtype=bool), 2)
    assert plan_encoding.orientation_blind(grasp_only), grasp_only
    plan[:, 10:12] = plan[:, 7:9]
    both = plan_encoding.yaw_informative(plan, np.ones(4, dtype=bool), 2)
    assert not plan_encoding.orientation_blind(both), both


def check_schema():
    program = prog.canonical_program()
    assert program.retry == {"policy": "abort", "max_attempts": 0}, program.retry
    assert prog.canonical_program(max_attempts=2).retry["max_attempts"] == 2
    assert prog.canonical_program(redetect=True).redetects == [6], prog.canonical_program(redetect=True).redetects

    round_tripped = prog.parse("```json\n" + json.dumps(program.as_json()) + "\n```")
    assert round_tripped.as_json() == program.as_json()

    base = program.as_json()

    def broken(**edit):
        payload = deepcopy(base)
        payload.update(edit)
        return payload

    def with_ops(ops):
        return broken(ops=ops)

    ops = base["ops"]
    cases = [
        ("prose instead of JSON", "Sure, I would grasp the block first.", "not JSON"),
        ("unknown operation", with_ops([{"op": "wiggle"}]), "unknown operation"),
        ("wrong schema version", broken(schema_version=99), "schema_version must be"),
        ("unknown object", broken(task={"object": "green block", "destination": "green pad"}), "task.object"),
        ("destination is the object", broken(task={"object": "red block", "destination": "red block"}),
         "task.destination"),
        ("unbound ref", with_ops([{"op": "move_above", "ref": "src", "height_mm": 240}]), "never bound"),
        ("offset out of range",
         with_ops([{**op, "offset_mm": [500, 0]} if op["op"] == "descend_to" else op for op in ops]),
         "outside the allowed range"),
        ("grasp height out of range",
         with_ops([{**op, "height_mm": 400} if op["op"] == "descend_to" else op for op in ops]),
         "outside the allowed range"),
        ("unbounded retries",
         with_ops(ops + [{"op": "on_failure", "policy": "redetect_regrasp", "max_attempts": 99}]),
         "outside the allowed range"),
        ("non-integer retries",
         with_ops(ops + [{"op": "on_failure", "policy": "redetect_regrasp", "max_attempts": 1.5}]),
         "whole number"),
        ("unknown retry policy",
         with_ops(ops + [{"op": "on_failure", "policy": "keep_trying", "max_attempts": 1}]), "policy must be"),
        ("on_failure is not last",
         with_ops([ops[0], {"op": "on_failure", "policy": "abort", "max_attempts": 0}, *ops[1:]]),
         "must be the last operation"),
        ("release before grasp", with_ops([ops[0], {"op": "release"}]), "release() before grasp()"),
        ("never grasps", with_ops(ops[:4]), "must grasp()"),
        ("ends holding the object", with_ops([op for op in ops if op["op"] != "release"]), "still holding"),
        ("too many ops", with_ops(ops * 2), f"at most {prog.MAX_OPS}"),
        ("empty program", with_ops([]), "non-empty list"),
        ("unknown detect query",
         with_ops([{"op": "detect", "query": "the mug", "as": "src"}, *ops[1:]]), "detect() accepts"),
        ("yaw out of range",
         with_ops([{**op, "yaw_deg": 180} if op["op"] == "descend_to" else op for op in ops]),
         "outside the allowed range"),
        ("abort without a reason", with_ops([ops[0], {"op": "abort"}]), "needs"),
        ("abort after acting", with_ops([*ops, {"op": "abort", "reason": "changed my mind"}]),
         "cannot follow motion"),
        ("work after an abort",
         with_ops([ops[0], {"op": "abort", "reason": "nothing here"}, *ops[1:]]),
         "must be the last operation"),
    ]
    for name, payload, fragment in cases:
        try:
            prog.parse(payload if isinstance(payload, str) else json.dumps(payload))
        except ProgramError as error:
            assert fragment in str(error), (name, str(error))
        else:
            raise AssertionError(f"{name} should have been rejected")

    # Over-long programs are legal as source and rejected at grounding, where the arm's
    # eight-phase budget lives.
    long_ops = [*ops[:8], {"op": "move_above", "ref": "dst", "height_mm": 240,
                           "offset_mm": [0, 0], "direction": "top", "standoff_mm": 0}, *ops[8:]]
    try:
        prog.ground(prog.validate_program(with_ops(long_ops)), SCENE)
    except ProgramError as error:
        assert "at most 8" in str(error), error
    else:
        raise AssertionError("a nine-phase program should not ground")

    print(f"schema contract passed: canonical program accepted, {len(cases) + 1} rejections")


def check_grounding():
    grounded = prog.ground(prog.canonical_program(), SCENE)
    phases = [entry["phase"] for entry in grounded.trace]
    assert phases == ["approach", "descend", "close", "lift", "move", "place", "open", "retreat"], phases
    assert grounded.step.pick_place_shaped

    descend = next(e for e in grounded.trace if e["phase"] == "descend")
    assert descend["target"][:2] == SCENE.points["red block"][:2], descend
    assert abs(descend["target"][2] - 0.015) < 1e-9, descend
    place = next(e for e in grounded.trace if e["phase"] == "place")
    assert place["target"][:2] == SCENE.points["green pad"][:2], place

    offset = prog.ground(prog.canonical_program(grasp_offset_mm=(30.0, -20.0)), SCENE)
    moved = next(e for e in offset.trace if e["phase"] == "descend")["target"]
    assert abs(moved[0] - (SCENE.points["red block"][0] + 0.030)) < 1e-9, moved
    assert abs(moved[1] - (SCENE.points["red block"][1] - 0.020)) < 1e-9, moved

    standoff = prog.canonical_program().as_json()
    standoff["ops"][2].update(direction="-x", standoff_mm=50.0)
    approach = next(e for e in prog.ground(prog.validate_program(standoff), SCENE).trace
                    if e["phase"] == "approach")["target"]
    assert abs(approach[0] - (SCENE.points["red block"][0] - 0.050)) < 1e-9, approach

    # Programs are symbolic: no coordinate from the scene may appear in the source.
    source = json.dumps(prog.canonical_program().as_json())
    for name, point in SCENE.points.items():
        for value in point:
            assert f"{value:.4f}" not in source, f"{name} coordinate leaked into the program"

    # The same behaviour under a different spelling is the same candidate; a different
    # number, a retry, or a redetect is not.
    renamed = prog.canonical_program().as_json()
    for op in renamed["ops"]:
        for key in ("as", "ref"):
            if key in op:
                op[key] = {"src": "block", "dst": "goal"}[op[key]]
    renamed["strategy"], renamed["note"] = "different words", "same waypoints"
    assert prog.ground(prog.validate_program(renamed), SCENE).dedup_key() == \
        prog.ground(prog.canonical_program(), SCENE).dedup_key()
    for variant in (prog.canonical_program(lift_mm=200.0), prog.canonical_program(max_attempts=1),
                    prog.canonical_program(redetect=True), prog.canonical_program(grasp_offset_mm=(1.0, 0.0))):
        assert prog.ground(variant, SCENE).dedup_key() != prog.ground(prog.canonical_program(), SCENE).dedup_key()

    # Grounding is deterministic, and a missing object is a grounding error, not a crash.
    assert prog.ground(prog.canonical_program(), SCENE).as_json() == grounded.as_json()
    empty = SceneObservation({"green pad": [0.5, 0.3, 0.0]}, seed=0)
    try:
        prog.ground(prog.canonical_program(), empty)
    except ProgramError as error:
        assert "found nothing" in str(error), error
    else:
        raise AssertionError("grounding against a scene without the object should fail")

    # A commanded wrist heading reaches the trace in radians and persists to the descent.
    by_name = {name: program for name, _, program in prog.diagnostic_programs()}
    oriented = by_name["orientation_aware_grasp"]
    grounded_yaw = prog.ground(oriented, SCENE)
    descend = next(e for e in grounded_yaw.trace if e["phase"] == "descend")
    assert abs(descend["yaw"] - math.radians(0.0)) < 1e-9, descend
    assert next(e for e in grounded_yaw.trace if e["phase"] == "approach")["yaw"] == descend["yaw"]
    tilted = prog.canonical_program(grasp_yaw_deg=60.0)
    assert abs(next(e for e in prog.ground(tilted, SCENE).trace
                    if e["phase"] == "lift")["yaw"] - math.radians(60.0)) < 1e-9, "yaw must persist"
    assert "yaw" not in prog.ground(prog.canonical_program(), SCENE).trace[0], "yaw must default to free"

    # A declining candidate grounds to no trace at all, and is not the same candidate as one
    # that does the work.
    declining = by_name["abort_on_uncertainty"]
    aborted = prog.ground(declining, SCENE)
    assert aborted.step is None and aborted.trace == [], aborted
    assert aborted.aborts and "stopping" in aborted.aborts, aborted.aborts
    assert aborted.dedup_key() != prog.ground(prog.canonical_program(), SCENE).dedup_key()
    # Declining because the object is not there is the point of declining, not a bug.
    assert prog.ground(declining, SceneObservation({"green pad": [0.5, 0.3, 0.0]}, seed=0)).step is None

    diagnostics = prog.diagnostic_programs()
    names = {name: kind for name, kind, _ in diagnostics}
    assert set(names) == {*prog.STRATEGIES, *prog.FAULTS}, sorted(names)
    assert all(names[name] == "strategy" for name in prog.STRATEGIES), names
    assert all(names[name] == "fault" for name in prog.FAULTS), names
    keys = {name: prog.ground(program, SCENE).dedup_key() for name, _, program in diagnostics}
    assert len(set(keys.values())) == len(keys), keys
    print(f"grounding passed: canonical shape, offsets, yaw, abort, symbolic source, dedup, "
          f"{len(diagnostics)} diagnostics")


def check_runtime_policy():
    """Redetection and retry must change execution, not only artifact metadata."""
    updated = deepcopy(SCENE)
    updated.points["green pad"] = [0.55, 0.25, 0.0]
    recovering = prog.canonical_program(redetect=True, max_attempts=1)

    segments = list(prog.trace_segments(recovering, SCENE, lambda: updated))
    assert [[entry["phase"] for entry in segment] for segment in segments] == [
        ["approach", "descend", "close", "lift"],
        ["move", "place", "open", "retreat"],
    ], segments
    move = next(entry for entry in segments[1] if entry["phase"] == "move")
    assert move["target"][:2] == updated.points["green pad"][:2], move

    occluded = recovering.as_json()
    occluded["ops"] = [op for index, op in enumerate(occluded["ops"]) if index != 1]
    initial = deepcopy(SCENE)
    initial.points.pop("green pad")
    revealed = list(prog.trace_segments(prog.validate_program(occluded), initial, lambda: updated))
    assert next(entry for entry in revealed[1] if entry["phase"] == "move")["target"][:2] == \
        updated.points["green pad"][:2]

    outcomes = iter((SimpleNamespace(success=False, failure_mode="missed"),
                     SimpleNamespace(success=True, failure_mode=None)))
    observations, attempts = [], []

    def observe():
        observations.append(True)
        return updated

    def run_attempt(trace_segments):
        attempts.append(list(trace_segments))
        return next(outcomes)

    results = prog.execute(recovering, SCENE, observe, run_attempt)
    assert [result.success for result in results] == [False, True], results
    assert len(attempts) == 2 and observations, (attempts, observations)
    target_miss = lambda segments: SimpleNamespace(success=False, failure_mode="target_miss")
    assert len(prog.execute(recovering, SCENE, observe, target_miss)) == 1
    aborting = next(program for name, _, program in prog.diagnostic_programs()
                    if name == "abort_on_uncertainty")
    assert prog.execute(aborting, SCENE, observe, run_attempt) == []
    print("runtime policy passed: redetection rebinds live geometry, retry is bounded, abort does not run")


def check_pool_contract():
    from waddle_wm import pools
    from waddle_wm.pools import PREFIXES, check_pool

    candidates = []
    for index, (name, kind, program) in enumerate(prog.diagnostic_programs()):
        grounded = prog.ground(program, SCENE)
        candidates.append({"candidate_id": f"c{index:02d}", "index": index, "sample_index": index,
                           "program": program.as_json(),
                           "grounded_trace": grounded.step.summary()["trace"] if grounded.step else [],
                           "dedup_key": grounded.dedup_key(), "duplicate_of": None,
                           "validation": {"ok": True, "stage": "accepted", "error": None},
                           "retry": program.retry, "redetect_ops": program.redetects,
                           "aborts": program.aborts, "diagnostic": name, "diagnostic_kind": kind})
    ids = [candidate["candidate_id"] for candidate in candidates]
    generator = pools.generator_settings("diagnostic", "scripted", len(ids),
                                         "put the red block on the green pad", 1, 1.0, 1.0)
    pool = {"pool_id": "diagnostic-test", "kind": "diagnostic", "split": "train",
            "protocol": {"protocol_version": 1, "program_schema_version": prog.SCHEMA_VERSION,
                         "git_sha": "test", "git_dirty": False, "generator": generator,
                         "generator_hash": pools.generator_hash(generator)},
            "task": {"instruction": "put the red block on the green pad", "object": "red block",
                     "destination": "green pad"},
            "scene": {"seed": 0, "observation_id": SCENE.observation_id, "observation": "",
                      "detections": [], "hidden_truth": {"red_block": [0.3801, -0.1799, 0.018]}},
            "candidates": candidates,
            "prefixes": {str(n): ids[:n] for n in PREFIXES if n <= len(ids)},
            "pool_has_success": None, "summary": {}}
    assert check_pool(pool) == [], check_pool(pool)

    negatives = [
        ("duplicate candidate id", lambda p: p["candidates"][1].update(candidate_id=p["candidates"][0]["candidate_id"]),
         "duplicate candidate_id"),
        ("duplicate program", lambda p: p["candidates"][1].update(
            dedup_key=p["candidates"][0]["dedup_key"]), "duplicate program"),
        ("non-nested prefix", lambda p: p["prefixes"].__setitem__("4", list(reversed(p["prefixes"]["4"]))),
         "not nested"),
        ("short prefix", lambda p: p["prefixes"].__setitem__("4", p["prefixes"]["4"][:2]), "holds 2 candidates"),
        ("missing expected prefix", lambda p: p["prefixes"].pop("4"), "expected prefixes"),
        ("gap in the ranking", lambda p: p["candidates"][2].update(index=9),
         "not a contiguous ranking"),
        ("missing grounded trace", lambda p: p["candidates"][0].update(grounded_trace=[]), "empty grounded trace"),
        ("a declining candidate that also acts",
         lambda p: p["candidates"][0].update(aborts="changed my mind"), "must not carry a trace"),
        ("dropped pool_has_success", lambda p: p.pop("pool_has_success"), "generation coverage"),
        ("unlabelled diagnostic", lambda p: p["candidates"][0].update(diagnostic_kind=None),
         "labelled a strategy or a fault"),
        ("wrong schema version", lambda p: p["candidates"][0]["program"].update(schema_version=99),
         "wrong program schema version"),
        ("repeated diagnostic",
         lambda p: p["candidates"][1].update(diagnostic=p["candidates"][0]["diagnostic"]),
         "repeats a named strategy or fault"),
        ("scripted natural pool", lambda p: p.update(kind="natural"), "must be generated by Claude"),
        ("leaked ground truth",
         lambda p: p["candidates"][0]["program"]["ops"].append({"op": "note", "x": 0.3801}),
         "ground truth"),
        ("changed generator settings", lambda p: p["protocol"].update(
            generator={"model": "scripted", "pool_size": len(ids)}, generator_hash="wrong"),
         "generator hash"),
    ]
    for name, break_it, fragment in negatives:
        broken = deepcopy(pool)
        break_it(broken)
        problems = check_pool(broken)
        assert any(fragment in problem for problem in problems), (name, problems)

    class OfflineScene:
        seed = 0
        observation = SCENE

        @staticmethod
        def reachable(trace):
            return None

    stress, rejected = pools.stress_pool(OfflineScene(), "red block", "green pad", 64)
    assert not rejected and len(stress) == len({candidate.dedup_key for candidate in stress}) == 64

    calls = []
    original = pools.one_sample

    def sample(index, instruction, observation, model, timeout):
        calls.append(index)
        program = prog.canonical_program(target_offset_mm=(index * 2.0, 0.0))
        return {"index": index, "error": None, "raw": json.dumps(program.as_json()),
                "generation": {"model": model, "cost_usd": 0.01}}

    pools.one_sample = sample
    try:
        accepted, rejected = pools.natural_pool(
            OfflineScene(), "task", "red block", "green pad", 3, "test", 8, 1.5, 1.0)
        assert len(accepted) == 3 and not rejected and calls == [0, 1, 2], (len(accepted), rejected, calls)
        extended, rejected = pools.natural_pool(
            OfflineScene(), "task", "red block", "green pad", 5, "test", 8, 1.5, 1.0,
            accepted=accepted, rejected=rejected)
        assert [candidate.sample_index for candidate in extended] == list(range(5)), extended
        assert calls == list(range(5)), calls
    finally:
        pools.one_sample = original

    settings = pools.generator_settings("natural", "test", 64, "task", 8, 1.5, 180.0)
    for key in ("parameter_ranges", "generator_code_sha1", "attempt_budget", "workers", "timeout"):
        assert key in settings, (key, settings)

    seen_yaw = []

    class FakeEnv:
        data = SimpleNamespace(joint=lambda name: SimpleNamespace(qpos=[0.0]))

        @staticmethod
        def _ik(target, q, yaw=None):
            seen_yaw.append(yaw)
            return q

    reachable_scene = pools.Scene.__new__(pools.Scene)
    reachable_scene.env = FakeEnv()
    assert reachable_scene.reachable([{"phase": "descend", "target": [0.4, 0.0, 0.02], "yaw": 0.7}]) is None
    assert seen_yaw == [0.7], seen_yaw

    from waddle_wm.verifier import require_action_compatibility
    require_action_compatibility(prog.ground(prog.canonical_program(), SCENE).trace)
    try:
        require_action_compatibility(prog.ground(prog.canonical_program(grasp_yaw_deg=45), SCENE).trace)
    except ValueError as error:
        assert "yaw-aware" in str(error), error
    else:
        raise AssertionError("a legacy checkpoint must not silently collapse distinct grasp yaws")
    print(f"pool contract passed: clean pool accepted, {len(negatives)} integrity checks fire")


def check_live(scenes: int):
    """Run every diagnostic program in MuJoCo from the identical restored scene."""
    from waddle_wm.pools import Scene

    counts = {}
    for seed in range(scenes):
        scene_obj = Scene(seed)
        assert Scene(seed).observation.observation_id == scene_obj.observation.observation_id, \
            f"seed {seed}: the observation is not reproducible"
        print(f"\nseed {seed}  observation {scene_obj.observation.observation_id}")
        for name, kind, program in prog.diagnostic_programs():
            if program.aborts:
                counts.setdefault(name, []).append(False)
                print(f"  {name:26s} declined: {program.aborts}")
                continue
            scene_obj.restore()
            episodes = scene_obj.execute(program)
            episode = episodes[-1]
            counts.setdefault(name, []).append(bool(episode.success))
            print(f"  {name:26s} success={str(episode.success):5s} failure={episode.failure_mode or '-':12s} "
                  f"attempts={len(episodes)} lift={episode.state_after['max_block_z']:.3f} "
                  f"target={episode.state_after['target_distance']:.3f}")
        scene_obj.restore()
        retry_probe = prog.canonical_program(grasp_offset_mm=(35.0, 0.0), max_attempts=1)
        retry_episodes = scene_obj.execute(retry_probe)
        assert len(retry_episodes) == 2 and all(e.failure_mode == "missed" for e in retry_episodes), \
            [episode.failure_mode for episode in retry_episodes]
        print("  bounded retry probe        attempts=2 (both missed, then stopped)")
        scene_obj.close()

    print(f"\ndiagnostic suite over {scenes} scenes (a planted fault is a bug in the program, not a "
          f"guaranteed failure — physics decides):")
    kinds = {name: kind for name, kind, _ in prog.diagnostic_programs()}
    for name, outcomes in counts.items():
        print(f"  {kinds[name]:9s} {name:26s} {sum(outcomes)}/{len(outcomes)} succeeded")


def check_claude(model: str):
    """One real generation round trip, to prove the prompt still produces a valid program."""
    from waddle_wm.pools import Scene, one_sample

    scene_obj = Scene(0)
    result = one_sample(0, "pick up the red block and put it on the green pad",
                        scene_obj.observation.text, model, 180.0)
    assert not result["error"], result["error"]
    program = prog.parse(result["raw"])
    grounded = prog.ground(program, scene_obj.observation)
    unreachable = scene_obj.reachable(grounded.step.trace)
    assert unreachable is None, unreachable
    print(f"claude round trip passed ({model}, ${result['generation'].get('cost_usd') or 0:.3f}): "
          f"{program.strategy!r} -> {[e['phase'] for e in grounded.trace]}")
    scene_obj.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", type=int, default=0, help="execute the diagnostic suite on N scenes")
    ap.add_argument("--claude", action="store_true", help="also make one real Claude call")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = ap.parse_args()

    check_plan_encoding()
    check_schema()
    check_grounding()
    check_runtime_policy()
    check_pool_contract()
    if args.live:
        check_live(args.live)
    if args.claude:
        check_claude(args.model)


if __name__ == "__main__":
    main()
