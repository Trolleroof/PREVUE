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
from copy import deepcopy

from waddle_wm import program as prog
from waddle_wm.program import ProgramError, SceneObservation

SCENE = SceneObservation(
    {"red block": [0.3812, -0.1804, 0.0180], "blue block": [0.5031, -0.1612, 0.0180],
     "yellow block": [0.4507, 0.0503, 0.0180], "green pad": [0.5000, 0.3000, 0.0],
     "gripper": [0.4, 0.0, 0.3]}, pad_radius=0.105, seed=0)


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

    faults = dict(prog.fault_programs())
    assert set(faults) == {"correct", "correct_redetect_regrasp", *prog.FAULTS}, sorted(faults)
    keys = {name: prog.ground(program, SCENE).dedup_key() for name, program in faults.items()}
    assert len(set(keys.values())) == len(keys), keys
    print(f"grounding passed: canonical shape, offsets, symbolic source, dedup, {len(faults)} diagnostics")


def check_pool_contract():
    from waddle_wm.pools import PREFIXES, check_pool

    candidates = []
    for index, (name, program) in enumerate(prog.fault_programs()):
        grounded = prog.ground(program, SCENE)
        candidates.append({"candidate_id": f"c{index:02d}", "index": index, "sample_index": index,
                           "program": program.as_json(), "grounded_trace": grounded.step.summary()["trace"],
                           "dedup_key": grounded.dedup_key(), "duplicate_of": None,
                           "validation": {"ok": True, "stage": "accepted", "error": None},
                           "retry": program.retry, "redetect_ops": program.redetects, "fault": name})
    ids = [candidate["candidate_id"] for candidate in candidates]
    pool = {"pool_id": "diagnostic-test", "kind": "diagnostic", "split": "train",
            "protocol": {"protocol_version": 1, "program_schema_version": prog.SCHEMA_VERSION,
                         "git_sha": "test", "generator": {"model": "scripted", "pool_size": len(ids)}},
            "task": {"instruction": "put the red block on the green pad", "object": "red block",
                     "destination": "green pad"},
            "scene": {"seed": 0, "observation_id": SCENE.observation_id, "observation": "",
                      "detections": [], "hidden_truth": {"red_block": [0.3801, -0.1799, 0.018]}},
            "candidates": candidates,
            "prefixes": {str(n): ids[:n] for n in PREFIXES if n <= len(ids)},
            "summary": {}}
    assert check_pool(pool) == [], check_pool(pool)

    negatives = [
        ("duplicate candidate id", lambda p: p["candidates"][1].update(candidate_id=p["candidates"][0]["candidate_id"]),
         "duplicate candidate_id"),
        ("non-nested prefix", lambda p: p["prefixes"].__setitem__("4", list(reversed(p["prefixes"]["4"]))),
         "not nested"),
        ("short prefix", lambda p: p["prefixes"].__setitem__("4", p["prefixes"]["4"][:2]), "holds 2 candidates"),
        ("gap in the ranking", lambda p: p["candidates"][2].update(index=9),
         "not a contiguous ranking"),
        ("missing grounded trace", lambda p: p["candidates"][0].update(grounded_trace=[]), "empty grounded trace"),
        ("wrong schema version", lambda p: p["candidates"][0]["program"].update(schema_version=99),
         "wrong program schema version"),
        ("repeated fault", lambda p: p["candidates"][1].update(fault=p["candidates"][0]["fault"]),
         "repeats a fault"),
        ("scripted natural pool", lambda p: p.update(kind="natural"), "must be generated by Claude"),
        ("leaked ground truth",
         lambda p: p["candidates"][0]["program"]["ops"].append({"op": "note", "x": 0.3801}),
         "ground truth"),
    ]
    for name, break_it, fragment in negatives:
        broken = deepcopy(pool)
        break_it(broken)
        problems = check_pool(broken)
        assert any(fragment in problem for problem in problems), (name, problems)
    print(f"pool contract passed: clean pool accepted, {len(negatives)} integrity checks fire")


def check_live(scenes: int):
    """Run every diagnostic program in MuJoCo from the identical restored scene."""
    from waddle_wm.pools import Scene
    from waddle_wm.sim.env import PRELUDE_FRAMES

    counts = {}
    for seed in range(scenes):
        scene_obj = Scene(seed)
        assert Scene(seed).observation.observation_id == scene_obj.observation.observation_id, \
            f"seed {seed}: the observation is not reproducible"
        print(f"\nseed {seed}  observation {scene_obj.observation.observation_id}")
        for name, program in prog.fault_programs():
            grounded = prog.ground(program, scene_obj.observation)
            scene_obj.restore()
            episode = scene_obj.env.run_trace(grounded.step.trace, prelude_frames=PRELUDE_FRAMES,
                                              block="red_block", destination="green_pad")
            counts.setdefault(name, []).append(bool(episode.success))
            print(f"  {name:26s} success={str(episode.success):5s} failure={episode.failure_mode or '-':12s} "
                  f"lift={episode.state_after['max_block_z']:.3f} target={episode.state_after['target_distance']:.3f}")
        scene_obj.close()

    print(f"\ndiagnostic suite over {scenes} scenes (a planted fault is a bug in the program, not a "
          f"guaranteed failure — physics decides):")
    for name, outcomes in counts.items():
        print(f"  {name:26s} {sum(outcomes)}/{len(outcomes)} succeeded")


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

    check_schema()
    check_grounding()
    check_pool_contract()
    if args.live:
        check_live(args.live)
    if args.claude:
        check_claude(args.model)


if __name__ == "__main__":
    main()
