"""Checks for the planner contract and for the agent's live (non-dataset) operating point.

    uv run python -m waddle_wm.test_agent                 # offline: plan parsing and validation
    uv run python -m waddle_wm.test_agent --live 24       # + verifier vs physics on freshly rendered scenes
    uv run python -m waddle_wm.test_agent --claude        # + one real Claude Opus 5 round trip

The live check is the part worth running before trusting a demo: the verifier was trained
on decoded `.mp4` windows, so a window rendered live has to travel the same path
(`verifier.encode_live`). Encoding raw renderer output instead moves the frozen backbone
off-distribution badly enough to reject plans that succeed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from waddle_wm.agent import DEFAULT_CHECKPOINT
from waddle_wm.planner import MAX_PHASES, PlanError, parse

GOOD = {"intent": "put the block on the pad", "action": "execute", "note": "aimed at the observed centre",
        "trace": [{"phase": "approach", "target": [0.35, -0.2, 0.24]},
                  {"phase": "descend", "target": [0.35, -0.2, 0.015]},
                  {"phase": "close"},
                  {"phase": "lift", "target": [0.35, -0.2, 0.24]},
                  {"phase": "move", "target": [0.5, 0.3, 0.3]},
                  {"phase": "place", "target": [0.5, 0.3, 0.015]},
                  {"phase": "open"},
                  {"phase": "retreat", "target": [0.5, 0.3, 0.24]}]}


def check_contract():
    plan = parse("```json\n" + json.dumps(GOOD) + "\n```")
    assert plan.executable and plan.pick_place_shaped, plan
    assert plan.trace[2] == {"phase": "close", "value": 255.0}, plan.trace[2]
    assert [entry["phase"] for entry in plan.trace][0] == "approach"

    abstain = parse(json.dumps({**GOOD, "action": "abstain", "note": "nothing called 'the mug' is visible"}))
    assert not abstain.executable and abstain.trace == [], abstain

    free = parse(json.dumps({**GOOD, "trace": GOOD["trace"][:1]}))
    assert free.executable and not free.pick_place_shaped, free

    for name, payload, fragment in [
        ("prose instead of JSON", "Sure! I would approach the block first.", "not JSON"),
        ("unknown phase", {**GOOD, "trace": [{"phase": "wiggle", "target": [0.4, 0.0, 0.2]}]}, "phase must be one of"),
        ("missing target", {**GOOD, "trace": [{"phase": "approach"}]}, "needs"),
        ("outside the workspace", {**GOOD, "trace": [{"phase": "approach", "target": [0.4, 0.0, 3.0]}]}, "outside the workspace"),
        ("too many phases", {**GOOD, "trace": GOOD["trace"] * 2}, f"at most {MAX_PHASES}"),
        ("no trace to execute", {**GOOD, "trace": []}, "non-empty trace"),
        ("missing key", {"action": "execute"}, "missing key"),
    ]:
        try:
            parse(payload if isinstance(payload, str) else json.dumps(payload))
        except PlanError as error:
            assert fragment in str(error), (name, str(error))
        else:
            raise AssertionError(f"{name} should have been rejected")
    print(f"plan contract passed: 1 valid plan, 1 abstention, 1 free-form trace, 7 rejections")


def check_live(episodes: int, checkpoint: Path, seed: int):
    """Score randomly-aimed plans on freshly rendered scenes, then actually run them."""
    from waddle_wm.sim.env import TabletopEnv, pick_place_trace
    from waddle_wm.verifier import Verifier

    verifier = Verifier(checkpoint)
    manifest = verifier.manifest
    env = TabletopEnv(seed=seed, block_spawn_low=manifest["block_spawn_low"], block_spawn_high=manifest["block_spawn_high"])
    rng = np.random.default_rng(seed)
    rows, agree = [], 0
    for index in range(episodes):
        block = env.sample_scene()
        target = np.array([0.5, 0.3]) + (rng.uniform([-0.20, -0.06], [0.0, 0.06]) if rng.random() < 0.45 else 0.0)
        angle, radius = rng.uniform(0, 2 * np.pi), rng.uniform(*((0.024, 0.042) if rng.random() < 0.4 else (0.0, 0.012)))
        offset = [radius * np.cos(angle), radius * np.sin(angle)]

        env.reset(block)
        latent = verifier.encode_live(env.observation_frames(manifest["window_frames"]))
        result = verifier.verify(latent, pick_place_trace(block, target, offset))
        env.reset(block)
        episode = env.run_skill("pick_place", {"target_xy": target.tolist(), "grasp_offset_xy": offset})

        agree += result.approve == episode.success
        rows.append((result.approve, episode.success, episode.failure_mode))
        print(f"{index + 1:3d}/{episodes} p={result.success_probability:.3f} approve={result.approve} "
              f"actual={'success' if episode.success else episode.failure_mode}", flush=True)

    lifted_wrong = sum(1 for approve, success, mode in rows if approve and mode == "missed")
    target_wrong = sum(1 for approve, success, mode in rows if approve and mode == "target_miss")
    print(f"\nlive agreement: {agree}/{len(rows)} = {agree / len(rows):.3f}")
    print(f"false accepts: {lifted_wrong} grasp misses, {target_wrong} target misses "
          f"(of {sum(1 for _, success, _ in rows if not success)} real failures)")
    print(f"false rejects: {sum(1 for approve, success, _ in rows if success and not approve)} "
          f"(of {sum(1 for _, success, _ in rows if success)} real successes)")


def check_perception(scenes: int, seed: int):
    """Every detection must land close enough to the true pose to be graspable."""
    from waddle_wm.perception import QUERIES, SceneCamera
    from waddle_wm.sim.env import TabletopEnv

    env = TabletopEnv(seed=seed)
    camera = SceneCamera(env.model, env.data)
    joints = {"red block": "red_block_free", "blue block": "blue_block_free", "yellow block": "yellow_block_free"}
    errors = []
    for _ in range(scenes):
        env.reset(env.sample_scene())
        detections = camera.detect_all(QUERIES)
        assert len(detections) == len(QUERIES), [d.label for d in detections]
        for detection in detections:
            truth = env.data.joint(joints[detection.label]).qpos[:3]
            error = np.linalg.norm(np.subtract(detection.point_base[:2], truth[:2]))
            assert error < 0.015, (detection.summary(), truth.tolist())
            assert 0.025 < detection.size_m < 0.05, detection.summary()
            errors.append(error)
    assert camera.bounding_box("the purple teapot") is None, "matched an object that is not in the scene"
    print(f"perception passed: {len(errors)} detections, lateral error mean {np.mean(errors) * 1000:.1f} mm, "
          f"max {np.max(errors) * 1000:.1f} mm (grasp tolerance is about 28 mm)")


def check_claude(model: str):
    from waddle_wm.perception import QUERIES, SceneCamera
    from waddle_wm.planner import ClaudePlanner, describe_observation
    from waddle_wm.perception import landing_pad
    from waddle_wm.sim.env import TabletopEnv

    env = TabletopEnv(seed=1)
    env.reset(env.sample_scene())
    camera = SceneCamera(env.model, env.data)
    detections = camera.detect_all(QUERIES)
    observation = describe_observation(detections, landing_pad(env.model), env.state()["gripper_pos"])
    plan = ClaudePlanner(model=model).propose("pick up the red block and put it on the green pad", observation)
    assert plan.executable and plan.pick_place_shaped, plan.summary()
    grasp = plan.trace[1]["target"][:2]
    detected = next(d for d in detections if d.label == "red block").point_base[:2]
    assert np.linalg.norm(np.subtract(grasp, detected)) < 0.01, (grasp, detected)
    print(f"claude round trip passed: {model} returned a valid pick-and-place trace aimed at the detected block")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=8, help="scenes to check the perception primitives on")
    ap.add_argument("--live", type=int, default=0, help="render this many fresh scenes and score them")
    ap.add_argument("--claude", action="store_true", help="also make one real Claude call")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    check_contract()
    check_perception(args.scenes, args.seed)
    if args.live:
        check_live(args.live, args.checkpoint, args.seed)
    if args.claude:
        check_claude(args.model)


if __name__ == "__main__":
    main()
