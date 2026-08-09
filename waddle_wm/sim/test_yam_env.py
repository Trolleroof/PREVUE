"""Smoke tests for the YAM-arm tabletop environment.

    uv run python -m waddle_wm.sim.test_yam_env

Mirrors test_env.py, but the skills run on the I2RT YAM arm under Cartesian IK
rather than on the pseudo-gripper. `align` is a known gap: see the README.
"""

from __future__ import annotations

from waddle_wm.sim.yam_env import (
    ARM_DOF,
    GRASP_Z,
    TRAVEL_Z,
    YamTabletopEnv,
    top_down_rotation,
)

NOMINAL = {
    "top_grasp": {},
    "side_grasp": {"side": -1.0},
}
TRIALS = 5


def main() -> None:
    failures = []
    env = YamTabletopEnv(seed=5, width=128, height=128)

    # the IK has to put the gripper where the skills ask, or nothing else means much
    env.reset()
    for target in ([0, 0, TRAVEL_Z], [0, 0, GRASP_Z], [-0.05, 0.03, GRASP_Z]):
        _, residual = env.solve_ik(target, top_down_rotation(0.0))
        if residual > 2e-3:
            failures.append(f"ik for {target} off by {residual * 1000:.1f} mm")
    print("ik      reaches the skill waypoints")

    for skill, params in NOMINAL.items():
        ok = 0
        for _ in range(TRIALS):
            env.reset(env.sample_scene())
            ok += env.run_skill(skill, params).success
        print(f"nominal {skill:11s} {ok}/{TRIALS} succeeded")
        if ok < TRIALS - 1:
            failures.append(f"{skill} nominal success {ok}/{TRIALS}")

    # reaching across the bin wall has to fail, and fail for the right reason
    env.reset()
    ep = env.run_skill("side_grasp", {"side": 1.0})
    print(f"bad     side_grasp  -> {ep.failure_mode}")
    if ep.success or ep.failure_mode != "collision_with_bin_wall":
        failures.append(f"side_grasp side=+1: expected a wall collision, got {ep.failure_mode}")

    env.reset()
    ep = env.run_skill("top_grasp", {})
    if ep.frames.shape[1:] != (128, 128, 3) or len(ep.frames) < 10:
        failures.append(f"unexpected clip shape {ep.frames.shape}")
    if len(ep.state_after["arm_qpos"]) != ARM_DOF:
        failures.append("arm_qpos missing from the recorded state")
    print(f"clip    {ep.frames.shape} at {env.fps} fps")

    # Known gap, reported rather than asserted: YAM cannot push this block.
    env.reset()
    ep = env.run_skill("align", {})
    print(f"known   align       -> success={ep.success} failure={ep.failure_mode} (see README)")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
