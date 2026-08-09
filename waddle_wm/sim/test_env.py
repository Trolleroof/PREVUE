"""Smoke tests for the MuJoCo tabletop environment.

    uv run python -m waddle_wm.sim.test_env
"""

from __future__ import annotations

from waddle_wm.sim.env import CLOSED_WIDTH, TARGET_XY, TabletopEnv

NOMINAL = {
    "top_grasp": {"grip_width": CLOSED_WIDTH},
    "side_grasp": {"side": -1.0, "grip_width": CLOSED_WIDTH},
    "align": {"push_to_x": float(TARGET_XY[0])},
}

# a bad trace per skill, with the failure mode it is supposed to produce
BAD = [
    ("side_grasp", {"side": 1.0}, "collision_with_bin_wall"),
    ("top_grasp", {"offset_y": -0.04}, "missed"),  # +0.04 would graze the wall instead
    ("align", {"push_to_x": -0.10}, "undershoot"),
]


def main() -> None:
    failures = []

    env = TabletopEnv(seed=5)
    for skill, params in NOMINAL.items():
        ok = 0
        for _ in range(6):
            env.reset(env.sample_scene())
            ep = env.run_skill(skill, params)
            ok += ep.success
        print(f"nominal {skill:11s} {ok}/6 succeeded")
        if ok < 5:
            failures.append(f"{skill} nominal success {ok}/6 below 5/6")

    for skill, params, expected in BAD:
        env.reset()
        ep = env.run_skill(skill, params)
        print(f"bad     {skill:11s} -> {ep.failure_mode}")
        if ep.success or ep.failure_mode != expected:
            failures.append(f"{skill} {params}: expected {expected}, got {ep.failure_mode}")

    env.reset()
    ep = env.run_skill("top_grasp", NOMINAL["top_grasp"])
    if ep.frames.shape[1:] != (256, 256, 3) or len(ep.frames) < 10:
        failures.append(f"unexpected clip shape {ep.frames.shape}")
    print(f"clip    {ep.frames.shape} at {env.fps} fps")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
