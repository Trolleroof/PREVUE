"""Regression test for the UR5e/Robotiq pick-and-place physics.

    uv run python -m waddle_wm.sim.test_render_ur5e_demo
"""

from __future__ import annotations

import mujoco
import numpy as np

from waddle_wm.sim import relling_scene as scene
from waddle_wm.sim.render_ur5e_demo import run_pick_place

MIN_LIFT = 0.09
MIN_SEPARATION = 0.035


def main() -> None:
    model = scene.make_model()
    data = mujoco.MjData(model)
    scene.reset(model, data)
    renderer = mujoco.Renderer(model, 64, 64)
    frames = []

    max_lift, placed = run_pick_place(model, data, renderer, frames)

    failures = []
    if max_lift <= MIN_LIFT:
        failures.append(f"max_lift {max_lift:.3f}m did not clear {MIN_LIFT}m")

    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            sep = float(np.linalg.norm(a[:2] - b[:2]))
            if sep <= MIN_SEPARATION:
                failures.append(f"final blocks {sep:.3f}m apart, below {MIN_SEPARATION}m")

    print(f"max_lift={max_lift:.3f}m")
    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            print(f"separation {float(np.linalg.norm(a[:2] - b[:2])):.3f}m")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
