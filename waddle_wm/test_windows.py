"""Check the transition contract: executed and compiled actions describe the same plan.

    uv run python -m waddle_wm.test_windows
"""
import numpy as np

from waddle_wm import windows
from waddle_wm.actions import ACTION_DIM, PHASES
from waddle_wm.sim.env import FRAMES_TOTAL, PRELUDE_FRAMES, WINDOW_FRAMES, TabletopEnv
from waddle_wm.sim.generate_dataset import phase_frames


def as_record(episode, split):
    return {"episode_id": f"ep_{split}", "split": split,
            "skill": {"name": episode.skill, "params": episode.params, "trace": episode.skill_trace},
            "tracks": episode.tracks, "outcome": {"success": episode.success, "failure_mode": episode.failure_mode}}


def main():
    env = TabletopEnv(seed=11)
    records = []
    for split, params in (("train", {}), ("val", {"target_xy": [0.34, 0.3]})):
        env.reset(env.sample_scene())
        records.append(as_record(env.run_skill("pick_place", params), split))
    manifest = {"frames_total": FRAMES_TOTAL, "prelude_frames": PRELUDE_FRAMES, "window_frames": WINDOW_FRAMES,
                "windows": FRAMES_TOTAL // WINDOW_FRAMES, "home_waypoint": env.home_waypoint(),
                "phase_frames": phase_frames(records)}

    assert list(windows.anchors(manifest)) == [7, 15, 23, 31, 39, 47], windows.anchors(manifest)

    executed = windows.build(records, manifest)
    compiled = windows.build(records, manifest, planned=True)
    steps = manifest["windows"] - 1
    assert executed["action"].shape == (2, steps, WINDOW_FRAMES, ACTION_DIM), executed["action"].shape
    assert compiled["action"].shape == executed["action"].shape
    assert executed["state"].shape == (2, manifest["windows"], 6 + 2), executed["state"].shape

    for name, data in (("executed", executed), ("compiled", compiled)):
        phases = data["action"][0].reshape(-1, ACTION_DIM)[:, :len(PHASES)].argmax(1)
        assert set(phases.tolist()) >= {1, 3, 5}, (name, phases)   # approach, close, move all inside the chunks
    gripper = executed["action"][0].reshape(-1, ACTION_DIM)[:, -1]
    assert gripper.max() == 1.0 and gripper.min() == 0.0, gripper

    waypoints = windows.planned_actions(records[0], manifest)[:, len(PHASES):len(PHASES) + 3]
    place = np.asarray(records[0]["skill"]["trace"][5]["target"])
    assert np.isclose(waypoints, place).all(1).any(), "compiled plan never commands the place waypoint"

    terminal = executed["state"][:, -1]
    assert np.array_equal((terminal[:, 6] * terminal[:, 7]).astype(bool), executed["success"].astype(bool)), terminal

    print(f"window contract ok: {steps} transitions/episode, action dim {ACTION_DIM}, "
          f"phase_frames={ {k: int(v) for k, v in manifest['phase_frames'].items()} }")


if __name__ == "__main__":
    main()
