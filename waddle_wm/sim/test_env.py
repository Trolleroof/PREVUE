"""Small runnable check for the schema-3 UR5e dataset path.

    uv run python -m waddle_wm.sim.test_env
"""
import math

from waddle_wm.actions import PHASE_ID
from waddle_wm.sim.env import FRAMES_TOTAL, PRELUDE_FRAMES, TRACK_KEYS, TabletopEnv, pick_place_trace
from waddle_wm.sim.generate_dataset import YAW_PHASES, orient_source


def check_grid(episode):
    assert len(episode.frames) == FRAMES_TOTAL, len(episode.frames)
    assert episode.frames.shape[1:] == (256, 256, 3), episode.frames.shape
    for key in TRACK_KEYS:
        assert len(episode.tracks[key]) == FRAMES_TOTAL, (key, len(episode.tracks[key]))
    phase = episode.tracks["phase"]
    assert phase[:PRELUDE_FRAMES] == [PHASE_ID["idle"]] * PRELUDE_FRAMES, phase[:PRELUDE_FRAMES]
    assert phase[PRELUDE_FRAMES] == PHASE_ID["approach"], phase[PRELUDE_FRAMES]
    spans = [entry["frames"] for entry in episode.skill_trace]
    assert spans[0][0] == PRELUDE_FRAMES, spans[0]
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start == end + 1, (end, start)
    assert spans[-1][1] < FRAMES_TOTAL, spans[-1]


def main():
    env = TabletopEnv(seed=5)
    env.reset(env.sample_scene()); good = env.run_skill("pick_place")
    assert good.success, good.failure_mode
    check_grid(good)

    env.reset(env.sample_scene()); miss = env.run_skill("pick_place", {"target_xy": [0.32, 0.3]})
    assert not miss.success and miss.failure_mode == "target_miss", miss.failure_mode
    check_grid(miss)

    env.reset(env.sample_scene()); dropped = env.run_skill("pick_place", {"grasp_offset_xy": [0.036, 0.0]})
    assert not dropped.success and dropped.failure_mode == "missed", dropped.failure_mode
    assert not dropped.tracks["max_block_z"][-1] > 0.09, dropped.tracks["max_block_z"][-1]
    check_grid(dropped)

    env.reset(env.sample_scene())
    assert env.approach_until([[0.4, -0.2, 0.30], [0.4, -0.2, 0.02]], stop=env.pinch_below(0.12))
    assert env.data.site("2f85/pinch").xpos[2] <= 0.121, env.data.site("2f85/pinch").xpos[2]
    env.reset(env.sample_scene())
    assert not env.approach_until([[0.4, -0.2, 0.24]]), "no criterion should mean no early stop"

    env.reset(block_xy=(0.4, -0.18)); orient_source(env, "red_block", 35.0)
    oriented_trace = pick_place_trace(env.state()["block_pos"], env.state()["target_pos"])
    for entry in oriented_trace:
        if entry["phase"] in YAW_PHASES:
            entry["yaw"] = math.radians(35.0)
    oriented = env.run_trace(oriented_trace)
    assert oriented.success, oriented.failure_mode

    print(f"UR5e dataset check passed: {FRAMES_TOTAL} frames, modes={[good.failure_mode, miss.failure_mode, dropped.failure_mode]}, "
          f"approach_until stops on its criterion, rotated grasp succeeds")


if __name__ == "__main__":
    main()
