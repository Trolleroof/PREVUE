"""Small runnable check for the physical UR5e dataset path."""
from waddle_wm.sim.env import TabletopEnv

def main():
    env = TabletopEnv(seed=5)
    env.reset(env.sample_scene()); good = env.run_skill("pick_place")
    assert good.success, good.failure_mode
    assert good.frames.shape[1:] == (256, 256, 3) and len(good.frames) >= 10
    env.reset(env.sample_scene()); bad = env.run_skill("pick_place", {"target_xy": [0.32, 0.3]})
    assert not bad.success and bad.failure_mode == "target_miss", bad.failure_mode
    print(f"UR5e dataset check passed: {len(good.frames)} frames, bad={bad.failure_mode}")

if __name__ == "__main__": main()
