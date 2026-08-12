"""End-to-end check that the verifier scores plans from pixels, not from labels.

    uv run python -m waddle_wm.test_verifier
    uv run python -m waddle_wm.test_verifier --multiblock 3     # the normalisation guard, live

Encodes each episode's pre-execution window, compiles its recorded plan, rolls the
plan out, and compares the verdict with what actually happened.

`--multiblock` is the regression test for the guard in `verifier.standardise`. The
multi-block checkpoint has one plan feature that was constant while it was fitted, so its
standard deviation sits on the trainer's clamp floor; without the guard a millimetre of
camera noise in that dimension saturates the ensemble and *every* plan scores exactly 0.
The check scores three plans that physics separates and requires the verifier to separate
them too.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from waddle_wm.verifier import DEGENERATE_STD, Verifier, degenerate_dimensions, standardise


def check_standardise():
    """The guard itself: a degenerate dimension is held, a live one is not."""
    mean = torch.tensor([0.0, -0.003])
    std = torch.tensor([0.05, 1e-6])
    scaled = standardise(torch.tensor([[0.10, 0.0024]]), mean, std)
    assert abs(float(scaled[0, 0]) - 2.0) < 1e-6, scaled
    assert float(scaled[0, 1]) == 0.0, f"a constant-in-training dimension must not be divided: {scaled}"
    # Unguarded, that same 5.4 mm of perception noise is what reaches the network.
    raw = (torch.tensor([[0.10, 0.0024]]) - mean) / std
    assert float(raw[0, 1]) > 5000, raw

    clamped = standardise(torch.tensor([[9.0]]), torch.tensor([0.0]), torch.tensor([1.0]))
    assert float(clamped[0, 0]) == 5.0, clamped
    assert degenerate_dimensions({"plan_std": std, "plan_mean": mean}) == {"plan_std": [1]}
    assert degenerate_dimensions({"plan_std": torch.tensor([0.05, 0.05])}) == {}
    print(f"guard passed: degenerate dimensions (std <= {DEGENERATE_STD:g}) are held at their "
          f"training constant, the rest are clamped to +-5")


def check_multiblock(scenes: int, checkpoint: Path, seed: int = 100):
    """Score three plans from *camera* coordinates on fresh scenes and require separation."""
    from waddle_wm.perception import QUERIES, SceneCamera
    from waddle_wm.sim.env import TabletopEnv, WINDOW_FRAMES, pick_place_trace

    def trace(block_xy, target_xy, offset=(0.0, 0.0)):
        # float32 on the way in: a trace built with numpy carries float64, which MPS refuses.
        return [{**entry, **({"target": [np.float32(v) for v in entry["target"]]} if "target" in entry else {})}
                for entry in pick_place_trace(block_xy, target_xy, offset)]

    verifier = Verifier(checkpoint)
    assert verifier.model_type == "multiblock_state", verifier.model_type
    print(f"checkpoint degenerate dimensions: {verifier.degenerate}")
    env = TabletopEnv(seed=seed)
    camera = SceneCamera(env.model, env.data)
    try:
        for index in range(scenes):
            env.reset(env.sample_scene())
            latent = verifier.encode_live(env.observation_frames(WINDOW_FRAMES))
            positions = {d.label.replace(" ", "_"): d.point_base for d in camera.detect_all(QUERIES)}
            positions["green_pad"] = [0.5, 0.3, 0.0]
            red = np.asarray(positions["red_block"][:2])
            scores = {name: verifier.verify(latent, plan, "red_block", "green_pad", positions)
                      for name, plan in (("centred", trace(red, [0.5, 0.3])),
                                          ("grasp 5 cm off", trace(red, [0.5, 0.3], (0.05, 0.0))),
                                          ("place 25 cm off", trace(red, [0.72, 0.30])))}
            line = "  ".join(f"{name} {row.success_probability:.3f}" for name, row in scores.items())
            print(f"scene {index + 1}/{scenes}: {line}")
            distinct = {round(row.success_probability, 4) for row in scores.values()}
            assert len(distinct) == len(scores), f"the verifier gave {distinct} to three different plans"
            assert scores["centred"].success_probability > scores["grasp 5 cm off"].success_probability, scores
            assert scores["centred"].success_probability > 0.5, scores["centred"]
    finally:
        camera.close()
    print(f"multiblock check passed: {scenes} scene(s) scored from camera coordinates, plans separated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/ur5e_wm"))
    ap.add_argument("--checkpoint", type=Path, default=Path("models/latent_dynamics.pt"))
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--multiblock", type=int, default=0,
                    help="run the normalisation-guard check on N freshly rendered scenes and exit")
    ap.add_argument("--multiblock-checkpoint", type=Path,
                    default=Path("models/multiblock_world_model.pt"))
    args = ap.parse_args()

    check_standardise()
    if args.multiblock:
        check_multiblock(args.multiblock, args.multiblock_checkpoint)
        return

    records = [json.loads(line) for line in (args.data / "records.jsonl").open()]
    test = [record for record in records if record["split"] == "test"][:args.episodes]
    assert test, "no test-split episodes"
    verifier = Verifier(args.checkpoint)

    agree = 0
    for record in test:
        params = record["skill"]["params"]
        latent = verifier.observation_window(args.data, record["episode_id"])
        result = verifier.verify_pick_place(latent, record["state_before"]["block_pos"][:2],
                                            params["target_xy"], params.get("grasp_offset_xy", (0.0, 0.0)))
        assert 0.0 <= result.success_probability <= 1.0, result
        assert result.uncertainty >= 0.0 and len(result.predicted_block_xy) == 2, result
        assert result.approve == (result.success_probability >= verifier.threshold)
        assert (result.likely_failure is None) == (result.lifted_probability >= 0.5 and result.in_target_probability >= 0.5)
        truth = record["outcome"]["success"]
        agree += result.approve == truth
        print(f"{record['episode_id']}: p(success)={result.success_probability:.3f} +-{result.uncertainty:.3f} "
              f"approve={result.approve} actual={'success' if truth else record['outcome']['failure_mode']}"
              f"{'' if result.likely_failure is None else '  -> ' + result.likely_failure}")

    print(f"verifier check passed: {agree}/{len(test)} verdicts matched the real outcome")


if __name__ == "__main__":
    main()
