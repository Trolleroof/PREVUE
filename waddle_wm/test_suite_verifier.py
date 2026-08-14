"""Check that the serving path computes what the training path fitted.

The expensive bug in this project's history was not a modelling mistake, it was a train/serve
mismatch: a feature that was constant during fitting, divided at inference by a clamp floor,
turned every candidate into a confident rejection. Nothing about the loss curve showed it. So
the checks here are parity checks — same episode, same numbers, both sides — plus the refusals
that stop a checkpoint being used for a question it cannot answer.

No MuJoCo and no training: the checkpoint and the episode are both synthetic, with the shapes
and statistics a real one has.

    uv run python -m waddle_wm.test_suite_verifier
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from waddle_wm import plan_encoding
from waddle_wm.sim import relling_scene as scene
from waddle_wm.sim.generate_suite import FRAMES_TOTAL, PRELUDE_FRAMES, SUITE_BLOCK_SIZE, WINDOWS, build_trace
from waddle_wm.sim.test_generate_suite import subtask as make_subtask
from waddle_wm.train_task_suite_world_model import (SUBTASK_SLOTS, SuiteWorldModel, apply_normaliser,
                                                    assemble, fit_normaliser)

NAMES = tuple(scene.BLOCK_NAMES)
REST_Z = SUITE_BLOCK_SIZE[2]
PAD = [0.50, 0.30]
CONTEXT_DIM = 32
FAILURES: list[str] = []


def check(condition, message):
    if condition:
        print(f"PASS  {message}")
    else:
        print(f"FAIL  {message}")
        FAILURES.append(message)


def manifest() -> dict:
    return {"schema_version": 5, "fps": 10, "frames_total": FRAMES_TOTAL,
            "prelude_frames": PRELUDE_FRAMES, "window_frames": 8, "windows": WINDOWS,
            "block_names": list(NAMES), "home_waypoint": [-0.134, 0.492, 0.332],
            "phase_frames": {"approach": 6.0, "descend": 4.0, "close": 5.0, "lift": 4.0,
                             "move": 4.0, "place": 4.0, "open": 4.0, "retreat": 4.0}}


def spawn_positions() -> dict:
    return {"red_block": [0.35, -0.20, REST_Z], "blue_block": [0.55, -0.22, REST_Z],
            "yellow_block": [0.64, -0.12, REST_Z]}


def make_record(episode_id: str, subtasks: list[dict], family: str, split: str = "train") -> dict:
    positions = spawn_positions()
    trace = build_trace(subtasks, positions)
    blocks = np.array([positions[name] for name in NAMES])
    per_frame = np.repeat(blocks[None], FRAMES_TOTAL, axis=0)
    per_frame[FRAMES_TOTAL // 2:, 0, 2] = 0.20            # something is lifted at some point
    tracks = {"all_block_pos": per_frame.tolist(),
              "pinch_pos": [[-0.134, 0.492, 0.332]] * FRAMES_TOTAL}
    outcomes = [{"object": s["object"], "destination": s["destination"], "success": index == 0,
                 "lifted": True, "placed": index == 0, "resting": True,
                 "final_distance_m": 0.01, "failure_mode": None if index == 0 else "target_miss"}
                for index, s in enumerate(subtasks)]
    return {"schema_version": 5, "episode_id": episode_id, "split": split, "family": family,
            "observation": {"frames_path": f"clips/{episode_id}.mp4", "frames_total": FRAMES_TOTAL,
                            "prelude_frames": PRELUDE_FRAMES, "window_frames": 8, "fps": 10},
            "skill": {"name": "task_suite", "trace": trace,
                      "params": {"family": family, "subtasks": subtasks,
                                 "spawn_positions": positions,
                                 "block_yaws_deg": {name: 0.0 for name in NAMES}}},
            "tracks": tracks,
            "subtask_outcomes": outcomes,
            "outcome": {"success": all(o["success"] for o in outcomes), "failure_mode": None,
                        "failed_subtask": None}}


def build_checkpoint(records, cache, path: Path, threshold: float = 0.5) -> dict:
    """A structurally real checkpoint: correct shapes, statistics fitted on these records."""
    raw = assemble(records, manifest(), cache, 2)
    train = raw["splits"] == "train"
    normalisers = {key: fit_normaliser(raw[key], train) for key in ("context", "initial")}
    normalisers["plan"] = fit_normaliser(raw["plan"], train, raw["mask"])
    normalisers["final"] = normalisers["initial"]
    torch.manual_seed(0)
    members = [SuiteWorldModel(CONTEXT_DIM, raw["plan"].shape[2], raw["task"].shape[2],
                               hidden=32, context_width=16, dropout=0.0) for _ in range(2)]
    present = raw["plan"][train].reshape(-1, raw["plan"].shape[-1])[raw["mask"][train].reshape(-1) > 0]
    encoding = plan_encoding.yaw_informative(present, np.ones(len(present), dtype=bool), 2)
    saved = {"model_type": "task_suite_state",
             "members": [{k: v.cpu() for k, v in m.state_dict().items()} for m in members],
             "context_dim": CONTEXT_DIM, "plan_dim": raw["plan"].shape[2],
             "task_dim": raw["task"].shape[2], "member_count": len(members),
             "hidden": 32, "context_width": 16, "dropout": 0.0,
             "subtask_slots": SUBTASK_SLOTS, "manifest": manifest(),
             "normalisation": normalisers, "plan_encoding": encoding,
             "decision_threshold": threshold, "metrics": {}}
    torch.save(saved, path)
    return raw


def main():
    from waddle_wm.suite_verifier import SuiteVerifier
    from waddle_wm.verifier import Verifier

    aligned = make_subtask("red_block", "green_pad", PAD, grasp_yaw_deg=0.0, block_yaw_deg=0.0)
    across = make_subtask("red_block", "green_pad", PAD, grasp_yaw_deg=80.0, block_yaw_deg=0.0)
    second = make_subtask("blue_block", "red_block", PAD, place_z=REST_Z + 2 * REST_Z,
                          grasp_yaw_deg=-35.0, block_yaw_deg=-35.0)

    records = []
    for index in range(24):
        # A mix of one- and two-subtask episodes, with varied headings so nothing is degenerate.
        turn = float(index * 7 - 80) / 1.0
        first = make_subtask("red_block", "green_pad", PAD, grasp_yaw_deg=max(-90.0, min(90.0, turn)))
        if index % 2:
            records.append(make_record(f"e{index:03d}", [first], "pad_place",
                                       "train" if index < 18 else "test"))
        else:
            records.append(make_record(f"e{index:03d}", [first, second], "ordered_stack",
                                       "train" if index < 18 else "test"))

    torch.manual_seed(1)
    cache = {record["episode_id"]: torch.randn(1, CONTEXT_DIM) for record in records}

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        raw = build_checkpoint(records, cache, path)
        verifier = SuiteVerifier(path, device=torch.device("cpu"))

        # --- parity: the serving features equal the training features -------------------------
        worst_plan = worst_state = worst_task = 0.0
        for index, record in enumerate(records):
            plan, task, mask, state = verifier._features(
                record["skill"]["trace"], record["skill"]["params"]["subtasks"],
                {**record["skill"]["params"]["spawn_positions"],
                 "pinch": manifest()["home_waypoint"]}, record["family"])
            expected_plan = apply_normaliser(torch.from_numpy(raw["plan"][index : index + 1]),
                                             verifier.normalisation["plan"])
            expected_state = apply_normaliser(torch.from_numpy(raw["initial"][index : index + 1]),
                                              verifier.normalisation["initial"])
            worst_plan = max(worst_plan, float((plan - expected_plan).abs().max()))
            worst_state = max(worst_state, float((state - expected_state).abs().max()))
            worst_task = max(worst_task, float((task - torch.from_numpy(raw["task"][index:index + 1])).abs().max()))
            assert mask.tolist() == [raw["mask"][index].tolist()]
        check(worst_plan < 1e-6, f"serving plan features match training (max diff {worst_plan:.2e})")
        check(worst_state < 1e-6, f"serving state features match training (max diff {worst_state:.2e})")
        check(worst_task < 1e-6, f"serving task features match training (max diff {worst_task:.2e})")

        # --- the heading changes the verdict ---------------------------------------------------
        positions = spawn_positions()
        context = torch.randn(1, CONTEXT_DIM)
        a = verifier.verify(context, build_trace([aligned], positions), [aligned], positions)
        b = verifier.verify(context, build_trace([across], positions), [across], positions)
        check(abs(a.success_probability - b.success_probability) > 1e-6,
              "two plans differing only in commanded heading get different probabilities")

        # --- probabilities are usable numbers ---------------------------------------------------
        check(0.0 <= a.success_probability <= 1.0 and a.uncertainty >= 0.0,
              "p(success) is a probability and uncertainty is non-negative")
        check(len(a.subtasks) == 1 and a.subtasks[0].object == "red_block",
              "one verdict per subtask, naming the block it is about")

        two = verifier.verify(context, build_trace([aligned, second], positions),
                              [aligned, second], positions)
        check(len(two.subtasks) == 2 and two.subtasks[1].destination == "red_block",
              "a two-step plan is scored per step, with the stack's destination named")
        check(two.success_probability <= min(s.success_probability for s in two.subtasks) + 1e-6,
              "the plan is no more likely than its least likely step, as the label defines it")

        # --- the no-vision ablation is reachable and actually differs ----------------------------
        blind = verifier.verify(context, build_trace([aligned], positions), [aligned], positions,
                                use_context=False)
        check(abs(blind.success_probability - a.success_probability) > 1e-9,
              "the no-vision ablation gives a different answer than the visual path")

        # --- refusals ---------------------------------------------------------------------------
        try:
            Verifier(path)
            check(False, "Verifier refuses a task_suite checkpoint")
        except ValueError as error:
            check("suite_verifier" in str(error),
                  "Verifier refuses a task_suite checkpoint and names the class that serves it")

        blind_path = Path(directory) / "blind.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        saved["plan_encoding"] = {**saved["plan_encoding"], "yaw_informative": False}
        torch.save(saved, blind_path)
        try:
            SuiteVerifier(blind_path, device=torch.device("cpu"))
            check(False, "an orientation-blind checkpoint is refused")
        except (ValueError, plan_encoding.PlanEncodingError):
            check(True, "an orientation-blind checkpoint is refused unless asked for by name")
        SuiteVerifier(blind_path, device=torch.device("cpu"), allow_orientation_blind=True)
        check(True, "...and loads when the caller opts in explicitly")

        try:
            verifier.verify(context, build_trace([aligned], positions), [aligned, second], positions)
            check(False, "a trace that does not match its subtasks is refused")
        except ValueError:
            check(True, "a trace whose segments do not match its subtask list is refused")

    print(f"\n{'FAILED' if FAILURES else 'OK'}: {len(FAILURES)} failing checks")
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
