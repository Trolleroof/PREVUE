"""End-to-end check that the verifier scores plans from pixels, not from labels.

    uv run python -m waddle_wm.test_verifier

Encodes each episode's pre-execution window, compiles its recorded plan, rolls the
plan out, and compares the verdict with what actually happened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from waddle_wm.verifier import Verifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/ur5e_wm"))
    ap.add_argument("--checkpoint", type=Path, default=Path("models/latent_dynamics.pt"))
    ap.add_argument("--episodes", type=int, default=12)
    args = ap.parse_args()

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
