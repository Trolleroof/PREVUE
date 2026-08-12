"""Contract checks for the demo, and the regression that made the live verifier useless.

    uv run python -m waddle_wm.test_demo

Everything here is offline: the scenario definitions, the replay reconstruction, and the
normalisation guard are checked without MuJoCo, Claude, or a rendered frame. Pass
`--checkpoint` to also assert the guard against the real saved statistics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from waddle_wm.demo import ARMS, SCENARIOS, RecordedPlanner, recorded_plans, report
from waddle_wm.planner import PICK_PLACE_SHAPE, Plan
from waddle_wm.verifier import Verifier


def check_scenarios():
    """Every scenario must produce a plan the verifier and the simulator both accept."""
    for scenario in SCENARIOS:
        plan = scenario.opening_plan((0.42, -0.20, 0.013), (0.50, 0.30))
        assert plan.executable and plan.pick_place_shaped, scenario.name
        step = plan.steps[0]
        assert tuple(entry["phase"] for entry in step.trace) == PICK_PLACE_SHAPE, scenario.name
        grasp = next(e["target"] for e in step.trace if e["phase"] == "descend")
        place = next(e["target"] for e in step.trace if e["phase"] == "place")
        # The flaw has to be big enough to actually fail: the grasp tolerance is ~28 mm and the
        # landing zone radius is 105 mm. A scenario that only just misses would make the demo a
        # coin flip on the controller rather than a test of the verifier.
        offset = max(abs(scenario.grasp_offset[0]), abs(scenario.grasp_offset[1]))
        target_offset = max(abs(scenario.place_offset[0]), abs(scenario.place_offset[1]))
        assert offset > 0.028 or target_offset > 0.105, f"{scenario.name}: flaw is within tolerance"
        assert abs(grasp[0] - 0.42 - scenario.grasp_offset[0]) < 1e-6, scenario.name
        assert abs(place[1] - 0.30 - scenario.place_offset[1]) < 1e-6, scenario.name
    print(f"scenarios: {len(SCENARIOS)} definitions produce executable, out-of-tolerance plans")


def check_replay_roundtrip():
    """A saved trace must rebuild the exact plans the recorded run used."""
    scenario = SCENARIOS[0]
    opening = scenario.opening_plan((0.42, -0.20, 0.013), (0.50, 0.30))
    repaired = scenario.opening_plan((0.42, -0.20, 0.013), (0.50, 0.30))
    trace = {"rounds": [{"plan": opening.summary()}, {"plan": repaired.summary()}]}
    rebuilt_opening, repairs = recorded_plans(trace)
    assert rebuilt_opening.summary() == opening.summary()
    assert len(repairs) == 1

    planner = RecordedPlanner(repairs)
    assert isinstance(planner.repair("", "", opening, {}), Plan)
    try:
        planner.repair("", "", opening, {})
    except RuntimeError:
        pass
    else:
        raise AssertionError("a replay must refuse to invent a repair the recording does not have")
    try:
        planner.propose("", "")
    except RuntimeError:
        pass
    else:
        raise AssertionError("a replay must not propose; it starts from the recorded opening plan")
    print("replay: traces rebuild their plans, and the recorded planner refuses to improvise")


def check_report():
    """The report must name every arm and survive an arm that was never verified."""
    records = [{"scenario": SCENARIOS[0].name, "arm": arm, "rounds": [
        {"round": 0, "kind": "given", "verdict": None, "skipped_reason": "verifier disabled",
         "plan": {}} if arm == "none" else
        {"round": 0, "kind": "given", "skipped_reason": None, "plan": {},
         "verdict": {"imagined_success_probability": 0.07, "ensemble_uncertainty": 0.09,
                     "approved": False, "likely_failure": "grasp misses the block"}}],
        "decision": "executed", "executed_success": arm != "none", "failure_mode": "missed",
        "final_block_xy": [0.42, -0.2], "target_distance": 0.51} for arm in ARMS]
    text = report(records)
    for arm in ARMS:
        assert f"| {arm} |" in text, arm
    assert "verifier disabled" in text
    print("report: renders every arm, including the one with no verdict")


def check_constant_feature_guard(checkpoint: Path | None):
    """The bug: a constant training feature normalised by `clamp_min`'s floor explodes live.

    `grasp_z - block_z` is identical in all 900 recorded episodes, because every recorded trace
    descends to `GRASP_Z` and every recorded block height is read out of MuJoCo. Its stored std is
    therefore the 1e-6 clamp floor, not a scale. Live, the block height comes from the depth buffer,
    and the few-millimetre disagreement divided by 1e-6 reached the ensemble as ~3.5e3 sigma. Every
    verdict came back p(success) = 0.000, "grasp misses the block", with the imagined block position
    tens of metres off the table — for good plans as readily as bad ones.
    """
    class Stub(Verifier):
        def __init__(self, norm):
            self.norm = norm

    norm = {"plan_mean": torch.tensor([0.0, 0.0, -0.0029]),
            "plan_std": torch.tensor([0.0141, 0.0145, 1e-6])}
    live = torch.tensor([[0.001, -0.002, 0.0035]])
    naive = (live - norm["plan_mean"]) / norm["plan_std"]
    guarded = Stub(norm).constant_features(live, "plan")
    assert naive[0, 2].abs() > 1e3, "the fixture no longer reproduces the blow-up"
    assert guarded[0, 2] == 0.0, "a constant training feature must contribute what training saw"
    assert torch.allclose(guarded[0, :2], naive[0, :2]), "healthy features must be untouched"
    print(f"guard: degenerate feature {naive[0, 2]:.0f} sigma -> {guarded[0, 2]:.0f}, others unchanged")

    if checkpoint is None:
        return
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    degenerate = (saved["normalization"]["plan_std"] <= Verifier.DEGENERATE_STD).nonzero().flatten()
    assert len(degenerate), (f"{checkpoint} has no degenerate plan feature; if the corpus now varies "
                             "grasp height the guard is harmless, but this check should be revisited")
    print(f"guard: {checkpoint} has constant plan features at {degenerate.tolist()}, all pinned")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, help="also check the guard against real saved statistics")
    ap.add_argument("--traces", type=Path, help="also check that saved demo traces are replayable")
    args = ap.parse_args()

    check_scenarios()
    check_replay_roundtrip()
    check_report()
    check_constant_feature_guard(args.checkpoint)

    if args.traces:
        for scenario in SCENARIOS:
            for arm in ARMS:
                saved = json.loads((args.traces / f"{scenario.name}.{arm}.json").read_text())
                opening, repairs = recorded_plans(saved)
                assert opening.summary() == saved["opening_plan"], f"{scenario.name}/{arm}"
                assert len(repairs) == len(saved["rounds"]) - 1
        print(f"traces: every arm in {args.traces} rebuilds its plans")

    print("demo checks passed")


if __name__ == "__main__":
    main()
