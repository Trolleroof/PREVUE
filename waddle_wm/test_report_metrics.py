"""Check the §3 grouping that the reported metrics are decomposed by.

The grouping decides which decisions a plan-only control could already make, so
mislabelling it would quietly flatter the verifier.

    uv run python -m waddle_wm.test_report_metrics
"""
from waddle_wm.report_metrics import group_of
from waddle_wm.sim.env import LIFT_THRESHOLD, TARGET_RADIUS


def as_record(target_xy, max_block_z, target_pos=(0.5, 0.3)):
    return {"skill": {"params": {"target_xy": list(target_xy)}},
            "state_before": {"target_pos": list(target_pos)},
            "tracks": {"max_block_z": [0.02, max_block_z, 0.02]}}


def main():
    lifted, dropped = LIFT_THRESHOLD + 0.05, LIFT_THRESHOLD - 0.01
    assert group_of(as_record((0.5, 0.3), lifted)) == "A"
    assert group_of(as_record((0.5, 0.3), dropped)) == "C"
    # off the zone dominates: the plan alone condemns it however the grasp went
    for lift in (lifted, dropped):
        assert group_of(as_record((0.5 + 2 * TARGET_RADIUS, 0.3), lift)) == "B"
    # aiming inside the zone but off-centre is still a plan the grasp gets to decide
    assert group_of(as_record((0.5 + 0.5 * TARGET_RADIUS, 0.3), lifted)) == "A"
    # the boundary belongs to "on zone", matching env.py's `<=` success test
    assert group_of(as_record((0.5 + TARGET_RADIUS, 0.3), dropped)) == "C"
    print("ok")


if __name__ == "__main__":
    main()
