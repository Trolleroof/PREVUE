"""Small accounting check for the issue #26 budget contract."""
from waddle_wm.benchmark_feedback import summarize


def main():
    rows = [
        {"arm": "a", "solve": True, "mujoco_calls": 2, "proposals": 2, "cost_usd": 1.0, "verifier_approved": True},
        {"arm": "a", "solve": False, "mujoco_calls": 3, "proposals": 3, "cost_usd": 1.0, "verifier_approved": True},
        {"arm": "b", "solve": True, "mujoco_calls": 1, "proposals": 2, "cost_usd": 2.0, "verifier_approved": False},
    ]
    primary = summarize(rows, 3)
    assert primary["primary_budget"] == {"unit": "proposals_per_scene", "value": 3}
    assert primary["arms"]["a"]["solve_rate"] == 0.5
    assert primary["arms"]["a"]["mean_mujoco_calls"] == 2.5
    secondary = summarize(rows, 3, call_budget=2)
    assert secondary["arms"]["a"]["scenes"] == 1
    assert secondary["arms"]["b"]["scenes"] == 1
    print("benchmark feedback accounting passed: proposal-primary and call-secondary views")


if __name__ == "__main__":
    main()
