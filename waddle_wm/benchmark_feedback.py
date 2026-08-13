"""Paired bounded repair benchmark for issue #26.

The primary budget is proposals per scene. MuJoCo calls are measured, not matched;
``--call-budget`` is the secondary equal-call view.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from waddle_wm.agent import SkillAgent

#: arm -> (verifier mode, where its repairs come from). The first three repair from an
#: imagined failure before the arm moves; only `mujoco` is allowed to learn from a real one.
ARMS = {
    "claude-self-critique": ("none", False),
    "estimated-state": ("rules", False),
    "visual-world-model": ("world-model", False),
    "mujoco": ("none", True),
}


def row(arm, prompt, seed, payload, elapsed, proposal_budget):
    execution = payload.get("execution") or {}
    attempts = execution.get("attempts") or execution.get("steps") or [execution]
    calls = len(attempts)
    rounds = payload.get("rounds") or []
    verifier_approved = any((item.get("verdict") or {}).get("approved") for item in rounds)
    return {
        "arm": arm, "prompt": prompt, "seed": seed,
        "proposal_budget": proposal_budget,
        "proposals": len(payload.get("rounds") or []),
        "mujoco_calls": calls, "solve": bool(execution.get("success")),
        "first_try_success": bool(execution.get("first_try_success")),
        "verifier_approved": verifier_approved,
        "failed_executions": sum(item.get("success") is False for item in attempts),
        "repairs": max(0, calls - 1), "cost_usd": payload.get("cost_usd", 0.0),
        "latency_s": round(elapsed, 3), "decision": payload.get("decision"),
    }


def summarize(rows, proposal_budget, call_budget=None):
    """Return both fair views without treating the budgets as interchangeable."""
    report = {"primary_budget": {"unit": "proposals_per_scene", "value": proposal_budget},
              "secondary_budget": {"unit": "mujoco_calls_per_scene", "value": call_budget},
              "arms": {}}
    for arm in sorted({item["arm"] for item in rows}):
        subset = [item for item in rows if item["arm"] == arm]
        if call_budget is not None:
            subset = [item for item in subset if item["mujoco_calls"] <= call_budget]
        n = len(subset)
        report["arms"][arm] = {
            "scenes": n,
            "solve_rate": sum(item["solve"] for item in subset) / n if n else None,
            "mean_mujoco_calls": sum(item["mujoco_calls"] for item in subset) / n if n else None,
            "mean_proposals": sum(item["proposals"] for item in subset) / n if n else None,
            "mean_cost_usd": sum(item["cost_usd"] for item in subset) / n if n else None,
            "false_accepts": sum(item["verifier_approved"] and not item["solve"] for item in subset),
        }
    return report


def run_suite(prompts, seeds, args):
    rows, proposal_budget = [], args.repairs + 1
    for prompt in prompts:
        for seed in seeds:
            for arm, (mode, feedback) in ARMS.items():
                agent = SkillAgent(args.checkpoint, seed, args.model, args.repairs,
                                   verifier_mode=mode, feedback=feedback)
                started = time.time()
                run = agent.run(prompt, seed)
                rows.append(row(arm, prompt, seed, run.as_json(), time.time() - started,
                                proposal_budget))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", action="append", dest="prompts",
                    default=["pick up the red block and put it on the green pad"])
    ap.add_argument("--seed", action="append", type=int)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--repairs", type=int, default=2)
    ap.add_argument("--call-budget", type=int, default=None,
                    help="optional secondary equal-MuJoCo-call slice")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--checkpoint", type=Path, default=Path("models/multiblock_world_model.pt"))
    ap.add_argument("--out", type=Path, default=Path("results/benchmark_feedback.json"))
    args = ap.parse_args()
    seeds = args.seed if args.seed is not None else list(range(args.episodes))
    rows = run_suite(args.prompts, seeds, args)
    report = {"rows": rows, "summary": summarize(rows, args.repairs + 1, args.call_budget)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
