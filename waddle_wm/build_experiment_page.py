"""Build the standalone experiment page from the recorded demo traces.

The page is a self-contained HTML file: the traces in ``results/demo`` are inlined, so
it opens over ``file://`` with no server, no MuJoCo, and no Claude call. Nothing on the
page is invented — every number it shows is read out of a trace written by
``python -m waddle_wm.demo``.

    uv run python -m waddle_wm.build_experiment_page
    open results/demo/experiment.html
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).resolve().parent / "ui" / "experiment.html"
COMPARE = Path(__file__).resolve().parent / "ui" / "compare.html"
ARMS = ("none", "world-model")
SCENARIOS = ("grasp_miss", "place_miss")

#: Trace fields the page renders. Everything else in a trace is dropped so the inlined
#: payload stays small enough to read by hand.
ARM_FIELDS = (
    "decision",
    "reason",
    "executed_success",
    "failure_mode",
    "final_block_xy",
    "target_distance",
    "max_block_z",
    "seconds",
    "cost_usd",
)


def _round(value, places=4):
    """Recursively round floats so the inlined JSON stays readable."""
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, list):
        return [_round(item, places) for item in value]
    if isinstance(value, dict):
        return {key: _round(item, places) for key, item in value.items()}
    return value


def _round_trip(trace: dict) -> list[dict]:
    """The waypoints an arm actually ran, per repair round, as x/y/z targets."""
    rounds = []
    for entry in trace["rounds"]:
        plan = entry["plan"]
        waypoints = plan.get("trace") or plan["steps"][0]["trace"]
        rounds.append(
            {
                "round": entry["round"],
                "kind": entry["kind"],
                "note": entry["note"],
                "grasp_xy": entry["grasp_xy"],
                "place_xy": entry["place_xy"],
                "verdict": entry["verdict"],
                "skipped_reason": entry["skipped_reason"],
                "trace": [
                    {"phase": step["phase"], "target": step.get("target")}
                    for step in waypoints
                ],
            }
        )
    return rounds


def clips(traces: Path) -> tuple[dict, dict]:
    """MuJoCo footage per (scenario, arm), inlined as data URIs so the page stays one file.

    Arms that ran identical waypoints produce identical files; those share one entry.
    """
    by_digest: dict[str, str] = {}
    index: dict[str, str] = {}
    for scenario in SCENARIOS:
        for arm in ARMS:
            path = traces / f"{scenario}.{arm}.clip.mp4"
            if not path.exists():
                continue
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()[:16]
            by_digest.setdefault(digest, "data:video/mp4;base64," + base64.b64encode(raw).decode())
            index[f"{scenario}.{arm}"] = digest
    return by_digest, index


def collect(traces: Path) -> dict:
    """Read the six traces plus the sweep table into the page's data payload."""
    summary = json.loads((traces / "summary.json").read_text())
    sweep = json.loads((traces / "sweep.json").read_text())

    scenarios = {}
    for scenario in SCENARIOS:
        arms = {}
        for arm in ARMS:
            trace = json.loads((traces / f"{scenario}.{arm}.json").read_text())
            arms[arm] = {**{key: trace[key] for key in ARM_FIELDS}, "rounds": _round_trip(trace)}
        head = json.loads((traces / f"{scenario}.none.json").read_text())
        scenarios[scenario] = {
            "flaw": head["flaw"],
            "tests": head["tests"],
            "instruction": head["instruction"],
            "block_xy": head["block_xy"],
            "scene": head["scene"],
            "arms": arms,
        }

    # The sweep is the same two flawed plans over eight scenes; the page shows the rates.
    rates = {}
    for record in sweep["arms"]:
        if record["arm"] not in ARMS:
            continue
        key = (record["scenario"], record["arm"])
        bucket = rates.setdefault(key, {"scenes": 0, "caught": 0, "success": 0})
        bucket["scenes"] += 1
        bucket["success"] += bool(record["executed_success"])
        opening = record["rounds"][0]["verdict"]
        bucket["caught"] += bool(opening and not opening["approved"])

    videos, video_index = clips(traces)

    return _round(
        {
            "generated": summary["generated"],
            "videos": videos,
            "video_index": video_index,
            "model": summary["model"],
            "seed": summary["seed"],
            "checkpoint": summary["checkpoint"],
            "headline": {"seconds": summary["seconds"], "cost_usd": summary["cost_usd"]},
            "sweep": {
                "seconds": sweep["seconds"],
                "cost_usd": sweep["cost_usd"],
                "scenes": len(sweep["scenes"]),
                "rates": [
                    {"scenario": scenario, "arm": arm, **bucket}
                    for (scenario, arm), bucket in rates.items()
                ],
            },
            "scenarios": scenarios,
        }
    )


def build(traces: Path, out: Path, template: Path = TEMPLATE, payload: str | None = None) -> Path:
    source = template.read_text()
    marker = "/*__DATA__*/null"
    if marker not in source:
        raise SystemExit(f"{template} is missing the {marker} placeholder")
    payload = payload if payload is not None else json.dumps(collect(traces), separators=(",", ":"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(source.replace(marker, payload))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=REPO / "results" / "demo")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "demo" / "experiment.html")
    args = parser.parse_args()
    # Both pages read the same payload, so they can never disagree about a number.
    payload = json.dumps(collect(args.traces), separators=(",", ":"))
    for template, out in ((TEMPLATE, args.out), (COMPARE, args.out.with_name("compare.html"))):
        written = build(args.traces, out, template, payload)
        print(f"wrote {written} ({written.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
