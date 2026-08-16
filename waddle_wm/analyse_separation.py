"""What the verifier's probability is actually good for, from sweeps already on disk.

    uv run python -m waddle_wm.analyse_separation

Every sweep so far reported a *catch rate*: how often the verifier rejected a plan it was
supposed to reject. That number cannot distinguish two very different verifiers — one that
detects gross defects and one that predicts outcomes — because every flaw it was ever tested
on was gross. This script separates those two questions, using runs that already exist.

The pairing that makes it possible: `demo.py` hands **every arm the identical opening plan**,
and the `none` arm executes that plan regardless of any verdict. So for each scene we have

    the world-model arm's probability for a plan   <->   what that exact plan then did in MuJoCo

with no selection bias, which is the thing a rejection normally destroys — a rejected plan is
never run, so it has no outcome to score against. Here it does.

Two questions, and they do not have the same answer:

| question | population | what a good score means |
| --- | --- | --- |
| **separation** | flawed plans vs competent plans | the probability detects a defect |
| **ranking** | competent plans only | the probability predicts an outcome |

Reporting a single pooled AUC over both populations is Simpson's paradox and inflates the
result badly: scripted-flaw sweeps are ~all failures at low p, live-plan sweeps are ~all
successes at high p, so pooling measures "which sweep did this come from", which is trivial.
The pooled number is printed only so it can be seen next to the two that mean something.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Sweeps to read. `scripted` sweeps inject a known flaw into the opening plan; `live` sweeps
#: let the planner author its own, so their plans are the competent population.
DEFAULT_SWEEPS = (
    ("scripted-flaw", REPO / "results" / "demo" / "sweep.json"),
    ("live-plan", REPO / "results" / "demo-live-haiku24" / "sweep.json"),
    ("live-plan", REPO / "results" / "demo-live-opus24" / "sweep.json"),
)


def auc(positive: list[float], negative: list[float]) -> float | None:
    """P(a random positive scores above a random negative), ties counted as half.

    Returns None when either class is empty — an AUC over one class is undefined, and the
    per-sweep table has several such rows worth showing as gaps rather than as zeros.
    """
    if not positive or not negative:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in positive for b in negative)
    return wins / (len(positive) * len(negative))


def opening_verdicts(path: Path) -> list[dict]:
    """Every scene in one sweep as `{p, succeeded, scenario, scene}`.

    `p` is the world-model arm's verdict on the *opening* plan; `succeeded` is what the `none`
    arm's execution of that same opening plan did. Scenes whose opening plan was never scored,
    or that have no `none` arm to supply ground truth, are dropped.
    """
    sweep = json.loads(path.read_text())
    records = sweep["arms"]
    truth = {(r["scenario"], r.get("scene_index")): bool(r["executed_success"])
             for r in records if r["arm"] == "none"}

    rows = []
    for record in records:
        if record["arm"] != "world-model":
            continue
        key = (record["scenario"], record.get("scene_index"))
        verdicts = [r["verdict"] for r in record["rounds"] if r.get("verdict")]
        if key not in truth or not verdicts:
            continue
        rows.append({"p": verdicts[0]["imagined_success_probability"], "succeeded": truth[key],
                     "scenario": record["scenario"], "scene": record.get("scene_index"),
                     "model": sweep.get("model"), "live_plan": bool(sweep.get("live_plan", False))})
    return rows


def describe(values: list[float]) -> str:
    return f"n={len(values):<3} p in [{min(values):.3f}, {max(values):.3f}]  mean {sum(values) / len(values):.3f}"


def report(groups: dict[str, list[dict]]) -> str:
    flawed = [row["p"] for row in groups.get("scripted-flaw", [])]
    competent = [row["p"] for row in groups.get("live-plan", [])]
    lines = ["# Verifier probability: what it separates, and what it does not", ""]

    if flawed and competent:
        overlap = max(flawed) >= min(competent)
        lines += ["## 1. Separation — flawed plans vs competent plans", "",
                  f"- scripted-flawed plans: {describe(flawed)}",
                  f"- model-authored plans:  {describe(competent)}",
                  "",
                  f"Separation AUC **{auc(competent, flawed):.3f}** "
                  f"({len(competent)} competent vs {len(flawed)} flawed).",
                  ""]
        if not overlap:
            lines.append(f"The two populations do not overlap: the highest-scoring flawed plan is "
                         f"{max(flawed):.3f} and the lowest-scoring competent plan is {min(competent):.3f}, "
                         f"so any threshold in [{max(flawed):.3f}, {min(competent):.3f}] classifies every "
                         f"plan correctly.")
        else:
            lines.append(f"The populations overlap ({max(flawed):.3f} vs {min(competent):.3f}), so no "
                         f"threshold separates them cleanly.")
        lines.append("")

    # The question the catch rate cannot answer: among plans that all look reasonable, does a
    # higher probability actually mean a likelier success?
    lines += ["## 2. Ranking — within the competent plans only", ""]
    live = groups.get("live-plan", [])
    won = [row["p"] for row in live if row["succeeded"]]
    lost = [row["p"] for row in live if not row["succeeded"]]
    ranking = auc(won, lost)
    if ranking is None:
        lines.append(f"Undefined — {len(won)} successes and {len(lost)} failures, so one class is empty.")
    else:
        lines += [f"Ranking AUC **{ranking:.3f}** ({len(won)} successes vs {len(lost)} failures).",
                  "",
                  f"- mean p on plans that succeeded: {sum(won) / len(won):.3f}",
                  f"- mean p on plans that failed:    {sum(lost) / len(lost):.3f}",
                  "",
                  "Probabilities of the plans that failed: "
                  + ", ".join(f"{p:.3f}" for p in sorted(lost, reverse=True)) + ".",
                  ""]
        if len(lost) < 10:
            lines.append(f"**{len(lost)} failures is too few to estimate this reliably** — read it as "
                         f"'no evidence of ranking ability', not as a trustworthy point estimate.")
    lines.append("")

    pooled_pos = [row["p"] for rows in groups.values() for row in rows if row["succeeded"]]
    pooled_neg = [row["p"] for rows in groups.values() for row in rows if not row["succeeded"]]
    pooled = auc(pooled_pos, pooled_neg)
    lines += ["## 3. The pooled number, and why it is misleading", "",
              f"Pooled over both populations the AUC is **{pooled:.3f}** "
              f"({len(pooled_pos)} successes vs {len(pooled_neg)} failures) — far better than §2.",
              "",
              "That gap is Simpson's paradox, not a result. The scripted-flaw sweeps contribute "
              "almost only failures at low p and the live-plan sweeps almost only successes at high p, "
              "so a pooled AUC mostly measures which sweep a plan came from. Quote §1 and §2; "
              "this figure is here to be discounted, not cited.", ""]

    lines += ["## Per-sweep detail", "",
              "| sweep | model | scenes | successes | failures | mean p |",
              "| --- | --- | --- | --- | --- | --- |"]
    for kind, rows in groups.items():
        for model in dict.fromkeys(row["model"] for row in rows):
            subset = [row for row in rows if row["model"] == model]
            wins = sum(row["succeeded"] for row in subset)
            ps = [row["p"] for row in subset]
            lines.append(f"| {kind} | {model} | {len(subset)} | {wins} | {len(subset) - wins} | "
                         f"{sum(ps) / len(ps):.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", action="append", metavar="KIND=PATH",
                    help="extra sweep to include, e.g. live-plan=results/mine/sweep.json")
    ap.add_argument("--out", type=Path, help="write the report here as well as to stdout")
    args = ap.parse_args()

    sweeps = list(DEFAULT_SWEEPS)
    for entry in args.sweep or []:
        kind, _, path = entry.partition("=")
        sweeps.append((kind, Path(path)))

    groups: dict[str, list[dict]] = {}
    for kind, path in sweeps:
        if not path.exists():
            print(f"skipping {path} (not found)", flush=True)
            continue
        groups.setdefault(kind, []).extend(opening_verdicts(path))

    if not groups:
        raise SystemExit("no sweeps found — run `python -m waddle_wm.demo --sweep N` first")

    text = report(groups)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
