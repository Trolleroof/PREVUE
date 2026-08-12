"""#18: run all three selectors over one locked pool set and report what the choice was worth.

One command. It reads the frozen pools from #17 and the counterfactual outcomes from #23,
runs Claude self-rank, the estimated-state heuristic, and the visual world model over the
*same* nested prefixes, writes an artifact `benchmark_record.check_run` validates, and reports
paired results by pool size and by failure slice.

    uv run python -m waddle_wm.benchmark_selectors \
        --counterfactual data/counterfactual/test-natural.json --pools data/pools \
        --checkpoint models/multiblock_world_model.pt --out results/programs

Nothing is executed here. Every outcome in the report was measured by #23 before any selector
ran, so a selector cannot influence what it is scored against, and the answer key is applied
only after all three have committed to a choice. The selectors themselves live in
`waddle_wm.selectors`, which is also where the information boundary is enforced.

What the report says, and what it refuses to say:

* `selected_success` against candidate count, per selector, with the oracle's own curve as the
  hidden upper bound and `pool_has_success` as what generation made available at all;
* the plan-visible slice (obvious geometry or control defects) and the scene-dependent slice
  (grasp alignment, occlusion, obstruction, slip, state changes after an earlier action) are
  reported separately, because a selector that only catches malformed programs has not shown
  that vision helps;
* the causal verdict compares the visual arm with the same learned arm after zeroing only its
  visual context. The hand-coded estimated-state heuristic remains a secondary reference.
"""
from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

from waddle_wm import selectors as sel
from waddle_wm.benchmark_record import PRIMARY_OUTCOMES, NotComparable, aggregate, bootstrap_ci
from waddle_wm.counterfactual import check_execution, scenario_id_of, selector_view
from waddle_wm.perception_scenes import SLICES
from waddle_wm.pools import DEFAULT_ROOT as POOL_ROOT, PREFIXES, Scene, git_dirty

# The slice the visual arm exists to win: state the camera can see and a centre coordinate
# cannot carry. `latent_physics` is declared by #25 as not inferable from one initial frame,
# and `plan_visible_control` is decidable from the program text alone — both are reported,
# neither is the claim.
INTENDED_SLICE = "visible_omitted_by_coordinates"
CAUSAL_BASELINE = "matched_no_vision"
SLICE_GROUPS = {"plan_visible": ("plan_visible_control",),
                "scene_dependent": ("visible_omitted_by_coordinates", "latent_physics")}


# --------------------------------------------------------------------------- inputs


def load_pools(root: Path, artifact: dict) -> dict[str, dict]:
    """The cached pools behind this artifact, keyed by pool id. Every scene must find one."""
    pools = {}
    for path in sorted(Path(root).rglob("*.json")):
        if path.name == "index.json":
            continue
        pool = json.loads(path.read_text())
        pools[pool["pool_id"]] = pool
    wanted = {scene["pool_id"] for scene in artifact["scenes"]}
    missing = sorted(wanted - set(pools))
    if missing:
        raise SystemExit(f"no cached pool under {root} for {missing[:4]}; #18 ranks the frozen "
                         f"pools #23 executed, it does not regenerate them")
    return {pool_id: pools[pool_id] for pool_id in wanted}


def load_views(artifact_path: Path, pools: dict) -> dict[str, dict]:
    """#23's saved selector views, or the same projection rebuilt from the cached pools."""
    path = artifact_path.with_name(f"{artifact_path.stem}-views.json")
    views = json.loads(path.read_text()) if path.exists() else {}
    for pool_id, pool in pools.items():
        rebuilt = selector_view(pool)
        if pool_id not in views:
            views[pool_id] = rebuilt
        elif views[pool_id] != rebuilt:
            raise SystemExit(f"{pool_id}: the saved selector view does not match the cached pool; "
                             f"one of them changed after the run was locked")
    return views


def observation_window(pool: dict, frames: int) -> "object":
    """The raw camera window the visual arm reads, rendered from the pool's own scene.

    Rebuilt from the scene seed and the locked scene spec, then checked against the pool's
    observation id — a window from a scene that no longer reproduces the observation would be
    a different scene, and the scenario is excluded rather than scored on it.
    """
    import numpy as np

    suite = pool["scene"].get("suite") or {}
    scene_obj = Scene(pool["scene"]["seed"], suite.get("scene_spec"))
    try:
        if scene_obj.observation.observation_id != pool["scene"]["observation_id"]:
            raise ValueError("the scene no longer reproduces the pool's observation")
        return np.asarray(scene_obj.env.observation_frames(frames))
    finally:
        scene_obj.close()


# --------------------------------------------------------------------------- the run


def build_selectors(args) -> list[sel.Selector]:
    arms: list[sel.Selector] = []
    if not args.no_claude:
        arms.append(sel.ClaudeSelfRank(model=args.model, timeout=args.timeout,
                                       cache=args.cache / "claude_self_rank"))
    arms.append(sel.load_heuristic(args.weights))
    if not args.no_visual:
        arms.extend([
            sel.MatchedNoVision(args.checkpoint, args.encoder,
                                allow_orientation_blind=args.allow_orientation_blind),
            sel.VisualWorldModel(args.checkpoint, args.encoder,
                                 allow_orientation_blind=args.allow_orientation_blind),
        ])
    return arms


def run_selectors(artifact: dict, views: dict, pools: dict, arms: list[sel.Selector],
                  window_frames: int) -> tuple[dict, list[dict]]:
    """Score every (pool, prefix) once with every arm and fold the blocks into the scenes.

    A selector's input does not depend on the physics seed — it chooses before anything is
    executed, and it is never told which perturbation the executor will apply — so one block
    per (pool, prefix) is the honest record and is reused across the paired physics seeds.
    """
    artifact = deepcopy(artifact)
    prefixes = {}
    for scene in artifact["scenes"]:
        prefixes.setdefault((scene["pool_id"], scene["prefix"]), tuple(scene["pool_prefix"]))
    excluded, blocks, contexts = [], {}, {}

    for (pool_id, size), prefix in sorted(prefixes.items(), key=lambda item: (item[0][0], item[0][1])):
        if pool_id in contexts:
            context = contexts[pool_id]
        else:
            frames = None
            if any(arm.needs_frames for arm in arms):
                try:
                    frames = observation_window(pools[pool_id], window_frames)
                except (ValueError, RuntimeError, KeyError) as failure:
                    excluded.append({"scenario_id": scenario_id_of(pools[pool_id]), "pool_id": pool_id,
                                     "reason": f"could not render the observation window: {failure}"})
                    continue
            context = contexts[pool_id] = {
                "with_frames": sel.ScenarioContext(views[pool_id], frames),
                "without_frames": sel.ScenarioContext(views[pool_id]),
            }
        for arm in arms:
            source = context["with_frames" if arm.needs_frames else "without_frames"]
            started = time.time()
            blocks[(pool_id, size, arm.name)] = sel.rank(arm, source, list(prefix))
            print(f"  {pool_id} n={size:<3} {arm.name:<18} "
                  f"chose {blocks[(pool_id, size, arm.name)]['chosen']['candidate_id']} "
                  f"in {time.time() - started:.2f}s", flush=True)

    dropped = {item["pool_id"] for item in excluded}
    kept = []
    for scene in artifact["scenes"]:
        if scene["pool_id"] in dropped:
            continue
        for arm in arms:
            scene["selectors"][arm.name] = deepcopy(blocks[(scene["pool_id"], scene["prefix"], arm.name)])
        kept.append(scene)
    artifact["scenes"] = kept
    artifact["excluded"] = [*artifact.get("excluded", []), *excluded]
    artifact["metadata"] = deepcopy(artifact["metadata"])
    for arm in arms:
        artifact["metadata"]["selectors"][arm.name] = {
            "kind": "selector", "information_sources": list(arm.information_sources),
            "accept_threshold": arm.accept_threshold, "config_hash": arm.config_hash,
            "config": arm.config(), "cost_usd": round(arm.cost_usd, 6)}
    artifact["metadata"]["selector_protocol"] = {
        "selector_version": sel.SELECTOR_VERSION,
        "ranking_is_read_only": "candidates are fingerprinted before and after every call",
        "prefix_faithful": "a selector at N sees only the first N candidates",
        "physics_seed_blind": "one block per (pool, prefix), reused across paired physics seeds",
        "declined_probability": sel.DECLINED_PROBABILITY}
    return artifact, excluded


# --------------------------------------------------------------------------- reporting


def slice_of(pool: dict) -> tuple[str, str]:
    """(outcome slice, observability) for one pool, from #25's locked scene record."""
    suite = pool["scene"].get("suite") or {}
    return suite.get("outcome_slice", "unsliced"), suite.get("observability", "unsliced")


def prefix_coverage(artifact: dict, pools: dict) -> dict:
    """Which pools each prefix actually holds, and which fell short of it.

    A pool that ran out of accepted candidates before the requested size simply has no
    scene at the wider prefixes, so the `N = 32` and `N = 64` rows can quietly be averages
    over fewer scenes than the narrow ones. On a natural Claude pool that is the normal
    case, not an error — generation is sampled, not scripted — but a curve drawn over a
    shrinking population is not a curve about pool size. Every shortfall is named here
    with the count that caused it, and `balanced` says whether the comparison across
    prefixes was made on one fixed set of scenes.
    """
    candidates = {pool_id: len(pool["candidates"]) for pool_id, pool in pools.items()}
    ranked = sorted({scene["pool_id"] for scene in artifact["scenes"]})
    present = {size: set() for size in PREFIXES}
    for scene in artifact["scenes"]:
        present.setdefault(scene["prefix"], set()).add(scene["pool_id"])

    coverage = {}
    for size in sorted(present):
        short = [{"pool_id": pool_id,
                  "scenario_id": scenario_id_of(pools[pool_id]),
                  "candidates": candidates.get(pool_id),
                  "reason": f"pool holds {candidates.get(pool_id)} candidates, fewer than {size}"}
                 for pool_id in ranked if pool_id not in present[size]]
        coverage[str(size)] = {"pools": len(present[size]), "short_of_prefix": short}
    return {"ranked_pools": len(ranked), "requested_prefixes": list(PREFIXES),
            "by_prefix": coverage,
            "balanced": all(pool_ids == set(ranked) for pool_ids in present.values())}


def sub_artifact(artifact: dict, keep) -> dict | None:
    scenes = [scene for scene in artifact["scenes"] if keep(scene)]
    if not scenes:
        return None
    return {**artifact, "scenes": scenes}


def paired_difference(artifact: dict, left: str, right: str, metric: str) -> dict:
    """The per-scene paired difference `left - right` on one metric, over every prefix at once."""
    from waddle_wm.benchmark_record import pairing_key, scene_metrics

    values, seen = [], set()
    for scene in artifact["scenes"]:
        key = pairing_key(scene)
        if key in seen or left not in scene["selectors"] or right not in scene["selectors"]:
            continue
        seen.add(key)
        mine = scene_metrics(scene, left)[metric]
        theirs = scene_metrics(scene, right)[metric]
        if mine is None or theirs is None:
            continue
        values.append(float(mine) - float(theirs))
    return {"paired_scenes": len(values),
            "mean_difference": round(sum(values) / len(values), 4) if values else None,
            "ci95": bootstrap_ci(values)}


def report(artifact: dict, pools: dict, arms: list[str]) -> dict:
    """Overall and per-slice paired results, plus the one verdict the issue asks for."""
    by_pool = {pool_id: slice_of(pool) for pool_id, pool in pools.items()}
    overall = aggregate(artifact)

    slices = {}
    for name in sorted({row[0] for row in by_pool.values()}):
        part = sub_artifact(artifact, lambda scene, name=name: by_pool[scene["pool_id"]][0] == name)
        if part is None:
            continue
        try:
            slices[name] = {"observability": next(row[1] for row in by_pool.values() if row[0] == name),
                            "declared_observability": SLICES.get(name, "unsliced"),
                            **aggregate(part)}
        except NotComparable as failure:
            slices[name] = {"error": str(failure)}

    groups = {}
    for group, observabilities in SLICE_GROUPS.items():
        part = sub_artifact(artifact, lambda scene, o=observabilities: by_pool[scene["pool_id"]][1] in o)
        if part is not None:
            groups[group] = aggregate(part)

    intended = sub_artifact(artifact, lambda scene: by_pool[scene["pool_id"]][1] == INTENDED_SLICE)
    verdict = {"intended_slice": INTENDED_SLICE,
               "baseline": CAUSAL_BASELINE,
               "question": "does raw visual context improve selection over the otherwise "
                           "identical no-vision model on the scene-dependent slice?"}
    if intended is None:
        verdict.update({"answer": "not measured",
                        "statement": f"this run contains no {INTENDED_SLICE} scenes, so it says "
                                     f"nothing about whether visual information helps."})
    elif not {"visual_world_model", CAUSAL_BASELINE} <= set(arms):
        verdict.update({"answer": "not measured",
                        "statement": "both the visual world model and its matched no-vision "
                                     "ablation must run for the comparison to exist."})
    else:
        differences = {metric: paired_difference(intended, "visual_world_model", CAUSAL_BASELINE, metric)
                       for metric in PRIMARY_OUTCOMES}
        success = differences["selected_success"]
        interval = success["ci95"]
        beats = bool(success["mean_difference"] and success["mean_difference"] > 0
                     and interval and interval[0] > 0)
        verdict.update({
            "answer": "yes" if beats else "no",
            "scenes": success["paired_scenes"],
            "paired_differences": differences,
            "statement": (
                f"on the {INTENDED_SLICE} slice the visual world model selected a successful "
                f"program {success['mean_difference']:+.3f} more often than the matched no-vision "
                f"model (95% CI {interval}), over {success['paired_scenes']} paired scenes. "
                + ("The interval excludes zero, so visual information improved selection on the "
                   "slice it was meant to improve."
                   if beats else
                   "The interval does not exclude zero, so this run does not support a claim "
                   "that visual information improves selection."))})

    return {"artifact_scenes": len(artifact["scenes"]), "selectors": arms,
            "all_selectors": sorted(artifact["metadata"]["selectors"]),
            "primary_outcomes": list(PRIMARY_OUTCOMES),
            "overall": overall, "by_slice": slices, "by_observability": groups,
            "prefix_coverage": prefix_coverage(artifact, pools),
            "verdict": verdict, "excluded": artifact.get("excluded", [])}


def plot(report_data: dict, path: Path) -> None:
    """Selected success against candidate count, per selector, with the oracle above them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefixes = report_data["overall"]["prefixes"]
    sizes = sorted(int(key) for key in prefixes)
    figure, axes = plt.subplots(figsize=(7.2, 4.6))
    # Selectors that agree on every prefix draw the same line, so the markers have to differ:
    # an arm hidden under another arm's curve would read as an arm that was never run.
    markers, widths = ("o", "s", "^", "D", "v", "P"), (3.4, 2.6, 1.8, 1.2)
    for position, name in enumerate(report_data["all_selectors"]):
        values = [prefixes[str(size)]["selectors"][name]["selected_success"] for size in sizes]
        compared = name in report_data["selectors"]
        axes.plot(sizes, values, marker=markers[position % len(markers)],
                  markersize=9 - 1.4 * position, fillstyle="none" if position % 2 else "full",
                  linewidth=widths[position % len(widths)],
                  linestyle="-" if compared else ":", alpha=1.0 if compared else 0.7,
                  label=name if compared else f"{name} (reference)")
    oracle = [prefixes[str(size)]["pool_has_success"] for size in sizes]
    axes.plot(sizes, oracle, marker="*", linestyle="--", color="black",
              label="MuJoCo oracle (hidden upper bound)")
    axes.set_xscale("log", base=2)
    axes.set_xticks(sizes)
    axes.set_xticklabels([str(size) for size in sizes])
    axes.set_xlabel("candidates in the pool (nested prefixes)")
    axes.set_ylabel("selected-program success")
    axes.set_ylim(-0.02, 1.02)
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)
    axes.set_title("Actual success of the selected program vs candidate count")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def summarise(report_data: dict) -> str:
    lines = []
    prefixes = report_data["overall"]["prefixes"]
    names = report_data["all_selectors"]
    lines.append(f"{'N':>4}  {'pool_has_success':>16}  " + "  ".join(f"{name:>20}" for name in names))
    for size in sorted(int(key) for key in prefixes):
        row = prefixes[str(size)]
        cells = "  ".join(f"{row['selectors'][name]['selected_success']:>20.3f}" for name in names)
        lines.append(f"{size:>4}  {row['pool_has_success']:>16.3f}  {cells}")
    lines.append("")
    lines.append(report_data["verdict"]["statement"])
    return "\n".join(lines)


# --------------------------------------------------------------------------- cli


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--counterfactual", type=Path, required=True,
                    help="the #23 artifact whose pools and outcomes this run scores against")
    ap.add_argument("--pools", type=Path, default=POOL_ROOT)
    ap.add_argument("--out", type=Path, default=Path("results/programs"))
    ap.add_argument("--checkpoint", type=Path, default=Path("models/multiblock_world_model.pt"))
    ap.add_argument("--encoder", type=Path, default=Path("models/vjepa2-vitl-fpc64-256"))
    ap.add_argument("--weights", type=Path, help="heuristic weights fitted on train/calibration")
    ap.add_argument("--model", default=sel.MODEL, help="model behind the Claude self-rank arm")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--cache", type=Path, default=Path("data/selector_cache"),
                    help="where the self-rank replies are cached so a re-run reproduces the artifact")
    ap.add_argument("--window-frames", type=int, default=8,
                    help="camera frames handed to the visual arm; must match its checkpoint")
    ap.add_argument("--no-claude", action="store_true", help="skip the self-rank arm (no CLI calls)")
    ap.add_argument("--no-visual", action="store_true", help="skip the visual arm (no checkpoint)")
    ap.add_argument("--allow-orientation-blind", action="store_true",
                    help="run a checkpoint whose plan encoding cannot separate grasp yaws; the "
                         "limitation is recorded in the arm's config and travels with the result")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="rank from tracked edits; the artifact will fail validation, by design")
    args = ap.parse_args()

    if git_dirty() and not args.allow_dirty:
        raise SystemExit("refusing to rank from a dirty tracked worktree; commit the locked code "
                         "or pass --allow-dirty for an explicitly exploratory run")

    artifact = json.loads(args.counterfactual.read_text())
    pools = load_pools(args.pools, artifact)
    views = load_views(args.counterfactual, pools)
    arms = build_selectors(args)
    print(f"{len(pools)} pool(s), {len(artifact['scenes'])} scenes, "
          f"{len(arms)} selector(s): {', '.join(arm.name for arm in arms)}")

    scored, excluded = run_selectors(artifact, views, pools, arms, args.window_frames)
    problems = check_execution(scored, views)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.counterfactual.stem}-selectors"
    artifact_path = args.out / f"{stem}.json"
    artifact_path.write_text(json.dumps(scored, indent=2, default=float) + "\n")
    print(f"wrote {artifact_path} ({len(scored['scenes'])} scenes, {len(excluded)} newly excluded)")
    for problem in problems:
        print(f"  ! {problem}")
    if problems:
        raise SystemExit(1)

    report_data = report(scored, pools, [arm.name for arm in arms])
    report_data["artifact"] = str(artifact_path)
    report_path = args.out / f"{stem}-report.json"
    report_path.write_text(json.dumps(report_data, indent=2, default=float) + "\n")
    plot_path = args.out / f"{stem}.png"
    plot(report_data, plot_path)
    print(summarise(report_data))
    print(f"\nwrote {report_path} and {plot_path}")


if __name__ == "__main__":
    main()
