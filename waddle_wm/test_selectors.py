"""Checks for the #18 selectors and the benchmark that compares them.

    uv run python -m waddle_wm.test_selectors                 # offline: contract, boundary, report
    uv run python -m waddle_wm.test_selectors --live-visual 1 # + encode a real window and score it

What is tested here is the part #18 owns: that a selector cannot mutate, repair, add, or drop
a candidate; that each arm reads only what it declares; that the geometry heuristic is a real
baseline rather than a strawman; that a prefix of N is scored on N candidates; and that the
report refuses to claim visual value the paired numbers do not support.

The artifact schema, the oracle, and the tie-break belong to `benchmark_record` (#24), and the
execution records to `counterfactual` (#23); both have their own suites.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy

from waddle_wm import benchmark_record as br
from waddle_wm import benchmark_selectors as bs
from waddle_wm import counterfactual as cf
from waddle_wm import program as prog
from waddle_wm import selectors as sel
from waddle_wm.test_counterfactual import BUDGET, offline_pool, outcome
from waddle_wm.test_program import SCENE

DETECTIONS = [{"label": "red block", "box": [100, 110, 130, 140], "pixels": 900, "score": 1.0,
               "point_base": [0.3812, -0.1804, 0.018], "depth_m": 0.9, "size_m": 0.036},
              {"label": "blue block", "box": [150, 110, 180, 140], "pixels": 900, "score": 1.0,
               "point_base": [0.5031, -0.1612, 0.018], "depth_m": 0.9, "size_m": 0.036}]


def pool() -> dict:
    """The scripted diagnostic pool, on the offline scene, with perception estimates attached."""
    built = deepcopy(offline_pool())
    built["scene"]["detections"] = deepcopy(DETECTIONS)
    built["scene"]["landing_pad"] = {"centre": SCENE.points["green pad"][:2], "radius": SCENE.pad_radius}
    return built


def view() -> dict:
    return cf.selector_view(pool())


def named(source: dict | None = None) -> dict:
    """candidate id -> the diagnostic's name. Read from the *pool*: the selector view withholds
    it, because which planted fault a candidate is would be half the answer."""
    return {candidate["candidate_id"]: candidate["diagnostic"]
            for candidate in (source or pool())["candidates"]}


def all_ids(context: sel.ScenarioContext) -> list[str]:
    return [candidate["candidate_id"] for candidate in context.view["candidates"]]


# --------------------------------------------------------------------------- the contract


class Mutating(sel.EstimatedStateHeuristic):
    name = "mutating"

    def score(self, context, prefix):
        context.view["candidates"][0]["program"]["ops"][0]["query"] = "blue block"
        return super().score(context, prefix)


class Dropping(sel.EstimatedStateHeuristic):
    name = "dropping"

    def score(self, context, prefix):
        return super().score(context, prefix)[:-1]


class Inventing(sel.EstimatedStateHeuristic):
    name = "inventing"

    def score(self, context, prefix):
        rows = super().score(context, prefix)
        return [*rows, {"candidate_id": "repaired", "score": 9.0, "probability": 1.0,
                        "uncertainty": None, "rank": 0}]


def check_contract():
    context = sel.ScenarioContext(view())
    prefix = all_ids(context)[:4]
    for broken in (Mutating(), Dropping(), Inventing()):
        try:
            sel.rank(broken, context, prefix)
        except sel.SelectorError as error:
            print(f"  refused {broken.name}: {str(error)[:70]}")
        else:
            raise AssertionError(f"{broken.name} was allowed to break the ranking contract")

    # The frozen pool survives every one of those attempts.
    assert sel.fingerprint(context.view) == sel.fingerprint(view()), "the pool was mutated"

    heuristic = sel.EstimatedStateHeuristic()
    with_frames = sel.ScenarioContext(view(), frames=[[[0]]])
    try:
        sel.rank(heuristic, with_frames, prefix)
    except sel.SelectorError as error:
        print(f"  refused frames for {heuristic.name}: {str(error)[:70]}")
    else:
        raise AssertionError("the coordinate arm was handed raw frames")

    block = sel.rank(heuristic, context, prefix)
    assert sorted(row["candidate_id"] for row in block["scores"]) == sorted(prefix)
    assert block["chosen"] == br.selector_choice(block["scores"], prefix), block["chosen"]
    assert block["timing"]["selector_seconds"] >= 0
    assert not br._leaked_keys(block), br._leaked_keys(block)
    for source in block["information_sources"]:
        assert source in br.INFORMATION_SOURCES, source
        assert source not in br.FORBIDDEN_SOURCES, source
    print("contract: mutation, dropping, inventing, and unearned frames are all refused")


def check_prefix_faithful():
    """A selector at N sees N candidates, and the nested prefixes agree where they overlap."""
    context = sel.ScenarioContext(view())
    ids = all_ids(context)
    scores = {}
    for size in (1, 4, 16):
        block = sel.rank(sel.EstimatedStateHeuristic(), context, ids[:size])
        assert len(block["scores"]) == size, (size, len(block["scores"]))
        for row in block["scores"]:
            previous = scores.setdefault(row["candidate_id"], row["score"])
            assert abs(previous - row["score"]) < 1e-12, (row, previous)
    print(f"prefixes: 1/4/16 scored {len(scores)} candidates, unchanged where they overlap")


# --------------------------------------------------------------------------- the heuristic


def check_heuristic_is_a_baseline():
    """The coordinate arm has to actually rank geometry, or the comparison is a strawman."""
    context = sel.ScenarioContext(view())
    block = sel.rank(sel.EstimatedStateHeuristic(), context, all_ids(context))
    labels = named()
    scored = {labels[row["candidate_id"]]: row["score"] for row in block["scores"]}

    for fault in ("bad_grasp", "wrong_target", "early_release", "high_release", "stale_coordinates"):
        assert scored[fault] < scored["correct"], (fault, scored[fault], scored["correct"])
    assert scored["missing_lift"] < scored["correct"], scored
    assert scored["abort_on_uncertainty"] == sel.DECLINED_PROBABILITY, scored["abort_on_uncertainty"]
    assert scored["redetect_regrasp"] >= scored["correct"], scored
    assert scored["correct"] > 0.5, scored["correct"]
    print("heuristic: every planted fault ranks below the canonical program "
          f"({scored['correct']:.3f} vs worst {min(scored.values()):.3f})")


def check_features():
    context = sel.ScenarioContext(view())
    estimates, task = context.estimates(), context.view["task"]
    labels = named()
    by_name = {labels[candidate["candidate_id"]]: candidate for candidate in context.view["candidates"]}

    correct = sel.features(by_name["correct"], estimates, task)
    assert correct["grasp_offset_mm"] < 2.0, correct
    assert correct["malformed"] == 0.0 and correct["declined"] == 0.0, correct
    assert correct["place_margin_mm"] == 0.0, correct

    bad = sel.features(by_name["bad_grasp"], estimates, task)
    assert bad["grasp_offset_mm"] > correct["grasp_offset_mm"], (bad, correct)
    wrong = sel.features(by_name["wrong_target"], estimates, task)
    assert wrong["place_margin_mm"] > 0.0, wrong
    declined = sel.features(by_name["abort_on_uncertainty"], estimates, task)
    assert declined["declined"] == 1.0 and declined["grasp_offset_mm"] == 0.0, declined

    # Every feature is finite and reads only estimated coordinates: perturbing the *hidden*
    # truth in the pool cannot move a single number, because it never enters the view.
    assert set(correct) == set(sel.FEATURES), set(correct) ^ set(sel.FEATURES)
    print(f"features: {len(sel.FEATURES)} coordinate-derived features, "
          f"grasp offset {correct['grasp_offset_mm']:.1f} mm on the canonical program")


def check_fit_refuses_test():
    artifact = {"metadata": {"dataset": {"split": "test"}}, "scenes": []}
    try:
        sel.fit_heuristic([artifact], {})
    except sel.SelectorError as error:
        print(f"  refused fitting on test: {str(error)[:70]}")
    else:
        raise AssertionError("the heuristic was fitted on the locked split")


# --------------------------------------------------------------------------- self-rank


def check_self_rank_parsing():
    labels = ["program_01", "program_02", "program_03"]
    ranking, scores = sel.parse_ranking(
        '```json\n{"ranking": ["program_02", "program_01", "program_03"], '
        '"scores": {"program_02": 0.9, "program_01": 0.4, "program_03": 1.7}}\n```', labels)
    assert ranking == ["program_02", "program_01", "program_03"], ranking
    assert scores["program_03"] == 1.0, scores          # clamped, not trusted

    ranking, scores = sel.parse_ranking('{"ranking": ["program_02", "nope", "program_02"]}', labels)
    assert ranking == ["program_02"], ranking           # unknown and repeated labels dropped
    assert scores == {"program_02": 1.0}, scores

    assert sel.parse_ranking("I would rank the second one first.", labels) == ([], {})

    context = sel.ScenarioContext(view())
    arm = sel.ClaudeSelfRank()
    first = arm._labels(context, all_ids(context)[:4])
    again = arm._labels(context, all_ids(context)[:4])
    assert [label for label, _ in first] == [label for label, _ in again], "shuffle is not seeded"
    assert [candidate["candidate_id"] for _, candidate in first] != all_ids(context)[:4], \
        "candidates were presented in pool order; the ordering itself is a hint"
    prompt = sel.rank_prompt(context.view["task"], context.view["scene"]["observation"],
                             [(label, candidate["program"]) for label, candidate in first])
    for candidate_id in all_ids(context)[:4]:
        assert candidate_id not in prompt, "the prompt carries pool identity"
    for name in ("hidden_truth", "success", "0.3801"):
        assert name not in prompt, f"the prompt leaks {name}"
    print("self-rank: replies parse defensively, candidates are anonymised and seed-shuffled")


def check_self_rank_scoring():
    """A recorded reply is turned into scores without the arm ever seeing an outcome."""
    context = sel.ScenarioContext(view())
    prefix = all_ids(context)[:4]
    arm = sel.ClaudeSelfRank()
    labelled = arm._labels(context, prefix)
    reply = {"ranking": [label for label, _ in labelled][:3],
             "scores": {labelled[0][0]: 0.9, labelled[1][0]: 0.2}}
    arm._ask = lambda prompt, key: {"raw": json.dumps(reply), "error": None, "call": {}}

    block = sel.rank(arm, context, prefix)
    rows = {row["candidate_id"]: row for row in block["scores"]}
    assert rows[labelled[0][1]["candidate_id"]]["score"] == 0.9
    unranked = rows[labelled[3][1]["candidate_id"]]
    assert unranked["score"] == 0.0 and unranked["probability"] is None, unranked
    assert block["chosen"]["candidate_id"] == labelled[0][1]["candidate_id"], block["chosen"]
    print("self-rank: an unmentioned candidate scores the floor rather than being dropped")


# --------------------------------------------------------------------------- the benchmark


def scored_artifact(success_by_name: dict, slices: dict) -> tuple[dict, dict, dict]:
    """A #23 artifact over two scenes, with the three arms folded in, built without MuJoCo."""
    pools, views, scenes = {}, {}, []
    for index, (scenario, (outcome_slice, observability)) in enumerate(sorted(slices.items())):
        pool = deepcopy(offline_pool())
        pool["pool_id"] = f"pool-{scenario}"
        pool["split"] = "test"
        pool["scene"]["seed"] = 100 + index
        pool["scene"]["detections"] = deepcopy(DETECTIONS)
        pool["scene"]["landing_pad"] = {"centre": SCENE.points["green pad"][:2],
                                        "radius": SCENE.pad_radius}
        pool["scene"]["suite"] = {"scenario_id": scenario, "outcome_slice": outcome_slice,
                                  "observability": observability}
        pools[pool["pool_id"]] = pool
        views[pool["pool_id"]] = cf.selector_view(pool)

        ids = [candidate["candidate_id"] for candidate in pool["candidates"]]
        labels = {candidate["candidate_id"]: candidate["diagnostic"] for candidate in pool["candidates"]}
        outcomes = {cid: outcome(position, cid, success=success_by_name[scenario](labels[cid]),
                                 final_target_error_mm=10.0 if success_by_name[scenario](labels[cid]) else 200.0)
                    for position, cid in enumerate(ids)}
        for size in (1, 4, 16):
            scenes.append(br.SceneRun(
                scenario_id=f"sc{index}", split="test", scene_seed=100 + index, physics_seed=0,
                pool_id=pool["pool_id"], pool_kind="diagnostic",
                observation_id=pool["scene"]["observation_id"], prefix=size, pool_prefix=ids[:size],
                counterfactual={cid: outcomes[cid] for cid in ids[:size]},
                selectors={}, execution_budget=BUDGET).as_json())

    metadata = br.run_metadata(
        split="test", scene_seeds=[100 + i for i in range(len(slices))], physics_seeds=[0],
        pools={pool_id: {"kind": "diagnostic"} for pool_id in pools}, generator={"kind": "diagnostic"},
        perception={"camera": "demo"}, physics={"perturbation_mm": 0.0}, selectors={},
        git_sha="abc", git_dirty=False, created_at="2026-01-01T00:00:00")
    artifact = {"artifact_version": br.ARTIFACT_VERSION, "metadata": metadata, "scenes": scenes,
                "excluded": [], "kind": "diagnostic", "preflight": {}, "execution": {}}
    return artifact, views, pools


class Constant(sel.Selector):
    """A stand-in for an arm whose model is not loaded here: fixed, declared, and read-only."""

    def __init__(self, name, sources, table, needs_frames=False):
        self.name, self.information_sources, self.table = name, sources, table
        self.needs_frames = needs_frames
        self.labels = named()

    def score(self, context, prefix):
        return [{"candidate_id": cid, "score": self.table(self.labels[cid]),
                 "probability": self.table(self.labels[cid]), "uncertainty": 0.1, "rank": None}
                for cid in prefix]


def check_benchmark_end_to_end():
    """Three arms over two slices: the artifact validates and the report is paired by scene."""
    # On the plan-visible scene the planted faults fail; on the scene-dependent one the grasp
    # geometry is fine but the object is rotated, so only the arm that reads the frame is right.
    plan_visible = lambda name: name not in prog.FAULTS
    scene_dependent = lambda name: name in ("orientation_grasp_neg45", "orientation_grasp_pos45")
    scenarios = {"plan": ("obvious_target_miss", "plan_visible_control"),
                 **{f"scene{index}": ("block_orientation", "visible_omitted_by_coordinates")
                    for index in range(4)}}
    artifact, views, pools = scored_artifact(
        {name: plan_visible if name == "plan" else scene_dependent for name in scenarios}, scenarios)

    arms = [sel.EstimatedStateHeuristic(),
            Constant("visual_world_model", sel.VisualWorldModel.information_sources,
                     lambda name: 0.9 if scene_dependent(name) else 0.3, needs_frames=True),
            Constant("claude_self_rank", sel.ClaudeSelfRank.information_sources,
                     lambda name: 0.5)]

    class Frames(dict):
        pass

    original = bs.observation_window
    bs.observation_window = lambda pool, frames: [[0]]
    try:
        scored, excluded = bs.run_selectors(artifact, views, pools, arms, 8)
    finally:
        bs.observation_window = original
    assert not excluded, excluded

    problems = br.check_run(scored)
    assert not problems, problems
    names = sorted(scored["metadata"]["selectors"])
    assert names == ["claude_self_rank", "estimated_state", "visual_world_model"], names

    report = bs.report(scored, pools, [arm.name for arm in arms])
    prefixes = report["overall"]["prefixes"]
    assert set(prefixes) == {"1", "4", "16"}, sorted(prefixes)
    for size in prefixes.values():
        for name in names:
            row = size["selectors"][name]
            for metric in ("selected_success", "oracle_gap", "selector_seconds", "brier"):
                assert metric in row, metric

    intended = report["by_slice"]["block_orientation"]
    heuristic = intended["prefixes"]["16"]["selectors"]["estimated_state"]["selected_success"]
    visual = intended["prefixes"]["16"]["selectors"]["visual_world_model"]["selected_success"]
    assert visual > heuristic, (visual, heuristic)
    assert report["verdict"]["answer"] == "yes", report["verdict"]
    assert report["verdict"]["paired_differences"]["selected_success"]["ci95"][0] > 0, report["verdict"]
    assert "visible_omitted_by_coordinates" in report["verdict"]["statement"]
    assert set(report["by_observability"]) == {"plan_visible", "scene_dependent"}, report["by_observability"]
    print(f"benchmark: artifact validates, {len(scored['scenes'])} scenes, "
          f"verdict={report['verdict']['answer']} on the intended slice")


def check_no_unsupported_claim():
    """When the visual arm does not win its slice, the report says so and claims nothing."""
    same = lambda name: name not in prog.FAULTS
    artifact, views, pools = scored_artifact(
        {"scene": same}, {"scene": ("block_orientation", "visible_omitted_by_coordinates")})
    arms = [sel.EstimatedStateHeuristic(),
            Constant("visual_world_model", sel.VisualWorldModel.information_sources,
                     lambda name: 0.1 if name == "correct" else 0.9, needs_frames=True)]
    original = bs.observation_window
    bs.observation_window = lambda pool, frames: [[0]]
    try:
        scored, _ = bs.run_selectors(artifact, views, pools, arms, 8)
    finally:
        bs.observation_window = original
    report = bs.report(scored, pools, [arm.name for arm in arms])
    assert report["verdict"]["answer"] == "no", report["verdict"]
    assert "does not support" in report["verdict"]["statement"], report["verdict"]["statement"]
    print("verdict: a losing visual arm produces no claim of visual value")


# --------------------------------------------------------------------------- live


def check_live_visual(scenes: int, checkpoint, encoder):
    """Encode a real observation window and score a real pool with the visual arm."""
    from waddle_wm.pools import SPLITS, Scene

    arm = sel.VisualWorldModel(checkpoint, encoder)
    for seed in list(SPLITS["test"])[:scenes]:
        scene_obj = Scene(seed)
        try:
            pool = {"pool_id": f"live-{seed}", "kind": "diagnostic", "split": "test",
                    "task": {"instruction": "", "object": "red block", "destination": "green pad"},
                    "scene": {"seed": seed, "observation_id": scene_obj.observation.observation_id,
                              "observation": scene_obj.observation.text,
                              "detections": scene_obj.observation.detections,
                              "landing_pad": {"centre": scene_obj.observation.points["green pad"][:2],
                                              "radius": scene_obj.observation.pad_radius}},
                    "prefixes": {}, "candidates": []}
            for index, (name, kind, program) in enumerate(prog.diagnostic_programs()):
                grounded = prog.ground(program, scene_obj.observation)
                pool["candidates"].append(
                    {"candidate_id": f"c{index:02d}", "index": index, "sample_index": index,
                     "program": program.as_json(),
                     "grounded_trace": grounded.step.summary()["trace"] if grounded.step else [],
                     "dedup_key": grounded.dedup_key(), "duplicate_of": None,
                     "validation": {}, "retry": program.retry, "redetect_ops": program.redetects,
                     "aborts": program.aborts, "diagnostic": name, "strategy": program.strategy,
                     "note": program.note, "generation": {}, "raw": ""})
            frames = scene_obj.env.observation_frames(arm.manifest["window_frames"])
        finally:
            scene_obj.close()
        context = sel.ScenarioContext(cf.selector_view(pool), frames)
        ids = [candidate["candidate_id"] for candidate in pool["candidates"]]
        block = sel.rank(arm, context, ids)
        labels = {candidate["candidate_id"]: candidate["diagnostic"] for candidate in pool["candidates"]}
        ordered = sorted(block["scores"], key=lambda row: -row["score"])
        print(f"  seed {seed}: {block['timing']['selector_seconds']:.1f}s, best "
              f"{labels[ordered[0]['candidate_id']]} ({ordered[0]['score']:.3f}), worst "
              f"{labels[ordered[-1]['candidate_id']]} ({ordered[-1]['score']:.3f})")
        assert all(0.0 <= row["probability"] <= 1.0 for row in block["scores"])
        # The guard on the checkpoint's degenerate normalisation, as a regression test: without
        # it a millimetre of perception noise in a training-constant feature saturates every
        # candidate to the same p=0 and the arm ranks nothing.
        spread = {round(row["probability"], 3) for row in block["scores"]}
        assert len(spread) > 3, f"the visual arm returned {len(spread)} distinct scores: {spread}"
        assert labels[ordered[-1]["candidate_id"]] != "correct", "the canonical program ranked last"
    print(f"live visual arm scored {scenes} real scene(s)")


def check_yaw_aware(checkpoint, embeddings):
    """Same estimated scene and XYZ plan, yaw-only change -> distinct rows and scores."""
    import math
    import numpy as np
    import torch

    arm = sel.VisualWorldModel(checkpoint, device=torch.device("cpu"))
    selector_view, labels = view(), named()
    base = next(candidate for candidate in selector_view["candidates"]
                if labels[candidate["candidate_id"]] == "correct")
    candidates = []
    for yaw_deg in (0.0, 45.0):
        candidate = deepcopy(base)
        candidate["candidate_id"] = f"yaw-{yaw_deg:g}"
        for entry in candidate["grounded_trace"]:
            if entry["phase"] in ("approach", "descend", "lift"):
                entry["yaw"] = math.radians(yaw_deg)
        candidates.append(candidate)
    selector_view["candidates"] = candidates

    cache = torch.load(embeddings, weights_only=False, map_location="cpu")
    latent = arm._normalise(next(iter(cache.values()))[0].unsqueeze(0), "context")
    context = sel.ScenarioContext(selector_view, np.zeros((8, 1, 1, 3), dtype=np.uint8), latent)
    estimates, state = context.estimates(), arm._state(context.estimates())
    plan_rows = [arm.plan_row(candidate, estimates, selector_view["task"], state)
                 for candidate in candidates]
    assert not np.array_equal(*plan_rows), plan_rows
    scores = sel.rank(arm, context, [candidate["candidate_id"] for candidate in candidates])["scores"]
    probabilities = [row["probability"] for row in scores]
    assert probabilities[0] != probabilities[1], probabilities
    print(f"yaw-aware checkpoint: yaw-only rows and scores differ ({probabilities[0]:.4f} vs {probabilities[1]:.4f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live-visual", type=int, default=0)
    ap.add_argument("--checkpoint", default="models/multiblock_world_model.pt")
    ap.add_argument("--encoder", default="models/vjepa2-vitl-fpc64-256")
    ap.add_argument("--yaw-aware-checkpoint")
    ap.add_argument("--embeddings", default="data/ur5e_wm_oriented/window_embeddings.pt")
    args = ap.parse_args()

    check_contract()
    check_prefix_faithful()
    check_features()
    check_heuristic_is_a_baseline()
    check_fit_refuses_test()
    check_self_rank_parsing()
    check_self_rank_scoring()
    check_benchmark_end_to_end()
    check_no_unsupported_claim()
    if args.live_visual:
        check_live_visual(args.live_visual, args.checkpoint, args.encoder)
    if args.yaw_aware_checkpoint:
        check_yaw_aware(args.yaw_aware_checkpoint, args.embeddings)


if __name__ == "__main__":
    main()
