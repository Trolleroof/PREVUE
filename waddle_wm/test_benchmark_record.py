"""Negative fixtures for the benchmark artifact: every integrity check must fail loudly.

A validator nobody has watched reject anything is a validator that passes everything. So
this builds one artifact that is correct in every respect, proves `check_run` accepts it,
and then breaks it one way at a time — a dropped execution, a prefix that is not nested, a
budget that differs between arms, a test run reusing training seeds, a MuJoCo field inside a
selector block, a timing window that swallowed the simulator — and requires the matching
complaint each time.

It also pins the two orderings the benchmark's answers hang on: the oracle's, and the
selector tie-break's. Neither may drift silently.

    uv run python -m waddle_wm.test_benchmark_record

Offline: no MuJoCo rollout, no Claude, no checkpoint.
"""
from __future__ import annotations

from copy import deepcopy

from waddle_wm.benchmark_record import (
    ARTIFACT_VERSION, NotComparable, PRIMARY_OUTCOMES, SceneRun, SelectorRun, Timing,
    aggregate, check_run, definition_hash, oracle_best, run_metadata, scene_metrics,
    selector_choice,
)

SELECTORS = ("heuristic", "world-model")


def outcome(success, attempts, error_mm, seconds, order):
    return {"success": success, "failure_mode": None if success else "target_miss",
            "max_lift_mm": 180.0 if success else 20.0, "final_target_error_mm": error_mm,
            "failed_attempts": attempts, "timed_out": False, "error": None,
            "execution_seconds": seconds, "execution_order": order}


# Four candidates per scene. c1 is the only success, so a selector that takes c0 has missed an
# available success and a selector that takes c1 matches the oracle exactly.
def scene_outcomes(seed: int) -> dict:
    return {f"{seed}-c0": outcome(False, 0, 148.0, 6.1, 2),
            f"{seed}-c1": outcome(True, 0, 12.0, 5.9, 0),
            f"{seed}-c2": outcome(False, 1, 91.0, 7.4, 3),
            f"{seed}-c3": outcome(False, 0, 210.0, 6.0, 1)}


def selector_block(name: str, prefix: list[str], picks: str) -> dict:
    """One selector's scores over a prefix, peaked on `picks`."""
    scores = [{"candidate_id": cid, "score": round(0.9 if cid == picks else 0.3 - 0.01 * index, 4),
               "probability": round(0.9 if cid == picks else 0.3 - 0.01 * index, 4),
               "uncertainty": 0.08, "rank": index} for index, cid in enumerate(prefix)]
    timing = Timing(1000.0, 1000.25, {"perception": 0.10, "scoring": 0.12, "tie_break": 0.001})
    return SelectorRun(name, f"{name}-cfg-a1b2c3",
                       ["observation_text", "visual_model_latents"] if name == "world-model"
                       else ["observation_text", "heuristic_image_estimate"],
                       scores, selector_choice(scores, prefix), timing.as_json(),
                       cost_usd=0.0 if name == "heuristic" else 0.002).as_json()


def valid_run() -> dict:
    """Two held-out scenes, two nested prefixes, two selectors, everything recorded."""
    seeds = [100, 101]
    scenes = []
    for seed in seeds:
        outcomes = scene_outcomes(seed)
        order = [f"{seed}-c0", f"{seed}-c1", f"{seed}-c2", f"{seed}-c3"]
        for size in (1, 4):
            prefix = order[:size]
            # The heuristic takes the first candidate; the world model takes c1 where it can
            # see it. At prefix 1 there is nothing to disagree about.
            picks = {"heuristic": prefix[0],
                     "world-model": f"{seed}-c1" if size > 1 else prefix[0]}
            scenes.append(SceneRun(
                scenario_id=f"test-seed{seed:04d}-red_block-to-green_pad", split="test",
                scene_seed=seed, physics_seed=0,
                pool_id=f"natural-red_block-to-green_pad-seed{seed:04d}-deadbeef1234",
                pool_kind="natural", observation_id=f"obs-{seed}", prefix=size,
                pool_prefix=prefix,
                counterfactual={cid: outcomes[cid] for cid in prefix},
                selectors={name: selector_block(name, prefix, picks[name]) for name in SELECTORS},
                execution_budget={"max_attempts": 2, "frames_total": 480, "wall_clock_s": 90},
                claude_generation_seconds=41.2, mujoco_execution_seconds=25.4).as_json())

    metadata = run_metadata(
        split="test", scene_seeds=seeds, physics_seeds=[0],
        pools={"generator_hash": "deadbeef1234", "pool_size": 64, "prefixes": [1, 4],
               "pool_ids": sorted({scene["pool_id"] for scene in scenes}),
               "order": "sample order, as cached"},
        generator={"model": "claude-opus-5", "system_prompt_sha1": "aa11bb22cc33",
                   "sampling": {"temperature": "claude CLI default"}, "max_turns": 1},
        perception={"camera": "demo", "width": 256, "height": 256,
                    "queries": ["red block", "blue block", "yellow block", "green pad"],
                    "detector_sha1": "0f0f0f0f0f0f"},
        physics={"controller": "damped least squares IK", "timestep": 0.002, "frames_total": 480,
                 "snapshot": "restored before every candidate"},
        selectors={"heuristic": {"kind": "image_heuristic", "accept_threshold": 0.5,
                                 "checkpoint": None, "checkpoint_sha256": None},
                   "world-model": {"kind": "latent_dynamics", "accept_threshold": 0.5,
                                   "checkpoint": "models/latent_dynamics_wide.pt",
                                   "checkpoint_sha256": "9c" * 32}},
        git_sha="0" * 40, git_dirty=False, created_at="2026-08-11T09:00:00")
    return {"artifact_version": ARTIFACT_VERSION, "run_id": "programs-test-0001",
            "metadata": metadata, "scenes": scenes,
            "excluded": [{"scenario_id": "test-seed0102-red_block-to-green_pad",
                          "reason": "pool short of 64 candidates within the attempt budget"}]}


# --------------------------------------------------------------------------- the orderings


def check_oracle():
    prefix = ["a", "b", "c", "d"]

    def outcomes(**edits):
        rows = {"a": outcome(False, 0, 100.0, 6.0, 0), "b": outcome(True, 0, 20.0, 6.0, 1),
                "c": outcome(True, 0, 20.0, 6.0, 2), "d": outcome(True, 1, 5.0, 5.0, 3)}
        for cid, edit in edits.items():
            rows[cid].update(edit)
        return rows

    def best(**edits):
        return oracle_best(prefix, outcomes(**edits))

    # a fails and loses to every success, however far the successes land from the pad.
    assert best()["candidate_id"] == "b", best()
    assert best()["ranking"][-1] == "a" and best()["pool_has_success"] is True
    # d lands 15 mm closer but needed a retry to get there, so it ranks behind b and c.
    assert best()["ranking"] == ["b", "c", "d", "a"], best()["ranking"]
    assert oracle_best(["a", "b", "d"], outcomes())["decided_by"] == "failed_attempts"
    # b and c differ on nothing an outcome records, so the winner is decided by pool order.
    assert best()["decided_by"] == "pool index", best()["decided_by"]
    assert best()["tied_with"] == ["c"], best()["tied_with"]
    # b and c are identical on every outcome key, so the earlier pool index decides.
    assert best()["ranking"][:2] == ["b", "c"]
    # ... and it still decides when c is scored first in the pool.
    flipped = oracle_best(["c", "b", "a", "d"],
                          {"a": outcome(False, 0, 100.0, 6.0, 0), "b": outcome(True, 0, 20.0, 6.0, 1),
                           "c": outcome(True, 0, 20.0, 6.0, 2), "d": outcome(True, 1, 5.0, 5.0, 3)})
    assert flipped["candidate_id"] == "c", flipped["candidate_id"]

    # Inside one 0.5 mm bucket the two are the same candidate as far as the oracle is
    # concerned, so pool order still decides; a bucket lower and the closer one wins.
    assert best(c={"final_target_error_mm": 20.4})["candidate_id"] == "b"
    assert best(c={"final_target_error_mm": 19.4})["candidate_id"] == "c"
    assert best(c={"final_target_error_mm": 19.4})["decided_by"] == "final_target_error_mm"
    # Execution time only speaks once error ties, and only across a 50 ms bucket.
    assert best(c={"execution_seconds": 6.04})["candidate_id"] == "b"
    assert best(c={"execution_seconds": 5.90})["candidate_id"] == "c"
    assert best(c={"execution_seconds": 5.90})["decided_by"] == "execution_seconds"
    # No success anywhere is candidate-generation coverage, and the ordering still answers.
    none = best(b={"success": False}, c={"success": False}, d={"success": False})
    assert none["pool_has_success"] is False, none
    assert none["candidate_id"] == "b" and none["ranking"][-1] == "d", none
    assert oracle_best(["a"], {"a": outcome(False, 0, 100.0, 6.0, 0)})["decided_by"] == "only candidate"
    print("oracle ordering ok")


def check_selector_tie_break():
    prefix = ["a", "b", "c"]
    scores = [{"candidate_id": "a", "score": 0.4}, {"candidate_id": "b", "score": 0.8},
              {"candidate_id": "c", "score": 0.8}]
    chosen = selector_choice(scores, prefix)
    assert chosen["candidate_id"] == "b" and chosen["tie_break"] == "pool_index", chosen
    assert chosen["tied_with"] == ["c"] and chosen["score_margin"] == 0.0, chosen
    # The tie-break is the pool's order, not the order the scores happen to be listed in.
    assert selector_choice(list(reversed(scores)), prefix)["candidate_id"] == "b"
    clear = selector_choice([{"candidate_id": "a", "score": 0.9}, {"candidate_id": "b", "score": 0.2}],
                            prefix)
    assert clear["tie_break"] == "unique" and clear["score_margin"] == 0.7, clear
    assert selector_choice([{"candidate_id": "a", "score": 0.1}], prefix)["score_margin"] is None
    print("selector tie-break ok")


# --------------------------------------------------------------------------- negative fixtures


def drop_execution(run):
    run["scenes"][1]["counterfactual"].pop("100-c3")


def duplicate_candidate(run):
    run["scenes"][1]["pool_prefix"][2] = "100-c0"


def execution_outside_prefix(run):
    run["scenes"][1]["counterfactual"]["100-c9"] = outcome(False, 0, 50.0, 6.0, 9)


def duplicate_execution_order(run):
    run["scenes"][1]["counterfactual"]["100-c3"]["execution_order"] = 0


def partial_outcome(run):
    run["scenes"][1]["counterfactual"]["100-c2"].pop("failed_attempts")


def break_nesting(run):
    prefix = run["scenes"][1]["pool_prefix"]
    prefix[0], prefix[1] = prefix[1], prefix[0]
    counterfactual = run["scenes"][1]["counterfactual"]
    for block in run["scenes"][1]["selectors"].values():
        block["chosen"] = selector_choice(block["scores"], prefix)
    run["scenes"][1]["oracle"] = oracle_best(prefix, counterfactual)


def unequal_budgets(run):
    run["scenes"][2]["execution_budget"]["max_attempts"] = 4


def train_test_overlap(run):
    run["metadata"]["dataset"]["scene_seeds"] = [7, 100, 101]
    run["scenes"][0]["scene_seed"] = 7


def mismatched_observation(run):
    run["scenes"][1]["observation_id"] = "obs-somewhere-else"


def mismatched_protocol(run):
    run["metadata"]["protocol_version"] = 99


def changed_definitions(run):
    run["metadata"]["success_definition"]["pad_radius_m"] = 0.3


def leaked_mujoco_field(run):
    run["scenes"][1]["selectors"]["world-model"]["scores"][0]["target_distance"] = 0.02


def leaked_scene_physics(run):
    run["scenes"][1]["selectors"]["world-model"]["scores"][0]["red_block_mass_kg"] = 0.065


def leaked_oracle(run):
    run["scenes"][1]["selectors"]["world-model"]["oracle_gap"] = 0


def forbidden_source(run):
    run["scenes"][1]["selectors"]["world-model"]["information_sources"].append("counterfactual_outcome")


def missing_information_boundary(run):
    run["scenes"][1]["selectors"]["heuristic"]["information_sources"] = []


def incomplete_timing(run):
    run["scenes"][1]["selectors"]["world-model"]["timing"]["components"] = {}


def no_timing(run):
    run["scenes"][1]["selectors"]["heuristic"].pop("timing")


def timing_window_mismatch(run):
    run["scenes"][1]["selectors"]["world-model"]["timing"]["selector_seconds"] = 0.05


def execution_time_inside_selector(run):
    run["scenes"][1]["selectors"]["world-model"]["timing"]["components"]["mujoco_execution_seconds"] = 0.02


def component_overrun(run):
    run["scenes"][1]["selectors"]["heuristic"]["timing"]["components"]["scoring"] = 9.0


def unscored_candidate(run):
    run["scenes"][1]["selectors"]["heuristic"]["scores"].pop()


def double_scored_candidate(run):
    scores = run["scenes"][1]["selectors"]["heuristic"]["scores"]
    scores[-1] = deepcopy(scores[0])


def missing_arm(run):
    run["scenes"][1]["selectors"].pop("world-model")


def outcome_driven_choice(run):
    """The selector's own scores say c0; the artifact claims it chose the successful c1."""
    run["scenes"][1]["selectors"]["heuristic"]["chosen"]["candidate_id"] = "100-c1"


def wrong_oracle(run):
    run["scenes"][1]["oracle"]["candidate_id"] = "100-c0"


def wrong_coverage(run):
    run["scenes"][1]["pool_has_success"] = False


def duplicate_scene(run):
    run["scenes"].append(deepcopy(run["scenes"][1]))


def dirty_worktree(run):
    run["metadata"]["git_dirty"] = True


def missing_metadata(run):
    run["metadata"].pop("perception")


def repointed_primaries(run):
    run["metadata"]["primary_outcomes"] = ["brier"]


def no_exclusions(run):
    run.pop("excluded")


NEGATIVE_FIXTURES = (
    ("a candidate was never executed", drop_execution, "no counterfactual execution record"),
    ("the prefix repeats a candidate", duplicate_candidate, "repeats a candidate"),
    ("something was executed outside the pool", execution_outside_prefix, "not in the pool prefix"),
    ("two candidates share an execution order", duplicate_execution_order, "share an execution order"),
    ("an execution record is partial", partial_outcome, "outcome is missing failed_attempts"),
    ("prefix 4 is not prefix 1 extended", break_nesting, "not nested inside"),
    ("the arms ran under different budgets", unequal_budgets, "different execution budgets"),
    ("a training seed is in the test run", train_test_overlap, "train/test overlap"),
    ("one scenario, two observations", mismatched_observation, "different pool or observation"),
    ("the protocol is not this code's", mismatched_protocol, "protocol version 99"),
    ("success was redefined after the run", changed_definitions, "definitions were changed"),
    ("simulator state inside a selector", leaked_mujoco_field, "ground-truth fields inside"),
    ("scene physics inside a selector", leaked_scene_physics, "ground-truth fields inside"),
    ("the answer key inside a selector", leaked_oracle, "ground-truth fields inside"),
    ("a selector declares it saw the oracle", forbidden_source, "never selector inputs"),
    ("no information boundary", missing_information_boundary, "no information boundary"),
    ("component timings dropped", incomplete_timing, "component timings"),
    ("no selector timing at all", no_timing, "no selector timing"),
    ("latency does not match its window", timing_window_mismatch, "does not match"),
    ("MuJoCo time inside the selector window", execution_time_inside_selector, "keep Claude and MuJoCo"),
    ("components outrun the window", component_overrun, "sum past the selector window"),
    ("a candidate was never scored", unscored_candidate, "scored 3 of 4 candidates"),
    ("a candidate was scored twice", double_scored_candidate, "scored twice"),
    ("an arm is missing on one scene", missing_arm, "do not match the declared"),
    ("the choice is not the argmax", outcome_driven_choice, "locked tie-break picks"),
    ("the recorded oracle is not the ordering's", wrong_oracle, "is not the locked ordering"),
    ("coverage disagrees with the executions", wrong_coverage, "pool_has_success disagrees"),
    ("the same cell recorded twice", duplicate_scene, "duplicated at prefix"),
    ("generated from a dirty worktree", dirty_worktree, "dirty worktree"),
    ("metadata is incomplete", missing_metadata, "missing perception"),
    ("the primary outcomes were swapped", repointed_primaries, "predeclared"),
    ("no exclusion list", no_exclusions, "every attempted scene"),
)


def check_fixtures():
    baseline = check_run(valid_run())
    assert not baseline, baseline
    for label, mutate, expected in NEGATIVE_FIXTURES:
        run = valid_run()
        mutate(run)
        problems = check_run(run)
        assert problems, f"{label}: the validator accepted a broken run"
        assert any(expected in problem for problem in problems), \
            f"{label}: wanted {expected!r}, got {problems}"
    print(f"{len(NEGATIVE_FIXTURES)} negative fixtures rejected, the clean run accepted")


# --------------------------------------------------------------------------- metrics


def check_metrics():
    run = valid_run()
    scene = next(s for s in run["scenes"] if s["prefix"] == 4 and s["scene_seed"] == 100)
    heuristic = scene_metrics(scene, "heuristic")
    model = scene_metrics(scene, "world-model")
    assert heuristic["selected_success"] is False and model["selected_success"] is True
    assert heuristic["missed_available_success"] is True
    assert heuristic["selection_efficiency"] == 0.0 and model["selection_efficiency"] == 1.0
    # The oracle ordering ranks the only success first, so the heuristic sits behind it.
    assert model["oracle_gap"] == 0 and heuristic["oracle_gap"] > 0, (model, heuristic)
    assert heuristic["target_error_gap_mm"] == 136.0, heuristic["target_error_gap_mm"]
    # A confident score on a candidate that failed is a false accept, whoever made it.
    assert heuristic["false_accepts"] == 1 and model["false_accepts"] == 0, (heuristic, model)
    assert model["selector_seconds"] == 0.25

    # No success in the pool: the selector cannot have missed one, so efficiency is undefined
    # rather than zero. Averaging a zero in there would blame a selector for the generator.
    lean = next(s for s in run["scenes"] if s["prefix"] == 1 and s["scene_seed"] == 100)
    assert lean["pool_has_success"] is False
    assert scene_metrics(lean, "heuristic")["selection_efficiency"] is None
    assert scene_metrics(lean, "heuristic")["missed_available_success"] is False

    report = aggregate(run)
    assert report["primary_outcomes"] == list(PRIMARY_OUTCOMES)
    at_four = report["prefixes"]["4"]
    assert at_four["selectors"]["world-model"]["selected_success"] == 1.0
    assert at_four["selectors"]["heuristic"]["selected_success"] == 0.0
    paired = report["paired"]["heuristic - world-model"]["4"]
    assert paired["selected_success"]["mean_difference"] == -1.0, paired
    assert paired["selected_success"]["paired_scenes"] == 2
    assert paired["selected_success"]["ci95"] == [-1.0, -1.0], paired
    assert paired["oracle_gap"]["mean_difference"] > 0
    # At prefix 1 both arms see one candidate and neither can differ.
    assert report["paired"]["heuristic - world-model"]["1"]["selected_success"]["mean_difference"] == 0.0
    assert set(report["false_accepts_by_prefix"]["heuristic"]) == {"1", "4"}
    assert report["excluded"] and report["scenes"] == 4

    # Aggregation validates first rather than trusting every caller to remember the gate.
    unpaired = valid_run()
    unpaired["scenes"][1]["selectors"].pop("world-model")
    unequal = valid_run()
    unequal_budgets(unequal)
    shifted = valid_run()
    shifted["metadata"]["definition_hash"] = "0" * 12
    for broken, why in ((unpaired, "arm"), (duplicated_cells(), "cell"),
                        (unequal, "budget"), (shifted, "definition")):
        try:
            aggregate(broken)
        except NotComparable:
            continue
        raise AssertionError(f"aggregate accepted a run with an invalid {why}")
    assert definition_hash() == valid_run()["metadata"]["definition_hash"]
    print("paired metrics ok")


def duplicated_cells() -> dict:
    run = valid_run()
    run["scenes"].append(deepcopy(run["scenes"][0]))
    return run


def main():
    check_oracle()
    check_selector_tie_break()
    check_fixtures()
    check_metrics()
    print("ok")


if __name__ == "__main__":
    main()
