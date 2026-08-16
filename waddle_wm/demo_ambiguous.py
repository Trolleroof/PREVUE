"""Crowded-neighbour scenes: the same plan, the same coordinates, two different physical outcomes.

    uv run python -m waddle_wm.demo_ambiguous --scenes 8 --spacing 0.075 --spacing 0.22 --offset 0.020

`waddle_wm.demo` tests flaws that are fully recoverable from x/y arithmetic (a grasp aimed 6 cm
into empty space, a release 22 cm short of the pad), so the coordinate-only `rules` verifier ties
or beats the learned visual one. Every flaw it has been shown is legible in the plan's own numbers.

This module builds the case that is *not*. It holds the plan fixed — a grasp aimed `--offset`
metres from the red block's detected centre, always inside `SkillAgent.rule_verdict`'s 0.028 m
tolerance — and moves the **blue block** instead. At 0.020 m offset an isolated red block is
picked and delivered every time; with a blue block 0.075 m away the identical grasp fails every
time, because the far finger fouls the neighbour on the way down.

`rule_verdict` cannot tell the two apart even in principle: it measures the distance from the grasp
to the intended block's detected centre, which is 0.020 m in both, and never looks at any other
block. So it returns p=1.000 approve on both. The difference is in the image. Whether the learned
visual verifier sees it is the whole question.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from waddle_wm.agent import DEFAULT_CHECKPOINT, SkillAgent
from waddle_wm.demo import ARMS, Scenario, arm_record, outcome_word
from waddle_wm.perception import landing_pad
from waddle_wm.planner import MODEL
from waddle_wm.sim import relling_scene as scene

DEFAULT_OUT = Path("results/demo-ambiguous")
NEIGHBOUR = "blue_block"
PARKED_YELLOW = (0.66, -0.30)      # out of the way, so only one neighbour is ever ambiguous


def ambiguous_scenes(count: int, spacing: float, seed: int = 0) -> list[dict]:
    """`count` reproducible scenes: a red block, and a blue block `spacing` metres away in +x.

    The opposite of `demo.demo_scenes`, which deliberately keeps every other block at least
    0.075 m away. The red positions are identical across spacings, so two conditions differ only
    in where the neighbour is — and, because the plan is aimed from the red block, not in the plan.
    """
    rng = np.random.default_rng(seed)
    scenes = []
    while len(scenes) < count:
        xy = rng.uniform((0.36, -0.26), (0.50, -0.10))
        red = (round(float(xy[0]), 4), round(float(xy[1]), 4))
        blue = (round(red[0] + spacing, 4), red[1])
        scenes.append({"red": red, "blue": blue, "spacing": spacing})
    return scenes


def scenario_for(spacing: float, offset: float) -> Scenario:
    return Scenario(
        "ambiguous_grasp", "pick up the red block and put it on the green pad",
        block_xy=(0.42, -0.20), grasp_offset=(offset, 0.0), place_offset=(0.0, 0.0),
        flaw=f"grasp aimed {offset * 100:.1f} cm toward the blue block, which sits "
             f"{spacing * 100:.1f} cm away — inside the rules verifier's 2.8 cm tolerance of the "
             "red block, so its verdict is identical at every spacing",
        tests="the case the coordinates cannot express: the plan is the same, only the neighbour "
              "moved, and the physics flips.")


def pin_blocks(agent: SkillAgent, blocks: dict[str, list[float]]):
    """Force every `env.reset` in this run to place the non-red blocks where the scene wants them.

    `SkillAgent.run` resets internally and only threads `block_xy` (the red block) through, so the
    neighbour has to be pinned on the env itself rather than passed down.
    """
    env = agent.env
    original = env.reset

    def reset(block_xy=None, target_xy=None, blocks_=None):
        return original(block_xy, target_xy, blocks_ if blocks_ is not None else blocks)

    env.reset = reset


def delivery(agent: SkillAgent) -> dict:
    """What is actually on the pad when the run ends, and how far every block moved."""
    pad, radius = landing_pad(agent.env.model)
    pad_xy = np.asarray(pad[:2])
    out = {"pad_radius": round(float(radius), 4), "on_pad": [], "blocks": {}}
    for name, position in agent.env.block_positions().items():
        distance = float(np.linalg.norm(np.asarray(position[:2]) - pad_xy))
        out["blocks"][name] = {"xy": [round(v, 4) for v in position[:2]],
                               "z": round(float(position[2]), 4),
                               "pad_distance": round(distance, 4)}
        if distance <= radius:
            out["on_pad"].append(name)
    return out


def run_scene(index: int, scene_spec: dict, scenario: Scenario, arm: str, args) -> dict:
    """One ambiguous scene against one verifier arm, from the shared flawed plan through MuJoCo."""
    red, blue = list(scene_spec["red"]), list(scene_spec["blue"])
    blocks = {NEIGHBOUR: [*blue, scene.BLOCK_HALF],
              "yellow_block": [*PARKED_YELLOW, scene.BLOCK_HALF]}
    agent = SkillAgent(args.checkpoint, seed=args.seed, model=args.model, repairs=args.repairs,
                       verifier_mode=arm)
    pin_blocks(agent, blocks)
    agent.env.reset(red)
    detections = {d.label: d.point_base for d in agent.perceive()}
    pad_xy, _ = landing_pad(agent.env.model)
    opening = scenario.opening_plan(detections["red block"], pad_xy)
    emit = (lambda *_: None) if args.quiet else (
        lambda name, payload: print(f"  [{arm}/{name}] {json.dumps(payload, default=float)[:200]}", flush=True))
    run = agent.run(scenario.instruction, args.seed, red, on_event=emit, opening_plan=opening)

    record = arm_record(scenario, arm, run, opening, block_xy=red)
    record["scene_index"] = index
    record["red_xy"], record["blue_xy"], record["spacing"] = red, blue, scene_spec["spacing"]
    record["detections"] = {label: [round(v, 4) for v in point] for label, point in detections.items()}
    grasp = opening.steps[0].trace[1]["target"][:2]
    record["grasp_xy"] = [round(v, 4) for v in grasp]
    record["grasp_to_red"] = round(float(np.linalg.norm(
        np.asarray(grasp) - np.asarray(detections["red block"][:2]))), 4)
    record["grasp_to_blue"] = round(float(np.linalg.norm(
        np.asarray(grasp) - np.asarray(detections["blue block"][:2]))), 4)
    record["delivery"] = delivery(agent)
    record["wrong_block_delivered"] = any(name != "red_block" for name in record["delivery"]["on_pad"])
    record["neighbour_moved"] = round(float(np.linalg.norm(
        np.asarray(record["delivery"]["blocks"][NEIGHBOUR]["xy"]) - np.asarray(blue))), 4)
    first = record["rounds"][0]
    record["caught"] = bool(first["verdict"]) and not first["verdict"]["approved"]
    return record


def rate_table(records: list[dict], offset: float) -> str:
    lines = ["# Crowded-neighbour sweep", "",
             f"Every row runs the identical opening plan: a grasp aimed {offset * 100:.1f} cm from the red",
             "block's detected centre, inside `rule_verdict`'s 2.8 cm tolerance. Only the blue block's",
             "distance changes between blocks of rows. `caught` counts scenes where the verifier rejected",
             "the opening plan; `p (opening)` is the mean imagined success probability of that plan;",
             "`red on pad` is what MuJoCo actually did.", "",
             "| neighbour at | verifier | caught | p (opening) | red on pad | wrong block on pad | not executed |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for spacing in dict.fromkeys(r["spacing"] for r in records):
        for arm in ARMS:
            rows = [r for r in records if r["arm"] == arm and r["spacing"] == spacing]
            if not rows:
                continue
            caught = sum(r["caught"] for r in rows)
            wrong = sum(r["wrong_block_delivered"] for r in rows)
            right = sum("red_block" in r["delivery"]["on_pad"] for r in rows)
            halted = sum(r["decision"] in ("rejected", "abstained", "error") for r in rows)
            probs = [r["rounds"][0]["verdict"]["imagined_success_probability"]
                     for r in rows if r["rounds"][0]["verdict"]]
            mean_p = f"{sum(probs) / len(probs):.3f}" if probs else "—"
            lines.append(f"| {spacing * 100:.1f} cm | {arm} | {caught}/{len(rows)} | {mean_p} | "
                         f"{right}/{len(rows)} | {wrong}/{len(rows)} | {halted}/{len(rows)} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repairs", type=int, default=2)
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--spacing", type=float, action="append",
                    help="metres between the red and blue blocks; repeat for a gradient")
    ap.add_argument("--offset", type=float, default=0.020,
                    help="metres the grasp is aimed toward the neighbour (rules tolerance is 0.028)")
    ap.add_argument("--arm", action="append", choices=ARMS)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    args = ap.parse_args()

    arms = [arm for arm in ARMS if not args.arm or arm in args.arm]
    spacings = args.spacing or [0.075, 0.220]
    started, records = time.time(), []

    for spacing in spacings:
        scenes = ambiguous_scenes(args.scenes, spacing, args.seed)
        scenario = scenario_for(spacing, args.offset)
        for index, spec in enumerate(scenes):
            for arm in arms:
                record = run_scene(index, spec, scenario, arm, args)
                records.append(record)
                print(f"spacing {spacing * 100:.1f}cm scene {index} "
                      f"red=({spec['red'][0]:.3f}, {spec['red'][1]:.3f}) / {arm}: "
                      f"caught={record['caught']} {outcome_word(record)} "
                      f"on_pad={record['delivery']['on_pad'] or '[]'}", flush=True)

    text = rate_table(records, args.offset)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f".{args.tag}" if args.tag else ""
    (args.out / f"sweep{tag}.md").write_text(text)
    (args.out / f"sweep{tag}.json").write_text(json.dumps(
        {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": args.model, "seed": args.seed,
         "spacings": spacings, "offset": args.offset, "scenes_per_spacing": args.scenes,
         "seconds": round(time.time() - started, 1),
         "cost_usd": round(sum(r["cost_usd"] for r in records), 4),
         "arms": records}, indent=2, default=float))
    print("\n" + text)
    print(f"sweep {args.out / f'sweep{tag}.md'}")


if __name__ == "__main__":
    main()
