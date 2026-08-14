"""Render what the verifier said *before* the arm moved, beside what then happened.

A table of accuracies does not show you a false accept. This does: for each chosen episode it
scores the plan from the eight-frame observation window alone — no execution, no labels — and
then plays the episode next to that verdict, revealing the true outcome only once the arm has
finished.

Each clip carries three things worth reading together:

* the **per-subtask** breakdown, so a two-step plan shows *which* half the model doubts;
* the **no-vision control**, the separately trained no-pixels ensemble, so the contribution of
  the observation window is visible per episode rather than only in aggregate;
* the quadrant label — true accept, true reject, **false accept**, **false reject** — because
  the failures are the point of looking.

    uv run python -m waddle_wm.render_suite_verdicts --data data/ur5e_wm_suite
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

from waddle_wm.embed_windows import clip_frames
from waddle_wm.suite_verifier import SuiteVerifier

CLIP_SIZE = 384
PANEL_WIDTH = 480      # 384 + 480 = 864, divisible by 16, so h264 does not rescale the panel text
LINE = 17
FONT = cv2.FONT_HERSHEY_SIMPLEX

INK = (238, 238, 238)
DIM = (150, 150, 150)
GOOD = (120, 226, 140)
BAD = (120, 130, 245)
WARN = (110, 200, 250)
BACKGROUND = (22, 18, 15)

QUADRANTS = {(True, True): ("TRUE ACCEPT", GOOD), (False, False): ("TRUE REJECT", GOOD),
             (True, False): ("FALSE ACCEPT", BAD), (False, True): ("FALSE REJECT", WARN)}


def put(canvas, text, x, y, colour=INK, scale=0.42, thickness=1):
    cv2.putText(canvas, text, (x, y), FONT, scale, colour, thickness, cv2.LINE_AA)


def panel(record, verdict, blind, reveal: bool, frame_index: int, total: int) -> np.ndarray:
    """The text column: what was predicted, and — once the arm has finished — what happened."""
    canvas = np.full((CLIP_SIZE, PANEL_WIDTH, 3), BACKGROUND, dtype=np.uint8)
    truth = bool(record["outcome"]["success"])
    y = 26

    put(canvas, record["episode_id"], 16, y, INK, 0.52, 1)
    put(canvas, record["family"], 190, y, DIM, 0.45)
    y += LINE + 6

    window = record["observation"]["window_frames"]
    if frame_index < window:
        put(canvas, f"observation window  {frame_index + 1}/{window}", 16, y, WARN, 0.40)
    else:
        put(canvas, "executing the proposed plan", 16, y, DIM, 0.40)
    y += LINE + 8

    put(canvas, "VERDICT, BEFORE ANYTHING MOVED", 16, y, DIM, 0.38)
    y += LINE + 2
    decision = "APPROVE" if verdict.approve else "REJECT"
    put(canvas, decision, 16, y, GOOD if verdict.approve else BAD, 0.60, 2)
    put(canvas, f"p(success) {verdict.success_probability:.2f}", 130, y, INK, 0.46)
    put(canvas, f"+-{verdict.uncertainty:.2f}", 300, y, DIM, 0.42)
    y += LINE + 10

    for index, subtask in enumerate(verdict.subtasks):
        marker = ">" if index == verdict.blocking_subtask else " "
        colour = BAD if index == verdict.blocking_subtask else INK
        destination = "pad" if subtask.destination == "green_pad" else subtask.destination.split("_")[0]
        put(canvas, f"{marker} {index + 1}. {subtask.object.split('_')[0]} -> {destination}",
            16, y, colour, 0.44)
        put(canvas, f"lift {subtask.lifted_probability:.2f}  place {subtask.placed_probability:.2f}"
                    f"  = {subtask.success_probability:.2f}", 175, y, colour, 0.40)
        y += LINE
        if subtask.likely_failure:
            put(canvas, f'    "{subtask.likely_failure}"', 16, y, BAD, 0.38)
            y += LINE - 2
            put(canvas, f"    fix: {subtask.suggestion}"[:62], 16, y, DIM, 0.36)
            y += LINE - 2
        y += 3

    y += 6
    put(canvas, "no-vision control (coordinates only)", 16, y, DIM, 0.38)
    y += LINE
    if blind is None:
        put(canvas, "unavailable in this checkpoint", 16, y, DIM, 0.40)
    else:
        agrees = blind.approve == verdict.approve
        put(canvas, f"{'APPROVE' if blind.approve else 'REJECT'}  p {blind.success_probability:.2f}",
            16, y, DIM, 0.46)
        put(canvas, "(agrees)" if agrees else "(disagrees)", 175, y,
            DIM if agrees else WARN, 0.40)
    y += LINE + 12

    # The truth is withheld until the arm has actually finished, so the verdict reads as a
    # prediction rather than as a caption written after the fact.
    if reveal:
        label, colour = QUADRANTS[(verdict.approve, truth)]
        outcome = "SUCCESS" if truth else (record["outcome"]["failure_mode"] or "failure")
        put(canvas, f"actual: {outcome}", 16, y, INK, 0.50)
        y += LINE + 4
        put(canvas, label, 16, y, colour, 0.54, 2)
        if blind is not None and blind.approve != verdict.approve:
            y += LINE + 2
            put(canvas, "vision changed the answer here"
                if (verdict.approve == truth) else "vision changed it the wrong way",
                16, y, GOOD if verdict.approve == truth else BAD, 0.38)
    else:
        put(canvas, "outcome hidden until the plan finishes", 16, y, DIM, 0.38)

    bar = int(PANEL_WIDTH * (frame_index + 1) / total)
    canvas[CLIP_SIZE - 4:, :bar] = (90, 90, 90)
    return canvas


def render(record, frames, verdict, blind, out: Path, fps: int) -> None:
    total = len(frames)
    reveal_at = max(0, total - 14)
    composed = []
    for index, frame in enumerate(frames):
        clip = cv2.resize(frame, (CLIP_SIZE, CLIP_SIZE), interpolation=cv2.INTER_LINEAR)
        side = panel(record, verdict, blind, index >= reveal_at, index, total)
        composed.append(np.concatenate([clip, side], axis=1))
    iio.imwrite(out, np.stack(composed), fps=fps, codec="libx264")


def choose(records, verdicts, per_quadrant: int) -> list[str]:
    """A few of each quadrant, preferring two-subtask families so the breakdown has something to say."""
    buckets: dict[tuple, list] = {key: [] for key in QUADRANTS}
    order = sorted(records, key=lambda r: (len(r["skill"]["params"]["subtasks"]) < 2,
                                           r["episode_id"]))
    for record in order:
        verdict = verdicts[record["episode_id"]]
        key = (verdict.approve, bool(record["outcome"]["success"]))
        if len(buckets[key]) < per_quadrant:
            buckets[key].append(record["episode_id"])
    return [episode for key in QUADRANTS for episode in buckets[key]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/ur5e_wm_suite"))
    ap.add_argument("--checkpoint", type=Path, default=Path("models/task_suite_world_model.pt"))
    ap.add_argument("--out", type=Path, default=Path("results/suite_verdicts"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-quadrant", type=int, default=2)
    ap.add_argument("--pool", type=int, default=60, help="episodes to score before choosing")
    ap.add_argument("--episodes", nargs="*", help="render exactly these instead of choosing")
    args = ap.parse_args()

    records = [json.loads(line) for line in (args.data / "records.jsonl").open()]
    by_id = {record["episode_id"]: record for record in records}
    candidates = ([by_id[episode] for episode in args.episodes] if args.episodes else
                  [r for r in records if r["split"] == args.split][:args.pool])

    verifier = SuiteVerifier(args.checkpoint)
    manifest = verifier.manifest
    fps = manifest["fps"]

    scored, verdicts, blinds = [], {}, {}
    for record in candidates:
        episode = record["episode_id"]
        context = verifier.observation_window(args.data, episode)
        subtasks = record["skill"]["params"]["subtasks"]
        positions = record["skill"]["params"]["spawn_positions"]
        verdicts[episode] = verifier.verify(context, record["skill"]["trace"], subtasks,
                                            positions, record["family"])
        blinds[episode] = (verifier.verify(context, record["skill"]["trace"], subtasks, positions,
                                           record["family"], blind_control=True)
                           if verifier.blind_members else None)
        scored.append(record)
        print(f"  scored {episode}: p={verdicts[episode].success_probability:.2f} "
              f"{'approve' if verdicts[episode].approve else 'reject'} "
              f"(actual {'success' if record['outcome']['success'] else 'failure'})", flush=True)

    chosen = args.episodes or choose(scored, verdicts, args.per_quadrant)
    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    for episode in chosen:
        record = by_id[episode]
        frames = clip_frames(args.data / record["observation"]["frames_path"],
                             manifest["frames_total"])
        verdict, blind = verdicts[episode], blinds[episode]
        label = QUADRANTS[(verdict.approve, bool(record["outcome"]["success"]))][0]
        path = args.out / f"{label.lower().replace(' ', '-')}-{episode}.mp4"
        render(record, frames, verdict, blind, path, fps)
        index.append({"episode": episode, "family": record["family"], "quadrant": label,
                      "p_success": verdict.success_probability, "approve": verdict.approve,
                      "uncertainty": verdict.uncertainty,
                      "blind_p_success": None if blind is None else blind.success_probability,
                      "blind_approve": None if blind is None else blind.approve,
                      "actual": bool(record["outcome"]["success"]),
                      "failure_mode": record["outcome"]["failure_mode"],
                      "blocking_subtask": verdict.blocking_subtask, "clip": str(path)})
        print(f"{label:<13} {episode}  p={verdict.success_probability:.2f}  -> {path}")

    (args.out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote {len(index)} clips and {args.out / 'index.json'}")


if __name__ == "__main__":
    main()
