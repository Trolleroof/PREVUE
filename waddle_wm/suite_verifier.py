"""Score a multi-step pick-and-place plan from pixels, before anything moves.

    frames of the untouched scene + estimated block coordinates + a proposed trace
        -> approve / reject, per-subtask diagnosis, and an uncertainty to gate on

This is the serving path for `task_suite_state` checkpoints. It is deliberately separate from
`waddle_wm.verifier`, which serves the single-subtask checkpoints and whose `verify()` takes
one object and one destination — a signature that cannot express "place the blue block, then
stack the red one on it". `Verifier` detects this checkpoint type and delegates here, so
existing callers keep working and new callers get the sequence interface.

Two things it does that a caller cannot skip:

* **Frames go through h264 first.** Every cached training embedding came out of a decoded
  `.mp4`. Handing the frozen backbone raw renderer output is off-distribution, and measurably:
  the same true-positive plans scored 0.17/0.30/0.12 raw against 0.98/0.86/1.00 through the
  codec. `verify_frames` does the round trip for you.
* **Headings come from the pixels, never from the caller.** The plan carries the *commanded*
  wrist yaw; whether that heading matches the block is what the observation window is for. So
  there is no argument here to pass a block orientation in, and the no-vision path is only
  reachable through the explicit `use_context=False` ablation flag.

    uv run python -m waddle_wm.suite_verifier --episode ur5e_00007 --data data/ur5e_wm_suite
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from waddle_wm import plan_encoding
from waddle_wm.embed_windows import clip_frames
from waddle_wm.sim import relling_scene as scene
from waddle_wm.train_task_suite_world_model import (SUBTASK_SLOTS, SuiteWorldModel, apply_context_projection,
                                                    apply_normaliser, episode_logit,
                                                    planned_destinations, trace_segments)
from waddle_wm.verifier import through_codec

FAMILY_FALLBACK = "pad_place"


@dataclass
class SubtaskVerdict:
    object: str
    destination: str
    lifted_probability: float
    placed_probability: float
    success_probability: float
    likely_failure: str | None
    suggestion: str | None


@dataclass
class PlanVerdict:
    approve: bool
    success_probability: float
    uncertainty: float
    subtasks: list[SubtaskVerdict] = field(default_factory=list)
    predicted_block_xyz: dict[str, list[float]] = field(default_factory=dict)
    blocking_subtask: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def diagnose(lifted: float, placed: float) -> tuple[str | None, str | None]:
    """The most useful single sentence about why a subtask is predicted to fail."""
    if lifted < 0.5:
        return ("the grasp does not hold the block",
                "re-aim the grasp at the block's centre and turn the wrist to match its heading")
    if placed < 0.5:
        return ("the block does not reach its destination",
                "move the place waypoint onto the destination")
    return (None, None)


class SuiteVerifier:
    """`verify(frames, trace, subtasks, positions) -> PlanVerdict`."""

    def __init__(self, checkpoint: Path = Path("models/task_suite_world_model.pt"),
                 encoder: Path = Path("models/vjepa2-vitl-fpc64-256"),
                 threshold: float | None = None, device=None, allow_orientation_blind: bool = False):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("model_type") != "task_suite_state":
            raise ValueError(f"{checkpoint} is a {saved.get('model_type')!r} checkpoint; "
                             f"SuiteVerifier serves `task_suite_state`")
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.manifest = saved["manifest"]
        self.block_names = tuple(self.manifest["block_names"])
        self.normalisation = saved["normalisation"]
        self.context_projection = saved.get("context_projection")
        self.plan_encoding = plan_encoding.declared(saved)
        plan_encoding.require_orientation_aware("the task-suite verifier", self.plan_encoding,
                                                allow_orientation_blind, error=ValueError)
        self.threshold = float(saved["decision_threshold"] if threshold is None else threshold)
        self.encoder_path = encoder
        self._encoder = None
        self.members = []
        for state in saved["members"]:
            model = SuiteWorldModel(saved["context_dim"], saved["plan_dim"], saved["task_dim"],
                                    hidden=saved["hidden"], context_width=saved["context_width"],
                                    dropout=saved["dropout"]).to(self.device)
            model.load_state_dict(state)
            model.eval()
            self.members.append(model)
        self.task_dim = saved["task_dim"]

    # ------------------------------------------------------------------ observation

    def encode(self, frames) -> torch.Tensor:
        """One observation window of decoded RGB frames -> the normalised context vector."""
        window = self.manifest["window_frames"]
        if len(frames) != window:
            raise ValueError(f"observation window must be {window} frames, got {len(frames)}")
        if self._encoder is None:
            from transformers import AutoModel, AutoVideoProcessor
            self._encoder = (AutoVideoProcessor.from_pretrained(self.encoder_path, local_files_only=True),
                             AutoModel.from_pretrained(self.encoder_path, local_files_only=True)
                             .to(self.device).eval())
        processor, encoder = self._encoder
        with torch.inference_mode():
            pixels = processor(list(frames), return_tensors="pt")["pixel_values_videos"].to(self.device)
            latent = encoder(pixel_values_videos=pixels).last_hidden_state.mean(dim=1).float()
        # Same projection then the same normalisation the trainer fitted, in that order.
        return apply_normaliser(apply_context_projection(latent, self.context_projection),
                                self.normalisation["context"])

    def encode_live(self, frames) -> torch.Tensor:
        """Encode freshly rendered frames, through the codec the training clips went through."""
        return self.encode(through_codec(frames, self.manifest["fps"]))

    def observation_window(self, data: Path, episode_id: str) -> torch.Tensor:
        frames = clip_frames(data / "clips" / f"{episode_id}.mp4", self.manifest["frames_total"])
        return self.encode(frames[:self.manifest["window_frames"]])

    # ------------------------------------------------------------------ the plan

    def _features(self, trace, subtasks, positions, family):
        from waddle_wm.sim.generate_suite import FAMILIES

        segments = trace_segments(trace)
        if len(segments) != len(subtasks):
            raise ValueError(f"{len(segments)} trace segments for {len(subtasks)} subtasks")
        if len(subtasks) > SUBTASK_SLOTS:
            raise ValueError(f"this checkpoint takes at most {SUBTASK_SLOTS} subtasks")

        # The pinch defaults to the home pose, which is where the arm is in every observation
        # window; a caller that knows better can override it. Passing zeros here would be a
        # metre-scale error on a dimension the fit saw as near-constant.
        pinch = positions.get("pinch", self.manifest["home_waypoint"])
        state = np.array([value for name in self.block_names for value in positions[name][:3]]
                         + list(pinch), dtype=np.float32)
        aims = [{"object": s["object"], "destination": s["destination"],
                 "target_xy": next(e["target"] for e in segment if e["phase"] == "place")[:2]}
                for s, segment in zip(subtasks, segments)]
        destinations = planned_destinations(aims, state, self.block_names)
        destination_names = ("green_pad", *self.block_names)

        plans = np.zeros((SUBTASK_SLOTS, len(plan_encoding.fields(self.plan_encoding["version"]))),
                         dtype=np.float32)
        tasks = np.zeros((SUBTASK_SLOTS, self.task_dim), dtype=np.float32)
        mask = np.zeros(SUBTASK_SLOTS, dtype=np.float32)
        family_index = FAMILIES.index(family) if family in FAMILIES else FAMILIES.index(FAMILY_FALLBACK)
        for k, (subtask, segment, destination_xyz) in enumerate(zip(subtasks, segments, destinations)):
            source = self.block_names.index(subtask["object"])
            grasp = next(entry["target"] for entry in segment if entry["phase"] == "descend")
            place = next(entry["target"] for entry in segment if entry["phase"] == "place")
            grasp_yaw, approach_yaw = plan_encoding.trace_yaws(segment)
            plans[k] = plan_encoding.plan_vector(grasp, place, state[source * 3:source * 3 + 3],
                                                 destination_xyz, grasp_yaw, approach_yaw,
                                                 self.plan_encoding["version"])
            tasks[k, source] = 1.0
            tasks[k, len(self.block_names) + destination_names.index(subtask["destination"])] = 1.0
            tasks[k, len(self.block_names) + len(destination_names) + family_index] = 1.0
            mask[k] = 1.0

        to = lambda array: torch.from_numpy(array).to(self.device).unsqueeze(0)
        return (apply_normaliser(to(plans), self.normalisation["plan"]),
                to(tasks), to(mask),
                apply_normaliser(to(state), self.normalisation["initial"]))

    def verify(self, context: torch.Tensor, trace, subtasks, positions,
               family: str | None = None, use_context: bool = True) -> PlanVerdict:
        """Score a compiled plan. `positions` are the camera's estimates, one xyz per block."""
        plan, task, mask, state = self._features(trace, subtasks, positions,
                                                 family or self._infer_family(subtasks))
        if not use_context:
            context = torch.zeros_like(context)
        with torch.inference_mode():
            outputs = [member(context, state, plan, task, mask) for member in self.members]
            states = torch.stack([output[0] for output in outputs])
            logits = torch.stack([output[1] for output in outputs])
            # output[2] is the training-time heading readout; the verdict does not consult it.
            probability = torch.sigmoid(torch.stack([episode_logit(logit, mask) for logit in logits]))
            per_subtask = torch.sigmoid(logits).mean(0)[0]

        mean, spread = float(probability.mean()), float(probability.std())
        stats = self.normalisation["initial"]
        predicted = (states.mean(0)[0].cpu().numpy() * stats["std"]) + stats["mean"]

        verdicts, blocking = [], None
        for k, subtask in enumerate(subtasks):
            lifted, placed, success = (float(per_subtask[k][0]), float(per_subtask[k][1]),
                                       float(per_subtask[k][2]))
            failure, suggestion = diagnose(lifted, placed)
            verdicts.append(SubtaskVerdict(subtask["object"], subtask["destination"],
                                           lifted, placed, success, failure, suggestion))
            if blocking is None and failure is not None:
                blocking = k
        return PlanVerdict(
            approve=mean >= self.threshold, success_probability=mean, uncertainty=spread,
            subtasks=verdicts, blocking_subtask=blocking,
            predicted_block_xyz={name: predicted[i * 3:i * 3 + 3].tolist()
                                 for i, name in enumerate(self.block_names)})

    def verify_frames(self, frames, trace, subtasks, positions, family: str | None = None,
                      live: bool = True, use_context: bool = True) -> PlanVerdict:
        """The whole path from pixels: encode the window, then score the plan."""
        context = self.encode_live(frames) if live else self.encode(frames)
        return self.verify(context, trace, subtasks, positions, family, use_context)

    @staticmethod
    def _infer_family(subtasks) -> str:
        """Name the family from the plan's shape, so a caller need not know the taxonomy."""
        onto_block = [s["destination"] != "green_pad" for s in subtasks]
        if len(subtasks) == 1:
            return "stack" if onto_block[0] else "pad_place"
        return "ordered_stack" if any(onto_block) else "ordered_pad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/ur5e_wm_suite"))
    ap.add_argument("--checkpoint", type=Path, default=Path("models/task_suite_world_model.pt"))
    ap.add_argument("--episode", required=True, help="score this recorded episode's own plan")
    ap.add_argument("--no-vision", action="store_true", help="run the ablation: zero the context")
    args = ap.parse_args()

    records = {json.loads(line)["episode_id"]: json.loads(line)
               for line in (args.data / "records.jsonl").open()}
    record = records[args.episode]
    verifier = SuiteVerifier(args.checkpoint)
    context = verifier.observation_window(args.data, args.episode)
    positions = {name: value for name, value in record["skill"]["params"]["spawn_positions"].items()}
    verdict = verifier.verify(context, record["skill"]["trace"],
                              record["skill"]["params"]["subtasks"], positions,
                              record["family"], use_context=not args.no_vision)
    print(json.dumps(verdict.as_dict(), indent=2))
    print(f"\nactual outcome: "
          f"{'success' if record['outcome']['success'] else record['outcome']['failure_mode']}")


if __name__ == "__main__":
    main()
