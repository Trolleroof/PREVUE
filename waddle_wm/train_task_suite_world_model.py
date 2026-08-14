"""Train the yaw-aware, multi-task state world model.

    observation window (frozen V-JEPA) + block coordinates + a *sequence* of subtask plans
        -> where every block ends up, and whether each subtask and the whole plan succeeds

Three things separate this from `train_multiblock_world_model`:

**The plan is a sequence.** "Place the blue block, then the red one" is two pick-and-places
whose second half acts on the world the first half left. A flat plan vector cannot say that,
so the subtasks are consumed one at a time by a recurrent cell whose hidden state is the
model's belief about the scene after each one. Per-subtask heads are supervised directly, and
the episode's probability is the product of its subtasks' — which is the label's own
definition, not a separate thing to learn.

**Vision has exactly one job, and it is decidable.** The model is *given* every block's
coordinates. What the coordinates do not carry is each block's heading, and the corpus is
built so the heading decides the grasp: the blocks are 100 mm long against an 85 mm jaw
stroke, so a plan whose commanded wrist yaw is far from the block's own cannot close on it.
So `--no-context` is not a nuisance ablation, it is the control that the whole claim rests on:
same architecture, same coordinates, same plans, no pixels. If the two score the same, vision
contributed nothing.

**Degenerate dimensions are held, not divided.** A feature that was constant while the model
was fitted has a standard deviation at the clamp floor, and dividing an inference-time value
by it produces a thousand-sigma input — the bug that made an earlier checkpoint return p=0 for
every candidate. Such dimensions are pinned to zero instead, and the checkpoint records which.

    uv run python -m waddle_wm.train_task_suite_world_model --data data/ur5e_wm_suite \\
        --out models/task_suite_world_model.pt
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from waddle_wm import plan_encoding, windows
from waddle_wm.sim import relling_scene as scene
from waddle_wm.sim.generate_suite import FAMILIES, SUITE_BLOCK_SIZE

SUBTASK_SLOTS = 2                       # the longest family is two pick-and-places
STATE_DIM = 12                          # three block xyz, then the pinch xyz
DEGENERATE_STD = plan_encoding.DEGENERATE_STD
FEATURE_CLAMP = 5.0                     # standard deviations; see `apply_normaliser`


# --------------------------------------------------------------------------------- features


def trace_segments(trace: list[dict]) -> list[list[dict]]:
    """Split a concatenated trace back into one segment per subtask.

    Each pick-and-place starts with its `approach`, so a new approach opens a new segment.
    This reads the trace rather than the generator's bookkeeping on purpose: a candidate
    program handed to the verifier at inference has a trace and nothing else.
    """
    segments: list[list[dict]] = []
    for entry in trace:
        if entry["phase"] == "approach" or not segments:
            segments.append([])
        segments[-1].append(entry)
    return segments


def planned_destinations(subtasks: list[dict], initial: np.ndarray, block_names) -> list[np.ndarray]:
    """Where each subtask's destination is *expected* to be when that subtask runs.

    For the pad, the pad. For a block, wherever the plan already decided to leave that block —
    in `ordered_stack` the support was moved by the previous subtask, so its spawn position is
    the wrong reference and would make a correct plan look badly aimed.
    """
    expected = {name: initial[index * 3:index * 3 + 3].astype(float)
                for index, name in enumerate(block_names)}
    resolved = []
    for subtask in subtasks:
        destination = subtask["destination"]
        if destination == "green_pad":
            resolved.append(np.asarray(scene.TARGET_POS, dtype=float))
        else:
            resolved.append(np.asarray(expected[destination], dtype=float))
        # After this subtask the plan believes its own block is at its aim point.
        expected[subtask["object"]] = np.asarray([*subtask["target_xy"], SUITE_BLOCK_SIZE[2]], dtype=float)
    return resolved


def episode_features(record: dict, initial: np.ndarray, block_names, version: int):
    """(plans, tasks, mask) for one episode, all of it available before anything executes."""
    subtasks = record["skill"]["params"]["subtasks"]
    segments = trace_segments(record["skill"]["trace"])
    if len(segments) != len(subtasks):
        raise ValueError(f"{record['episode_id']}: {len(segments)} trace segments for "
                         f"{len(subtasks)} subtasks")
    # Aim points are read back out of the trace, not out of the generator's bookkeeping, so a
    # candidate program at serving time produces exactly the features training was fitted on.
    aims = [{"object": subtask["object"], "destination": subtask["destination"],
             "target_xy": next(entry["target"] for entry in segment if entry["phase"] == "place")[:2]}
            for subtask, segment in zip(subtasks, segments)]
    destinations = planned_destinations(aims, initial, block_names)
    destination_names = ("green_pad", *block_names)

    plans = np.zeros((SUBTASK_SLOTS, len(plan_encoding.fields(version))), dtype=np.float32)
    tasks = np.zeros((SUBTASK_SLOTS, len(block_names) + len(destination_names) + len(FAMILIES)),
                     dtype=np.float32)
    mask = np.zeros(SUBTASK_SLOTS, dtype=np.float32)
    family = FAMILIES.index(record["family"])

    for k, (subtask, segment, destination_xyz) in enumerate(zip(subtasks, segments, destinations)):
        if k >= SUBTASK_SLOTS:
            raise ValueError(f"{record['episode_id']} has more than {SUBTASK_SLOTS} subtasks")
        source = block_names.index(subtask["object"])
        source_xyz = initial[source * 3:source * 3 + 3]
        grasp = next(entry["target"] for entry in segment if entry["phase"] == "descend")
        place = next(entry["target"] for entry in segment if entry["phase"] == "place")
        grasp_yaw, approach_yaw = plan_encoding.trace_yaws(segment)
        plans[k] = plan_encoding.plan_vector(grasp, place, source_xyz, destination_xyz,
                                             grasp_yaw, approach_yaw, version)
        tasks[k, source] = 1.0
        tasks[k, len(block_names) + destination_names.index(subtask["destination"])] = 1.0
        tasks[k, len(block_names) + len(destination_names) + family] = 1.0
        mask[k] = 1.0
    return plans, tasks, mask


def assemble(records: list[dict], manifest: dict, cache: dict, version: int) -> dict:
    """Every model input and label, as flat arrays in record order."""
    block_names = tuple(manifest["block_names"])
    states = np.stack([windows.window_states(record, manifest) for record in records])
    initial, final = states[:, 0, :STATE_DIM], states[:, -1, :STATE_DIM]

    plans, tasks, masks = [], [], []
    for record, start in zip(records, initial):
        plan, task, mask = episode_features(record, start, block_names, version)
        plans.append(plan), tasks.append(task), masks.append(mask)

    labels = np.zeros((len(records), SUBTASK_SLOTS, 3), dtype=np.float32)
    for index, record in enumerate(records):
        for k, outcome in enumerate(record["subtask_outcomes"][:SUBTASK_SLOTS]):
            labels[index, k] = [outcome["lifted"], outcome["placed"], outcome["success"]]

    return {
        "episode_ids": [record["episode_id"] for record in records],
        "splits": np.array([record["split"] for record in records]),
        "families": np.array([record["family"] for record in records]),
        "context": np.stack([cache[record["episode_id"]][0].numpy() for record in records]).astype(np.float32),
        "initial": initial.astype(np.float32),
        "final": final.astype(np.float32),
        "plan": np.stack(plans),
        "task": np.stack(tasks),
        "mask": np.stack(masks),
        "subtask_labels": labels,
        "success": np.array([float(record["outcome"]["success"]) for record in records], dtype=np.float32),
        "headings": block_headings(records, block_names),
    }


def block_headings(records: list[dict], block_names) -> np.ndarray:
    """(episodes, 2 * blocks) of `sin 2y, cos 2y` per block: the auxiliary target.

    Doubled angle because the jaws are symmetric — a block turned by 180 degrees is the same
    grasp — and sin/cos because a heading is circular and 89 degrees is not far from -89.
    """
    rows = []
    for record in records:
        yaws = record["skill"]["params"]["block_yaws_deg"]
        row = []
        for name in block_names:
            angle = 2.0 * math.radians(float(yaws[name]))
            row.extend([math.sin(angle), math.cos(angle)])
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


# --------------------------------------------------------------------------------- normalisation


def fit_normaliser(values: np.ndarray, train: np.ndarray, rows_mask: np.ndarray | None = None) -> dict:
    """Mean/std over the training rows, with constant dimensions flagged rather than divided.

    `rows_mask` drops padded subtask slots. A one-subtask family contributes an all-zero second
    slot that the model gates away, but leaving those rows in the statistics would drag the mean
    toward zero and shrink the spread of every real subtask.
    """
    rows = values[train].reshape(-1, values.shape[-1])
    if rows_mask is not None:
        rows = rows[rows_mask[train].reshape(-1) > 0]
    mean, std = rows.mean(0), rows.std(0)
    degenerate = std <= DEGENERATE_STD
    return {"mean": mean.astype(np.float32), "std": np.where(degenerate, 1.0, std).astype(np.float32),
            "degenerate": degenerate}


def fit_context_projection(context: np.ndarray, train: np.ndarray, components: int) -> dict | None:
    """Leading principal components of the training contexts, or None to use the latent whole.

    Kept for comparison, and measured to be the wrong compression here. PCA keeps the
    highest-*variance* directions, which in these latents are block positions and arm pose —
    things the model is already handed as coordinates. The heading lives in low-variance
    directions, and projecting onto 32 components nearly doubles the error a ridge readout
    achieves on the full latent (12.2 -> 21.5 degrees). Use `fit_heading_readout` instead.
    """
    if not components:
        return None
    rows = context[train]
    mean = rows.mean(0)
    _, _, basis = np.linalg.svd(rows - mean, full_matrices=False)
    return {"kind": "pca", "mean": mean.astype(np.float32),
            "components": basis[:components].astype(np.float32)}


def fit_heading_readout(context: np.ndarray, headings: np.ndarray, train: np.ndarray,
                        alpha: float = 30.0) -> dict:
    """A ridge probe of the frozen latent onto every block's heading, fitted on train only.

    This is the perception step stated plainly: the coordinates a planner already has carry
    where each block is and not which way it faces, and the heading is what decides whether a
    commanded grasp can close. A linear probe recovers it from the frozen latent to about 12
    degrees, so the visual pathway can be exactly those few numbers rather than a thousand-
    dimensional latent the network will memorise before it finds them.

    Fitted on the training split, folded into the same `(x - mean) @ components.T` form the PCA
    path uses, and stored in the checkpoint — so training and serving apply one identical map.
    Nothing here is available at inference beyond the pixels.
    """
    rows = context[train]
    mean = rows.mean(0)
    scale = rows.std(0) + 1e-6
    scaled = (rows - mean) / scale
    gram = scaled.T @ scaled + alpha * np.eye(scaled.shape[1], dtype=scaled.dtype)
    weights = np.linalg.solve(gram, scaled.T @ headings[train])          # (latent, targets)
    return {"kind": "heading_readout", "mean": mean.astype(np.float32),
            "components": (weights / scale[:, None]).T.astype(np.float32), "alpha": alpha}


def apply_context_projection(context: torch.Tensor, projection: dict | None) -> torch.Tensor:
    if projection is None:
        return context
    mean = torch.as_tensor(projection["mean"], device=context.device)
    components = torch.as_tensor(projection["components"], device=context.device)
    return (context - mean) @ components.T


def apply_normaliser(values: torch.Tensor, stats: dict, clamp: float = FEATURE_CLAMP) -> torch.Tensor:
    """Normalise, holding degenerate dimensions at zero and clamping to the fitted range.

    The clamp is applied during *training* as well as inference on purpose. At inference the
    coordinates come from perception rather than from the simulator, and an input the fit never
    covered would otherwise arrive as a many-sigma value and saturate the ensemble — the failure
    that once made a checkpoint answer p=0 to every candidate. Clamping only at inference would
    fix the overflow while introducing a train/serve mismatch, so both paths clamp identically.
    """
    mean = torch.as_tensor(stats["mean"], device=values.device)
    std = torch.as_tensor(stats["std"], device=values.device)
    keep = torch.as_tensor(~stats["degenerate"], device=values.device)
    return (((values - mean) / std) * keep).clamp(-clamp, clamp)


# --------------------------------------------------------------------------------- the model


class SuiteWorldModel(nn.Module):
    """Scene belief from pixels + coordinates, then one recurrent step per subtask.

    The heading head is the load-bearing detail. Measured on this corpus, a ridge readout
    recovers each block's heading from the frozen latent to about 10 degrees — the information
    is plainly there. But trained on the binary outcome alone the network overfits long before
    it learns to extract it, and scores exactly like its own no-pixels ablation. So each block's
    heading is supervised directly, off the context projection, which forces that pathway to
    carry the one thing the coordinates cannot.

    The labels come from the simulator during training only. At inference the head is not read
    at all: the verifier still sees nothing but pixels, coordinates, and the plan.
    """

    def __init__(self, context_dim: int, plan_dim: int, task_dim: int,
                 hidden: int = 192, context_width: int = 128, dropout: float = 0.1,
                 blocks: int = 3):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(context_dim, context_width), nn.GELU(),
                                     nn.Dropout(dropout))
        self.heading = nn.Linear(context_width, 2 * blocks)
        self.scene = nn.Sequential(nn.Linear(context_width + STATE_DIM, hidden), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        # The context is fed to the subtask encoder as well as to the scene belief. Judging a
        # grasp means comparing the *commanded* heading, which is in the plan, against the
        # *block's* heading, which is only in the pixels — and "compare" here is a product:
        # cos(2(a-b)) = cos 2a cos 2b + sin 2a sin 2b. Routing the context only into the initial
        # hidden state leaves that product to be formed through the GRU's gates across a step
        # boundary. Concatenating it here lets one MLP form it directly, which is the difference
        # between a term the network can express cheaply and one it has to discover.
        self.subtask = nn.Sequential(nn.Linear(plan_dim + task_dim + context_width, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU())
        self.cell = nn.GRUCell(hidden, hidden)
        self.outcome = nn.Linear(hidden, 3)         # lifted, placed, success — per subtask
        self.state = nn.Linear(hidden, STATE_DIM)

    def forward(self, context, state, plan, task, mask):
        projected = self.context(context)
        hidden = self.scene(torch.cat([projected, state], dim=-1))
        logits = []
        for k in range(plan.shape[1]):
            token = self.subtask(torch.cat([plan[:, k], task[:, k], projected], dim=-1))
            stepped = self.cell(token, hidden)
            # A padded slot must not move the belief, so the carry is gated by the mask.
            gate = mask[:, k : k + 1]
            hidden = gate * stepped + (1 - gate) * hidden
            logits.append(self.outcome(hidden))
        return state + self.state(hidden), torch.stack(logits, dim=1), self.heading(projected)


def episode_logit(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log p(plan succeeds) = sum of log p(subtask succeeds), which is what the label means."""
    per_subtask = nn.functional.logsigmoid(logits[..., 2]) * mask
    log_p = per_subtask.sum(dim=1).clamp(min=-30.0)
    # Back to a logit so the loss can stay in log-space and calibrated: log p - log(1 - p).
    log_not_p = torch.log1p(-log_p.exp().clamp(max=1 - 1e-6))
    return log_p - log_not_p


# --------------------------------------------------------------------------------- training


def train_member(data: dict, index: torch.Tensor, val_index: torch.Tensor, args, device, seed: int):
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = SuiteWorldModel(data["context"].shape[1], data["plan"].shape[2], data["task"].shape[2],
                            hidden=args.hidden, context_width=args.context_width,
                            dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bootstrap = index[torch.randint(len(index), (len(index),), generator=generator)].to(device)

    def batch_parts(rows):
        predicted, logits, headings = model(data["context"][rows], data["initial"][rows],
                                            data["plan"][rows], data["task"][rows], data["mask"][rows])
        mask = data["mask"][rows]
        subtask = nn.functional.binary_cross_entropy_with_logits(
            logits, data["subtask_labels"][rows], reduction="none")
        subtask = (subtask.mean(-1) * mask).sum() / mask.sum().clamp_min(1.0)
        episode = nn.functional.binary_cross_entropy_with_logits(
            episode_logit(logits, mask), data["success"][rows])
        state = nn.functional.mse_loss(predicted, data["final"][rows])
        heading = nn.functional.mse_loss(headings, data["headings"][rows])
        return subtask, episode, state, heading

    def batch_loss(rows):
        subtask, episode, state, heading = batch_parts(rows)
        return (args.subtask_weight * subtask + episode + args.state_weight * state
                + args.heading_weight * heading)

    def selection_loss(rows):
        """What the checkpoint is chosen on: the decision, not the scenery.

        Predicting where the blocks end up is a useful auxiliary signal to train against, but it
        is not what a verifier is graded on, and its scale swamps the classification terms. Early
        stopping on the total therefore picks the epoch with the best *state* fit, which is not
        the epoch with the best verdicts.
        """
        subtask, episode, _, _ = batch_parts(rows)
        return args.subtask_weight * subtask + episode

    best = (float("inf"), 0, None)
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = bootstrap[torch.randperm(len(bootstrap), generator=generator).to(device)]
        for start in range(0, len(order), args.batch):
            optimizer.zero_grad()
            batch_loss(order[start:start + args.batch]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            value = float(selection_loss(val_index))
        if value < best[0] - 1e-4:
            best = (value, epoch, {k: v.clone() for k, v in model.state_dict().items()})
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
        if epoch % 10 == 0:
            with torch.inference_mode():
                subtask, episode, state, heading = (float(x) for x in batch_parts(val_index))
            print(f"    epoch {epoch}: val decision {value:.4f} (best {best[0]:.4f} @ {best[1]}) "
                  f"[subtask {subtask:.4f} episode {episode:.4f} state {state:.4f} "
                  f"heading {heading:.4f}]", flush=True)
    model.load_state_dict(best[2])
    model.eval()
    return model, best


def ensemble_scores(members, data, rows, use_context=True):
    """(p(success), per-subtask probabilities, predicted final state, disagreement)."""
    with torch.inference_mode():
        context = data["context"][rows] if use_context else torch.zeros_like(data["context"][rows])
        outputs = [member(context, data["initial"][rows], data["plan"][rows],
                          data["task"][rows], data["mask"][rows]) for member in members]
        states = torch.stack([output[0] for output in outputs])
        logits = torch.stack([output[1] for output in outputs])
        # output[2] is the auxiliary heading readout — a training signal, not a verdict.
        probability = torch.sigmoid(torch.stack([episode_logit(logit, data["mask"][rows])
                                                 for logit in logits]))
    return (probability.mean(0), torch.sigmoid(logits).mean(0), states.mean(0), probability.std(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/ur5e_wm_suite"))
    ap.add_argument("--embeddings", type=Path, help="default: <data>/window_embeddings.pt")
    ap.add_argument("--out", type=Path, default=Path("models/task_suite_world_model.pt"))
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--context-width", type=int, default=64)
    ap.add_argument("--context-mode", choices=("heading", "pca", "latent"), default="heading",
                    help="how the frozen latent reaches the model. `heading` is a ridge probe "
                         "onto each block's heading, fitted on train — the few numbers the "
                         "coordinates omit. `pca` keeps leading principal components, which "
                         "measurably discards the heading. `latent` passes all 1024 dims, which "
                         "overfits within a handful of epochs.")
    ap.add_argument("--context-pca", type=int, default=32, help="components when --context-mode pca")
    ap.add_argument("--subtask-weight", type=float, default=1.0)
    ap.add_argument("--state-weight", type=float, default=0.2)
    ap.add_argument("--heading-weight", type=float, default=2.0,
                    help="auxiliary supervision on each block's heading, read off the context "
                         "projection. Set 0 to ablate it — the model then has to discover the "
                         "heading from the binary outcome alone, which is what it failed to do.")
    ap.add_argument("--max-false-accept", type=float, default=0.10,
                    help="the verifier's operating point: the highest false-accept rate allowed "
                         "on val when picking the decision threshold")
    ap.add_argument("--plan-encoding", type=int, default=plan_encoding.PLAN_ENCODING_VERSION,
                    choices=sorted(plan_encoding.PLAN_FIELDS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    manifest = json.loads((args.data / "manifest.json").read_text())
    records = [json.loads(line) for line in (args.data / "records.jsonl").open()]
    cache = torch.load(args.embeddings or args.data / "window_embeddings.pt", weights_only=False)
    raw = assemble(records, manifest, cache, args.plan_encoding)
    print(f"{len(records)} episodes, context {raw['context'].shape[1]}-d, "
          f"plan {raw['plan'].shape[1]}x{raw['plan'].shape[2]}", flush=True)

    train = raw["splits"] == "train"
    if args.context_mode == "heading":
        projection = fit_heading_readout(raw["context"], raw["headings"], train)
    elif args.context_mode == "pca":
        projection = fit_context_projection(raw["context"], train, args.context_pca)
    else:
        projection = None
    context = apply_context_projection(torch.from_numpy(raw["context"]), projection).numpy()
    if projection is not None:
        print(f"context projected {raw['context'].shape[1]} -> {context.shape[1]} dims", flush=True)
    normalisers = {"context": fit_normaliser(context, train),
                   "initial": fit_normaliser(raw["initial"], train)}
    normalisers["plan"] = fit_normaliser(raw["plan"], train, raw["mask"])
    normalisers["final"] = normalisers["initial"]

    # Informativeness is judged over subtasks that exist. A padded slot is all zeros, so
    # counting it would make the `pinned` flag look variable when no heading varied at all.
    present = raw["plan"][train].reshape(-1, raw["plan"].shape[-1])[raw["mask"][train].reshape(-1) > 0]
    encoding = plan_encoding.yaw_informative(present, np.ones(len(present), dtype=bool),
                                             args.plan_encoding)
    if plan_encoding.orientation_blind(encoding):
        raise SystemExit(f"refusing to train an orientation-blind checkpoint: "
                         f"{plan_encoding.blindness_reason(encoding)}")

    data = {key: torch.from_numpy(np.asarray(raw[key])).to(device)
            for key in ("initial", "final", "plan", "task", "mask",
                        "subtask_labels", "success", "headings")}
    data["context"] = torch.from_numpy(context).to(device)
    for key in ("context", "plan", "initial", "final"):
        data[key] = apply_normaliser(data[key], normalisers[key])

    split = {name: torch.from_numpy(np.flatnonzero(raw["splits"] == name)).to(device)
             for name in ("train", "val", "test")}
    print({name: len(index) for name, index in split.items()}, flush=True)

    members, histories = [], []
    for member in range(args.members):
        print(f"  member {member + 1}/{args.members}", flush=True)
        model, best = train_member(data, split["train"], split["val"], args, device,
                                   args.seed * 1000 + member)
        members.append(model)
        histories.append({"best_val_loss": best[0], "best_epoch": best[1]})

    # The honest no-vision control, trained rather than ablated at test time. Zeroing the context
    # of a model fitted *with* context feeds it an input it never saw, and the resulting number
    # says nothing about what the pixels were worth — measured, it came out *better* than the
    # visual model, which is a diagnostic of that mistake and not a finding. This arm sees the
    # same coordinates, the same plans and the same labels, and no pixels at any point.
    blind_data = {**data, "context": torch.zeros_like(data["context"])}
    blind_args = argparse.Namespace(**{**vars(args), "heading_weight": 0.0})
    blind_members, blind_histories = [], []
    for member in range(args.members):
        print(f"  blind control {member + 1}/{args.members}", flush=True)
        model, best = train_member(blind_data, split["train"], split["val"], blind_args, device,
                                   args.seed * 1000 + member)
        blind_members.append(model)
        blind_histories.append({"best_val_loss": best[0], "best_epoch": best[1]})

    # The operating point is chosen on val: the most accurate threshold whose false-accept rate
    # stays under budget. Test is never consulted for it.
    val_probability, _, _, _ = ensemble_scores(members, data, split["val"])
    val_truth = data["success"][split["val"]]
    candidates = torch.unique(val_probability).sort().values.tolist()
    affordable = [value for value in candidates
                  if float((((val_probability >= value) & (val_truth == 0)).float().sum() /
                            max(1, int((val_truth == 0).sum())))) <= args.max_false_accept]
    threshold = max(affordable, key=lambda value: float(
        ((val_probability >= value).float() == val_truth).float().mean())) if affordable else 1.0

    def quick(rows, use_context=True):
        probability, subtask, _, uncertainty = ensemble_scores(members, data, rows, use_context)
        truth = data["success"][rows]
        verdict = (probability >= threshold).float()
        negatives, positives = max(1, int((truth == 0).sum())), max(1, int((truth == 1).sum()))
        return {"episodes": len(rows),
                "success_accuracy": float((verdict == truth).float().mean()),
                "brier": float((probability - truth).pow(2).mean()),
                "false_accept_rate": float(((verdict == 1) & (truth == 0)).float().sum() / negatives),
                "false_reject_rate": float(((verdict == 0) & (truth == 1)).float().sum() / positives),
                "mean_uncertainty": float(uncertainty.mean())}

    metrics = {"members": histories, "blind_members": blind_histories,
               "decision_threshold": float(threshold),
               "plan_encoding": encoding,
               "majority_class": float(max(raw["success"][raw["splits"] == "test"].mean(),
                                           1 - raw["success"][raw["splits"] == "test"].mean())),
               "test": quick(split["test"]),
               "test_without_vision": quick(split["test"], use_context=False)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": "task_suite_state",
                "members": [{k: v.cpu() for k, v in member.state_dict().items()} for member in members],
                "blind_members": [{k: v.cpu() for k, v in member.state_dict().items()}
                                  for member in blind_members],
                "context_dim": context.shape[1], "latent_dim": raw["context"].shape[1],
                "context_projection": projection, "plan_dim": raw["plan"].shape[2],
                "task_dim": raw["task"].shape[2], "member_count": len(members),
                "hidden": args.hidden, "context_width": args.context_width, "dropout": args.dropout,
                "subtask_slots": SUBTASK_SLOTS, "manifest": manifest,
                "normalisation": normalisers, "plan_encoding": encoding,
                "decision_threshold": float(threshold), "metrics": metrics}, args.out)
    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
