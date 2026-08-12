"""How a candidate's action reaches the world model, and which encoding a checkpoint speaks.

The multi-block state model reads a plan as a fixed-width vector. Version 1 carried only
where the grasp and the release were aimed, as offsets from the estimated coordinates they
aim at — six numbers, no heading. Two candidates that differ only in `yaw_deg` therefore
produced *byte-identical* model inputs, which is the representational failure #35 is about:
the pixels carry the block's heading, the pool carries a program that uses it, and the
encoding threw the action away before the network saw it.

Version 2 appends the wrist heading of the grasp and of the approach that precedes it:

    [grasp - source (3), place - destination (3),
     grasp_yaw_pinned, sin 2y, cos 2y, approach_yaw_pinned, sin 2y, cos 2y]

Bounded and stable by construction. `sin 2y, cos 2y` is continuous across the +-90 degree
wrap the program schema allows and identical for a heading and its 180-degree flip, which is
what a two-fingered gripper's symmetry means; a heading left to the solver is not a heading
of zero, so it is flagged separately and carries no angle at all.

A checkpoint records the encoding it was fitted with, and whether that encoding's yaw
dimensions actually varied while it was fitted. Both are needed: a v2 checkpoint trained on a
corpus where every episode left the wrist free is yaw-aware in schema and orientation-blind in
evidence — its yaw dimensions are constant, the trainer's normalisation collapses them, and it
would score yaw-distinct candidates identically while claiming not to. `orientation_blind`
covers both cases, and the readers refuse such a checkpoint unless the caller asks for it by
name.
"""
from __future__ import annotations

import math

import numpy as np

PLAN_ENCODING_VERSION = 2

PLAN_FIELDS = {
    1: ("grasp_dx", "grasp_dy", "grasp_dz", "place_dx", "place_dy", "place_dz"),
    2: ("grasp_dx", "grasp_dy", "grasp_dz", "place_dx", "place_dy", "place_dz",
        "grasp_yaw_pinned", "grasp_yaw_sin2", "grasp_yaw_cos2",
        "approach_yaw_pinned", "approach_yaw_sin2", "approach_yaw_cos2"),
}
YAW_FIELDS = tuple(name for name in PLAN_FIELDS[2] if name not in PLAN_FIELDS[1])

# A dimension whose training standard deviation sits at or below the trainer's clamp floor was
# constant while the model was fitted. Same constant the verifier and the visual selector use
# for their normalisation guard, and the same reasoning: no variance, no information.
DEGENERATE_STD = 1e-5


class PlanEncodingError(RuntimeError):
    """A checkpoint's plan encoding cannot express the distinction being asked of it."""


def fields(version: int = PLAN_ENCODING_VERSION) -> tuple[str, ...]:
    if version not in PLAN_FIELDS:
        raise PlanEncodingError(f"unknown plan encoding version {version!r}; "
                                f"this build writes {PLAN_ENCODING_VERSION}")
    return PLAN_FIELDS[version]


def yaw_terms(yaw_rad: float | None) -> tuple[float, float, float]:
    """One wrist heading -> (pinned, sin 2y, cos 2y). A free wrist is all zeros, not zero yaw."""
    if yaw_rad is None:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * float(yaw_rad)
    return (1.0, math.sin(angle), math.cos(angle))


def plan_vector(grasp_xyz, place_xyz, source_xyz, destination_xyz,
                grasp_yaw: float | None = None, approach_yaw: float | None = None,
                version: int = PLAN_ENCODING_VERSION) -> np.ndarray:
    """One candidate's action, as the checkpoint's plan vector. Radians in, float32 out."""
    names = fields(version)
    grasp = np.asarray(grasp_xyz, dtype=np.float64)[:3] - np.asarray(source_xyz, dtype=np.float64)[:3]
    place = np.asarray(place_xyz, dtype=np.float64)[:3] - np.asarray(destination_xyz, dtype=np.float64)[:3]
    row = [*grasp.tolist(), *place.tolist()]
    if version >= 2:
        row.extend(yaw_terms(grasp_yaw))
        row.extend(yaw_terms(approach_yaw))
    if len(row) != len(names):
        raise PlanEncodingError(f"encoded {len(row)} numbers for a {len(names)}-field encoding")
    return np.asarray(row, dtype=np.float32)


def trace_yaws(trace, grasp_phase: str = "descend", approach_phase: str = "approach"):
    """(grasp yaw, approach yaw) from the last matching waypoints before the close."""
    grasp = approach = None
    for entry in trace:
        phase = entry.get("phase")
        yaw = entry.get("yaw")
        if phase == grasp_phase:
            grasp = None if yaw is None else float(yaw)
        elif phase == approach_phase:
            approach = None if yaw is None else float(yaw)
        if phase == "close":
            break
    return grasp, approach


# --------------------------------------------------------------------------- what a checkpoint says


def declared(saved: dict) -> dict:
    """The plan encoding a checkpoint declares, filled in for checkpoints written before #35.

    A checkpoint with no `plan_encoding` block predates versioning, and there is exactly one
    thing it can be: the six-number, orientation-blind v1 encoding. Saying so here is what
    keeps an old file from being read as a new one.
    """
    block = dict(saved.get("plan_encoding") or {})
    version = int(block.get("version", 1))
    names = block.get("fields") or list(fields(version))
    yaw_informative = bool(block.get("yaw_informative", False)) and version >= 2
    return {"version": version, "fields": list(names), "yaw_informative": yaw_informative,
            "yaw_dims": list(block.get("yaw_dims") or []),
            "yaw_std": [float(value) for value in (block.get("yaw_std") or [])],
            "declared": "plan_encoding" in saved}


def yaw_informative(plan: np.ndarray, train_mask, version: int = PLAN_ENCODING_VERSION) -> dict:
    """Did the yaw dimensions actually vary over the rows the model was fitted on?

    Called by the trainer with the raw (unnormalised) plan matrix and its train split. A yaw
    dimension that was constant is one the normalisation will collapse, so a checkpoint fitted
    that way is orientation-blind however new its schema is.
    """
    names = fields(version)
    dims = [index for index, name in enumerate(names) if name in YAW_FIELDS]
    values = np.asarray(plan, dtype=np.float64)[np.asarray(train_mask, dtype=bool)]
    stds = [float(values[:, index].std()) for index in dims] if len(values) else [0.0] * len(dims)
    by_name = {names[index]: std for index, std in zip(dims, stds)}
    groups = {heading: any(by_name.get(f"{heading}_yaw_{term}", 0.0) > DEGENERATE_STD
                           for term in ("sin2", "cos2"))
              for heading in ("grasp", "approach")}
    return {"version": version, "fields": list(names), "yaw_dims": dims, "yaw_std": stds,
            "yaw_informative": bool(dims) and all(groups.values())}


def orientation_blind(encoding: dict) -> bool:
    """True when this checkpoint cannot separate two candidates that differ only in yaw."""
    return int(encoding.get("version", 1)) < 2 or not encoding.get("yaw_informative")


def blindness_reason(encoding: dict) -> str:
    if int(encoding.get("version", 1)) < 2:
        return (f"its plan encoding is version {encoding.get('version', 1)} "
                f"({len(encoding.get('fields') or fields(1))} fields, no wrist heading)")
    return ("its plan encoding is version 2 but at least one heading had no angular variation while it was "
            f"fitted (std {['%.2g' % s for s in encoding.get('yaw_std') or []]}), so the "
            f"normalisation collapses them")


def require_orientation_aware(what: str, encoding: dict, allow_blind: bool = False,
                              error=PlanEncodingError) -> None:
    """Fail loudly on an orientation-blind checkpoint unless the caller asked for one by name."""
    if not orientation_blind(encoding) or allow_blind:
        return
    raise error(f"{what} cannot rank grasp orientations: {blindness_reason(encoding)}, so two "
                f"candidates differing only in `yaw_deg` get identical model inputs. Retrain a "
                f"yaw-aware checkpoint on a corpus with commanded headings (plan encoding "
                f"v{PLAN_ENCODING_VERSION}), or opt in explicitly to run it as a declared limitation.")
