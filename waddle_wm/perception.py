"""Perception primitives: a text query becomes a pixel box, then a point in the robot base frame.

This mirrors the primitive layer Waddle Labs describes — `bounding_box(text query -> box
in frame)`, `detect_in_base(box -> point in the robot base frame)` — so the planner never
reads simulator state. Everything here comes out of the camera: the box from the rendered
image, the depth from the depth buffer, the transform from the camera and base poses.

In simulation the open-vocabulary detector is stood in for by MuJoCo's segmentation
buffer plus a small word list, so `bounding_box("the red block")` resolves the same way a
real detector would resolve a phrase to one object — but without a network, and without
the pixel-level failure modes a real detector has. Everything downstream of the box is the
real geometry: unprojection through the intrinsics, the camera extrinsics, and the base
transform. Swapping in a real detector means replacing `bounding_box` and nothing else.

    uv run python -m waddle_wm.perception --query "the red block"
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import mujoco
import numpy as np

from waddle_wm.sim import relling_scene as scene

# What a phrase may call each nameable thing in the scene. A real detector would be
# open-vocabulary; this is the sim stand-in for it.
VOCABULARY = {
    "red_block_geom": ("red", "block", "cube", "brick"),
    "blue_block_geom": ("blue", "block", "cube", "brick"),
    "yellow_block_geom": ("yellow", "block", "cube", "brick"),
}
LABELS = {"red_block_geom": "red block", "blue_block_geom": "blue block", "yellow_block_geom": "yellow block"}
COLOURS = {"red", "blue", "yellow", "green"}
QUERIES = ("the red block", "the blue block", "the yellow block")   # what the agent looks for each turn


@dataclass
class Detection:
    label: str
    query: str
    box: list[int]                       # [x0, y0, x1, y1] in pixels
    pixels: int                          # how many pixels of the object were visible
    score: float                         # fraction of the query's words the label matched
    point_base: list[float] = field(default_factory=list)   # [x, y, z] metres, robot base frame
    depth_m: float = 0.0
    size_m: float = 0.0                  # measured extent along the camera's view axis

    def summary(self) -> dict:
        return {"label": self.label, "box": self.box, "pixels": self.pixels, "score": round(self.score, 2),
                "point_base": [round(v, 4) for v in self.point_base],
                "depth_m": round(self.depth_m, 4), "size_m": round(self.size_m, 4)}


class SceneCamera:
    """One calibrated camera on the scene: RGB, depth, segmentation, and the two primitives."""

    def __init__(self, model, data, camera: str = "demo", width: int = 256, height: int = 256):
        self.model, self.data, self.camera = model, data, camera
        self.width, self.height = width, height
        self._renderer = mujoco.Renderer(model, height, width)
        self._camera_id = model.camera(camera).id
        self._base = model.body("base").id

    def close(self):
        self._renderer.close()

    def rgb(self) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render().copy()

    def depth(self) -> np.ndarray:
        """Metres along the camera's view axis, per pixel."""
        self._renderer.enable_depth_rendering()
        try:
            self._renderer.update_scene(self.data, camera=self.camera)
            return self._renderer.render().copy()
        finally:
            self._renderer.disable_depth_rendering()

    def segmentation(self) -> np.ndarray:
        """Per-pixel geom id (-1 where nothing was drawn)."""
        self._renderer.enable_segmentation_rendering()
        try:
            self._renderer.update_scene(self.data, camera=self.camera)
            buffer = self._renderer.render().copy()
        finally:
            self._renderer.disable_segmentation_rendering()
        ids = buffer[..., 0].astype(int)
        ids[buffer[..., 1] != mujoco.mjtObj.mjOBJ_GEOM.value] = -1
        return ids

    @property
    def intrinsics(self):
        """(fx, fy, cx, cy) from the camera's vertical field of view and square pixels."""
        fovy = np.deg2rad(float(self.model.cam_fovy[self._camera_id]))
        focal = 0.5 * self.height / np.tan(0.5 * fovy)
        return focal, focal, (self.width - 1) / 2.0, (self.height - 1) / 2.0

    def base_frame(self):
        """The frame waypoints are commanded in.

        On real hardware that is the robot base frame. Here the arm is commanded in MuJoCo's
        world frame, and the `base` body's own frame is rotated 180 degrees about z relative
        to it — reporting points in the body frame would silently flip x and y under every
        waypoint in the repo, so the command frame is what `detect_in_base` returns.
        """
        return np.zeros(3), np.eye(3)

    def bounding_box(self, query: str, segmentation: np.ndarray | None = None) -> Detection | None:
        """Text query -> the tightest box around the best-matching object, or None."""
        words = {word.strip(".,").lower() for word in query.split()}
        best, best_score = None, 0.0
        for name, vocabulary in VOCABULARY.items():
            hits = words & set(vocabulary)
            if not hits or (words & COLOURS and not (words & COLOURS) & set(vocabulary)):
                continue                                  # a named colour has to be the right one
            score = len(hits) / max(1, len(words & (set(vocabulary) | COLOURS)))
            if score > best_score:
                best, best_score = name, score
        if best is None:
            return None

        segmentation = self.segmentation() if segmentation is None else segmentation
        mask = segmentation == self.model.geom(best).id
        if not mask.any():
            return None                                   # occluded or out of frame
        rows, columns = np.where(mask)
        box = [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())]
        return Detection(LABELS[best], query, box, int(mask.sum()), best_score)

    def detect_in_base(self, box, depth: np.ndarray | None = None) -> tuple[list[float], float, float]:
        """Box -> the object's centre in the base frame, unprojected from the depth buffer.

        Every pixel inside the box that is nearer than the background behind it is
        unprojected into a small point cloud. The cloud is only the camera-facing surface,
        so its centroid sits half an object in front of the true centre; pushing it back
        along the view direction by half the cloud's depth extent recovers the centre. That
        correction is what takes the lateral error from ~16 mm to ~7 mm, well inside the
        grasp tolerance. Returns (point, distance, measured size along the view axis).
        """
        depth = self.depth() if depth is None else depth
        x0, y0, x1, y1 = box
        crop = depth[y0:y1 + 1, x0:x1 + 1]
        rows, columns = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        near = crop <= np.median(crop) + 0.02          # drop the table showing through the box
        distances, rows, columns = crop[near], rows[near], columns[near]

        fx, fy, cx, cy = self.intrinsics
        # MuJoCo cameras look down -z with +x right and +y up; image rows run downwards.
        in_camera = np.stack([(columns - cx) * distances / fx,
                              -(rows - cy) * distances / fy,
                              -distances], axis=-1)
        rotation = self.data.cam_xmat[self._camera_id].reshape(3, 3)
        cloud = in_camera @ rotation.T + self.data.cam_xpos[self._camera_id]

        view = rotation @ np.array([0.0, 0.0, -1.0])   # unit vector pointing into the scene
        along = cloud @ view
        size = float(along.max() - along.min())
        centre = cloud.mean(axis=0) + view * (size / 2)

        base_pos, base_mat = self.base_frame()
        return (base_mat.T @ (centre - base_pos)).tolist(), float(np.median(distances)), size

    def detect(self, query: str) -> Detection | None:
        """`bounding_box` then `detect_in_base`, the pair the planner actually uses."""
        detection = self.bounding_box(query)
        if detection is None:
            return None
        detection.point_base, detection.depth_m, detection.size_m = self.detect_in_base(detection.box)
        return detection

    def detect_all(self, queries) -> list[Detection]:
        segmentation, depth = self.segmentation(), self.depth()
        found = []
        for query in queries:
            detection = self.bounding_box(query, segmentation)
            if detection is None:
                continue
            detection.point_base, detection.depth_m, detection.size_m = self.detect_in_base(detection.box, depth)
            found.append(detection)
        return found


def landing_pad(model) -> tuple[list[float], float]:
    """The green pad is a fixed site, not a detected object: its pose is part of the task."""
    site = model.site("target").id
    return model.site_pos[site][:2].tolist(), float(model.site_size[site][0])


def main():
    ap = argparse.ArgumentParser(description="Run the perception primitives on one scene")
    ap.add_argument("--query", default="the red block")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from waddle_wm.sim.env import TabletopEnv

    env = TabletopEnv(seed=args.seed)
    env.reset(env.sample_scene())
    camera = SceneCamera(env.model, env.data)
    detection = camera.detect(args.query)
    if detection is None:
        print(f"nothing matched {args.query!r}")
        return
    truth = env.data.joint("red_block_free").qpos[:3]
    print(f"{detection.summary()}")
    print(f"ground truth (red block body frame): {np.round(truth, 4).tolist()}")
    print(f"xy error: {np.linalg.norm(np.subtract(detection.point_base[:2], truth[:2])) * 1000:.1f} mm")


if __name__ == "__main__":
    main()
