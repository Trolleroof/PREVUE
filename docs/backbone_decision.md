# Backbone Decision: frozen V-JEPA 2 ViT-L + trained dynamics head

**Decision: V-JEPA 2-AC is out of scope for this milestone.** The runnable
backbone is the already-downloaded frozen V-JEPA 2 ViT-L encoder plus a small
trainable action-conditioned dynamics head. Recorded 2026-08-09.

## What is actually on this machine

| artefact | state |
| --- | --- |
| `models/vjepa2-vitl-fpc64-256/` | complete — 1.3 GB, 326 M params, loads on MPS |
| `models/vjepa2-ac/` | **README only, 16 KB** — `vjepa2-ac-vitg.pt` never downloaded |
| `transformers` 5.14.1 | exports `VJEPA2Config/Model/ForVideoClassification/VideoProcessor` |

Measured: an 8-frame 256x256 window through the frozen ViT-L trunk is **0.23 s on
MPS** (0.47 s for 16 frames). Caching every window of a 1000-episode corpus is
~20 minutes, once.

## Why not V-JEPA 2-AC

1. **The weights are not here.** What was fetched is a third-party HF-endpoints
   wrapper card, not Meta's checkpoint. The real one is ViT-G scale (~1 B trunk +
   ~300 M action predictor).
2. **`transformers` cannot load the AC predictor.** `VJEPA2Model` does expose a
   `predictor`, but it is the *mask-token* JEPA predictor — it in-fills masked
   tokens of the same clip. It takes no action input. Action conditioning lives
   only in `facebookresearch/vjepa2`, which would have to be vendored, along with
   its own config/loading path, and run on MPS unverified.
3. **The action interface does not match ours.** AC is conditioned on Droid-style
   end-effector state `(x, y, z, rx, ry, rz, gripper)` at 4 Hz. Our action is a
   skill-trace phase plus a commanded waypoint at 10 Hz. Using AC means writing a
   conversion layer and inheriting its frame rate, which is a second project.
4. **Its intended use is planning, not verification.** AC is driven by CEM over
   action sequences against a *goal image*. Pre-execution verification of a named
   skill has no goal image; it has a proposed trace. Bending AC into that shape
   costs more than training a 2-layer head.
5. **Fine-tuning a 1 B trunk on an M4 Pro is already out of scope** per the
   [`project.md`](project.md), so AC would be frozen anyway — and a frozen model we cannot
   condition on our own actions predicts nothing useful for us.

## What replaces it

```text
frames  --(frozen ViT-L, cached)-->  z_k (1024-d)
z_k, chunk_k  --(trained MLP f)-->   z_hat_{k+1} = z_k + f(z_k, chunk_k)
z             --(trained head g)-->  block xyz, pinch xyz, lifted, in_target
```

`f` and `g` together are a few hundred thousand parameters; training is seconds on
the cached latents. Uncertainty comes from an ensemble of `K` independently
initialised `f`s trained on bootstrap resamples — disagreement between imagined
futures, which is the quantity the verifier actually needs and which a single
frozen AC model would not give us either.

This keeps every property [`project.md`](project.md) asks for: latent (not pixel)
prediction, frozen visual backbone, skill-level action conditioning, laptop-sized.

## Re-entry criteria

Revisit AC when **all** of these hold, and keep it behind the same
`predict_rollout(z0, chunks) -> [z_hat]` interface so it is a drop-in swap:

- the trained head plateaus and the residual error is shown to be *perceptual*
  (readout from a real latent is also wrong), not dynamical;
- `vjepa2-ac-vitg.pt` is downloaded and loads on MPS within memory;
- our waypoint actions have a tested mapping onto the 7-d EE convention.

Until then the frozen-trunk baseline is the thing that runs this week.
