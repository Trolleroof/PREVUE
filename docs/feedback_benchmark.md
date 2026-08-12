# Issue #26 feedback benchmark

The paired repair benchmark uses two explicit budget views:

- **Primary:** equal candidate proposals per scene. At a repair limit of `R`, every arm gets
  `R + 1` proposal rounds. Solve rate is the primary outcome; MuJoCo calls are the efficiency
  axis.
- **Secondary:** equal MuJoCo calls per scene. This is a solve-rate curve, not the primary
  claim, and is selected with `--call-budget`.

The four named arms are Claude self-critique, the estimated-state heuristic, the visual
world-model verifier, and MuJoCo feedback. Every arm receives the same prompt, seed, model,
repair limit, and initial scene. A learned approval never counts as success: the final result
comes from the simulator execution record.

Run the accounting check with:

```bash
uv run python -m waddle_wm.test_benchmark_feedback
```

Run the benchmark with `--repairs 2` for three proposal rounds per scene. Use a separate
`--call-budget` run for the equal-call view; do not compare its filtered solve rate to the
proposal-primary result without reporting the unit.
