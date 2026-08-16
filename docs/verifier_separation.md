# Verifier probability: what it separates, and what it does not

## 1. Separation — flawed plans vs competent plans

- scripted-flawed plans: n=16  p in [0.013, 0.207]  mean 0.086
- model-authored plans:  n=47  p in [0.441, 0.981]  mean 0.849

Separation AUC **1.000** (47 competent vs 16 flawed).

The two populations do not overlap: the highest-scoring flawed plan is 0.207 and the lowest-scoring competent plan is 0.441, so any threshold in [0.207, 0.441] classifies every plan correctly.

## 2. Ranking — within the competent plans only

Ranking AUC **0.070** (43 successes vs 4 failures).

- mean p on plans that succeeded: 0.839
- mean p on plans that failed:    0.958

Probabilities of the plans that failed: 0.971, 0.971, 0.944, 0.944.

**4 failures is too few to estimate this reliably** — read it as 'no evidence of ranking ability', not as a trustworthy point estimate.

## 3. The pooled number, and why it is misleading

Pooled over both populations the AUC is **0.814** (43 successes vs 20 failures) — far better than §2.

That gap is Simpson's paradox, not a result. The scripted-flaw sweeps contribute almost only failures at low p and the live-plan sweeps almost only successes at high p, so a pooled AUC mostly measures which sweep a plan came from. Quote §1 and §2; this figure is here to be discounted, not cited.

## Per-sweep detail

| sweep | model | scenes | successes | failures | mean p |
| --- | --- | --- | --- | --- | --- |
| scripted-flaw | claude-opus-5 | 16 | 0 | 16 | 0.086 |
| live-plan | claude-haiku-4-5-20251001 | 23 | 21 | 2 | 0.853 |
| live-plan | claude-opus-5 | 24 | 22 | 2 | 0.846 |
