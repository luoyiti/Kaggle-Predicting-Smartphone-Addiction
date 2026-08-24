# Decision Brain — S6E8

Use this loop before adding a model, feature family, or Kernel job.

1. **Start from the decision.** The only downstream action is which `submission.csv` we upload. Ranking (ROC-AUC) is the metric, not a threshold.
2. **Who cares.** The submitter. False positives/negatives are not product costs; mis-ranking is. Compute cost is Kernel minutes.
3. **Hypotheses must be falsifiable.** Example: “dropping categoricals raises OOF AUC” — tested (`lgbm_nocat`, Kernel 0.963771). “Exact-value TE adds identity LightGBM missed” — falsified (−0.040 on full-data 3-fold).
4. **Prior art.** Playground tabular GBMs: LightGBM first, then HistGB/XGB/CatBoost diversity, then a cheap blend. Linear models rarely win on this generator (logreg diagnostic 0.911 vs GBM ~0.954).
5. **Score choices.** (probability the path helps, confidence) × (Kernel hours, complexity, slice risk). Prefer one primary YAML change per experiment.
6. **Adversarial / shift.** Train/test PSI has been negligible. LB probing is not a strategy here. Original-source concat is a shift risk, not extra n.
7. **Simplest fix.** Current best single model is raw 9 numerics + LightGBM, cats dropped. Remaining hard-band errors look like generator noise (column AUC ≈ 0.50). New features must beat that residual, not the easy bulk.
8. **Capture.** Ledger: `docs/mle/observation_ledger.md` and `experiments/LOG.md`. Code/config, not chat.

## Scoring rubric for a new path

Ship the path as YAML+code if any of these is true:

- The trainer/transform does not exist yet (surface gap), even if we expect a dead-end on Kernel.
- Diagnostics have not tested that **primary variable**.
- It is required for promotion (slices, calibration, gates, ensemble).

Do **not** spend a full 5-fold Kernel on a path already falsified at >10× fold std.
