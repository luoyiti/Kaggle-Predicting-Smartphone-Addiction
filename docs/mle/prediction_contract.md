# Prediction contract — S6E8

Kaggle-native mapping of mle-workflow “Define the Prediction Contract”.

| Field | Contract |
| --- | --- |
| Product behavior | Rank test users by addiction probability for ROC-AUC |
| Target | `addicted_label` ∈ {0, 1}; submit **P(class=1)** |
| Entity grain | One row per `id` |
| Output schema | `id,addicted_label` CSV; probabilities in (0, 1) preferred, [0, 1] allowed |
| Serving | Batch only: `scripts/train.py` → `submissions/<exp>.csv` → optional `kaggle competitions submit` |
| Latency | Not applicable (Kernel wall-clock, typically 15–180 min for 5-fold GBM) |
| Fallback | Previous experiment artifacts; do not call an API |
| Human override | None |
| Privacy | Synthetic playground table; still do not commit `data/raw/` or tokens |
| Confidence | Optional calibrated probability; AUC does not require calibration, Brier/ECE are guardrails |
| Model version | `experiment.name` + git SHA in `metrics.json` |

## Do not ship

- Hard 0/1 labels
- Row order different from `sample_submission.csv` without writing `id`
- A diagnostic (`*_diag*`) CSV as if it were a full-data model
