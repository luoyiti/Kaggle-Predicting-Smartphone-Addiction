# Iteration Compact — Playground Series S6E8

Mapped from the mle-workflow Iteration Compact onto this Kaggle repo.
There is no online serving, feature store, or production dashboard.

```text
Goal:
  Exhaust config-driven modeling paths for Predicting Smartphone Addiction
  (playground-series-s6e8) and leave a reviewable HTML report. Optimize
  OOF ROC-AUC of P(addicted_label=1) without leakage or irreproducible notebooks.

Who cares:
  The competitor (submitter). Kaggle scores ranking quality, not a product
  threshold. No downstream human review queue.

Decision owner:
  The repo maintainer promoting a Kernel experiment to a submission CSV.
  Rollback = keep the previous unique experiment name / OOF / submission.

User or system action changed by the model:
  A batch submission file (id, addicted_label probability). Leaderboard rank
  is the only product action.

Success metric:
  ROC-AUC on P(class=1). Official number requires full-data 5-fold OOF
  (n_splits=5, no row subsample) in oof/<name>/metrics.json from a Kernel
  (or equivalent) run. Diagnostic 80k×3 AUC is ranking-only.

Guardrail metrics:
  Per-fold std; slice AUC (gender, stress_level, academic_work_impact,
  missingness cohorts, hard-probability band); Brier / ECE; OOF correlation
  of ensemble members; train/test PSI from the data contract audit.

Mistake budget:
  Tiny OOF lifts (< ~1× fold std, ~0.0006 on full 5-fold) are noise. Do not
  ship complexity for <0.0001 AUC. Synthetic-data AUC is never a score.

Unacceptable mistakes:
  Label leakage (target in features, fold-unaware TE, mixing original 7500
  rows into train). Submitting hard labels. Treating diagnostic AUC as
  leaderboard. Overwriting another experiment's OOF. Claiming a Kernel
  score without metrics.json. GPU-forcing the CPU baseline LightGBM.

Acceptable mistakes:
  Well-calibrated probabilities that slightly compress a hard band.
  Dropping the three categoricals (noise on original + playground).
  Not using coverage/ratio features that trees already surrogate.

Assumptions:
  Playground is a smoothed version of an original OR-of-usage rule plus
  exact-value identity on notifications/app_opens. Missingness on daily
  screen is largely MCAR vs label. id is a sequential split, not a feature.

Constraints:
  YAML-driven hyperparameters. New experiments = new configs/<name>.yaml.
  Full 5-fold on 691k rows only on Kaggle Kernels. GitHub Actions and this
  VM: smoke + optional sample diagnostics. No K8s / MLflow Server / feature store.

Labels and data snapshot:
  Target addicted_label in train.csv only. Test labels unknown until LB.
  Snapshot = Kaggle competition CSVs (data/raw locally, /kaggle/input on Kernel).
  Record git SHA when available; never invent one.

Baseline:
  configs/baseline.yaml (LightGBM CPU + ratios/n_missing). Beaten on
  diagnostics by raw columns. Current production-equivalent:
  configs/lgbm_nocat.yaml, full 5-fold OOF 0.963771 (Kernel).
  Optional blend_nocat 0.963806.

Candidate signals:
  Missing flags; leisure/work and screen/sleep ratios; pairwise interactions;
  quantile bins; ordinal cats; native cats vs drop; CatBoost / ExtraTrees /
  XGB on nocat; isotonic calibration; rank / AUC-weighted / logistic stack.

Threshold or config plan:
  No decision threshold (AUC ranking). Promotion: candidate OOF AUC >=
  lgbm_nocat (delta >= 0) and ECE not exploding; fail closed if artifacts missing.

Eval slices:
  gender, stress_level, academic_work_impact, any-missing, daily-screen-missing,
  OOF p in (0.3, 0.7) hard band.

Known risks:
  Exact-value TE looks strong univariately and hijacks GBM early stopping.
  Residual in the hard band has ~0.50 column AUC (generator noise).
  HistGB on Kaggle sklearn 1.6.1 may cap trees without X_val.

Next experiment:
  After this surface lands: Kernel-run CatBoost/XGB nocat and ExtraTrees only
  if local smoke + promotion gates pass. Do not re-run dead-end TE / ratios.

Rollback or fallback:
  Keep old YAML + oof/<old>/. Submit submissions/lgbm_nocat.csv or
  blend_nocat.csv. Never retrain to roll back.
```
