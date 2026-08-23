# Observation ledger — S6E8

Cumulative mle-workflow ledger. Competition scores appear only when
`oof/<name>/metrics.json` exists from a non-diagnostic run. Diagnostic rows
are labeled. Synthetic smoke is never a score.

## Iteration 0 — repo baseline (pre-this-work)

```text
Iteration: 0
Change: Config-driven LightGBM CPU pipeline; later nocat / HistGB / blend / TE
Why this mattered: Reproducible Kernel training without notebooks
Metric movement: Kernel lgbm_nocat OOF 0.963771; blend_nocat 0.963806
Slice movement: Hard band p∈(0.3,0.7) ~93k rows, AUC 0.641, labels 50%
False positives / negatives: Concentrated in the hard band; bulk usage is easy
Unexpected errors: Exact-value TE univariate-strong, GBM-harmful
Decision: Promote lgbm_nocat as default single model; optional 0.85/0.15 blend
Tradeoff accepted: Drop categoricals; ignore ratio engineering
Lesson captured: Identity in notif/app_opens is already inside deep LGBM
Regression added: Fold-safe TE unit tests; HistGB sklearn 1.6 X_val shim
Debt created: No CatBoost/ExtraTrees YAML; no automated slice/calibration/gates
Next iteration: Close those surface gaps without mutating old configs
```

## Iteration 1 — mle-workflow surface (this PR)

```text
Iteration: 1
Change: Contracts + remaining YAML surfaces; 80k×3 real-data ranking of new flags
Why this mattered: Close high-value gaps in code/config and falsify them cheaply
Metric movement (NOT 5-fold): lgbm_nocat_diag80000 reproduced 0.954317.
  missflags +0.000144 (noise). interactions −0.00135 (stop).
  ExtraTrees 0.925, grid weight vs LGBM = 1.0/0. Full Kernel 5-fold unchanged.
Slice movement: Real 80k hard band n=12219 AUC 0.637; gender slices all ~0.954
False positives / negatives: Residual column AUC max 0.516 in the hard band
Unexpected errors: Platt calibration raised ECE 0.0042 → 0.0348
Decision: Keep promoting lgbm_nocat / blend_nocat. Do not 5-fold missflags,
  interactions, bins, ordinal, NA-level, ExtraTrees, or linear stacks.
Tradeoff accepted: CatBoost/XGB remain optional extras (not installed here)
Lesson captured: LGBM probabilities are already well calibrated (ECE 0.0042)
Regression added: join_oof_with_train (target-name collision); promotion tests
Debt created: catboost_nocat / xgb_nocat still need a Kernel if we want a third GBDT
Next iteration: Optional Kernel CatBoost/XGB nocat only. No new arithmetic features.
```

## How to append

After every Kernel or honest diagnostic, add an Iteration block. Do not edit
historical metric numbers. Point at `oof/<name>/metrics.json`.
