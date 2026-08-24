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

## Iteration 2 — remaining GBDT / seed surfaces (this PR)

```text
Iteration: 2
Change: bag_seeds in train.py; histgb_nocat_moreiter; lgbm lowlr / extra_trees /
  seed43 / seed2026 / seedbag; catboost_raw; adversarial AUC in audit_tables;
  blend YAML for LGBM+CatBoost and seedbag+CatBoost
Why this mattered: The previous iteration left GBDT hypers, seed averaging,
  CatBoost-native-cats, and a train/test classifier unexpressed as YAML
Metric movement (NOT 5-fold, official 80k×3, seed 42 subsample):
  Control lgbm_nocat_diag80000 0.954317 (reproduced).
  seedbag 0.956035 (+0.00172). lowlr 0.954917 (+0.00060).
  catboost_nocat 0.954249; catboost_raw 0.954273.
  xgb_nocat 0.952417. histgb moreiter 0.953986 (flat vs histgb_nocat).
  LightGBM extra_trees 0.939472 (−0.015).
  Best diagnostic blend: 0.70 seedbag + 0.30 CatBoost = 0.956368.
  LGBM+CatBoost 0.50/0.50 = 0.955326.
  Full Kernel 5-fold unchanged (lgbm_nocat 0.963771 / blend_nocat 0.963806).
Slice / shift: adversarial is_test AUC 0.559 (warn; missing-rate gaps 2–3%).
False positives / negatives: unchanged hard-band story; seedbag is the ensemble
  the error-analysis loop asked for
Unexpected errors: LightGBM extra_trees collapsed; CatBoost cats ≈ nocat
Decision: Next Kernel job is lgbm_nocat_seedbag, then catboost_nocat if extras
  are on the image. Do not 5-fold extra_trees / moreiter-on-sklearn≥1.7 /
  catboost_raw. Promote.py correctly refused the diagnostic seedbag.
Tradeoff accepted: 3× Kernel wall-clock for seedbag; CatBoost still an extra pip
Lesson captured: Different CV seeds move 80k OOF more than new arithmetic features
Regression added: parse_bag_seeds + seedbag smoke; adversarial unit tests;
  optional xgb/catboost smokes (importorskip)
Debt created: No full 5-fold seedbag/CatBoost metrics.json yet
Next iteration: Kernel 5-fold seedbag (and optional CatBoost). No new feature flags.
```

## How to append

After every Kernel or honest diagnostic, add an Iteration block. Do not edit
historical metric numbers. Point at `oof/<name>/metrics.json`.
