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
Change: Contracts, extra feature families, ExtraTrees, CatBoost/XGB nocat YAML,
  slice metrics, isotonic/platt calibration, rank/AUC-weighted/logistic stack,
  promotion gates (fail closed), data-contract audit, error analysis → next YAML
Why this mattered: Modeling paths were notes or partial; Kernel cannot run
  a path that has no trainer/config
Metric movement: No new full-data 5-fold Kernel in this iteration.
  Synthetic smoke only proves the pipeline runs.
Slice movement: Slice helper now writes per-cohort AUC whenever train columns exist
False positives / negatives: Error-analysis clusters hard-band / missingness /
  cat slices and emits a falsifiable next experiment (or “stop”)
Unexpected errors: (filled after smoke/audit)
Decision: Keep lgbm_nocat as the promotion baseline. New models are candidates
  only after Kernel OOF and scripts/promote.py pass
Tradeoff accepted: ExtraTrees needs median impute (no native NaN). CatBoost
  stays an optional extra dependency, not in default requirements.txt
Lesson captured: Exhaustion means the *surface* is implemented and dead-ends
  are documented; Kernel scores remain the only competition evidence
Regression added: Unit tests for transforms, metrics, ensemble, gates, audit
Debt created: CatBoost/XGB/ExtraTrees full 5-fold still Kernel-only
Next iteration: Kernel jobs for catboost_nocat / xgb_nocat if smoke is green;
  do not re-open TE-in-GBM or ratio blocks
```

## How to append

After every Kernel or honest diagnostic, add an Iteration block. Do not edit
historical metric numbers. Point at `oof/<name>/metrics.json`.
