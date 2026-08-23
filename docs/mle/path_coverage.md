# Path coverage matrix — S6E8

Status key:

- **implemented**: code + YAML exist and are runnable
- **scored-kernel**: full-data 5-fold `metrics.json` exists
- **scored-diag**: 80k×3 or full-data fewer-fold ranking only
- **dead-end**: implemented or tested; do not spend more Kernel budget
- **blocked**: needs Kernel / extra dependency / real OOF of parents
- **skipped**: written evidence that the path is out of policy

| # | Path | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Data contract & leakage audit | implemented | `s6e8/contracts.py`, `scripts/audit_data.py`; EDA report already scored id leak / PSI on real tables |
| 2a | Missing flags | scored-diag dead-end | 80k missflags 0.954461 vs nocat 0.954317 (+0.000144, noise) |
| 2b | Ratios (screen/sleep, weekend/weekday, notif/open) | dead-end | `configs/baseline.yaml` vs `lgbm_raw` diagnostic −0.001 |
| 2c | Leisure vs work ratio | scored-diag dead-end | 80k 0.954284 vs 0.954317 |
| 2d | Pairwise interactions | scored-diag dead-end | 80k **0.952966 (−0.00135)** |
| 2e | Quantile bins as cats | scored-diag dead-end | 80k 0.954196 |
| 2f | Count of missings | dead-end | baseline `add_n_missing`; lost to raw |
| 2g | Coverage / strong3 / OR-score | dead-end | `lgbm_strong3_mean` diagnostic flat; LOG.md |
| 3a | LightGBM | scored-kernel | `lgbm_nocat` OOF **0.963771** |
| 3b | HistGB | scored-kernel | `histgb_nocat` OOF 0.962140 |
| 3c | XGBoost | scored-diag (raw); nocat YAML blocked on extra dep | `xgb_raw` 80k 0.952; `xgb_nocat.yaml` not run (xgboost not installed) |
| 3d | CatBoost | implemented / blocked on extra dep | `catboost_nocat.yaml`; catboost not installed in this environment |
| 3e | Logistic (regularized) | scored-diag | `logreg_raw` 80k 0.911; too weak to blend |
| 3e2 | Logistic nocat + missing flags | scored-diag dead-end | 80k 0.913096; stack/blend with LGBM hurts |
| 3f | ExtraTrees | scored-diag dead-end | 80k 0.925; grid blend vs LGBM picks weight 1.0/0.0 |
| 3g | MLP | skipped | Pipeline has no neural trainer; not a clean add vs GBMs already at 0.96 |
| 4 | Missingness as signal | implemented | indicators + n_missing; median impute only for linear/ET |
| 5a | Native cats | scored-diag | `lgbm_raw` keeps cats; nocat won |
| 5b | Drop cats | scored-kernel | `lgbm_nocat` |
| 5c | One-hot | implemented | logreg ColumnTransformer |
| 5d | Ordinal maps | scored-diag dead-end | 80k 0.954108 vs nocat 0.954317 |
| 5e | Missing as explicit level | scored-diag dead-end | 80k 0.954091 |
| 6 | Calibration isotonic/platt | scored-diag dead-end for AUC | LGBM ECE already 0.0042; isotonic −0.00029 AUC; Platt ECE 0.035 |
| 7a | Rank average | implemented | `blend_oof.py --method rank` |
| 7b | Grid / mean blend | scored-kernel | `blend_nocat` 0.963806 |
| 7c | AUC-weighted blend | scored-diag dead-end vs weak partners | LGBM+logreg 80k 0.945653 (hurts) |
| 7d | Logistic stacker | scored-diag dead-end vs logreg | 80k stack 0.953912 vs LGBM 0.954317; nocat+TE stack already hurt |
| 8 | Slice metrics | implemented | written into `metrics.json` when columns exist; CLI `eval_slices.py` |
| 9 | Error analysis → next experiment | implemented | `scripts/error_analysis.py` |
| 10 | Promotion gates vs baseline | implemented | `scripts/promote.py`, fail closed |
| 11 | Original 7500-row concat | skipped | LOG.md: different component dependence; leakage/shift |
| 12 | GPU LightGBM baseline | skipped | AGENTS.md: do not force baseline onto GPU |

## Exhaustion claim (local / config-driven)

All high-value **surfaces** that this repo’s YAML pipeline can reasonably express are implemented, and the new arithmetic / encoding / ExtraTrees / linear-stack paths were **falsified on real 80k×3 data** (same protocol as the existing ranking log). Remaining optional work is a Kernel job for CatBoost or XGBoost nocat (optional extras, not installed here) — not new feature families. Hard-band residual column AUC is 0.516; error analysis says stop_or_ensemble. Do not 5-fold missflags for a +0.000144 that sits inside fold std.
