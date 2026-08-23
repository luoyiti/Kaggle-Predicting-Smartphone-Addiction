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
| 2a | Missing flags | implemented | `add_missing_indicators`; `configs/lgbm_nocat_missflags.yaml`. `n_missing` dead-end on 80k (baseline < raw) |
| 2b | Ratios (screen/sleep, weekend/weekday, notif/open) | dead-end | `configs/baseline.yaml` vs `lgbm_raw` diagnostic −0.001 |
| 2c | Leisure vs work ratio | implemented | `add_leisure_work_ratio` + `configs/lgbm_nocat_leisure_work.yaml` |
| 2d | Pairwise interactions | implemented | `add_interactions` + `configs/lgbm_nocat_interactions.yaml` |
| 2e | Quantile bins as cats | implemented | `add_quantile_bins` + `configs/lgbm_nocat_bins.yaml` |
| 2f | Count of missings | dead-end | baseline `add_n_missing`; lost to raw |
| 2g | Coverage / strong3 / OR-score | dead-end | `lgbm_strong3_mean` diagnostic flat; LOG.md |
| 3a | LightGBM | scored-kernel | `lgbm_nocat` OOF **0.963771** |
| 3b | HistGB | scored-kernel | `histgb_nocat` OOF 0.962140 |
| 3c | XGBoost | scored-diag (raw); implemented nocat YAML | `xgb_raw` 80k 0.952; `configs/xgb_nocat.yaml` not Kernel-scored |
| 3d | CatBoost | implemented | trainer already in `train.py`; `configs/catboost_nocat.yaml`; optional pip extra |
| 3e | Logistic (regularized) | scored-diag | `logreg_raw` 80k 0.911; too weak to blend |
| 3e2 | Logistic nocat + missing flags | implemented | `configs/logreg_nocat.yaml` (linear path with missingness signal) |
| 3f | ExtraTrees | implemented | `extratrees` backend + `configs/extratrees_nocat.yaml` |
| 3g | MLP | skipped | Pipeline has no neural trainer; not a clean add vs GBMs already at 0.96 |
| 4 | Missingness as signal | implemented | indicators + n_missing; median impute only for linear/ET |
| 5a | Native cats | scored-diag | `lgbm_raw` keeps cats; nocat won |
| 5b | Drop cats | scored-kernel | `lgbm_nocat` |
| 5c | One-hot | implemented | logreg ColumnTransformer |
| 5d | Ordinal maps | implemented | `add_ordinal_cats` + `configs/lgbm_ordinal_cats.yaml` |
| 5e | Missing as explicit level | implemented | `encode_missing_as_category` + `configs/lgbm_cat_missing_level.yaml` |
| 6 | Calibration isotonic/platt | implemented | `scripts/calibrate_oof.py`; needs parent OOF (Kernel artifact) |
| 7a | Rank average | implemented | `blend_oof.py --method rank` |
| 7b | Grid / mean blend | scored-kernel | `blend_nocat` 0.963806 |
| 7c | AUC-weighted blend | implemented | `--method auc_weighted` + ensemble YAML |
| 7d | Logistic stacker | implemented | `scripts/stack_oof.py`; LOG already: nocat+TE stack **hurt** |
| 8 | Slice metrics | implemented | written into `metrics.json` when columns exist; CLI `eval_slices.py` |
| 9 | Error analysis → next experiment | implemented | `scripts/error_analysis.py` |
| 10 | Promotion gates vs baseline | implemented | `scripts/promote.py`, fail closed |
| 11 | Original 7500-row concat | skipped | LOG.md: different component dependence; leakage/shift |
| 12 | GPU LightGBM baseline | skipped | AGENTS.md: do not force baseline onto GPU |

## Exhaustion claim (local / config-driven)

All high-value **surfaces** that this repo’s YAML pipeline can reasonably express are implemented. Remaining work is **Kernel compute** on CatBoost/XGB/ExtraTrees nocat and post-hoc calibrate/stack using real OOF — not new feature families or trainers. Arithmetic extras on the nocat residual were already ~0 correlated in prior error analysis; new interaction/bin flags exist so that claim can be re-checked without writing code.
