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
| 1 | Data contract & leakage audit | implemented | `s6e8/contracts.py`, `scripts/audit_data.py`; real tables: schema OK, id overlap 0, id-vs-label 0.5007, PSI ~1e-5, adversarial AUC **0.559** (warn, missing-rate gaps) |
| 2a | Missing flags | scored-diag dead-end | 80k missflags 0.954461 vs nocat 0.954317 (+0.000144, noise) |
| 2b | Ratios (screen/sleep, weekend/weekday, notif/open) | dead-end | `configs/baseline.yaml` vs `lgbm_raw` diagnostic −0.001 |
| 2c | Leisure vs work ratio | scored-diag dead-end | 80k 0.954284 vs 0.954317 |
| 2d | Pairwise interactions | scored-diag dead-end | 80k **0.952966 (−0.00135)** |
| 2e | Quantile bins as cats | scored-diag dead-end | 80k 0.954196 |
| 2f | Count of missings | dead-end | baseline `add_n_missing`; lost to raw |
| 2g | Coverage / strong3 / OR-score | dead-end | `lgbm_strong3_mean` diagnostic flat; LOG.md |
| 3a | LightGBM nocat | scored-kernel | `lgbm_nocat` OOF **0.963771** |
| 3b | HistGB nocat | scored-kernel | `histgb_nocat` OOF 0.962140 (Kernel sklearn 1.6.1 capped at 500) |
| 3b2 | HistGB more trees | scored-diag dead-end here | 80k max_iter=2000 **0.953986**, best_iter ~530–542 on sklearn 1.9 with X_val — same as histgb_nocat 80k 0.953985. Optional only on Kernel images that still cap at 500 |
| 3c | XGBoost nocat | scored-diag | 80k **0.952417** (−0.0019 vs LGBM); grid blend 0.75/0.25 → 0.954512 |
| 3d | CatBoost nocat | scored-diag | 80k **0.954249** (tied with LGBM); grid 0.50/0.50 vs LGBM → **0.955326** |
| 3d2 | CatBoost + native cats | scored-diag dead-end vs nocat | `catboost_raw` 80k 0.954273 |
| 3e | Logistic (regularized) | scored-diag | `logreg_raw` 80k 0.911; too weak to blend |
| 3e2 | Logistic nocat + missing flags | scored-diag dead-end | 80k 0.913096; stack/blend with LGBM hurts |
| 3f | ExtraTrees (sklearn) | scored-diag dead-end | 80k 0.925; grid blend weight vs LGBM = **1.0 / 0.0** |
| 3g | MLP | skipped | Pipeline has no neural trainer; not a clean add vs GBMs already at 0.96 |
| 3h | LightGBM extra_trees | scored-diag dead-end | 80k **0.939472 (−0.015)** |
| 3i | LightGBM lr=0.02 | scored-diag | 80k **0.954917 (+0.00060)**; corr 0.996 with default LGBM; grid 0.25/0.75 → 0.954991 |
| 3j | LightGBM 3-seed bag | scored-diag (**Kernel next**) | `lgbm_nocat_seedbag` 80k **0.956035 (+0.00172)**; members 0.954317 / 0.954575 / 0.954699 |
| 4 | Missingness as signal | implemented | indicators + n_missing; median impute only for linear/ET |
| 5a | Native cats | scored-diag | `lgbm_raw` keeps cats; nocat won; CatBoost raw ≈ nocat |
| 5b | Drop cats | scored-kernel | `lgbm_nocat` |
| 5c | One-hot | implemented | logreg ColumnTransformer |
| 5d | Ordinal maps | scored-diag dead-end | 80k 0.954108 vs nocat 0.954317 |
| 5e | Missing as explicit level | scored-diag dead-end | 80k 0.954091 |
| 6 | Calibration isotonic/platt | scored-diag dead-end for AUC | LGBM ECE already 0.0042; isotonic −0.00029 AUC; Platt ECE 0.035 |
| 7a | Rank average | scored-diag | LGBM+HistGB rank 80k 0.954880 ≈ grid 0.954896 |
| 7b | Grid / mean blend | scored-kernel | `blend_nocat` 0.963806 |
| 7c | AUC-weighted blend | scored-diag dead-end vs weak partners | LGBM+logreg 80k 0.945653 (hurts) |
| 7d | Logistic stacker | scored-diag dead-end vs logreg | 80k stack 0.953912 vs LGBM 0.954317 |
| 7e | LGBM+CatBoost grid | scored-diag | 80k **0.955326** |
| 7f | Seedbag+CatBoost grid | scored-diag | 80k **0.956368** (best diagnostic blend) |
| 7g | Three-GBDT grid | scored-diag | LGBM+CB+HistGB 80k 0.955574 < seedbag |
| 8 | Slice metrics | implemented | written into `metrics.json` when columns exist; CLI `eval_slices.py` |
| 9 | Error analysis → next experiment | implemented | `scripts/error_analysis.py` |
| 10 | Promotion gates vs baseline | implemented | `scripts/promote.py`, fail closed (rejected diagnostic seedbag despite +0.0017) |
| 11 | Original 7500-row concat | skipped | LOG.md: different component dependence; leakage/shift |
| 12 | GPU LightGBM baseline | skipped | AGENTS.md: do not force baseline onto GPU |

## Exhaustion claim (local / config-driven)

Every high-value **surface** this YAML pipeline can reasonably express is implemented (trainers, flags, seed bag, CatBoost-with-cats, HistGB moreiter, adversarial audit, blends, gates). Arithmetic / encoding / ExtraTrees / linear / LightGBM-`extra_trees` / CatBoost-cats paths were **falsified on real 80k×3**.

The remaining **compute** (not missing code) is a Kernel 5-fold of `lgbm_nocat_seedbag` and optionally `catboost_nocat` + grid blend. Those 80k lifts (+0.0017 seedbag, +0.0010 LGBM+CB) are ranking-only and must not be quoted as competition scores. Do not 5-fold missflags, interactions, extra_trees, or TE.

Hard-band residual column AUC is 0.516; error analysis says stop_or_ensemble. Seed bagging is that ensemble step.
