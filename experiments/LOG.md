# S6E8 experiment log

Diagnostic rows use a **stratified 80,000-row train subsample**, 3 folds, seed 42.
They rank hypotheses. They are **not** competition scores. A result counts only when
`diagnostic` is false, `n_splits=5`, full train, and `oof/<name>/metrics.json` exists.

Full-data screens with `--n-splits 3` (no row subsample) are marked `*_diag`. They
are stronger than 80k ranking runs but still **not** official 5-fold scores.

## Full 5-fold (Kaggle Kernels, 691,369 train, seed 42)

Source: `oof/<name>/metrics.json` from kernels
[yitiluo/s6e8-lgbm-nocat](https://www.kaggle.com/code/yitiluo/s6e8-lgbm-nocat) and
[yitiluo/s6e8-histgb-nocat](https://www.kaggle.com/code/yitiluo/s6e8-histgb-nocat).
Feature set is the 9 numerics including `notifications_per_day` and `app_opens_per_day`.
The three categoricals are dropped. No coverage features.

| experiment | hypothesis | change | CV AUC | fold std | runtime | conclusion | next step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lgbm_nocat | Drop only the three categoricals | LightGBM, 9 numeric cols, 5-fold | **0.963771** | 0.000593 | 715s | Primary single model. Folds 0.96293–0.96468. Best iter 1549–1912. | Default submit candidate |
| histgb_nocat | Same cols, second tree family | sklearn HistGB, max_iter=500 | 0.962140 | 0.000458 | 178s | −0.00163 vs LGBM. Kaggle sklearn 1.6.1 has no `X_val`; hit the 500-iter cap on every fold. | Optional: more trees / sklearn≥1.7 later |
| blend_nocat | Complementary tree errors | Grid 0.85 LGBM + 0.15 HistGB | **0.963806** | — | — | +0.000035 vs LGBM. Pearson 0.992. Tiny, consistent lift. | Prefer this CSV if submitting a blend |

Did **not** submit to the leaderboard. Did **not** add coverage features. Did **not** drop notifications/app_opens.

## Full-data screens (691,369 train, fewer folds — not a 5-fold score)

Exact-value TE must not be judged on an 80k subsample: repeat frequency collapses and
unseen-value rates are fake. These rows use the **full train**.

| experiment | hypothesis | change | CV AUC | fold std | runtime | conclusion | next step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lgbm_nocat_exact_te_v1_diag | Exact numeric values carry playground identity that raw splits miss | `lgbm_nocat` + fold-safe LOO TE on notif/app/sleep/age/gaming/work; 3-fold full data | 0.923636 | 0.000601 | 17s | **−0.040 vs lgbm_nocat.** Unseen=0 (not leakage). Best iter 20–31. `age_exact_te` stole 17% gain; predictions compressed. Identity is real univariately (notif TE 0.76 vs raw 0.49) but already inside deep LGBM. **Stop TE-in-GBM. No 5-fold.** | Do not inject lookup TE into LightGBM. Residual of nocat is not in generator arithmetic either |

## Diagnostic ranking, this PR (real 80k rows × 3-fold, seed 42 — not a 5-fold score)

Run on official Kaggle `train.csv` (691,369 rows subsampled to 80k). Control `lgbm_nocat_diag80000` **reproduced 0.954317**, matching the previous log exactly.

| experiment | hypothesis | change | CV AUC | vs nocat | conclusion |
| --- | --- | --- | --- | --- | --- |
| lgbm_nocat_diag80000 | Control | 9 raw numerics | **0.954317** | 0 | Reproduced |
| lgbm_nocat_missflags_diag80000 | Missingness is signal | per-column NA flags | 0.954461 | +0.000144 | Within fold std (~0.0003). Not a 5-fold candidate |
| lgbm_nocat_leisure_work_diag80000 | Leisure vs work ratio | `add_leisure_work_ratio` | 0.954284 | −0.000033 | Dead-end |
| lgbm_nocat_interactions_diag80000 | Explicit products | pairwise usage products | 0.952966 | **−0.00135** | Harmful. Stop |
| lgbm_nocat_bins_diag80000 | Quantile stumps | 5-qbins of daily/sleep/weekend | 0.954196 | −0.00012 | Dead-end |
| lgbm_ordinal_cats_diag80000 | Ordered cat maps | ordinal + drop strings | 0.954108 | −0.00021 | Dead-end vs nocat |
| lgbm_cat_missing_level_diag80000 | `__NA__` cat level | keep cats, fill missing | 0.954091 | −0.00023 | Dead-end vs nocat |
| logreg_nocat_diag80000 | Linear + miss flags | logreg nocat | 0.913096 | −0.041 | Still too weak (raw logreg was 0.911) |
| extratrees_nocat_diag80000 | Diversity vs GBM | ExtraTrees, median impute | 0.925209 | −0.029 | Grid blend weight vs LGBM = **1.0 / 0.0** |
| lgbm_nocat_diag80000_cal_isotonic | Isotonic on OOF | inner-CV isotonic | 0.954025 | −0.00029 | ECE 0.0042→0.0035; AUC drop. Skip |
| lgbm_nocat_diag80000_cal_platt | Platt on OOF | inner-CV logistic | 0.954272 | −0.000045 | ECE **worsens** 0.0042→0.0348. Stop |
| stack_nocat_logreg_diag80000 | Logistic stack | LGBM+logreg OOF | 0.953912 | −0.00041 | Hurts |
| blend_nocat_logreg_diag80000 | AUC-weighted | LGBM+logreg | 0.945653 | −0.0087 | Hurts (logreg too weak) |

Real-data contract audit (full 691,369 / 296,302): schema OK, id overlap 0, id-vs-label AUC 0.5007, `daily < social+gaming+work` rows **0**, numeric PSI ~1e-5, positive rate **0.7094**. Missing-rate train/test gaps of 2–3% on a few columns (warn, not error).

Error analysis on `lgbm_nocat_diag80000`: hard band n=12,219, AUC 0.637, max residual-column AUC 0.516 → **stop_or_ensemble** (same story as full-data ~93k / 0.641).

CatBoost and XGBoost nocat YAML are implemented but **not scored here** (`xgboost`/`catboost` not installed). Kernel-only.

## Diagnostic ranking, remaining GBDT / seed surfaces (real 80k × 3-fold, seed 42 subsample — not a 5-fold score)

Same protocol as the previous 80k table. Control `lgbm_nocat_diag80000` still **0.954317**. Packages `xgboost==3.4.1` and `catboost==1.2.10` were installed for this ranking only (not added to `requirements.txt`).

| experiment | hypothesis | change | CV AUC | vs nocat | conclusion |
| --- | --- | --- | --- | --- | --- |
| lgbm_nocat_diag80000 | Control | 9 raw numerics | **0.954317** | 0 | Reproduced |
| lgbm_nocat_seedbag_diag80000 | Mean of 3 CV seeds | bag_seeds 42/43/2026 | **0.956035** | **+0.00172** | Members 0.954317 / 0.954575 / 0.954699. Best remaining Kernel job |
| lgbm_nocat_lowlr_diag80000 | Slower schedule | lr=0.02 | 0.954917 | +0.00060 | ~2× fold std; corr 0.996 with default LGBM |
| catboost_nocat_diag80000 | Ordered boosting | CatBoost, nocat | 0.954249 | −0.00007 | Tied with LGBM; useful blend partner (corr 0.986) |
| catboost_raw_diag80000 | CatBoost native cats | cats kept | 0.954273 | −0.00004 | Cats still noise |
| xgb_nocat_diag80000 | Third GBDT on nocat | XGBoost hist | 0.952417 | −0.00190 | Same gap as xgb_raw |
| histgb_nocat_moreiter_diag80000 | Uncap HistGB | max_iter=2000 | 0.953986 | −0.00033 | best_iter 531–542 on sklearn 1.9; flat vs histgb_nocat 80k |
| lgbm_nocat_extra_trees_diag80000 | Random splits | extra_trees=true | 0.939472 | **−0.015** | Harmful. Stop |
| blend_lgbm_cb_diag80000 | CB diversity | grid 0.50/0.50 | **0.955326** | +0.00101 | Better 80k blend than LGBM+HistGB |
| blend_seedbag_cb_diag80000 | Seedbag + CB | grid 0.70/0.30 | **0.956368** | +0.00205 | Best diagnostic blend |
| blend_three_gbdt_diag80000 | LGBM+CB+HistGB | grid 0.3/0.4/0.3 | 0.955574 | +0.00126 | Below seedbag alone |
| blend_lgbm_histgb_moreiter_diag80000 | Uncapped HistGB partner | grid 0.55/0.45 | 0.954896 | +0.00058 | Same as old histgb blend |
| blend_rank_lgbm_histgb_diag80000 | Rank vs grid | rank 0.5/0.5 | 0.954880 | +0.00056 | Rank ≈ grid |
| blend_lgbm_xgb_diag80000 | XGB partner | grid 0.75/0.25 | 0.954512 | +0.00020 | Tiny |
| blend_lgbm_lowlr_diag80000 | Two LGBM schedules | grid 0.25/0.75 | 0.954991 | +0.00067 | Almost just lowlr |

Adversarial train/test AUC (40k subsample, numeric + missing flags): **0.5585** (warn ≥ 0.55). Matches the known 2–3% missing-rate gaps; not a reason to add shift features.

`scripts/promote.py --candidate lgbm_nocat_seedbag_diag80000 --baseline lgbm_nocat_diag80000` failed closed (`diagnostic runs cannot be promoted`) even though delta was +0.00172.

## Diagnostic ranking (80k rows, 3-fold, seed 42 — not a leaderboard number)

| experiment | hypothesis | change | CV AUC | fold std | runtime | conclusion | next step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_diag80000 | Current ratios + n_missing are a fine GBM start | LightGBM, 17 cols | 0.953145 | 0.000256 | 18s | Control. Engineering is not free. | Compare raw |
| lgbm_raw_diag80000 | Ratios dilute raw usage | All engineering off | **0.954115** | 0.000345 | 18s | **+0.001 vs baseline.** Drop the ratio/leisure/n_missing block. | Keep raw as default feature set |
| lgbm_strong3_mean_diag80000 | Skip-NA mean of daily/weekend/social fills missing daily | raw + `strong3_row_mean` | 0.954120 | 0.000293 | 18s | Flat vs raw. Worse on the daily-missing slice (0.9149 vs 0.9153). Trees already use surrogates. **Stop.** | Do not ship this feature |
| histgb_raw_diag80000 | Different tree family, same raw cols | sklearn HistGB | 0.954050 | 0.000357 | 12s | Matches LightGBM. Corr 0.989. | Blend partner |
| xgb_raw_diag80000 | Third boosted-tree family | XGBoost hist | 0.952247 | 0.000398 | 28s | −0.002 vs LGBM. Not worth extra complexity at these params. | Stop unless GPU/tune later |
| lgbm_usage_core_diag80000 | Original-noise columns can be dropped | Drop cats + notifications + app_opens | 0.938491 | 0.000541 | 9s | **−0.016. Failed.** Notifications/app_opens are weak univariate but high GBM gain on playground. | Split the drop |
| lgbm_nocat_diag80000 | Only categoricals are noise | Drop gender/stress/academic | **0.954317** | 0.000303 | 15s | Small gain vs raw. Best single model in this ranking. | Full 5-fold on Kaggle |
| logreg_raw_diag80000 | Surface is nearly linear in usage | Logistic + median impute | 0.911040 | 0.001395 | 1s | Far below trees. Corr 0.87 with LGBM, but too weak to help a mean blend. | Not a stacker at this strength |
| histgb_nocat_diag80000 | HistGB on the nocat feature set | HistGB + drop cats | 0.953985 | 0.000313 | 10s | Matches LGBM within 0.0003. | Blend with lgbm_nocat |
| blend_nocat_diag80000 | Complementary tree errors | Grid 0.55 LGBM + 0.45 HistGB | **0.954887** | — | — | +0.00057 vs best single. Corr 0.990 — small, consistent lift. | Repeat on full 5-fold OOF |

## Answers so far

**What actually determines `addicted_label`?**

- Original 7,500-row source: label is exactly `addiction_level ∈ {Moderate, Severe}`. A depth-3 tree of `daily_screen ≳ 8` or `social_media ≳ 4` already has ~0.99 AUC. Notifications, app opens, cats, gaming, work are ~0.50 there.
- Playground keeps the ~71% positive rate and the usage ranking, but **smooths** the hard rule (continuous daily AUC 0.890 >> stump 0.81) and **masks** ~14% of daily_screen (MCAR vs label).
- Playground also **entangles notifications and app_opens with the label via exact-value identity**, not a monotone ranking (raw notif AUC 0.49, fold-safe exact-value TE 0.76). Deep LightGBM already harvests that identity: dropping the columns costs ~0.016, but *explicit* TE does not add ranking on top of `lgbm_nocat`.
- Categoricals remain noise on both tables. Safe to drop.
- `id` is a sequential split, not a leak. Train/test value drift is negligible.

**What model fits this generator?**

- Trees, not linear models (logreg 0.911 vs GBM 0.954 on the same 80k protocol).
- LightGBM ≳ HistGB > XGBoost at the default-ish params used here.
- Explicit coverage features (`strong3_row_mean`) and the original OR-score do not beat native missing handling.
- Fold-safe exact-value TE is a real univariate signal and a harmful GBM feature (early-stopping hijack). Do not put lookup TE into `lgbm_nocat`.

**Is CV trustworthy?**

- Fold std ≈ 0.0003 on 80k×3 and 0.0006 on full 5-fold. Rankings held: `lgbm_nocat` still beats HistGB; blend still helps, but the full-data lift is much smaller.
- The TE screen’s −0.040 is ~70× fold std. Not a noisy CV flip. Unseen-value share was 0, so this is overfit of lookup features, not val→train leakage.

**Complementary errors?**

- On 80k diagnostics, LGBM vs HistGB Pearson ≈ 0.99 and grid blend added ~0.0006.
- On full 5-fold OOF the same pair is Pearson 0.992. Grid 0.85/0.15 adds only **+0.000035**. HistGB is a weaker partner here because it capped at 500 trees on sklearn 1.6.1 without `X_val`.
- Do not average in failed ablations (`usage_core`, `lgbm_nocat_exact_te_v1`).
- Grid-blending `lgbm_nocat` OOF with fold-safe notif/app TE is ≤ **+0.00002**. Logistic stack of nocat+TE **hurts** (−0.0007).

**Where is `lgbm_nocat` still wrong?**

- ~93k-row hard band (OOF p ∈ (0.3, 0.7)): 50.0% positive, OOF AUC only 0.641.
- Inside that band **every raw column has AUC ≈ 0.50**. `other_screen`, `component_sum`, weekend−daily, value-frequency, and fractional parts have residual correlation ≈ 0 with `y − p_nocat`.
- Remaining errors look like generator noise / conflicting usage, not a missing arithmetic feature.

## Full 5-fold jobs (done)

1. `configs/lgbm_nocat.yaml` (CPU) — OOF **0.963771**.
2. `configs/histgb_nocat.yaml` (CPU) — OOF 0.962140.
3. `python scripts/blend_oof.py --experiments lgbm_nocat histgb_nocat --method grid --name blend_nocat` — OOF **0.963806**.

Submission CSVs are local/Kaggle artifacts (`submissions/lgbm_nocat.csv`, `submissions/blend_nocat.csv`). Do not `kaggle competitions submit` unless explicitly asked.

## Full-data screens (done, no 5-fold follow-up)

1. `configs/lgbm_nocat_exact_te_v1.yaml` with `--n-splits 3` — OOF 0.923636. Report: `reports/lgbm_nocat_exact_te_v1.html`.

## Not worth more budget

- Baseline ratio engineering.
- `strong3_row_mean` / `strong3_row_max` / `or_usage_score` for GBMs.
- Dropping notifications or app_opens.
- Mixing original 7,500 rows into train (component dependence differs: original `daily` ⟂ social+gaming+work; playground never violates `daily ≥ sum`).
- Stacking extra coverage features on top of `lgbm_nocat`.
- Fold-safe exact-value target encoding **inside** LightGBM (lookup hijacks early stopping; already in the deep tree).
- Mean-blending nocat with OOF exact-value TE (≤ +0.00002).
- `other_screen` / `component_sum` / `weekend − daily` / value-frequency / fractional parts as extra GBM columns (residual vs nocat ≈ 0).
- Another LGBM+HistGB probability blend pass.
- LightGBM `extra_trees` (80k 0.939).
- CatBoost with categoricals kept (`catboost_raw` tied with nocat).

## Full 5-fold jobs (queued — not scored)

1. `configs/lgbm_nocat_seedbag.yaml` — 80k +0.00172; ~3× `lgbm_nocat` wall-clock.
2. Optional `configs/catboost_nocat.yaml` then grid blend with seedbag.

## Modeling surface added for Kernel follow-up

Configs/scripts exist so remaining families can run without new trainers. **Competition scores** still require non-diagnostic `oof/<name>/metrics.json`.

| name | primary variable | 80k×3 (this VM) | Kernel? |
| --- | --- | --- | --- |
| `lgbm_nocat_seedbag` | `experiment.bag_seeds` | **0.956035** | **Yes — next 5-fold** |
| `catboost_nocat` | model.name catboost | 0.954249 | Yes, if catboost on image |
| `lgbm_nocat_lowlr` | learning_rate 0.02 | 0.954917 | Optional; highly correlated with default |
| `xgb_nocat` | model.name xgboost | 0.952417 | Low priority |
| `histgb_nocat_moreiter` | max_iter 2000 | 0.953986 | Only if Kernel sklearn still caps at 500 |
| `lgbm_nocat_extra_trees` | extra_trees | 0.939472 | **No** |
| `catboost_raw` | keep cats | 0.954273 | **No** (tied with nocat) |
| `lgbm_nocat_seed43` / `seed2026` | extra seeds | (inside seedbag) | Alternative to one 3× job |
| rank / auc_weighted / LGBM+CB grid | ensemble | see table | After parent OOFs exist |

```bash
python scripts/train.py --config configs/lgbm_nocat_seedbag.yaml
python scripts/train.py --config configs/catboost_nocat.yaml
python scripts/blend_oof.py --experiments lgbm_nocat_seedbag catboost_nocat --method grid --name blend_seedbag_catboost --write-experiment-record
python scripts/promote.py --candidate lgbm_nocat_seedbag --baseline lgbm_nocat
```

Human report: `reports/mle_modeling_report.html`. Contracts: `docs/mle/`.

