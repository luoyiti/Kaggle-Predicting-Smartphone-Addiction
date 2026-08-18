# S6E8 experiment log

Diagnostic rows use a **stratified 80,000-row train subsample**, 3 folds, seed 42.
They rank hypotheses. They are **not** competition scores. A result counts only when
`diagnostic` is false, `n_splits=5`, full train, and `oof/<name>/metrics.json` exists.

Full-data screens with `--n-splits 3` (no row subsample) are marked `*_diag`. They
are stronger than 80k ranking runs but still **not** official 5-fold scores.

## Full 5-fold (Kaggle Kernels, 691,369 train, seed 42)

Source: `oof/<name>/metrics.json` from kernels
[yitiluo/s6e8-lgbm-nocat](https://www.kaggle.com/code/yitiluo/s6e8-lgbm-nocat) and
[yitiluo/s6e8-histgb-nocat](https://www.kaggle.com/code/yitiluo/s6e8-histgb-nocat), plus
[yitiluo/s6e8-histgb-nocat-long-v1](https://www.kaggle.com/code/yitiluo/s6e8-histgb-nocat-long-v1).
Feature set is the 9 numerics including `notifications_per_day` and `app_opens_per_day`.
The three categoricals are dropped. No coverage features.

| experiment | hypothesis | change | CV AUC | fold std | runtime | conclusion | next step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lgbm_nocat | Drop only the three categoricals | LightGBM, 9 numeric cols, 5-fold | **0.963771** | 0.000593 | 715s | Primary single model. Folds 0.96293–0.96468. Best iter 1549–1912. | Default submit candidate |
| histgb_nocat | Same cols, second tree family | sklearn HistGB, max_iter=500 | 0.962140 | 0.000458 | 178s | −0.00163 vs LGBM. Kaggle sklearn 1.6.1 has no `X_val`; hit the 500-iter cap on every fold. | Optional: more trees / sklearn≥1.7 later |
| blend_nocat | Complementary tree errors | Grid 0.85 LGBM + 0.15 HistGB | **0.963806** | — | — | +0.000035 vs LGBM. Pearson 0.992. Tiny, consistent lift. | Prefer this CSV if submitting a blend |
| histgb_nocat_long_v1 | The 500-tree HistGB is undertrained | Same HistGB with max_iter=1400 only | 0.963468 | 0.000465 | 410s | **+0.001329 vs capped HistGB.** Best iter 1117, 1400, 1226, 1146, 1215. Undertraining confirmed; one fold still capped. | Use as the stronger blend partner |
| blend_nocat_long_v1 | A stronger HistGB should contribute more complementary ranking | Grid 0.60 LGBM + 0.40 long HistGB | **0.964087** | — | — | **New best: +0.000316 vs LGBM and +0.000281 vs old blend.** Pearson 0.9937. Improved all five folds. | Default submission candidate; no submit yet |

Did **not** submit to the leaderboard. Did **not** add coverage features. Did **not** drop notifications/app_opens.

## Full-data screens (691,369 train, fewer folds — not a 5-fold score)

Exact-value TE must not be judged on an 80k subsample: repeat frequency collapses and
unseen-value rates are fake. These rows use the **full train**.

| experiment | hypothesis | change | CV AUC | fold std | runtime | conclusion | next step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lgbm_nocat_exact_te_v1_diag | Exact numeric values carry playground identity that raw splits miss | `lgbm_nocat` + fold-safe LOO TE on notif/app/sleep/age/gaming/work; 3-fold full data | 0.923636 | 0.000601 | 17s | **−0.040 vs lgbm_nocat.** Unseen=0 (not leakage). Best iter 20–31. `age_exact_te` stole 17% gain; predictions compressed. Identity is real univariately (notif TE 0.76 vs raw 0.49) but already inside deep LGBM. **Stop TE-in-GBM. No 5-fold.** | Do not inject lookup TE into LightGBM. Residual of nocat is not in generator arithmetic either |

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
- The new 0.60/0.40 blend improves every fold over `lgbm_nocat` by +0.000224 to +0.000404. Leave-one-fold-out weight selection chose an LGBM weight of 0.55 or 0.60 on all five held-out folds, so 0.60 is not driven by one fold.

**Complementary errors?**

- On 80k diagnostics, LGBM vs HistGB Pearson ≈ 0.99 and grid blend added ~0.0006.
- On full 5-fold OOF the same pair is Pearson 0.992. Grid 0.85/0.15 adds only **+0.000035**. HistGB is a weaker partner here because it capped at 500 trees on sklearn 1.6.1 without `X_val`.
- Extending only HistGB's budget to 1400 raises it from 0.962140 to **0.963468**. Its correlation with LGBM is 0.9937, but a 0.60/0.40 grid blend reaches **0.964087**. A 1%-step search peaks at 0.58/0.42 by only +0.00000066, so keep the coarser 0.60/0.40 weights.
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
4. `configs/histgb_nocat_long_v1.yaml` (CPU) — OOF 0.963468; +0.001329 vs capped HistGB.
5. `python scripts/blend_oof.py --experiments lgbm_nocat histgb_nocat_long_v1 --method grid --name blend_nocat_long_v1` — OOF **0.964087**, weights 0.60/0.40.

Submission CSVs are local/Kaggle artifacts (`submissions/lgbm_nocat.csv`, `submissions/blend_nocat_long_v1.csv`). Do not `kaggle competitions submit` unless explicitly asked.

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
- Another blend pass with the capped 500-tree HistGB; its longer-trained replacement is strictly better.
