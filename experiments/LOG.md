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
| lgbm_freq_v1 | Fold-safe log1p frequency | `lgbm_nocat` + freq of notif/app/sleep/age/gaming/work; 5-fold Kaggle | 0.963801 | 0.000522 | 750s | **+0.000029 vs lgbm_nocat.** 80k +0.001 did not scale. Unseen=0 every fold. | Weak solo; blend partner |
| blend_lgbm_freq | Complementary LGBM errors | Grid 0.50 nocat + 0.50 freq | **0.964208** | — | — | +0.00040 vs nocat. Pearson 0.994. | Include in later CatBoost blend |
| blend_freq_histlong | Freq + long HistGB | Grid 0.55 freq + 0.45 histlong (max_iter **1400**) | 0.964283 | — | — | +0.00007 vs blend_lgbm_freq. Pearson 0.991. | Optional |
| blend_nocat_freq_histlong | Three-way cats-dropped trees | Grid 0.30 nocat + 0.40 freq + 0.30 histlong | **0.964367** | — | — | Best this-branch **cats-dropped** 5-fold from existing OOF. Still far below CatBoost. | Wait for drop-cats CatBoost then re-blend |

Did **not** submit to the leaderboard. Did **not** add coverage features. Did **not** drop notifications/app_opens.

## Full-data screens (691,369 train, fewer folds — not a 5-fold score)

Exact-value TE must not be judged on an 80k subsample: repeat frequency collapses and
unseen-value rates are fake. These rows use the **full train**.

| experiment | hypothesis | change | CV AUC | fold std | runtime | conclusion | next step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lgbm_nocat_exact_te_v1_diag | Exact numeric values carry playground identity that raw splits miss | `lgbm_nocat` + fold-safe LOO TE on notif/app/sleep/age/gaming/work; 3-fold full data | 0.923636 | 0.000601 | 17s | **−0.040 vs lgbm_nocat.** Unseen=0 (not leakage). Best iter 20–31. `age_exact_te` stole 17% gain; predictions compressed. Identity is real univariately (notif TE 0.76 vs raw 0.49) but already inside deep LGBM. **Stop TE-in-GBM. No 5-fold.** | Do not inject lookup TE into LightGBM. Residual of nocat is not in generator arithmetic either |
| catboost_exactcat_budget_v1_diag | CatBoost ordered CTRs on exact-value copies + screen budget | 9 numeric + 9 `__exact` cats + budget arithmetic; 3-fold full 691,369 rows | **0.967878** | 0.000296 | 2026s | **+0.004 vs lgbm_nocat 5-fold 0.963771, but n_splits differs (3 vs 5).** Folds 0.96747–0.96817. Best iter 2629–3069. Same protocol family as the TE screen (full data, 3-fold), which failed. Not an official 5-fold. | **Kaggle 5-fold** `configs/catboost_exactcat_budget_v1.yaml`; then grid-blend with `lgbm_nocat` |

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

### 80k ranking of remaining paths (this iteration — still not a leaderboard number)

Same protocol as the table above: stratified 80,000-row subsample, 3 folds, seed 42.
Control `lgbm_nocat_diag80000` reproduced **0.954317** (matches the original ranking row).

| experiment | hypothesis | change | CV AUC | fold std | runtime | conclusion | next step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| catboost_numeric_v1_diag80000 | CatBoost family on nocat numerics | 9 numeric cols, no exact cats | 0.952448 | 0.000279 | 31s | −0.0019 vs LGBM. Family alone is not the win. | Control only |
| catboost_exactcat_v1_diag80000 | Exact values as CatBoost categories | + 9 `__exact` string cats | **0.960525** | 0.000173 | 115s | **+0.0062 vs LGBM.** Pearson vs LGBM 0.963 (more diverse than HistGB 0.99). | Full 5-fold on Kaggle |
| catboost_exactcat_budget_v1_diag80000 | Screen-budget arithmetic on top | remainder/share/awake | **0.961641** | 0.000200 | 124s | **+0.0011 vs exactcat.** Best single 80k model here. | Full 5-fold on Kaggle |
| catboost_exactcat_lattice_v1_diag80000 | Fractional / first-decimal | decimal_lattice.enabled | 0.960671 | 0.000124 | 107s | Flat vs exactcat (+0.00015). | Stop |
| catboost_origcats_v1_diag80000 | Keep gender/stress/academic | no drop, no exact cats | 0.951988 | 0.000373 | 38s | Original cats still noise in CatBoost. | Stop |
| lgbm_freq_v1_diag80000 | Fold-safe value frequency | log1p counts, no labels | 0.955410 | 0.000082 | 18s | **+0.0011 vs LGBM.** Small, real on this protocol. | Optional full-data screen |
| lgbm_exactcat_v1_diag80000 | LGBM categorical splits on exact copies | same copies as CatBoost | 0.955291 | 0.000351 | 10s | Tiny lift; not CatBoost CTR. | Stop as a solo; optional blend |
| lgbm_nocat_lr02_diag80000 | Slower learning rate | lr 0.02, 8000 rounds | 0.954934 | 0.000297 | 65s | +0.0006 vs nocat. Small. | Optional |
| lgbm_nocat_mono_diag80000 | Monotone usage/sleep | constraints | 0.952208 | 0.000257 | 18s | Hurts. Identity columns are not monotone. | Stop |
| lgbm_nocat_extratrees_diag80000 | extra_trees | extra_trees true | 0.939472 | 0.000548 | 32s | **−0.015. Failed.** | Stop |
| xgb_nocat_diag80000 | XGB on nocat view | drop 3 cats | 0.952417 | 0.000419 | 29s | Matches xgb_raw. Cats were not XGB's problem. | Stop unless GPU HPO |
| histgb_nocat_long_v1_diag80000 | More HistGB trees | max_iter 2500 | 0.954014 | 0.000231 | 14s | Flat vs histgb_nocat 80k (0.953985). The 500-iter cap was a full-data issue. | Full 5-fold still useful |
| histgb_exactcat_v1_diag80000 | HistGB native exact cats | un-hashed copies | — | — | 4s | **Failed:** cardinality 1244 > max_bins 255. | Use hashed variant / skip |
| histgb_exactcat_hashed_v1_diag80000 | Hash exact cats into 128 bins | hash_bins=128 | 0.944809 | 0.000469 | 14s | Hashing destroys identity. **−0.009 vs nocat.** | Stop |
| logreg_exact_te_v1_diag80000 | Linear identity stacker | logistic + TE of 9 nums | 0.942985 | 0.000631 | 5s | +0.032 vs logreg_raw 0.911. Still below trees. Possible weak stacker. | Optional stack component |
| mlp_nocat_diag80000 | sklearn MLP diversity | 64-32 MLP | 0.931215 | 0.000739 | 14s | Too weak to stack. | Stop |
| blend_lgbm_cb_budget_diag80000 | Complementary errors | grid 0.2 LGBM + 0.8 CB budget | **0.961967** | — | — | +0.00033 vs best single. Corr 0.961. | Repeat on full 5-fold OOF |
| stack_lgbm_cb_freq_diag80000 | Logistic stack of 3 | inner-CV logistic | 0.961897 | — | — | No better than 2-model grid. | Prefer grid of LGBM+CB budget |
| catboost_exactcat_budget_refdist_v1_diag80000 | Target-free original 7500-row CDF / robust-z / source-frequency / kNN | `features.reference.enabled` on top of budget; labels dropped; overlap filter on train+test | 0.961669 | 0.000225 | ~159s | **Flat vs budget 0.961641** (+0.00003). retained=7500, overlap_removed=0, +18 columns. | Do not prioritize a 5-fold; PR11 lattice+refdist 5-fold only +0.00021 |
| catboost_exactcat_budget_joint_v1_diag80000 | Explicit notif\\|app exact-value token | `exact_categorical.joint_pairs` on top of budget; n_cat=10 | 0.961698 | 0.000280 | 155s | **Flat vs budget 0.961641** (+0.00006). Pairwise CatBoost CTRs already cover the joint key. | Do not prioritize a 5-fold |
| catboost_exactcat_budget_identity_v1_diag80000 | Exact cats only on notif+app | `exact_categorical.columns` subset; n_cat=2 | 0.958548 | 0.000051 | 48s | **−0.0031 vs budget 0.961641.** The other seven exact copies add ranking, not noise. | Stop; keep all-numeric exact cats |
| entity_mlp_hash_v1_diag80000 | Hashed-entity residual MLP (not Lookup-Transformer) | torch CPU; hash_buckets=256; 12 epochs | 0.921705 | 0.000943 | 27s | **−0.033 vs lgbm_nocat 0.954317.** Worse than sklearn MLP 0.931. Hashing destroys identity (same lesson as hashed HistGB). Hit epoch cap on every fold. | Stop as a solo; do not 5-fold |
| catboost_exactcat_budget_plain_v1_diag80000 | Ordered vs Plain boosting | `model.params.boosting_type: Plain` | 0.961641 | 0.000200 | 120s | **Identical to budget Ordered 0.961641** (same OOF, std, folds to reported precision). Not a ranking lever on 80k. | Do not 5-fold for lift; optional runtime tweak |
| catboost_exactcat_budget_bernoulli_v1_diag80000 | Bernoulli row subsample | `model.params.bootstrap_type: Bernoulli` | 0.961279 | 0.000240 | 119s | **−0.00036 vs budget 0.961641.** Slightly worse solo. | Do not 5-fold |

**Promote to Kaggle 5-fold (in this order):** `catboost_exactcat_budget_v1` drop-cats CPU kernel still **RUNNING** as of 2026-08-23 17:52 UTC, then grid-blend with `lgbm_nocat` (and `lgbm_freq_v1` / `blend_lgbm_freq`). Seed-average after seed kernels land.

**Stop on 80k evidence:** monotone LGBM, extra_trees, original CatBoost cats, decimal lattice, hashed HistGB exact cats, MLP, XGB-nocat without HPO, **target-free original-source refdist** (flat vs budget), **explicit notif\\|app joint exact token** (flat vs budget), **identity-only exact cats** (notif+app only), **hashed-entity MLP**, **Plain vs Ordered CatBoost** (identical 80k), **Bernoulli bootstrap** (−0.00036 vs budget).

## Answers so far

**What actually determines `addicted_label`?**

- Original 7,500-row source: label is exactly `addiction_level ∈ {Moderate, Severe}`. A depth-3 tree of `daily_screen ≳ 8` or `social_media ≳ 4` already has ~0.99 AUC. Notifications, app opens, cats, gaming, work are ~0.50 there.
- Playground keeps the ~71% positive rate and the usage ranking, but **smooths** the hard rule (continuous daily AUC 0.890 >> stump 0.81) and **masks** ~14% of daily_screen (MCAR vs label).
- Playground also **entangles notifications and app_opens with the label via exact-value identity**, not a monotone ranking (raw notif AUC 0.49, fold-safe exact-value TE 0.76). Deep LightGBM already harvests that identity: dropping the columns costs ~0.016, but *explicit* TE does not add ranking on top of `lgbm_nocat`.
- Categoricals remain noise on both tables. Safe to drop.
- `id` is a sequential split, not a leak. Train/test value drift is negligible.

**What model fits this generator?**

- Trees, not linear models (logreg 0.911 vs GBM 0.954 on the same 80k protocol).
- LightGBM ≳ HistGB > XGBoost at the default-ish params used here **on raw/nocat numerics**.
- **CatBoost ordered CTRs on exact-value copies are a different mechanism than LGBM TE** and win on both 80k (+0.006) and full-data 3-fold (0.967878 diagnostic). Do not confuse this with TE-in-LightGBM, which failed.
- Explicit coverage features (`strong3_row_mean`) and the original OR-score do not beat native missing handling.
- Fold-safe exact-value TE is a real univariate signal and a harmful GBM feature (early-stopping hijack). Do not put lookup TE into `lgbm_nocat`.

**Is CV trustworthy?**

- Fold std ≈ 0.0003 on 80k×3 and 0.0006 on full 5-fold. Rankings held: `lgbm_nocat` still beats HistGB; blend still helps, but the full-data lift is much smaller.
- The TE screen’s −0.040 is ~70× fold std. Not a noisy CV flip. Unseen-value share was 0, so this is overfit of lookup features, not val→train leakage.

**Complementary errors?**

- On 80k diagnostics, LGBM vs HistGB Pearson ≈ 0.99 and grid blend added ~0.0006.
- On full 5-fold OOF the same pair is Pearson 0.992. Grid 0.85/0.15 adds only **+0.000035**. HistGB is a weaker partner here because it capped at 500 trees on sklearn 1.6.1 without `X_val`.
- Kernel `histgb_nocat_long_v1` (max_iter 1400, not this branch's 2500) is 0.963468. Grid 0.60 LGBM + 0.40 long-HistGB is **0.964087** (Pearson 0.994) — better than blend_nocat 0.963806, still far below CatBoost.
- This-branch `lgbm_freq_v1` 5-fold is 0.963801. Grid 0.50 nocat + 0.50 freq is **0.964208** (Pearson 0.994), slightly above the long-HistGB blend.
- Three-way grid 0.30 nocat + 0.40 freq + 0.30 histlong is **0.964367**. Still far below CatBoost.
- Do not average in failed ablations (`usage_core`, `lgbm_nocat_exact_te_v1`).
- Grid-blending `lgbm_nocat` OOF with fold-safe notif/app TE is ≤ **+0.00002**. Logistic stack of nocat+TE **hurts** (−0.0007).
- PR #11 CatBoost exact-cat+budget 5-fold OOF (GPU, **original cats kept**) Pearson vs `lgbm_nocat` is 0.979. Grid 0.15 LGBM + 0.85 CB budget = **0.967843**. Adding PR #11 lattice+refdist reaches **0.968084** without Lookup-Transformer.

**Where is `lgbm_nocat` still wrong?**

- Reproducible via `python scripts/analyze_error_band.py --experiment lgbm_nocat --compare catboost_exactcat_budget_v1 --train data/raw/train.csv` (report: `experiments/error_band_lgbm_nocat.json`).
- 93,459-row hard band (OOF p ∈ (0.3, 0.7), 13.5% of train): 50.05% positive, OOF AUC **0.64113**. Outside-band AUC 0.980.
- Inside that band every raw numeric has AUC 0.496–0.506. Residual corr vs `y − p` is |r| ≤ 0.039.
- PR #11 CatBoost budget (orig cats kept, **not this-branch YAML**) scores **0.7049** on those same LGBM-hard rows and has a smaller own band (81,398). CatBoost shrinks the uncertain slice; do not treat 0.7049 as this-branch drop-cats.

## Full 5-fold jobs (this repo, main-tracked YAML)

1. `configs/lgbm_nocat.yaml` (CPU) — OOF **0.963771**.
2. `configs/histgb_nocat.yaml` (CPU) — OOF 0.962140.
3. `python scripts/blend_oof.py --experiments lgbm_nocat histgb_nocat --method grid --name blend_nocat` — OOF **0.963806**.

This is still the **main-tracked official SOTA** for YAMLs that landed on `main`. This branch additionally landed `lgbm_freq_v1` **0.963801**, `blend_lgbm_freq` **0.964208**, and `blend_nocat_freq_histlong` **0.964367**. Do **not** `kaggle competitions submit` unless explicitly asked.

## Full 5-fold from existing Kaggle kernels (PR #11 / earlier; not this-branch YAML)

Downloaded 2026-08-23 with the Kaggle API. `diagnostic=false`, `n_splits=5`, `n_train=691369`. CatBoost kernels kept the original 3 categoricals (`n_cat=12`) and ran **GPU**. This branch's `catboost_exactcat_budget_v1.yaml` **drops** those cats. Do not treat PR #11 CatBoost rows as a score for this-branch YAML.

| experiment | kernel | OOF AUC | notes |
| --- | --- | ---: | --- |
| lgbm_nocat | [yitiluo/s6e8-lgbm-nocat](https://www.kaggle.com/code/yitiluo/s6e8-lgbm-nocat) | **0.963771** | Same as main-tracked |
| histgb_nocat_long_v1 | [yitiluo/s6e8-histgb-nocat-long-v1](https://www.kaggle.com/code/yitiluo/s6e8-histgb-nocat-long-v1) | 0.963468 | max_iter **1400** (this branch YAML is 2500); one fold hit the 1400 cap. Still below LGBM |
| catboost_exactcat_v1 | [yitiluo/s6e8-catboost-exactcat-v1](https://www.kaggle.com/code/yitiluo/s6e8-catboost-exactcat-v1) | 0.966982 | GPU; orig cats kept |
| catboost_exactcat_budget_v1 | [yitiluo/s6e8-catboost-exactcat-budget-v1](https://www.kaggle.com/code/yitiluo/s6e8-catboost-exactcat-budget-v1) | **0.967681** | GPU; orig cats kept; 1202s |
| catboost_exactcat_budget_lattice_refdist_v1 | [yitiluo/s6e8-catboost-exactcat-budget-lattice-refdist-v1](https://www.kaggle.com/code/yitiluo/s6e8-catboost-exactcat-budget-lattice-refdist-v1) | **0.967890** | GPU; lattice + target-free refdist; orig cats kept |
| lookup_transformer_v1 | [yitiluo/s6e8-lookup-transformer-v1](https://www.kaggle.com/code/yitiluo/s6e8-lookup-transformer-v1) | 0.965339 | **Solo gate vs CatBoost failed** (−0.0023 vs budget). Beats LGBM. Do not revive the architecture |

Grid blends of those downloaded OOF dumps (same train ids; not submitted):

| blend | weights | OOF AUC |
| --- | --- | ---: |
| lgbm_nocat + histgb_nocat_long_v1 | 0.60 / 0.40 | 0.964087 |
| lgbm_nocat + catboost_exactcat_budget_v1 | 0.15 / 0.85 | 0.967843 |
| lgbm_nocat + lattice_refdist | 0.15 / 0.85 | 0.967985 |
| lgbm + budget + exactcat | 0.15 / 0.70 / 0.15 | 0.967875 |
| lgbm + budget + lattice_refdist | 0.10 / 0.35 / 0.55 | **0.968084** |
| budget + lattice_refdist | 0.40 / 0.60 | 0.968011 |
| lgbm + lattice_refdist + lookup | 0.10 / 0.65 / 0.25 | 0.968294 |

The 0.968294 row uses Lookup-Transformer **only as an already-trained blender**. It is not a reason to reimplement that architecture (solo 0.965339 < CatBoost). Preferred blend **without** Lookup: **0.968084**.

## Full-data screens (this branch; not a 5-fold score)

1. `configs/lgbm_nocat_exact_te_v1.yaml` with `--n-splits 3` — OOF 0.923636.
2. `configs/catboost_exactcat_budget_v1.yaml` with `--n-splits 3` — OOF **0.967878** (cats dropped). **Not a 5-fold score.**

## Kernel kick for this-branch YAML (cats dropped)

`gh workflow run` returned **HTTP 403** (integration cannot `workflow_dispatch`).
Local `kaggle kernels push` only. `submit_to_kaggle` was **not** used. Do **not** overwrite PR #11 slugs
`yitiluo/s6e8-catboost-exactcat-budget-v1` or `yitiluo/s6e8-catboost-exactcat-v1`.

Status snapshot **2026-08-23 17:59 UTC** (`python3 -m kaggle kernels status`):

| experiment YAML | kernel | acc | status |
| --- | --- | --- | --- |
| `catboost_exactcat_budget_v1` | [yitiluo/s6e8-cb-exactcat-budget-dropcats-v1](https://www.kaggle.com/code/yitiluo/s6e8-cb-exactcat-budget-dropcats-v1) | CPU | **RUNNING** (this-branch 5-fold; cats dropped) |
| `lgbm_freq_v1` | [yitiluo/s6e8-lgbm-freq-v1](https://www.kaggle.com/code/yitiluo/s6e8-lgbm-freq-v1) | CPU | **COMPLETE** — 5-fold OOF **0.963801** (`diagnostic=false`, n_train=691369) |
| `catboost_exactcat_budget_seed7` | [yitiluo/s6e8-cb-budget-seed7-v1](https://www.kaggle.com/code/yitiluo/s6e8-cb-budget-seed7-v1) | CPU | **RUNNING** |
| `catboost_exactcat_budget_seed2026` | [yitiluo/s6e8-cb-budget-seed2026-v1](https://www.kaggle.com/code/yitiluo/s6e8-cb-budget-seed2026-v1) | CPU | **RUNNING** |
| `catboost_exactcat_v1` | [yitiluo/s6e8-cb-exactcat-dropcats-v1](https://www.kaggle.com/code/yitiluo/s6e8-cb-exactcat-dropcats-v1) | CPU | **RUNNING** (this-branch ablation; cats dropped; unique slug) |
| `catboost_exactcat_budget_gpu_v1` | [yitiluo/s6e8-cb-budget-dropcats-gpu-v1](https://www.kaggle.com/code/yitiluo/s6e8-cb-budget-dropcats-gpu-v1) | GPU | **RUNNING** (was queued; do not stampede with depth6) |

`gpu_depth6_v1` was **not** pushed (do not stampede GPU). Joint-pair YAML was **not** pushed (80k flat).

## Not worth more budget

- Baseline ratio engineering.
- `strong3_row_mean` / `strong3_row_max` / `or_usage_score` for GBMs.
- Dropping notifications or app_opens.
- Mixing original 7,500 rows into train (component dependence differs: original `daily` ⟂ social+gaming+work; playground never violates `daily ≥ sum`).
- Stacking extra coverage features on top of `lgbm_nocat`.
- Fold-safe exact-value target encoding **inside** LightGBM (lookup hijacks early stopping; already in the deep tree).
- Mean-blending nocat with OOF exact-value TE (≤ +0.00002).
- `other_screen` / `component_sum` / `weekend − daily` / value-frequency / fractional parts as extra GBM columns (residual vs nocat ≈ 0).
- Another LGBM+HistGB probability blend pass (long-HistGB blend is +0.00028 vs blend_nocat; CatBoost dominates).
- `extra_trees` LightGBM on nocat (80k −0.015).
- Monotone constraints on usage/sleep (identity columns are not monotone).
- Keeping original gender/stress/academic in CatBoost as a **solo** model (80k origcats 0.952). PR #11 5-fold still kept them *alongside* exact cats; this-branch YAML drops them.
- Decimal-lattice extras on top of CatBoost exact-cat (80k flat).
- Hashing exact-value cats into ≤255 bins for HistGB (destroys identity; raw exact cats exceed max_bins).
- sklearn MLP on nocat numerics (0.931 on 80k).
- Lookup-Transformer as a **solo** model (PR #11 5-fold 0.965339 < CatBoost 0.967681). Do not revive it; optional-torch path is hashed-entity MLP.
- Target-free original-source refdist on CatBoost budget (80k **0.961669 vs 0.961641**). Code is landed; a dedicated 5-fold is optional confirmation only.
- Explicit `notifications_per_day|app_opens_per_day` joint exact-cat token on CatBoost budget (80k **0.961698 vs 0.961641**). Pairwise CTRs already cover it; do not 5-fold.
- Restricting CatBoost exact-cats to notifications+app_opens only (80k **0.958548 vs budget 0.961641**). The other seven exact copies help; do not 5-fold.
- Hashed-entity residual MLP (80k **0.921705**). Hashing trick destroys identity; not a solo or 5-fold candidate. Not Lookup-Transformer.
- CatBoost `boosting_type: Plain` vs default Ordered (80k **identical** 0.961641).
- CatBoost `bootstrap_type: Bernoulli` vs default Bayesian (80k **0.961279 vs 0.961641**).

## Code-level paths landed this iteration

| config | isolated variable | 80k / kernel note |
| --- | --- | --- |
| `catboost_exactcat_budget_refdist_v1` | target-free original CDF / robust-z / frequency / kNN | 80k **flat**; labels never appended |
| `catboost_exactcat_budget_seed7` / `_seed2026` | seed only | mean-blend after 5-folds exist |
| `catboost_exactcat_budget_gpu_v1` | `runtime.accelerator: gpu` | YAML only; LightGBM unchanged |
| `catboost_exactcat_budget_gpu_depth6_v1` | depth 6 vs GPU v1 | YAML only |
| `entity_mlp_hash_v1` | hashed-entity residual MLP | 80k **0.921705**; torch CPU installed this turn; **stop as solo** |
| `catboost_exactcat_budget_joint_v1` | `exact_categorical.joint_pairs` notif\\|app | 80k **flat** vs budget; YAML kept for the isolated lever |
| `catboost_exactcat_budget_identity_v1` | exact cats = notif+app only | 80k **0.958548** (−0.003 vs budget); stop |
| `catboost_exactcat_budget_plain_v1` | `boosting_type: Plain` | 80k **identical** to Ordered budget |
| `catboost_exactcat_budget_bernoulli_v1` | `bootstrap_type: Bernoulli` | 80k **0.961279** (−0.00036 vs budget); stop |

`scripts/prepare_kaggle_kernel.py` copies `features.reference.dataset_source` into kernel `dataset_sources`. `scripts/blend_oof.py --method mean` is the seed-average glue. `scripts/analyze_error_band.py` writes `experiments/error_band_<name>.json` from saved OOF (no training).

## Remaining *runtime-only* work (this VM cannot finish)

- **Blocker for this-branch CatBoost SOTA:** wait for https://www.kaggle.com/code/yitiluo/s6e8-cb-exactcat-budget-dropcats-v1 then download OOF/metrics. **Drop-cats 5-fold is NOT yet in LOG.md.** Then:
  `python scripts/blend_oof.py --experiments lgbm_nocat catboost_exactcat_budget_v1 --method grid --name blend_lgbm_cb_budget`
  and optionally include `lgbm_freq_v1` / `blend_lgbm_freq`.
- Seed-averaged CatBoost 5-folds (`seed7`, `seed2026` kernels **RUNNING**) then `blend_oof.py --method mean`.
- This-branch `catboost_exactcat_v1` drop-cats ablation kernel **RUNNING**.
- `lgbm_freq_v1` 5-fold **COMPLETE** (0.963801); `blend_lgbm_freq` **0.964208**; `blend_nocat_freq_histlong` **0.964367**.
- GPU `catboost_exactcat_budget_gpu_v1` kernel **RUNNING**. `gpu_depth6_v1` still unrun (do not stampede GPU).
- Optional-torch `entity_mlp_hash_v1` 80k diagnostic **done** (0.921705). Do not 5-fold.
- Pseudo-labelling (high leak risk) and calibration (cannot move ROC-AUC) stay rejected.

```bash
# GHA path (403 for this cloud agent; run from a repo admin):
gh workflow run kaggle-train.yml \
  -f config=configs/catboost_exactcat_budget_v1.yaml \
  -f accelerator=cpu \
  -f submit_to_kaggle=false \
  -f kernel_slug=s6e8-cb-exactcat-budget-dropcats-v1

# After the drop-cats kernel completes:
python scripts/blend_oof.py --experiments lgbm_nocat catboost_exactcat_budget_v1 --method grid --name blend_lgbm_cb_budget
python scripts/blend_oof.py --experiments catboost_exactcat_budget_v1 catboost_exactcat_budget_seed7 catboost_exactcat_budget_seed2026 --method mean --name catboost_exactcat_budget_seedavg
```

Original source features (do **not** append labelled rows):

```bash
python3 -m kaggle datasets download -p data/raw --unzip jayjoshi37/smartphone-usage-and-addiction-prediction
```
