# S6E8 experiment log

Scores in the **diagnostic** rows use a stratified train subsample and fewer folds.
They are for ranking hypotheses only. A number is a competition result only when
`diagnostic` is false, `n_splits=5`, full train, and `oof/<name>/metrics.json` exists.

| experiment | hypothesis | change | CV AUC | fold std | runtime | conclusion | next step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Current YAML (ratios + n_missing + leisure) is a usable GBM starting point | LightGBM 5-fold, seed 42 | _pending full 5-fold_ | | | Control. Do not overwrite. | Compare `lgbm_raw` |
| lgbm_raw | Ratio/leisure/n_missing dilute raw usage levels | All engineering flags off; same LGBM | | | | | |
| lgbm_strong3_mean | Missing daily_screen is the main hole; skip-NA mean of daily/weekend/social restores coverage | `lgbm_raw` + `add_strong3_row_mean` | | | | | |

## Modeling diagnosis (from EDA + original 7,500-row source)

What actually drives `addicted_label`:

- On Jay Joshi's original table, `addicted_label` is **exactly** `addiction_level in {Moderate, Severe}` (Mild and missing level are 0). Playground dropped `addiction_level`.
- A depth-3 tree on original data reaches ~0.989 AUC using almost only `daily_screen_time_hours` (split ~8.0h) and `social_media_hours` (split ~4.0h). Weekend is a scaled copy of daily (r=0.96). Gaming, work, notifications, app opens, stress, academic impact, gender are ~0.50 AUC on original.
- Playground **preserves the positive rate** (~70.9% vs original 70.8%) and the strong screen/social/weekend univariate ranking, but:
  - Labels are smoothed (daily≥8 stump AUC 0.81 vs continuous daily 0.890). Keep raw continuous values; do not binarize the original rule.
  - Features are masked (~14% daily missing, MCAR vs label, missing-indicator AUC≈0.50). Original had **zero** feature missingness. The modeling problem is recovering the latent usage when the main column is gone.
  - Component algebra changed: original `daily` is independent of social+gaming+work (60% "violations"); playground never violates `daily >= social+gaming+work`. So playground `work`/`gaming` AUC (0.65/0.62) is mostly **through daily**, not an independent original cause.
- Train/test value PSI/KS are negligible. Missing-rate gaps exist (up to 3.4%) but are not label-informative.
- Current baseline ratios (`screen_sleep_ratio`, `weekend_weekday_ratio`, `leisure_hours`, `notif_per_open`) are univariate-weaker than their components. `n_missing` is noise.
- `id` is a sequential split (train 0..691368, test after). No id leak (AUC 0.501).
- Do **not** mix original 7,500 rows into train until a dedicated experiment: the joint law of screen components is different, so it can inject the wrong dependence.

Implication: a GBM on raw usage columns should already be strong (80k×3-fold probe OOF ~0.95, not a competition score). The first attributable questions are (1) whether the current engineering hurts, (2) whether an explicit coverage feature for missing daily helps, (3) whether other tree families have complementary errors.

## Probe notes (not competition scores)

- 80k-row 3-fold LightGBM on raw-ish columns: OOF ≈ 0.953.
- Same protocol, `daily_screen_time_hours` only: OOF ≈ 0.869 (univariate 0.890 is on **observed** rows only).
- `or_usage_score` = max(daily/8, social/4, weekend/9.92): univariate 0.888, coverage 97.8%. Weaker than EDA `strong3_row_mean` 0.916. Original hard OR rule is not the playground surface.
