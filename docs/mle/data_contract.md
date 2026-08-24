# Data contract — S6E8

Kaggle-native mapping of mle-workflow “Lock the Data Contract”.
Machine-readable twin: `s6e8/contracts.py`. Audit CLI: `python scripts/audit_data.py`.

## Grain and keys

- Primary key: `id` (unique, sequential; train then test).
- Label: `addicted_label` present only on train.
- Label timing: generated with the row (no delayed outcome). Point-in-time joins are N/A.

## Columns

Numeric (nulls allowed): `age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, `weekend_screen_time`.

Categorical (nulls allowed): `gender` {Male, Female, Other}, `stress_level` {Low, Medium, High}, `academic_work_impact` {Yes, No}.

## Split policy

- Local/Kernel CV: `StratifiedKFold` on `addicted_label`, shuffle=true, seed from YAML (42).
- Default `n_splits=5`. Diagnostic ranking may use `--max-train-rows` and `--n-splits 3`; those runs rename the experiment to `*_diag*` and **must not** be written as `experiments/<name>.json` for competition claims.
- Test is the Kaggle test file. Do not peek at LB to tune repeatedly (one primary variable per experiment).

## Leakage rules

| Risk | Policy |
| --- | --- |
| `id` as a feature | Forbidden (sequential split). Audit may score id-vs-label AUC. |
| Target encoding | Fold-safe only (`s6e8.target_encoding`). Proven harmful inside LightGBM — do not re-enable on `lgbm_nocat`. |
| Original 7,500-row source | Do not concat into playground train (different component dependence). |
| Test statistics in features | Row-wise transforms only before CV. TE maps fit on the training fold. |
| Future information | None in this table; all features are contemporaneous. |

## Missingness

Absence is allowed on every feature column. Trees may use native NaN. Missingness-as-signal is an explicit YAML flag (`add_n_missing`, `add_missing_indicators`), not median fill unless the backend requires it (logreg, ExtraTrees).

## Shift audit

`scripts/audit_data.py` records numeric/categorical PSI and an **adversarial** logistic AUC (is_test ~ features + missing flags, 40k subsample). PSI on the official tables is ~1e-5. Adversarial AUC on this snapshot was **0.559** (warn ≥ 0.55), consistent with 2–3% missing-rate gaps, not a feature overhaul.

## Snapshot

- Local: `data/raw/train.csv`, `test.csv`, `sample_submission.csv` (gitignored).
- Kernel: `/kaggle/input/playground-series-s6e8/` or `/kaggle/input/competitions/playground-series-s6e8/`.
- Record `data_version` in YAML (`v1` = official Kaggle playground tables).

## Train/serve equivalence

Train, OOF, and test predictions share `s6e8.features.transform`. There is no separate serving preprocessor. Ensemble/calibration scripts consume saved OOF/test npy/parquet only.
