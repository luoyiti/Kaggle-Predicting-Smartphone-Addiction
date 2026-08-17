# Predicting Smartphone Addiction (Playground Series S6E8)

Kaggle: https://www.kaggle.com/competitions/playground-series-s6e8

Config-driven tabular pipeline: YAML configs, modular `s6e8/` package, OOF + test predictions saved every run.

## Competition

| Item | Value |
| --- | --- |
| Task | Binary classification (target `addicted_label` is 0/1) |
| Metric | ROC-AUC on predicted probability of class 1 |
| Train | 691,369 rows |
| Test | 296,302 rows |
| Original data | ~7,500-row synthetic smartphone usage dataset |

`addicted_label` is the **target column**, not the evaluation metric. The leaderboard scores predicted probabilities with ROC-AUC. Predicting 0/1 as a regression problem is a modeling trick some people use; this repo evaluates and submits as classification.

## Data notes

Numeric (many contain NaNs): `age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, `weekend_screen_time`.

Categorical: `gender` (Male / Female / Other), `stress_level` (Low / Medium / High), `academic_work_impact` (Yes / No).

Missingness is widespread (including categoricals). Positive class is roughly 43%. Trees handle NaNs natively; baseline also adds `n_missing` and a few usage ratios.

## Layout

```
configs/          experiment YAML (hyperparams, feature lists, paths)
data/raw/         train.csv / test.csv / sample_submission.csv (gitignored)
oof/              per-experiment OOF + test predictions (npy + parquet)
submissions/      Kaggle submission CSVs
scripts/          CLI entrypoints
s6e8/             package: data, features, models
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# download competition files (requires kaggle CLI + API token)
kaggle competitions download -c playground-series-s6e8 -p data/raw
unzip -o data/raw/playground-series-s6e8.zip -d data/raw
```

## Train baseline

```bash
python scripts/train.py --config configs/baseline.yaml
```

Expect:

- OOF AUC printed per fold and overall
- `oof/baseline/oof.npy`, `oof/baseline/test.npy` (float64)
- `oof/baseline/oof.parquet`, `oof/baseline/test.parquet`
- `oof/baseline/metrics.json`
- `submissions/baseline.csv`
