# Predicting Smartphone Addiction (Playground Series S6E8)

Kaggle: https://www.kaggle.com/competitions/playground-series-s6e8

Config-driven tabular pipeline: YAML configs, modular `s6e8/` package, OOF + test predictions saved every run. Heavy training runs on **Kaggle Kernels**; GitHub is the source of truth; GitHub Actions only orchestrates.

```text
AI Agent / Codex Cloud
        ↓
      GitHub
        ↓
GitHub Actions          (CI, kernel push, status, artifacts)
        ↓
 Kaggle API
        ↓
Kaggle Kernel           (CPU / GPU training)
        ↓
metrics / OOF / test prediction / submission.csv
```

Local machines are optional. A cloud experiment can finish while your laptop is off.

## Competition

| Item | Value |
| --- | --- |
| Task | Binary classification (target `addicted_label` is 0/1) |
| Metric | ROC-AUC on predicted probability of class 1 |
| Train | 691,369 rows |
| Test | 296,302 rows |
| Original data | ~7,500-row synthetic smartphone usage dataset |

`addicted_label` is the **target column**, not the evaluation metric. The leaderboard scores predicted probabilities with ROC-AUC.

## Layout

```
configs/            experiment YAML (hyperparams, features, runtime)
data/raw/           train.csv / test.csv / sample_submission.csv (gitignored)
oof/                per-experiment OOF + test predictions (gitignored)
submissions/        Kaggle submission CSVs (gitignored)
experiments/        small experiment metadata JSON (ok to commit)
reports/            self-contained HTML reports and `index.html` overview
scripts/            CLI entrypoints (`train.py`, `eda.py`, `build_reports_index.py`)
s6e8/               package: data, features, models, runtime
kaggle/             Kernel runner + metadata template
.github/workflows/  CI + Kaggle Train orchestration
```

## Initial setup

### 1. Kaggle API

1. Open https://www.kaggle.com/settings/api
2. Click **Generate New Token**
3. Prefer the current `KAGGLE_API_TOKEN` value from that page

Locally you can put it in `.env` (gitignored) or export it:

```bash
cp .env.example .env
# edit .env: KAGGLE_API_TOKEN=...
export KAGGLE_API_TOKEN=...
```

If the token is the new opaque string, also set your Kaggle username (needed to build `username/kernel-slug`):

```bash
export KAGGLE_USERNAME=your-kaggle-username
```

Legacy fallback still works: `KAGGLE_USERNAME` + `KAGGLE_KEY`, or `~/.kaggle/kaggle.json`. Never commit those files.

### 2. GitHub Secrets

Repo → **Settings → Secrets and variables → Actions**:

| Secret | Required | What it is |
| --- | --- | --- |
| `KAGGLE_API_TOKEN` | yes | Token from https://www.kaggle.com/settings/api |
| `KAGGLE_USERNAME` | yes if token is not legacy JSON | Your Kaggle username |
| `KAGGLE_KEY` | no | Legacy API key, only if you still use `kaggle.json` |

If `KAGGLE_API_TOKEN` is the full legacy JSON (`{"username":"...","key":"..."}`), username is inferred and `KAGGLE_USERNAME` is optional.

### 3. Python (local or CI)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Optional local data download (not used on Kaggle Kernels):

```bash
kaggle competitions download -c playground-series-s6e8 -p data/raw
unzip -o data/raw/playground-series-s6e8.zip -d data/raw
```

On Kaggle, data is mounted at `/kaggle/input/playground-series-s6e8/`. The code maps `data/raw/*.csv` automatically.

## Train baseline (local)

```bash
python scripts/train.py --config configs/baseline.yaml
```

Baseline is LightGBM **CPU**. Do not switch it to GPU just to use a GPU.

Expect:

- OOF AUC printed per fold and overall
- `oof/baseline/oof.npy`, `oof/baseline/test.npy`
- `oof/baseline/metrics.json`, `oof/baseline/experiment.json`
- `submissions/baseline.csv`
- `experiments/baseline.json`

Cheap **ranking** runs (subsample, not a leaderboard/CV claim):

```bash
python scripts/train.py --config configs/lgbm_nocat.yaml --max-train-rows 80000 --n-splits 3
```

That renames the experiment to `lgbm_nocat_diag80000` and skips writing `experiments/*.json`. Record the ranking in `experiments/LOG.md`. Full 5-fold jobs belong on Kaggle Kernels.

Current best **official 5-fold** single-model YAML on `main`: `configs/lgbm_nocat.yaml` (OOF 0.963771). Blend partner: `configs/histgb_nocat.yaml` (blend 0.963806).

This-branch CatBoost exact-cat+budget 3-fold diagnostic is 0.967878 (not a 5-fold score). A CPU 5-fold of `configs/catboost_exactcat_budget_v1.yaml` (original cats dropped) is running at https://www.kaggle.com/code/yitiluo/s6e8-cb-exactcat-budget-dropcats-v1. PR #11 GPU kernels that **kept** original cats are a different YAML; see `experiments/LOG.md`.

```bash
python scripts/blend_oof.py --experiments lgbm_nocat histgb_nocat --method grid
```

Once Kaggle 5-fold OOF exists for CatBoost exact-cat:

```bash
python scripts/blend_oof.py --experiments lgbm_nocat catboost_exactcat_budget_v1 --method grid
python scripts/blend_oof.py --experiments lgbm_nocat catboost_exactcat_budget_v1 --method stack_logistic
python scripts/blend_oof.py --experiments catboost_exactcat_budget_v1 catboost_exactcat_budget_seed7 catboost_exactcat_budget_seed2026 --method mean --name catboost_exactcat_budget_seedavg
```

Ready Kernel YAMLs: `configs/catboost_exactcat_budget_v1.yaml` (primary CPU; drop-cats 5-fold kernel already pushed), seed variants `*_seed7` / `*_seed2026`, GPU `configs/catboost_exactcat_budget_gpu_v1.yaml`, HPO `configs/catboost_exactcat_budget_gpu_depth6_v1.yaml`. Original-distribution features `configs/catboost_exactcat_budget_refdist_v1.yaml` (80k **flat** vs budget; attach Kaggle dataset `jayjoshi37/smartphone-usage-and-addiction-prediction`; do **not** append those labelled rows to train). Optional torch NN: `configs/entity_mlp_hash_v1.yaml` (hash embeddings, not Lookup-Transformer).

Kick the primary 5-fold via GitHub → **Actions → Kaggle Train → Run workflow**:

```text
config=configs/catboost_exactcat_budget_v1.yaml
accelerator=cpu
submit_to_kaggle=false
```

Equivalent CLI (needs `workflow_dispatch` write permission):

```bash
gh workflow run kaggle-train.yml -f config=configs/catboost_exactcat_budget_v1.yaml -f accelerator=cpu -f submit_to_kaggle=false
```

## Cloud workflow (daily loop)

1. Agent adds `configs/xgb_gpu_v2.yaml` (new file, unique `experiment.name`)
2. Commit / merge to GitHub
3. GitHub → **Actions → Kaggle Train → Run workflow**
4. Inputs:
   - `config`: `configs/xgb_gpu_v2.yaml`
   - `accelerator`: `gpu` (or `cpu`)
   - `submit_to_kaggle`: leave **false** unless you really want a leaderboard submission
5. Actions packages the repo, `kaggle kernels push`, waits, downloads outputs
6. Open the workflow **Summary** for OOF ROC-AUC / runtime
7. Download the workflow artifact for OOF / test prediction / `submission.csv`

First cloud run of the current baseline:

1. Add the secrets above
2. Actions → **Kaggle Train** → Run workflow
3. `config=configs/baseline.yaml`
4. `accelerator=cpu`

Do **not** expect a Summary in the first few minutes. `Wait for kernel` is supposed to stay yellow while Kaggle trains. Full 5-fold LightGBM on this dataset often takes **30–180 minutes** on CPU. Expand that step to see heartbeats; the yellow job is not a hang by itself.

If `Configure Kaggle credentials` fails: the new opaque token also needs secret `KAGGLE_USERNAME`. Token-only is not enough to build `username/s6e8-cloud-train`.

Live kernel page (login required if private): `https://www.kaggle.com/code/<kaggle-username>/s6e8-cloud-train`

## Runtime YAML

```yaml
runtime:
  accelerator: cpu   # or gpu
  enable_internet: true
```

`--accelerator` on `scripts/train.py` and the workflow input override YAML. Model code only sets GPU device keys when `accelerator=gpu` and the backend understands it (LightGBM `device_type`, future XGBoost/CatBoost hooks).

## Tests

These do **not** claim a competition AUC. The smoke test uses a tiny synthetic dataframe.

```bash
python -m compileall -q .
python scripts/train.py --help
python scripts/eda.py --help
python scripts/build_reports_index.py --help
python scripts/validate_configs.py
pytest -q
```

## Reports

HTML reports live in `reports/`. `python scripts/build_reports_index.py` scans that directory and writes `reports/index.html`. Use `--check` to fail if the committed overview is stale.

## Optional GPU dependencies

Install extra libraries only when an experiment needs them (not in the default baseline):

```text
xgboost, catboost, torch
```

## Data notes

Numeric (many contain NaNs): `age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, `weekend_screen_time`.

Categorical: `gender` (Male / Female / Other), `stress_level` (Low / Medium / High), `academic_work_impact` (Yes / No).

Missingness is widespread (including categoricals). Positive class is roughly 43%. Trees handle NaNs natively; baseline also adds `n_missing` and a few usage ratios.
