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

The completed structural-tree phase moved the validated target-free 5-fold frontier
from the previous 0.60/0.40 LightGBM/HistGB blend (OOF AUC 0.964087) to
`configs/catboost_exactcat_budget_lattice_refdist_v1.yaml` (OOF AUC
**0.9678902672**). The main attributable steps were exact numeric values copied as
CatBoost categories (+0.007855663 over the numeric CatBoost control, 5/5 folds) and
screen-budget constraints (+0.000703812, 5/5 folds). The target-free 7,500-row
reference-distribution block added a further +0.000147928 in all five folds.

The separately marked label-aware reference run reached 0.9679204006, but added only
+0.000030133 over the target-free reference model in 3/5 folds and did not satisfy
its stricter preregistered promotion rule. It is not the default frontier.

The fixed-5-fold Lookup-Transformer v1 reached 0.9653386597 versus RefDist's
0.9678902672 (delta -0.0025516076) and fails its solo promotion gate. The neural family
therefore stops at v1 without a hyperparameter sweep. Its errors are complementary,
however: honest leave-one-fold-out weight selection produced a target-free tree
control of 0.9679961033, then reached **0.9683015509** with
RefDist/LGBM/HistGB/Lookup weights 0.65/0.05/0.05/0.25. This is +0.0003054476 over
the tree control in all five folds and +0.0004405748 over the earlier honest blend,
so Lookup is promoted only as a blend component. This four-model result is the new
validated target-free OOF frontier.

The direct RefDist/Lookup blend reached 0.9682403616. Replacing RefDist with the
externally supervised RefLabel model reached 0.9682632105, only +0.0000228489 over
the direct target-free comparison. RefLabel remains excluded from the primary gate
and is not promoted.

The original source is used only to derive reference features: it is never appended
to competition training rows. Its complete rows violate the screen-budget relation
in 60.6933% of cases, while competition data has no such violations. No public
prediction CSV is consumed, and no leaderboard submission has been made. See
`experiments/LOG.md` for per-fold results, provenance, and promotion decisions.

```bash
python scripts/blend_oof.py \
  --experiments lgbm_nocat histgb_nocat_long_v1 \
  --method grid \
  --name blend_nocat_long_v1
```

## Cloud workflow (daily loop)

1. Agent adds `configs/xgb_gpu_v2.yaml` (new file, unique `experiment.name`)
2. Commit / merge to GitHub
3. GitHub → **Actions → Kaggle Train → Run workflow**
4. Inputs:
   - `config`: `configs/xgb_gpu_v2.yaml`
   - `accelerator`: `gpu` (or `cpu`)
   - `kernel_slug`: optional isolated kernel slug (default `s6e8-cloud-train`)
   - `gpu_machine_shape`: GPU shape when `accelerator=gpu` (default `NvidiaTeslaT4`)
5. Actions packages the repo, runs the Kaggle Kernel, and downloads outputs. It keeps
   `submission.csv` in the workflow artifact and never submits it to the leaderboard.
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

Missingness is widespread (including categoricals). Positive class is roughly 71%. Trees handle NaNs natively; baseline also adds `n_missing` and a few usage ratios.
