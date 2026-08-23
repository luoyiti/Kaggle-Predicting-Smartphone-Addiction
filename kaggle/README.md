# Kaggle Cloud Runner

This directory is the Kaggle-side entry for cloud experiments. Training still
lives in `scripts/train.py` and `s6e8/`. Do not copy model logic into `runner.py`.

## Why a bundle?

`kaggle kernels push` uploads **only** `code_file` (`runner.py`). Extra files in
this folder are not sent to Kaggle. GitHub Actions therefore runs:

```bash
python scripts/prepare_kaggle_kernel.py \
  --config configs/baseline.yaml \
  --accelerator cpu \
  --username "$KAGGLE_USERNAME"
```

That command writes `.kernel-staging/` with:

- generated `runner.py` (run context + source archive)
- `kernel-metadata.json` (CPU/GPU + competition data source)
- `run_context.json` (debug copy of the same context)

Then:

```bash
kaggle kernels push -p .kernel-staging
```

On Kaggle the runner extracts the archive into `/kaggle/working/src` (or uses an
already-present checkout) and calls `scripts/train.py`.

Competition CSVs come from the mounted dataset:

```text
/kaggle/input/playground-series-s6e8/train.csv
```

not from a second `kaggle competitions download`.

Configs that set `features.reference.enabled` also need the original ~7,500-row
**source dataset** attached (features only; do not concatenate those labelled
rows into train). `scripts/prepare_kaggle_kernel.py` copies
`features.reference.dataset_source` into `kernel-metadata.json` `dataset_sources`.
Expected typed mount:

```text
/kaggle/input/datasets/jayjoshi37/smartphone-usage-and-addiction-prediction/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
```

## Local dry-run of the runner

From the repo root, with data already in `data/raw/`:

```bash
python kaggle/runner.py --config configs/baseline.yaml --accelerator cpu
```

This uses the local checkout and does not push anything.
