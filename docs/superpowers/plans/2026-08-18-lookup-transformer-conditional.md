# Conditional Lookup-Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed-five-fold GPU Lookup-Transformer that combines exact-value embeddings, smooth numeric embeddings, and screen-budget tokens after the tree-model phase gate is satisfied.

**Architecture:** A target-free preprocessor creates deterministic exact-value ids and robust normalized numeric values from combined train/test predictors, then each outer fold trains a fresh Transformer using only fold-training labels. The model creates one token per raw numeric field plus structural budget tokens, fuses exact lookup and periodic smooth branches, applies random feature masking and EMA, and averages five fold test predictions.

**Tech Stack:** Python 3.11+, pandas, NumPy, scikit-learn, PyTorch 2.x, PyYAML, pytest, Kaggle T4 GPU.

**Spec:** `docs/plans/2026-08-18-structural-feature-engineering-design.md`

## Global Constraints

- Execute this plan only when Task 8 of `2026-08-18-structural-tree-and-reference-features.md` records a positive Lookup gate decision.
- Validation remains `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Predictor-only vocabulary/scaling may be transductive across competition train and test, but no target may be used outside the fold trainer.
- Each outer fold initializes a fresh model, optimizer, scheduler, scaler, and EMA state.
- All architecture, augmentation, optimizer, and epoch settings are YAML-driven.
- Every formal run saves OOF, test predictions, fold ids, metrics, resolved config, and preprocessing provenance.
- No public prediction file is consumed and no Kaggle leaderboard submission is automated.
- Implement tests first and run a tiny CPU smoke test before Kaggle GPU training.

## File structure

- Create `s6e8/models/lookup_transformer.py`: preprocessing state, dataset, model modules, EMA, and one-fold trainer.
- Modify `s6e8/models/train.py`: register the `lookup_transformer` backend and pass fold indices/provenance.
- Modify `kaggle/runner.py`: install PyTorch only when the backend requests it and it is unavailable.
- Create `configs/lookup_transformer_v1.yaml`: fixed-five-fold exact+budget model configuration.
- Create `tests/test_lookup_transformer.py`: preprocessing, tensor-shape, target-isolation, and CPU smoke tests.
- Modify `tests/test_config.py` and `tests/test_cloud_helpers.py`.
- Modify `README.md`, `experiments/LOG.md`, and Draft PR #11 only after a successful formal cloud run.

---

### Task 1: Build target-free lookup preprocessing

**Files:**
- Create: `s6e8/models/lookup_transformer.py`
- Create: `tests/test_lookup_transformer.py`

**Interfaces:**
- Consumes: row-local transformed competition predictors and configured token columns.
- Produces: `LookupPreprocessor.fit(train, test)`, `LookupPreprocessor.transform(frame)`, and serializable provenance.

- [ ] **Step 1: Write failing deterministic-vocabulary test**

Create `tests/test_lookup_transformer.py`:

```python
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from s6e8.models.lookup_transformer import LookupPreprocessor


def test_lookup_preprocessor_is_target_free_and_deterministic():
    train = pd.DataFrame({
        "age": [20.0, 21.0, np.nan],
        "daily_screen_time_hours": [3.10, 3.20, 3.10],
        "screen_remainder_complete": [1.0, 1.2, np.nan],
    })
    test = pd.DataFrame({
        "age": [22.0],
        "daily_screen_time_hours": [3.30],
        "screen_remainder_complete": [1.1],
    })
    pre = LookupPreprocessor(
        lookup_columns=["age", "daily_screen_time_hours"],
        numeric_columns=["age", "daily_screen_time_hours", "screen_remainder_complete"],
        decimal_places={"age": 0, "daily_screen_time_hours": 2},
    ).fit(train, test)
    first = pre.transform(train)
    second = pre.transform(train.copy())
    assert np.array_equal(first.lookup_ids, second.lookup_ids)
    assert np.allclose(first.numeric_values, second.numeric_values, equal_nan=False)
    assert first.lookup_ids.shape == (3, 2)
    assert first.numeric_values.shape == (3, 3)
    assert first.missing_mask.shape == (3, 3)
    assert pre.provenance()["transductive_predictor_preprocessing"] is True
```

- [ ] **Step 2: Run preprocessing test and verify red**

Run: `pytest tests/test_lookup_transformer.py::test_lookup_preprocessor_is_target_free_and_deterministic -v`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement preprocessing dataclasses**

Create:

```python
@dataclass(frozen=True)
class LookupBatchArrays:
    lookup_ids: np.ndarray
    numeric_values: np.ndarray
    missing_mask: np.ndarray


class LookupPreprocessor:
    def __init__(
        self,
        lookup_columns: list[str],
        numeric_columns: list[str],
        decimal_places: dict[str, int],
    ) -> None:
        self.lookup_columns = list(lookup_columns)
        self.numeric_columns = list(numeric_columns)
        self.decimal_places = dict(decimal_places)
        self.value_to_id: dict[str, dict[str, int]] = {}
        self.medians: dict[str, float] = {}
        self.scales: dict[str, float] = {}
```

Use Task 2's canonical numeric serializer. Reserve id `0` for missing and `1` for out-of-vocabulary; assign sorted canonical values from combined train/test predictors starting at `2`. Fit numeric median and IQR from the same combined predictors, replacing non-positive IQR with `1.0`. Transform missing numeric values to normalized zero and mark them in a boolean mask.

Expose cardinalities with:

```python
@property
def lookup_cardinalities(self) -> list[int]:
    return [len(self.value_to_id[column]) + 2 for column in self.lookup_columns]
```

Import and reuse `canonical_numeric_value` from `s6e8.structural_features` so
CatBoost and Lookup features share identical grid keys.

- [ ] **Step 4: Run preprocessing tests and verify green**

Run: `pytest tests/test_lookup_transformer.py::test_lookup_preprocessor_is_target_free_and_deterministic -v`

Expected: PASS.

- [ ] **Step 5: Add OOV and serialization tests**

Assert a value absent during fit maps to id `1`, missing maps to `0`, and `provenance()` returns lookup cardinalities, numeric medians/scales, column names, and the transductive flag without any target-related key.

Run: `pytest tests/test_lookup_transformer.py -v`

Expected: PASS.

- [ ] **Step 6: Commit preprocessing**

```bash
git add s6e8/models/lookup_transformer.py tests/test_lookup_transformer.py
git commit -m "Add target-free lookup preprocessing"
```

---

### Task 2: Implement exact-plus-smooth Transformer tokens

**Files:**
- Modify: `s6e8/models/lookup_transformer.py`
- Modify: `tests/test_lookup_transformer.py`

**Interfaces:**
- Consumes: `LookupBatchArrays` from Task 1 and model hyperparameters.
- Produces: `LookupTransformer.forward(lookup_ids, numeric_values, missing_mask) -> logits`.

- [ ] **Step 1: Write failing model-shape and masking tests**

Add:

```python
from s6e8.models.lookup_transformer import LookupTransformer


def test_lookup_transformer_returns_one_finite_logit_per_row():
    model = LookupTransformer(
        lookup_cardinalities=[8, 10],
        n_numeric=3,
        d_model=32,
        plr_frequencies=8,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        mask_probability=0.2,
    )
    model.train()
    logits = model(
        lookup_ids=torch.tensor([[2, 3], [4, 0], [1, 5]], dtype=torch.long),
        numeric_values=torch.tensor([[0.1, 0.2, 0.3], [0.0, -0.2, 0.4], [1.0, 0.5, -0.1]]),
        missing_mask=torch.tensor([[False, False, False], [True, False, False], [False, False, False]]),
    )
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
```

- [ ] **Step 2: Run shape test and verify red**

Run: `pytest tests/test_lookup_transformer.py::test_lookup_transformer_returns_one_finite_logit_per_row -v`

Expected: FAIL because the model is absent.

- [ ] **Step 3: Implement token branches**

Implement one embedding table per lookup column, one learned numeric-column embedding, learned missing tokens, and a periodic numeric projection:

```python
class PeriodicNumericEmbedding(nn.Module):
    def __init__(self, n_features: int, n_frequencies: int, d_model: int) -> None:
        super().__init__()
        self.frequency = nn.Parameter(torch.randn(n_features, n_frequencies) * 0.1)
        self.projection = nn.Linear(2 * n_frequencies, d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * torch.pi * values.unsqueeze(-1) * self.frequency.unsqueeze(0)
        periodic = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.projection(periodic)
```

For the first `len(lookup_cardinalities)` numeric tokens, add the corresponding exact lookup embedding to the periodic numeric token. Structural budget columns without lookup ids use the smooth branch only. Replace missing positions with learned per-column missing tokens.

- [ ] **Step 4: Implement attention and masking**

Prepend a learned CLS token, add learned column-position embeddings, pass tokens through `nn.TransformerEncoder` with `batch_first=True` and GELU activation, then classify the CLS output with LayerNorm and a two-layer MLP. During training, independently replace non-CLS feature tokens with a learned mask token at `mask_probability`; disable random masking in evaluation mode.

- [ ] **Step 5: Run model tests and verify green**

Run: `pytest tests/test_lookup_transformer.py -v`

Expected: PASS.

- [ ] **Step 6: Commit model architecture**

```bash
git add s6e8/models/lookup_transformer.py tests/test_lookup_transformer.py
git commit -m "Implement exact and smooth Transformer tokens"
```

---

### Task 3: Add fold training, EMA, and early stopping

**Files:**
- Modify: `s6e8/models/lookup_transformer.py`
- Modify: `tests/test_lookup_transformer.py`

**Interfaces:**
- Consumes: fold train/validation/test arrays, labels, device, and YAML model settings.
- Produces: `train_lookup_fold(...) -> tuple[np.ndarray, np.ndarray, int, dict[str, float]]`.

- [ ] **Step 1: Write failing one-epoch CPU fold test**

Add a deterministic 64-row binary fixture, fit preprocessing, and call:

```python
va_pred, te_pred, best_epoch, diagnostics = train_lookup_fold(
    train_arrays=train_arrays,
    train_y=y_train,
    valid_arrays=valid_arrays,
    valid_y=y_valid,
    test_arrays=test_arrays,
    lookup_cardinalities=pre.lookup_cardinalities,
    params={
        "d_model": 16,
        "plr_frequencies": 4,
        "n_layers": 1,
        "n_heads": 2,
        "dropout": 0.0,
        "mask_probability": 0.0,
        "batch_size": 16,
        "epochs": 1,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "embedding_weight_decay": 1.0e-3,
        "ema_decay": 0.99,
        "patience": 1,
        "num_workers": 0,
    },
    seed=42,
    device="cpu",
)
assert va_pred.shape == (len(y_valid),)
assert te_pred.shape == (len(test_arrays.lookup_ids),)
assert np.isfinite(va_pred).all()
assert best_epoch == 1
assert diagnostics["best_auc"] >= 0.0
```

- [ ] **Step 2: Run fold test and verify red**

Run the named CPU test with `pytest -v`.

Expected: FAIL because fold training is absent.

- [ ] **Step 3: Implement dataset, deterministic loaders, and EMA**

Create a tensor dataset returning lookup ids, numeric values, missing mask, and optional labels. Seed Python, NumPy, PyTorch, CUDA, and the DataLoader generator. Create separate AdamW parameter groups so lookup embedding tables use `embedding_weight_decay` and other parameters use `weight_decay`.

Implement an EMA state dictionary updated after each optimizer step:

```python
shadow[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
```

Use an `ema_scope` context manager to swap shadow parameters in for validation/test prediction and restore training parameters afterward.

- [ ] **Step 4: Implement optimization and early stopping**

Use `BCEWithLogitsLoss`, automatic mixed precision only on CUDA, gradient clipping at configured `max_grad_norm`, and `OneCycleLR` across `epochs * len(train_loader)` steps. Evaluate EMA predictions each epoch, keep the highest validation AUC state, and stop after `patience` non-improving epochs. Return sigmoid probabilities from the best EMA state.

- [ ] **Step 5: Run CPU fold tests and verify green**

Run: `pytest tests/test_lookup_transformer.py -v`

Expected: PASS in under one minute on CPU.

- [ ] **Step 6: Commit fold training**

```bash
git add s6e8/models/lookup_transformer.py tests/test_lookup_transformer.py
git commit -m "Add Lookup-Transformer fold training"
```

---

### Task 4: Register the backend and formal configuration

**Files:**
- Modify: `s6e8/models/train.py:28-43`
- Modify: `s6e8/models/train.py:167-283`
- Create: `configs/lookup_transformer_v1.yaml`
- Modify: `tests/test_config.py`
- Modify: `tests/test_smoke_train.py`

**Interfaces:**
- Consumes: the shared row-local exact+budget feature pipeline and Task 3 fold trainer.
- Produces: full fixed-five-fold `lookup_transformer` training through the standard `train_cv` and `save_artifacts` APIs.

- [ ] **Step 1: Write failing backend-resolution test**

Add:

```python
def test_lookup_transformer_backend_resolves(baseline_config_path):
    config = load_config(baseline_config_path)
    config["model"]["name"] = "lookup_transformer"
    assert resolve_backend(config) == "lookup_transformer"
```

- [ ] **Step 2: Run backend test and verify red**

Expected: FAIL with unsupported backend.

- [ ] **Step 3: Register a specialized CV path**

Add the alias and dispatch `lookup_transformer` to `_run_lookup_cv` because its target-free vocabulary must be fitted once before folds. `_run_lookup_cv` must:

1. call existing row-local `transform` on train and test;
2. select raw lookup columns and structural numeric token columns from config;
3. fit one target-free `LookupPreprocessor` on combined train/test predictors;
4. allocate OOF, test, and zero-based fold ids;
5. call `train_lookup_fold` with only fold-training labels;
6. average five fold test predictions;
7. return the same artifact keys as `_run_cv` plus preprocessing provenance.

Assert no key containing `target` or `label` appears in preprocessing provenance.

- [ ] **Step 4: Create `configs/lookup_transformer_v1.yaml`**

Use exact+budget row-local features and:

```yaml
experiment:
  name: lookup_transformer_v1
  seed: 42
  formal: true
  validation_protocol: fixed5_seed42_v1
  data_version: v1
  feature_version: exact_budget_lookup_v1
  model_version: lookup_transformer_v1
runtime:
  accelerator: gpu
features:
  exact_categorical: {enabled: false}
  screen_budget: {enabled: true, tolerance: 1.0e-9}
  decimal_lattice: {enabled: false}
  external_reference: {enabled: false}
model:
  name: lookup_transformer
  lookup_columns: [age, daily_screen_time_hours, social_media_hours, gaming_hours, work_study_hours, sleep_hours, notifications_per_day, app_opens_per_day, weekend_screen_time]
  numeric_token_columns: [age, daily_screen_time_hours, social_media_hours, gaming_hours, work_study_hours, sleep_hours, notifications_per_day, app_opens_per_day, weekend_screen_time, screen_component_sum_complete, screen_remainder_complete, screen_component_share_complete, screen_remainder_share_complete, weekend_minus_component_sum, weekend_minus_remainder, awake_non_screen_hours]
  params:
    d_model: 128
    plr_frequencies: 24
    n_layers: 4
    n_heads: 8
    dropout: 0.15
    mask_probability: 0.10
    batch_size: 2048
    epochs: 32
    learning_rate: 1.0e-3
    weight_decay: 1.0e-4
    embedding_weight_decay: 1.0e-3
    ema_decay: 0.995
    patience: 6
    max_grad_norm: 1.0
    num_workers: 2
```

Set the nine-column decimal precision map to the same values as the CatBoost exact-category configs.

- [ ] **Step 5: Add config and two-fold CPU smoke tests**

Load the formal config, set `formal: false`, CPU, two folds, one epoch, 16-dimensional model, one layer, and batch size 32. Train on 120 synthetic rows, save artifacts, and assert finite OOF/test predictions, two fold ids, and preprocessing provenance.

Run: `pytest tests/test_lookup_transformer.py tests/test_smoke_train.py tests/test_config.py -v`

Expected: PASS.

- [ ] **Step 6: Commit backend integration**

```bash
git add s6e8/models/train.py configs/lookup_transformer_v1.yaml tests/test_lookup_transformer.py tests/test_smoke_train.py tests/test_config.py
git commit -m "Register fixed-fold Lookup-Transformer backend"
```

---

### Task 5: Package, verify, and run the formal GPU experiment

**Files:**
- Modify: `kaggle/runner.py`
- Modify: `tests/test_cloud_helpers.py`
- Create after success: `experiments/lookup_transformer_v1.json`
- Modify after success: `experiments/LOG.md`
- Modify after success: `README.md`
- Modify: Draft PR #11 through the GitHub connector.

**Interfaces:**
- Consumes: Tasks 1-4 and the existing GitHub Actions/Kaggle path.
- Produces: one real five-fold Lookup-Transformer result and honest blends against the best trees.

- [ ] **Step 1: Add optional PyTorch dependency mapping**

Extend `optional_requirements`:

```python
if backend in {"lookup_transformer", "lookup"}:
    return [("torch", "torch>=2.2,<3")]
```

Test that no installation is attempted when `torch` is already importable.

- [ ] **Step 2: Run full local verification**

Run:

```bash
pytest -q
python -m compileall -q s6e8 scripts kaggle
python scripts/validate_configs.py
python scripts/build_reports_index.py --check
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 3: Request independent code review and publish**

Use `superpowers:requesting-code-review`, fix every Critical or Important issue tests-first, rerun Step 2, and publish commits to the existing Draft PR #11. Confirm no raw data, notebook, OOF, test, submission, or credential file is committed.

- [ ] **Step 4: Dispatch formal Kaggle GPU training**

Use:

```text
config=configs/lookup_transformer_v1.yaml
accelerator=gpu
kernel_slug=s6e8-lookup-transformer-v1
gpu_machine_shape=NvidiaTeslaT4
```

Require successful workflow and kernel completion, five folds, 691,369 OOF rows, 296,302 test rows, finite probabilities, preprocessing provenance, and a matching git SHA.

- [ ] **Step 5: Evaluate single and honest blend results**

Compare Lookup alone to `lgbm_nocat`, `histgb_nocat_long_v1`, the best structural CatBoost, and their promoted blend. Run `scripts/blend_oof.py --method lofo_grid` for two- and three-model candidates. Report per-fold AUC, overall AUC, Pearson/Spearman correlations, fold-selected weights, and honest blend delta.

- [ ] **Step 6: Record results and update the Draft PR**

Commit compact experiment JSON and documentation only after validation. Include the cloud run URL, git SHA, runtime, best epoch per fold, single-model metrics, honest blends, limitations, and `No Kaggle leaderboard submission was made.` Keep the PR Draft until final review completes.
