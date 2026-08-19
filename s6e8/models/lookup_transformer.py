"""Target-free preprocessing and exact-plus-smooth lookup models."""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from s6e8.structural_features import canonical_numeric_value

try:
    import torch
    from torch import nn
except ImportError:  # Keep predictor-only preprocessing usable without PyTorch.
    torch = None
    nn = None


@dataclass(frozen=True)
class LookupBatchArrays:
    """Predictor arrays consumed by the Lookup-Transformer backend."""

    lookup_ids: np.ndarray
    numeric_values: np.ndarray
    missing_mask: np.ndarray


class LookupPreprocessor:
    """Fit deterministic lookup vocabularies and robust numeric scaling."""

    def __init__(
        self,
        lookup_columns: list[str],
        numeric_columns: list[str],
        decimal_places: dict[str, int],
    ) -> None:
        self.lookup_columns = list(lookup_columns)
        self.numeric_columns = list(numeric_columns)
        lookup_prefix = self.numeric_columns[: len(self.lookup_columns)]
        if lookup_prefix != self.lookup_columns:
            raise ValueError(
                "numeric_columns must start with lookup_columns in identical order"
            )
        self.decimal_places = {
            column: int(decimals) for column, decimals in decimal_places.items()
        }
        self.value_to_id: dict[str, dict[str, int]] = {}
        self.medians: dict[str, float] = {}
        self.scales: dict[str, float] = {}
        self._is_fitted = False

    @property
    def lookup_cardinalities(self) -> list[int]:
        """Return embedding cardinalities including missing and OOV ids."""
        return [len(self.value_to_id[column]) + 2 for column in self.lookup_columns]

    def fit(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> LookupPreprocessor:
        """Fit state from configured train/test predictor columns only."""
        self._is_fitted = False
        required = list(dict.fromkeys([*self.lookup_columns, *self.numeric_columns]))
        self._require_columns(train, required, "train")
        self._require_columns(test, required, "test")
        combined = pd.concat(
            [train.loc[:, required], test.loc[:, required]], ignore_index=True
        )

        self.value_to_id = {}
        for column in self.lookup_columns:
            decimals = self.decimal_places.get(column, 8)
            values = {
                canonical_numeric_value(value, decimals, "__MISSING__")
                for value in combined[column]
                if not pd.isna(value)
            }
            self.value_to_id[column] = {
                value: index for index, value in enumerate(sorted(values), start=2)
            }

        self.medians = {}
        self.scales = {}
        for column in self.numeric_columns:
            values = pd.to_numeric(combined[column], errors="coerce")
            median = float(values.median())
            scale = float(values.quantile(0.75) - values.quantile(0.25))
            if not np.isfinite(median):
                median = 0.0
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0
            self.medians[column] = median
            self.scales[column] = scale
        self._is_fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> LookupBatchArrays:
        """Transform predictors with missing id 0 and unseen-value id 1."""
        if not self._is_fitted:
            raise RuntimeError("LookupPreprocessor must be fitted before transform")
        required = list(dict.fromkeys([*self.lookup_columns, *self.numeric_columns]))
        self._require_columns(frame, required, "transform")

        lookup_ids = np.empty((len(frame), len(self.lookup_columns)), dtype=np.int64)
        for index, column in enumerate(self.lookup_columns):
            values = frame[column]
            missing = values.isna().to_numpy()
            decimals = self.decimal_places.get(column, 8)
            canonical = values.map(
                lambda value, d=decimals: canonical_numeric_value(
                    value, d, "__MISSING__"
                )
            )
            ids = canonical.map(self.value_to_id[column]).fillna(1).to_numpy(
                dtype=np.int64
            )
            ids[missing] = 0
            lookup_ids[:, index] = ids

        numeric_values = np.empty(
            (len(frame), len(self.numeric_columns)), dtype=np.float32
        )
        missing_mask = np.empty(
            (len(frame), len(self.numeric_columns)), dtype=np.bool_
        )
        for index, column in enumerate(self.numeric_columns):
            values = pd.to_numeric(frame[column], errors="coerce")
            missing = values.isna().to_numpy()
            normalized = (
                values.fillna(self.medians[column]) - self.medians[column]
            ) / self.scales[column]
            numeric_values[:, index] = normalized.to_numpy(dtype=np.float32)
            missing_mask[:, index] = missing

        return LookupBatchArrays(
            lookup_ids=lookup_ids,
            numeric_values=numeric_values,
            missing_mask=missing_mask,
        )

    def provenance(self) -> dict[str, Any]:
        """Return JSON-serializable predictor preprocessing state."""
        if not self._is_fitted:
            raise RuntimeError("LookupPreprocessor must be fitted before provenance")
        return {
            "lookup_columns": list(self.lookup_columns),
            "numeric_columns": list(self.numeric_columns),
            "decimal_places": dict(self.decimal_places),
            "lookup_cardinalities": list(self.lookup_cardinalities),
            "numeric_medians": dict(self.medians),
            "numeric_scales": dict(self.scales),
            "transductive_predictor_preprocessing": True,
        }

    @staticmethod
    def _require_columns(
        frame: pd.DataFrame, columns: list[str], frame_name: str
    ) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise KeyError(f"{frame_name} columns are missing: {missing}")


def _validation_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute binary AUC, using a stable neutral score for one-class folds."""
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    if labels.shape != predictions.shape:
        raise ValueError("validation labels and predictions must have equal length")
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError("validation labels and predictions must be finite")
    if np.unique(labels).size < 2:
        return 0.5
    return float(roc_auc_score(labels, predictions))


if nn is not None:

    class LookupTensorDataset(torch.utils.data.Dataset):
        """Tensor-backed lookup dataset with optional binary labels."""

        def __init__(
            self,
            arrays: LookupBatchArrays,
            labels: np.ndarray | None = None,
        ) -> None:
            self.lookup_ids = torch.from_numpy(
                np.ascontiguousarray(arrays.lookup_ids, dtype=np.int64)
            )
            self.numeric_values = torch.from_numpy(
                np.ascontiguousarray(arrays.numeric_values, dtype=np.float32)
            )
            self.missing_mask = torch.from_numpy(
                np.ascontiguousarray(arrays.missing_mask, dtype=np.bool_)
            )
            self.labels = (
                None
                if labels is None
                else torch.from_numpy(
                    np.ascontiguousarray(labels, dtype=np.float32).reshape(-1)
                )
            )

        def __len__(self) -> int:
            return int(self.lookup_ids.shape[0])

        def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
            predictors = (
                self.lookup_ids[index],
                self.numeric_values[index],
                self.missing_mask[index],
            )
            if self.labels is None:
                return predictors
            return (*predictors, self.labels[index])


    class _ExponentialMovingAverage:
        """Maintain and temporarily apply an EMA of model parameters."""

        def __init__(self, model: nn.Module, decay: float) -> None:
            self.decay = float(decay)
            self.shadow = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
            }

        @torch.no_grad()
        def update(self, model: nn.Module) -> None:
            for name, parameter in model.named_parameters():
                self.shadow[name].mul_(self.decay).add_(
                    parameter.detach(), alpha=1.0 - self.decay
                )

        def state_dict(self) -> dict[str, torch.Tensor]:
            return {name: value.detach().clone() for name, value in self.shadow.items()}

        def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
            if state.keys() != self.shadow.keys():
                raise ValueError("EMA state parameter names do not match the model")
            self.shadow = {
                name: value.detach().clone() for name, value in state.items()
            }

        @contextmanager
        def scope(self, model: nn.Module) -> Iterator[None]:
            backup = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
            }
            try:
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        parameter.copy_(self.shadow[name])
                yield
            finally:
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        parameter.copy_(backup[name])

    class PeriodicNumericEmbedding(nn.Module):
        """Learn periodic smooth embeddings for each numeric feature."""

        def __init__(
            self,
            n_features: int,
            n_frequencies: int,
            d_model: int,
        ) -> None:
            super().__init__()
            if n_features <= 0:
                raise ValueError("n_features must be positive")
            if n_frequencies <= 0:
                raise ValueError("n_frequencies must be positive")
            self.n_features = int(n_features)
            self.frequency = nn.Parameter(
                torch.randn(self.n_features, n_frequencies) * 0.1
            )
            self.projection = nn.Linear(2 * n_frequencies, d_model)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            if values.ndim != 2 or values.shape[1] != self.n_features:
                raise ValueError(
                    "values must have shape "
                    f"(batch_size, {self.n_features}); got {tuple(values.shape)}"
                )
            phase = (
                2.0
                * torch.pi
                * values.unsqueeze(-1)
                * self.frequency.unsqueeze(0)
            )
            periodic = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
            return self.projection(periodic)


    class LookupTransformer(nn.Module):
        """Fuse exact lookup ids and smooth numeric tokens with attention."""

        def __init__(
            self,
            lookup_cardinalities: list[int],
            n_numeric: int,
            d_model: int,
            plr_frequencies: int,
            n_layers: int,
            n_heads: int,
            dropout: float,
            mask_probability: float,
        ) -> None:
            super().__init__()
            self.n_lookup = len(lookup_cardinalities)
            if any(
                type(cardinality) is not int or cardinality < 2
                for cardinality in lookup_cardinalities
            ):
                raise ValueError(
                    "every lookup cardinality must be an integer of at least 2"
                )
            positive_integer_parameters = {
                "n_numeric": n_numeric,
                "d_model": d_model,
                "plr_frequencies": plr_frequencies,
                "n_layers": n_layers,
                "n_heads": n_heads,
            }
            for name, value in positive_integer_parameters.items():
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")

            self.n_numeric = n_numeric
            if self.n_numeric < self.n_lookup:
                raise ValueError(
                    "n_numeric must be at least the number of lookup columns"
                )
            if d_model % n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads")
            for name, value in {
                "dropout": dropout,
                "mask_probability": mask_probability,
            }.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not 0.0 <= float(value) <= 1.0
                ):
                    raise ValueError(f"{name} must be between 0 and 1")

            self.mask_probability = float(mask_probability)
            self.exact_embeddings = nn.ModuleList(
                [
                    nn.Embedding(int(cardinality), d_model)
                    for cardinality in lookup_cardinalities
                ]
            )
            self.periodic_embedding = PeriodicNumericEmbedding(
                n_features=self.n_numeric,
                n_frequencies=plr_frequencies,
                d_model=d_model,
            )
            self.numeric_column_embedding = nn.Parameter(
                torch.empty(self.n_numeric, d_model)
            )
            self.missing_tokens = nn.Parameter(torch.empty(self.n_numeric, d_model))
            self.mask_token = nn.Parameter(torch.empty(1, 1, d_model))
            self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
            self.position_embedding = nn.Parameter(
                torch.empty(1, self.n_numeric + 1, d_model)
            )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            self._reset_token_parameters()

        def _reset_token_parameters(self) -> None:
            nn.init.normal_(self.numeric_column_embedding, std=0.02)
            nn.init.normal_(self.missing_tokens, std=0.02)
            nn.init.normal_(self.mask_token, std=0.02)
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.position_embedding, std=0.02)

        def forward(
            self,
            lookup_ids: torch.Tensor,
            numeric_values: torch.Tensor,
            missing_mask: torch.Tensor,
        ) -> torch.Tensor:
            self._validate_inputs(lookup_ids, numeric_values, missing_mask)
            feature_tokens = self.periodic_embedding(numeric_values)
            feature_tokens = (
                feature_tokens + self.numeric_column_embedding.unsqueeze(0)
            )

            if self.n_lookup:
                exact_tokens = torch.stack(
                    [
                        embedding(lookup_ids[:, index])
                        for index, embedding in enumerate(self.exact_embeddings)
                    ],
                    dim=1,
                )
                feature_tokens = torch.cat(
                    [
                        feature_tokens[:, : self.n_lookup] + exact_tokens,
                        feature_tokens[:, self.n_lookup :],
                    ],
                    dim=1,
                )

            learned_missing = self.missing_tokens.unsqueeze(0)
            feature_tokens = torch.where(
                missing_mask.unsqueeze(-1), learned_missing, feature_tokens
            )
            if self.training and self.mask_probability > 0.0:
                feature_mask = (
                    torch.rand(
                        missing_mask.shape,
                        device=feature_tokens.device,
                    )
                    < self.mask_probability
                )
                feature_tokens = torch.where(
                    feature_mask.unsqueeze(-1), self.mask_token, feature_tokens
                )

            cls_tokens = self.cls_token.expand(feature_tokens.shape[0], -1, -1)
            tokens = torch.cat([cls_tokens, feature_tokens], dim=1)
            encoded = self.encoder(tokens + self.position_embedding)
            return self.classifier(encoded[:, 0]).squeeze(-1)

        def _validate_inputs(
            self,
            lookup_ids: torch.Tensor,
            numeric_values: torch.Tensor,
            missing_mask: torch.Tensor,
        ) -> None:
            if lookup_ids.ndim != 2 or lookup_ids.shape[1] != self.n_lookup:
                raise ValueError(
                    "lookup_ids must have shape "
                    f"(batch_size, {self.n_lookup}); got {tuple(lookup_ids.shape)}"
                )
            if numeric_values.ndim != 2 or numeric_values.shape[1] != self.n_numeric:
                raise ValueError(
                    "numeric_values must have shape "
                    f"(batch_size, {self.n_numeric}); got {tuple(numeric_values.shape)}"
                )
            if missing_mask.shape != numeric_values.shape:
                raise ValueError(
                    "missing_mask must have the same shape as numeric_values"
                )
            if lookup_ids.shape[0] != numeric_values.shape[0]:
                raise ValueError(
                    "lookup_ids and numeric_values must have the same batch size"
                )


else:

    class LookupTensorDataset:
        """Unavailable PyTorch dataset placeholder."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("LookupTensorDataset requires PyTorch")

    class PeriodicNumericEmbedding:
        """Unavailable PyTorch module placeholder."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PeriodicNumericEmbedding requires PyTorch")


    class LookupTransformer:
        """Unavailable PyTorch model placeholder."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("LookupTransformer requires PyTorch")


def _validate_fold_arrays(
    *,
    train_arrays: LookupBatchArrays,
    train_y: np.ndarray,
    valid_arrays: LookupBatchArrays,
    valid_y: np.ndarray,
    test_arrays: LookupBatchArrays,
    lookup_cardinalities: list[int],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate array alignment and return normalized labels and token count."""
    splits = {
        "train": train_arrays,
        "valid": valid_arrays,
        "test": test_arrays,
    }
    if not all(isinstance(arrays, LookupBatchArrays) for arrays in splits.values()):
        raise TypeError("fold arrays must be LookupBatchArrays instances")
    if len(train_arrays.lookup_ids) == 0 or len(valid_arrays.lookup_ids) == 0:
        raise ValueError("train and valid fold arrays must be non-empty")

    n_lookup = len(lookup_cardinalities)
    n_numeric = train_arrays.numeric_values.shape[1]
    if n_numeric <= 0:
        raise ValueError("fold arrays must contain at least one numeric token")
    if n_lookup > n_numeric:
        raise ValueError("lookup cardinalities cannot exceed numeric token count")

    for split_name, arrays in splits.items():
        expected_rows = len(arrays.lookup_ids)
        if arrays.lookup_ids.ndim != 2 or arrays.lookup_ids.shape[1] != n_lookup:
            raise ValueError(
                f"{split_name} lookup_ids must have {n_lookup} columns"
            )
        if arrays.numeric_values.ndim != 2 or arrays.numeric_values.shape != (
            expected_rows,
            n_numeric,
        ):
            raise ValueError(
                f"{split_name} numeric_values must have {n_numeric} columns"
            )
        if arrays.missing_mask.shape != arrays.numeric_values.shape:
            raise ValueError(
                f"{split_name} missing_mask must match numeric_values shape"
            )
        if not np.isfinite(arrays.numeric_values).all():
            raise ValueError(f"{split_name} numeric_values must be finite")
        if n_lookup:
            if (arrays.lookup_ids < 0).any():
                raise ValueError(f"{split_name} lookup_ids must be non-negative")
            for column, cardinality in enumerate(lookup_cardinalities):
                if type(cardinality) is not int or cardinality < 2:
                    raise ValueError(
                        "every lookup cardinality must be an integer of at least 2"
                    )
                if (arrays.lookup_ids[:, column] >= cardinality).any():
                    raise ValueError(
                        f"{split_name} lookup_ids exceed cardinality in column {column}"
                    )

    normalized_labels: list[np.ndarray] = []
    for split_name, labels, arrays in (
        ("train", train_y, train_arrays),
        ("valid", valid_y, valid_arrays),
    ):
        values = np.asarray(labels, dtype=np.float32).reshape(-1)
        if len(values) != len(arrays.lookup_ids):
            raise ValueError(f"{split_name} labels must align with fold arrays")
        if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
            raise ValueError(f"{split_name} labels must be finite binary values")
        normalized_labels.append(values)
    return normalized_labels[0], normalized_labels[1], n_numeric


def _integer_param(
    params: dict[str, Any], name: str, *, minimum: int = 1
) -> int:
    value = params.get(name)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _float_param(
    params: dict[str, Any],
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
    maximum_inclusive: bool = True,
) -> float:
    value = params.get(name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    upper_invalid = maximum is not None and (
        value > maximum if maximum_inclusive else value >= maximum
    )
    if not np.isfinite(value) or value < minimum or upper_invalid:
        upper = "" if maximum is None else f" and below {maximum}"
        raise ValueError(f"{name} must be at least {minimum}{upper}")
    return value


def _predict_lookup(
    model: Any,
    loader: Any,
    device: Any,
) -> np.ndarray:
    """Return sigmoid probabilities while preserving the caller's train mode."""
    was_training = model.training
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for lookup_ids, numeric_values, missing_mask in loader:
            logits = model(
                lookup_ids.to(device, non_blocking=True),
                numeric_values.to(device, non_blocking=True),
                missing_mask.to(device, non_blocking=True),
            )
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    model.train(was_training)
    if not predictions:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(predictions).astype(np.float64, copy=False)


def train_lookup_fold(
    *,
    train_arrays: LookupBatchArrays,
    train_y: np.ndarray,
    valid_arrays: LookupBatchArrays,
    valid_y: np.ndarray,
    test_arrays: LookupBatchArrays,
    lookup_cardinalities: list[int],
    params: dict[str, Any],
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, float | int | bool]]:
    """Train one fresh Lookup-Transformer fold and predict from its best EMA."""
    if torch is None or nn is None:
        raise ImportError("train_lookup_fold requires PyTorch")
    if not isinstance(params, dict):
        raise TypeError("params must be a dictionary")
    train_y, valid_y, n_numeric = _validate_fold_arrays(
        train_arrays=train_arrays,
        train_y=train_y,
        valid_arrays=valid_arrays,
        valid_y=valid_y,
        test_arrays=test_arrays,
        lookup_cardinalities=lookup_cardinalities,
    )
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    batch_size = _integer_param(params, "batch_size")
    epochs = _integer_param(params, "epochs")
    patience = _integer_param(params, "patience")
    num_workers = _integer_param(params, "num_workers", minimum=0)
    learning_rate = _float_param(params, "learning_rate", minimum=0.0)
    if learning_rate == 0.0:
        raise ValueError("learning_rate must be positive")
    weight_decay = _float_param(params, "weight_decay", minimum=0.0)
    embedding_weight_decay = _float_param(
        params, "embedding_weight_decay", minimum=0.0
    )
    ema_decay = _float_param(
        params,
        "ema_decay",
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    optimizer_params = dict(params)
    optimizer_params.setdefault("max_grad_norm", 1.0)
    max_grad_norm = _float_param(
        optimizer_params, "max_grad_norm", minimum=0.0
    )
    if max_grad_norm == 0.0:
        raise ValueError("max_grad_norm must be positive")

    try:
        torch_device = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid torch device: {device}") from error
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = LookupTransformer(
        lookup_cardinalities=list(lookup_cardinalities),
        n_numeric=n_numeric,
        d_model=_integer_param(params, "d_model"),
        plr_frequencies=_integer_param(params, "plr_frequencies"),
        n_layers=_integer_param(params, "n_layers"),
        n_heads=_integer_param(params, "n_heads"),
        dropout=_float_param(
            params, "dropout", minimum=0.0, maximum=1.0
        ),
        mask_probability=_float_param(
            params, "mask_probability", minimum=0.0, maximum=1.0
        ),
    ).to(torch_device)

    def seed_worker(_worker_id: int) -> None:
        worker_seed = torch.initial_seed() % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch_device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    train_loader = torch.utils.data.DataLoader(
        LookupTensorDataset(train_arrays, train_y),
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    valid_loader = torch.utils.data.DataLoader(
        LookupTensorDataset(valid_arrays), shuffle=False, **loader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        LookupTensorDataset(test_arrays), shuffle=False, **loader_kwargs
    )

    embedding_parameters = list(model.exact_embeddings.parameters())
    embedding_parameter_ids = {id(parameter) for parameter in embedding_parameters}
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in embedding_parameter_ids
    ]
    optimizer_groups: list[dict[str, Any]] = []
    if embedding_parameters:
        optimizer_groups.append(
            {
                "params": embedding_parameters,
                "weight_decay": embedding_weight_decay,
            }
        )
    optimizer_groups.append(
        {"params": other_parameters, "weight_decay": weight_decay}
    )
    optimizer = torch.optim.AdamW(optimizer_groups, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )
    loss_function = nn.BCEWithLogitsLoss()
    amp_enabled = torch_device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except TypeError:  # PyTorch versions where device is keyword-only/absent.
            scaler = torch.amp.GradScaler(enabled=amp_enabled)
    else:  # pragma: no cover - compatibility with PyTorch before torch.amp.
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    ema = _ExponentialMovingAverage(model, decay=ema_decay)

    best_auc = float("-inf")
    best_epoch = 0
    best_ema_state: dict[str, torch.Tensor] | None = None
    epochs_trained = 0
    non_improving_epochs = 0
    for epoch_index in range(epochs):
        model.train()
        for lookup_ids, numeric_values, missing_mask, labels in train_loader:
            lookup_ids = lookup_ids.to(torch_device, non_blocking=True)
            numeric_values = numeric_values.to(torch_device, non_blocking=True)
            missing_mask = missing_mask.to(torch_device, non_blocking=True)
            labels = labels.to(torch_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp_enabled
            ):
                logits = model(lookup_ids, numeric_values, missing_mask)
                loss = loss_function(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

        epochs_trained = epoch_index + 1
        with ema.scope(model):
            valid_predictions = _predict_lookup(model, valid_loader, torch_device)
        validation_auc = _validation_auc(valid_y, valid_predictions)
        if validation_auc > best_auc + 1.0e-12:
            best_auc = validation_auc
            best_epoch = epochs_trained
            best_ema_state = ema.state_dict()
            non_improving_epochs = 0
        else:
            non_improving_epochs += 1
            if non_improving_epochs >= patience:
                break

    if best_ema_state is None:  # Defensive: epochs is validated as positive.
        raise RuntimeError("fold training completed without an EMA checkpoint")
    ema.load_state_dict(best_ema_state)
    with ema.scope(model):
        valid_predictions = _predict_lookup(model, valid_loader, torch_device)
        test_predictions = _predict_lookup(model, test_loader, torch_device)

    diagnostics: dict[str, float | int | bool] = {
        "best_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "epochs_trained": int(epochs_trained),
        "early_stopped": bool(epochs_trained < epochs),
        "training_rows": int(len(train_arrays.lookup_ids)),
        "validation_rows": int(len(valid_arrays.lookup_ids)),
        "test_rows": int(len(test_arrays.lookup_ids)),
    }
    return valid_predictions, test_predictions, best_epoch, diagnostics
