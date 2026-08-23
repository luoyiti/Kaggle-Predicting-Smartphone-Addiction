"""Optional-torch hashed-entity residual MLP.

Distinct from the Lookup-Transformer (PR #11): there is no attention stack and
no per-value embedding table over the full exact-key vocabulary. Exact-value
strings are mapped through a **hashing trick** into a fixed number of buckets,
then a small residual MLP. Install torch to use ``model.name: entity_mlp``;
trainers raise ImportError when torch is missing.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def hashed_bucket_ids(values: pd.Series, n_buckets: int) -> np.ndarray:
    """Stable MD5 hashing trick. Missing → bucket 0 reserved for NA.

    Unique non-null tokens are hashed once and scattered back, so a 296k-row
    test frame does not pay a Python MD5 per cell.
    """
    if n_buckets < 2:
        raise ValueError("hash_buckets must be >= 2")
    usable = int(n_buckets) - 1
    out = np.zeros(len(values), dtype=np.int64)
    mask = values.isna().to_numpy()
    finite_idx = np.flatnonzero(~mask)
    if finite_idx.size == 0:
        return out
    tokens = values.astype("string").to_numpy()[finite_idx]
    uniques, inverse = np.unique(tokens, return_inverse=True)
    buckets = np.empty(len(uniques), dtype=np.int64)
    for i, token in enumerate(uniques):
        digest = hashlib.md5(str(token).encode("utf-8")).hexdigest()
        buckets[i] = 1 + (int(digest, 16) % usable)
    out[finite_idx] = buckets[inverse]
    return out


def _split_columns(X: pd.DataFrame, cat_cols: list[str]) -> tuple[list[str], list[str]]:
    cats = [c for c in cat_cols if c in X.columns]
    nums = [c for c in X.columns if c not in cats]
    return nums, cats


def fold_predict(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    X_test: pd.DataFrame,
    *,
    cat_cols: list[str],
    seed: int,
    accelerator: str,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Train one fold. Raises ImportError if torch is not installed."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(
            "Install torch to use model.name: entity_mlp. "
            "This backend is optional and skipped in CI when torch is absent."
        ) from exc

    params = dict(params or {})
    hidden = params.get("hidden_dims", [64, 32])
    if isinstance(hidden, int):
        hidden = [hidden]
    hidden_dims = [int(h) for h in hidden]
    embed_dim = int(params.get("embed_dim", 8))
    hash_buckets = int(params.get("hash_buckets", 256))
    dropout = float(params.get("dropout", 0.1))
    epochs = int(params.get("epochs", params.get("max_iter", 12)))
    batch_size = int(params.get("batch_size", 1024))
    lr = float(params.get("lr", params.get("learning_rate", 1e-3)))
    patience = int(params.get("patience", 3))
    weight_decay = float(params.get("weight_decay", 1e-4))

    num_cols, cats = _split_columns(X_tr, cat_cols)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    if num_cols:
        xtr_num = scaler.fit_transform(imputer.fit_transform(X_tr[num_cols])).astype(np.float32)
        xva_num = scaler.transform(imputer.transform(X_va[num_cols])).astype(np.float32)
        xte_num = scaler.transform(imputer.transform(X_test[num_cols])).astype(np.float32)
    else:
        xtr_num = np.zeros((len(X_tr), 0), dtype=np.float32)
        xva_num = np.zeros((len(X_va), 0), dtype=np.float32)
        xte_num = np.zeros((len(X_test), 0), dtype=np.float32)

    def cat_stack(frame: pd.DataFrame) -> np.ndarray | None:
        if not cats:
            return None
        parts = [hashed_bucket_ids(frame[c], hash_buckets) for c in cats]
        return np.stack(parts, axis=1).astype(np.int64)

    xtr_cat = cat_stack(X_tr)
    xva_cat = cat_stack(X_va)
    xte_cat = cat_stack(X_test)

    use_cuda = str(accelerator).lower() == "gpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(int(seed))
    if use_cuda:
        torch.cuda.manual_seed_all(int(seed))

    n_num = int(xtr_num.shape[1])
    n_cat = 0 if xtr_cat is None else int(xtr_cat.shape[1])
    in_dim = n_num + n_cat * embed_dim
    if in_dim <= 0:
        raise ValueError("entity_mlp needs at least one numeric or categorical column")

    class ResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.ModuleList(
                [nn.Embedding(hash_buckets, embed_dim) for _ in range(n_cat)]
            )
            layers: list[nn.Module] = []
            prev = in_dim
            for width in hidden_dims:
                layers.append(nn.Linear(prev, width))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev = width
            self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
            self.skip = nn.Linear(in_dim, prev) if prev != in_dim else nn.Identity()
            self.head = nn.Linear(prev, 1)

        def encode(self, nums: torch.Tensor, cat_ids: torch.Tensor | None) -> torch.Tensor:
            pieces: list[torch.Tensor] = []
            if n_num:
                pieces.append(nums)
            if cat_ids is not None and n_cat:
                pieces.extend(emb(cat_ids[:, i]) for i, emb in enumerate(self.embeddings))
            return torch.cat(pieces, dim=1)

        def forward(self, nums: torch.Tensor, cat_ids: torch.Tensor | None) -> torch.Tensor:
            encoded = self.encode(nums, cat_ids)
            return self.head(self.trunk(encoded) + self.skip(encoded)).squeeze(-1)

    model = ResidualMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    def as_num(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=device)

    def as_cat(arr: np.ndarray | None) -> torch.Tensor | None:
        if arr is None:
            return None
        return torch.tensor(arr, dtype=torch.int64, device=device)

    tr_num = as_num(xtr_num)
    va_num = as_num(xva_num)
    te_num = as_num(xte_num)
    tr_cat = as_cat(xtr_cat)
    va_cat = as_cat(xva_cat)
    te_cat = as_cat(xte_cat)
    tr_y = torch.tensor(np.asarray(y_tr, dtype=np.float32), device=device)

    if tr_cat is None:
        dataset = TensorDataset(tr_num, tr_y)
    else:
        dataset = TensorDataset(tr_num, tr_cat, tr_y)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, len(X_tr))),
        shuffle=True,
        drop_last=False,
    )

    best_state = None
    best_auc = -1.0
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            if tr_cat is None:
                batch_num, batch_y = batch
                logits = model(batch_num, None)
            else:
                batch_num, batch_cat, batch_y = batch
                logits = model(batch_num, batch_cat)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_prob = torch.sigmoid(model(va_num, va_cat)).detach().cpu().numpy()
        try:
            auc = float(roc_auc_score(y_va, va_prob))
        except ValueError:
            auc = float("nan")
        if np.isfinite(auc) and auc > best_auc + 1e-6:
            best_auc = auc
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va_pred = torch.sigmoid(model(va_num, va_cat)).detach().cpu().numpy()
        te_pred = torch.sigmoid(model(te_num, te_cat)).detach().cpu().numpy()
    return va_pred.astype(np.float64), te_pred.astype(np.float64), int(best_epoch)
