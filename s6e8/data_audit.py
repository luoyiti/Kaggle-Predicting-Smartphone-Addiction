"""Train/test schema, missingness, shift, and leakage audit (batch, not production drift dashboards)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from s6e8.contracts import (
    CATEGORICAL_COLUMNS,
    CATEGORICAL_VALUES,
    COMPONENT_SUM_PARTS,
    COMPONENT_SUM_TOTAL,
    ID_COL,
    NUMERIC_BOUNDS,
    NUMERIC_COLUMNS,
    TARGET,
    required_test_columns,
    required_train_columns,
)


def _psi(train: pd.Series, test: pd.Series, bins: int = 10) -> float:
    t = train.dropna().astype(float)
    s = test.dropna().astype(float)
    if t.empty or s.empty:
        return float("nan")
    edges = np.unique(np.quantile(t, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    t_hist, _ = np.histogram(t, bins=edges)
    s_hist, _ = np.histogram(s, bins=edges)
    p = np.clip(t_hist / max(t_hist.sum(), 1), 1e-4, None)
    q = np.clip(s_hist / max(s_hist.sum(), 1), 1e-4, None)
    return float(np.sum((p - q) * np.log(p / q)))


def _cat_psi(train: pd.Series, test: pd.Series) -> float:
    def rates(s: pd.Series) -> pd.Series:
        filled = s.astype("string").fillna("__NA__")
        return filled.value_counts(normalize=True)

    p = rates(train)
    q = rates(test)
    keys = sorted(set(p.index).union(q.index))
    a = np.array([float(p.get(k, 0.0)) for k in keys])
    b = np.array([float(q.get(k, 0.0)) for k in keys])
    a = np.clip(a, 1e-4, None)
    b = np.clip(b, 1e-4, None)
    a = a / a.sum()
    b = b / b.sum()
    return float(np.sum((a - b) * np.log(a / b)))


def audit_tables(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    psi_notable: float = 0.10,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    payload: dict[str, Any] = {"findings": findings}

    missing_train = [c for c in required_train_columns() if c not in train.columns]
    missing_test = [c for c in required_test_columns() if c not in test.columns]
    payload["missing_train_columns"] = missing_train
    payload["missing_test_columns"] = missing_test
    if missing_train:
        findings.append({"severity": "error", "title": "train schema gap", "detail": missing_train})
    if missing_test:
        findings.append({"severity": "error", "title": "test schema gap", "detail": missing_test})

    if ID_COL in train.columns:
        payload["train_id_unique"] = bool(train[ID_COL].is_unique)
        if not train[ID_COL].is_unique:
            findings.append({"severity": "error", "title": "duplicate train id"})
    if ID_COL in test.columns:
        payload["test_id_unique"] = bool(test[ID_COL].is_unique)
        overlap = set(train[ID_COL]).intersection(set(test[ID_COL])) if ID_COL in train.columns else set()
        payload["id_overlap"] = int(len(overlap))
        if overlap:
            findings.append({"severity": "error", "title": "train/test id overlap", "detail": len(overlap)})

    if TARGET in train.columns:
        y = train[TARGET]
        payload["positive_rate"] = float(y.mean())
        payload["n_train"] = int(len(train))
        payload["n_test"] = int(len(test))
        if ID_COL in train.columns:
            id_auc = float(roc_auc_score(y, train[ID_COL].astype(float)))
            payload["id_vs_label_auc"] = id_auc
            payload["id_vs_label_auc_flipped"] = max(id_auc, 1.0 - id_auc)
            if max(id_auc, 1.0 - id_auc) >= 0.53:
                findings.append(
                    {
                        "severity": "warn",
                        "title": "id may leak split structure vs label",
                        "detail": payload["id_vs_label_auc_flipped"],
                    }
                )

    miss_train = {}
    miss_test = {}
    miss_gap = {}
    for col in list(NUMERIC_COLUMNS) + list(CATEGORICAL_COLUMNS):
        if col in train.columns:
            miss_train[col] = float(train[col].isna().mean())
        if col in test.columns:
            miss_test[col] = float(test[col].isna().mean())
        if col in miss_train and col in miss_test:
            miss_gap[col] = abs(miss_train[col] - miss_test[col])
            if miss_gap[col] >= 0.02:
                findings.append(
                    {
                        "severity": "warn",
                        "title": f"missing-rate gap {col}",
                        "detail": miss_gap[col],
                    }
                )
    payload["missing_rate_train"] = miss_train
    payload["missing_rate_test"] = miss_test
    payload["missing_rate_gap"] = miss_gap

    bounds_hits: dict[str, int] = {}
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in train.columns:
            continue
        s = train[col].dropna()
        n_out = int(((s < lo) | (s > hi)).sum())
        bounds_hits[col] = n_out
        if n_out:
            findings.append(
                {"severity": "info", "title": f"out-of-bound {col}", "detail": n_out}
            )
    payload["numeric_bound_violations_train"] = bounds_hits

    unknown_cats: dict[str, list[str]] = {}
    for col, allowed in CATEGORICAL_VALUES.items():
        if col not in train.columns:
            continue
        observed = set(train[col].dropna().astype(str).unique())
        extra = sorted(observed - set(allowed))
        unknown_cats[col] = extra
        if extra:
            findings.append({"severity": "warn", "title": f"unexpected {col} levels", "detail": extra})
    payload["unknown_categories"] = unknown_cats

    impossible = 0
    parts_ok = all(c in train.columns for c in COMPONENT_SUM_PARTS) and COMPONENT_SUM_TOTAL in train.columns
    if parts_ok:
        total = train[COMPONENT_SUM_TOTAL]
        summed = train[list(COMPONENT_SUM_PARTS)].sum(axis=1, min_count=len(COMPONENT_SUM_PARTS))
        mask = total.notna() & summed.notna() & (total + 1e-6 < summed)
        impossible = int(mask.sum())
        payload["daily_lt_component_sum_rows"] = impossible
        if impossible:
            findings.append(
                {
                    "severity": "info",
                    "title": "daily_screen < social+gaming+work",
                    "detail": impossible,
                }
            )
    else:
        payload["daily_lt_component_sum_rows"] = None

    psi_num = {}
    for col in NUMERIC_COLUMNS:
        if col in train.columns and col in test.columns:
            psi_num[col] = _psi(train[col], test[col])
            if np.isfinite(psi_num[col]) and psi_num[col] >= psi_notable:
                findings.append({"severity": "warn", "title": f"numeric PSI {col}", "detail": psi_num[col]})
    psi_cat = {}
    for col in CATEGORICAL_COLUMNS:
        if col in train.columns and col in test.columns:
            psi_cat[col] = _cat_psi(train[col], test[col])
            if np.isfinite(psi_cat[col]) and psi_cat[col] >= psi_notable:
                findings.append({"severity": "warn", "title": f"categorical PSI {col}", "detail": psi_cat[col]})
    payload["psi_numeric"] = psi_num
    payload["psi_categorical"] = psi_cat

    payload["ok"] = not any(f["severity"] == "error" for f in findings)
    return payload
