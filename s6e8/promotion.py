"""Fail-closed promotion gates versus a named baseline experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from s6e8.contracts import DEFAULT_PROMOTION_GATES


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def raise_if_failed(self) -> None:
        if self.passed:
            return
        raise ValueError("promotion gates failed: " + "; ".join(self.failures))


def _require(metrics: dict[str, Any], key: str, fail_closed: bool, failures: list[str]) -> Any:
    if key in metrics and metrics[key] is not None:
        return metrics[key]
    if fail_closed:
        failures.append(f"missing required metric {key!r}")
    return None


def evaluate_promotion(
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None,
    gates: dict[str, Any] | None = None,
) -> GateResult:
    """Compare candidate metrics.json-like dicts to gates. Missing metrics fail closed."""
    spec = dict(DEFAULT_PROMOTION_GATES)
    if gates:
        spec.update(gates)
    fail_closed = bool(spec.get("fail_closed_on_missing", True))
    failures: list[str] = []
    checks: dict[str, Any] = {}

    cand_auc = _require(candidate, "oof_auc", fail_closed, failures)
    if cand_auc is None and "cv_auc" in candidate:
        cand_auc = candidate["cv_auc"]
        failures[:] = [f for f in failures if "oof_auc" not in f]
    checks["candidate_oof_auc"] = cand_auc

    if spec.get("require_oof_and_test", True):
        n_train = candidate.get("n_train")
        n_test = candidate.get("n_test")
        checks["n_train"] = n_train
        checks["n_test"] = n_test
        if fail_closed and not n_train:
            failures.append("candidate missing n_train")
        if fail_closed and not n_test:
            failures.append("candidate missing n_test")

    if candidate.get("diagnostic") is True:
        failures.append("diagnostic runs cannot be promoted")
        checks["diagnostic"] = True

    if baseline is not None:
        base_auc = baseline.get("oof_auc", baseline.get("cv_auc"))
        checks["baseline_oof_auc"] = base_auc
        if base_auc is None and fail_closed:
            failures.append("baseline missing oof_auc")
        elif cand_auc is not None and base_auc is not None:
            delta = float(cand_auc) - float(base_auc)
            checks["oof_auc_delta"] = delta
            min_delta = float(spec.get("min_oof_auc_delta", 0.0))
            if delta < min_delta:
                failures.append(
                    f"OOF AUC delta {delta:.6f} < min_delta {min_delta}"
                )

    max_ece = spec.get("max_ece")
    require_ece = bool(spec.get("require_ece", False))
    ece = None
    cal = candidate.get("calibration") or {}
    if isinstance(cal, dict):
        ece = cal.get("ece")
    if ece is None:
        ece = candidate.get("ece")
    checks["ece"] = ece
    if max_ece is not None:
        if ece is None:
            if require_ece and fail_closed:
                failures.append("missing ECE for max_ece gate")
        elif float(ece) > float(max_ece):
            failures.append(f"ECE {ece:.4f} > max_ece {max_ece}")

    min_slice_auc = spec.get("min_slice_auc")
    min_slice_n = int(spec.get("min_slice_n", 200))
    require_slices = bool(spec.get("require_slices", False))
    slices = candidate.get("slices") or {}
    slice_failures = []
    if require_slices and not slices and fail_closed:
        failures.append("missing slices for require_slices gate")
    if min_slice_auc is not None and slices:
        for col, table in slices.items():
            if col in {"overall_auc", "n", "hard_band"} or not isinstance(table, dict):
                continue
            for key, row in table.items():
                if not isinstance(row, dict):
                    continue
                n = int(row.get("n") or 0)
                auc = row.get("auc")
                if n < min_slice_n or auc is None:
                    continue
                if float(auc) < float(min_slice_auc):
                    slice_failures.append(f"{col}={key} auc={auc:.4f} n={n}")
        checks["slice_failures"] = slice_failures
        failures.extend(f"slice below floor: {item}" for item in slice_failures)

    passed = not failures
    return GateResult(passed=passed, failures=failures, checks=checks)
