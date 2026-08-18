"""Config-driven EDA. All conclusions are computed from the loaded tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from s6e8.features import add_engineered_features
from s6e8.runtime import detect_environment, get_git_commit, utc_timestamp

# Analysis policy (not data conclusions). Follow-ups fire only when stats cross these.
STRONG_AUC = 0.60
WEAK_AUC = 0.53
MISS_INFORMATIVE_AUC = 0.52
PSI_NOTABLE = 0.10
MISS_RATE_GAP = 0.02
CAT_RATE_SPREAD = 0.03
AGE_RATE_SPREAD = 0.08
ID_LEAK_AUC = 0.53
RATIO_DILUTION = 0.005
CORR_REDUNDANT = 0.70
SCATTER_DEFAULT = 8000
QUANTILE_BINS = 15
KS_SAMPLE = 80_000
EPS = 1e-9


@dataclass
class Finding:
    severity: str
    title: str
    phenomenon: str
    hypothesis: str
    modeling: str
    experiment: str
    score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EDAResult:
    meta: dict[str, Any]
    quality: dict[str, Any]
    target: dict[str, Any]
    univariate: pd.DataFrame
    missing: dict[str, Any]
    categoricals: dict[str, Any]
    relations: dict[str, Any]
    shift: dict[str, Any]
    outliers: dict[str, Any]
    leakage: dict[str, Any]
    engineering: dict[str, Any]
    candidates: pd.DataFrame
    followups: dict[str, Any]
    findings: list[Finding] = field(default_factory=list)
    sample_train: pd.DataFrame | None = None
    hist_bins: dict[str, dict[str, Any]] = field(default_factory=dict)


def _safe_auc(y: pd.Series, scores: pd.Series) -> dict[str, Any]:
    mask = scores.notna() & np.isfinite(scores.to_numpy(dtype=float, copy=False))
    n = int(mask.sum())
    if n < 200 or int(y[mask].nunique()) < 2:
        return {
            "auc": np.nan,
            "auc_raw": np.nan,
            "direction": None,
            "n": n,
            "coverage": float(n) / max(len(y), 1),
        }
    x = scores[mask].astype(float)
    yy = y[mask]
    raw = float(roc_auc_score(yy, x))
    flipped = float(roc_auc_score(yy, -x))
    if raw >= flipped:
        return {
            "auc": raw,
            "auc_raw": raw,
            "direction": "+",
            "n": n,
            "coverage": float(n) / len(y),
        }
    return {
        "auc": flipped,
        "auc_raw": raw,
        "direction": "-",
        "n": n,
        "coverage": float(n) / len(y),
    }


def _index_key(value: Any) -> str:
    if value is None:
        return "__NA__"
    try:
        if pd.isna(value):
            return "__NA__"
    except (TypeError, ValueError):
        pass
    return str(value)


def _psi_categorical(p: pd.Series, q: pd.Series) -> float:
    p2 = p.copy()
    q2 = q.copy()
    p2.index = [_index_key(i) for i in p2.index]
    q2.index = [_index_key(i) for i in q2.index]
    p2 = p2.groupby(level=0).sum()
    q2 = q2.groupby(level=0).sum()
    keys = sorted(set(p2.index).union(q2.index))
    a = np.array([float(p2.get(k, 0.0)) for k in keys], dtype=float)
    b = np.array([float(q2.get(k, 0.0)) for k in keys], dtype=float)
    a = np.clip(a, 1e-4, None)
    b = np.clip(b, 1e-4, None)
    a = a / a.sum()
    b = b / b.sum()
    return float(np.sum((b - a) * np.log(b / a)))


def _psi_numeric(train_s: pd.Series, test_s: pd.Series, n_bins: int = 10) -> float:
    a = train_s.dropna().to_numpy(dtype=float)
    b = test_s.dropna().to_numpy(dtype=float)
    if len(a) < 50 or len(b) < 50:
        return float("nan")
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(a, qs))
    if len(edges) < 3:
        return 0.0
    e_cnt, _ = np.histogram(a, bins=edges)
    t_cnt, _ = np.histogram(b, bins=edges)
    e = np.clip(e_cnt / max(e_cnt.sum(), 1), 1e-4, None)
    t = np.clip(t_cnt / max(t_cnt.sum(), 1), 1e-4, None)
    e = e / e.sum()
    t = t / t.sum()
    return float(np.sum((t - e) * np.log(t / e)))


def _quantile_target_table(x: pd.Series, y: pd.Series, n_bins: int = QUANTILE_BINS) -> pd.DataFrame:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < n_bins * 20 or d["x"].nunique() < 4:
        return pd.DataFrame(columns=["bin", "left", "right", "mid", "rate", "n"])
    d["bin"] = pd.qcut(d["x"], n_bins, duplicates="drop")
    rows = []
    for interval, g in d.groupby("bin", observed=True):
        rows.append(
            {
                "bin": str(interval),
                "left": float(interval.left),
                "right": float(interval.right),
                "mid": float(g["x"].mean()),
                "rate": float(g["y"].mean()),
                "n": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def _best_threshold_stump(x: pd.Series, y: pd.Series, n_grid: int = 25) -> dict[str, Any] | None:
    mask = x.notna()
    if int(mask.sum()) < 1000:
        return None
    xx = x[mask].to_numpy(dtype=float)
    yy = y[mask].to_numpy()
    qs = np.linspace(0.05, 0.95, n_grid)
    grid = np.unique(np.quantile(xx, qs))
    best: dict[str, Any] | None = None
    for t in grid:
        pred = (xx >= t).astype(float)
        if pred.min() == pred.max():
            continue
        auc = float(roc_auc_score(yy, pred))
        acc = float((pred == yy).mean())
        rec = {"threshold": float(t), "auc": auc, "accuracy": acc, "pred_rate": float(pred.mean())}
        if best is None or auc > best["auc"]:
            best = rec
    return best


def _bitpack_missing_patterns(df: pd.DataFrame, cols: list[str], top_k: int = 12) -> list[dict[str, Any]]:
    miss = df[cols].isna().to_numpy()
    n_cols = miss.shape[1]
    keys = miss.dot(1 << np.arange(n_cols, dtype=np.uint64))
    vc = pd.Series(keys).value_counts().head(top_k)
    out = []
    for key, cnt in vc.items():
        bits = int(key)
        missing_cols = [cols[i] for i in range(n_cols) if bits & (1 << i)]
        out.append(
            {
                "n_missing": len(missing_cols),
                "missing_cols": missing_cols,
                "count": int(cnt),
                "share": float(cnt) / len(df),
                "complete": bits == 0,
            }
        )
    return out


def _fmt_pct(x: float, digits: int = 1) -> str:
    if x != x:
        return "NA"
    return f"{100.0 * x:.{digits}f}%"


def _fmt_auc(x: float) -> str:
    if x != x:
        return "NA"
    return f"{x:.3f}"


def _severity_rank(sev: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(sev, 9)


class EDASession:
    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        config: dict[str, Any],
        sample_size: int = SCATTER_DEFAULT,
    ) -> None:
        self.train = train
        self.test = test
        self.config = config
        self.sample_size = int(sample_size)
        self.seed = int(config["experiment"]["seed"])
        self.id_col = config["competition"]["id_col"]
        self.target_col = config["competition"]["target"]
        self.numeric = list(config["features"]["numeric"])
        self.categorical = list(config["features"]["categorical"])
        self.feature_cols = [c for c in self.numeric + self.categorical if c in train.columns]
        self.y = train[self.target_col].astype("int64")
        self.rng = np.random.default_rng(self.seed)
        self.findings: list[Finding] = []
        self.engineered = add_engineered_features(train, config)
        self.test_engineered = add_engineered_features(test, config)

    def run(self) -> EDAResult:
        quality = self._quality()
        target = self._target()
        univariate = self._univariate()
        missing = self._missing()
        cats = self._categoricals()
        relations = self._relations(univariate)
        shift = self._shift()
        outliers = self._outliers()
        leakage = self._leakage()
        engineering = self._engineering(univariate)
        candidates = self._candidates(univariate)
        followups = self._followups(univariate, relations)
        sample = self._sample()
        hist_bins = self._hist_bins(univariate)
        self._build_findings(
            quality=quality,
            target=target,
            univariate=univariate,
            missing=missing,
            cats=cats,
            relations=relations,
            shift=shift,
            outliers=outliers,
            leakage=leakage,
            engineering=engineering,
            candidates=candidates,
            followups=followups,
        )
        self.findings.sort(key=lambda f: (_severity_rank(f.severity), -abs(f.score)))
        meta = {
            "experiment": self.config["experiment"]["name"],
            "seed": self.seed,
            "git_commit": get_git_commit(),
            "timestamp": utc_timestamp(),
            "environment": detect_environment(),
            "config_path": self.config.get("_config_path"),
            "data_version": self.config["experiment"].get("data_version"),
            "feature_version": self.config["experiment"].get("feature_version"),
            "n_train": int(len(self.train)),
            "n_test": int(len(self.test)),
            "sample_size": int(min(self.sample_size, len(self.train))),
            "numeric": self.numeric,
            "categorical": self.categorical,
            "engineered_flags": dict(self.config["features"].get("engineering") or {}),
        }
        return EDAResult(
            meta=meta,
            quality=quality,
            target=target,
            univariate=univariate,
            missing=missing,
            categoricals=cats,
            relations=relations,
            shift=shift,
            outliers=outliers,
            leakage=leakage,
            engineering=engineering,
            candidates=candidates,
            followups=followups,
            findings=self.findings,
            sample_train=sample,
            hist_bins=hist_bins,
        )

    def _quality(self) -> dict[str, Any]:
        train_null = {c: float(self.train[c].isna().mean()) for c in self.train.columns}
        test_null = {c: float(self.test[c].isna().mean()) for c in self.test.columns}
        dtypes = {c: str(self.train[c].dtype) for c in self.train.columns}
        int_like = {}
        nunique = {}
        bounds = {}
        for c in self.numeric:
            if c not in self.train.columns:
                continue
            s = self.train[c].dropna()
            nunique[c] = int(s.nunique())
            if len(s) == 0:
                int_like[c] = float("nan")
                continue
            arr = s.to_numpy(dtype=float)
            int_like[c] = float(np.mean(np.isclose(arr, np.round(arr), atol=1e-8)))
            bounds[c] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "p01": float(s.quantile(0.01)),
                "p50": float(s.median()),
                "p99": float(s.quantile(0.99)),
                "at_min": float(np.mean(np.isclose(arr, arr.min()))),
                "at_max": float(np.mean(np.isclose(arr, arr.max()))),
            }
        train_dup = int(self.train.duplicated().sum())
        test_dup = int(self.test.duplicated().sum())
        feat_dup_train = int(self.train[self.feature_cols].duplicated().sum()) if self.feature_cols else 0
        feat_dup_test = int(self.test[self.feature_cols].duplicated().sum()) if self.feature_cols else 0
        return {
            "train_null": train_null,
            "test_null": test_null,
            "dtypes": dtypes,
            "int_like": int_like,
            "nunique": nunique,
            "bounds": bounds,
            "train_full_duplicates": train_dup,
            "test_full_duplicates": test_dup,
            "train_feature_duplicates": feat_dup_train,
            "test_feature_duplicates": feat_dup_test,
            "train_mean_n_missing": float(self.train[self.feature_cols].isna().sum(axis=1).mean()),
            "test_mean_n_missing": float(self.test[self.feature_cols].isna().sum(axis=1).mean()),
            "complete_case_share": float((~self.train[self.feature_cols].isna().any(axis=1)).mean()),
            "missing_patterns": _bitpack_missing_patterns(self.train, self.feature_cols),
        }

    def _target(self) -> dict[str, Any]:
        y = self.y
        counts = y.value_counts(dropna=False).sort_index()
        pos = float(y.mean())
        return {
            "name": self.target_col,
            "n": int(len(y)),
            "n_positive": int((y == 1).sum()),
            "n_negative": int((y == 0).sum()),
            "n_null": int(y.isna().sum()),
            "positive_rate": pos,
            "negative_rate": float(1.0 - pos),
            "nunique": int(y.nunique(dropna=False)),
            "counts": {str(k): int(v) for k, v in counts.items()},
            "majority_baseline_auc": 0.5,
            "naive_constant": pos,
        }

    def _univariate(self) -> pd.DataFrame:
        rows = []
        for c in self.numeric:
            stats = _safe_auc(self.y, self.train[c])
            spearman = float(self.train[c].corr(self.y, method="spearman")) if c in self.train else float("nan")
            miss = float(self.train[c].isna().mean())
            miss_auc = _safe_auc(self.y, self.train[c].isna().astype(float))["auc"]
            rows.append(
                {
                    "feature": c,
                    "kind": "numeric",
                    "auc": stats["auc"],
                    "auc_raw": stats["auc_raw"],
                    "direction": stats["direction"],
                    "n": stats["n"],
                    "coverage": stats["coverage"],
                    "spearman": spearman,
                    "missing_rate": miss,
                    "missing_auc": miss_auc,
                    "source": "raw",
                }
            )
        for c in self.categorical:
            miss = float(self.train[c].isna().mean())
            miss_auc = _safe_auc(self.y, self.train[c].isna().astype(float))["auc"]
            rates = self.train.groupby(c, dropna=False)[self.target_col].mean()
            spread = float(rates.max() - rates.min()) if len(rates) else 0.0
            # Use rate-encoding for a cheap univariate AUC of the categorical.
            mapped = self.train[c].map(self.train.groupby(c)[self.target_col].mean())
            # In-sample encoding overstates AUC; still useful as an upper bound.
            cat_auc = _safe_auc(self.y, pd.to_numeric(mapped, errors="coerce"))["auc"]
            rows.append(
                {
                    "feature": c,
                    "kind": "categorical",
                    "auc": cat_auc,
                    "auc_raw": cat_auc,
                    "direction": "+",
                    "n": int(self.train[c].notna().sum()),
                    "coverage": float(self.train[c].notna().mean()),
                    "spearman": np.nan,
                    "missing_rate": miss,
                    "missing_auc": miss_auc,
                    "rate_spread": spread,
                    "source": "raw",
                    "in_sample_encoding": True,
                }
            )
        id_stats = _safe_auc(self.y, self.train[self.id_col])
        rows.append(
            {
                "feature": self.id_col,
                "kind": "id",
                "auc": id_stats["auc"],
                "auc_raw": id_stats["auc_raw"],
                "direction": id_stats["direction"],
                "n": id_stats["n"],
                "coverage": 1.0,
                "spearman": float(self.train[self.id_col].corr(self.y, method="spearman")),
                "missing_rate": 0.0,
                "missing_auc": 0.5,
                "source": "id",
            }
        )
        df = pd.DataFrame(rows)
        return df.sort_values("auc", ascending=False, na_position="last").reset_index(drop=True)

    def _missing(self) -> dict[str, Any]:
        rows = []
        for c in self.feature_cols:
            miss = self.train[c].isna()
            rate = float(miss.mean())
            y_miss = float(self.y[miss].mean()) if miss.any() else float("nan")
            y_obs = float(self.y[~miss].mean()) if (~miss).any() else float("nan")
            auc = _safe_auc(self.y, miss.astype(float))["auc"]
            rows.append(
                {
                    "feature": c,
                    "train_missing": rate,
                    "test_missing": float(self.test[c].isna().mean()) if c in self.test.columns else float("nan"),
                    "missing_auc": auc,
                    "target_when_missing": y_miss,
                    "target_when_observed": y_obs,
                    "delta_target": (y_miss - y_obs) if y_miss == y_miss and y_obs == y_obs else float("nan"),
                }
            )
        table = pd.DataFrame(rows).sort_values("train_missing", ascending=False)
        n_miss = self.train[self.feature_cols].isna().sum(axis=1)
        by_n = (
            pd.DataFrame({"n_missing": n_miss, "y": self.y})
            .groupby("n_missing")["y"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "target_rate", "count": "n"})
        )
        n_miss_auc = _safe_auc(self.y, n_miss.astype(float))
        max_miss_auc = float(table["missing_auc"].max()) if len(table) else 0.5
        return {
            "table": table,
            "by_n_missing": by_n,
            "n_missing_auc": n_miss_auc["auc"],
            "max_missing_indicator_auc": max_miss_auc,
            "informative": bool(max_miss_auc >= MISS_INFORMATIVE_AUC),
        }

    def _categoricals(self) -> dict[str, Any]:
        out: dict[str, Any] = {"tables": {}, "spreads": {}}
        for c in self.categorical:
            g = (
                self.train.groupby(c, dropna=False)[self.target_col]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "target_rate", "count": "n", c: "level"})
            )
            g["level"] = g["level"].astype(str)
            out["tables"][c] = g
            out["spreads"][c] = float(g["target_rate"].max() - g["target_rate"].min()) if len(g) else 0.0
            test_vc = self.test[c].value_counts(dropna=False, normalize=True) if c in self.test.columns else None
            train_vc = self.train[c].value_counts(dropna=False, normalize=True)
            if test_vc is not None:
                out.setdefault("psi", {})[c] = _psi_categorical(train_vc, test_vc)
        return out

    def _relations(self, univariate: pd.DataFrame) -> dict[str, Any]:
        num_present = [c for c in self.numeric if c in self.train.columns]
        corr = self.train[num_present].corr(method="pearson")
        spearman = self.train[num_present].corr(method="spearman")
        top = (
            univariate.loc[univariate["kind"].eq("numeric") & univariate["source"].eq("raw")]
            .sort_values("auc", ascending=False)
        )
        top_name = str(top.iloc[0]["feature"]) if len(top) else None
        daily = self.train.get("daily_screen_time_hours")
        social = self.train.get("social_media_hours")
        gaming = self.train.get("gaming_hours")
        work = self.train.get("work_study_hours")
        weekend = self.train.get("weekend_screen_time")
        consistency: dict[str, Any] = {}
        if daily is not None and social is not None and gaming is not None and work is not None:
            component_sum = social + gaming + work
            both = daily.notna() & social.notna() & gaming.notna() & work.notna()
            diff = daily - component_sum
            viol = float(((daily + 1e-6) < component_sum).mean())
            viol_obs = float(((daily[both] + 1e-6) < component_sum[both]).mean()) if both.any() else float("nan")
            other = diff.clip(lower=0)
            consistency = {
                "component_sum_auc": _safe_auc(self.y, component_sum)["auc"],
                "other_screen_auc": _safe_auc(self.y, other)["auc"],
                "other_screen_mean": float(other.mean()) if other.notna().any() else float("nan"),
                "other_screen_coverage": float(other.notna().mean()),
                "violation_rate_all_rows": viol,
                "violation_rate_complete_components": viol_obs,
                "n_complete_components": int(both.sum()),
                "daily_vs_components_corr": float(daily[both].corr(component_sum[both])) if both.sum() > 10 else float("nan"),
            }
            if weekend is not None:
                wboth = daily.notna() & weekend.notna()
                wr = (weekend / daily.replace(0, np.nan))
                consistency["weekend_daily_corr"] = float(daily[wboth].corr(weekend[wboth])) if wboth.sum() > 10 else float("nan")
                consistency["weekend_daily_ratio_median"] = float(wr.median())
                consistency["weekend_minus_daily_auc"] = _safe_auc(self.y, weekend - daily)["auc"]
        redundant = []
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                val = float(corr.loc[a, b])
                if abs(val) >= CORR_REDUNDANT:
                    redundant.append({"a": a, "b": b, "corr": val})
        return {
            "corr": corr,
            "spearman": spearman,
            "top_numeric": top_name,
            "consistency": consistency,
            "redundant_pairs": redundant,
        }

    def _shift(self) -> dict[str, Any]:
        rows = []
        rng = np.random.default_rng(self.seed)
        for c in self.numeric:
            tr = self.train[c].dropna()
            te = self.test[c].dropna()
            n_tr = min(len(tr), KS_SAMPLE)
            n_te = min(len(te), KS_SAMPLE)
            if n_tr < 50 or n_te < 50:
                ks_stat, ks_p = float("nan"), float("nan")
            else:
                tr_s = tr.iloc[rng.choice(len(tr), size=n_tr, replace=False)]
                te_s = te.iloc[rng.choice(len(te), size=n_te, replace=False)]
                res = ks_2samp(tr_s.to_numpy(dtype=float), te_s.to_numpy(dtype=float))
                ks_stat, ks_p = float(res.statistic), float(res.pvalue)
            miss_tr = float(self.train[c].isna().mean())
            miss_te = float(self.test[c].isna().mean())
            rows.append(
                {
                    "feature": c,
                    "kind": "numeric",
                    "ks": ks_stat,
                    "ks_pvalue": ks_p,
                    "psi_values": _psi_numeric(self.train[c], self.test[c]),
                    "mean_train": float(tr.mean()) if len(tr) else float("nan"),
                    "mean_test": float(te.mean()) if len(te) else float("nan"),
                    "missing_train": miss_tr,
                    "missing_test": miss_te,
                    "missing_gap": miss_te - miss_tr,
                    "psi_missing": _psi_categorical(
                        pd.Series({0: 1 - miss_tr, 1: miss_tr}),
                        pd.Series({0: 1 - miss_te, 1: miss_te}),
                    ),
                }
            )
        for c in self.categorical:
            miss_tr = float(self.train[c].isna().mean())
            miss_te = float(self.test[c].isna().mean())
            tr_vc = self.train[c].value_counts(dropna=False, normalize=True)
            te_vc = self.test[c].value_counts(dropna=False, normalize=True)
            rows.append(
                {
                    "feature": c,
                    "kind": "categorical",
                    "ks": np.nan,
                    "ks_pvalue": np.nan,
                    "psi_values": _psi_categorical(tr_vc, te_vc),
                    "mean_train": np.nan,
                    "mean_test": np.nan,
                    "missing_train": miss_tr,
                    "missing_test": miss_te,
                    "missing_gap": miss_te - miss_tr,
                    "psi_missing": _psi_categorical(
                        pd.Series({0: 1 - miss_tr, 1: miss_tr}),
                        pd.Series({0: 1 - miss_te, 1: miss_te}),
                    ),
                }
            )
        table = pd.DataFrame(rows).sort_values("psi_values", ascending=False)
        return {
            "table": table,
            "max_psi_values": float(table["psi_values"].max(skipna=True)),
            "max_abs_missing_gap": float(table["missing_gap"].abs().max(skipna=True)),
            "max_ks": float(table["ks"].max(skipna=True)) if table["ks"].notna().any() else 0.0,
        }

    def _outliers(self) -> dict[str, Any]:
        rows = []
        for c in self.numeric:
            s = self.train[c].dropna()
            if len(s) < 10:
                continue
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            out = (s < lo) | (s > hi)
            y_out = float(self.y.loc[s.index[out]].mean()) if out.any() else float("nan")
            y_in = float(self.y.loc[s.index[~out]].mean()) if (~out).any() else float("nan")
            rows.append(
                {
                    "feature": c,
                    "q1": q1,
                    "q3": q3,
                    "fence_low": lo,
                    "fence_high": hi,
                    "outlier_share": float(out.mean()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "target_outlier": y_out,
                    "target_inlier": y_in,
                }
            )
        table = pd.DataFrame(rows)
        return {
            "table": table,
            "any_material": bool((table["outlier_share"] > 0.01).any()) if len(table) else False,
        }

    def _leakage(self) -> dict[str, Any]:
        id_tr = self.train[self.id_col]
        id_te = self.test[self.id_col]
        overlap_ids = int(len(set(id_tr).intersection(set(id_te))))
        id_auc = _safe_auc(self.y, id_tr)
        q = pd.qcut(id_tr, 10, duplicates="drop")
        id_bins = (
            pd.DataFrame({"bin": q.astype(str), "y": self.y})
            .groupby("bin", observed=True)["y"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "target_rate", "count": "n"})
        )
        tr_h = pd.util.hash_pandas_object(self.train[self.feature_cols], index=False)
        te_h = pd.util.hash_pandas_object(self.test[self.feature_cols], index=False)
        common = set(tr_h).intersection(set(te_h))
        tr_hit = self.train.loc[tr_h.isin(common), [self.id_col, self.target_col] + self.feature_cols]
        te_hit = self.test.loc[te_h.isin(common), [self.id_col] + self.feature_cols]
        overlap_mostly_null = False
        if len(tr_hit):
            overlap_mostly_null = bool(
                tr_hit[self.feature_cols].isna().mean(axis=1).mean() >= 0.7
            )
        return {
            "train_id_min": int(id_tr.min()),
            "train_id_max": int(id_tr.max()),
            "test_id_min": int(id_te.min()),
            "test_id_max": int(id_te.max()),
            "train_id_unique": int(id_tr.nunique()),
            "test_id_unique": int(id_te.nunique()),
            "train_id_monotonic": bool(id_tr.is_monotonic_increasing),
            "test_id_monotonic": bool(id_te.is_monotonic_increasing),
            "id_overlap": overlap_ids,
            "id_auc": id_auc["auc"],
            "id_bins": id_bins,
            "feature_hash_overlap": int(len(common)),
            "overlap_train_rows": int(len(tr_hit)),
            "overlap_test_rows": int(len(te_hit)),
            "overlap_mostly_null": overlap_mostly_null,
            "overlap_preview": tr_hit.head(5).to_dict(orient="records"),
        }

    def _engineering(self, univariate: pd.DataFrame) -> dict[str, Any]:
        flags = dict(self.config["features"].get("engineering") or {})
        names = [
            "n_missing",
            "leisure_hours",
            "screen_sleep_ratio",
            "weekend_weekday_ratio",
            "notif_per_open",
            "other_screen_hours",
            "component_sum",
            "screen_imputed_weekend",
            "strong3_row_mean",
            "strong3_row_max",
            "or_usage_score",
        ]
        rows = []
        raw_auc = {
            r.feature: r.auc
            for r in univariate.itertuples(index=False)
            if r.kind == "numeric" and r.source == "raw"
        }
        components = {
            "leisure_hours": ["social_media_hours", "gaming_hours"],
            "screen_sleep_ratio": ["daily_screen_time_hours", "sleep_hours"],
            "weekend_weekday_ratio": ["weekend_screen_time", "daily_screen_time_hours"],
            "notif_per_open": ["notifications_per_day", "app_opens_per_day"],
            "n_missing": [],
            "other_screen_hours": ["daily_screen_time_hours"],
            "component_sum": ["social_media_hours", "gaming_hours", "work_study_hours"],
            "screen_imputed_weekend": ["daily_screen_time_hours", "weekend_screen_time"],
            "strong3_row_mean": [
                "daily_screen_time_hours",
                "weekend_screen_time",
                "social_media_hours",
            ],
            "strong3_row_max": [
                "daily_screen_time_hours",
                "weekend_screen_time",
                "social_media_hours",
            ],
            "or_usage_score": [
                "daily_screen_time_hours",
                "social_media_hours",
                "weekend_screen_time",
            ],
        }
        for name in names:
            if name not in self.engineered.columns:
                continue
            s = self.engineered[name]
            stats = _safe_auc(self.y, s)
            finite = s.replace([np.inf, -np.inf], np.nan)
            best_comp = None
            best_comp_auc = np.nan
            for c in components.get(name, []):
                if c in raw_auc and (best_comp is None or raw_auc[c] > best_comp_auc):
                    best_comp = c
                    best_comp_auc = raw_auc[c]
            delta = stats["auc"] - best_comp_auc if stats["auc"] == stats["auc"] and best_comp_auc == best_comp_auc else np.nan
            p99 = float(finite.quantile(0.99)) if finite.notna().any() else float("nan")
            vmax = float(finite.max()) if finite.notna().any() else float("nan")
            exploded = bool(vmax == vmax and p99 == p99 and vmax > 5 * max(p99, EPS))
            if name == "n_missing":
                verdict = "无信息" if (stats["auc"] != stats["auc"] or stats["auc"] < WEAK_AUC) else "弱信号"
            elif delta == delta and delta <= -RATIO_DILUTION:
                verdict = "稀释强信号"
            elif exploded:
                verdict = "比值尾部放大"
            elif delta == delta and delta > 0.005:
                verdict = "优于最强组分"
            else:
                verdict = "接近组分，增量有限"
            rows.append(
                {
                    "feature": name,
                    "enabled": bool(
                        flags.get(
                            {
                                "n_missing": "add_n_missing",
                                "leisure_hours": "add_leisure_hours",
                                "screen_sleep_ratio": "add_screen_sleep_ratio",
                                "weekend_weekday_ratio": "add_weekend_weekday_ratio",
                                "notif_per_open": "add_notif_per_open",
                                "other_screen_hours": "add_other_screen_hours",
                                "component_sum": "add_component_sum",
                                "screen_imputed_weekend": "add_screen_imputed_weekend",
                                "strong3_row_mean": "add_strong3_row_mean",
                                "strong3_row_max": "add_strong3_row_max",
                                "or_usage_score": "add_or_usage_score",
                            }[name],
                            False,
                        )
                    ),
                    "auc": stats["auc"],
                    "auc_raw": stats["auc_raw"],
                    "direction": stats["direction"],
                    "coverage": stats["coverage"],
                    "missing_rate": float(finite.isna().mean()),
                    "p99": p99,
                    "max": vmax,
                    "best_component": best_comp,
                    "best_component_auc": best_comp_auc,
                    "delta_vs_component": delta,
                    "exploded_tail": exploded,
                    "verdict": verdict,
                }
            )
        table = pd.DataFrame(rows)
        return {"flags": flags, "table": table}

    def _candidates(self, univariate: pd.DataFrame) -> pd.DataFrame:
        t = self.train
        y = self.y
        daily = t.get("daily_screen_time_hours")
        social = t.get("social_media_hours")
        gaming = t.get("gaming_hours")
        work = t.get("work_study_hours")
        weekend = t.get("weekend_screen_time")
        sleep = t.get("sleep_hours")
        feats: dict[str, pd.Series] = {}
        notes: dict[str, str] = {}
        if daily is not None and social is not None and gaming is not None and work is not None:
            other = (daily - social - gaming - work).clip(lower=0)
            feats["other_screen_hours"] = other
            notes["other_screen_hours"] = "daily - social - gaming - work（负值裁 0）"
            feats["component_sum"] = social + gaming + work
            notes["component_sum"] = "social + gaming + work"
        if daily is not None and social is not None:
            feats["social_share"] = social / (daily + EPS)
            notes["social_share"] = "social_media_hours / daily_screen_time_hours"
        if daily is not None and work is not None:
            feats["work_share"] = work / (daily + EPS)
            notes["work_share"] = "work_study_hours / daily_screen_time_hours"
        if daily is not None and gaming is not None:
            feats["gaming_share"] = gaming / (daily + EPS)
            notes["gaming_share"] = "gaming_hours / daily_screen_time_hours"
        if daily is not None and weekend is not None:
            wr = float((weekend / daily.replace(0, np.nan)).median())
            if wr == wr and wr > 0:
                feats["screen_imputed_weekend"] = daily.fillna(weekend / wr)
                notes["screen_imputed_weekend"] = f"用 weekend / {wr:.3f} 回填缺失的 daily_screen"
                feats["weekend_daily_ratio"] = weekend / (daily + EPS)
                notes["weekend_daily_ratio"] = "与现有 weekend_weekday_ratio 同类，作对照"
        strong = (
            univariate.loc[univariate["kind"].eq("numeric") & univariate["source"].eq("raw")]
            .sort_values("auc", ascending=False)["feature"]
            .head(3)
            .tolist()
        )
        if strong:
            scaled = []
            for c in strong:
                s = t[c]
                med = float(s.median())
                if med == 0 or med != med:
                    scaled.append(s)
                else:
                    scaled.append(s / med)
            stacked = pd.concat(scaled, axis=1)
            feats["strong3_row_mean"] = stacked.mean(axis=1)
            notes["strong3_row_mean"] = f"最强三列（按中位数缩放后）行均值: {', '.join(strong)}"
            feats["strong3_row_max"] = stacked.max(axis=1)
            notes["strong3_row_max"] = f"最强三列（按中位数缩放后）行最大: {', '.join(strong)}"
        if daily is not None and sleep is not None:
            feats["screen_plus_sleep"] = daily + (9.0 - sleep)
            notes["screen_plus_sleep"] = "screen + (9 - sleep)，检验睡眠是否提供增量"

        rows = []
        raw_best = float(
            univariate.loc[univariate["kind"].eq("numeric") & univariate["source"].eq("raw"), "auc"].max()
        )
        for name, s in feats.items():
            stats = _safe_auc(y, s)
            rows.append(
                {
                    "feature": name,
                    "auc": stats["auc"],
                    "auc_raw": stats["auc_raw"],
                    "direction": stats["direction"],
                    "coverage": stats["coverage"],
                    "missing_rate": float(s.replace([np.inf, -np.inf], np.nan).isna().mean()),
                    "delta_vs_best_raw": stats["auc"] - raw_best if stats["auc"] == stats["auc"] else np.nan,
                    "note": notes.get(name, ""),
                    "beats_best_raw": bool(stats["auc"] == stats["auc"] and stats["auc"] > raw_best + 0.002),
                }
            )
        df = pd.DataFrame(rows)
        if len(df):
            df = df.sort_values("auc", ascending=False).reset_index(drop=True)
        return df

    def _followups(self, univariate: pd.DataFrame, relations: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        num = univariate.loc[univariate["kind"].eq("numeric") & univariate["source"].eq("raw")].copy()
        top_rows = num.sort_values("auc", ascending=False).head(3)
        out["top_numeric"] = top_rows.to_dict(orient="records")
        quantile_tables: dict[str, pd.DataFrame] = {}
        stumps: dict[str, Any] = {}
        for rec in top_rows.itertuples(index=False):
            if rec.auc == rec.auc and rec.auc >= STRONG_AUC:
                quantile_tables[rec.feature] = _quantile_target_table(self.train[rec.feature], self.y)
                stumps[rec.feature] = _best_threshold_stump(self.train[rec.feature], self.y)
        out["quantile_tables"] = quantile_tables
        out["stumps"] = stumps

        top_name = str(top_rows.iloc[0]["feature"]) if len(top_rows) else None
        if top_name and float(top_rows.iloc[0]["auc"]) >= STRONG_AUC:
            miss = self.train[top_name].isna()
            fallback_rows = []
            if miss.any() and miss.mean() >= 0.02:
                yy = self.y[miss]
                for c in self.numeric + self.categorical:
                    if c == top_name or c not in self.train.columns:
                        continue
                    s = self.train.loc[miss, c]
                    if c in self.categorical:
                        mapped = s.map(self.train.groupby(c)[self.target_col].mean())
                        stats = _safe_auc(yy, pd.to_numeric(mapped, errors="coerce"))
                    else:
                        stats = _safe_auc(yy, s)
                    fallback_rows.append(
                        {
                            "feature": c,
                            "auc": stats["auc"],
                            "coverage": stats["coverage"],
                            "n": stats["n"],
                        }
                    )
                out["fallback_when_top_missing"] = {
                    "top_feature": top_name,
                    "n_missing": int(miss.sum()),
                    "missing_rate": float(miss.mean()),
                    "target_rate_when_missing": float(yy.mean()),
                    "table": pd.DataFrame(fallback_rows).sort_values("auc", ascending=False)
                    if fallback_rows
                    else pd.DataFrame(),
                }

        if "age" in self.train.columns:
            age = self.train["age"]
            age_tbl = (
                pd.DataFrame({"age": age, "y": self.y})
                .dropna(subset=["age"])
                .groupby("age")["y"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "target_rate", "count": "n"})
            )
            spread = float(age_tbl["target_rate"].max() - age_tbl["target_rate"].min()) if len(age_tbl) else 0.0
            out["age_table"] = age_tbl
            out["age_spread"] = spread
            if spread >= AGE_RATE_SPREAD and top_name and top_name != "age":
                tmp = self.train[[top_name, "age", self.target_col]].copy()
                tmp["_q"] = pd.qcut(tmp[top_name], 5, duplicates="drop")
                residual = (
                    tmp.dropna(subset=["age", "_q"])
                    .pivot_table(
                        index="age",
                        columns="_q",
                        values=self.target_col,
                        aggfunc="mean",
                        observed=True,
                    )
                )
                out["age_within_top_quintiles"] = residual

        if quantile_tables and top_name in quantile_tables:
            qt = quantile_tables[top_name]
            if len(qt) >= 4:
                jumps = qt["rate"].diff().abs()
                out["steepest_jump"] = {
                    "feature": top_name,
                    "from_mid": float(qt.loc[jumps.idxmax(), "mid"]) if jumps.notna().any() else None,
                    "delta": float(jumps.max()) if jumps.notna().any() else None,
                }

        cons = relations.get("consistency") or {}
        if cons.get("other_screen_auc") == cons.get("other_screen_auc"):
            other = (
                self.train["daily_screen_time_hours"]
                - self.train["social_media_hours"]
                - self.train["gaming_hours"]
                - self.train["work_study_hours"]
            ).clip(lower=0)
            out["other_screen_quantiles"] = _quantile_target_table(other, self.y, n_bins=12)
        return out

    def _sample(self) -> pd.DataFrame:
        cols = [c for c in [self.id_col, self.target_col] + self.numeric[:6] + self.categorical if c in self.train.columns]
        n = min(self.sample_size, len(self.train))
        idx = self.rng.choice(len(self.train), size=n, replace=False)
        return self.train.iloc[idx].loc[:, cols].reset_index(drop=True)

    def _hist_bins(self, univariate: pd.DataFrame) -> dict[str, dict[str, Any]]:
        top = (
            univariate.loc[univariate["kind"].eq("numeric") & univariate["source"].eq("raw")]
            .sort_values("auc", ascending=False)["feature"]
            .head(4)
            .tolist()
        )
        out: dict[str, dict[str, Any]] = {}
        for c in top:
            tr = self.train[c].dropna().to_numpy(dtype=float)
            te = self.test[c].dropna().to_numpy(dtype=float) if c in self.test.columns else np.array([])
            if len(tr) < 20:
                continue
            combined = np.concatenate([tr, te]) if len(te) else tr
            edges = np.histogram_bin_edges(combined, bins=36)
            ht, _ = np.histogram(tr, bins=edges, density=True)
            he, _ = np.histogram(te, bins=edges, density=True) if len(te) else (np.zeros_like(ht), None)
            centers = 0.5 * (edges[:-1] + edges[1:])
            out[c] = {
                "centers": centers.tolist(),
                "train_density": ht.tolist(),
                "test_density": he.tolist() if len(te) else [],
            }
        return out

    def _add(self, **kwargs: Any) -> None:
        self.findings.append(Finding(**kwargs))

    def _build_findings(
        self,
        *,
        quality: dict[str, Any],
        target: dict[str, Any],
        univariate: pd.DataFrame,
        missing: dict[str, Any],
        cats: dict[str, Any],
        relations: dict[str, Any],
        shift: dict[str, Any],
        outliers: dict[str, Any],
        leakage: dict[str, Any],
        engineering: dict[str, Any],
        candidates: pd.DataFrame,
        followups: dict[str, Any],
    ) -> None:
        pos = target["positive_rate"]
        maj = "正类" if pos >= 0.5 else "负类"
        self._add(
            severity="high" if abs(pos - 0.5) >= 0.15 else "medium",
            title="目标分布决定 AUC 解释方式",
            phenomenon=(
                f"训练集 {target['n']:,} 行中 `{self.target_col}=1` 共 {target['n_positive']:,} 行，"
                f"正类率 {_fmt_pct(pos)}。无缺失标签。常数预测 {pos:.4f} 的 ROC-AUC 为 0.5。"
            ),
            hypothesis=(
                f"{maj}占优，但这是排序问题而非准确率问题。把任务当成少数类识别（盲目过采样 / 以 0.5 为阈值）"
                "会优化错误目标。"
            ),
            modeling=(
                "LightGBM 用 `binary` + AUC 早停与正类率无冲突。评估只看概率排序。"
                f"提交文件应对齐 sample_submission 的 id 范围 [{leakage['test_id_min']}, {leakage['test_id_max']}]。"
            ),
            experiment="任何新实验先记录 OOF AUC，并与常数基线 0.5 及单列 stump 对照，而不是看 Accuracy。",
            score=abs(pos - 0.5),
        )

        num = univariate.loc[univariate["kind"].eq("numeric") & univariate["source"].eq("raw")]
        if len(num):
            best = num.iloc[0]
            worst = num.sort_values("auc").iloc[0]
            qt = followups.get("quantile_tables", {}).get(best["feature"])
            qtxt = ""
            if isinstance(qt, pd.DataFrame) and len(qt):
                qtxt = (
                    f" {QUANTILE_BINS} 分箱正类率从 {_fmt_pct(float(qt['rate'].iloc[0]))} "
                    f"升至 {_fmt_pct(float(qt['rate'].iloc[-1]))}。"
                )
            stump = followups.get("stumps", {}).get(best["feature"]) or {}
            stump_txt = ""
            if stump:
                stump_txt = (
                    f" 最佳单阈值 stump（≥ {stump['threshold']:.2f}）AUC={_fmt_auc(stump['auc'])}，"
                    f"低于连续值 AUC，说明关系是分级/非线性的，而不是单一硬切。"
                )
            sev = "high" if best["auc"] >= 0.75 else ("medium" if best["auc"] >= STRONG_AUC else "low")
            self._add(
                severity=sev,
                title=f"最强单变量信号：`{best['feature']}`",
                phenomenon=(
                    f"`{best['feature']}` 单变量 ROC-AUC={_fmt_auc(best['auc'])}"
                    f"（方向 {best['direction']}，覆盖 {_fmt_pct(best['coverage'])}，"
                    f"Spearman={best['spearman']:.3f}）。"
                    f"最弱数值列 `{worst['feature']}` AUC={_fmt_auc(worst['auc'])}。{qtxt}{stump_txt}"
                ),
                hypothesis=(
                    "标签与用量类变量高度单调相关，更像由屏幕/社交行为规则（或强相关潜变量）生成，"
                    "而不是与特征独立的心理量表。树模型会把大量分裂给该列。"
                ),
                modeling=(
                    "这是 ROC-AUC 的主杠杆：保留原始连续值，避免无根据的粗分箱。"
                    f"该列缺失率 {_fmt_pct(best['missing_rate'])}，缺失指示 AUC={_fmt_auc(best['missing_auc'])}，"
                    "缺失本身几乎不含标签信息，真正的风险是缺失时失去主信号。"
                ),
                experiment=(
                    f"做单列 `{best['feature']}` 的 LightGBM/stump 作为 AUC 下限；"
                    "完整模型应明显高于该下限。不要为了“特征多样性”削弱该列。"
                ),
                score=float(best["auc"] - 0.5),
            )

        fb = followups.get("fallback_when_top_missing")
        if fb and isinstance(fb.get("table"), pd.DataFrame) and len(fb["table"]):
            top2 = fb["table"].head(2)
            names = "、".join(
                f"`{r.feature}` (AUC={_fmt_auc(r.auc)})" for r in top2.itertuples(index=False)
            )
            self._add(
                severity="high",
                title=f"`{fb['top_feature']}` 缺失时仍有强替代变量",
                phenomenon=(
                    f"`{fb['top_feature']}` 缺失 {fb['n_missing']:,} 行（{_fmt_pct(fb['missing_rate'])}），"
                    f"缺失子集正类率 {_fmt_pct(fb['target_rate_when_missing'])}。"
                    f"在该子集上最强替代为 {names}。"
                ),
                hypothesis="主信号缺失时，高度相关的周末时长/社交时长可以近似同一潜变量。",
                modeling="树模型会自动走替代分裂；显式回填或 `strong3_row_mean` 可能让线性模型/浅树更稳，对深度树未必必要。",
                experiment="新增实验只加 `screen_imputed_weekend` 或 `strong3_row_mean`，对比 baseline OOF AUC，其它超参保持不变。",
                score=float(top2.iloc[0]["auc"] - 0.5) if len(top2) else 0.0,
            )

        if float(missing["max_missing_indicator_auc"]) < MISS_INFORMATIVE_AUC:
            self._add(
                severity="medium",
                title="缺失对标签近似无信息",
                phenomenon=(
                    f"各列缺失指示的最大 AUC={_fmt_auc(missing['max_missing_indicator_auc'])}；"
                    f"`n_missing` AUC={_fmt_auc(missing['n_missing_auc'])}。"
                    f"完整样本占比 {_fmt_pct(quality['complete_case_share'])}，"
                    f"行均缺失 {quality['train_mean_n_missing']:.2f} 个字段。"
                ),
                hypothesis="缺失机制相对标签接近完全随机（或至少不通过标签选择）。缺失更像数据生成/掩码过程，而不是成瘾本身的表现。",
                modeling=(
                    "LightGBM 原生处理 NaN 即可。`add_n_missing` 与 `add_missing_indicators` 对 AUC 预期贡献接近零；"
                    "保留它们几乎无害，但不应期待提升。"
                ),
                experiment="对照实验：`add_n_missing: false` 且不打开 missing indicators，验证 OOF AUC 是否持平。",
                score=abs(float(missing["max_missing_indicator_auc"]) - 0.5),
            )

        max_spread = max(cats.get("spreads", {}).values()) if cats.get("spreads") else 0.0
        if max_spread < CAT_RATE_SPREAD:
            detail = "；".join(
                f"`{k}` 正类率极差 {_fmt_pct(v)}" for k, v in cats.get("spreads", {}).items()
            )
            self._add(
                severity="low",
                title="类别特征主效应很弱",
                phenomenon=f"{detail}。In-sample 目标编码 AUC 也只是类别可分性的上限。",
                hypothesis="性别/压力/学业影响要么不是标签生成因子，要么效应被用量变量吸收。",
                modeling="可保留给树做浅交互，但不要把类别交互当成提分主攻方向。目标编码泄漏风险大于收益。",
                experiment="一次实验去掉全部类别列，看 OOF AUC 掉多少；若 <0.001 则可在后续模型中降权。",
                score=float(max_spread),
            )

        age_spread = float(followups.get("age_spread") or 0.0)
        if age_spread >= AGE_RATE_SPREAD:
            tbl = followups["age_table"]
            hi = tbl.loc[tbl["target_rate"].idxmax()]
            lo = tbl.loc[tbl["target_rate"].idxmin()]
            residual_note = ""
            if "age_within_top_quintiles" in followups:
                residual_note = " 在最强用量特征的五分位内，年龄正类率差并未消失，说明不完全是用量混杂。"
            age_auc_row = num.loc[num["feature"].eq("age"), "auc"]
            age_auc_txt = _fmt_auc(float(age_auc_row.iloc[0])) if len(age_auc_row) else "NA"
            self._add(
                severity="medium",
                title="年龄主效应弱，但存在非单调锯齿",
                phenomenon=(
                    f"年龄单变量 AUC={age_auc_txt}，"
                    f"但各整数年龄正类率极差 {_fmt_pct(age_spread)}"
                    f"（最高 age={hi['age']:.0f} 为 {_fmt_pct(float(hi['target_rate']))}，"
                    f"最低 age={lo['age']:.0f} 为 {_fmt_pct(float(lo['target_rate']))}）。{residual_note}"
                ),
                hypothesis="更像原始小样本被 GMM 放大后留下的年龄段基线差异，而不是真实的平滑年龄风险。线性 age 项会估偏。",
                modeling="树可以按具体年龄分裂，不必手工分箱。不要把 age 当连续线性风险。",
                experiment="保持 age 为数值（树自行切点）。若做线性模型，把 age 当类别或加样条，而不是一个斜率。",
                score=age_spread,
            )

        cons = relations.get("consistency") or {}
        if cons.get("other_screen_auc") == cons.get("other_screen_auc"):
            self._add(
                severity="high" if cons["other_screen_auc"] >= STRONG_AUC else "medium",
                title="屏幕时长在分量上内部一致，且“其它屏幕”有独立信号",
                phenomenon=(
                    f"当 social/gaming/work 均非缺失时，`daily_screen` 小于三者之和的比例为 "
                    f"{_fmt_pct(cons.get('violation_rate_complete_components', float('nan')))}；"
                    f"`other_screen_hours` 单变量 AUC={_fmt_auc(cons['other_screen_auc'])}，"
                    f"覆盖 {_fmt_pct(cons.get('other_screen_coverage', float('nan')))}。"
                    f"weekend 与 daily 相关 {cons.get('weekend_daily_corr', float('nan')):.3f}，"
                    f"中位比值 {cons.get('weekend_daily_ratio_median', float('nan')):.3f}。"
                ),
                hypothesis="日屏幕是分量加残差的生成式结构；残差（其它 App/未标注用途）仍与成瘾标签相关。",
                modeling="`other_screen_hours` 是可验证的新特征，且不替代 daily 本身。比值特征会丢掉水平信息。",
                experiment="只新增 `other_screen_hours`（其它配置不变）跑一次 CV，看 OOF AUC 是否高于 baseline。",
                score=float(cons["other_screen_auc"] - 0.5),
            )

        eng = engineering["table"]
        if len(eng):
            diluted = eng.loc[eng["verdict"].eq("稀释强信号")]
            useless = eng.loc[eng["verdict"].isin(["无信息", "比值尾部放大"])]
            if len(diluted):
                lines = "；".join(
                    f"`{r.feature}` AUC={_fmt_auc(r.auc)} vs 组分 `{r.best_component}` {_fmt_auc(r.best_component_auc)}"
                    for r in diluted.itertuples(index=False)
                )
                self._add(
                    severity="medium",
                    title="现有部分工程特征稀释了更强的原始信号",
                    phenomenon=lines + "。",
                    hypothesis="把弱变量加进强变量（或把两个高度相关的水平量相除）会降低单变量可分性，对树也可能增加噪声分裂。",
                    modeling="baseline 里这些开关可以关掉做消融。树仍能从原始列学到交互，不一定需要预先相加/相除。",
                    experiment="新建 YAML：关闭被判为稀释的 engineering 开关，只改这一处，比较 OOF AUC。",
                    score=float((-diluted["delta_vs_component"]).max()),
                )
            if len(useless):
                names = "、".join(f"`{x}`" for x in useless["feature"])
                self._add(
                    severity="low",
                    title="若干工程特征对 AUC 几乎无贡献",
                    phenomenon=f"{names} 的单变量 AUC 接近 0.5 或比值尾部被放大。",
                    hypothesis="缺失计数与标签独立；部分比值在分母接近 0 时产生极端值，对 ROC 排序无益。",
                    modeling="可保留以免破坏已有实验对比，但新实验不应继续堆同类比值。",
                    experiment="与上一实验合并：一次只关一个开关，避免无法归因。",
                    score=0.01,
                )

        if len(candidates):
            keep = candidates.head(3)
            txt = "；".join(
                f"`{r.feature}` AUC={_fmt_auc(r.auc)}（覆盖 {_fmt_pct(r.coverage)}）"
                for r in keep.itertuples(index=False)
            )
            best_raw = float(num["auc"].max()) if len(num) else 0.5
            self._add(
                severity="high" if float(keep.iloc[0]["auc"]) >= STRONG_AUC else "medium",
                title="新特征假设已用单变量 AUC 预筛",
                phenomenon=f"在全量训练集上即时计算（非 CV 模型）：{txt}。当前最强原始列 AUC={_fmt_auc(best_raw)}。",
                hypothesis="有增量的候选应满足：覆盖缺失行，或捕捉组分结构（份额/残差），而不是重复 daily 的单调变换。",
                modeling=(
                    "单变量 AUC 低于最强原始列并不等于无用（可在主特征缺失时补位，或提供正交方向）。"
                    "若 `delta_vs_best_raw` 明显为负且覆盖更差，则优先级低。"
                ),
                experiment=(
                    "按候选表从上到下，每次 YAML 只加 1 个特征："
                    + "、".join(f"`{x}`" for x in keep["feature"].head(3))
                    + "，用同一 seed / 同一 CV。"
                ),
                score=float(keep.iloc[0]["auc"] - 0.5),
            )

        if shift["max_psi_values"] < PSI_NOTABLE and shift["max_ks"] < 0.05:
            gap_tbl = shift["table"].assign(abs_gap=lambda d: d["missing_gap"].abs()).sort_values("abs_gap", ascending=False)
            top_gap = gap_tbl.head(3)
            gap_txt = "；".join(
                f"`{r.feature}` 缺失率 train {_fmt_pct(r.missing_train)} / test {_fmt_pct(r.missing_test)}"
                for r in top_gap.itertuples(index=False)
            )
            self._add(
                severity="medium" if shift["max_abs_missing_gap"] >= MISS_RATE_GAP else "low",
                title="取值分布几乎无 drift，但缺失率存在差异",
                phenomenon=(
                    f"数值列最大 KS={shift['max_ks']:.3f}，最大取值 PSI={shift['max_psi_values']:.3f}（均低于常用 0.10 阈值）。"
                    f"缺失率最大绝对差 { _fmt_pct(shift['max_abs_missing_gap']) }。{gap_txt}。"
                ),
                hypothesis="train/test 由同一生成过程切分；差异主要在掩码概率，而不是特征边缘分布。",
                modeling="不必做对抗验证优先。更应保证强特征缺失时的替代路径在 test 上同样可用。",
                experiment="无需专门做 shift 校正。若后续模型在 test 上异常，再查强特征缺失子集的预测校准。",
                score=float(shift["max_abs_missing_gap"]),
            )
        elif shift["max_psi_values"] >= PSI_NOTABLE:
            worst = shift["table"].iloc[0]
            self._add(
                severity="high",
                title="检测到可能影响泛化的分布偏移",
                phenomenon=(
                    f"`{worst['feature']}` 取值 PSI={worst['psi_values']:.3f}，KS={worst['ks']}。"
                ),
                hypothesis="test 的该列边缘分布与 train 不同，可能伤害依赖该列的模型。",
                modeling="检查是否只发生在弱特征；若发生在强特征，考虑分箱或稳健变换。",
                experiment="对该列做 train/test 分位对照，并在 OOF 与 public LB 差偏大时优先怀疑它。",
                score=float(worst["psi_values"]),
            )

        if not outliers["any_material"]:
            self._add(
                severity="low",
                title="IQR 意义下几乎没有离群点",
                phenomenon="多数数值列的 min/max 落在生成器硬边界内，1.5 IQR 外点占比普遍 <1%。",
                hypothesis="特征被截断或从有界仿真器抽样，winsorize 收益很小。",
                modeling="不要删“异常行”。边界堆积（若有）可当树的切点，不必单独做指示。",
                experiment="跳过outlier clipping 实验，把预算留给主信号与替代特征。",
                score=0.0,
            )
        else:
            ot = outliers["table"].sort_values("outlier_share", ascending=False).iloc[0]
            self._add(
                severity="low",
                title="存在少量 IQR 离群点",
                phenomenon=(
                    f"`{ot['feature']}` IQR 外占比 {_fmt_pct(float(ot['outlier_share']))}，"
                    f"离群子集正类率 {_fmt_pct(float(ot['target_outlier']))} vs 箱内 {_fmt_pct(float(ot['target_inlier']))}。"
                ),
                hypothesis="若离群正类率接近边界饱和区，它们只是分布尾部而非错误值。",
                modeling="树对尾部稳健；线性模型才需要 winsorize。",
                experiment="仅当该列进入线性 stacking 时再考虑截尾。",
                score=float(ot["outlier_share"]),
            )

        leak_score = abs(float(leakage["id_auc"]) - 0.5)
        overlap_note = (
            f"特征哈希重叠 {leakage['feature_hash_overlap']} 对"
            + ("，且重叠行高度缺失，更像掩码碰撞而非复制泄漏。" if leakage["overlap_mostly_null"] else "。")
        )
        self._add(
            severity="high" if leakage["id_auc"] >= ID_LEAK_AUC or leakage["id_overlap"] else "low",
            title="ID / 重复样本泄漏检查",
            phenomenon=(
                f"train id=[{leakage['train_id_min']}, {leakage['train_id_max']}] "
                f"（唯一 {leakage['train_id_unique']:,}，单调={leakage['train_id_monotonic']}）；"
                f"test id=[{leakage['test_id_min']}, {leakage['test_id_max']}]；"
                f"id 交集 {leakage['id_overlap']}；id 对标签 AUC={_fmt_auc(leakage['id_auc'])}。"
                f"全行重复 train={quality['train_full_duplicates']} / test={quality['test_full_duplicates']}。"
                f"{overlap_note}"
            ),
            hypothesis=(
                "id 是生成顺序下标，train 在前、test 在后。若 id-AUC≈0.5 且分箱正类率平坦，则不存在可用的时间泄漏。"
            ),
            modeling="禁止把 `id` 当特征。行级泄漏不是当前风险。",
            experiment="无需针对泄漏的模型实验。若未来加入 original 数据，再查 original↔train 重叠。",
            score=leak_score,
        )

        red = relations.get("redundant_pairs") or []
        if red:
            pair_txt = "；".join(f"`{p['a']}`–`{p['b']}` r={p['corr']:.2f}" for p in red[:4])
            self._add(
                severity="medium",
                title="强相关特征对：水平量重复，比值会丢信号",
                phenomenon=pair_txt + "。",
                hypothesis="周末时长很大程度上是日时长的缩放。比值近似常数加噪声，单变量 AUC 会明显弱于水平值。",
                modeling="同时保留两列给树做缺失替代是合理的；不要用比值替换它们。",
                experiment="关闭 `add_weekend_weekday_ratio`，保留两列原始值。",
                score=max(abs(p["corr"]) for p in red) - CORR_REDUNDANT,
            )


def run_eda(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
    sample_size: int = SCATTER_DEFAULT,
) -> EDAResult:
    return EDASession(train, test, config, sample_size=sample_size).run()
