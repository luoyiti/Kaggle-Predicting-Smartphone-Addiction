"""Self-contained Chinese HTML report for EDAResult. No CDN, no notebooks."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from s6e8.eda import EDAResult, Finding

PLOT_FONT = dict(family="Segoe UI, PingFang SC, Noto Sans SC, Microsoft YaHei, sans-serif", size=13)
PLOT_CONFIG = {"displaylogo": False, "responsive": True}
COLORS = {
    "pos": "#c0392b",
    "neg": "#2471a3",
    "train": "#1f6aa5",
    "test": "#d35400",
    "accent": "#1a5276",
    "weak": "#7f8c8d",
    "good": "#1e8449",
}


def _esc(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _fmt(x: Any, kind: str = "num") -> str:
    if x is None:
        return "—"
    try:
        if pd.isna(x):
            return "—"
    except (TypeError, ValueError):
        pass
    if kind == "pct":
        return f"{100.0 * float(x):.2f}%"
    if kind == "auc":
        return f"{float(x):.4f}"
    if kind == "int":
        return f"{int(x):,}"
    if kind == "float":
        return f"{float(x):.4f}"
    return str(x)


def _df_table(df: pd.DataFrame, fmt: dict[str, str] | None = None, max_rows: int = 40) -> str:
    if df is None or len(df) == 0:
        return '<p class="muted">无数据。</p>'
    view = df.head(max_rows).copy()
    fmt = fmt or {}
    thead = "".join(f"<th>{_esc(c)}</th>" for c in view.columns)
    body = []
    for row in view.itertuples(index=False, name=None):
        tds = []
        for col, val in zip(view.columns, row, strict=True):
            tds.append(f"<td>{_esc(_fmt(val, fmt.get(str(col), 'raw')))}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    extra = f'<p class="muted">仅展示前 {max_rows} 行，共 {len(df):,} 行。</p>' if len(df) > max_rows else ""
    return f'<div class="table-wrap"><table><thead><tr>{thead}</tr></thead><tbody>{"".join(body)}</tbody></table></div>{extra}'


def _base_layout(title: str, **kwargs: Any) -> dict[str, Any]:
    layout = dict(
        title=dict(text=title, font=dict(size=16)),
        font=PLOT_FONT,
        template="plotly_white",
        margin=dict(l=60, r=24, t=56, b=52),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    layout.update(kwargs)
    return layout


def _fig_target(result: EDAResult) -> go.Figure:
    t = result.target
    fig = go.Figure()
    fig.add_bar(
        x=["0 未成瘾", "1 成瘾"],
        y=[t["n_negative"], t["n_positive"]],
        marker_color=[COLORS["neg"], COLORS["pos"]],
        text=[_fmt(t["n_negative"], "int"), _fmt(t["n_positive"], "int")],
        textposition="outside",
        hovertemplate="%{x}<br>n=%{y:,}<extra></extra>",
    )
    fig.update_layout(
        **_base_layout(
            f"目标 `{t['name']}`：正类率 {100 * t['positive_rate']:.2f}%",
            yaxis_title="行数",
            showlegend=False,
        )
    )
    return fig


def _fig_univariate(result: EDAResult) -> go.Figure:
    df = result.univariate.copy()
    df = df[df["kind"] != "id"]
    df = df.sort_values("auc", ascending=True)
    colors = [COLORS["pos"] if (a == a and a >= 0.7) else (COLORS["accent"] if (a == a and a >= 0.6) else COLORS["weak"]) for a in df["auc"]]
    fig = go.Figure()
    fig.add_bar(
        x=df["auc"],
        y=df["feature"],
        orientation="h",
        marker_color=colors,
        customdata=np.stack(
            [
                df["kind"].astype(str),
                df["coverage"].fillna(0),
                df["missing_rate"].fillna(0),
            ],
            axis=1,
        ),
        hovertemplate="%{y}<br>AUC=%{x:.4f}<br>%{customdata[0]} · 覆盖=%{customdata[1]:.1%} · 缺失=%{customdata[2]:.1%}<extra></extra>",
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="#999")
    fig.update_layout(
        **_base_layout("单变量 ROC-AUC（全量数据；类别列为 in-sample 编码上限）", xaxis_title="AUC", height=max(420, 28 * len(df) + 120))
    )
    fig.update_xaxes(range=[0.45, max(0.95, float(np.nanmax(df["auc"].to_numpy())) + 0.03)])
    return fig


def _fig_quantile_rates(result: EDAResult) -> go.Figure | None:
    tables: dict[str, pd.DataFrame] = result.followups.get("quantile_tables") or {}
    if not tables:
        return None
    fig = go.Figure()
    for name, tbl in tables.items():
        if tbl is None or len(tbl) == 0:
            continue
        fig.add_scatter(
            x=tbl["mid"],
            y=tbl["rate"],
            mode="lines+markers",
            name=name,
            hovertemplate=name + "<br>x≈%{x:.3f}<br>正类率=%{y:.3f}<extra></extra>",
        )
    if not fig.data:
        return None
    fig.update_layout(
        **_base_layout("最强数值特征：分位数中点 vs 正类率（全量）", xaxis_title="特征取值（箱内均值）", yaxis_title="正类率")
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def _fig_missing(result: EDAResult) -> go.Figure:
    tbl = result.missing["table"]
    fig = go.Figure()
    fig.add_bar(x=tbl["feature"], y=tbl["train_missing"], name="train 缺失率", marker_color=COLORS["train"])
    fig.add_bar(x=tbl["feature"], y=tbl["test_missing"], name="test 缺失率", marker_color=COLORS["test"])
    fig.update_layout(
        **_base_layout("缺失率：train vs test", yaxis_title="缺失率", barmode="group", xaxis_tickangle=-28)
    )
    return fig


def _fig_hist_overlay(result: EDAResult) -> go.Figure | None:
    items = list(result.hist_bins.items())
    if not items:
        return None
    n = len(items)
    fig = make_subplots(rows=n, cols=1, subplot_titles=[k for k, _ in items], vertical_spacing=0.08)
    for i, (name, payload) in enumerate(items, start=1):
        centers = payload["centers"]
        fig.add_bar(
            x=centers,
            y=payload["train_density"],
            name="train",
            marker_color=COLORS["train"],
            opacity=0.75,
            showlegend=(i == 1),
            row=i,
            col=1,
        )
        if payload.get("test_density"):
            fig.add_bar(
                x=centers,
                y=payload["test_density"],
                name="test",
                marker_color=COLORS["test"],
                opacity=0.55,
                showlegend=(i == 1),
                row=i,
                col=1,
            )
    fig.update_layout(
        **_base_layout("最强数值特征密度（全量预分箱，非原始散点）", barmode="overlay", height=260 * n + 80)
    )
    return fig


def _fig_corr(result: EDAResult) -> go.Figure | None:
    corr = result.relations.get("corr")
    if corr is None or len(corr) == 0:
        return None
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            hovertemplate="%{y} vs %{x}<br>r=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("数值特征 Pearson 相关（成对删除缺失）", height=520, xaxis_tickangle=-30))
    return fig


def _fig_scatter(result: EDAResult) -> go.Figure | None:
    sample = result.sample_train
    if sample is None:
        return None
    xcol = "daily_screen_time_hours"
    ycol = "weekend_screen_time"
    if xcol not in sample.columns or ycol not in sample.columns:
        num_cols = [c for c in result.meta["numeric"] if c in sample.columns]
        if len(num_cols) < 2:
            return None
        xcol, ycol = num_cols[0], num_cols[1]
    tgt = result.target["name"]
    d = sample.dropna(subset=[xcol, ycol])
    if len(d) < 20:
        return None
    fig = go.Figure()
    for label, color, name in [(0, COLORS["neg"], "0"), (1, COLORS["pos"], "1")]:
        sl = d[d[tgt] == label] if tgt in d.columns else d.iloc[0:0]
        fig.add_scattergl(
            x=sl[xcol],
            y=sl[ycol],
            mode="markers",
            name=f"{tgt}={name}",
            marker=dict(size=5, opacity=0.35, color=color),
            hoverinfo="skip",
        )
    fig.update_layout(
        **_base_layout(
            f"抽样散点（n={len(d):,}，seed={result.meta['seed']}）：{xcol} vs {ycol}",
            xaxis_title=xcol,
            yaxis_title=ycol,
        )
    )
    return fig


def _fig_categoricals(result: EDAResult) -> go.Figure | None:
    tables = result.categoricals.get("tables") or {}
    if not tables:
        return None
    names = list(tables.keys())
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=names)
    for i, name in enumerate(names, start=1):
        tbl = tables[name]
        fig.add_bar(
            x=tbl["level"].astype(str),
            y=tbl["target_rate"],
            name=name,
            marker_color=COLORS["accent"],
            showlegend=False,
            customdata=tbl["n"],
            hovertemplate="%{x}<br>正类率=%{y:.3f}<br>n=%{customdata:,}<extra></extra>",
            row=1,
            col=i,
        )
    fig.update_layout(**_base_layout("类别水平的正类率（含缺失水平）", yaxis_title="正类率"))
    fig.update_yaxes(range=[0, 1])
    return fig


def _fig_age(result: EDAResult) -> go.Figure | None:
    tbl = result.followups.get("age_table")
    if tbl is None or len(tbl) == 0:
        return None
    fig = go.Figure()
    fig.add_bar(
        x=tbl["age"],
        y=tbl["target_rate"],
        marker_color=COLORS["accent"],
        customdata=tbl["n"],
        hovertemplate="age=%{x}<br>正类率=%{y:.3f}<br>n=%{customdata:,}<extra></extra>",
    )
    fig.update_layout(**_base_layout("各整数年龄的正类率（全量）", xaxis_title="age", yaxis_title="正类率"))
    return fig


def _fig_engineering(result: EDAResult) -> go.Figure | None:
    eng = result.engineering["table"]
    cand = result.candidates
    rows = []
    if len(eng):
        tmp = eng[["feature", "auc"]].copy()
        tmp["group"] = "现有工程特征"
        rows.append(tmp)
    if len(cand):
        tmp = cand[["feature", "auc"]].copy()
        tmp["group"] = "EDA 候选特征"
        rows.append(tmp)
    raw = result.univariate
    raw = raw.loc[raw["kind"].eq("numeric") & raw["source"].eq("raw"), ["feature", "auc"]].copy()
    raw["group"] = "原始数值"
    rows.append(raw)
    df = pd.concat(rows, ignore_index=True).dropna(subset=["auc"]).sort_values("auc")
    color_map = {"原始数值": COLORS["train"], "现有工程特征": COLORS["test"], "EDA 候选特征": COLORS["good"]}
    fig = go.Figure()
    for g, sl in df.groupby("group", sort=False):
        fig.add_bar(
            x=sl["auc"],
            y=sl["feature"],
            orientation="h",
            name=g,
            marker_color=color_map.get(str(g), COLORS["weak"]),
            hovertemplate="%{y}<br>AUC=%{x:.4f}<extra>" + str(g) + "</extra>",
        )
    fig.add_vline(x=0.5, line_dash="dash", line_color="#999")
    fig.update_layout(
        **_base_layout("原始 / 现有工程 / 新候选 的单变量 AUC", xaxis_title="AUC", barmode="overlay", height=max(480, 22 * len(df) + 140))
    )
    return fig


def _fig_fallback(result: EDAResult) -> go.Figure | None:
    fb = result.followups.get("fallback_when_top_missing")
    if not fb:
        return None
    tbl = fb.get("table")
    if tbl is None or len(tbl) == 0:
        return None
    sl = tbl.head(10).sort_values("auc")
    fig = go.Figure()
    fig.add_bar(
        x=sl["auc"],
        y=sl["feature"],
        orientation="h",
        marker_color=COLORS["accent"],
        hovertemplate="%{y}<br>AUC=%{x:.4f}<extra></extra>",
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="#999")
    fig.update_layout(
        **_base_layout(
            f"`{fb['top_feature']}` 缺失子集上的替代变量 AUC（n={fb['n_missing']:,}）",
            xaxis_title="AUC",
            height=max(360, 26 * len(sl) + 120),
        )
    )
    return fig


def _fig_id_bins(result: EDAResult) -> go.Figure | None:
    tbl = result.leakage.get("id_bins")
    if tbl is None or len(tbl) == 0:
        return None
    fig = go.Figure()
    fig.add_scatter(
        x=list(range(len(tbl))),
        y=tbl["target_rate"],
        mode="lines+markers",
        marker_color=COLORS["accent"],
        customdata=tbl["bin"],
        hovertemplate="%{customdata}<br>正类率=%{y:.4f}<extra></extra>",
        name="id 十分位正类率",
    )
    fig.add_hline(y=result.target["positive_rate"], line_dash="dash", line_color=COLORS["pos"], annotation_text="全局正类率")
    fig.update_layout(**_base_layout("id 十分位的正类率（泄漏探针）", xaxis_title="十分位序号", yaxis_title="正类率"))
    return fig


def _fig_n_missing(result: EDAResult) -> go.Figure | None:
    tbl = result.missing.get("by_n_missing")
    if tbl is None or len(tbl) == 0:
        return None
    fig = go.Figure()
    fig.add_bar(
        x=tbl["n_missing"],
        y=tbl["target_rate"],
        marker_color=COLORS["accent"],
        customdata=tbl["n"],
        hovertemplate="n_missing=%{x}<br>正类率=%{y:.3f}<br>n=%{customdata:,}<extra></extra>",
    )
    fig.update_layout(**_base_layout("行缺失个数 vs 正类率", xaxis_title="n_missing", yaxis_title="正类率"))
    return fig


def build_figures(result: EDAResult) -> list[tuple[str, str, go.Figure]]:
    specs: list[tuple[str, str, go.Figure | None]] = [
        ("target", "目标分布", _fig_target(result)),
        ("univariate", "单变量 AUC", _fig_univariate(result)),
        ("quantiles", "分箱正类率", _fig_quantile_rates(result)),
        ("missing", "缺失率对照", _fig_missing(result)),
        ("hist", "train/test 密度", _fig_hist_overlay(result)),
        ("corr", "相关矩阵", _fig_corr(result)),
        ("scatter", "抽样散点", _fig_scatter(result)),
        ("cats", "类别正类率", _fig_categoricals(result)),
        ("age", "年龄正类率", _fig_age(result)),
        ("eng", "工程特征对照", _fig_engineering(result)),
        ("fallback", "主特征缺失时的替代", _fig_fallback(result)),
        ("id", "id 泄漏探针", _fig_id_bins(result)),
        ("nmiss", "缺失个数", _fig_n_missing(result)),
    ]
    return [(k, title, fig) for k, title, fig in specs if fig is not None]


def _finding_card(f: Finding) -> str:
    sev = _esc(f.severity)
    return f"""
<article class="finding {sev}">
  <div class="finding-head"><span class="tag {sev}">{sev}</span><h3>{_esc(f.title)}</h3></div>
  <p><strong>现象</strong> — { _esc(f.phenomenon) }</p>
  <p><strong>解释/假设</strong> — { _esc(f.hypothesis) }</p>
  <p><strong>对建模的意义</strong> — { _esc(f.modeling) }</p>
  <p><strong>下一步实验建议</strong> — { _esc(f.experiment) }</p>
</article>
"""


def _exec_summary(result: EDAResult) -> str:
    cards = "".join(_finding_card(f) for f in result.findings)
    n_high = sum(1 for f in result.findings if f.severity == "high")
    n_med = sum(1 for f in result.findings if f.severity == "medium")
    n_low = sum(1 for f in result.findings if f.severity == "low")
    return f"""
<p>共生成 <strong>{len(result.findings)}</strong> 条数据驱动结论（high={n_high}，medium={n_med}，low={n_low}）。
条目按严重度与效应量排序；数字全部来自本次运行的全量统计，不是预设文案。</p>
{cards}
"""


def _meta_block(result: EDAResult) -> str:
    m = result.meta
    git = m.get("git_commit") or "（当前环境无法读取，未伪造）"
    return f"""
<dl class="meta">
  <dt>实验配置</dt><dd>{_esc(m.get("experiment"))} · {_esc(m.get("config_path"))}</dd>
  <dt>数据规模</dt><dd>train={_fmt(m["n_train"], "int")} 行 · test={_fmt(m["n_test"], "int")} 行</dd>
  <dt>随机种子</dt><dd>{_esc(m.get("seed"))}（散点抽样 n={_fmt(m.get("sample_size"), "int")}）</dd>
  <dt>Git commit</dt><dd><code>{_esc(git)}</code></dd>
  <dt>环境 / 时间</dt><dd>{_esc(m.get("environment"))} · {_esc(m.get("timestamp"))}</dd>
  <dt>版本标记</dt><dd>data={_esc(m.get("data_version"))} · features={_esc(m.get("feature_version"))}</dd>
</dl>
"""


def _quality_section(result: EDAResult) -> str:
    q = result.quality
    bounds = pd.DataFrame(q["bounds"]).T.reset_index().rename(columns={"index": "feature"})
    miss = result.missing["table"]
    pat = pd.DataFrame(q["missing_patterns"])
    if len(pat):
        pat["missing_cols"] = pat["missing_cols"].map(lambda xs: "（完整）" if not xs else ", ".join(xs))
    int_like = pd.DataFrame(
        {"feature": list(q["int_like"].keys()), "integer_share": list(q["int_like"].values()), "nunique": [q["nunique"].get(k) for k in q["int_like"]]}
    )
    return f"""
<h2 id="quality">数据质量</h2>
<p>完整样本占比 {_fmt(q["complete_case_share"], "pct")}；行均缺失 train={q["train_mean_n_missing"]:.3f}、test={q["test_mean_n_missing"]:.3f}。
全行重复 train={q["train_full_duplicates"]}、test={q["test_full_duplicates"]}；去 id/标签后的特征重复 train={q["train_feature_duplicates"]}、test={q["test_feature_duplicates"]}。</p>
<h3>缺失率</h3>
{_df_table(miss, {"train_missing": "pct", "test_missing": "pct", "missing_auc": "auc", "target_when_missing": "pct", "target_when_observed": "pct", "delta_target": "float"})}
<h3>高频缺失模式（bit-pack 计数，全量）</h3>
{_df_table(pat, {"count": "int", "share": "pct", "n_missing": "int"})}
<h3>数值范围与边界堆积</h3>
{_df_table(bounds, {"min": "float", "max": "float", "mean": "float", "std": "float", "p01": "float", "p50": "float", "p99": "float", "at_min": "pct", "at_max": "pct"})}
<h3>是否像整数编码</h3>
{_df_table(int_like, {"integer_share": "pct", "nunique": "int"})}
"""


def _target_section(result: EDAResult) -> str:
    t = result.target
    return f"""
<h2 id="target">目标变量</h2>
<p><code>{_esc(t["name"])}</code> 正类 { _fmt(t["n_positive"], "int") } / { _fmt(t["n"], "int") }
= {_fmt(t["positive_rate"], "pct")}。缺失标签 {t["n_null"]}。竞赛指标是 ROC-AUC，常数预测的 AUC 恒为 0.5；
sample_submission 若填写训练集正类率，只是校准后的常数，不含排序信息。</p>
"""


def _univ_section(result: EDAResult) -> str:
    df = result.univariate.copy()
    fmt = {
        "auc": "auc",
        "auc_raw": "auc",
        "n": "int",
        "coverage": "pct",
        "spearman": "float",
        "missing_rate": "pct",
        "missing_auc": "auc",
        "rate_spread": "pct",
    }
    qhtml = ""
    for name, tbl in (result.followups.get("quantile_tables") or {}).items():
        stump = (result.followups.get("stumps") or {}).get(name) or {}
        stump_txt = ""
        if stump:
            stump_txt = (
                f" 最佳单阈值 ≥ {_fmt(stump['threshold'], 'float')} 的 stump AUC={_fmt(stump['auc'], 'auc')}，"
                f"准确率={_fmt(stump['accuracy'], 'pct')}。"
            )
        qhtml += f"<h3>{_esc(name)} 分箱正类率</h3><p>{stump_txt}</p>"
        qhtml += _df_table(tbl, {"left": "float", "right": "float", "mid": "float", "rate": "pct", "n": "int"})
    return f"""
<h2 id="univariate">单变量信号（面向 ROC-AUC）</h2>
<p>数值列在非缺失行上计算 <code>max(AUC(x), AUC(−x))</code>。类别列的 AUC 使用 <strong>in-sample</strong> 目标均值编码，只作为可分性上限，不能当 CV 分数。</p>
{_df_table(df, fmt)}
{qhtml}
"""


def _missing_section(result: EDAResult) -> str:
    by_n = result.missing["by_n_missing"]
    fb = result.followups.get("fallback_when_top_missing")
    fb_html = ""
    if fb and isinstance(fb.get("table"), pd.DataFrame):
        fb_html = f"""
<h3>当 `{_esc(fb["top_feature"])}` 缺失时</h3>
<p>缺失 { _fmt(fb["n_missing"], "int") } 行（{_fmt(fb["missing_rate"], "pct")}），该子集正类率 {_fmt(fb["target_rate_when_missing"], "pct")}。</p>
{_df_table(fb["table"], {"auc": "auc", "coverage": "pct", "n": "int"})}
"""
    return f"""
<h2 id="missing">缺失机制</h2>
<p>缺失指示最大 AUC={_fmt(result.missing["max_missing_indicator_auc"], "auc")}；
<code>n_missing</code> AUC={_fmt(result.missing["n_missing_auc"], "auc")}。
若两者都接近 0.5，则缺失对标签几乎没有直接信息，但对强特征而言仍会造成“主信号缺席”。</p>
<h3>按行缺失个数</h3>
{_df_table(by_n, {"target_rate": "pct", "n": "int", "n_missing": "int"})}
{fb_html}
"""


def _cat_section(result: EDAResult) -> str:
    blocks = []
    for name, tbl in (result.categoricals.get("tables") or {}).items():
        spread = result.categoricals.get("spreads", {}).get(name, float("nan"))
        psi = (result.categoricals.get("psi") or {}).get(name, float("nan"))
        blocks.append(
            f"<h3>{_esc(name)}</h3><p>正类率极差={_fmt(spread, 'pct')}，train/test 水平 PSI={_fmt(psi, 'float')}。</p>"
            + _df_table(tbl, {"target_rate": "pct", "n": "int"})
        )
    age = result.followups.get("age_table")
    age_html = ""
    if age is not None:
        age_html = "<h3>age 明细</h3>" + _df_table(age, {"age": "float", "target_rate": "pct", "n": "int"})
        residual = result.followups.get("age_within_top_quintiles")
        if residual is not None and len(residual):
            show = residual.reset_index()
            age_html += "<p>在最强用量特征五分位内的年龄正类率（检验混杂）：</p>" + _df_table(show)
    return f"<h2 id='cats'>类别与年龄</h2>{''.join(blocks)}{age_html}"


def _rel_section(result: EDAResult) -> str:
    cons = result.relations.get("consistency") or {}
    cons_df = pd.DataFrame([cons]) if cons else pd.DataFrame()
    other_q = result.followups.get("other_screen_quantiles")
    other_html = ""
    if other_q is not None and len(other_q):
        other_html = "<h3>other_screen_hours 分箱正类率</h3>" + _df_table(
            other_q, {"left": "float", "right": "float", "mid": "float", "rate": "pct", "n": "int"}
        )
    red = pd.DataFrame(result.relations.get("redundant_pairs") or [])
    return f"""
<h2 id="relations">特征关系与内部一致性</h2>
<p>若 daily_screen 始终不小于 social+gaming+work，则数据更像“分量加总 + 残差”，残差本身可能携带标签信息。</p>
{_df_table(cons_df, {k: "float" for k in cons_df.columns})}
{other_html}
<h3>高相关对 (|r|≥0.70)</h3>
{_df_table(red, {"corr": "float"})}
"""


def _shift_section(result: EDAResult) -> str:
    tbl = result.shift["table"]
    return f"""
<h2 id="shift">Train / Test 差异</h2>
<p>最大取值 PSI={_fmt(result.shift["max_psi_values"], "float")}，最大 KS={_fmt(result.shift["max_ks"], "float")}，
缺失率最大绝对差={_fmt(result.shift["max_abs_missing_gap"], "pct")}。
PSI&lt;0.10 且 KS 很小通常表示边缘分布可对齐；此时应把注意力放在缺失率而不是做复杂的域适应。</p>
{_df_table(tbl, {"ks": "float", "ks_pvalue": "float", "psi_values": "float", "mean_train": "float", "mean_test": "float", "missing_train": "pct", "missing_test": "pct", "missing_gap": "pct", "psi_missing": "float"})}
"""


def _out_section(result: EDAResult) -> str:
    return f"""
<h2 id="outliers">异常值</h2>
{_df_table(result.outliers["table"], {"q1": "float", "q3": "float", "fence_low": "float", "fence_high": "float", "outlier_share": "pct", "min": "float", "max": "float", "target_outlier": "pct", "target_inlier": "pct"})}
"""


def _leak_section(result: EDAResult) -> str:
    L = result.leakage
    preview = pd.DataFrame(L.get("overlap_preview") or [])
    return f"""
<h2 id="leakage">泄漏与 ID 结构</h2>
<p>train id [{L["train_id_min"]}, {L["train_id_max"]}] 唯一={_fmt(L["train_id_unique"], "int")} 单调={L["train_id_monotonic"]}；
test id [{L["test_id_min"]}, {L["test_id_max"]}] 唯一={_fmt(L["test_id_unique"], "int")} 单调={L["test_id_monotonic"]}；
id 交集={L["id_overlap"]}；id→标签 AUC={_fmt(L["id_auc"], "auc")}。
特征哈希重叠 {L["feature_hash_overlap"]} 对（重叠行高度缺失={L["overlap_mostly_null"]}）。</p>
<h3>id 十分位正类率</h3>
{_df_table(L["id_bins"], {"target_rate": "pct", "n": "int"})}
<h3>哈希重叠 train 预览</h3>
{_df_table(preview)}
"""


def _eng_section(result: EDAResult) -> str:
    flags = result.engineering["flags"]
    flag_txt = ", ".join(f"{k}={v}" for k, v in flags.items())
    return f"""
<h2 id="engineering">现有特征工程评估</h2>
<p>当前 YAML <code>features.engineering</code>：{_esc(flag_txt)}。评估口径是<strong>单变量 AUC + 覆盖率 + 是否稀释最强组分</strong>，不是 CV 模型分数。没有 OOF 就不能声称某开关“提升了竞赛 AUC”。</p>
{_df_table(result.engineering["table"], {"auc": "auc", "auc_raw": "auc", "coverage": "pct", "missing_rate": "pct", "p99": "float", "max": "float", "best_component_auc": "auc", "delta_vs_component": "float"})}
<h3>新特征假设（全量单变量预筛）</h3>
<p>下列特征只在本次 EDA 内存中构造，<strong>尚未写入</strong> <code>s6e8/features.py</code>。建议每次实验只加入其中一个，并用同一 CV 验证。
<code>strong3_row_*</code> 的成员列来自同一份 train 的单变量 AUC 排名，组合 AUC 带有轻微选择乐观偏差，只能当作优先级，不能当作已验证的提分。</p>
{_df_table(result.candidates, {"auc": "auc", "auc_raw": "auc", "coverage": "pct", "missing_rate": "pct", "delta_vs_best_raw": "float"})}
"""


def _method_section(result: EDAResult) -> str:
    return f"""
<h2 id="method">方法、抽样与边界</h2>
<ul>
  <li>统计与 AUC 使用<strong>全量</strong> train / test；散点图仅使用 seed={result.meta["seed"]} 的 {result.meta["sample_size"]} 行，避免把几十万点写进 HTML。</li>
  <li>密度图先在全量上预分箱再绘图；相关矩阵成对删除缺失。</li>
  <li>KS 在最多 {80_000:,} 行子样本上计算以控制耗时，分位数由固定种子抽取。</li>
  <li>类别 AUC 为 in-sample 编码上限；数值 AUC 为缺失行删除后的双方向最大值。</li>
  <li>本报告<strong>不训练</strong> 5-fold LightGBM，因此没有任何竞赛 OOF AUC 数字。</li>
  <li>图表由 Plotly 内联 JS 渲染，无 CDN / 无 MathJax / 无外部字体依赖。</li>
</ul>
"""


CSS = """
:root {
  --bg: #f4f1ea;
  --card: #fffcf7;
  --ink: #1c2430;
  --muted: #5c6b7a;
  --line: #d9d2c5;
  --high: #9b2c2c;
  --medium: #9a6700;
  --low: #1f6a4d;
  --accent: #1a5276;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  font-family: "Segoe UI", "PingFang SC", "Noto Sans SC", "Microsoft YaHei", sans-serif;
  color: var(--ink);
  background: var(--bg);
  line-height: 1.55;
}
header.hero {
  background: linear-gradient(135deg, #13293d, #1a5276 55%, #1e8449);
  color: #fff;
  padding: 36px 28px 28px;
}
header.hero h1 { margin: 0 0 8px; font-size: 28px; }
header.hero p { margin: 0; opacity: 0.92; max-width: 920px; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 20px 18px 64px; }
nav.toc {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px 18px;
  margin: 18px 0;
}
nav.toc a { color: var(--accent); margin-right: 14px; text-decoration: none; }
nav.toc a:hover { text-decoration: underline; }
h2 { margin-top: 36px; padding-top: 8px; border-top: 1px solid var(--line); }
h3 { margin-top: 22px; }
.meta { display: grid; grid-template-columns: 140px 1fr; gap: 6px 12px; background: var(--card); padding: 16px; border-radius: 12px; border: 1px solid var(--line); }
.meta dt { color: var(--muted); font-weight: 600; }
.meta dd { margin: 0; }
.finding {
  background: var(--card);
  border: 1px solid var(--line);
  border-left-width: 6px;
  border-radius: 10px;
  padding: 12px 16px;
  margin: 12px 0 18px;
}
.finding.high { border-left-color: var(--high); }
.finding.medium { border-left-color: var(--medium); }
.finding.low { border-left-color: var(--low); }
.finding-head { display: flex; gap: 10px; align-items: center; }
.finding-head h3 { margin: 0; font-size: 17px; }
.tag { font-size: 12px; padding: 2px 8px; border-radius: 999px; color: #fff; text-transform: uppercase; }
.tag.high { background: var(--high); }
.tag.medium { background: var(--medium); }
.tag.low { background: var(--low); }
.table-wrap { overflow-x: auto; background: var(--card); border: 1px solid var(--line); border-radius: 10px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; white-space: nowrap; }
th { background: #efe8da; position: sticky; top: 0; }
.muted { color: var(--muted); }
.plot { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 8px; margin: 16px 0; }
code { background: #efe8da; padding: 1px 4px; border-radius: 4px; }
footer { color: var(--muted); margin-top: 40px; font-size: 13px; }
"""


def render_html(result: EDAResult) -> str:
    figures = build_figures(result)
    plot_parts = []
    for i, (key, title, fig) in enumerate(figures):
        div = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            include_mathjax=False,
            config=PLOT_CONFIG,
            div_id=f"plot-{key}",
        )
        plot_parts.append(f'<section class="plot" id="plot-wrap-{_esc(key)}"><h3>{_esc(title)}</h3>{div}</section>')

    toc_links = [
        ("summary", "执行摘要"),
        ("quality", "数据质量"),
        ("target", "目标"),
        ("univariate", "单变量 AUC"),
        ("missing", "缺失"),
        ("cats", "类别"),
        ("relations", "关系"),
        ("shift", "Train/Test"),
        ("outliers", "异常值"),
        ("leakage", "泄漏"),
        ("engineering", "特征工程"),
        ("method", "方法"),
    ]
    toc = " ".join(f'<a href="#{k}">{lab}</a>' for k, lab in toc_links)
    plotly_js = get_plotlyjs()
    body = f"""
<header class="hero">
  <h1>Playground S6E8 探索性数据分析</h1>
  <p>面向 ROC-AUC 的可复现诊断。结论由 <code>scripts/eda.py</code> 根据当前 train/test 动态生成；未训练完整模型，也未提交 Kaggle。</p>
</header>
<div class="wrap">
  {_meta_block(result)}
  <nav class="toc">{toc}</nav>
  <h2 id="summary">执行摘要</h2>
  {_exec_summary(result)}
  <h2 id="plots">图表</h2>
  {"".join(plot_parts)}
  {_quality_section(result)}
  {_target_section(result)}
  {_univ_section(result)}
  {_missing_section(result)}
  {_cat_section(result)}
  {_rel_section(result)}
  {_shift_section(result)}
  {_out_section(result)}
  {_leak_section(result)}
  {_eng_section(result)}
  {_method_section(result)}
  <footer>生成时间 { _esc(result.meta.get("timestamp")) }。HTML 为单文件自包含。</footer>
</div>
"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta name="referrer" content="no-referrer"/>
<title>S6E8 EDA 报告</title>
<style>{CSS}</style>
<script type="text/javascript">{plotly_js}</script>
</head>
<body>
{body}
</body>
</html>
"""


def write_html_report(result: EDAResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render_html(result)
    path.write_text(text, encoding="utf-8")
    return path
