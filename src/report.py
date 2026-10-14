"""风控报告：Markdown、Excel 和图表输出。"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

METRIC_LABELS = {
    "initial_capital": "初始资金（元）",
    "final_value": "期末净值（元）",
    "period_return": "区间收益率",
    "annualized_return": "年化收益率",
    "annualized_vol": "年化波动率（历史）",
    "annualized_vol_ewma": "年化波动率（EWMA）",
    "var95_hist": "95% VaR（历史模拟，1日）",
    "var95_param": "95% VaR（参数法，1日）",
    "cvar95": "95% CVaR（1日）",
    "max_drawdown": "最大回撤",
    "worst_day_return": "历史最差单日收益",
    "best_day_return": "历史最好单日收益",
    "beta": "组合 Beta（对沪深300）",
    "bench_vol": "基准年化波动率",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "calmar": "Calmar",
    "gross_exposure": "总敞口",
}


def _fmt(key: str, value) -> str:
    if key in ("initial_capital", "final_value"):
        return f"{value:,.0f}"
    if key in ("sharpe", "sortino", "calmar", "beta"):
        return f"{value:.2f}"
    return f"{value:.2%}"


def _metric_rows(metrics: dict) -> list:
    rows = []
    for key, label in METRIC_LABELS.items():
        rows.append([label, _fmt(key, metrics[key])])
    concentration = metrics["concentration"]
    rows.append(["集中度 HHI", f"{concentration['hhi']:.3f}"])
    rows.append(
        [
            "最大单票权重",
            f"{concentration['top_weight']:.2%}（{concentration['top_code']}）",
        ]
    )
    return rows


def _to_native(value):
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    return value


def _conclusion(checks: list) -> str:
    statuses = [c["状态"] for c in checks]
    if "FAIL" in statuses:
        return "触发限额：暂停新增仓位，优先降低超限敞口，并人工复核。"
    if "WARN" in statuses:
        return "接近限额：暂不触发熔断，但需要控制仓位并加强监控。"
    return "运行正常：当前风险指标均在限额以内。"


def _markdown_table(headers: list, rows: list, widths: list = None) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _stock_row_formatter(columns: list, row) -> list:
    formatted = []
    for col, value in zip(columns, row):
        if col == "Beta":
            formatted.append(f"{value:.2f}")
        elif isinstance(value, float):
            formatted.append(f"{value:.2%}")
        else:
            formatted.append(value)
    return formatted


def generate_markdown(result: dict, config: dict, report_dir: Path) -> Path:
    market = result["market"]
    date_str = dt.date.today().isoformat()
    path = report_dir / f"风控报告_{date_str}.md"
    meta = result["meta"]

    lines = [
        f"# A股风控报告 {date_str}",
        "",
        f"- 数据来源：{market.source}；{'虚拟数据，仅供流程演示' if market.is_demo else '真实行情数据'}",
        f"- 数据区间：{meta['start']} ~ {meta['end']}",
        f"- 股票池：{len(market.stocks)} 只；基准：{market.benchmark_name}",
        "",
        "## 组合概览",
        "",
        _markdown_table(["指标", "数值"], _metric_rows(result["metrics"])),
        "",
        "## 限额检查",
        "",
        _markdown_table(
            ["项目", "当前值", "限额", "状态"],
            [
                [c["项目"], f"{c['当前值']:.2%}", f"{c['限额']:.2%}", c["状态"]]
                for c in result["checks"]
            ],
        ),
        "",
        "## 压力测试",
        "",
        _markdown_table(
            ["情景", "持有期（天）", "组合累计亏损", "是否触发熔断", "说明"],
            [
                [
                    s["情景"],
                    s["持有期（天）"],
                    f"{s['组合累计亏损']:.2%}",
                    "是" if s["是否触发熔断"] else "否",
                    s["说明"],
                ]
                for s in result["stress"]
            ],
        ),
        "",
        "## 个股风险",
        "",
        _markdown_table(
            list(result["per_stock"].columns),
            [_stock_row_formatter(result["per_stock"].columns, row) for row in result["per_stock"].values],
        ),
        "",
        "## 结论",
        "",
        _conclusion(result["checks"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def plot_overview(result: dict, figures_dir: Path) -> Path:
    nav = result["nav"]
    bench_nav = result["bench_nav"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(nav.index, nav / nav.iloc[0], label="组合", linewidth=1.6)
    axes[0].plot(bench_nav.index, bench_nav / bench_nav.iloc[0], label=result["market"].benchmark_name, linewidth=1.2)
    axes[0].set_title("净值曲线（归一化）")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    drawdown = nav / nav.cummax() - 1
    axes[1].fill_between(drawdown.index, drawdown * 100, 0, color="red", alpha=0.4)
    axes[1].set_title("组合回撤（%）")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path = figures_dir / "净值与回撤.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_heatmap(result: dict, figures_dir: Path) -> Path:
    corr = result["correlation"]
    labels = [result["market"].stock_info[code]["name"] for code in corr.columns]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax)
    ax.set_title("个股收益相关性")
    fig.tight_layout()
    path = figures_dir / "相关性热力图.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def export_excel(result: dict, data_dir: Path) -> Path:
    date_str = dt.date.today().isoformat()
    path = data_dir / f"风控明细_{date_str}.xlsx"
    metrics_df = pd.DataFrame(
        [(label, _fmt(key, result["metrics"][key])) for key, label in METRIC_LABELS.items()],
        columns=["指标", "数值"],
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="组合指标", index=False)
        result["per_stock"].to_excel(writer, sheet_name="个股风险", index=False)
        pd.DataFrame(result["checks"]).to_excel(writer, sheet_name="限额检查", index=False)
        pd.DataFrame(result["stress"]).to_excel(writer, sheet_name="压力测试", index=False)
        result["correlation"].to_excel(writer, sheet_name="相关性")
    return path


def export_json(result: dict, data_dir: Path) -> Path:
    date_str = dt.date.today().isoformat()
    path = data_dir / f"风险指标_{date_str}.json"
    payload = _to_native(result["metrics"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def generate_report(result: dict, config: dict) -> dict:
    dirs = {
        "reports": Path(config["报告"]["输出目录"]),
        "figures": Path(config["报告"]["图表目录"]),
        "data": Path(config["报告"]["数据目录"]),
    }
    root = Path(__file__).resolve().parents[1]
    for key in dirs:
        dirs[key] = root / dirs[key]
        dirs[key].mkdir(parents=True, exist_ok=True)

    md_path = generate_markdown(result, config, dirs["reports"])
    overview_path = plot_overview(result, dirs["figures"])
    heatmap_path = plot_heatmap(result, dirs["figures"])
    excel_path = export_excel(result, dirs["data"])
    json_path = export_json(result, dirs["data"])
    return {
        "markdown": md_path,
        "overview_png": overview_path,
        "heatmap_png": heatmap_path,
        "excel": excel_path,
        "json": json_path,
    }
