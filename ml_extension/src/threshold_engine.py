# -*- coding: utf-8 -*-
"""第四步：阈值引擎（可配置风控规则）。

规则来源：
- 配置文件：config/risk_limits.json（risk_limits 字典）；
- 未找到配置文件时使用内置 DEFAULT_LIMITS，方便聚宽 notebook 直接运行。

规则含义：所有风控规则均为上限约束，value <= limit 为 PASS，超过为 FAIL。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_LIMITS = {
    "portfolio_volatility": 0.20,   # 组合年化波动率上限 20%
    "single_stock_weight": 0.10,    # 单票最大仓位 10%
    "sector_weight": 0.30,          # 前 3 大行业合计仓位上限 30%
    "portfolio_var_95": 0.03,       # 日频 95% VaR 上限 3%
    "max_drawdown": 0.15,           # 历史最大回撤警戒线 15%
}

__all__ = [
    "DEFAULT_LIMITS",
    "load_limits",
    "check_limits",
    "check_portfolio",
    "get_overall_status",
]


def load_limits(path=None):
    """读取 JSON 规则文件；path 为空且文件不存在时返回内置默认值。"""
    if path is None:
        try:
            default_path = (
                Path(__file__).resolve().parent.parent
                / "config" / "risk_limits.json"
            )
        except NameError:  # 聚宽 notebook 中没有 __file__
            default_path = None
        path = default_path

    if path is not None and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        limits = cfg.get("risk_limits", cfg)
        if not isinstance(limits, dict):
            raise ValueError("规则配置中 risk_limits 必须是字典")
        return {k: float(v) for k, v in limits.items()}
    return dict(DEFAULT_LIMITS)


def check_limits(metrics, limits=None):
    """逐条检查指标是否超过阈值，返回规则检查表。"""
    limits = limits or load_limits()
    rows = []
    for rule, limit in limits.items():
        value = metrics.get(rule)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            rows.append(
                {
                    "rule": rule,
                    "value": None,
                    "limit": limit,
                    "status": "NO_DATA",
                }
            )
            continue
        value = float(value)
        status = "PASS" if value <= float(limit) else "FAIL"
        rows.append(
            {
                "rule": rule,
                "value": value,
                "limit": float(limit),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def _top_n_industry_share(weights, industry_map, top_n=3):
    """前 top_n 大行业持仓占比之和。"""
    w = pd.Series(weights, dtype=float)
    by_industry = w.groupby(industry_map).sum().sort_values(ascending=False)
    return float(by_industry.head(top_n).sum())


def check_portfolio(
    portfolio_summary,
    weights,
    limits=None,
    industry_map=None,
    max_drawdown=None,
):
    """把第三步的组合风险汇总转成指标，再执行阈值检查。"""
    limits = limits or load_limits()
    w = pd.Series(weights, dtype=float)

    metrics = {
        "portfolio_volatility": portfolio_summary.get("volatility"),
        "portfolio_var_95": portfolio_summary.get("var_historical"),
        "single_stock_weight": float(w.max()),
        "sector_weight": portfolio_summary.get("industry_concentration"),
        "max_drawdown": max_drawdown,
    }
    if metrics["sector_weight"] is None and industry_map is not None:
        metrics["sector_weight"] = _top_n_industry_share(w, industry_map)
    return check_limits(metrics, limits)


def get_overall_status(rule_table):
    """汇总规则表状态：任一 FAIL 即 FAIL，有 NO_DATA 则返回 NO_DATA。"""
    if rule_table.empty:
        return "NO_DATA"
    if "FAIL" in set(rule_table["status"]):
        return "FAIL"
    if "NO_DATA" in set(rule_table["status"]):
        return "NO_DATA"
    return "PASS"
