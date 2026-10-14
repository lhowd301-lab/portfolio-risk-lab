# -*- coding: utf-8 -*-
"""第二步：单票风险指标计算模块。

输入约定：
- returns / prices / market_returns 均为 pandas Series，索引为日期；
- 建议直接使用第一步输出的 price_daily.csv / return_daily.csv；
- 停牌、涨跌停日收益在第一步已置为 NaN，滚动计算时自动跳过。

函数：
- calc_volatility(returns, window=60)      滚动年化波动率
- calc_max_drawdown(prices)                历史最大回撤（正值）
- calc_var(returns, confidence=0.95)       历史模拟法 VaR（正值）
- calc_beta(stock_returns, market_returns, window=60)  滚动 Beta
- calc_risk_summary(...)                   60/120/250 日窗口汇总
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_WINDOWS = (60, 120, 250)

__all__ = [
    "TRADING_DAYS",
    "DEFAULT_WINDOWS",
    "calc_volatility",
    "calc_max_drawdown",
    "calc_var",
    "calc_beta",
    "calc_risk_summary",
]


def _ensure_series(x, name):
    """把单列输入统一成 Series，避免后续函数处理逻辑分支。"""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"{name} 必须是单列 Series/DataFrame")
        return x.iloc[:, 0]
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x, name=name)


def calc_volatility(returns, window=60):
    """滚动年化波动率 = 滚动标准差 x sqrt(252)。"""
    returns = _ensure_series(returns, "returns").astype(float)
    return (
        returns.rolling(window=window, min_periods=window)
        .std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )


def calc_max_drawdown(prices):
    """历史最大回撤（从峰值到谷底的最大跌幅，返回正值）。"""
    prices = _ensure_series(prices, "prices").astype(float).dropna()
    if prices.empty:
        return np.nan
    drawdown = prices / prices.cummax() - 1.0
    return float(-drawdown.min())


def calc_var(returns, confidence=0.95, method="historical", window=None):
    """历史模拟法 VaR。

    window=None 时用全部历史；传入 window 时输出滚动 VaR。
    返回值为正数，表示最大可能亏损幅度。
    """
    if method != "historical":
        raise ValueError("本模块目前只支持 method='historical'")
    returns = _ensure_series(returns, "returns").astype(float).dropna()
    if window is None:
        return float(-returns.quantile(1 - confidence))
    var = -returns.rolling(
        window=window, min_periods=window
    ).quantile(1 - confidence)
    var.name = "var"
    return var


def calc_beta(stock_returns, market_returns, window=60):
    """滚动 Beta = Cov(个股, 市场) / Var(市场)。"""
    stock = _ensure_series(stock_returns, "stock_returns").astype(float)
    market = _ensure_series(market_returns, "market_returns").astype(float)
    data = pd.concat([stock, market], axis=1, keys=["stock", "market"]).dropna(
        how="any"
    )
    cov = data["stock"].rolling(
        window=window, min_periods=window
    ).cov(data["market"])
    var = data["market"].rolling(
        window=window, min_periods=window
    ).var(ddof=1)
    beta = cov / var
    beta.name = "beta"
    return beta


def calc_risk_summary(
    prices,
    returns,
    market_returns=None,
    confidence=0.95,
    windows=DEFAULT_WINDOWS,
):
    """按 60/120/250 日窗口输出单票风险汇总表。"""
    prices = _ensure_series(prices, "prices").astype(float)
    returns = _ensure_series(returns, "returns").astype(float)

    summary = pd.DataFrame(index=returns.index)
    summary["close"] = prices
    for window in windows:
        summary[f"volatility_{window}"] = calc_volatility(returns, window=window)
        summary[f"var_{confidence}_{window}"] = calc_var(
            returns, confidence=confidence, window=window
        )
        if market_returns is not None:
            summary[f"beta_{window}"] = calc_beta(
                returns, market_returns, window=window
            )
    return summary
