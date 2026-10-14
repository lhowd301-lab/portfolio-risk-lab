# -*- coding: utf-8 -*-
"""第六步：历史回测验证（带/不带风控对比）。

模拟规则：
- 基准组合：按权重每日再平衡，满仓持有；
- 风控组合：单日亏损超过 daily_loss_limit，或回撤超过 max_drawdown_limit 时，
  把仓位降到 risk_off_weight，冷却 cooling_days 天后恢复满仓。
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252

__all__ = [
    "TRADING_DAYS",
    "backtest_portfolio",
    "calc_performance",
    "run_comparison",
]


def _as_weights(weights, columns):
    w = pd.Series(weights, dtype=float).reindex(columns)
    if w.isna().any():
        raise ValueError("权重与收益列无法对齐")
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError(f"权重之和应为 1，当前为 {w.sum():.6f}")
    return w


def backtest_portfolio(
    returns,
    weights,
    risk_control=True,
    risk_off_weight=0.50,
    cooling_days=5,
    daily_loss_limit=0.02,
    max_drawdown_limit=0.15,
):
    """回测一条组合净值曲线。

    risk_control=False 时是满仓基准组合；True 时模拟触发风控后降仓冷却。
    """
    w = _as_weights(weights, returns.columns)
    raw_return = returns.fillna(0.0).dot(w)

    nav = 1.0
    peak = 1.0
    exposure = 1.0
    cooling = 0
    rows = []

    for date, raw in raw_return.items():
        # 冷却期结束先恢复仓位，再计算当日收益
        if risk_control and cooling > 0:
            cooling -= 1
            if cooling == 0:
                exposure = 1.0

        daily_return = float(raw) * exposure
        nav *= 1.0 + daily_return
        peak = max(peak, nav)
        drawdown = nav / peak - 1.0

        # 当日触发风控，从次日开始执行降仓
        if (
            risk_control
            and cooling == 0
            and (
                daily_return <= -daily_loss_limit
                or drawdown <= -max_drawdown_limit
            )
        ):
            cooling = cooling_days
            exposure = risk_off_weight

        rows.append(
            {
                "raw_return": float(raw),
                "strategy_return": daily_return,
                "nav": nav,
                "drawdown": drawdown,
                "exposure": exposure,
                "risk_off": exposure < 1.0,
            }
        )

    return pd.DataFrame(rows, index=raw_return.index)


def calc_performance(strategy_returns, nav, rf=0.02):
    """计算一组绩效指标，输入为策略收益序列和净值序列。"""
    strategy_returns = pd.Series(strategy_returns, dtype=float)
    nav = pd.Series(nav, dtype=float)
    n = len(strategy_returns)
    if n == 0:
        raise ValueError("收益序列为空")

    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    annual_return = float((nav.iloc[-1] / nav.iloc[0]) ** (TRADING_DAYS / n) - 1.0)
    annual_vol = float(strategy_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    max_drawdown = float((1.0 - nav / nav.cummax()).max())
    sharpe = (
        float((annual_return - rf) / annual_vol)
        if annual_vol > 0
        else np.nan
    )
    var_95 = float(-strategy_returns.quantile(0.05))
    return {
        "累计收益": total_return,
        "年化收益": annual_return,
        "年化波动率": annual_vol,
        "最大回撤": max_drawdown,
        "夏普比率": sharpe,
        "95% VaR": var_95,
    }


def run_comparison(
    returns,
    weights,
    risk_off_weight=0.50,
    cooling_days=5,
    daily_loss_limit=0.02,
    max_drawdown_limit=0.15,
    rf=0.02,
):
    """基准组合 vs 风控组合，返回两条净值曲线和指标对比表。"""
    baseline = backtest_portfolio(
        returns, weights, risk_control=False
    )
    controlled = backtest_portfolio(
        returns,
        weights,
        risk_control=True,
        risk_off_weight=risk_off_weight,
        cooling_days=cooling_days,
        daily_loss_limit=daily_loss_limit,
        max_drawdown_limit=max_drawdown_limit,
    )

    metrics = pd.DataFrame(
        {
            "基准组合（无风控）": calc_performance(
                baseline["strategy_return"], baseline["nav"], rf=rf
            ),
            "风控组合": calc_performance(
                controlled["strategy_return"], controlled["nav"], rf=rf
            ),
        }
    ).T
    metrics["风控触发天数"] = [0, int(controlled["risk_off"].sum())]

    return {
        "baseline": baseline,
        "controlled": controlled,
        "metrics": metrics,
    }
