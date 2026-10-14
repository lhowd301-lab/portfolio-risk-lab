# -*- coding: utf-8 -*-
"""第五步：压力测试模块（极端情景亏损预估）。

输入：
- portfolio_summary: 第三步 calc_portfolio_summary 的返回字典；
- weights: dict / Series，持仓权重，和必须为 1；
- limits: 可选，第四步阈值配置字典。

输出：
- DataFrame：情景、持有期、组合累计亏损、是否触发熔断、对比阈值、说明。
"""

import numpy as np
import pandas as pd

__all__ = ["run_stress_test"]


def _limit_down_shocks(weights):
    """按板块返回跌停冲击：创业板/科创板 -20%，其余 -10%。"""
    shocks = {}
    for code in weights.index:
        if str(code).startswith(("300", "301", "688", "689")):
            shocks[code] = -0.20
        else:
            shocks[code] = -0.10
    return pd.Series(shocks)


def _portfolio_loss(weights, shocks_list):
    """按权重计算多日冲击下的组合累计亏损（正值）。"""
    cumulative = 1.0
    for shock in shocks_list:
        if isinstance(shock, pd.Series):
            daily_return = float((weights * shock).sum())
        else:
            daily_return = float(shock)
        cumulative *= 1.0 + daily_return
    return -(cumulative - 1.0)


def _make_row(name, days, loss, threshold, threshold_name):
    triggered = loss > threshold
    return {
        "情景": name,
        "持有期(天)": days,
        "组合累计亏损": round(float(loss), 6),
        "是否触发熔断": bool(triggered),
        "对比阈值": round(float(threshold), 6),
        "说明": f"超过{threshold_name}" if triggered else f"未超过{threshold_name}",
    }


def run_stress_test(
    portfolio_summary,
    weights,
    limits=None,
    daily_loss_limit=0.02,
):
    """执行压力测试，返回各极端情景的亏损预估表。"""
    limits = limits or {}
    w = pd.Series(weights, dtype=float)
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError(f"权重之和应为 1，当前为 {w.sum():.6f}")

    drawdown_limit = limits.get("max_drawdown", 0.15)
    var_limit = limits.get("portfolio_var_95", 0.03)
    single_day_limit = limits.get("daily_loss_limit", daily_loss_limit)

    rows = []

    # 1. 单日跌停冲击：主板约 -10%，创业板/科创板约 -20%
    limit_down = _limit_down_shocks(w)
    loss = _portfolio_loss(w, [limit_down])
    rows.append(
        _make_row(
            "单日跌停冲击（主板-10%/创业板-20%）",
            1,
            loss,
            single_day_limit,
            "单日亏损熔断线",
        )
    )

    # 2. 连续三日下跌
    loss = _portfolio_loss(w, [-0.04, -0.03, -0.025])
    rows.append(
        _make_row(
            "连续三日下跌（-4%/-3%/-2.5%）",
            3,
            loss,
            drawdown_limit,
            "回撤熔断线",
        )
    )

    # 3. 股灾式五日崩盘
    loss = _portfolio_loss(
        w, [-0.08, -0.06, -0.05, -0.04, -0.03]
    )
    rows.append(
        _make_row(
            "股灾式五日崩盘",
            5,
            loss,
            drawdown_limit,
            "回撤熔断线",
        )
    )

    # 4. 流动性冻结：跌停后无法卖出，次日续跌 5%
    loss = _portfolio_loss(w, [limit_down, -0.05])
    rows.append(
        _make_row(
            "流动性冻结（跌停后次日续跌5%）",
            2,
            loss,
            drawdown_limit,
            "回撤熔断线",
        )
    )

    # 5. 波动率翻倍：VaR 直接翻倍
    base_var = portfolio_summary.get("var_historical")
    if base_var is None:
        base_var = portfolio_summary.get("var_parametric")
    if base_var is None:
        raise ValueError("portfolio_summary 中缺少 var_historical/var_parametric")
    loss = float(base_var) * 2.0
    rows.append(
        _make_row(
            "波动率翻倍（VaR x2）",
            1,
            loss,
            var_limit,
            "组合VaR上限",
        )
    )

    return pd.DataFrame(rows)
