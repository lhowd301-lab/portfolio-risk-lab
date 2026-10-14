"""压力测试模块：用情景冲击计算组合损失并和熔断线比较。"""

from __future__ import annotations

import numpy as np


def run_stress(metrics: dict, limits: dict) -> list:
    single_day = limits.get("单日亏损熔断", 0.02)
    drawdown_limit = limits.get("最大回撤熔断", 0.15)
    var_limit = limits.get("组合VaR上限", 0.03)
    results = []

    shock_scenarios = [
        ("单日跌停冲击（主板约-9.5%）", [-0.095]),
        ("连续三日下跌（-4%/-3%/-2.5%）", [-0.04, -0.03, -0.025]),
        ("股灾式五日崩盘", [-0.08, -0.06, -0.05, -0.04, -0.03]),
        ("流动性冻结（跌停后无法卖出，次日续跌5%）", [-0.095, -0.05]),
    ]
    for name, shocks in shock_scenarios:
        cum_return = np.prod([1 + s for s in shocks]) - 1
        loss = -cum_return
        multi_day = len(shocks) > 1
        triggered = loss > drawdown_limit if multi_day else loss > single_day
        limit_name = "回撤熔断线" if multi_day else "单日亏损熔断线"
        results.append(
            {
                "情景": name,
                "持有期（天）": len(shocks),
                "组合累计亏损": float(loss),
                "是否触发熔断": bool(triggered),
                "说明": f"累计亏损超过{limit_name}" if triggered else "未超过熔断线",
            }
        )

    var = metrics["var95_hist"]
    new_var = var * 2.0
    results.append(
        {
            "情景": "波动率翻倍",
            "持有期（天）": 1,
            "组合累计亏损": float(new_var),
            "是否触发熔断": bool(new_var > var_limit),
            "说明": f"VaR 由 {var:.2%} 升至 {new_var:.2%}",
        }
    )
    return results
