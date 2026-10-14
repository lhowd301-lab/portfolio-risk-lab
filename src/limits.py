"""限额检查模块：把当前风险指标和配置限额做比较。"""

from __future__ import annotations


def _status(value: float, limit: float) -> str:
    if limit is None or limit <= 0:
        return "N/A"
    utilization = value / limit
    if utilization > 1.0:
        return "FAIL"
    if utilization >= 0.8:
        return "WARN"
    return "PASS"


def _industry_exposure(config: dict):
    stocks = config["股票池"]
    if not stocks or "行业" not in stocks[0]:
        return None
    total = sum(item.get("weight", 0) for item in stocks)
    exposure: dict = {}
    for item in stocks:
        industry = item.get("行业", "未知")
        exposure[industry] = exposure.get(industry, 0) + item.get("weight", 0)
    return max(exposure.values()) / total if total else None


def check_limits(metrics: dict, weights: dict, config: dict) -> list:
    lim = config["限额"]
    checks = []
    top_weight = metrics["concentration"]["top_weight"]
    checks.append(
        {
            "项目": "单票最大仓位",
            "当前值": top_weight,
            "限额": lim.get("单票仓位上限"),
            "状态": _status(top_weight, lim.get("单票仓位上限")),
        }
    )

    var = metrics["var95_hist"]
    checks.append(
        {
            "项目": "组合95% VaR（1日）",
            "当前值": var,
            "限额": lim.get("组合VaR上限"),
            "状态": _status(var, lim.get("组合VaR上限")),
        }
    )

    worst_day = abs(metrics["worst_day_return"])
    checks.append(
        {
            "项目": "历史单日最大亏损",
            "当前值": worst_day,
            "限额": lim.get("单日亏损熔断"),
            "状态": _status(worst_day, lim.get("单日亏损熔断")),
        }
    )

    mdd = abs(metrics["max_drawdown"])
    checks.append(
        {
            "项目": "最大回撤",
            "当前值": mdd,
            "限额": lim.get("最大回撤熔断"),
            "状态": _status(mdd, lim.get("最大回撤熔断")),
        }
    )

    exposure = metrics["gross_exposure"]
    checks.append(
        {
            "项目": "总敞口（杠杆）",
            "当前值": exposure,
            "限额": lim.get("杠杆上限"),
            "状态": _status(exposure, lim.get("杠杆上限")),
        }
    )

    industry = _industry_exposure(config)
    if industry is not None:
        checks.append(
            {
                "项目": "最大行业集中度",
                "当前值": industry,
                "限额": lim.get("行业集中度上限"),
                "状态": _status(industry, lim.get("行业集中度上限")),
            }
        )
    return checks
