"""工作流编排：把六个阶段串成一条可单独调用的流水线。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loader, limits as limits_mod, metrics as metrics_mod, report as report_mod, stress as stress_mod


def stage_fetch(config: dict, force_refresh: bool = False):
    return data_loader.load_market_data(config, force_refresh=force_refresh)


def stage_metrics(market, config: dict) -> dict:
    stock_returns = pd.DataFrame(
        {code: metrics_mod.to_returns(df) for code, df in market.stocks.items()}
    )
    weights = {code: float(item["weight"]) for code, item in market.stock_info.items()}
    portfolio_ret = metrics_mod.portfolio_returns(stock_returns, weights)
    nav = (1 + portfolio_ret).cumprod() * float(config["账户"]["初始资金"])
    bench_ret = metrics_mod.to_returns(market.benchmark)
    bench_nav = (1 + bench_ret).cumprod() * float(config["账户"]["初始资金"])
    all_metrics = metrics_mod.compute_all(
        stock_returns, portfolio_ret, bench_ret, nav, weights, config
    )
    per_stock = metrics_mod.per_stock_table(stock_returns, bench_ret, market.stock_info, config)
    correlation = metrics_mod.correlation_matrix(stock_returns)

    dirs = data_loader.ensure_dirs()
    portfolio_ret.to_frame("组合日收益").to_csv(dirs["processed"] / "组合日收益.csv", encoding="utf-8-sig")
    stock_returns.to_csv(dirs["processed"] / "个股日收益.csv", encoding="utf-8-sig")

    return {
        "market": market,
        "stock_returns": stock_returns,
        "weights": weights,
        "portfolio_ret": portfolio_ret,
        "bench_ret": bench_ret,
        "nav": nav,
        "bench_nav": bench_nav,
        "metrics": all_metrics,
        "per_stock": per_stock,
        "correlation": correlation,
    }


def stage_limits(result: dict, config: dict) -> list:
    checks = limits_mod.check_limits(result["metrics"], result["weights"], config)
    result["checks"] = checks
    return checks


def stage_stress(result: dict, config: dict) -> list:
    stress_results = stress_mod.run_stress(result["metrics"], config["限额"])
    result["stress"] = stress_results
    return stress_results


def stage_report(result: dict, config: dict) -> dict:
    meta = {
        "start": config["数据"]["回测起点"],
        "end": result["nav"].index[-1].date().isoformat(),
    }
    result["meta"] = meta
    return report_mod.generate_report(result, config)
