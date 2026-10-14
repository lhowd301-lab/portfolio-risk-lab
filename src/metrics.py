"""风险度量模块：波动率、VaR/CVaR、回撤、Beta、相关性与集中度。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def to_returns(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().dropna()


def portfolio_returns(stock_returns: pd.DataFrame, weights: dict) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    aligned = stock_returns[w.index].dropna()
    return aligned.mul(w, axis=1).sum(axis=1)


def historical_vol(returns: pd.Series, window: int = 252, periods_per_year: int = 252) -> float:
    sample = returns.tail(window)
    return float(sample.std(ddof=1) * np.sqrt(periods_per_year))


def ewma_vol(returns: pd.Series, lam: float = 0.94, periods_per_year: int = 252) -> float:
    r2 = returns ** 2
    ewma = r2.ewm(alpha=1 - lam).mean().iloc[-1]
    return float(np.sqrt(ewma * periods_per_year))


def var_historical(
    returns: pd.Series,
    level: float = 0.95,
    horizon: int = 1,
    window: int = 252,
) -> float:
    r = returns.tail(window)
    q = r.quantile(1 - level)
    return float(-q * np.sqrt(horizon))


def var_parametric(returns: pd.Series, level: float = 0.95, horizon: int = 1) -> float:
    sigma = returns.std(ddof=1)
    return float(stats.norm.ppf(level) * sigma * np.sqrt(horizon))


def cvar(
    returns: pd.Series,
    level: float = 0.95,
    horizon: int = 1,
    window: int = 252,
) -> float:
    r = returns.tail(window)
    q = r.quantile(1 - level)
    tail = r[r <= q]
    if tail.empty:
        return 0.0
    return float(-tail.mean() * np.sqrt(horizon))


def drawdown_series(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1


def max_drawdown(nav: pd.Series) -> float:
    return float(drawdown_series(nav).min())


def beta(ret: pd.Series, bench: pd.Series) -> float:
    joined = pd.concat([ret, bench], axis=1, join="inner").dropna()
    if joined.empty:
        return 0.0
    var_b = joined.iloc[:, 1].var(ddof=1)
    if var_b == 0:
        return 0.0
    return float(joined.iloc[:, 0].cov(joined.iloc[:, 1]) / var_b)


def correlation_matrix(stock_returns: pd.DataFrame) -> pd.DataFrame:
    return stock_returns.corr()


def concentration(weights: dict) -> dict:
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    return {
        "hhi": float((w ** 2).sum()),
        "top_weight": float(w.max()),
        "top_code": str(w.idxmax()),
    }


def compute_all(
    stock_returns: pd.DataFrame,
    portfolio_ret: pd.Series,
    bench_ret: pd.Series,
    nav: pd.Series,
    weights: dict,
    config: dict,
) -> dict:
    risk_cfg = config["风险度量"]
    level = risk_cfg["VaR置信度"]
    horizon = risk_cfg["持有期"]
    window = risk_cfg["历史窗口"]
    rf = risk_cfg["无风险利率"]

    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (252 / len(nav)) - 1
    vol = historical_vol(portfolio_ret, window)
    vol_ewma = ewma_vol(portfolio_ret, risk_cfg["EWMA衰减因子"])
    downside_ret = portfolio_ret[portfolio_ret < 0]
    downside = downside_ret.std(ddof=1) * np.sqrt(252) if not downside_ret.empty else 0.0
    mdd = max_drawdown(nav)

    return {
        "initial_capital": float(config["账户"]["初始资金"]),
        "final_value": float(nav.iloc[-1]),
        "period_return": float(nav.iloc[-1] / nav.iloc[0] - 1),
        "annualized_return": float(ann_ret),
        "annualized_vol": vol,
        "annualized_vol_ewma": vol_ewma,
        "var95_hist": var_historical(portfolio_ret, level, horizon, window),
        "var95_param": var_parametric(portfolio_ret, level, horizon),
        "cvar95": cvar(portfolio_ret, level, horizon, window),
        "max_drawdown": mdd,
        "worst_day_return": float(portfolio_ret.min()),
        "best_day_return": float(portfolio_ret.max()),
        "beta": beta(portfolio_ret, bench_ret),
        "bench_vol": historical_vol(bench_ret, window),
        "sharpe": float((ann_ret - rf) / vol) if vol > 0 else 0.0,
        "sortino": float((ann_ret - rf) / downside) if downside > 0 else 0.0,
        "calmar": float(ann_ret / abs(mdd)) if mdd != 0 else 0.0,
        "gross_exposure": float(sum(abs(w) for w in weights.values())),
        "concentration": concentration(weights),
    }


def per_stock_table(
    stock_returns: pd.DataFrame,
    bench_ret: pd.Series,
    stock_info: dict,
    config: dict,
) -> pd.DataFrame:
    risk_cfg = config["风险度量"]
    rows = []
    for code, ret in stock_returns.items():
        nav = (1 + ret).cumprod()
        rows.append(
            {
                "代码": code,
                "名称": stock_info[code]["name"],
                "区间收益率": float((1 + ret).prod() - 1),
                "年化波动率": historical_vol(ret, risk_cfg["历史窗口"]),
                "95%VaR(1日)": var_historical(
                    ret,
                    risk_cfg["VaR置信度"],
                    risk_cfg["持有期"],
                    risk_cfg["历史窗口"],
                ),
                "Beta": beta(ret, bench_ret),
                "最大回撤": max_drawdown(nav),
                "日胜率": float((ret > 0).mean()),
            }
        )
    return pd.DataFrame(rows)
