# -*- coding: utf-8 -*-
"""第三步：组合风险聚合器。

输入约定：
- returns: DataFrame，行=交易日，列=股票代码；
- weights: dict / Series / array，必须与 returns 列对齐且权重和为 1；
- 建议直接使用第一步输出的 return_daily.csv 和持仓权重。

函数：
- calc_covariance_matrix(returns, window=None)  协方差矩阵（年化）
- calc_portfolio_volatility(weights, cov_matrix) 组合波动率
- calc_risk_contribution(weights, cov_matrix)    MCTR / RC / 占比
- calc_portfolio_var(...)                        组合 VaR（历史/参数法）
- calc_single_stock_concentration(...)           单票集中度
- calc_industry_concentration(...)               行业集中度
- calc_portfolio_summary(...)                    组合风险汇总
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252

_Z_TABLE = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.99: 2.3263,
}

__all__ = [
    "TRADING_DAYS",
    "calc_covariance_matrix",
    "calc_portfolio_volatility",
    "calc_risk_contribution",
    "calc_portfolio_var",
    "calc_single_stock_concentration",
    "calc_industry_concentration",
    "calc_portfolio_summary",
]


def _as_weights(weights, columns):
    """把权重统一成与 returns 列对齐的 Series，并校验和为 1。"""
    if isinstance(weights, pd.Series):
        w = weights.astype(float).reindex(columns)
    elif isinstance(weights, dict):
        w = pd.Series(weights, dtype=float).reindex(columns)
    else:
        w = pd.Series(np.asarray(weights, dtype=float), index=columns)
    if w.isna().any():
        raise ValueError("权重与收益列无法对齐，存在缺失权重")
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError(f"权重之和应为 1，当前为 {w.sum():.6f}")
    return w


def _z_score(confidence):
    """正态分布分位数，优先内置表，缺失时尝试 scipy。"""
    if confidence in _Z_TABLE:
        return _Z_TABLE[confidence]
    try:
        from scipy.stats import norm

        return float(-norm.ppf(1 - confidence))
    except ImportError as exc:
        raise ValueError(
            f"confidence={confidence} 不在内置表内，且环境没有 scipy"
        ) from exc


def calc_covariance_matrix(returns, window=None, annualize=TRADING_DAYS):
    """计算收益协方差矩阵，默认按 252 个交易日年化。"""
    returns = returns.dropna(how="any")
    if window is not None:
        returns = returns.tail(window)
    if len(returns) < 2:
        raise ValueError("有效收益数据不足，无法计算协方差矩阵")
    # 旧版 pandas 的 cov() 不支持 ddof 参数，默认即为样本协方差(ddof=1)
    return returns.cov() * annualize


def calc_portfolio_volatility(weights, cov_matrix):
    """组合波动率 sigma_p = sqrt(w^T @ Sigma @ w)。"""
    w = _as_weights(weights, cov_matrix.columns)
    return float(np.sqrt(w.values @ cov_matrix.values @ w.values))


def calc_risk_contribution(weights, cov_matrix):
    """MCTR / RC / RC 占比分解。

    MCTR_i = (Sigma @ w)_i / sigma_p
    RC_i = w_i * MCTR_i
    """
    w = _as_weights(weights, cov_matrix.columns)
    sigma_p = calc_portfolio_volatility(w, cov_matrix)
    if sigma_p <= 0:
        raise ValueError("组合波动率为 0，无法做风险贡献分解")

    mctr = cov_matrix.values @ w.values / sigma_p
    rc = w.values * mctr
    df = pd.DataFrame(
        {
            "weight": w.values,
            "mctr": mctr,
            "risk_contribution": rc,
            "rc_share": rc / sigma_p,
        },
        index=w.index,
    )
    return df.sort_values("rc_share", ascending=False)


def calc_portfolio_var(
    returns,
    weights,
    confidence=0.95,
    method="historical",
    horizon=1,
    cov_matrix=None,
):
    """组合 VaR，返回正值。

    - historical：组合历史收益序列的 1-confidence 分位数；
    - parametric：sigma_p * Z_alpha * sqrt(horizon)。
    """
    w = _as_weights(weights, returns.columns)
    if method == "historical":
        portfolio_returns = returns.dropna(how="any").dot(w)
        if portfolio_returns.empty:
            return np.nan
        return float(-portfolio_returns.quantile(1 - confidence))
    if method == "parametric":
        if cov_matrix is None:
            raise ValueError("参数法需要传入 cov_matrix")
        sigma_p = calc_portfolio_volatility(w, cov_matrix)
        return float(sigma_p * _z_score(confidence) * np.sqrt(horizon))
    raise ValueError("method 只支持 historical 或 parametric")


def calc_single_stock_concentration(weights, top_n=1):
    """单票集中度：最大（或前 top_n）持仓占比。"""
    w = pd.Series(weights, dtype=float).sort_values(ascending=False)
    return float(w.head(top_n).sum())


def calc_industry_concentration(weights, industry_map, top_n=3):
    """行业集中度：前 top_n 大行业持仓占比之和。"""
    w = pd.Series(weights, dtype=float)
    by_industry = w.groupby(industry_map).sum().sort_values(ascending=False)
    return float(by_industry.head(top_n).sum())


def calc_portfolio_summary(
    returns,
    weights,
    confidence=0.95,
    window=None,
    horizon=1,
    industry_map=None,
):
    """组合风险汇总：波动率、VaR、风险贡献、集中度。"""
    returns = returns.dropna(how="any")
    if window is not None:
        returns = returns.tail(window)
    if len(returns) < 2:
        raise ValueError("有效收益数据不足，无法计算组合风险")

    w = _as_weights(weights, returns.columns)
    cov = calc_covariance_matrix(returns)
    sigma_p = calc_portfolio_volatility(w, cov)
    rc = calc_risk_contribution(w, cov)

    summary = {
        "volatility": sigma_p,
        "var_historical": calc_portfolio_var(
            returns, w, confidence=confidence, method="historical"
        ),
        "var_parametric": calc_portfolio_var(
            returns,
            w,
            confidence=confidence,
            method="parametric",
            horizon=horizon,
            cov_matrix=cov,
        ),
        "risk_contribution": rc,
        "covariance_matrix": cov,
        "single_stock_concentration": calc_single_stock_concentration(w),
    }
    if industry_map is not None:
        summary["industry_concentration"] = calc_industry_concentration(
            w, industry_map
        )
    return summary
