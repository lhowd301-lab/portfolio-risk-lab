# -*- coding: utf-8 -*-
"""单票风险分析 + 组合风险指标与基准对比。

本模块完全自包含，聚宽 notebook 可直接整段粘贴运行，本地也可作为脚本运行。

输出：
- 单票风险分析表：波动率 60/120/250 日、最大回撤、VaR 95%、Beta 60 日
- 组合风险指标表：组合 vs 沪深300（年化收益、波动率、回撤、夏普、VaR、Beta）
- 风险贡献分解表：MCTR / RC / RC 占比
- 回测基准对比表：基准组合（无风控） vs 风控组合（来自 回测指标对比.csv）
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252

STOCKS = {
    "600519.XSHG": "贵州茅台",
    "300750.XSHE": "宁德时代",
    "600036.XSHG": "招商银行",
    "300760.XSHE": "迈瑞医疗",
    "002415.XSHE": "海康威视",
}

_Z_TABLE = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.99: 2.3263,
}


def _z_score(confidence):
    if confidence in _Z_TABLE:
        return _Z_TABLE[confidence]
    try:
        from scipy.stats import norm

        return float(-norm.ppf(1 - confidence))
    except ImportError as exc:
        raise ValueError(
            f"confidence={confidence} 不在内置表内，且环境没有 scipy"
        ) from exc


def _as_weights(weights, columns):
    w = pd.Series(weights, dtype=float).reindex(columns)
    if w.isna().any():
        raise ValueError("权重与收益列无法对齐")
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError(f"权重之和应为 1，当前为 {w.sum():.6f}")
    return w


def _rolling_vol_last(returns, window):
    return float(returns.rolling(window).std(ddof=1).iloc[-1] * np.sqrt(TRADING_DAYS))


def _max_drawdown(prices):
    prices = prices.dropna()
    if prices.empty:
        return np.nan
    return float((1.0 - prices / prices.cummax()).max())


def _var_hist(returns, confidence=0.95):
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return float(-returns.quantile(1 - confidence))


def _rolling_beta(stock, market, window=60):
    stock = pd.Series(stock, dtype=float)
    market = pd.Series(market, dtype=float)
    stock.index = pd.to_datetime(stock.index)
    market.index = pd.to_datetime(market.index)
    data = pd.concat([stock, market], axis=1, keys=["stock", "market"]).dropna()
    if len(data) < window:
        return pd.Series(dtype=float)
    cov = data["stock"].rolling(window).cov(data["market"])
    var = data["market"].rolling(window).var(ddof=1)
    return cov / var


def _cov_matrix(returns, window):
    returns = returns.tail(window).dropna(how="any")
    if len(returns) < 2:
        raise ValueError("有效收益数据不足，无法计算协方差矩阵")
    return returns.cov() * TRADING_DAYS


def _portfolio_vol(weights, cov):
    w = _as_weights(weights, cov.columns)
    return float(np.sqrt(w.values @ cov.values @ w.values))


def _portfolio_returns(returns, weights, window=None):
    sample = returns.tail(window) if window is not None else returns
    w = _as_weights(weights, sample.columns)
    return sample.fillna(0.0).dot(w)


def single_stock_risk_table(price, returns, market_returns=None, confidence=0.95):
    """每只股票的单票风险分析表。"""
    if market_returns is not None:
        market_returns = pd.Series(market_returns, dtype=float)
        market_returns.index = pd.to_datetime(market_returns.index)
    rows = []
    for code in price.columns:
        r = returns[code].dropna()
        beta_last = None
        if market_returns is not None:
            beta_last = _rolling_beta(r, market_returns, window=60).iloc[-1]
        rows.append(
            {
                "股票": STOCKS.get(code, code),
                "代码": code,
                "波动率60日": _rolling_vol_last(r, 60),
                "波动率120日": _rolling_vol_last(r, 120),
                "波动率250日": _rolling_vol_last(r, 250),
                "最大回撤": _max_drawdown(price[code]),
                "VaR 95%": _var_hist(r, confidence=confidence),
                "Beta 60日": beta_last,
            }
        )
    df = pd.DataFrame(rows).set_index("股票")
    return df.round(4)


def portfolio_risk_metrics(
    price,
    returns,
    weights,
    market_returns=None,
    window=250,
    confidence=0.95,
    rf=0.02,
):
    """组合风险指标 + 与沪深300 的对比。"""
    w = _as_weights(weights, returns.columns)
    port_ret = _portfolio_returns(returns, w, window=window)
    nav = (1.0 + port_ret).cumprod()

    cov = _cov_matrix(returns, window)
    sigma_p = _portfolio_vol(w, cov)
    var_hist = float(-port_ret.quantile(1 - confidence))
    var_param = float(sigma_p * _z_score(confidence))

    def _metrics(series):
        series = series.dropna()
        if len(series) == 0:
            return dict.fromkeys(
                ["年化收益", "年化波动率", "最大回撤", "夏普比率", "95% VaR", "Beta"], np.nan
            )
        nav_s = (1.0 + series).cumprod()
        annual_return = float(nav_s.iloc[-1] ** (TRADING_DAYS / len(series)) - 1.0)
        annual_vol = float(series.std(ddof=1) * np.sqrt(TRADING_DAYS))
        mdd = float((1.0 - nav_s / nav_s.cummax()).max())
        sharpe = (annual_return - rf) / annual_vol if annual_vol > 0 else np.nan
        var = float(-series.quantile(1 - confidence))
        return {
            "年化收益": annual_return,
            "年化波动率": annual_vol,
            "最大回撤": mdd,
            "夏普比率": sharpe,
            "95% VaR": var,
            "Beta": np.nan,
        }

    rows = {"组合": _metrics(port_ret)}
    rows["组合"]["Beta"] = (
        float(_rolling_beta(port_ret, market_returns, window=60).iloc[-1])
        if market_returns is not None
        else np.nan
    )
    rows["组合"]["95% VaR（参数法）"] = var_param
    rows["组合"]["组合波动率（参数）"] = sigma_p

    if market_returns is not None:
        bench = _metrics(market_returns)
        bench["Beta"] = 1.0
        bench["95% VaR（参数法）"] = float(
            market_returns.tail(window).std(ddof=1)
            * np.sqrt(TRADING_DAYS)
            * _z_score(confidence)
        )
        rows["沪深300"] = bench

    metrics = pd.DataFrame(rows).T.round(4)

    mctr = cov.values @ w.values / sigma_p
    rc = w.values * mctr
    rc_table = pd.DataFrame(
        {
            "股票": [STOCKS.get(c, c) for c in w.index],
            "代码": w.index,
            "权重": w.values,
            "MCTR": mctr,
            "风险贡献": rc,
            "风险贡献占比": rc / sigma_p,
        }
    ).sort_values("风险贡献占比", ascending=False).reset_index(drop=True).round(4)

    return {
        "metrics": metrics,
        "risk_contribution": rc_table,
        "covariance_matrix": cov,
        "var_historical": var_hist,
        "var_parametric": var_param,
        "portfolio_volatility": sigma_p,
        "single_stock_concentration": float(w.max()),
    }


def load_backtest_comparison(path):
    """读取第六步输出的回测指标对比.csv。"""
    df = pd.read_csv(path, index_col=0)
    df.index.name = None
    return df.round(4)


def load_market_returns(path):
    """读取沪深300收益 CSV：支持 date/close 或 date/return 两种格式。"""
    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    if "return" in df.columns:
        return df["return"].dropna()
    close_col = "close" if "close" in df.columns else df.columns[0]
    return df[close_col].pct_change().dropna()


def build_risk_analysis_report(
    data_dir,
    output_dir,
    weights=None,
    window=250,
    confidence=0.95,
    rf=0.02,
    backtest_path=None,
    benchmark_path=None,
):
    """读取数据并输出全部风险分析表，保存 CSV。"""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    price = pd.read_csv(data_dir / "price_daily.csv", index_col=0)
    returns = pd.read_csv(data_dir / "return_daily.csv", index_col=0)
    price.index = pd.to_datetime(price.index)
    returns.index = pd.to_datetime(returns.index)

    if weights is None:
        weights = {code: 1.0 / len(returns.columns) for code in returns.columns}

    market_returns = None
    if benchmark_path is not None and Path(benchmark_path).exists():
        market_returns = load_market_returns(benchmark_path)
        print(f"基准收益已加载：{benchmark_path}")
    elif "get_price" in globals():
        try:
            idx = get_price(
                "000300.XSHG",
                start_date=str(returns.index.min().date()),
                end_date=str(returns.index.max().date()),
                frequency="daily",
                fields=["close"],
                fq="pre",
                panel=False,
            )
            market_returns = idx["close"].pct_change().dropna()
            market_returns.index = pd.to_datetime(market_returns.index)
            print("基准收益已从聚宽获取：000300.XSHG")
        except Exception as exc:
            print(f"聚宽基准获取失败，跳过对比：{exc}")
    else:
        print("未提供沪深300收益文件，Beta 与基准对比跳过")

    if market_returns is not None:
        market_returns = pd.Series(market_returns, dtype=float)
        market_returns.index = pd.to_datetime(market_returns.index)
        print(
            f"沪深300样本：{len(market_returns)} 行，"
            f"{market_returns.index.min().date()} ~ {market_returns.index.max().date()}"
        )

    single = single_stock_risk_table(price, returns, market_returns, confidence)
    portfolio = portfolio_risk_metrics(
        price,
        returns,
        weights,
        market_returns,
        window=window,
        confidence=confidence,
        rf=rf,
    )

    print("\n单票风险分析：")
    print(single.to_string())
    print("\n组合风险指标（组合 vs 沪深300）：")
    print(portfolio["metrics"].to_string())
    print("\n风险贡献分解：")
    print(portfolio["risk_contribution"].to_string(index=False))

    single.to_csv(output_dir / "单票风险分析.csv", encoding="utf-8-sig")
    portfolio["metrics"].to_csv(output_dir / "组合风险指标.csv", encoding="utf-8-sig")
    portfolio["risk_contribution"].to_csv(
        output_dir / "风险贡献分解.csv", index=False, encoding="utf-8-sig"
    )

    backtest = None
    if backtest_path is not None and Path(backtest_path).exists():
        backtest = load_backtest_comparison(backtest_path)
        backtest.to_csv(output_dir / "回测基准对比.csv", encoding="utf-8-sig")
        print("\n回测基准对比（来自 回测指标对比.csv）：")
        print(backtest.to_string())
    else:
        print("\n未找到回测指标对比.csv，跳过回测基准对比")

    print(f"\n文件已保存到：{output_dir}")
    return {
        "single_stock": single,
        "portfolio": portfolio,
        "backtest": backtest,
    }


def main():
    parser = argparse.ArgumentParser(description="单票风险 + 组合风险与基准对比")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--rf", type=float, default=0.02)
    parser.add_argument("--backtest", default=None)
    parser.add_argument("--benchmark", default=None)
    args = parser.parse_args()

    project = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else project / "data"
    output_dir = (
        Path(args.output_dir) if args.output_dir else project / "output" / "风险分析"
    )
    backtest_path = (
        Path(args.backtest)
        if args.backtest
        else project / "data" / "回测指标对比.csv"
    )
    benchmark_path = (
        Path(args.benchmark)
        if args.benchmark
        else project / "data" / "index_000300.csv"
    )

    build_risk_analysis_report(
        data_dir,
        output_dir,
        window=args.window,
        confidence=args.confidence,
        rf=args.rf,
        backtest_path=backtest_path,
        benchmark_path=benchmark_path,
    )


if __name__ == "__main__":
    main()
