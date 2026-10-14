# -*- coding: utf-8 -*-
"""机器学习特征工程模块。

输入（宽表，行=日期，列=股票代码）：
- price_daily: 前复权收盘价
- return_daily: 日收益率
- volume_daily: 可选，成交量
- market_returns: 可选，基准指数日收益（Series）

输出：
- build_features() -> MultiIndex(日期, 股票代码) 特征表
- make_volatility_label() -> 未来 horizon 日已实现波动率
- make_future_return() -> 未来 horizon 日收益（VaR 回归标签）
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_WINDOWS = (5, 21, 60)

__all__ = [
    "TRADING_DAYS",
    "DEFAULT_WINDOWS",
    "build_features",
    "make_volatility_label",
    "make_future_return",
]


def _build_stock_features(
    price,
    returns,
    volume=None,
    market_returns=None,
    windows=DEFAULT_WINDOWS,
):
    df = pd.DataFrame({"close": price.astype(float), "ret": returns.astype(float)})
    if volume is not None:
        df["volume"] = volume.astype(float)
    if market_returns is not None:
        df["market_ret"] = market_returns.astype(float)

    for w in windows:
        ret_std = df["ret"].rolling(w).std(ddof=1)
        df[f"ret_mean_{w}"] = df["ret"].rolling(w).mean()
        df[f"ret_std_{w}"] = ret_std
        df[f"vol_{w}"] = ret_std * np.sqrt(TRADING_DAYS)
        rolling_max = df["close"].rolling(w).max()
        rolling_min = df["close"].rolling(w).min()
        df[f"dist_high_{w}"] = df["close"] / rolling_max - 1.0
        df[f"dist_low_{w}"] = df["close"] / rolling_min - 1.0
        df[f"range_{w}"] = (rolling_max - rolling_min) / df["close"]

    if volume is not None:
        volume_chg = df["volume"].pct_change()
        for w in windows:
            df[f"volume_chg_{w}"] = volume_chg.rolling(w).mean()

    if market_returns is not None:
        df["excess_ret"] = df["ret"] - df["market_ret"]
        df["mkt_vol_21"] = (
            df["market_ret"].rolling(21).std(ddof=1) * np.sqrt(TRADING_DAYS)
        )
    return df


def build_features(
    price_daily,
    return_daily,
    volume_daily=None,
    market_returns=None,
    windows=DEFAULT_WINDOWS,
):
    """宽表转成 MultiIndex(日期, 股票代码) 特征表。"""
    frames = []
    for code in price_daily.columns:
        volume = volume_daily[code] if volume_daily is not None else None
        feats = _build_stock_features(
            price_daily[code],
            return_daily[code],
            volume=volume,
            market_returns=market_returns,
            windows=windows,
        )
        feats.index.name = "date"
        feats = feats.reset_index()
        feats["code"] = code
        frames.append(feats)
    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    return result.set_index(["date", "code"]).sort_index()


def make_volatility_label(returns, horizon=21):
    """未来 horizon 个交易日的已实现波动率（年化）。"""
    future = returns.shift(-horizon)
    label = future.rolling(horizon).std(ddof=1) * np.sqrt(TRADING_DAYS)
    label.columns.name = None
    return label


def make_future_return(returns, horizon=1):
    """未来 horizon 个交易日后的收益，作为 VaR 回归标签。"""
    label = returns.shift(-horizon)
    label.columns.name = None
    return label


def wide_labels_to_long(labels_wide):
    """把宽表标签转成与 build_features 对齐的 MultiIndex Series。"""
    labels = labels_wide.stack()
    labels.index.names = ["date", "code"]
    labels.name = "label"
    return labels
