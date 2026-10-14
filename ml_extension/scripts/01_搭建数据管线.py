# -*- coding: utf-8 -*-
"""第一步：搭建数据管线（聚宽 JQData）。

股票池：贵州茅台、宁德时代、招商银行、迈瑞医疗、海康威视
数据：日频 OHLCV，前复权，最近 3 年

输出（data/ 目录）：
    price_daily.csv    宽表：行=交易日，列=股票代码，值=前复权收盘价
    return_daily.csv   宽表：行=交易日，列=股票代码，值=日收益率
                       停牌日与涨跌停日的收益按工作流程置为 NaN
    ohlcv_daily.csv    长表：OHLCV + 成交额 + 复权因子 + 停牌/涨跌停标记

运行方式：
1. 聚宽研究环境：把本文件内容放进 notebook 单元格直接运行；
2. 本地 jqdatasdk：先 pip install jqdatasdk，再取消下方 auth 注释并填入账号。
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

# ============================ 配置 ============================
DEFAULT_STOCKS = {
    "600519.XSHG": "贵州茅台",
    "300750.XSHE": "宁德时代",
    "600036.XSHG": "招商银行",
    "300760.XSHE": "迈瑞医疗",
    "002415.XSHE": "海康威视",
}

# 本地运行时优先读取 config/stock_pool.json，新增股票只改配置文件即可；
# 聚宽 notebook 直接粘贴本代码时使用 DEFAULT_STOCKS，新增股票在下方字典加一行。
try:
    POOL_PATH = Path(__file__).resolve().parent.parent / "config" / "stock_pool.json"
    if POOL_PATH.exists():
        import json

        with open(POOL_PATH, "r", encoding="utf-8") as _f:
            STOCKS = json.load(_f)
        print(f"股票池从配置文件加载：{POOL_PATH}")
    else:
        STOCKS = DEFAULT_STOCKS
except NameError:  # 聚宽 notebook 中没有 __file__
    STOCKS = DEFAULT_STOCKS

YEARS_BACK = 3
END_DATE = dt.date.today()
START_DATE = END_DATE - dt.timedelta(days=YEARS_BACK * 365)

# 聚宽 get_price 日线字段字典：
#   open/high/low/close  开高低收价（前复权后）
#   volume/money         成交量(股)/成交额(元)
#   pre_close            昨收价
#   factor               前复权因子
#   paused               是否停牌
#   high_limit/low_limit 涨停价/跌停价
FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "money",
    "pre_close",
    "factor",
    "paused",
    "high_limit",
    "low_limit",
]

# 按工作流程：涨跌停日的收益率不能直接用于波动率，置为 NaN
MASK_LIMIT_DAYS = True
# 停牌日价格由聚宽填充为前收盘，收益并非真实成交，置为 NaN
MASK_PAUSED_DAYS = True

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    OUTPUT_DIR = SCRIPT_DIR.parent / "data"
except NameError:  # 聚宽 notebook 中没有 __file__
    # 聚宽研究环境的当前目录是 /home，根级 data 目录没有写权限，
    # 所以 notebook 模式写入用户主目录 ~/data
    OUTPUT_DIR = Path.home() / "data"

# 本地使用 jqdatasdk 时，取消下面两行注释并填入账号
# import jqdatasdk
# jqdatasdk.auth("手机号/邮箱", "密码")


def fetch_stock_daily(code: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """用聚宽 get_price 拉取单只股票的前复权日线。"""
    df = get_price(
        security=code,
        start_date=start,
        end_date=end,
        frequency="daily",
        fields=FIELDS,
        skip_paused=False,   # 保留停牌日，用 paused 字段标记
        fill_paused=True,    # 停牌日价格填充为前一交易日收盘价，保证面板连续
        fq="pre",            # 前复权
        panel=False,         # 返回 DataFrame 而不是 Panel
    )
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    date_col = "date" if "date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["code"] = code
    df["name"] = STOCKS[code]
    df["paused"] = df["paused"].fillna(False).astype(bool)

    # 收盘价达到当日涨停价/跌停价时标记，用于后续收益清洗
    df["limit_up"] = df["high_limit"].notna() & np.isclose(
        df["close"], df["high_limit"], rtol=0.0, atol=1e-6
    )
    df["limit_down"] = df["low_limit"].notna() & np.isclose(
        df["close"], df["low_limit"], rtol=0.0, atol=1e-6
    )
    return df


def build_pipeline() -> None:
    """执行数据管线并输出面板 CSV。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = [fetch_stock_daily(code, START_DATE, END_DATE) for code in STOCKS]
    panel = pd.concat(frames, ignore_index=True)
    if panel.empty:
        raise RuntimeError("所有股票均未取到数据，请检查聚宽账号权限")
    all_dates = pd.to_datetime(panel["date"])
    print(
        f"数据范围：{all_dates.min().date()} ~ {all_dates.max().date()}，"
        f"共 {panel['date'].nunique()} 个交易日"
    )

    # 前复权收盘价宽表：索引=日期，列=股票代码
    price_daily = panel.pivot_table(
        index="date", columns="code", values="close", aggfunc="last"
    ).sort_index()
    price_daily.columns.name = None

    # 日收益率宽表
    return_daily = price_daily.pct_change()

    if MASK_PAUSED_DAYS:
        paused = panel.pivot_table(
            index="date", columns="code", values="paused", aggfunc="last"
        ).astype(bool)
        return_daily[paused] = np.nan

    if MASK_LIMIT_DAYS:
        limit_up = panel.pivot_table(
            index="date", columns="code", values="limit_up", aggfunc="last"
        ).astype(bool)
        limit_down = panel.pivot_table(
            index="date", columns="code", values="limit_down", aggfunc="last"
        ).astype(bool)
        return_daily[limit_up | limit_down] = np.nan

    # 输出文件（utf-8-sig 便于 Excel 直接打开）
    price_daily.to_csv(OUTPUT_DIR / "price_daily.csv", encoding="utf-8-sig")
    return_daily.to_csv(OUTPUT_DIR / "return_daily.csv", encoding="utf-8-sig")
    panel.sort_values(["code", "date"]).to_csv(
        OUTPUT_DIR / "ohlcv_daily.csv", index=False, encoding="utf-8-sig"
    )

    # 沪深300 基准收益（供 Beta 与基准对比使用）
    try:
        index = get_price(
            security="000300.XSHG",
            start_date=START_DATE,
            end_date=END_DATE,
            frequency="daily",
            fields=["close"],
            fq="pre",
            panel=False,
        )
        index = index.reset_index()
        date_col = "date" if "date" in index.columns else index.columns[0]
        index = index.rename(columns={date_col: "date"})
        index["date"] = pd.to_datetime(index["date"]).dt.normalize()
        index = index[["date", "close"]].sort_values("date")
        index.to_csv(
            OUTPUT_DIR / "index_000300.csv", index=False, encoding="utf-8-sig"
        )
        print(f"沪深300基准已保存：{OUTPUT_DIR / 'index_000300.csv'}")
    except Exception as exc:
        print(f"[警告] 沪深300获取失败：{exc}")

    print(f"\n数据已保存到：{OUTPUT_DIR}")
    print(f"price_daily: {price_daily.shape[0]} 行 x {price_daily.shape[1]} 列")
    print(f"return_daily: {return_daily.shape[0]} 行 x {return_daily.shape[1]} 列")

    for code, name in STOCKS.items():
        if code not in price_daily.columns:
            print(f"  [警告] {name}({code}) 无数据")
            continue
        s = price_daily[code].dropna()
        print(
            f"  {name}({code}): {len(s)} 个交易日, "
            f"{s.index.min().date()} ~ {s.index.max().date()}"
        )

    print("\n各股票日收益率 NaN 数（停牌/涨跌停已屏蔽）：")
    print(return_daily.isna().sum().to_string())


if __name__ == "__main__":
    build_pipeline()
