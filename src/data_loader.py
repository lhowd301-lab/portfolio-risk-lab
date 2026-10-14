"""数据获取与缓存层：优先使用 akshare 拉取 A 股日线，失败时回退为虚拟数据。"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Optional[Path] = None) -> dict:
    path = path or ROOT / "config" / "风控参数.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> Dict[str, Path]:
    dirs = {
        "raw": ROOT / "data" / "raw",
        "processed": ROOT / "data" / "processed",
        "reports": ROOT / "output" / "reports",
        "figures": ROOT / "output" / "figures",
        "data": ROOT / "output" / "data",
        "logs": ROOT / "output" / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, encoding="utf-8-sig")


def _fetch_stock_akshare(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    import akshare as ak

    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
        adjust=adjust,
    )
    df = df.rename(
        columns={
            "日期": "date",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "close", "volume", "amount", "turnover"]].set_index("date").sort_index()
    return df


def _fetch_index_akshare(symbol: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    em_symbol = symbol[2:] if symbol[:2] in ("sh", "sz", "bj") else symbol
    errors = []
    candidates = [
        ("stock_zh_index_daily", symbol),
        ("stock_zh_index_daily_em", em_symbol),
    ]
    for fn_name, arg in candidates:
        try:
            fn = getattr(ak, fn_name)
            df = fn(arg)
            if "date" not in df.columns and "日期" in df.columns:
                df = df.rename(columns={"日期": "date"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            if "close" not in df.columns and "收盘" in df.columns:
                df = df.rename(columns={"收盘": "close"})
            df = df[["close", "volume"]] if "volume" in df.columns else df[["close"]]
            return df.loc[start:end]
        except Exception as exc:
            errors.append(f"{fn_name}: {exc}")
    raise RuntimeError("指数获取失败: " + " | ".join(errors))


def _demo_daily(code: str, start: str, end: str, seed: int) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    drift = rng.normal(0.0002, 0.0004)
    noise = rng.normal(drift, 0.018, len(dates))
    close = 100 * np.cumprod(1 + noise)
    volume = rng.integers(1_000_000, 5_000_000, len(dates))
    amount = close * volume
    turnover = rng.uniform(0.5, 3.0, len(dates))
    return pd.DataFrame(
        {"close": close, "volume": volume, "amount": amount, "turnover": turnover},
        index=dates,
    )


@dataclass
class MarketData:
    stocks: Dict[str, pd.DataFrame]
    stock_info: Dict[str, dict]
    benchmark: pd.DataFrame
    benchmark_name: str
    is_demo: bool
    source: str


def load_market_data(config: dict, force_refresh: bool = False) -> MarketData:
    dirs = ensure_dirs()
    data_cfg = config["数据"]
    start = data_cfg["回测起点"]
    end = dt.date.today().isoformat()
    stocks: Dict[str, pd.DataFrame] = {}
    stock_info: Dict[str, dict] = {}
    is_demo = False

    for item in config["股票池"]:
        code = item["code"]
        stock_info[code] = item
        path = dirs["raw"] / f"{code}_{start}.csv"
        if path.exists() and not force_refresh:
            df = _read_csv(path)
        else:
            try:
                df = _fetch_stock_akshare(code, start, end, data_cfg["复权方式"])
            except Exception as exc:
                print(f"警告：{item['name']}({code}) 拉取失败，使用虚拟数据。原因：{exc}")
                seed = sum(ord(c) for c in code) % (2 ** 32)
                df = _demo_daily(code, start, end, seed=seed)
                is_demo = True
            _write_csv(df, path)
        stocks[code] = df

    bench_cfg = config["基准指数"]
    bench_path = dirs["raw"] / f"index_{bench_cfg['code']}.csv"
    if bench_path.exists() and not force_refresh:
        benchmark = _read_csv(bench_path)
    else:
        try:
            benchmark = _fetch_index_akshare(bench_cfg["symbol"], start, end)
        except Exception as exc:
            print(f"警告：基准指数 {bench_cfg['name']} 拉取失败，使用虚拟数据。原因：{exc}")
            benchmark = _demo_daily(bench_cfg["code"], start, end, seed=42)
            is_demo = True
        _write_csv(benchmark, bench_path)

    note = "真实数据" if not is_demo else "虚拟数据：akshare 拉取失败时生成的演示样本"
    meta = {
        "source": data_cfg["数据源"],
        "start": start,
        "end": end,
        "is_demo": is_demo,
        "note": note,
    }
    with open(dirs["data"] / "数据来源.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return MarketData(
        stocks=stocks,
        stock_info=stock_info,
        benchmark=benchmark,
        benchmark_name=bench_cfg["name"],
        is_demo=is_demo,
        source=data_cfg["数据源"],
    )
