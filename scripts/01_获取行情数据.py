"""阶段 1：获取行情数据并缓存到 data/raw。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loader


def main() -> None:
    config = data_loader.load_config()
    force_refresh = "--refresh" in sys.argv
    market = data_loader.load_market_data(config, force_refresh=force_refresh)
    print(f"数据源: {market.source}")
    print("数据性质: " + ("虚拟数据（演示用）" if market.is_demo else "真实数据"))
    for code, df in market.stocks.items():
        name = market.stock_info[code]["name"]
        print(f"  {code} {name}: {len(df)} 行, {df.index.min().date()} ~ {df.index.max().date()}")
    bench = market.benchmark
    print(
        f"  基准 {market.benchmark_name}: {len(bench)} 行, "
        f"{bench.index.min().date()} ~ {bench.index.max().date()}"
    )


if __name__ == "__main__":
    main()
