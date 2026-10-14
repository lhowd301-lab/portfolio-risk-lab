"""阶段 4：限额检查。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loader, pipeline


def main() -> None:
    config = data_loader.load_config()
    market = pipeline.stage_fetch(config)
    result = pipeline.stage_metrics(market, config)
    checks = pipeline.stage_limits(result, config)
    print(f"{'项目':<24}{'当前值':>10}{'限额':>10}  状态")
    for c in checks:
        print(f"{c['项目']:<24}{c['当前值']:>10.2%}{c['限额']:>10.2%}  {c['状态']}")


if __name__ == "__main__":
    main()
