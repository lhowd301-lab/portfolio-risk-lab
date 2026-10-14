"""阶段 5：压力测试。"""

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
    stress_results = pipeline.stage_stress(result, config)
    print(f"{'情景':<40}{'持有期':>6}{'累计亏损':>10}{'触发熔断':>10}")
    for s in stress_results:
        trigger = "是" if s["是否触发熔断"] else "否"
        print(
            f"{s['情景']:<40}{s['持有期（天）']:>6}{s['组合累计亏损']:>10.2%}{trigger:>10}"
        )


if __name__ == "__main__":
    main()
