"""阶段 6：生成风控报告（Markdown + Excel + 图表）。"""

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
    pipeline.stage_limits(result, config)
    pipeline.stage_stress(result, config)
    outputs = pipeline.stage_report(result, config)
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
