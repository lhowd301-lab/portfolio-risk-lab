"""阶段 2-3：构建组合并计算风险指标。"""

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
    m = result["metrics"]
    print(f"数据性质: {'虚拟数据' if market.is_demo else '真实数据'}")
    print(f"期末净值: {m['final_value']:,.0f} 元")
    print(f"区间收益率: {m['period_return']:.2%}")
    print(f"年化收益率: {m['annualized_return']:.2%}")
    print(f"年化波动率: {m['annualized_vol']:.2%} / EWMA: {m['annualized_vol_ewma']:.2%}")
    print(f"95% VaR（1日）: {m['var95_hist']:.2%} / 参数法: {m['var95_param']:.2%}")
    print(f"95% CVaR（1日）: {m['cvar95']:.2%}")
    print(f"最大回撤: {m['max_drawdown']:.2%}")
    print(f"Beta: {m['beta']:.2f}, Sharpe: {m['sharpe']:.2f}, Calmar: {m['calmar']:.2f}")
    print("\n个股风险：")
    print(result["per_stock"].to_string(index=False))


if __name__ == "__main__":
    main()
