"""A股风控模型端到端工作流：一条命令跑完六个阶段。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loader, pipeline


def main() -> None:
    config = data_loader.load_config()

    print("== 阶段 1/6：数据获取 ==")
    market = pipeline.stage_fetch(config, force_refresh="--refresh" in sys.argv)
    print("数据性质: " + ("虚拟数据（演示用）" if market.is_demo else "真实数据"))

    print("\n== 阶段 2/6：组合构建 ==")
    result = pipeline.stage_metrics(market, config)
    print(f"组合净值: {result['nav'].iloc[-1]:,.0f} 元，样本 {len(result['nav'])} 个交易日")

    print("\n== 阶段 3/6：风险度量 ==")
    m = result["metrics"]
    print(
        f"年化波动率 {m['annualized_vol']:.2%}，95% VaR {m['var95_hist']:.2%}，"
        f"CVaR {m['cvar95']:.2%}，最大回撤 {m['max_drawdown']:.2%}"
    )

    print("\n== 阶段 4/6：限额检查 ==")
    checks = pipeline.stage_limits(result, config)
    for c in checks:
        print(f"  {c['项目']}: {c['当前值']:.2%} / {c['限额']:.2%} -> {c['状态']}")

    print("\n== 阶段 5/6：压力测试 ==")
    stress_results = pipeline.stage_stress(result, config)
    for s in stress_results:
        trigger = "触发" if s["是否触发熔断"] else "未触发"
        print(f"  {s['情景']}: 累计亏损 {s['组合累计亏损']:.2%} -> {trigger}")

    print("\n== 阶段 6/6：生成报告 ==")
    outputs = pipeline.stage_report(result, config)
    for key, path in outputs.items():
        print(f"  {key}: {path}")

    print("\n工作流执行完成。")


if __name__ == "__main__":
    main()
