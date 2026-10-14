# -*- coding: utf-8 -*-
"""本地运行入口：第六步历史回测验证。"""

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from backtest import run_comparison  # noqa: E402
from threshold_engine import load_limits  # noqa: E402


def main():
    data_dir = PROJECT_DIR / "data"
    output_dir = PROJECT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    ret = pd.read_csv(data_dir / "return_daily.csv", index_col=0)
    weights = {code: 1.0 / ret.shape[1] for code in ret.columns}
    limits = load_limits()

    result = run_comparison(
        ret,
        weights,
        daily_loss_limit=0.02,
        max_drawdown_limit=limits.get("max_drawdown", 0.15),
    )

    print("回测指标对比：")
    print(result["metrics"].round(4).to_string())

    result["metrics"].round(6).to_csv(
        output_dir / "回测指标对比.csv", encoding="utf-8-sig"
    )
    result["baseline"].round(6).to_csv(
        output_dir / "基准净值.csv", encoding="utf-8-sig"
    )
    result["controlled"].round(6).to_csv(
        output_dir / "风控净值.csv", encoding="utf-8-sig"
    )

    lines = [
        "# 历史回测验证报告",
        "",
        "## 指标对比（基准 vs 风控）",
        "",
        result["metrics"].round(4).to_markdown(),
        "",
        "风控规则：单日亏损超过 2%，或回撤超过 15% 时，降仓至 50%，冷却 5 天后恢复。",
        "",
    ]
    (output_dir / "回测报告.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已保存到：{output_dir}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        (result["baseline"]["nav"] * 100).plot(
            label="基准组合（无风控）", linewidth=1.5
        )
        (result["controlled"]["nav"] * 100).plot(
            label="风控组合", linewidth=1.5
        )
        plt.title("组合净值对比")
        plt.ylabel("净值")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "回测净值对比.png", dpi=150)
        plt.close()
        print("净值图已保存：", output_dir / "回测净值对比.png")
    except ImportError:
        print("未安装 matplotlib，跳过净值图")


if __name__ == "__main__":
    main()
