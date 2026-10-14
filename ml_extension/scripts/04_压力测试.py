# -*- coding: utf-8 -*-
"""本地运行入口：压力测试。"""

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from portfolio_risk import calc_portfolio_summary  # noqa: E402
from stress_test import run_stress_test  # noqa: E402


def main():
    data_dir = SCRIPT_DIR.parent / "data"
    ret = pd.read_csv(data_dir / "return_daily.csv", index_col=0)
    weights = {code: 1.0 / ret.shape[1] for code in ret.columns}
    summary = calc_portfolio_summary(ret, weights, window=250)
    table = run_stress_test(summary, weights)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
