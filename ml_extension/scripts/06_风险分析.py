# -*- coding: utf-8 -*-
"""本地运行入口：单票风险分析 + 组合风险与基准对比。"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from risk_analysis import main  # noqa: E402


if __name__ == "__main__":
    main()
