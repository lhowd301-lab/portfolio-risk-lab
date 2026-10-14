# -*- coding: utf-8 -*-
"""本地运行入口：机器学习风险增强实验。"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from train_ml import main  # noqa: E402


if __name__ == "__main__":
    main()
