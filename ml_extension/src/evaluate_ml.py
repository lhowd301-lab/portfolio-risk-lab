# -*- coding: utf-8 -*-
"""机器学习模型评估工具。

包含：
- rmse / mae：波动率预测误差；
- kupiec_failure_rate：VaR 回测失败率（Kupiec 检验）。
"""

import numpy as np

__all__ = ["rmse", "mae", "kupiec_failure_rate"]


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def kupiec_failure_rate(y_true, y_pred_var, confidence=0.95):
    """VaR 回测：统计实际损失超过 VaR 的比例，并给出 Kupiec 检验。

    y_true 为实际收益（可为负），y_pred_var 为 VaR 正值。
    95% VaR 理论失败率 5%，一般 4%~6% 视为可接受。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_var = np.asarray(y_pred_var, dtype=float)
    n = len(y_true)
    failures = int(np.sum(y_true < -y_pred_var))
    expected = 1.0 - confidence
    rate = failures / n if n > 0 else np.nan

    stat = None
    p_value = None
    if 0 < failures < n:
        p = expected
        phat = rate
        lr = -2.0 * (
            (n - failures) * np.log(1.0 - p)
            + failures * np.log(p)
            - (n - failures) * np.log(1.0 - phat)
            - failures * np.log(phat)
        )
        stat = float(lr)
        try:
            from scipy.stats import chi2

            p_value = float(chi2.sf(lr, 1))
        except Exception:
            p_value = None

    return {
        "sample_size": n,
        "failures": failures,
        "failure_rate": rate,
        "expected_rate": expected,
        "ratio": rate / expected if expected > 0 else np.nan,
        "kupiec_stat": stat,
        "kupiec_p_value": p_value,
    }
