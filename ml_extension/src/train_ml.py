# -*- coding: utf-8 -*-
"""机器学习训练脚本（walk-forward 回测 + 最终模型保存）。

用法（本地运行）：
    py -3.12 scripts/03_机器学习实验.py --task volatility --horizon 21
    py -3.12 scripts/03_机器学习实验.py --task var --confidence 0.95
"""

import argparse
import joblib
from pathlib import Path

import numpy as np
import pandas as pd

from features import (
    build_features,
    make_future_return,
    make_volatility_label,
    wide_labels_to_long,
)
from evaluate_ml import mae, rmse, kupiec_failure_rate

__all__ = [
    "expanding_splits",
    "run_walk_forward",
    "fit_final_model",
]


def _default_model(seed=42, objective="regression", alpha=None):
    """优先 LightGBM，缺失时退回 sklearn。"""
    try:
        import lightgbm as lgb

        params = dict(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            random_state=seed,
            verbose=-1,
        )
        if objective == "quantile":
            params.update(objective="quantile", alpha=alpha)
        return lgb.LGBMRegressor(**params)
    except ImportError:
        from sklearn.ensemble import (
            GradientBoostingRegressor,
            RandomForestRegressor,
        )

        if objective == "quantile":
            return GradientBoostingRegressor(
                loss="quantile", alpha=alpha, n_estimators=200, random_state=seed
            )
        return RandomForestRegressor(n_estimators=200, random_state=seed)


def expanding_splits(dates, n_splits=4):
    """按时间顺序做 expanding window 切分，返回 (train_mask, test_mask)。"""
    dates = np.asarray(dates)
    unique = np.array(sorted(set(pd.Index(dates))))
    n = len(unique)
    block = n // (n_splits + 1)
    if block < 1:
        raise ValueError("样本量不足以做 walk-forward 切分")

    splits = []
    for k in range(1, n_splits + 1):
        train_cut_pos = k * block
        test_start_pos = train_cut_pos
        test_end_pos = min((k + 1) * block, n)
        train_cut = unique[train_cut_pos]
        test_start = unique[test_start_pos]
        test_end = unique[test_end_pos - 1]
        splits.append(
            (
                dates < train_cut,
                (dates >= test_start) & (dates <= test_end),
            )
        )
    return splits


def run_walk_forward(
    features,
    labels,
    objective="regression",
    alpha=None,
    n_splits=4,
    seed=42,
):
    """walk-forward 训练与样本外预测，返回预测表、分段评估、每段模型。"""
    data = features.join(labels).dropna().sort_index()
    if len(data) == 0:
        raise ValueError("特征与标签对齐后没有可用样本")

    feature_cols = list(features.columns)
    dates = data.index.get_level_values("date").values
    predictions = []
    models = []
    split_metrics = []

    for train_mask, test_mask in expanding_splits(dates, n_splits=n_splits):
        train = data.loc[train_mask]
        test = data.loc[test_mask]
        if len(train) == 0 or len(test) == 0:
            continue

        model = _default_model(seed=seed, objective=objective, alpha=alpha)
        model.fit(train[feature_cols], train["label"])
        pred = pd.Series(
            model.predict(test[feature_cols]),
            index=test.index,
            name="pred",
        )
        predictions.append(pred)
        models.append(model)
        split_metrics.append(
            {
                "train_start": train.index.get_level_values("date").min(),
                "train_end": train.index.get_level_values("date").max(),
                "test_start": test.index.get_level_values("date").min(),
                "test_end": test.index.get_level_values("date").max(),
                "rmse": rmse(test["label"], pred),
                "mae": mae(test["label"], pred),
            }
        )

    out = pd.concat(predictions).sort_index()
    out.name = "pred"
    return out, pd.DataFrame(split_metrics), models


def fit_final_model(features, labels, objective="regression", alpha=None, seed=42):
    """用全部样本训练最终模型，供每日预测使用。"""
    data = features.join(labels).dropna()
    if len(data) == 0:
        raise ValueError("没有可用训练样本")
    model = _default_model(seed=seed, objective=objective, alpha=alpha)
    model.fit(data[list(features.columns)], data["label"])
    return model


def main():
    parser = argparse.ArgumentParser(description="聚宽数据 -> ML 风险模型训练")
    parser.add_argument("--task", choices=["volatility", "var"], default="volatility")
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--n-splits", type=int, default=4)
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data"
    price = pd.read_csv(data_dir / "price_daily.csv", index_col=0)
    ret = pd.read_csv(data_dir / "return_daily.csv", index_col=0)
    price.index = pd.to_datetime(price.index)
    ret.index = pd.to_datetime(ret.index)
    ohlcv = pd.read_csv(data_dir / "ohlcv_daily.csv")
    volume = ohlcv.pivot_table(
        index="date", columns="code", values="volume", aggfunc="last"
    )
    volume.index = pd.to_datetime(volume.index)

    print("构建特征...")
    features = build_features(price, ret, volume_daily=volume)
    print(f"特征表：{features.shape[0]} 行 x {features.shape[1]} 列")

    if args.task == "volatility":
        labels = wide_labels_to_long(
            make_volatility_label(ret, horizon=args.horizon)
        )
        objective = "regression"
        alpha = None
    else:
        labels = wide_labels_to_long(make_future_return(ret, horizon=1))
        objective = "quantile"
        alpha = 1.0 - args.confidence

    print("walk-forward 训练...")
    pred, metrics, _ = run_walk_forward(
        features,
        labels,
        objective=objective,
        alpha=alpha,
        n_splits=args.n_splits,
    )
    print(metrics.round(6).to_string(index=False))

    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(model_dir / f"ml_{args.task}_predictions.csv", encoding="utf-8-sig")

    final_model = fit_final_model(
        features, labels, objective=objective, alpha=alpha
    )
    joblib.dump(final_model, model_dir / f"ml_{args.task}_model.pkl")
    print(f"预测与模型已保存到：{model_dir}")

    if args.task == "var":
        # 预测的是收益率 5% 分位数，VaR 取负号
        var_pred = -pred
        result = kupiec_failure_rate(
            labels.reindex(pred.index),
            var_pred,
            confidence=args.confidence,
        )
        print("Kupiec 回测：", {k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()})


if __name__ == "__main__":
    main()
