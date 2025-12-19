#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build (train + save) BalancedRandomForest artifact using FIXED best hyperparameters
already found from PR-AUC optimization.

Input:
  - data/processed/train.csv
  - (optional) data/processed/test.csv for sanity check metrics

Output:
  - artifacts/best_pr_auc_balancedrf.joblib (compressed joblib)

Artifact contains:
  - pipeline: preprocess(RobustScaler+OneHot) + BalancedRandomForestClassifier(best_params)
  - best_params: dict
  - cv_best_pr_auc: float (optional, from your past experiment)
  - test_metrics: dict (if test has label)
  - num_cols, cat_cols, target_col, meta
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score

from imblearn.ensemble import BalancedRandomForestClassifier


# === 고정: 이미 찾은 best params ===
BEST_PARAMS = {
    "sampling_strategy": 0.5,
    "replacement": False,
    "n_estimators": 300,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
    "max_features": 0.3,
    "max_depth": 8,
}

# 기록한 값도 artifact에 같이 저장
PAST_BEST_CV_PR_AUC = 0.7559602666233075


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build artifact with fixed best PR-AUC params (BalancedRF).")
    p.add_argument("--train", type=str, default="data/processed/train.csv", help="Path to train.csv")
    p.add_argument("--test", type=str, default="data/processed/test.csv", help="Path to test.csv (optional for eval)")
    p.add_argument("--target", type=str, default="Revenue", help="Target column name")
    p.add_argument("--out", type=str, default="../app/artifacts/best_pr_auc_balancedrf.joblib", help="Output joblib path")

    p.add_argument("--random_state", type=int, default=42, help="Random seed")
    p.add_argument("--no_eval", action="store_true", help="Skip evaluation even if test has label")

    # 용량 줄이기용 joblib 압축 (기본 xz:3)
    p.add_argument("--compress", type=str, default="xz:3",
                   help="joblib compress option, e.g. 'xz:3', 'zlib:3', or 'none'")
    return p.parse_args()


def parse_compress_arg(s: str):
    s = (s or "").strip().lower()
    if s in ("none", "0", "false", "off"):
        return 0
    if ":" in s:
        method, lvl = s.split(":", 1)
        return (method, int(lvl))
    if s.isdigit():
        return int(s)
    return ("xz", 3)


def main() -> None:
    args = parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    # ---- Load train
    train_df = pd.read_csv(train_path)
    if args.target not in train_df.columns:
        raise ValueError(f"Target '{args.target}' not found in train columns: {list(train_df.columns)}")

    X_train = train_df.drop(columns=[args.target])
    y_train = train_df[args.target].astype(int)

    # ---- Column typing (auto)
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    brf = BalancedRandomForestClassifier(
        **BEST_PARAMS,
        n_jobs=-1,
        random_state=args.random_state,
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", brf),
    ])

    # ---- Fit full train
    pipeline.fit(X_train, y_train)

    print("\n=== Using fixed best params ===")
    print(json.dumps({f"model__{k}": v for k, v in BEST_PARAMS.items()}, indent=2))

    # ---- Optional eval on test
    test_metrics = {}
    if (not args.no_eval) and test_path.exists():
        test_df = pd.read_csv(test_path)
        if args.target in test_df.columns:
            X_test = test_df.drop(columns=[args.target])
            y_test = test_df[args.target].astype(int)

            test_proba = pipeline.predict_proba(X_test)[:, 1]
            test_metrics = {
                "test_pr_auc": float(average_precision_score(y_test, test_proba)),
                "test_roc_auc": float(roc_auc_score(y_test, test_proba)),
            }
            print("\n=== TEST (reference) ===")
            print("TEST PR-AUC :", test_metrics["test_pr_auc"])
            print("TEST ROC-AUC:", test_metrics["test_roc_auc"])
        else:
            print("\n(test.csv에 target 컬럼이 없어 평가를 스킵합니다.)")

    # ---- Save artifact (compressed)
    artifact = {
        "pipeline": pipeline,
        "best_params": {f"model__{k}": v for k, v in BEST_PARAMS.items()},
        "cv_best_pr_auc": float(PAST_BEST_CV_PR_AUC),
        "test_metrics": test_metrics,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target_col": args.target,
        "meta": {
            "builder": "build_best_pr_auc_balancedrf.py",
            "scoring_origin": "average_precision (PR-AUC)",
            "note": "Fixed params from prior RandomizedSearchCV run",
            "random_state": args.random_state,
        }
    }

    compress_opt = parse_compress_arg(args.compress)
    joblib.dump(artifact, out_path, compress=compress_opt)
    print(f"\nSaved artifact to: {out_path.resolve()}")
    print(f"joblib.compress = {compress_opt}")


if __name__ == "__main__":
    main()
