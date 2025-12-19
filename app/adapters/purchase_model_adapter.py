from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd

ModelStrategy = Literal["roc_auc", "pr_auc"]


@dataclass
class PurchaseModelAdapterConfig:
    root_dir: Path
    roc_auc_model_path: Path
    pr_auc_model_path: Path

    @classmethod
    def from_default_layout(cls) -> "PurchaseModelAdapterConfig":
        # 현재 파일: ROOT/app/adapter/purchase_model_adapter.py
        root_dir = Path(__file__).resolve().parents[2]  # ROOT
        artifact_dir = root_dir / "artifact"
        return cls(
            root_dir=root_dir,
            roc_auc_model_path=artifact_dir / "best_balancedrf_pipeline.joblib",
            pr_auc_model_path=artifact_dir / "best_pr_auc_balancedrf.joblib",
        )


class PurchaseModelAdapter:
    """
    DataFrame -> 구매 확률
    """

    def __init__(self, config: Optional[PurchaseModelAdapterConfig] = None):
        self.config = config or PurchaseModelAdapterConfig.from_default_layout()
        self._roc_auc_model = None
        self._pr_auc_model = None

    def _load_roc_auc_model(self):
        if self._roc_auc_model is None:
            self._roc_auc_model = joblib.load(self.config.roc_auc_model_path)
        return self._roc_auc_model

    def _load_pr_auc_model(self):
        if self._pr_auc_model is None:
            self._pr_auc_model = joblib.load(self.config.pr_auc_model_path)
        return self._pr_auc_model

    def _get_model(self, strategy: ModelStrategy):
        if strategy == "roc_auc":
            return self._load_roc_auc_model()
        elif strategy == "pr_auc":
            return self._load_pr_auc_model()
        else:
            raise ValueError(f"Unknown model strategy: {strategy}")

    def predict_proba(
        self,
        session_df: pd.DataFrame,
        strategy: ModelStrategy = "roc_auc",
    ):
        model = self._get_model(strategy)
        # 모델이 pipeline 이라고 가정 (전처리 포함)
        return model.predict_proba(session_df)

    def predict_purchase_probability(
        self,
        session_df: pd.DataFrame,
        strategy: ModelStrategy = "roc_auc",
    ) -> float:
        proba = self.predict_proba(session_df, strategy=strategy)
        return float(proba[0][1])
