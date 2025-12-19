from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd


@dataclass(frozen=True)
class ModelArtifact:
    pipeline: Any
    best_threshold: float
    meta: Dict[str, Any]


class PurchaseIntentModelAdapter:
    """
    - artifacts/*.joblib 로딩 (pipeline + threshold)
    - 서비스에서 predict/predict_proba 호출할 수 있게 제공
    """

    def __init__(self, model_path: str | Path):
        self._model_path = Path(model_path)
        self._artifact: Optional[ModelArtifact] = None

    def load(self) -> ModelArtifact:
        if self._artifact is not None:
            return self._artifact

        if not self._model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self._model_path}")

        raw = joblib.load(self._model_path)

        # 우리가 저장한 형태: {"pipeline": ..., "best_threshold": ..., ...}
        if "pipeline" not in raw or "best_threshold" not in raw:
            raise ValueError("Invalid artifact format: expected keys ['pipeline', 'best_threshold'].")

        self._artifact = ModelArtifact(
            pipeline=raw["pipeline"],
            best_threshold=float(raw["best_threshold"]),
            meta={k: v for k, v in raw.items() if k not in ("pipeline", "best_threshold")},
        )
        return self._artifact

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        art = self.load()
        proba = art.pipeline.predict_proba(features)[:, 1]
        return pd.Series(proba, index=features.index, name="purchase_proba")

    def predict(self, features: pd.DataFrame, threshold: Optional[float] = None) -> pd.Series:
        art = self.load()
        thr = art.best_threshold if threshold is None else float(threshold)
        proba = self.predict_proba(features)
        pred = (proba >= thr).astype(int)
        return pd.Series(pred.values, index=features.index, name="purchase_pred")

    def get_threshold(self) -> float:
        return float(self.load().best_threshold)
