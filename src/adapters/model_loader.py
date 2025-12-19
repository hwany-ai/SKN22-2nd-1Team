from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib


@dataclass(frozen=True)
class ModelArtifact:
    """
    로딩된 모델 아티팩트 구조체.

    Attributes:
        pipeline:
            scikit-learn Pipeline 또는 estimator.
            보통 preprocess + model이 묶여 있어서 바로 predict_proba() 호출 가능.
        meta:
            pipeline을 제외한 나머지 메타 정보(파라미터, 컬럼 정보, 평가 지표 등).
            예: {"best_params": ..., "target_col": "Revenue", ...}
    """
    pipeline: Any
    meta: Dict[str, Any]


class JoblibArtifactLoader:
    """
    joblib(.joblib)로 저장된 모델 아티팩트를 로드하는 로더.

    ✅ 역할
    - artifacts/*.joblib 파일을 읽어서 내부 객체(pipeline + meta)를 반환
    - 한 번 로드한 결과를 메모리에 캐시하여, 서비스 요청마다 디스크 I/O가 반복되지 않게 함

    ✅ 기대하는 joblib 포맷 (dict)
    - 최소 키: "pipeline"
    - 추가 키들은 meta로 담아 제공됨
      예) {"pipeline": <Pipeline>, "best_params": {...}, "target_col": "Revenue", ...}

    ✅ 사용 예시
    ------------------------------------------------------------------
    from src.adapters.model_loader import JoblibArtifactLoader

    loader = JoblibArtifactLoader("artifacts/best_pr_auc_balancedrf.joblib")
    artifact = loader.load()

    pipe = artifact.pipeline
    proba = pipe.predict_proba(features_df)[:, 1]

    # 메타 정보 확인
    print(artifact.meta.get("best_params"))
    ------------------------------------------------------------------

    ⚠️ 주의사항
    - joblib로 저장된 sklearn 모델은 로드 시점에 sklearn/imbalanced-learn 버전 호환이 중요함.
      (학습/서빙 환경의 패키지 버전을 맞추는 게 안전)
    """

    def __init__(self, path: str | Path):
        """
        Args:
            path: joblib 아티팩트 파일 경로 (상대/절대 모두 가능)
        """
        self.path = Path(path)
        self._cached: Optional[ModelArtifact] = None

    def load(self) -> ModelArtifact:
        """
        아티팩트를 로드하여 반환한다. 이미 로드했다면 캐시된 값을 반환한다.

        Returns:
            ModelArtifact: (pipeline, meta)로 구성된 객체

        Raises:
            FileNotFoundError: path에 파일이 없을 때
            ValueError: joblib 내부 포맷이 예상(dict + pipeline 키)과 다를 때
        """
        if self._cached is not None:
            return self._cached

        if not self.path.exists():
            raise FileNotFoundError(f"Artifact not found: {self.path}")

        raw = joblib.load(self.path)

        # 우리가 저장한 artifact는 dict 형태를 기대
        if not isinstance(raw, dict):
            raise ValueError("Invalid artifact format: expected dict saved via joblib.dump({...}).")

        if "pipeline" not in raw:
            raise ValueError("Invalid artifact format: missing required key 'pipeline'.")

        self._cached = ModelArtifact(
            pipeline=raw["pipeline"],
            meta={k: v for k, v in raw.items() if k != "pipeline"},
        )
        return self._cached
