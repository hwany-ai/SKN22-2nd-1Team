# app/pages/01_session_prob.py

from __future__ import annotations

import streamlit as st
import pandas as pd

from service.session_probability_service import (
    SessionProbabilityService,
    SessionPredictionResult,
)

from ui.header import render_header


render_header()

# ======================
# 0. Streamlit 설정 & 서비스 초기화
# ======================

st.set_page_config(page_title="세션 구매 확률 계산기", page_icon="🛒", layout="wide")

# CSS
st.markdown(
    """
    <style>
    .result-card {
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin-top: 1rem;
    }
    .high-prob {
        background: linear-gradient(135deg, #16a34a, #22c55e);
    }
    .medium-prob {
        background: linear-gradient(135deg, #eab308, #facc15);
        color: #1f2933;
    }
    .low-prob {
        background: linear-gradient(135deg, #b91c1c, #ef4444);
    }
    .sub-text {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_session_probability_service() -> SessionProbabilityService:
    """
    - 모델/어댑터는 여기서 한 번만 로드 (Streamlit 캐싱)
    - Global 평균 값은 추후 실제 데이터 기준으로 수정 가능
    """
    return SessionProbabilityService(global_avg_purchase_prob=0.15)


service = get_session_probability_service()


# ======================
# 1. UI Layout
# ======================

st.title("🛒 세션 구매 확률 계산기")
st.caption("UCI Online Shoppers Purchasing Intention Dataset 기반 예측 데모")

left_col, right_col = st.columns([1.1, 1])


# ----------------------
# 1-1. 입력 폼 (좌측)
# ----------------------
with left_col:
    st.subheader("세션 정보 입력")

    st.markdown("#### 📌 세션 활동 정보")
    col1, col2, col3 = st.columns(3)

    with col1:
        administrative = st.number_input(
            "Administrative (관리 페이지 수)",
            min_value=0,
            max_value=30,
            value=2,
            step=1,
        )
    with col2:
        informational = st.number_input(
            "Informational (정보 페이지 수)",
            min_value=0,
            max_value=30,
            value=1,
            step=1,
        )
    with col3:
        product_related = st.number_input(
            "ProductRelated (상품 페이지 수)",
            min_value=0,
            max_value=500,
            value=8,
            step=1,
        )

    st.markdown("#### 📊 행동 지표")
    col4, col5, col6 = st.columns(3)
    with col4:
        bounce_rates = st.slider(
            "BounceRates",
            min_value=0.0,
            max_value=1.0,
            value=0.02,
            step=0.01,
        )
    with col5:
        exit_rates = st.slider(
            "ExitRates",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
        )
    with col6:
        page_values = st.number_input(
            "PageValues",
            min_value=0.0,
            max_value=500.0,
            value=10.0,
            step=1.0,
        )

    st.markdown("#### 🧩 기타 세션 속성")

    months = [
        "Feb",
        "Mar",
        "Apr",
        "May",
        "June",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    month = st.selectbox("Month (방문 월)", options=months, index=10)

    visitor_type = st.selectbox(
        "VisitorType (방문자 유형)",
        options=["New_Visitor", "Returning_Visitor", "Other"],
        index=1,
    )

    weekend = st.radio(
        "Weekend (주말 방문 여부)",
        options=[False, True],
        format_func=lambda x: "주말 방문" if x else "평일 방문",
        index=0,
    )

    traffic_type = st.number_input(
        "TrafficType",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
    )

    # 모델 전략 선택 (선택 사항)
    model_strategy_label = st.selectbox(
        "사용할 모델 기준",
        options=[
            "ROC-AUC 기준 베스트 모델 사용",
            "PR-AUC 기준 베스트 모델 사용",
        ],
        index=0,
    )
    strategy_map = {
        "ROC-AUC 기준 베스트 모델 사용": "roc_auc",
        "PR-AUC 기준 베스트 모델 사용": "pr_auc",
    }
    selected_strategy = strategy_map[model_strategy_label]

    predict_btn = st.button("🔮 구매 확률 예측하기", type="primary")


def build_input_dataframe() -> pd.DataFrame:
    """
    Service/Adapter에 넘길 원본 DataFrame 생성
    (컬럼명은 학습 시 사용한 이름과 동일해야 함)
    """
    data = {
        "Administrative": [administrative],
        "Informational": [informational],
        "ProductRelated": [product_related],
        "BounceRates": [bounce_rates],
        "ExitRates": [exit_rates],
        "PageValues": [page_values],
        "Month": [month],
        "VisitorType": [visitor_type],
        "Weekend": [weekend],
        "TrafficType": [traffic_type],
    }
    df = pd.DataFrame(data)
    return df


def risk_band_to_css_class(risk_band: str) -> str:
    if risk_band == "high":
        return "high-prob"
    elif risk_band == "medium":
        return "medium-prob"
    else:
        return "low-prob"


# ----------------------
# 1-2. 결과 영역 (우측)
# ----------------------
with right_col:
    st.subheader("예측 결과")

    if predict_btn:
        input_df = build_input_dataframe()

        try:
            result: SessionPredictionResult = service.predict_session(
                input_df,
                strategy=selected_strategy,  # "roc_auc" 또는 "pr_auc"
            )
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
            st.stop()

        css_class = risk_band_to_css_class(result.risk_band)

        st.markdown(
            f"""
            <div class="result-card {css_class}">
                <h3>🧮 구매 확률: {result.probability * 100:.1f}%</h3>
                <p class="sub-text">{result.status_label}</p>
                <p class="sub-text">{result.compare_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("🔍 왜 이런 결과가 나왔나요? (설명 보기)", expanded=True):
            st.markdown("**설명 요약**")
            for r in result.reasons:
                st.markdown(f"- {r}")
            st.markdown("---")
            st.markdown(f"**평균 대비:** {result.average_text}")

        with st.expander("📁 디버깅용 입력 데이터 보기"):
            st.dataframe(input_df)
    else:
        st.info(
            "왼쪽에서 세션 정보를 입력하고 **'구매 확률 예측하기'** 버튼을 눌러주세요."
        )
