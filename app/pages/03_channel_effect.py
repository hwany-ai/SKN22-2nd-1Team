import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

# --------------------------------------------------------------------------------
# 0. 경로 설정
# --------------------------------------------------------------------------------
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.adapters.dataset_loader import DatasetLoader

# --------------------------------------------------------------------------------
# 1. 페이지 설정 및 데이터 로드
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="채널 및 지역 효과 분석",
    page_icon="📢",
    layout="wide"
)

@st.cache_data
def load_data_from_adapter():
    loader = DatasetLoader(base_path=root_path)
    try:
        return loader.load_train_data()
    except FileNotFoundError as e:
        st.error(f"❌ 데이터 로드 실패: {e}")
        return None

df = load_data_from_adapter()

if df is not None:
    st.title("📢 채널 및 지역 효과 분석")
    st.markdown("---")

    st.header("1. 유입 채널(TrafficType) 및 지역(Region)별 효율 분석")
    st.info("💡 **전환율(Conversion Rate)**: 해당 채널/지역 방문자 중 실제로 구매(Revenue)한 비율")

    # 그래프 종류 선택 옵션 추가
    plot_type = st.radio(
        "📊 그래프 스타일 선택:", 
        ["Bar Chart (막대)", "Line Chart (선)", "Area Chart (영역)", "Scatter Plot (산점도)"], 
        horizontal=True
    )

    col1, col2 = st.columns(2)

    def create_dynamic_plot(data, x_col, y_col, 
                            chart_type, 
                            color_scale='Blues', 
                            x_label=None, y_label=None):
        """선택된 차트 타입에 따라 Plotly Figure 생성"""
        common_args = {
            'data_frame': data,
            'x': x_col,
            'y': y_col,
            'labels': {y_col: y_label, x_col: x_label}
        }
        
        if "Bar" in chart_type:
            fig = px.bar(**common_args, color=y_col, color_continuous_scale=color_scale, text_auto='.1f')
        elif "Line" in chart_type:
            fig = px.line(**common_args, markers=True)
            fig.update_traces(line_color=color_scale.lower() if isinstance(color_scale, str) and color_scale in ['red', 'blue', 'green'] else None)
        elif "Area" in chart_type:
            fig = px.area(**common_args)
        elif "Scatter" in chart_type:
            fig = px.scatter(**common_args, color=y_col, size=y_col, color_continuous_scale=color_scale)
        else:
            fig = px.bar(**common_args)
        
        return fig

    # TrafficType
    with col1:
        st.subheader("🚦 Traffic Type 별 구매 전환율")
        traffic_eff = df.groupby('TrafficType')['Revenue'].mean().reset_index()
        traffic_eff['Revenue'] = traffic_eff['Revenue'] * 100
        traffic_eff = traffic_eff.sort_values(by='Revenue', ascending=False)
        # 카테고리 순서 유지를 위해
        traffic_eff['TrafficType'] = traffic_eff['TrafficType'].astype(str)

        fig_traffic = create_dynamic_plot(
            traffic_eff, 'TrafficType', 'Revenue', 
            plot_type, 
            color_scale='Blues',
            x_label='Traffic Type ID', y_label='구매 전환율 (%)'
        )
        fig_traffic.update_layout(xaxis_type='category')
        st.plotly_chart(fig_traffic, use_container_width=True)

    # Region
    with col2:
        st.subheader("🌍 지역(Region) 별 구매 전환율")
        region_eff = df.groupby('Region')['Revenue'].mean().reset_index()
        region_eff['Revenue'] = region_eff['Revenue'] * 100
        region_eff = region_eff.sort_values(by='Revenue', ascending=False)
        region_eff['Region'] = region_eff['Region'].astype(str)

        fig_region = create_dynamic_plot(
            region_eff, 'Region', 'Revenue', 
            plot_type, 
            color_scale='Greens',
            x_label='Region ID', y_label='구매 전환율 (%)'
        )
        fig_region.update_layout(xaxis_type='category')
        st.plotly_chart(fig_region, use_container_width=True)
