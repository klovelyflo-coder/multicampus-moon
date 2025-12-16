"""
Phase 3: 역 상세 라인차트 + 히트맵 대시보드
Streamlit 기반 지하철 혼잡도 시각화 앱
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="지하철 혼잡도 대시보드",
    page_icon="🚇",
    layout="wide"
)

# 타이틀
st.title("🚇 지하철 혼잡도 대시보드 - 역별 분석")
st.markdown("---")

# 데이터 로딩 함수 (캐싱)
@st.cache_data
def load_data():
    """정제된 데이터 로딩 (Parquet 포맷)"""
    data_path = Path(__file__).parent / "data" / "subway_crowding_tidy.parquet"
    df = pd.read_parquet(data_path)
    return df

# 히트맵 데이터 집계 함수
def prepare_heatmap_data(df, selected_day, selected_line, selected_direction, sort_by="avg_desc"):
    """
    호선의 모든 역에 대한 히트맵 데이터 생성
    
    Args:
        df: 전체 데이터프레임
        selected_day: 선택된 요일
        selected_line: 선택된 호선
        selected_direction: 선택된 방향
        sort_by: 정렬 기준 ("avg_desc", "name", "code")
    
    Returns:
        pivot_df: 역(행) × 시간대(열) 피벗 테이블
        station_order: 정렬된 역 리스트
    """
    # 선택한 조건으로 필터링
    filtered = df[
        (df['day_type'] == selected_day) &
        (df['line'] == selected_line) &
        (df['direction'] == selected_direction)
    ]
    
    # 피벗 테이블 생성
    pivot_df = filtered.pivot_table(
        index='station_name',
        columns='time_label',
        values='crowding',
        aggfunc='mean'
    )
    
    # 정렬
    if sort_by == "avg_desc":
        # 평균 혼잡도 내림차순
        avg_crowding = pivot_df.mean(axis=1).sort_values(ascending=False)
        station_order = avg_crowding.index.tolist()
    elif sort_by == "name":
        # 가나다순
        station_order = sorted(pivot_df.index.tolist())
    elif sort_by == "code":
        # 역번호순
        station_codes = filtered[['station_name', 'station_code']].drop_duplicates()
        station_codes = station_codes.sort_values('station_code')
        station_order = station_codes['station_name'].tolist()
    else:
        station_order = pivot_df.index.tolist()
    
    pivot_df = pivot_df.reindex(station_order)
    
    return pivot_df, station_order

# 색상 스케일 범위 계산 (호선별)
def get_color_scale_range(df, selected_line):
    """
    호선별 분위수 기반 색상 범위 계산
    
    Returns:
        (vmin, vmax): 색상 스케일 범위
    """
    line_data = df[df['line'] == selected_line]['crowding']
    vmin = line_data.quantile(0.0)
    vmax = line_data.quantile(1.0)
    return vmin, vmax

# 메인 로직
def main():
    # Session State 초기화
    if 'selected_station_from_heatmap' not in st.session_state:
        st.session_state['selected_station_from_heatmap'] = None
    
    # 데이터 로드
    try:
        df = load_data()
        st.sidebar.success(f"✅ 데이터 로딩 완료: {len(df):,}행")
    except Exception as e:
        st.error(f"❌ 데이터 로딩 실패: {e}")
        st.stop()
    
    # 사이드바 - 필터
    st.sidebar.header("🔍 필터")
    
    # 필터 1: 요일구분 (평일 → 토요일 → 일요일 순서)
    day_types_raw = df['day_type'].unique().tolist()
    day_types_order = ["평일", "토요일", "일요일"]
    day_types = [d for d in day_types_order if d in day_types_raw]
    selected_day = st.sidebar.selectbox(
        "요일구분",
        options=day_types,
        index=0
    )
    
    # 필터 2: 호선
    lines = sorted(df['line'].unique().tolist())
    selected_line = st.sidebar.selectbox(
        "호선",
        options=lines,
        index=0
    )
    
    # 필터 3: 출발역 (선택된 호선의 역만 표시)
    stations_in_line = df[df['line'] == selected_line]['station_name'].unique()
    stations_sorted = sorted(stations_in_line)
    
    # 히트맵에서 선택된 역이 있으면 해당 역을 기본값으로 설정
    default_station_idx = 0
    if st.session_state['selected_station_from_heatmap'] and \
       st.session_state['selected_station_from_heatmap'] in stations_sorted:
        default_station_idx = stations_sorted.index(st.session_state['selected_station_from_heatmap'])
        # 한 번 사용 후 초기화
        st.session_state['selected_station_from_heatmap'] = None
    
    selected_station = st.sidebar.selectbox(
        "출발역",
        options=stations_sorted,
        index=default_station_idx
    )
    
    # 필터 4: 상하구분 (선택된 호선의 방향만 표시)
    directions_in_line = sorted(df[df['line'] == selected_line]['direction'].unique())
    selected_direction = st.sidebar.selectbox(
        "상하구분",
        options=directions_in_line,
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 필터를 변경하면 차트가 자동으로 업데이트됩니다.")
    
    # 데이터 필터링
    filtered_df = df[
        (df['day_type'] == selected_day) &
        (df['line'] == selected_line) &
        (df['station_name'] == selected_station) &
        (df['direction'] == selected_direction)
    ].sort_values('time_order')
    
    # 데이터 검증
    if filtered_df.empty:
        st.warning("⚠️ 선택한 조건에 해당하는 데이터가 없습니다.")
        st.stop()
    
    # 본문 영역
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("선택된 역", selected_station)
    with col2:
        avg_crowding = filtered_df['crowding'].mean()
        st.metric("평균 혼잡도", f"{avg_crowding:.1f}")
    with col3:
        max_crowding = filtered_df['crowding'].max()
        st.metric("최대 혼잡도", f"{max_crowding:.1f}")
    
    st.markdown("---")
    
    # 방향 설명 추가 (모든 호선)
    def get_direction_description(line, direction):
        """각 호선별 방향 설명 추가"""
        direction_info = {
            "1호선": {
                "상선": "상선 (서울역 방향)",
                "하선": "하선 (청량리 방향)"
            },
            "2호선": {
                "내선": "내선 (시계방향)",
                "외선": "외선 (반시계방향)"
            },
            "3호선": {
                "상선": "상선 (대화 방향)",
                "하선": "하선 (오금 방향)"
            },
            "4호선": {
                "상선": "상선 (당고개 방향)",
                "하선": "하선 (오이도 방향)"
            },
            "5호선": {
                "상선": "상선 (방화 방향)",
                "하선": "하선 (하남검단산 방향)"
            },
            "6호선": {
                "상선": "상선 (봉화산 방향)",
                "하선": "하선 (응암 방향)"
            },
            "7호선": {
                "상선": "상선 (장암 방향)",
                "하선": "하선 (부평구청 방향)"
            },
            "8호선": {
                "상선": "상선 (암사 방향)",
                "하선": "하선 (모란 방향)"
            }
        }
        
        if line in direction_info and direction in direction_info[line]:
            return direction_info[line][direction]
        return direction
    
    direction_display = get_direction_description(selected_line, selected_direction)
    
    # 라인차트 생성
    fig = px.line(
        filtered_df,
        x='time_label',
        y='crowding',
        title=f'{selected_station} ({selected_line}, {direction_display}) - {selected_day}',
        labels={'time_label': '시간대', 'crowding': '혼잡도'},
        markers=True
    )
    
    # 차트 스타일 개선
    fig.update_traces(
        line_color='#1f77b4',
        marker=dict(size=6),
        hovertemplate='<b>시간대</b>: %{x}<br><b>혼잡도</b>: %{y:.1f}<extra></extra>'
    )
    
    # 출근 시간대 강조 (07:30 ~ 09:30)
    fig.add_vrect(
        x0="07:30", x1="09:30",
        fillcolor="rgba(0, 100, 255, 0.1)",
        layer="below",
        line_width=0,
        annotation_text="출근",
        annotation_position="top left",
        annotation=dict(font_size=12, font_color="blue")
    )
    
    # 퇴근 시간대 강조 (17:30 ~ 19:30)
    fig.add_vrect(
        x0="17:30", x1="19:30",
        fillcolor="rgba(255, 100, 0, 0.1)",
        layer="below",
        line_width=0,
        annotation_text="퇴근",
        annotation_position="top left",
        annotation=dict(font_size=12, font_color="red")
    )
    
    # 레이아웃 설정
    fig.update_layout(
        height=500,
        xaxis_title="시간대",
        yaxis_title="혼잡도",
        hovermode='x unified',
        xaxis=dict(
            tickangle=-45,
            tickmode='linear'
        ),
        yaxis=dict(
            rangemode='tozero'
        )
    )
    
    # 차트 표시
    st.plotly_chart(fig, use_container_width=True)
    
    # 추가 정보
    with st.expander("📊 상세 데이터 보기"):
        st.dataframe(
            filtered_df[['time_label', 'crowding']].rename(
                columns={'time_label': '시간대', 'crowding': '혼잡도'}
            ),
            hide_index=True,
            use_container_width=True
        )
    
    # ============================================
    # 히트맵 섹션
    # ============================================
    st.markdown("---")
    st.markdown("## 📊 역×시간대 혼잡도 히트맵")
    st.markdown(f"**{selected_line} {direction_display}** 의 모든 역에 대한 시간대별 혼잡도를 한눈에 확인하세요.")
    
    # 정렬 옵션
    col_sort, col_info = st.columns([1, 3])
    with col_sort:
        sort_options = {
            "평균 혼잡도 내림차순": "avg_desc",
            "가나다순": "name",
            "역번호순": "code"
        }
        sort_label = st.selectbox(
            "역 정렬 기준",
            options=list(sort_options.keys()),
            index=0
        )
        sort_by = sort_options[sort_label]
    
    with col_info:
        st.info("💡 히트맵에서 특정 역을 확인하려면 아래에서 역을 선택하면 위의 라인차트가 자동으로 업데이트됩니다.")
    
    # 히트맵 데이터 준비
    try:
        heatmap_df, station_order = prepare_heatmap_data(
            df, selected_day, selected_line, selected_direction, sort_by
        )
        
        # 역이 없는 경우 처리
        if heatmap_df.empty or len(station_order) == 0:
            st.warning("⚠️ 히트맵을 표시할 데이터가 없습니다.")
        else:
            # 색상 범위 계산
            vmin, vmax = get_color_scale_range(df, selected_line)
            
            # 히트맵 생성
            fig_heatmap = px.imshow(
                heatmap_df,
                labels=dict(x="시간대", y="역명", color="혼잡도"),
                x=heatmap_df.columns,
                y=heatmap_df.index,
                color_continuous_scale="RdYlGn_r",  # 빨강-노랑-초록 역순
                aspect="auto",
                title=f"역×시간대 혼잡도 히트맵 ({selected_line}, {direction_display}) - {selected_day}",
                zmin=vmin,
                zmax=vmax
            )
            
            # 스타일 설정
            fig_heatmap.update_traces(
                hovertemplate='<b>역</b>: %{y}<br><b>시간대</b>: %{x}<br><b>혼잡도</b>: %{z:.1f}<extra></extra>'
            )
            
            # 높이를 역 수에 비례하여 조정 (최소 400px, 역당 약 25px)
            heatmap_height = max(400, len(station_order) * 25)
            
            fig_heatmap.update_layout(
                height=heatmap_height,
                xaxis_title="시간대",
                yaxis_title="역명",
                xaxis=dict(
                    side="bottom", 
                    tickangle=-45,
                    tickmode='linear'
                ),
                yaxis=dict(
                    autorange="reversed"  # 상단부터 표시
                )
            )
            
            # 히트맵 표시
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # 히트맵에서 역 선택 UI
            st.markdown("### 🔍 히트맵에서 역 상세보기")
            
            col_select, col_button = st.columns([3, 1])
            with col_select:
                selected_from_heatmap = st.selectbox(
                    "역을 선택하면 위의 라인차트가 업데이트됩니다",
                    options=["선택하세요..."] + station_order,
                    key="heatmap_station_selector"
                )
            
            with col_button:
                st.write("")  # 여백
                st.write("")  # 여백
                if st.button("라인차트로 이동", type="primary"):
                    if selected_from_heatmap and selected_from_heatmap != "선택하세요...":
                        st.session_state['selected_station_from_heatmap'] = selected_from_heatmap
                        st.rerun()
            
            # 자동 이동 (버튼 없이 선택만으로)
            if selected_from_heatmap and selected_from_heatmap != "선택하세요..." and selected_from_heatmap != selected_station:
                if st.button(f"'{selected_from_heatmap}' 역 상세보기", key="auto_move"):
                    st.session_state['selected_station_from_heatmap'] = selected_from_heatmap
                    st.rerun()
    
    except Exception as e:
        st.error(f"❌ 히트맵 생성 중 오류 발생: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
