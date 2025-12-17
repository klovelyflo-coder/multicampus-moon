"""
Phase 4: 역 상세 라인차트 + 히트맵 + Top N 랭킹 대시보드
Streamlit 기반 지하철 혼잡도 시각화 앱
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# 출근/퇴근 시간대 정의 (프로젝트 요구사항 기준)
RUSH_HOUR_MORNING = ["07:30", "08:00", "08:30", "09:00", "09:30"]
RUSH_HOUR_EVENING = ["17:30", "18:00", "18:30", "19:00", "19:30"]

# 페이지 설정
st.set_page_config(
    page_title="지하철 혼잡도 대시보드",
    page_icon="🚇",
    layout="wide"
)

# 타이틀
st.title("🚇 지하철 혼잡도 대시보드")
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

# KPI 계산 함수
def calculate_kpi(df, selected_day, selected_line, selected_direction):
    """
    선택된 조건에 대한 전체 KPI 계산
    
    Args:
        df: 전체 데이터프레임
        selected_day: 선택된 요일
        selected_line: 선택된 호선
        selected_direction: 선택된 방향
    
    Returns:
        dict: KPI 딕셔너리 또는 None
    """
    # 선택된 조건으로 필터링
    filtered = df[
        (df['day_type'] == selected_day) &
        (df['line'] == selected_line) &
        (df['direction'] == selected_direction)
    ]
    
    if filtered.empty:
        return None
    
    # 전체 평균 혼잡도
    avg_crowding = filtered['crowding'].mean()
    
    # 역별 평균 혼잡도 계산
    station_avg = filtered.groupby('station_name')['crowding'].mean()
    max_station = station_avg.idxmax()
    max_crowding = station_avg.max()
    
    # 시간대별 평균 혼잡도 계산하여 피크 시간 찾기
    time_avg = filtered.groupby('time_label')['crowding'].mean()
    peak_time = time_avg.idxmax()
    
    # 총 역 수
    total_stations = filtered['station_name'].nunique()
    
    # 출퇴근 시간대 평균
    morning_data = filtered[filtered['time_label'].isin(RUSH_HOUR_MORNING)]
    evening_data = filtered[filtered['time_label'].isin(RUSH_HOUR_EVENING)]
    
    morning_avg = morning_data['crowding'].mean() if not morning_data.empty else 0
    evening_avg = evening_data['crowding'].mean() if not evening_data.empty else 0
    
    return {
        'avg_crowding': avg_crowding,
        'max_station': max_station,
        'max_crowding': max_crowding,
        'peak_time': peak_time,
        'total_stations': total_stations,
        'morning_avg': morning_avg,
        'evening_avg': evening_avg
    }

# 출퇴근 시간대 랭킹 계산 함수
def calculate_rush_hour_ranking(df, selected_day, rush_hour_type="morning", top_n=10):
    """
    출근/퇴근 시간대의 혼잡한 역 Top N 계산
    
    Args:
        df: 전체 데이터프레임
        selected_day: 선택된 요일
        rush_hour_type: "morning", "evening", 또는 "all_day"
        top_n: 상위 몇 개 역
    
    Returns:
        ranking_df: 랭킹 데이터프레임
    """
    # 시간대 선택
    if rush_hour_type == "all_day":
        # 주말: 전체 시간대 데이터 사용
        rush_df = df[df['day_type'] == selected_day]
    else:
        # 평일: 출퇴근 시간대만 사용
        time_labels = RUSH_HOUR_MORNING if rush_hour_type == "morning" else RUSH_HOUR_EVENING
        rush_df = df[
            (df['day_type'] == selected_day) &
            (df['time_label'].isin(time_labels))
        ]
    
    if rush_df.empty:
        return pd.DataFrame()
    
    # 역x방향 단위로 집계
    grouped_data = []
    for (station, line, direction), group in rush_df.groupby(['station_name', 'line', 'direction']):
        avg_crowding = group['crowding'].mean()
        # 피크 시간: 해당 역×방향에서 가장 혼잡한 시간
        peak_idx = group['crowding'].idxmax()
        peak_time = group.loc[peak_idx, 'time_label']
        
        grouped_data.append({
            'station_name': station,
            'line': line,
            'direction': direction,
            'avg_crowding': avg_crowding,
            'peak_time': peak_time
        })
    
    ranking_df = pd.DataFrame(grouped_data)
    
    # 평균 혼잡도 기준 내림차순 정렬 후 Top N
    ranking_df = ranking_df.sort_values('avg_crowding', ascending=False).head(top_n)
    
    # 순위 추가
    ranking_df.insert(0, 'rank', range(1, len(ranking_df) + 1))
    
    return ranking_df.reset_index(drop=True)

# 노선별 출퇴근 시간대 랭킹 계산 함수
def calculate_rush_hour_ranking_by_line(df, selected_day, selected_line, selected_direction, rush_hour_type="morning", top_n=10):
    """
    특정 노선의 출근/퇴근 시간대 혼잡한 역 Top N 계산
    
    Args:
        df: 전체 데이터프레임
        selected_day: 선택된 요일
        selected_line: 선택된 호선
        selected_direction: 선택된 방향
        rush_hour_type: "morning" 또는 "evening"
        top_n: 상위 몇 개 역
    
    Returns:
        ranking_df: 랭킹 데이터프레임
    """
    # 시간대 선택
    time_labels = RUSH_HOUR_MORNING if rush_hour_type == "morning" else RUSH_HOUR_EVENING
    
    # 선택한 노선, 방향, 시간대로 필터링
    rush_df = df[
        (df['day_type'] == selected_day) &
        (df['line'] == selected_line) &
        (df['direction'] == selected_direction) &
        (df['time_label'].isin(time_labels))
    ]
    
    if rush_df.empty:
        return pd.DataFrame()
    
    # 역 단위로 집계
    grouped_data = []
    for station, group in rush_df.groupby('station_name'):
        avg_crowding = group['crowding'].mean()
        # 피크 시간: 해당 역에서 가장 혼잡한 시간
        peak_idx = group['crowding'].idxmax()
        peak_time = group.loc[peak_idx, 'time_label']
        
        grouped_data.append({
            'station_name': station,
            'avg_crowding': avg_crowding,
            'peak_time': peak_time
        })
    
    ranking_df = pd.DataFrame(grouped_data)
    
    # 평균 혼잡도 기준 내림차순 정렬 후 Top N
    ranking_df = ranking_df.sort_values('avg_crowding', ascending=False).head(top_n)
    
    # 순위 추가
    ranking_df.insert(0, 'rank', range(1, len(ranking_df) + 1))
    
    return ranking_df.reset_index(drop=True)

# 메인 로직
def main():
    # Session State 초기화
    if 'selected_station_from_heatmap' not in st.session_state:
        st.session_state['selected_station_from_heatmap'] = None
    
    # 데이터 로드
    try:
        df = load_data()
    except Exception as e:
        st.error(f"❌ 데이터 로딩 실패: {e}")
        st.stop()
    
    # 사이드바 - 탭 선택
    st.sidebar.header("🚇 지하철 혼잡도 대시보드")
    
    selected_tab = st.sidebar.radio(
        "메뉴 선택",
        ["📈 역 상세 분석", "🏆 전체 혼잡도 랭킹", "📊 노선별 분석"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # 요일 타입 준비 (여러 탭에서 사용)
    day_types_raw = df['day_type'].unique().tolist()
    day_types_order = ["평일", "토요일", "일요일"]
    day_types = [d for d in day_types_order if d in day_types_raw]
    
    # 방향 설명 함수 (여러 탭에서 사용)
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
    
    # ============================================
    # 선택된 메뉴에 따라 콘텐츠 표시
    # ============================================
    
    if selected_tab == "📈 역 상세 분석":
        # 사이드바 - 필터
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 필터")
        
        # 필터 1: 요일구분
        selected_day = st.sidebar.selectbox(
            "요일구분",
            options=day_types,
            index=0,
            key="tab1_day"
        )
        
        # 필터 2: 호선
        lines = sorted(df['line'].unique().tolist())
        selected_line = st.sidebar.selectbox(
            "호선",
            options=lines,
            index=0,
            key="tab1_line"
        )
        
        # 필터 3: 출발역 (선택된 호선의 역만 표시)
        stations_in_line = df[df['line'] == selected_line]['station_name'].unique()
        stations_sorted = sorted(stations_in_line)
        
        # 히트맵이나 랭킹에서 선택된 역이 있으면 해당 역을 기본값으로 설정
        default_station_idx = 0
        if st.session_state['selected_station_from_heatmap'] and \
           st.session_state['selected_station_from_heatmap'] in stations_sorted:
            default_station_idx = stations_sorted.index(st.session_state['selected_station_from_heatmap'])
            # 한 번 사용 후 초기화
            st.session_state['selected_station_from_heatmap'] = None
        
        selected_station = st.sidebar.selectbox(
            "출발역",
            options=stations_sorted,
            index=default_station_idx,
            key=f"tab1_station_{selected_line}"
        )
        
        # 필터 4: 상하구분 (선택된 호선의 방향만 표시)
        directions_in_line = sorted(df[df['line'] == selected_line]['direction'].unique())
        selected_direction = st.sidebar.selectbox(
            "상하구분",
            options=directions_in_line,
            index=0,
            key=f"tab1_direction_{selected_line}"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info("💡 필터를 변경하면 차트가 자동으로 업데이트됩니다.")
        
        direction_display = get_direction_description(selected_line, selected_direction)
        
        # KPI 요약 카드 섹션
        st.markdown("## 📈 역 상세 분석")
        st.markdown("### 핵심 지표 요약")
        st.markdown(f"**{selected_line} {direction_display}** ({selected_day})")
        
        # KPI 계산
        kpi_data = calculate_kpi(df, selected_day, selected_line, selected_direction)
        
        if kpi_data:
            # 주요 KPI 4개
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "총 역 수",
                    f"{kpi_data['total_stations']}개",
                    help="선택한 호선과 방향의 총 역 수"
                )
            
            with col2:
                st.metric(
                    "전체 평균 혼잡도",
                    f"{kpi_data['avg_crowding']:.1f}",
                    help="모든 역과 시간대의 평균 혼잡도"
                )
            
            with col3:
                st.metric(
                    "최고 혼잡역",
                    kpi_data['max_station'],
                    f"평균 {kpi_data['max_crowding']:.1f}",
                    help="가장 혼잡한 역 (하루 평균)"
                )
            
            with col4:
                st.metric(
                    "피크 시간대",
                    kpi_data['peak_time'],
                    help="전체적으로 가장 혼잡한 시간대"
                )
            
            # 출퇴근 시간대 평균
            col_morning, col_evening = st.columns(2)
            
            with col_morning:
                st.metric(
                    "출근 시간대 평균",
                    f"{kpi_data['morning_avg']:.1f}",
                    help="07:30-09:30 평균 혼잡도"
                )
            
            with col_evening:
                st.metric(
                    "퇴근 시간대 평균",
                    f"{kpi_data['evening_avg']:.1f}",
                    help="17:30-19:30 평균 혼잡도"
                )
        else:
            st.warning("⚠️ KPI를 계산할 수 있는 데이터가 없습니다.")
        
        st.markdown("---")
        
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
            height=350,
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
    
    elif selected_tab == "🏆 전체 혼잡도 랭킹":
        st.markdown("## 🏆 전체 혼잡도 랭킹")
        st.caption("모든 노선에서 가장 혼잡한 역을 보여줍니다")
        
        # 요일 선택 (독립적인 필터)
        ranking_day = st.selectbox(
            "요일 선택",
            options=day_types,
            index=0,
            key="ranking_day_select"
        )
        
        # 평일/주말 구분
        if ranking_day == "평일":
            # 평일: 출근/퇴근 시간대 선택
            col_toggle, col_info = st.columns([1, 3])
            
            with col_toggle:
                rush_hour_option = st.radio(
                    "시간대 선택",
                    options=["출근 (07:30-09:30)", "퇴근 (17:30-19:30)"],
                    index=0,
                    horizontal=True
                )
                
                rush_type = "morning" if "출근" in rush_hour_option else "evening"
            
            with col_info:
                st.info(f"💡 {rush_hour_option} 시간대에서 가장 혼잡한 역을 표시합니다.")
        else:
            # 주말: 전체 시간대
            rush_type = "all_day"
            st.info(f"💡 전체 시간대 ({ranking_day})의 평균 혼잡도를 기준으로 랭킹을 표시합니다.")
        
        # Top N 슬라이더
        top_n = st.slider(
            "표시할 역 수",
            min_value=5,
            max_value=20,
            value=10,
            step=5,
            key="top_n_slider"
        )
        
        # 랭킹 계산
        try:
            ranking_df = calculate_rush_hour_ranking(df, ranking_day, rush_type, top_n=top_n)
            
            if ranking_df.empty:
                st.warning("⚠️ 랭킹 데이터가 없습니다.")
            else:
                # 방향 설명 추가
                ranking_df['direction_display'] = ranking_df.apply(
                    lambda row: get_direction_description(row['line'], row['direction']),
                    axis=1
                )
                
                # 랭킹 테이블
                st.markdown("### 📋 혼잡도 랭킹 테이블")
                
                display_df = ranking_df.copy()
                display_df['avg_crowding'] = display_df['avg_crowding'].round(1)
                
                st.dataframe(
                    display_df[['rank', 'station_name', 'line', 'direction_display', 
                                'avg_crowding', 'peak_time']].rename(columns={
                        'rank': '순위',
                        'station_name': '역명',
                        'line': '호선',
                        'direction_display': '방향',
                        'avg_crowding': '평균 혼잡도',
                        'peak_time': '피크 시간'
                    }),
                    hide_index=True,
                    use_container_width=True,
                    height=min(300, 35 * len(ranking_df) + 40)
                )
                
                # 막대 차트
                st.markdown("### 📊 혼잡도 막대 차트")
                
                # 라벨 생성
                chart_df = ranking_df.copy()
                chart_df['label'] = chart_df['station_name'] + '\n(' + chart_df['line'] + ')'
                
                fig_bar = px.bar(
                    chart_df,
                    x='label',
                    y='avg_crowding',
                    color='avg_crowding',
                    color_continuous_scale='Reds',
                    labels={'label': '역', 'avg_crowding': '평균 혼잡도'},
                    title=f"혼잡도 Top {top_n} ({rush_hour_option}, {ranking_day})",
                    text='avg_crowding'
                )
                
                # 스타일 설정
                fig_bar.update_traces(
                    texttemplate='%{text:.1f}',
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>평균 혼잡도: %{y:.1f}<extra></extra>'
                )
                
                fig_bar.update_layout(
                    height=350,
                    xaxis_title="",
                    yaxis_title="평균 혼잡도",
                    xaxis=dict(tickangle=-45),
                    showlegend=False
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # 각 역별 상세 차트 (Expander)
                st.markdown("---")
                st.markdown("### 📈 역별 상세 혼잡도 차트")
                st.caption("역을 펼쳐서 시간대별 혼잡도 추이를 확인하세요")
                
                for idx, row in ranking_df.iterrows():
                    with st.expander(f"{row['rank']}위. {row['station_name']} ({row['line']} {row['direction_display']}) - 평균 {row['avg_crowding']:.1f}"):
                        # 해당 역의 시간대별 데이터 가져오기
                        station_detail_df = df[
                            (df['day_type'] == ranking_day) &
                            (df['line'] == row['line']) &
                            (df['station_name'] == row['station_name']) &
                            (df['direction'] == row['direction'])
                        ].sort_values('time_order')
                        
                        if not station_detail_df.empty:
                            # 라인차트 생성
                            fig_station = px.line(
                                station_detail_df,
                                x='time_label',
                                y='crowding',
                                markers=True,
                                title=f"{row['station_name']}역 시간대별 혼잡도 ({ranking_day})",
                                labels={'time_label': '시간대', 'crowding': '혼잡도'}
                            )
                            
                            # 차트 스타일
                            fig_station.update_traces(
                                line_color='#1f77b4',
                                marker=dict(size=6),
                                hovertemplate='<b>시간대</b>: %{x}<br><b>혼잡도</b>: %{y:.1f}<extra></extra>'
                            )
                            
                            # 출퇴근 시간대 강조 (평일인 경우만)
                            if ranking_day == "평일":
                                fig_station.add_vrect(
                                    x0="07:30", x1="09:30",
                                    fillcolor="rgba(0, 100, 255, 0.1)",
                                    layer="below",
                                    line_width=0,
                                    annotation_text="출근",
                                    annotation_position="top left",
                                    annotation=dict(font_size=10, font_color="blue")
                                )
                                fig_station.add_vrect(
                                    x0="17:30", x1="19:30",
                                    fillcolor="rgba(255, 100, 0, 0.1)",
                                    layer="below",
                                    line_width=0,
                                    annotation_text="퇴근",
                                    annotation_position="top left",
                                    annotation=dict(font_size=10, font_color="red")
                                )
                            
                            fig_station.update_layout(
                                height=300,
                                xaxis_title="시간대",
                                yaxis_title="혼잡도",
                                hovermode='x unified',
                                xaxis=dict(tickangle=-45, tickmode='linear'),
                                yaxis=dict(rangemode='tozero')
                            )
                            
                            st.plotly_chart(fig_station, use_container_width=True)
                        else:
                            st.warning("⚠️ 데이터가 없습니다.")
        
        except Exception as e:
            st.error(f"❌ 랭킹 생성 중 오류 발생: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    else:  # 노선별 분석
        st.markdown("## 📊 노선별 혼잡도 분석")
        st.caption("특정 노선의 혼잡도를 심층 분석합니다")
        
        # 노선별 필터 (사이드바와 독립적)
        col_line, col_day, col_dir = st.columns([1, 1, 1])
        
        with col_line:
            lines_for_analysis = sorted(df['line'].unique().tolist())
            analysis_line = st.selectbox(
                "분석할 호선",
                options=lines_for_analysis,
                index=0,
                key="analysis_line_select"
            )
        
        with col_day:
            analysis_day = st.selectbox(
                "요일 선택",
                options=day_types,
                index=0,
                key="analysis_day_select"
            )
        
        with col_dir:
            analysis_directions = sorted(df[df['line'] == analysis_line]['direction'].unique())
            analysis_direction = st.selectbox(
                "방향 선택",
                options=analysis_directions,
                index=0,
                key=f"analysis_dir_{analysis_line}"
            )
        
        # 방향 설명 추가
        analysis_direction_display = get_direction_description(analysis_line, analysis_direction)
        
        # 히트맵 (노선별 분석용)
        st.markdown("### 🔥 역×시간대 혼잡도 히트맵")
        st.markdown(f"**{analysis_line} {analysis_direction_display}** 의 모든 역에 대한 시간대별 혼잡도를 한눈에 확인하세요.")
        
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
                index=0,
                key="analysis_sort_option"
            )
            sort_by = sort_options[sort_label]
        
        with col_info:
            pass  # 메시지 제거됨
        
        # 히트맵 데이터 준비
        try:
            heatmap_df, station_order = prepare_heatmap_data(
                df, analysis_day, analysis_line, analysis_direction, sort_by
            )
            
            # 역이 없는 경우 처리
            if heatmap_df.empty or len(station_order) == 0:
                st.warning("⚠️ 히트맵을 표시할 데이터가 없습니다.")
            else:
                # 색상 범위 계산
                vmin, vmax = get_color_scale_range(df, analysis_line)
                
                # 히트맵 생성
                fig_heatmap = px.imshow(
                    heatmap_df,
                    labels=dict(x="시간대", y="역명", color="혼잡도"),
                    x=heatmap_df.columns,
                    y=heatmap_df.index,
                    color_continuous_scale="RdYlGn_r",  # 빨강-노랑-초록 역순
                    aspect="auto",
                    title=f"역×시간대 혼잡도 히트맵 ({analysis_line}, {analysis_direction_display}) - {analysis_day}",
                    zmin=vmin,
                    zmax=vmax
                )
                
                # 스타일 설정
                fig_heatmap.update_traces(
                    hovertemplate='<b>역</b>: %{y}<br><b>시간대</b>: %{x}<br><b>혼잡도</b>: %{z:.1f}<extra></extra>'
                )
                
                # 높이를 역 수에 비례하여 조정 (최소 400px, 역당 약 40px)
                heatmap_height = max(400, len(station_order) * 40)
                
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
                st.markdown("#### 🔍 역 선택하여 상세 보기")
                
                selected_station_for_chart = st.selectbox(
                    "역을 선택하면 해당 역의 시간대별 혼잡도 추이를 확인할 수 있습니다",
                    options=["선택하세요..."] + station_order,
                    key="analysis_station_selector"
                )
                
                # 역이 선택되면 해당 역의 라인차트 표시
                if selected_station_for_chart and selected_station_for_chart != "선택하세요...":
                    st.markdown(f"### 📈 {selected_station_for_chart}역 시간대별 혼잡도")
                    
                    # 선택한 역의 데이터 필터링
                    station_df = df[
                        (df['day_type'] == analysis_day) &
                        (df['line'] == analysis_line) &
                        (df['station_name'] == selected_station_for_chart) &
                        (df['direction'] == analysis_direction)
                    ].sort_values('time_order')
                    
                    if not station_df.empty:
                        # 라인차트 생성
                        fig_station = px.line(
                            station_df,
                            x='time_label',
                            y='crowding',
                            markers=True,
                            title=f"{selected_station_for_chart}역 ({analysis_line} {analysis_direction_display}) - {analysis_day}"
                        )
                        
                        # 출퇴근 시간대 강조
                        fig_station.add_vrect(
                            x0="07:30", x1="09:30",
                            fillcolor="yellow", opacity=0.2,
                            layer="below", line_width=0,
                            annotation_text="출근시간", annotation_position="top left"
                        )
                        fig_station.add_vrect(
                            x0="17:30", x1="19:30",
                            fillcolor="orange", opacity=0.2,
                            layer="below", line_width=0,
                            annotation_text="퇴근시간", annotation_position="top left"
                        )
                        
                        fig_station.update_layout(
                            height=350,
                            xaxis_title="시간대",
                            yaxis_title="혼잡도",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_station, use_container_width=True)
                    else:
                        st.warning("⚠️ 선택한 역의 데이터가 없습니다.")
        
        except Exception as e:
            st.error(f"❌ 히트맵 생성 중 오류 발생: {e}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
