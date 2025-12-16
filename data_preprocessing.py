"""
Phase 1: 데이터 준비 및 정제
원본 CSV를 롱 포맷(tidy data)로 변환하고 품질 검증
"""
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

TIME_COL_RE = re.compile(r"^\s*\d{1,2}시\d{2}분\s*$")

def read_raw_csv(path: str) -> pd.DataFrame:
    """원본 CSV 읽기 (인코딩 자동 감지)"""
    print(f"[*] 원본 CSV 읽기: {path}")
    last_err = None
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(path, encoding=enc)
            df.columns = [c.strip() for c in df.columns]
            print(f"[OK] 인코딩 성공: {enc}, 원본 행 수: {len(df):,}")
            return df
        except Exception as e:
            last_err = e
    raise last_err

def _timecol_to_hhmm(col: str) -> str:
    """시간 컬럼명(예: '5시30분') → 'HH:MM' 포맷으로 변환"""
    col = col.strip()
    m = re.match(r"^(\d{1,2})시(\d{2})분$", col)
    if not m:
        raise ValueError(f"시간 컬럼 파싱 실패: {col}")
    hh = int(m.group(1))
    mm = int(m.group(2))
    return f"{hh:02d}:{mm:02d}"

def to_tidy_long(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Wide 포맷 → Long(Tidy) 포맷 변환"""
    print("\n[*] 롱 포맷으로 변환 시작...")
    
    base_cols = ["요일구분", "호선", "역번호", "출발역", "상하구분"]
    missing = [c for c in base_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    time_cols = [c for c in df_raw.columns if TIME_COL_RE.match(c)]
    if not time_cols:
        raise ValueError("시간대 컬럼(예: 5시30분)이 탐지되지 않았습니다.")
    
    print(f"  - 시간대 컬럼 수: {len(time_cols)}")

    # 원본 시간 컬럼 순서를 정렬 기준으로 보존
    time_order_map = {c: i for i, c in enumerate(time_cols)}
    time_label_map = {c: _timecol_to_hhmm(c.strip()) for c in time_cols}

    df = df_raw.copy()

    # 혼잡도 숫자 클린업
    print("  - 혼잡도 값 파싱 중...")
    for c in time_cols:
        df[c] = (
            df[c]
            .astype(str)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    tidy = df.melt(
        id_vars=base_cols,
        value_vars=time_cols,
        var_name="time_col",
        value_name="crowding",
    )

    tidy["time_label"] = tidy["time_col"].map(time_label_map)
    tidy["time_order"] = tidy["time_col"].map(time_order_map)

    # 컬럼 리네이밍
    tidy = tidy.rename(
        columns={
            "요일구분": "day_type",
            "호선": "line",
            "역번호": "station_code",
            "출발역": "station_name",
            "상하구분": "direction",
        }
    )

    tidy["station_code"] = tidy["station_code"].astype(str).str.strip()

    tidy = tidy[
        ["day_type", "line", "station_code", "station_name", "direction", 
         "time_label", "time_order", "crowding"]
    ]
    
    # 정렬
    tidy = tidy.sort_values(["day_type", "line", "station_code", "direction", "time_order"]).reset_index(drop=True)
    
    print(f"[OK] 롱 포맷 변환 완료: {len(tidy):,} 행")
    return tidy

def quality_report(tidy: pd.DataFrame) -> dict:
    """데이터 품질 검증 리포트 생성"""
    print("\n[*] 데이터 품질 검증 중...")
    
    # 중복 체크
    key_cols = ["day_type", "line", "station_code", "station_name", "direction", "time_label"]
    dup_cnt = int(tidy.duplicated(key_cols).sum())

    # 결측
    crowding_nan = int(tidy["crowding"].isna().sum())
    crowding_nan_rate = float(tidy["crowding"].isna().mean())

    # 범위 체크
    neg_cnt = int((tidy["crowding"] < 0).sum(skipna=True))
    over200_cnt = int((tidy["crowding"] > 200).sum(skipna=True))

    # 전 시간대 0인 그룹
    grp_cols = ["day_type", "line", "station_code", "station_name", "direction"]
    g = tidy.groupby(grp_cols, dropna=False)["crowding"]
    all_zero_rows = int((g.apply(lambda s: s.notna().all() and (s == 0).all())).sum())
    
    # 분포
    day_types = tidy["day_type"].unique().tolist()
    lines = sorted(tidy["line"].unique().tolist())
    directions = sorted(tidy["direction"].unique().tolist())
    
    unique_stations = tidy[["station_code", "station_name"]].drop_duplicates()
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "rows_tidy": int(len(tidy)),
        "unique_stations": int(len(unique_stations)),
        "unique_station_direction_combinations": int(g.size().shape[0]),
        "day_types": day_types,
        "lines": lines,
        "directions": directions,
        "duplicate_key_rows": dup_cnt,
        "crowding_nan_count": crowding_nan,
        "crowding_nan_rate": round(crowding_nan_rate, 4),
        "negative_crowding_count": neg_cnt,
        "crowding_over_200_count": over200_cnt,
        "all_time_zero_station_direction_rows": all_zero_rows,
        "crowding_stats": {
            "min": float(tidy["crowding"].min()) if not tidy["crowding"].isna().all() else None,
            "max": float(tidy["crowding"].max()) if not tidy["crowding"].isna().all() else None,
            "mean": float(tidy["crowding"].mean()) if not tidy["crowding"].isna().all() else None,
            "median": float(tidy["crowding"].median()) if not tidy["crowding"].isna().all() else None,
        }
    }
    
    return report

def print_report(report: dict):
    """리포트 출력"""
    print("\n" + "="*60)
    print("[REPORT] Phase 1 Data Quality Report")
    print("="*60)
    print(f"Timestamp: {report['timestamp']}")
    print(f"\n[Basic Info]")
    print(f"  - Total rows: {report['rows_tidy']:,}")
    print(f"  - Unique stations: {report['unique_stations']:,}")
    print(f"  - Station x Direction combinations: {report['unique_station_direction_combinations']:,}")
    print(f"  - Day types: {', '.join(report['day_types'])}")
    print(f"  - Lines: {', '.join(report['lines'])}")
    print(f"  - Directions: {', '.join(report['directions'])}")
    
    print(f"\n[Quality Check]")
    status = "[OK] PASS" if report['duplicate_key_rows'] == 0 else "[FAIL]"
    print(f"  - Duplicate keys: {report['duplicate_key_rows']:,} {status}")
    
    status = "[OK] PASS" if report['negative_crowding_count'] == 0 else "[FAIL]"
    print(f"  - Negative crowding: {report['negative_crowding_count']:,} {status}")
    
    print(f"  - Missing (NaN): {report['crowding_nan_count']:,} ({report['crowding_nan_rate']*100:.2f}%)")
    print(f"  - Crowding >200 (warning): {report['crowding_over_200_count']:,}")
    print(f"  - All-time-zero groups: {report['all_time_zero_station_direction_rows']:,}")
    
    print(f"\n[Crowding Stats]")
    stats = report['crowding_stats']
    print(f"  - Min: {stats['min']:.1f}" if stats['min'] is not None else "  - Min: N/A")
    print(f"  - Max: {stats['max']:.1f}" if stats['max'] is not None else "  - Max: N/A")
    print(f"  - Mean: {stats['mean']:.1f}" if stats['mean'] is not None else "  - Mean: N/A")
    print(f"  - Median: {stats['median']:.1f}" if stats['median'] is not None else "  - Median: N/A")
    
    print("\n" + "="*60)
    
    # Phase 1 완료 조건 체크
    passed = (
        report['duplicate_key_rows'] == 0 and
        report['negative_crowding_count'] == 0
    )
    
    if passed:
        print("[OK] Phase 1 Completion Criteria: PASSED")
    else:
        print("[FAIL] Phase 1 Completion Criteria: FAILED - Please check data")
    print("="*60 + "\n")

def main():
    """Phase 1 메인 실행"""
    print("[START] Phase 1: Data Preparation and Cleaning\n")
    
    # 경로 설정
    base_dir = Path(__file__).parent
    raw_csv = base_dir / "서울교통공사_지하철혼잡도정보_20250930.csv"
    
    # 출력 폴더 생성
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    output_csv = data_dir / "subway_crowding_tidy.csv"
    output_parquet = data_dir / "subway_crowding_tidy.parquet"
    report_json = data_dir / "quality_report.json"
    
    # 1. 원본 읽기
    raw = read_raw_csv(str(raw_csv))
    
    # 2. 롱 포맷 변환
    tidy = to_tidy_long(raw)
    
    # 3. 품질 검증
    report = quality_report(tidy)
    print_report(report)
    
    # 4. 저장
    print("[*] Saving cleaned data...")
    tidy.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"  [OK] CSV saved: {output_csv}")
    
    tidy.to_parquet(output_parquet, index=False)
    print(f"  [OK] Parquet saved: {output_parquet}")
    
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Quality report saved: {report_json}")
    
    print(f"\n[COMPLETE] Phase 1 Finished!")
    print(f"   - Cleaned data: {len(tidy):,} rows")
    print(f"   - Output directory: {data_dir}")

if __name__ == "__main__":
    main()
