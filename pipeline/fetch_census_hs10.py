"""미국 Census 수입 통계 API에서 한국발 K-뷰티 품목을 HS 10단위 세목별로 받아 CSV로 저장한다.

출력 (data/extracted/ 폴더)
  hs10_monthly.csv           세목 x 월 (미국 전체 합계)
  hs10_district_monthly.csv  세목 x 세관 구역 x 월 (금액 0인 행 제외)

원본: API 응답 원문은 data/raw/census/ 에 그대로 보관하고 manifest.csv 에 기록한다 (census_snapshot.py).
      보관본이 있으면 다시 호출하지 않는다. --offline 이면 보관본만 사용한다.

사용법
  python fetch_census_hs10.py                          # 2024-01부터 최신 공개 월까지, 기본 품목
  python fetch_census_hs10.py --start 2024-01 --end 2025-12
  python fetch_census_hs10.py --hs 3304 --hs 3305      # 품목(HS 2~6단위 접두어) 지정

API 키: 환경변수 CENSUS_API_KEY 또는 같은 폴더의 .env 파일 (CENSUS_API_KEY=...)
"""

import argparse
import csv
import os
import sys
from datetime import date
from pathlib import Path

import census_snapshot

BASE_URL = "https://api.census.gov/data/timeseries/intltrade/imports/hs"
KOREA = "5800"

# 기본 조회 품목 (HS 4단위). 아래 접두어로 시작하는 HS 10단위 세목을 모두 받는다.
DEFAULT_HS = {
    "3303": "향수·화장수",
    "3304": "메이크업·기초화장품·선크림",
    "3305": "헤어 제품",
    "3307": "면도·탈취·입욕 제품 등",
}

FIELDS = [
    "I_COMMODITY", "I_COMMODITY_LDESC", "DISTRICT", "DIST_NAME",
    "GEN_VAL_MO",                 # 일반 수입 금액 (USD)
    "VES_VAL_MO", "VES_WGT_MO",   # 해상 금액 (USD), 해상 중량 (kg)
    "AIR_VAL_MO", "AIR_WGT_MO",   # 항공 금액 (USD), 항공 중량 (kg)
    "GEN_QY1_MO", "UNIT_QY1",     # 수량과 단위 (화장품은 대부분 KG)
]

MONTHLY_COLUMNS = [
    "month", "hs10", "hs10_desc", "value_usd",
    "vessel_value_usd", "air_value_usd", "vessel_weight_kg", "air_weight_kg",
    "quantity", "quantity_unit",
]
DISTRICT_COLUMNS = [
    "month", "hs10", "district_code", "district_name", "value_usd",
    "vessel_value_usd", "air_value_usd", "vessel_weight_kg", "air_weight_kg",
    "quantity", "quantity_unit",
]


def load_api_key() -> str:
    key = os.environ.get("CENSUS_API_KEY")
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "CENSUS_API_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        sys.exit("CENSUS_API_KEY가 없습니다. 환경변수나 .env 파일에 넣어 주세요.")
    return key


def fetch(hs_prefix: str, start: str, end: str, country: str, key, offline: bool = False) -> list[dict]:
    params = {"get": ",".join(FIELDS), "time": f"from {start} to {end}", "CTY_CODE": country,
              "COMM_LVL": "HS10", "I_COMMODITY": f"{hs_prefix}*"}
    body = census_snapshot.get(f"hs10_{hs_prefix}_{start}_{end}_{country}", BASE_URL, params, key, offline)
    return census_snapshot.parse_rows(body)


def to_num(value: str):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def latest_available_month(country: str, key, offline: bool = False) -> str:
    """최근 13개월을 조회해 데이터가 있는 마지막 월을 찾는다. 이 응답도 원본으로 보관하며,
    보관본이 있으면 그것을 쓴다 (실행 날짜에 따라 결과가 바뀌지 않도록)."""
    today = date.today()
    params = {"get": "GEN_VAL_MO", "time": f"from {today.year - 1}-{today.month:02d} to {today.year}-{today.month:02d}",
              "CTY_CODE": country, "COMM_LVL": "HS4", "I_COMMODITY": "3304"}
    rows = census_snapshot.parse_rows(census_snapshot.get(f"latest_month_probe_{country}", BASE_URL, params, key, offline))
    return max(r["time"] for r in rows)


def main():
    parser = argparse.ArgumentParser(description="한국발 K-뷰티 HS10 세목별 미국 수입 통계 수집")
    parser.add_argument("--start", default="2024-01", help="시작 월 YYYY-MM (기본 2024-01)")
    parser.add_argument("--end", help="종료 월 YYYY-MM (기본: 최신 공개 월)")
    parser.add_argument("--hs", action="append", help="HS 접두어 (여러 번 지정 가능)")
    parser.add_argument("--country", default=KOREA, help="Census 국가 코드 (기본 5800 = 한국)")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "extracted"))
    parser.add_argument("--offline", action="store_true", help="보관된 원본만 사용 (외부 요청 0회)")
    args = parser.parse_args()

    key = None if args.offline else load_api_key()
    end = args.end or latest_available_month(args.country, key, args.offline)
    prefixes = args.hs or list(DEFAULT_HS)
    print(f"조회 기간 {args.start} ~ {end}, 품목 {', '.join(prefixes)}, 국가 {args.country}")

    monthly, district = [], []
    for prefix in prefixes:
        rows = fetch(prefix, args.start, end, args.country, key, args.offline)
        print(f"  {prefix}: {len(rows)}행")
        for r in rows:
            base = {
                "month": r["time"],
                "hs10": r["I_COMMODITY"],
                "value_usd": to_num(r["GEN_VAL_MO"]),
                "vessel_value_usd": to_num(r["VES_VAL_MO"]),
                "air_value_usd": to_num(r["AIR_VAL_MO"]),
                "vessel_weight_kg": to_num(r["VES_WGT_MO"]),
                "air_weight_kg": to_num(r["AIR_WGT_MO"]),
                "quantity": to_num(r["GEN_QY1_MO"]),
                "quantity_unit": r["UNIT_QY1"],
            }
            if r["DISTRICT"] == "-":  # "TOTAL FOR ALL DISTRICTS" 행 = 미국 전체
                monthly.append({**base, "hs10_desc": r["I_COMMODITY_LDESC"]})
            elif base["value_usd"]:
                district.append({**base, "district_code": r["DISTRICT"],
                                 "district_name": r["DIST_NAME"]})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    monthly.sort(key=lambda x: (x["month"], x["hs10"]))
    district.sort(key=lambda x: (x["month"], x["hs10"], -(x["value_usd"] or 0), x["district_code"]))
    for name, columns, data in [("hs10_monthly.csv", MONTHLY_COLUMNS, monthly),
                                ("hs10_district_monthly.csv", DISTRICT_COLUMNS, district)]:
        # utf-8-sig: 엑셀에서 바로 열어도 글자가 깨지지 않게 한다.
        with open(out / name, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)
        print(f"저장: {out / name} ({len(data)}행)")


if __name__ == "__main__":
    main()
