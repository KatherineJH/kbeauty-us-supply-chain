"""미국 Census 항구별 수입 통계 API(porths)에서 한국발 K-뷰티 품목을 HS 6단위 x 항구 x 월로 받아 CSV로 저장한다.

세관 구역(hs) 데이터와 달리 항구·공항 단위이고, 해상 중 컨테이너 금액과 항공 금액이 따로 나온다.
대신 세목은 HS 6단위까지만 제공된다.

출력 (output/)
  hs6_port_monthly.csv   세목(HS6) x 항구 x 월 (금액 0인 행 제외)

사용법
  python fetch_census_port.py                       # 2024-01부터 최신 공개 월까지
  python fetch_census_port.py --start 2025-01 --end 2025-12

원본: API 응답 원문은 data/raw/census/ 에 보관 (census_snapshot.py). --offline 이면 보관본만 사용.
API 키: 환경변수 CENSUS_API_KEY 또는 .env
"""

import argparse
import csv
from pathlib import Path

import census_snapshot
from fetch_census_hs10 import DEFAULT_HS, KOREA, latest_available_month, load_api_key, to_num

BASE_URL = "https://api.census.gov/data/timeseries/intltrade/imports/porths"
FIELDS = [
    "PORT", "PORT_NAME", "I_COMMODITY", "I_COMMODITY_LDESC",
    "GEN_VAL_MO",                 # 일반 수입 금액 (USD)
    "VES_VAL_MO", "VES_WGT_MO",   # 해상 금액, 해상 중량 (kg)
    "CNT_VAL_MO", "CNT_WGT_MO",   # 해상 중 컨테이너 금액, 중량 (kg)
    "AIR_VAL_MO", "AIR_WGT_MO",   # 항공 금액, 항공 중량 (kg)
]
COLUMNS = [
    "month", "port_code", "port_name", "hs6", "hs6_desc", "value_usd",
    "vessel_value_usd", "vessel_weight_kg", "container_value_usd", "container_weight_kg",
    "air_value_usd", "air_weight_kg",
]


def fetch(hs_prefix: str, start: str, end: str, country: str, key, offline: bool = False) -> list[dict]:
    params = {"get": ",".join(FIELDS), "time": f"from {start} to {end}", "CTY_CODE": country,
              "COMM_LVL": "HS6", "I_COMMODITY": f"{hs_prefix}*"}
    body = census_snapshot.get(f"porths_{hs_prefix}_{start}_{end}_{country}", BASE_URL, params, key, offline)
    return census_snapshot.parse_rows(body)


def main():
    parser = argparse.ArgumentParser(description="한국발 K-뷰티 HS6 항구별 미국 수입 통계 수집")
    parser.add_argument("--start", default="2024-01")
    parser.add_argument("--end", help="기본: 최신 공개 월")
    parser.add_argument("--hs", action="append")
    parser.add_argument("--country", default=KOREA)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "extracted"))
    parser.add_argument("--offline", action="store_true", help="보관된 원본만 사용 (외부 요청 0회)")
    args = parser.parse_args()

    key = None if args.offline else load_api_key()
    end = args.end or latest_available_month(args.country, key, args.offline)
    prefixes = args.hs or list(DEFAULT_HS)
    print(f"조회 기간 {args.start} ~ {end}, 품목 {', '.join(prefixes)}")

    rows = []
    for prefix in prefixes:
        data = fetch(prefix, args.start, end, args.country, key, args.offline)
        kept = 0
        for r in data:
            value = to_num(r["GEN_VAL_MO"])
            if r["PORT"] == "-" or not value:  # 전체 합계 행과 0인 행 제외
                continue
            kept += 1
            rows.append({
                "month": r["time"], "port_code": r["PORT"], "port_name": r["PORT_NAME"],
                "hs6": r["I_COMMODITY"], "hs6_desc": r["I_COMMODITY_LDESC"], "value_usd": value,
                "vessel_value_usd": to_num(r["VES_VAL_MO"]), "vessel_weight_kg": to_num(r["VES_WGT_MO"]),
                "container_value_usd": to_num(r["CNT_VAL_MO"]), "container_weight_kg": to_num(r["CNT_WGT_MO"]),
                "air_value_usd": to_num(r["AIR_VAL_MO"]), "air_weight_kg": to_num(r["AIR_WGT_MO"]),
            })
        print(f"  {prefix}: {kept}행")

    rows.sort(key=lambda x: (x["month"], x["port_code"], x["hs6"]))
    out = Path(args.out) / "hs6_port_monthly.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"저장: {out} ({len(rows)}행)")


if __name__ == "__main__":
    main()
