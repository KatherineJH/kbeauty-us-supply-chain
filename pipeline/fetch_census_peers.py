"""비교 국가의 미국 수입 통계를 HS10 세목별로 받는다 (사전 등록 docs/preregistration.md 2절의 규칙).

1. 2024년 연간 HS 3304 미국 수입액으로 국가 순위를 매기고, 집계 코드와 한국을 뺀 상위 8개국을 고른다.
   결과: warehouse/seeds/peer_countries.csv (한 번 만들면 바꾸지 않는다. 다시 만들려면 --reselect)
2. 고른 국가마다 한국과 같은 품목(HS 3303·3304·3305·3307)·기간의 HS10 × 월 미국 전체 합계를 받는다.
   결과: data/extracted/peer_hs10_monthly.csv

API 응답 원문은 data/raw/census/ 에 보관하고 manifest.csv 에 기록한다(census_snapshot.py). --offline 이면 보관본만 쓴다.

사용법
  python pipeline/fetch_census_peers.py --end 2026-08
  python pipeline/fetch_census_peers.py --end 2026-08 --offline
"""

import argparse
import csv
from pathlib import Path

import census_snapshot
from fetch_census_hs10 import BASE_URL, DEFAULT_HS, KOREA, fetch, load_api_key, to_num

REPO = Path(__file__).resolve().parents[1]
SEED = REPO / "warehouse" / "seeds" / "peer_countries.csv"
N_PEERS = 8
COLUMNS = ["month", "cty_code", "cty_name", "hs10", "value_usd", "vessel_value_usd", "air_value_usd",
           "vessel_weight_kg", "air_weight_kg"]


def is_country(code: str, name: str) -> bool:
    """Census groups regions and trade blocs under codes starting with 0 (e.g. 0003 EU) or containing X."""
    return code.isdigit() and not code.startswith("0") and "TOTAL" not in name.upper()


def select_peers(key, offline: bool) -> list[dict]:
    params = {"get": "CTY_CODE,CTY_NAME,GEN_VAL_YR", "time": "2024-12", "COMM_LVL": "HS4", "I_COMMODITY": "3304"}
    rows = census_snapshot.parse_rows(census_snapshot.get("country_rank_3304_2024", BASE_URL, params, key, offline))
    ranked = sorted((r for r in rows if is_country(r["CTY_CODE"], r["CTY_NAME"]) and r["CTY_CODE"] != KOREA),
                    key=lambda r: -(to_num(r["GEN_VAL_YR"]) or 0))
    return [{"rank": i + 1, "cty_code": r["CTY_CODE"], "cty_name": r["CTY_NAME"].title(),
             "hs3304_value_usd_2024": to_num(r["GEN_VAL_YR"])} for i, r in enumerate(ranked[:N_PEERS])]


def main():
    parser = argparse.ArgumentParser(description="비교 국가 HS10 미국 수입 통계 수집")
    parser.add_argument("--start", default="2024-01")
    parser.add_argument("--end", required=True, help="종료 월 YYYY-MM (한국 데이터와 같은 월)")
    parser.add_argument("--out", default=str(REPO / "data" / "extracted"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--reselect", action="store_true", help="peer_countries.csv를 다시 만든다(사전 등록 위반 주의)")
    args = parser.parse_args()
    key = None if args.offline else load_api_key()

    if args.reselect or not SEED.exists():
        peers = select_peers(key, args.offline)
        with open(SEED, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(peers[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(peers)
        print(f"선정: {SEED.relative_to(REPO)}")
    with open(SEED, encoding="utf-8") as f:
        peers = list(csv.DictReader(f))
    print("비교 국가:", ", ".join(f"{p['rank']}.{p['cty_name']}({p['cty_code']})" for p in peers))

    out_rows = []
    for p in peers:
        for prefix in DEFAULT_HS:
            for r in fetch(prefix, args.start, args.end, p["cty_code"], key, args.offline):
                if r["DISTRICT"] != "-":  # 미국 전체 합계 행만
                    continue
                out_rows.append({"month": r["time"], "cty_code": p["cty_code"], "cty_name": p["cty_name"],
                                 "hs10": r["I_COMMODITY"], "value_usd": to_num(r["GEN_VAL_MO"]),
                                 "vessel_value_usd": to_num(r["VES_VAL_MO"]), "air_value_usd": to_num(r["AIR_VAL_MO"]),
                                 "vessel_weight_kg": to_num(r["VES_WGT_MO"]), "air_weight_kg": to_num(r["AIR_WGT_MO"])})
        print(f"  {p['cty_name']}: 누적 {len(out_rows)}행")

    out_rows.sort(key=lambda x: (x["month"], x["cty_code"], x["hs10"]))
    out = Path(args.out) / "peer_hs10_monthly.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"저장: {out} ({len(out_rows)}행)")


if __name__ == "__main__":
    main()
