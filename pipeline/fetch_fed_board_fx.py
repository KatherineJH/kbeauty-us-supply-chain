"""Federal Reserve Board G.5 월평균 환율을 Board 통계 릴리스 페이지의 XML에서 받는다(API 키 불필요).

출처: Board of Governors of the Federal Reserve System, H.10/G.5 Foreign Exchange Rates
  릴리스 페이지 https://www.federalreserve.gov/releases/h10/current/default.htm 의 "XML" 링크
  (FRB_h10_xml.zip). Board는 Data Download Program(DDP)을 단계적으로 폐지하면서
  "Historical data will remain available for download as XML files on statistical release pages"라고
  밝혔다(https://www.federalreserve.gov/data/data-download-fred-information.htm, 2026-07-16).
  이용조건: https://www.federalreserve.gov/disclaimer.htm ("public domain … Please cite to the Board").

원문: zip 파일을 그대로 data/raw/federal_reserve_board/FRB_h10_xml.zip 에 보관하고 manifest.csv 에
      URL·수집 시각·크기·SHA-256을, series.csv 에 실제로 쓰는 7개 계열의 ID·단위·방향을 기록한다.
      zip에는 G.5/H.10 전체 계열이 들어 있고, 이 저장소는 아래 SERIES의 월간 계열 7개만 쓴다.
결과: data/extracted/fx_monthly.csv  (month, currency, lcu_per_usd) — 모든 통화를 '1달러당 현지 통화'로 맞춘다.
      기간은 설정값(--start, --end)으로 고정한다. 기본 --end 는 kbeauty.paths.FX_DATA_END 이므로
      나중에 공표된 달이 보관본에 있어도 자동으로 들어오지 않는다.

단위 주의: 파운드 계열(RXI$US_N.M.UK)의 XML 속성 UNIT_MULT 는 "0.00999999978"로 적혀 있지만 값 자체는
  파운드당 달러(예: 2024-01 1.2701)이며 FRED EXUSUK·Board 공표표와 일치한다. 메타데이터 표기 오류로 보고
  UNIT_MULT 는 적용하지 않는다(tests/test_pipeline.py 가 이 동작을 고정한다).

사용법
  python pipeline/fetch_fed_board_fx.py            # 보관본이 없으면 받는다
  python pipeline/fetch_fed_board_fx.py --offline  # 보관본만 사용
"""

import argparse
import csv
import hashlib
import io
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kbeauty.paths import FX_DATA_END, FX_DATA_START  # noqa: E402

RAW = REPO / "data" / "raw" / "federal_reserve_board"
ZIP_NAME = "FRB_h10_xml.zip"
XML_NAME = "H10_data.xml"
MANIFEST = RAW / "manifest.csv"
SERIES_FILE = RAW / "series.csv"
URL = "https://www.federalreserve.gov/releases/h10/data/FRB_h10_xml.zip"

# Board series name -> (currency, quoted as USD per unit of currency?, FRED series used before 2026-11-06)
SERIES = {
    "RXI_N.M.KO": ("KRW", False, "EXKOUS"),
    "RXI$US_N.M.EU": ("EUR", True, "EXUSEU"),   # U.S. dollars per euro
    "RXI_N.M.CA": ("CAD", False, "EXCAUS"),
    "RXI_N.M.CH": ("CNY", False, "EXCHUS"),
    "RXI_N.M.MX": ("MXN", False, "EXMXUS"),
    "RXI$US_N.M.UK": ("GBP", True, "EXUSUK"),   # U.S. dollars per pound (UNIT_MULT not applied, see above)
    "RXI_N.M.JA": ("JPY", False, "EXJPUS"),
}


def get(offline: bool) -> bytes:
    path = RAW / ZIP_NAME
    if path.exists():
        return path.read_bytes()
    if offline:
        sys.exit(f"오프라인 모드: 보관본 없음 {path.relative_to(REPO)}")
    RAW.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (kbeauty-us-supply-chain data collector)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
    if body[:4] != b"PK\x03\x04":
        sys.exit(f"zip이 아닌 응답: {URL} ({len(body)} bytes)")
    path.write_bytes(body)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "url", "collected_at", "bytes", "sha256"], lineterminator="\n")
        w.writeheader()
        w.writerow({"file": ZIP_NAME, "url": URL, "collected_at": datetime.now().isoformat(timespec="seconds"),
                    "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
    return body


def parse(body: bytes) -> dict[str, dict]:
    """The 7 monthly series from the release XML: attributes as published and {YYYY-MM: value string}."""
    xml = zipfile.ZipFile(io.BytesIO(body)).read(XML_NAME)
    out = {}
    for _, el in ET.iterparse(io.BytesIO(xml), events=("end",)):
        if el.tag.endswith("}Series") and el.get("SERIES_NAME") in SERIES:
            obs = {o.get("TIME_PERIOD")[:7]: o.get("OBS_VALUE") for o in el if o.tag.endswith("}Obs")}
            out[el.get("SERIES_NAME")] = {"attrs": dict(el.attrib), "obs": obs}
            el.clear()
    missing = set(SERIES) - set(out)
    if missing:
        sys.exit(f"XML에 없는 계열: {sorted(missing)}")
    return out


def lcu_per_usd(series: str, value: str) -> float:
    """Local currency units per U.S. dollar. UNIT_MULT is deliberately not applied (see module docstring)."""
    v = float(value)
    return round(1 / v if SERIES[series][1] else v, 6)


def write_series_table(parsed: dict[str, dict]) -> None:
    with open(SERIES_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["series_name", "currency", "freq", "unit", "unit_mult_published",
                                          "unit_mult_applied", "direction", "conversion_to_lcu_per_usd",
                                          "fred_series_before_2026_11_06"], lineterminator="\n")
        w.writeheader()
        for s, (ccy, usd_per_unit, fred) in SERIES.items():
            a = parsed[s]["attrs"]
            w.writerow({"series_name": s, "currency": ccy, "freq": a.get("FREQ"), "unit": a.get("UNIT"),
                        "unit_mult_published": a.get("UNIT_MULT"), "unit_mult_applied": "1",
                        "direction": "USD per " + ccy if usd_per_unit else ccy + " per USD",
                        "conversion_to_lcu_per_usd": "1/value" if usd_per_unit else "value",
                        "fred_series_before_2026_11_06": fred})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=FX_DATA_START)
    parser.add_argument("--end", default=FX_DATA_END)
    parser.add_argument("--out", default=str(REPO / "data" / "extracted"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    parsed = parse(get(args.offline))
    if not args.offline or not SERIES_FILE.exists():
        write_series_table(parsed)
    out_rows = []
    for series, (ccy, _, _) in SERIES.items():
        for month, value in parsed[series]["obs"].items():
            if not args.start <= month <= args.end or value in ("", "ND", "NA"):
                continue
            out_rows.append({"month": month, "currency": ccy, "lcu_per_usd": lcu_per_usd(series, value)})
    out_rows.sort(key=lambda x: (x["month"], x["currency"]))
    out = Path(args.out) / "fx_monthly.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["month", "currency", "lcu_per_usd"], lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    print(f"저장: {out} ({len(out_rows)}행, 통화 {len(SERIES)}개, {args.start}~{args.end})")


if __name__ == "__main__":
    main()
