"""관세청 품목별 국가별 수출입실적(GW)에서 한국 → 미국 수출을 월별로 받아 화장품 세목만 남긴다.

API: https://apis.data.go.kr/1220000/nitemtrade/getNitemtradeList  (data.go.kr, 키: .env 의 DATA_GO_KR_KEY)
- 품목 필터(hsSgn)는 한국 HSK 10단위 '정확한 코드'만 받는다. 미국 HTS 코드와 다르므로 월별 전체 품목을 받아 걸러낸다.
- 응답 원문(XML, 월당 약 1.8MB)은 gzip으로 data/raw/kcs/US_<YYYYMM>.xml.gz 에 보관한다(압축을 풀면 원문 바이트 그대로).
  manifest.csv 에 키를 뺀 요청 URL, 수집 시각, 원문 크기와 SHA-256을 남긴다.

결과: data/extracted/kr_exports_us_hs10_monthly.csv
  month, hsk10, export_usd(FOB), export_kg, import_usd, import_kg   (HS 3303·3304·3305·3307)

사용법
  python pipeline/fetch_kcs_exports.py --start 2024-01 --end 2026-09
  python pipeline/fetch_kcs_exports.py --start 2024-01 --end 2026-09 --offline
"""

import argparse
import csv
import gzip
import hashlib
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "kcs"
MANIFEST = RAW / "manifest.csv"
URL = "https://apis.data.go.kr/1220000/nitemtrade/getNitemtradeList"
PREFIXES = ("3303", "3304", "3305", "3307")
COLUMNS = ["month", "hsk10", "export_usd", "export_kg", "import_usd", "import_kg"]


def load_key() -> str:
    env = REPO / ".env"
    for line in env.read_text(encoding="utf-8").splitlines() if env.exists() else []:
        name, _, value = line.partition("=")
        if name.strip() == "DATA_GO_KR_KEY":
            # data.go.kr issues an URL-encoded key; decode once so urlencode does not double-encode it.
            return urllib.parse.unquote(value.strip().strip('"').strip("'"))
    sys.exit("DATA_GO_KR_KEY가 없습니다. .env 파일에 넣어 주세요.")


def months(start: str, end: str):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y}{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def get(yyyymm: str, key, offline: bool) -> bytes:
    path = RAW / f"US_{yyyymm}.xml.gz"
    if path.exists():
        return gzip.decompress(path.read_bytes())
    if offline:
        sys.exit(f"오프라인 모드: 보관본 없음 {path.name}")
    params = {"strtYymm": yyyymm, "endYymm": yyyymm, "cntyCd": "US"}
    with urllib.request.urlopen(f"{URL}?{urllib.parse.urlencode({'serviceKey': key, **params})}", timeout=180) as r:
        body = r.read()
    code = re.search(rb"<resultCode>(\d+)</resultCode>", body)
    if not code or code.group(1) != b"00":
        sys.exit(f"[{yyyymm}] API 오류: {body[:300]!r}")
    RAW.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(body, mtime=0))  # mtime=0: same input -> same .gz bytes
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8"))) if MANIFEST.exists() else []
    rows = [x for x in rows if x["month"] != yyyymm] + [{
        "month": yyyymm, "request_url_without_key": f"{URL}?{urllib.parse.urlencode(params)}",
        "collected_at": datetime.now().isoformat(timespec="seconds"), "bytes": len(body),
        "sha256_uncompressed": hashlib.sha256(body).hexdigest()}]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: x["month"]))
    return body


def parse(body: bytes, yyyymm: str) -> list[dict]:
    text = body.decode("utf-8", "replace")  # Korean item names (statKor) are read by build_hsk_names.py
    out = []
    for item in re.findall(r"<item>(.*?)</item>", text, flags=re.S):
        f = dict(re.findall(r"<(\w+)>([^<]*)</\1>", item))
        hs = f.get("hsCd", "")
        if not hs.startswith(PREFIXES):
            continue
        if f.get("year") != f"{yyyymm[:4]}.{yyyymm[4:]}":
            continue
        out.append({"month": f"{yyyymm[:4]}-{yyyymm[4:]}", "hsk10": hs,
                    "export_usd": int(f["expDlr"]), "export_kg": int(f["expWgt"]),
                    "import_usd": int(f["impDlr"]), "import_kg": int(f["impWgt"])})
    return out


def main():
    parser = argparse.ArgumentParser(description="관세청 한국 → 미국 화장품 수출 실적 수집")
    parser.add_argument("--start", default="2024-01")
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", default=str(REPO / "data" / "extracted"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    key = None if args.offline else load_key()
    rows = []
    for ym in months(args.start, args.end):
        got = parse(get(ym, key, args.offline), ym)
        if not got:
            sys.exit(f"[{ym}] 화장품 행이 없습니다(아직 공개 전일 수 있음). --end 를 앞당기세요.")
        rows += got
        print(f"  {ym}: {len(got)}개 세목")
    rows.sort(key=lambda x: (x["month"], x["hsk10"]))
    out = Path(args.out) / "kr_exports_us_hs10_monthly.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"저장: {out} ({len(rows)}행)")


if __name__ == "__main__":
    main()
