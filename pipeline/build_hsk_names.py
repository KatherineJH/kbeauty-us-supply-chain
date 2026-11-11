"""관세청 응답 원문(data/raw/kcs/*.xml.gz)에서 화장품 HSK 10단위 세목의 한글 품목명을 뽑는다.

API 응답은 UTF-8이고 품목명(statKor)도 정상이다. 세목마다 가장 최근 월의 이름을 쓴다(새 API 호출 없음).

결과: data/extracted/hsk10_names.csv  (hsk10, name_ko, last_seen_month)

사용법
  python pipeline/build_hsk_names.py
"""

import argparse
import csv
import gzip
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "kcs"
PREFIXES = ("3303", "3304", "3305", "3307")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "data" / "extracted"))
    parser.add_argument("--offline", action="store_true", help="항상 오프라인(원문만 읽음). run_all 호환용")
    args = parser.parse_args()
    names = {}
    for path in sorted(RAW.glob("US_*.xml.gz")):  # chronological, so later months overwrite earlier names
        ym = path.name[3:9]
        text = gzip.decompress(path.read_bytes()).decode("utf-8")
        for item in re.findall(r"<item>(.*?)</item>", text, flags=re.S):
            f = dict(re.findall(r"<(\w+)>([^<]*)</\1>", item))
            if f.get("hsCd", "").startswith(PREFIXES):
                names[f["hsCd"]] = (f.get("statKor", "").strip(), f"{ym[:4]}-{ym[4:]}")
    out = Path(args.out) / "hsk10_names.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["hsk10", "name_ko", "last_seen_month"])
        for code in sorted(names):
            w.writerow([code, *names[code]])
    print(f"저장: {out} ({len(names)}개 세목)")


if __name__ == "__main__":
    main()
