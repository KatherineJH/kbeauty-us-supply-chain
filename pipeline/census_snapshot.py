"""Census API 응답 원문을 그대로 보관하고, 보관본이 있으면 다시 호출하지 않는다.

보관 위치: data/raw/census/<이름>.json   (응답 바이트를 수정하지 않고 저장. 204 빈 응답은 0바이트 파일)
manifest : data/raw/census/manifest.csv   (이름, API 키를 뺀 요청 URL·파라미터, HTTP 상태, 수집 시각, 바이트 수, SHA-256)

offline=True 이면 보관본만 읽고, 없으면 실패한다 (외부 요청 0회).
"""

import csv
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "census"
MANIFEST = DIR / "manifest.csv"
COLUMNS = ["name", "request_url_without_key", "params_json", "http_status", "collected_at", "bytes", "sha256"]


def _load_manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST, newline="", encoding="utf-8-sig") as f:
        return {r["name"]: r for r in csv.DictReader(f)}


def _save_manifest(m: dict):
    with open(MANIFEST, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(m[k] for k in sorted(m))


def get(name: str, base_url: str, params: dict, key: str | None, offline: bool) -> bytes:
    """보관본이 있으면 그 바이트를, 없으면 호출해서 저장한 바이트를 돌려준다."""
    path = DIR / f"{name}.json"
    if path.exists():
        return path.read_bytes()
    if offline:
        sys.exit(f"오프라인 모드: Census 원본 없음 {path.name}")
    DIR.mkdir(parents=True, exist_ok=True)
    url_wo_key = f"{base_url}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(f"{url_wo_key}&{urllib.parse.urlencode({'key': key})}", timeout=180) as resp:
            final_url, status, body = resp.geturl(), resp.status, resp.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"[{name}] API 오류 {e.code}")
    if "invalid_key" in final_url or "missing_key" in final_url:
        sys.exit("API 키가 유효하지 않거나 활성화되지 않았습니다.")
    path.write_bytes(body)
    m = _load_manifest()
    m[name] = {"name": name, "request_url_without_key": url_wo_key,
               "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True), "http_status": status,
               "collected_at": datetime.now().isoformat(timespec="seconds"), "bytes": len(body),
               "sha256": hashlib.sha256(body).hexdigest()}
    _save_manifest(m)
    return body


def parse_rows(body: bytes) -> list[dict]:
    """Census JSON 배열 -> dict 목록. 조건으로 넘긴 변수가 끝에 한 번 더 붙으므로 첫 위치만 쓴다."""
    text = body.decode("utf-8")
    if not text.strip():
        return []
    header, *rows = json.loads(text)
    idx = {}
    for i, n in enumerate(header):
        idx.setdefault(n, i)
    return [{n: row[i] for n, i in idx.items()} for row in rows]
