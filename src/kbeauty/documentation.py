"""Generated reference documents: the data dictionary and the forecasting model card.

Both are rebuilt by `report.generate_all()` from the warehouse, dbt's manifest and the result tables, so they
cannot describe a column or a number that no longer exists.
- reports/data_dictionary.csv: every warehouse column with its type, meaning, unit, dbt tests and upstream models.
  Column meanings come from warehouse/column_glossary.yml; table descriptions from dbt schema.yml.
- reports/model_card.md: what is forecast, how models are compared and selected, results and limits.
"""

import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from .paths import BACKTEST_FIRST_ORIGIN, CENSUS_DATA_END, DATA_START, DUCKDB_PATH, REPO, TABLES_DIR, WAREHOUSE_DIR

GLOSSARY = WAREHOUSE_DIR / "column_glossary.yml"
MANIFEST = WAREHOUSE_DIR / "target" / "manifest.json"
FROZEN_DIR = REPO / "reports" / "frozen"
SEED_DESCRIPTIONS = {
    "hs10_category": "HTS10 코드 → 분석 품목군(skincare, makeup, hair, nail, fragrance, other)",
    "state_region": "미국 주 → 하역 권역",
    "policy_events": "정책 사건과 첫 완전 적용 월(2025-08 상호관세 15%, 소액면세 중단)",
    "peer_countries": "사전 등록 규칙으로 고른 비교 8개국(2024년 미국 HS 3304 수입액 상위)",
    "country_currency": "Census 국가 코드 → 국가·통화, 처리군 표시",
}


def layer_of(name: str, resource_type: str) -> str:
    if resource_type == "seed":
        return "seed"
    return "staging" if name.startswith("stg_") else "mart"


def data_dictionary(db_path: Path = DUCKDB_PATH, manifest_path: Path = MANIFEST) -> pd.DataFrame:
    glossary = yaml.safe_load(GLOSSARY.read_text(encoding="utf-8"))
    cols, overrides = glossary["columns"], glossary.get("overrides", {})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nodes = {v["name"]: v for v in manifest["nodes"].values() if v["resource_type"] in ("model", "seed")}
    tests: dict[tuple[str, str | None], list[str]] = {}
    for v in manifest["nodes"].values():
        if v["resource_type"] != "test":
            continue
        meta = v.get("test_metadata")
        for dep in v["depends_on"]["nodes"]:
            table = dep.split(".")[-1]
            if meta:  # generic test on a column of the node it is attached to
                if v.get("attached_node") == dep:
                    tests.setdefault((table, v.get("column_name")), []).append(meta["name"])
            else:     # singular test: listed on every table it reads
                tests.setdefault((table, None), []).append(v["name"])
    with duckdb.connect(str(db_path), read_only=True) as con:
        info = con.execute("select table_name, column_name, data_type, ordinal_position from information_schema.columns "
                           "where table_schema = 'main' order by table_name, ordinal_position").fetchall()
    rows = []
    for table, column, dtype, _ in info:
        node = nodes.get(table)
        if node is None:
            continue
        entry = overrides.get(f"{table}.{column}") or cols.get(column) or {}
        rtype = node["resource_type"]
        rows.append({
            "layer": layer_of(table, rtype),
            "table": table,
            "table_description": node.get("description") or SEED_DESCRIPTIONS.get(table, ""),
            "column": column,
            "data_type": dtype,
            "description": entry.get("description", ""),
            "unit": entry.get("unit", ""),
            "column_tests": ", ".join(sorted(set(tests.get((table, column), [])))),
            "table_tests": ", ".join(sorted(set(tests.get((table, None), [])))),
            "upstream": ", ".join(d.split(".")[-1] for d in node.get("depends_on", {}).get("nodes", [])),
        })
    order = {"seed": 0, "staging": 1, "mart": 2}
    return (pd.DataFrame(rows).sort_values(["layer", "table"], key=lambda s: s.map(order) if s.name == "layer" else s,
                                           kind="stable").reset_index(drop=True))


def seeds(manifest_path: Path = MANIFEST) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return sorted(v["name"] for v in manifest["nodes"].values() if v["resource_type"] == "seed")


def lineage(manifest_path: Path = MANIFEST) -> pd.DataFrame:
    """Edges between dbt nodes (seed/staging -> mart), for the lineage diagram."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for v in manifest["nodes"].values():
        if v["resource_type"] != "model":
            continue
        for dep in v["depends_on"]["nodes"]:
            rows.append({"parent": dep.split(".")[-1], "child": v["name"]})
    return pd.DataFrame(rows).sort_values(["child", "parent"]).reset_index(drop=True)


def _w(lw: pd.DataFrame, model: str, method: str, h: int, level: str = "total") -> float:
    return float(lw[(lw.model == model) & (lw.method == method) & (lw.horizon == h) & (lw.level == level)].wape.iloc[0])


def _frozen_status() -> list[str]:
    out = []
    for f in sorted(FROZEN_DIR.glob("*_frozen_*.csv")):
        recorded = f.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        ok = hashlib.sha256(f.read_bytes()).hexdigest() == recorded
        scored = (FROZEN_DIR / f"score_{f.stem}.csv").exists()
        out.append(f"| `{f.name}` | {len(pd.read_csv(f))}행 | {'일치' if ok else '불일치'} | {'채점됨' if scored else '채점 대기(Census 미공개)'} |")
    return out


PLATFORM_DIR = REPO / "reports" / "platform"


def _platform_section() -> str:
    """Section 8: the canonical (pinned Linux) values next to the earlier observed runs, kept as diagnostic
    history: the author environment (Windows) and two unpinned Linux CI runs."""
    from .platform_rules import SENSITIVE, recommendations
    runs = [("linux", "Linux 고정 환경(공식)"), ("windows", "Windows 작성 환경(이력)"),
            ("linux_ci_run1", "Linux 고정 전 1회(이력)"), ("linux_ci_run2", "Linux 고정 전 2회(이력)")]
    names = [n for n in SENSITIVE if n.startswith("forecast_")]
    runs = [(k, label) for k, label in runs if all((PLATFORM_DIR / k / n).exists() for n in names)]
    if not runs or runs[0][0] != "linux":
        return "## 8. 실행 환경과 재현성 (모델 위험)\n\n- 공식 기준(`reports/platform/linux/`)이 아직 없다."
    t = {k: {n: pd.read_csv(PLATFORM_DIR / k / n) for n in names} for k, _ in runs}

    def w(k, model, method, h):
        return f"{_w(t[k]['forecast_hierarchy_wape.csv'], model, method, h):.3f}"

    def pv(k, name):
        return f"{float(t[k]['forecast_dm_tests.csv'].set_index('comparison').loc[name, 'p_value']):.2f}"

    spec = [
        ("ML 검증 선택, 합계 WAPE 1개월", lambda k: w(k, "ml_selected", "bottom_up", 1)),
        ("직전 값(기준선), 합계 WAPE 1개월", lambda k: w(k, "naive", "base", 1)),
        ("ML 검증 선택 vs 직전 값, DM p(1개월)", lambda k: pv(k, "ml_selected(BU) vs naive (h1)")),
        ("ML 검증 선택, 합계 WAPE 3개월", lambda k: w(k, "ml_selected", "bottom_up", 3)),
        ("계절 ETS 합, 합계 WAPE 3개월", lambda k: w(k, "ets_seasonal", "bottom_up", 3)),
        ("ML 검증 선택 vs 계절 ETS 합, DM p(3개월)", lambda k: pv(k, "ml_selected(BU) vs ets_seasonal(BU) (h3)")),
        ("계절 ETS(합산 전), 합계 WAPE 1개월", lambda k: w(k, "ets_seasonal", "base", 1)),
        ("MLP, 합계 WAPE 3개월", lambda k: w(k, "ml_mlp", "bottom_up", 3)),
    ]
    header = "| 지표 | " + " | ".join(label for _, label in runs) + " |"
    table = "\n".join([header, "|---|" + "---|" * len(runs)] +
                      [f"| {name} | " + " | ".join(f(k) for k, _ in runs) + " |" for name, f in spec])
    recs = [recommendations(t[k]) for k, _ in runs]
    same = "같다" if all(r == recs[0] for r in recs) else "**다르다**"
    r0 = recs[0]
    sel, nv = w("linux", "ml_selected", "bottom_up", 1), w("linux", "naive", "base", 1)
    p1 = pv("linux", "ml_selected(BU) vs naive (h1)")
    return f"""## 8. 실행 환경과 재현성 (모델 위험)

- 모든 학습은 단일 스레드·고정 시드이고, 같은 컴퓨터에서 다시 실행하면 결과 표가 바이트 단위로 같다.
- 그러나 예측 표 {len(names)}개는 실행 환경에 민감했다. 원인은 GitHub runner가 실행마다 다른 CPU를 받는 것이었다(`docs/experiments/linux_determinism.md`).
  - AVX-512가 있는 CPU에서는 numpy가 AVX-512 경로를, OpenBLAS가 SkylakeX 커널을 쓴다. 이때 마지막 비트가 달라진다.
  - 처음 갈라지는 곳은 통계 모델(감쇠 ETS)과 ridge다. ETS 시작값 탐색·L-BFGS-B, SARIMAX, MLP가 그 차이를 키운다.
- **공식 기준은 Linux 고정 환경이다.** 스레드 1, OpenBLAS Haswell 커널, numpy AVX-512 비활성화로 고정했다. 이 환경에서는 CPU 모델 6종(AMD EPYC 7763·9V74·9V45, Intel Xeon 8370C·8573C·6973P)에서 31개 표가 바이트 단위로 같았다.
  - 공식 결과표와 생성 기록(커밋, 패키지·BLAS 버전, 환경 변수, runner, 생성 시각, 해시)은 `reports/platform/linux/`에 있다.
  - CI는 같은 고정 환경에서 31개 표가 이 기준과 바이트 단위로 같은지 검사한다. 같은 job에서 예측·ML 단계를 두 번 더 실행해 결과가 달라지지 않는지도 검사한다.
- 나머지 열은 **진단 이력**이다. 고정 전에 관측한 값이며 보장 구간이 아니다. Windows 작성 환경 값(노트북의 숫자)도 공식 기준과 다르다. 통계 모델은 1e-4 수준에서 다르지만, ML 선택 결과는 크게 달라진다.

{table}

- **결론**: 공식 기준에서 ML 검증 선택의 1개월 앞 WAPE는 {sel}로 직전 값 {nv}보다 낮다. 그러나 차이를 통계적으로 입증하지 못했다(DM p {p1}, 기준 시점 8개로 검정력이 낮다). 관측한 모든 실행에서도 DM p > 0.1이었다. 그래서 **통계적으로 구분되는 개선을 입증하지 못해 운영 규칙(1개월 직전 값, 3개월 계절 ETS 합)을 유지한다**.
- **모델 위험**: ML 결과는 수치 환경(CPU 계열, BLAS 커널)에 민감하다. 환경 고정은 결과를 재현 가능하게 만들 뿐, 모델의 안정성을 보장하지 않는다. MLP fold의 약 11%는 반복 상한(2000회)에서 멈춘다.
- **운영 권고는 공식 기준과 진단 이력 모두에서 {same}**: 1개월 앞 {r0['1m_rule']}, 3개월 앞 {r0['3m_rule']}, ML 채택 보류.
"""


def _official(r: dict) -> dict | None:
    """The model card's forecast numbers come only from the official pinned-Linux reference. If it is missing,
    there are no official numbers; this run's (possibly other-environment) tables are never used instead."""
    ref = PLATFORM_DIR / "linux"
    names = [k for k in r if k.startswith("forecast_") and (TABLES_DIR / f"{k}.csv").exists()]  # saved tables only
    if not all((ref / f"{k}.csv").exists() for k in names):
        return None
    official = {k: pd.read_csv(ref / f"{k}.csv") for k in names}
    return {**r, **official}


def model_card(r: dict) -> str:
    official = _official(r)
    if official is None:
        return ("# 모델 카드 — K-뷰티 대미 수입 흐름 예측\n\n"
                "> 공식 기준(`reports/platform/linux/`)이 없어 예측 수치를 싣지 않는다. "
                "다른 실행 환경의 값으로 대체하지 않는다.\n")
    r = official
    source = ("예측 수치는 모두 공식 기준인 Linux 고정 환경의 결과표(`reports/platform/linux/`)에서 읽는다. "
              "작성 환경(Windows) 값은 8절의 진단 이력 열에만 있다")
    lw = r["forecast_hierarchy_wape"]
    dm = r["forecast_dm_tests"].set_index("comparison")
    imp = r["forecast_ml_importance"]
    sel = r["forecast_ml_selection"]
    cov = r["forecast_conformal"].query("level == 0.8").coverage
    picked = sel[sel.model == "ml_selected"]
    boost = [_w(lw, m, "bottom_up", 1) for m in ("ml_lightgbm", "ml_xgboost", "ml_gradient_boosting")]
    top = imp[imp.horizon == 1].sort_values("wape_rise_mean", ascending=False).head(3)

    def p(name):
        v = dm.loc[name, "p_value"]
        return f"p {v:.3f}" if v < 0.01 else f"p {v:.2f}"

    rows = [
        ("직전 값", "naive", "base"), ("작년 같은 달", "seasonal_naive", "base"),
        ("SARIMAX(환율·정책)", "sarimax_fx_policy", "base"), ("계절 ETS, 품목·운송수단별 합", "ets_seasonal", "bottom_up"),
        ("ML 검증 선택", "ml_selected", "bottom_up"), ("LightGBM (사후 비교)", "ml_lightgbm", "bottom_up"),
    ]
    table = "\n".join(f"| {name} | {_w(lw, m, k, 1):.3f} | {_w(lw, m, k, 3):.3f} | {_w(lw, m, k, 1, 'bottom'):.3f} |"
                      for name, m, k in rows)
    return f"""# 모델 카드 — K-뷰티 대미 수입 흐름 예측

> 이 문서는 `scripts/run_all.py`가 결과 표에서 자동으로 만든다(`src/kbeauty/documentation.py`). 숫자를 손으로 고치지 않는다. {source}.

## 1. 무엇을 예측하나

| 항목 | 내용 |
|---|---|
| 대상 | 미국 Census가 기록한 한국산 화장품(HS 3303·3304·3305·3307) **해상 + 항공 수입 금액**, 월 단위 |
| 계층 | 품목군 4개 × 운송수단 2개 = 하위 8개 시계열 → 품목군 4 · 운송수단 2 · 합계 1, 모두 15개 노드 |
| 기간 | {DATA_START} ~ {CENSUS_DATA_END}({len(pd.period_range(DATA_START, CENSUS_DATA_END, freq="M"))}개월). 예측 기준 시점 {BACKTEST_FIRST_ORIGIN[:7]} ~ {sel.origin.max()[:7]}, 1·2·3개월 앞 |
| 범위 | **공개 수입통계의 예측**이다. 기업 내부 판매·재고·발주 데이터를 쓰는 수요예측이 아니며, 발주 판단을 돕는 외부 시장 흐름 지표로만 쓴다 |

## 2. 후보 모델

- **통계 모델 6종**: 직전 값, 작년 같은 달, 작년 같은 달 × 성장률, ETS(감쇠 추세), ETS(계절), SARIMAX(1,1,0) + 정책 더미 + 원/달러. 노드마다 따로 맞추고 bottom-up·OLS로 정합한다.
- **머신러닝 9종**: Ridge, Elastic Net, kNN, Random Forest, Extra Trees, Gradient Boosting, LightGBM, XGBoost, MLP. 하위 8개 시계열을 한 모델로 학습하는 전역 모델, 예측 시점별 직접 예측.
  - 특징: 기준 시점의 최근 3개월 로그 값, 월, 시계열, 정책 후 여부, 원/달러(+ 선택 시 작년 같은 달)
  - 함께 조정: 모델별 하이퍼파라미터 × 목표(로그 수준·로그 변화) × 특징 묶음 = **76개 설정**

## 3. 평가와 선택

- **rolling origin**: 각 기준 시점에서 그 시점까지의 자료로만 학습한다. 미래 값을 바꿔도 예측이 변하지 않는지 테스트로 확인한다(`tests/test_forecast.py`, `tests/test_ml.py`).
- **ML 선택 규칙(시험 달을 보지 않는다)**: 기준 시점마다 이미 관측된 마지막 3개월로 76개 설정을 채점하고, 가장 좋은 설정을 `ml_selected`로 쓴다. 이 절차 자체의 표본 외 성적을 보고한다.
- **지표**: WAPE(낮을수록 좋음). 차이의 유의성은 Diebold-Mariano(HLN 소표본 보정), 구간은 leave-one-origin-out split-conformal.
- **재현성**: 모든 학습은 단일 스레드·고정 시드다. 같은 환경에서 다시 실행하면 같은 숫자가 나온다.

## 4. 결과 (WAPE)

| 모델 | 합계 1개월 앞 | 합계 3개월 앞 | 하위 8개 1개월 앞 |
|---|---|---|---|
{table}

- 1개월 앞: 직전 값 vs 작년 같은 달 {p('naive vs seasonal_naive (h1)')}, `ml_selected` vs 직전 값 {p('ml_selected(BU) vs naive (h1)')}
- 3개월 앞: 계절 ETS 합 vs 작년 같은 달 {p('ets_seasonal(BU) vs seasonal_naive (h3)')}, `ml_selected` vs 계절 ETS 합 {p('ml_selected(BU) vs ets_seasonal(BU) (h3)')}
- 부스팅 3종의 1개월 앞 합계 WAPE는 {min(boost):.3f}~{max(boost):.3f}로 직전 값보다 낮지만, **시험 구간을 보고 고른 사후 비교**다(LightGBM vs 직전 값 {p('[post hoc] ml_lightgbm(BU) vs naive (h1)')}).
- `ml_selected`가 고른 설정은 기준 시점마다 바뀌었다(1개월 앞 기준 시점 {picked[picked.horizon == 1].origin.nunique()}개에서 서로 다른 설정 {picked[picked.horizon == 1].chosen.nunique()}가지).
- 표본 외 순열 중요도(1개월 앞, WAPE 상승): {", ".join(f"{r.feature} +{r.wape_rise_mean:.2f}" for r in top.itertuples())}. 원/달러의 기여는 0에 가깝다.
- 80% 예측 구간의 실제 포함률은 {cov.min() * 100:.0f}~{cov.max() * 100:.0f}%로 명목보다 낮다.

## 5. 운영 규칙과 ML을 채택하지 않은 이유

| 의사결정 기간 | 사용할 예측 | 근거 |
|---|---|---|
| 1개월 앞 | 직전 값 | 합계 WAPE {_w(lw, 'naive', 'base', 1):.3f}, 비교한 모델 중 가장 낮음(순위는 통계적으로 확정되지 않음) |
| 3개월 앞 | 계절 ETS, 품목·운송수단별 합 | 합계 WAPE {_w(lw, 'ets_seasonal', 'bottom_up', 3):.3f}, '작년 같은 달' 대비 우위 {p('ets_seasonal(BU) vs seasonal_naive (h3)')} |
| 머신러닝 | **보류** | 검증 성적으로 고른 모델은 1개월 앞 직전 값과 통계적으로 구분되는 개선을 입증하지 못했고(공식 기준 Linux 고정 환경 값과 진단 이력은 8절), 3개월 앞 차이도 구분되지 않는다. 부스팅의 우위는 아래 고정 예측 채점으로 판정한다 |

복잡한 모델을 썼다는 것이 성과가 아니라, 같은 조건에서 검증하고 **기준선 대비 개선을 입증하지 못한 선택 절차를 채택하지 않은 것**이 이 단계의 결론이다.

## 6. 공개 전 고정 예측 (표본 외 검증)

| 파일 | 크기 | SHA-256 | 상태 |
|---|---|---|---|
{chr(10).join(_frozen_status())}

채점 규칙(사전 등록 변경 기록): 3개월을 합친 하위 8개 WAPE에서 부스팅 3종이 **모두** 주 예측보다 낮으면 부스팅 우위를 확인한 것으로 본다.

## 7. 알려진 한계

- 기준 시점 8개(3개월 앞 6개), ML 학습 행 56~200개. 순위는 데이터가 쌓이면 바뀔 수 있다.
- 미국 측 금액은 신고 가격 변화(04 노트북)의 영향을 받는다. 예측 대상은 '미국 통계에 기록될 금액'이다.
- SARIMAX의 미래 환율은 마지막 값으로 고정했다.
- ML 결과는 실행 환경에 따라 달라진다(8절). 그래서 ML의 점수는 운영 판단의 근거로 쓰지 않는다.

{_platform_section()}"""
