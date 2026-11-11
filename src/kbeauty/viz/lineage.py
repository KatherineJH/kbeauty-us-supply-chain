"""Data lineage diagram: 7 sources -> dbt staging and seeds -> marts -> analysis modules -> outputs.
dbt edges are read from the manifest (documentation.lineage), so the diagram follows the warehouse."""

import matplotlib.patches as mpatches
import pandas as pd

from ..paths import FIGURES_DIR
from . import annotations as an
from .export import save
from .theme import COLORS, TEXT

SOURCES = [  # (label, agency, staging models it feeds)
    ("미국 수입 HS10 (한국산)", "Census", ["stg_census__hs10_monthly"]),
    ("하역항별 수입 HS6", "Census", ["stg_census__port_hs6_monthly"]),
    ("CF-1400 입항 신고", "CBP", ["stg_vessel__routes"]),
    ("PORT-MIS 출항 기록", "해양수산부", ["stg_vessel__routes"]),
    ("미국 수입 HS10 (비교 8개국)", "Census", ["stg_census__peer_hs10_monthly"]),
    ("품목별 국가별 수출실적", "관세청", ["stg_kcs__kr_exports_us", "stg_kcs__hsk10_names"]),
    ("월평균 환율 7종", "Fed Board G.5", ["stg_fed_board__fx_monthly"]),
]
MODULES = ["decompose · 단가·환율 분해", "events · 전후 비교", "comparison · 국가 비교", "mirror · 미러 분석",
           "structure · 집중도·리드타임", "forecast · hierarchy · ml · 예측", "decision · 재고·경로", "synthesis · 주장·가정·권고"]
X = {"source": 0.0, "staging": 1.25, "core": 2.5, "mart": 3.6, "module": 4.85, "output": 6.05}
W = {"source": 1.05, "staging": 1.1, "core": 0.95, "mart": 1.0, "module": 1.1, "output": 0.95}


def _box(ax, x, y, w, h, text, sub=None, face=None, edge=None, weight=500, size=None, color=None):
    ax.add_patch(mpatches.FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.008,rounding_size=0.03",
                                         facecolor=face or COLORS["panel"], edgecolor=edge or COLORS["grid"], lw=0.9, zorder=2))
    ax.text(x + 0.04, y + (0.012 if sub else 0), text, ha="left", va="center", fontsize=size or TEXT["note"] - 0.5,
            fontweight=weight, color=color or COLORS["ink"], zorder=3)
    if sub:
        ax.text(x + 0.04, y - 0.022, sub, ha="left", va="center", fontsize=TEXT["note"] - 1.5, color=COLORS["muted"], zorder=3)


def _edge(ax, x0, y0, x1, y1, color=None, lw=0.7):
    xm = (x0 + x1) / 2
    path = mpatches.Path([(x0, y0), (xm, y0), (xm, y1), (x1, y1)],
                         [mpatches.Path.MOVETO, mpatches.Path.CURVE4, mpatches.Path.CURVE4, mpatches.Path.CURVE4])
    ax.add_patch(mpatches.PathPatch(path, fill=False, edgecolor=color or COLORS["faint"], lw=lw, zorder=1))


def _column(names, top=0.9, bottom=0.08):
    step = (top - bottom) / max(len(names) - 1, 1)
    return {n: top - i * step for i, n in enumerate(names)}


def lineage(edges: pd.DataFrame, results: dict, seed_names=None):
    staging = sorted({c for c in edges.parent if c.startswith("stg_")} | {t for s in SOURCES for t in s[2]})
    seeds = sorted(seed_names or {p for p in edges.parent if not p.startswith("stg_") and p not in set(edges.child)})
    marts = sorted(set(edges.child))
    core = [m for m in marts if m in set(edges.parent)]          # marts that other marts read
    final = [m for m in marts if m not in core]
    y = {**_column([s[0] for s in SOURCES], 0.9, 0.3)}
    ys = _column(staging + seeds, 0.92, 0.06)
    yc = _column(core, 0.72, 0.42)
    ym = _column(final, 0.9, 0.12)
    ymod = _column(MODULES, 0.86, 0.16)
    n_fig = len({p.stem for p in FIGURES_DIR.glob("*.png")} | {"00_data_lineage"})
    outputs = [(f"결과 표 {len(results)}개", "reports/tables"), (f"그림 {n_fig}개", "PNG 2배 + SVG"),
               ("노트북 7개", "00_summary ~ 06_decision"), ("공개 전 고정 예측", "SHA-256 · reports/frozen"),
               ("데이터 사전 · 모델 카드", "자동 생성 문서"), ("사례 연구", "HTML · PDF")]
    yo = _column([o[0] for o in outputs], 0.82, 0.22)

    fig, ax = an.figure((14.0, 7.6), subtitle_lines=1)
    ax.set_xlim(-0.05, X["output"] + W["output"] + 0.05)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for col, label in (("source", "원천 7개 · 기관 5곳"), ("staging", "dbt 스테이징 · 시드"), ("core", "공통 마트"),
                       ("mart", "분석 마트"), ("module", "분석 모듈 (src/kbeauty)"), ("output", "산출물")):
        ax.text(X[col], 0.985, label, ha="left", va="top", fontsize=TEXT["label"], fontweight=700, color=COLORS["ink"])

    for label, agency, targets in SOURCES:
        for t in targets:
            _edge(ax, X["source"] + W["source"], y[label], X["staging"], ys[t])
    pos = {**{n: (X["staging"], ys[n]) for n in staging + seeds}, **{n: (X["core"], yc[n]) for n in core},
           **{n: (X["mart"], ym[n]) for n in final}}
    wid = {**{n: W["staging"] for n in staging + seeds}, **{n: W["core"] for n in core}, **{n: W["mart"] for n in final}}
    for e in edges.itertuples():
        (x0, y0), (x1, y1) = pos[e.parent], pos[e.child]
        _edge(ax, x0 + wid[e.parent], y0, x1, y1)
    for n in final:
        _edge(ax, X["mart"] + W["mart"], ym[n], X["module"], 0.51, color=COLORS["grid"], lw=0.6)
    for o in outputs:
        _edge(ax, X["module"] + W["module"], 0.51, X["output"], yo[o[0]], color=COLORS["grid"], lw=0.6)

    for label, agency, _ in SOURCES:
        _box(ax, X["source"], y[label], W["source"], 0.06, label, agency, face=COLORS["ink"], edge=COLORS["ink"],
             color=COLORS["paper"], weight=600)
    for n in staging:
        _box(ax, X["staging"], ys[n], W["staging"], 0.045, n)
    for n in seeds:
        _box(ax, X["staging"], ys[n], W["staging"], 0.045, f"seed · {n}", face=COLORS["paper"], color=COLORS["muted"])
    for n in core:
        _box(ax, X["core"], yc[n], W["core"], 0.05, n, face=COLORS["panel"], edge=COLORS["muted"], weight=600)
    for n in final:
        _box(ax, X["mart"], ym[n], W["mart"], 0.05, n, edge=COLORS["muted"], weight=600)
    ax.add_patch(mpatches.FancyBboxPatch((X["module"] - 0.03, 0.1), W["module"] + 0.06, 0.82,
                                         boxstyle="round,pad=0.01,rounding_size=0.04", facecolor=COLORS["panel"],
                                         edgecolor=COLORS["primary"], lw=1.4, zorder=1.5))
    for m in MODULES:
        ax.text(X["module"] + 0.02, ymod[m], m, ha="left", va="center", fontsize=TEXT["note"] - 0.5, color=COLORS["ink"], zorder=3)
    for label, sub in outputs:
        _box(ax, X["output"], yo[label], W["output"], 0.065, label, sub, face=COLORS["panel"], edge=COLORS["primary"], weight=700)

    an.header(fig, "원천 7개에서 결론까지, 모든 숫자는 한 경로로 다시 만들어진다",
              f"원문 해시 검증 → 오프라인 재추출 → dbt 스테이징 {len(staging)}·마트 {len(marts)}·시드 {len(seeds)} → 분석 모듈 → 산출물. 선 = dbt 의존 관계(manifest)",
              None)
    an.footer(fig, "warehouse/target/manifest.json · scripts/run_all.py", "dbt 데이터 테스트와 pytest가 각 단계를 검증")
    return save(fig, "00_data_lineage")
