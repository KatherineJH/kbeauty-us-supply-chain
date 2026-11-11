"""Write notebooks/01_data_overview.ipynb (sources, quality checks, missing values, outliers, distributions).

    uv run python scripts/run_all.py        # builds the warehouse and dbt run results this notebook reads
    uv run python scripts/build_nb_01_data_overview.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_overview.ipynb

Interpretation text lives in scripts/nb01_interpretation.py and is written after reading the executed outputs.
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "01_data_overview.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
INTERP = Path(__file__).with_name("nb01_interpretation.py")
interp = {}
if INTERP.exists():
    exec(INTERP.read_text(encoding="utf-8"), interp)
note = lambda key: md(interp.get(key, f"_(해석: 실행 결과를 확인한 뒤 작성 — {key})_"))

cells = [
    md("""# 01. 왜 미국 통계 하나로는 답할 수 없나 — 7개 원천과 데이터 품질

**미국 통계에서 이상 신호가 보였다.** 정책 후 12개월 한국산 화장품 수입 금액은 +11.7%인데 해상+항공 중량은 +39.5%, kg당 단가는 크게 떨어졌다.
이 데이터만으로 할 수 있는 해석과 할 수 없는 해석을 나누면, 무엇을 더 모아야 하는지가 정해진다.

| 미국 통계만으로 답할 수 없는 질문 | 필요한 원천 | 이 프로젝트의 원천 |
|---|---|---|
| 한국만의 변화인가, 미국 화장품 수입 전체의 변화인가 | 같은 정의의 다른 나라 통계 | Census 비교 8개국 |
| 소포가 통계에 새로 잡힌 것인가 | 같은 흐름의 다른 쪽 기록 | 관세청 대미 수출 |
| 환율 때문인가 | 월별 환율 | Federal Reserve Board G.5 7개 통화 |
| 어느 항만으로 들어오나 | 하역항별 통계 | Census 하역항 × HS6 |
| 운송에 얼마나 걸리나 | 선박의 출항·입항 기록 | PORT-MIS × CBP CF-1400 |

이 노트북은 7개 원천의 역할과 규모를 먼저 보고, 이후 노트북(02~06)의 결론이 서는 데이터 품질을 확인한다: 자동 품질 검사(dbt), 결측·0값, 이상치, 분포, 선박 연결 품질."""),

    code("""import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kbeauty import report, warehouse
from kbeauty.paths import REPO, TABLES_DIR
from kbeauty.viz import annotations as an
from kbeauty.viz import notebook as nbviz
from kbeauty.viz.theme import COLORS as C

%matplotlib inline
nbviz.setup()
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
pd.set_option("display.max_rows", 60)
load = lambda t: warehouse.load(t).pipe(lambda d: d.assign(**{c: pd.to_datetime(d[c]) for c in ("month",) if c in d}))
us = load("stg_census__hs10_monthly")
ports = load("stg_census__port_hs6_monthly")
peers = load("stg_census__peer_hs10_monthly")
kr = load("stg_kcs__kr_exports_us")
fx = load("stg_fed_board__fx_monthly")
vessel = warehouse.load("stg_vessel__routes")
summary = load("mart_monthly_summary")"""),

    md("""## 1. 7개 원천과 각 원천의 역할"""),

    code("""def span(d, col="month"):
    return f"{d[col].min():%Y-%m} ~ {d[col].max():%Y-%m}"

sources = pd.DataFrame([
    {"원천": "Census 한국산 수입 (HS10)", "단위": "HS10 × 월", "행": len(us), "기간": span(us), "키 수": us.hs10.nunique()},
    {"원천": "Census 한국산 수입 (하역항)", "단위": "하역항 × HS6 × 월", "행": len(ports), "기간": span(ports), "키 수": ports.port_code.nunique()},
    {"원천": "Census 비교국 수입 (HS10)", "단위": "국가 × HS10 × 월", "행": len(peers), "기간": span(peers), "키 수": peers.cty_code.nunique()},
    {"원천": "관세청 한국 → 미국 수출 (HSK10)", "단위": "HSK10 × 월", "행": len(kr), "기간": span(kr), "키 수": kr.hsk10.nunique()},
    {"원천": "Fed Board 환율", "단위": "통화 × 월", "행": len(fx), "기간": span(fx), "키 수": fx.currency.nunique()},
    {"원천": "부산 출항 × 미국 입항 신고 (선박)", "단위": "선박 1회 운항", "행": len(vessel),
     "기간": f"{pd.to_datetime(vessel.us_filing_date).min():%Y-%m} ~ {pd.to_datetime(vessel.us_filing_date).max():%Y-%m}",
     "키 수": vessel.call_sign.nunique()},
])
raw_files = sorted(p.relative_to(REPO).as_posix() for p in (REPO / "data" / "raw").rglob("*") if p.is_file())
print(f"보관된 원문 파일 {len(raw_files)}개, 해시 기록 {len((REPO / 'data' / 'SHA256SUMS').read_text().splitlines())}개")
sources"""),

    note("sources"),

    md("""## 2. 자동 품질 검사 (dbt)

`scripts/run_all.py`가 실행한 dbt build의 결과 파일(`warehouse/target/run_results.json`)을 읽는다. 모델과 데이터 테스트가 모두 통과해야 분석이 시작된다."""),

    code("""rr = json.loads((REPO / "warehouse" / "target" / "run_results.json").read_text(encoding="utf-8"))
res = pd.DataFrame([{"node": r["unique_id"].split(".")[-1] if r["unique_id"].startswith("test.") else r["unique_id"].split(".")[-1],
                     "type": r["unique_id"].split(".")[0], "status": r["status"]} for r in rr["results"]])
print(res.groupby(["type", "status"]).size().to_string())
singular = res[(res.type == "test") & res.node.str.startswith("assert_")]
singular[["node", "status"]].reset_index(drop=True)"""),

    note("dbt"),

    md("""## 3. 결측과 0값

Census는 값이 0인 품목×월 행을 내려주지 않는다. 따라서 '행이 없음'은 결측이 아니라 0이다. 품목별로 32개월 중 몇 달에 값이 있는지 본다."""),

    code("""n_months = us.month.nunique()
cover = (us.groupby("hs10").agg(months_with_value=("month", "nunique"), value_musd=("value_usd", lambda s: s.sum() / 1e6))
           .assign(coverage=lambda d: d.months_with_value / n_months).sort_values("value_musd", ascending=False))
cover["share_of_value"] = cover.value_musd / cover.value_musd.sum()
print("US HS10 lines present every month:", int((cover.coverage == 1).sum()), "of", len(cover),
      "| value share of lines with gaps:", round(cover.loc[cover.coverage < 1, "share_of_value"].sum(), 4))
cover"""),

    code("""mode_gap = pd.DataFrame({
    "other (육로·우편·미상) 금액 비중": [us.other_value_usd.sum() / us.value_usd.sum()],
    "해상 금액은 있는데 해상 중량 0인 행": [int(((us.vessel_value_usd > 0) & (us.vessel_weight_kg == 0)).sum())],
    "항공 금액은 있는데 항공 중량 0인 행": [int(((us.air_value_usd > 0) & (us.air_weight_kg == 0)).sum())],
    "그 행들의 금액 합 (달러)": [int(us.loc[((us.vessel_value_usd > 0) & (us.vessel_weight_kg == 0)), "vessel_value_usd"].sum()
                                  + us.loc[((us.air_value_usd > 0) & (us.air_weight_kg == 0)), "air_value_usd"].sum())],
}).T.rename(columns={0: "값"})
mode_gap"""),

    note("missing"),

    md("""## 4. 이상치

**4-1. 월별 합계**: 계절성이 있으므로 전년 동월 대비 로그 성장률(YoY)을 기준으로, 중앙값에서 MAD(중앙값 절대 편차)의 3배 이상 벗어난 달을 표시한다."""),

    code("""s = summary.set_index("month")
def mad_flags(x, k=3.0):
    med = x.median(); mad = (x - med).abs().median() * 1.4826
    return (x - med).abs() > k * mad
yoy = np.log(s[["value_usd", "measured_weight_kg", "vessel_weight_kg", "air_weight_kg"]]).diff(12).dropna()
flags = yoy.apply(mad_flags)
print("flagged months:")
print(yoy[flags.any(axis=1)].round(3).to_string())
fig, ax = nbviz.figure(title="월별 YoY 로그 성장률: MAD 3배를 벗어난 달(점)", subtitle="계절성을 없애려 전년 동월 대비로 본다. 점 = 이상치 후보")
items = []
for c, color, name in (("value_usd", C["ink"], "수입 금액"), ("measured_weight_kg", C["primary"], "해상+항공 중량"), ("air_weight_kg", C["muted"], "항공 중량")):
    ax.plot(yoy.index, yoy[c], color=color, lw=1.8)
    ax.scatter(yoy.index[flags[c]], yoy.loc[flags[c], c], color=color, s=55, zorder=3, edgecolor=C["ink"])
    items.append((yoy.index[-1], yoy[c].iloc[-1], name, C["primary_dark"] if color == C["primary"] else color, 600))
ax.axhline(0, color=C["ink"], lw=0.6)
an.policy_marker(ax)
ax.set_xlim(yoy.index.min(), yoy.index.max() + pd.Timedelta(days=120))
an.end_labels(ax, items)
plt.show()"""),

    md("""**4-2. 품목 × 월 단가**: 같은 HS10 안에서 kg당 단가가 그 품목의 중앙값에서 크게 벗어난 달(해상·항공 각각, 금액 10만 달러 이상인 행만)."""),

    code("""fct = load("fct_imports_hs10_mode_monthly").query("mode != 'other' and value_usd >= 100000 and weight_kg > 0")
fct["log_uv"] = np.log(fct.value_usd / fct.weight_kg)
fct["robust_z"] = fct.groupby(["hs10", "mode"]).log_uv.transform(
    lambda x: (x - x.median()) / ((x - x.median()).abs().median() * 1.4826 + 1e-9))
out = fct[fct.robust_z.abs() > 4].sort_values("value_usd", ascending=False)
print(f"{len(out)} of {len(fct)} rows ({len(out) / len(fct):.1%}), value share {out.value_usd.sum() / fct.value_usd.sum():.2%}")
out[["month", "hs10", "category", "mode", "value_usd", "weight_kg", "unit_value_usd_per_kg", "robust_z"]].head(12)"""),

    note("outliers"),

    md("""## 5. 분포"""),

    code("""from kbeauty.viz.formats import percent_axis
fig, axes = nbviz.figure("three_panel", 1, 3, title="금액 비중: 품목군 · 운송수단 · 하역항", subtitle="32개월 합계 기준")
cat = us.merge(warehouse.load("hs10_category"), on="hs10").groupby("category_ko").value_usd.sum().sort_values()
axes[0].barh(cat.index, cat / cat.sum(), color=[C["primary"] if v == cat.max() else C["muted"] for v in cat]); axes[0].set_title("품목군별")
modes = pd.Series({"해상": us.vessel_value_usd.sum(), "항공": us.air_value_usd.sum(), "기타": us.other_value_usd.sum()})
axes[1].bar(modes.index, modes / modes.sum(), color=[C["ink"], C["primary"], C["faint"]]); axes[1].set_title("운송수단별")
top_ports = ports.groupby("port_name").value_usd.sum().sort_values().tail(8)
axes[2].barh(top_ports.index.str.slice(0, 24), top_ports / ports.value_usd.sum(), color=C["muted"])
axes[2].set_title("하역항 상위 8곳"); axes[2].tick_params(labelsize=8)
percent_axis(axes[0], "x"); percent_axis(axes[1], "y"); percent_axis(axes[2], "x")
plt.show()
pd.DataFrame({"category_share": (cat / cat.sum()).sort_values(ascending=False)})"""),

    code("""panel = warehouse.load("mart_country_panel_monthly")
countries = (panel.groupby("country").agg(value_bn=("value_usd", lambda s: s.sum() / 1e9),
                                          sea_air_share=("sea_air_share_value", "mean"),
                                          air_usd_per_kg=("air_usd_per_kg", "median"),
                                          vessel_usd_per_kg=("vessel_usd_per_kg", "median"))
             .sort_values("value_bn", ascending=False))
countries"""),

    code("""names = warehouse.load("stg_kcs__hsk10_names").set_index("hsk10").name_ko
kr_top = kr.groupby("hsk10").agg(export_musd=("export_usd", lambda s: s.sum() / 1e6), export_kt=("export_kg", lambda s: s.sum() / 1e6))
kr_top["usd_per_kg"] = kr_top.export_musd / kr_top.export_kt
kr_top["share"] = kr_top.export_musd / kr_top.export_musd.sum()
kr_top.insert(0, "name_ko", names.reindex(kr_top.index).values)
kr_top.sort_values("export_musd", ascending=False).head(10)"""),

    note("distributions"),

    md("""## 6. 선박 관측치: 리드타임 원천의 연결 품질

부산 출항(PORT-MIS)과 미국 입항 신고(CF-1400)는 공통 식별자가 없어 호출부호·선박명으로 연결했다. 연결 방식과 통계 사용 여부를 본다."""),

    code("""v = vessel.assign(use=vessel.use_for_stats.map({True: "통계 사용", False: "제외"}))
link = v.pivot_table(index="match_method", columns="use", values="observation_id", aggfunc="count", fill_value=0)
link["합계"] = link.sum(axis=1)
print(f"관측 {len(v)}건 중 통계 사용 {int(v.use_for_stats.sum())}건 ({v.use_for_stats.mean():.0%})")
display(link.sort_values("합계", ascending=False))
used = v[v.use_for_stats]
fig, ax = nbviz.figure(title="통계에 쓰는 관측치의 리드타임: 서부 13일, 걸프·동부 26~28일 두 무리", subtitle="부산 출항 → 미국 입항 신고 일수의 분포")
ax.hist(used.departure_to_filing_days, bins=range(5, 46), color=C["muted"], edgecolor=C["paper"])
ax.set_xlabel("부산 출항 → 미국 입항 신고 (일)")
plt.show()"""),

    note("vessel"),

    md("""## 7. 일치 확인

이 노트북이 원천에서 직접 더한 월별 합계가 마트(`mart_monthly_summary`)와 `run_all` 표(`reports/tables/window_totals.csv`)와 같은지 확인한다."""),

    code("""direct = us.groupby("month").value_usd.sum()
assert (direct.values == summary.set_index("month").value_usd.reindex(direct.index).values).all()
wt = pd.read_csv(TABLES_DIR / "window_totals.csv").set_index("metric")
assert wt.loc["value_usd", "post_12m"] == direct.loc["2025-09-01":"2026-08-01"].sum()
print("source totals match mart_monthly_summary and reports/tables/window_totals.csv")"""),

    note("conclusion"),

]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
nbf.write(nb, OUT)
print(f"wrote {OUT}")
