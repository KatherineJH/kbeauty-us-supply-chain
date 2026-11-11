"""Write notebooks/03_cross_country.ipynb (analysis notebook for pre-registration Q1). Edit here, then:

    uv run python scripts/build_nb_03_cross_country.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_cross_country.ipynb
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "03_cross_country.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
INTERP = Path(__file__).with_name("nb03_interpretation.py")
interp = {}
if INTERP.exists():
    exec(INTERP.read_text(encoding="utf-8"), interp)
note = lambda key: md(interp.get(key, f"_(해석: 실행 결과를 확인한 뒤 작성 — {key})_"))

cells = [
    md("""# 03. 한국만의 변화였나 — 8개국 비교

**질문(사전 등록 Q1)**: 2025년 9월 이후 K-뷰티 대미 수입의 변화는 한국 특유의 변화인가, 미국 화장품 수입 전체에 공통된 변화인가?

| 항목 | 내용 |
|---|---|
| 처리군 | 한국 |
| 비교군 | 데이터를 받기 전에 정한 규칙(2024년 미국의 HS 3304 수입액 상위 8개국)으로 고른 프랑스·캐나다·이탈리아·중국·멕시코·영국·독일·일본 |
| 지표 | 수입 금액, 해상+항공 중량, 항공 중량, 항공 kg당 단가, 해상 kg당 단가 |
| 판정 기준 | Holm 보정 후 위약 p < 0.05 ([사전 등록](../docs/preregistration.md)) |

**답**: 사전 등록 기준을 통과한 지표가 없다(Holm 보정 후 최소 p 0.38). 항공 단가는 한국이 가장 크게 떨어졌지만 영국·프랑스·일본도 함께 떨어졌고, 이 설계의 검출 한계가 커서 정확한 결론은 "효과 없음"이 아니라 "판정 불가"다.

진행 순서: 비교 대상 → 원자료 → **평행 추세 확인** → 이중차분 → 위약 순위와 검출 한계 → 결론. 합성 통제 진단과 강건성 상세는 부록에 둔다.
계산 공식은 `src/kbeauty/comparison.py`에 있고 단위 테스트(`tests/test_comparison.py`)로 검증했다. 이 노트북은 그 함수들을 단계별로 호출하며 중간 결과를 보여 준다."""),

    code("""import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kbeauty import comparison as cmp
from kbeauty import report  # noqa: F401  (chart style)
from kbeauty import warehouse
from kbeauty.paths import POLICY_MONTH, TABLES_DIR
from kbeauty.viz import annotations as an
from kbeauty.viz import notebook as nbviz
from kbeauty.viz.theme import COLORS as C
from kbeauty.viz.comparison import COUNTRY_KO

%matplotlib inline
nbviz.setup()

pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
BREAK = pd.Timestamp(POLICY_MONTH)
panel = warehouse.load("mart_country_panel_monthly")
panel["month"] = pd.to_datetime(panel["month"])
print(panel.shape, panel["country"].nunique(), "countries,", panel["month"].nunique(), "months",
      panel["month"].min().date(), "~", panel["month"].max().date())"""),

    md("""## 1. 데이터 확인: 누구와 무엇을 비교하나

비교의 전제는 '같은 정의로 측정된 같은 품목'이다. 모든 국가가 같은 Census 엔드포인트, 같은 HS 4개 류, 같은 32개월이다.
먼저 국가별 규모와 운송수단 구성을 본다."""),

    code("""profile = (panel.groupby("country")
           .agg(value_bn=("value_usd", lambda s: s.sum() / 1e9),
                sea_air_share=("sea_air_share_value", "mean"),
                air_share_value=("air_value_usd", "sum"))
           .assign(air_share_value=lambda d: d.air_share_value / (panel.groupby("country").value_usd.sum()))
           .sort_values("value_bn", ascending=False))
profile"""),

    note("profile"),

    md("""## 2. 원자료 탐색: 국가별 전년 대비 성장률

국가마다 계절성이 다르므로 모든 분석은 **전년 동월 대비 로그 성장률**(YoY)로 한다. 2025-01부터 계산된다(정책 전 8개월, 후 12개월)."""),

    code("""def yoy_for(metric):
    countries = cmp.countries_for(metric, panel)
    return cmp.yoy(cmp.wide(panel, cmp.METRICS[metric][0], countries))

fig, axes = nbviz.figure("wide", 1, 2, title="항공 지표의 YoY: 한국과 비교 8개국", subtitle="굵은 선 = 한국, 가는 회색 선 = 비교국", sharex=True)
for ax, metric in zip(axes, ["air_weight_kg", "air_usd_per_kg"]):
    y = yoy_for(metric)
    for c in y.columns:
        if c != cmp.TREATED:
            ax.plot(y.index, y[c], lw=1, color=C["faint"])
    ax.plot(y.index, y[cmp.TREATED], lw=2.6, color=C["primary"])
    an.note(ax, y.index[-1], y[cmp.TREATED].iloc[-1], "한국", xytext=(5, 0), color=C["primary_dark"], weight=700)
    an.policy_marker(ax, label=False)
    ax.axhline(0, color=C["ink"], lw=0.6)
    ax.set_title(cmp.METRIC_KO[metric] + " YoY")
plt.show()"""),

    md("""정책 전후의 평균 YoY 변화를 국가별로 나란히 놓는다(통제 없는 기술 통계). D2의 핵심 발견이 처음 나온 표다."""),

    code("""raw_change = pd.DataFrame({
    cmp.METRIC_KO[m]: (lambda y: y[y.index >= BREAK].mean() - y[y.index < BREAK].mean())(yoy_for(m))
    for m in cmp.METRICS})
raw_change.style.format("{:+.3f}").highlight_min(axis=0, color="#f3d6cf").highlight_max(axis=0, color="#d6e4f0")"""),

    note("raw_change"),

    md("""## 3. 평행 추세 확인 (이중차분의 전제)

이중차분은 '정책이 없었다면 한국은 비교국과 같은 흐름이었을 것'을 전제로 한다. 정책 전 기간에 이 전제를 두 가지로 확인한다.
- **선행 F 검정**: 정책 전 각 월의 '한국 × 월' 효과가 모두 0인가 (p가 크면 전제와 모순 없음)
- **추세 이탈**: 정책 전 한국과 비교국 평균의 차이가 시간에 따라 벌어지는가

정책 전 YoY는 8개월뿐이라 두 검정 모두 힘이 약하다. 기각되지 않았다고 전제가 증명되는 것은 아니다."""),

    code("""pre = pd.DataFrame({m: cmp.pretrend(yoy_for(m), cmp.TREATED, BREAK) for m in cmp.METRICS}).T
pre.insert(0, "label", [cmp.METRIC_KO[m] for m in pre.index])
pre"""),

    code("""fig, axes = nbviz.figure((14, 3.8), 1, 5, title="한국 − 비교국 평균 YoY 격차", subtitle="정책 전 평균 = 0. 회색 막대 = 정책 전, 노란 막대 = 정책 후")
for ax, m in zip(axes, cmp.METRICS):
    path = cmp.event_path(yoy_for(m), cmp.TREATED, BREAK)
    ax.bar(path.index, path.values, width=20, color=np.where(path.index < BREAK, C["muted"], C["primary"]))
    ax.axhline(0, color=C["ink"], lw=0.6)
    ax.set_title(cmp.METRIC_KO[m], fontsize=10)
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
plt.show()"""),

    note("pretrend"),

    md("""## 4. 이중차분: 한국 특유의 변화 크기

YoY 패널에 국가·월 고정효과를 두고, 한국의 정책 후 추가 변화(β)를 추정한다. 단가 지표는 달러 기준이므로 각국 환율의 YoY 변화를 통제한다."""),

    code("""did_tbl = pd.DataFrame([{
    "label": cmp.METRIC_KO[m],
    "beta_unadjusted": cmp.did(yoy_for(m), cmp.TREATED, BREAK),
    "beta_fx_adjusted": cmp.did(yoy_for(m), cmp.TREATED, BREAK,
                                cmp.yoy(cmp.wide(panel, "lcu_per_usd", cmp.countries_for(m, panel))).reindex(yoy_for(m).index)),
} for m in cmp.METRICS], index=list(cmp.METRICS))
did_tbl["pct_effect"] = np.expm1(did_tbl.beta_fx_adjusted)
did_tbl"""),

    note("did"),

    md("""## 5. 위약 분포: 한국의 β는 얼마나 특별한가

처리군이 한국 하나뿐이면 표준오차를 믿기 어렵다. 그래서 **각 비교국을 한국처럼, 각 월을 정책 시점처럼** 두고 같은 β를 다시 추정한다.
한국의 실제 β보다 절댓값이 크거나 같은 위약의 비율이 위약 p다. 위약 단위는 한국을 비교군으로 쓰지 않는다."""),

    code("""placebos = {}
for m in cmp.METRICS:
    y = yoy_for(m)
    fx = cmp.yoy(cmp.wide(panel, "lcu_per_usd", cmp.countries_for(m, panel))).reindex(y.index)
    placebos[m] = cmp.placebo(y, cmp.TREATED, BREAK, fx)

fig, axes = nbviz.figure((14, 3.8), 1, 5, title="위약 분포 속 한국의 위치", subtitle="회색 = 각 국가·시점을 정책처럼 둔 β의 분포, 선 = 한국의 β")
for ax, m in zip(axes, cmp.METRICS):
    dist = placebos[m]["distribution"]
    ax.hist(dist.beta, bins=25, color=C["faint"])
    ax.axvline(placebos[m]["beta"], color=C["primary_dark"], lw=2.2)
    ax.set_title(f"{cmp.METRIC_KO[m]}\\np={placebos[m]['p_space_time']:.2f} (n={placebos[m]['n_space_time']})", fontsize=9.5)
plt.show()"""),

    code("""inf = pd.DataFrame({m: {k: v for k, v in placebos[m].items() if k != "distribution"} for m in cmp.METRICS}).T
inf["mde_80pct_power"] = [cmp.mde(placebos[m]["distribution"].query("unit != @cmp.TREATED").beta) for m in cmp.METRICS]
from statsmodels.stats.multitest import multipletests
inf["p_holm"] = multipletests(inf.p_space_time.astype(float), method="holm")[1]
inf["q_bh"] = multipletests(inf.p_space_time.astype(float), method="fdr_bh")[1]
inf["detectable"] = inf.beta.abs() >= inf.mde_80pct_power
inf.insert(0, "label", [cmp.METRIC_KO[m] for m in inf.index])
inf[["label", "beta", "p_time", "p_space", "p_space_time", "p_holm", "q_bh", "mde_80pct_power", "detectable"]]"""),

    note("placebo"),

    note("conclusion"),

    md("""---

## 부록"""),

    md("""### A1. 합성 통제 진단

비교국의 가중 조합으로 정책 전 20개월 동안 한국(로그 수준, 정책 전 평균 0)을 따라가는 '합성 한국'을 만든다.
**믿을 수 있으려면 정책 전에 잘 맞아야 한다**(정책 전 RMSPE가 작아야 한다). 그리고 한국의 정책 후 이탈이 각 비교국을 한국처럼 둔 위약보다 커야 한다."""),

    code("""synth = {m: cmp.synthetic_placebo(cmp.wide(panel, cmp.METRICS[m][0], cmp.countries_for(m, panel)), cmp.TREATED, BREAK)
         for m in cmp.METRICS}
diag = pd.DataFrame({m: {"label": cmp.METRIC_KO[m], "rmspe_pre": s["rmspe_pre"], "rmspe_post": s["rmspe_post"],
                         "ratio": s["ratio"], "p_rank": s["p_rank"], "top_donor": s["weights"].idxmax(),
                         "top_weight": s["weights"].max()} for m, s in synth.items()}).T
diag"""),

    code("""fig, axes = nbviz.figure((14, 4.0), 1, 5, title="합성 통제의 RMSPE 비(정책 후 ÷ 전) 순위", subtitle="노란 막대 = 한국. 길수록 정책 후 격차가 정책 전 적합 오차보다 크다")
for ax, m in zip(axes, cmp.METRICS):
    r = synth[m]["placebo_ratios"].sort_values()
    ax.barh([COUNTRY_KO.get(c, c) for c in r.index], r.values, color=[C["primary"] if c == cmp.TREATED else C["muted"] for c in r.index])
    ax.set_title(cmp.METRIC_KO[m], fontsize=10)
    ax.tick_params(labelsize=8)
plt.show()"""),

    note("synth"),

    md("""### A2. 강건성: 결론이 선택에 좌우되지 않는가

분석자가 내린 선택을 하나씩 바꿔 β와 위약 p가 어떻게 움직이는지 본다.
- 비교국: 영국 제외, 중국 제외, (해상·항공 지표에) 캐나다·멕시코 포함
- 정책 시점: 2025-08(관세 적용 월), 2025-10
- 환율 통제 제외"""),

    code("""def estimate(metric, drop=(), add_land=False, break_month=BREAK, use_fx=True):
    countries = [c for c in cmp.countries_for(metric, panel) if c not in drop]
    if add_land:
        countries += [c for c in cmp.LAND_BORDER if c not in countries]
    y = cmp.yoy(cmp.wide(panel, cmp.METRICS[metric][0], countries))
    fx = cmp.yoy(cmp.wide(panel, "lcu_per_usd", countries)).reindex(y.index) if use_fx else None
    p = cmp.placebo(y, cmp.TREATED, break_month, fx)
    return pd.Series({"beta": p["beta"], "p_space_time": p["p_space_time"]})

variants = {
    "기준": {},
    "영국 제외": {"drop": ("United Kingdom",)},
    "중국 제외": {"drop": ("China",)},
    "캐나다·멕시코 포함": {"add_land": True},
    "시점 2025-08": {"break_month": pd.Timestamp("2025-08-01")},
    "시점 2025-10": {"break_month": pd.Timestamp("2025-10-01")},
    "환율 통제 없음": {"use_fx": False},
}
rob = pd.concat({name: pd.DataFrame({cmp.METRIC_KO[m]: estimate(m, **kw) for m in cmp.METRICS}).T
                 for name, kw in variants.items()}, names=["variant", "metric"])
rob.unstack("variant").swaplevel(axis=1).sort_index(axis=1, level=0).loc[:, list(variants)]"""),

    code("""# β만 한눈에: 행 = 지표, 열 = 변형
rob["beta"].unstack("variant")[list(variants)].style.format("{:+.3f}").background_gradient(cmap="RdBu", vmin=-0.8, vmax=0.8, axis=None)"""),

    note("robustness"),

    md("""### A3. 일치 확인

이 노트북의 기준 추정치가 `scripts/run_all.py`가 만든 표(`reports/tables/comparison_did.csv`)와 같은지 확인한다. 다르면 이 셀이 실패한다."""),

    code("""saved = pd.read_csv(TABLES_DIR / "comparison_did.csv").set_index("metric")
assert np.allclose(saved.loc[list(cmp.METRICS), "beta_fx_adjusted"], inf["beta"].astype(float), atol=1e-9)
assert np.allclose(saved.loc[list(cmp.METRICS), "p_space_time"], inf["p_space_time"].astype(float), atol=1e-12)
print("notebook estimates match reports/tables/comparison_did.csv")"""),

]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
nbf.write(nb, OUT)
print(f"wrote {OUT}")
