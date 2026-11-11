"""Write notebooks/04_mirror.ipynb (analysis notebook for pre-registration Q2). Edit here, then:

    uv run python scripts/build_nb_04_mirror.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_mirror.ipynb

Interpretation text lives in scripts/nb04_interpretation.py and is written after reading the executed outputs.
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "04_mirror.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
INTERP = Path(__file__).with_name("nb04_interpretation.py")
interp = {}
if INTERP.exists():
    exec(INTERP.read_text(encoding="utf-8"), interp)
note = lambda key: md(interp.get(key, f"_(해석: 실행 결과를 확인한 뒤 작성 — {key})_"))

cells = [
    md("""# 04. 소포 가설은 왜 지지되지 않았나 — 한국 수출 대 미국 수입 미러 분석

**가설(사전 등록 Q2)**: 2025년 9월 이후 미국의 한국산 항공 수입 증가와 항공 단가 하락은 **측정 범위 변화** 때문이다. 이전에 미국 통계에 잡히지 않던 소액면세 소포가 통계에 들어오기 시작했다는 뜻이다.

| 단계 | 내용 |
|---|---|
| 가설이 맞다면 | 소액면세(800달러 이하) 화물은 2025-08-29 이전 미국 Census 통계에 없었지만 한국 수출 신고에는 있을 수 있다. 그렇다면 **미국 수입 / 한국 수출 비율이 정책 후 올라가야 한다** |
| 실제로는 | 금액 비율은 **−14%**(HAC p 0.005). 예측과 반대 방향이다 |
| 판정 | **소포 가설은 지지되지 않는다** |
| 반전 | 같은 흐름의 kg당 기록 금액이 한국에서는 오르고 미국에서는 떨어졌다 |
| 남는 해석 | 미국 측 신고 가격의 변화. 통관 건별 자료 없이는 확정할 수 없다 |

데이터: 한국 관세청 품목별 국가별 수출입실적(미국, FOB)과 미국 Census 한국산 수입(과세가격)을, 두 나라 세번 체계가 공유하는 가장 세밀한 단위인 **HS 6단위 × 월**로 맞췄다.
판정 규칙은 데이터를 받기 전에 정했다([사전 등록](../docs/preregistration.md)): 정책 후 비율이 유의하게 상승(p < 0.05)하면 가설 지지."""),

    code("""import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kbeauty import mirror, report, warehouse
from kbeauty.paths import POLICY_MONTH, POST_WINDOW, PRE_WINDOW, TABLES_DIR
from kbeauty.viz import annotations as an
from kbeauty.viz import notebook as nbviz
from kbeauty.viz.theme import COLORS as C

%matplotlib inline
nbviz.setup()
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
BREAK = pd.Timestamp(POLICY_MONTH)
mir_raw = warehouse.load("mart_mirror_monthly")
mir_raw["month"] = pd.to_datetime(mir_raw["month"])
m = mirror.monthly(mir_raw)
print("HS6 lines:", mir_raw.hs6.nunique(), "| months with both sides:", len(m), m.index.min().date(), "~", m.index.max().date(),
      "| Korea also has:", sorted(mir_raw.loc[~mir_raw.us_published, "month"].dt.strftime("%Y-%m").unique()))"""),

    md("""## 1. 두 기록 맞추기

두 나라의 10단위 세번은 서로 다르게 나뉘지만(예: 미국은 3304.99를 1개 세목, 한국은 4개 세목으로 나눔) 6단위는 국제 공통이다.
양쪽 모두 같은 16개 HS6에 값이 있는지, 규모가 어떻게 다른지 본다."""),

    code("""by_hs6 = (mir_raw[mir_raw.us_published].groupby("hs6")[["kr_export_usd", "us_import_usd"]].sum() / 1e6)
by_hs6["us_over_kr"] = by_hs6.us_import_usd / by_hs6.kr_export_usd
by_hs6["kr_share"] = by_hs6.kr_export_usd / by_hs6.kr_export_usd.sum()
by_hs6.sort_values("kr_export_usd", ascending=False)"""),

    note("data"),

    md("""## 2. 원자료: 같은 흐름을 두 나라가 어떻게 기록했나"""),

    code("""fig, axes = nbviz.figure((10, 6.6), 2, 1, title="같은 흐름의 두 기록과 그 비율", subtitle="위 = 월별 금액, 아래 = 미국/한국 비율(1 = 같은 금액·중량)", sharex=True)
axes[0].plot(m.index, m.kr_export_usd / 1e6, color=C["ink"], lw=2)
axes[0].plot(m.index, m.us_import_usd / 1e6, color=C["primary"], lw=2.4)
axes[0].set_ylabel("백만 달러")
axes[1].plot(m.index, m.ratio_value, color=C["ink"], lw=2)
axes[1].plot(m.index, m.ratio_weight, color=C["muted"], lw=1.6, ls="--")
axes[1].axhline(1, color=C["ink"], lw=0.6)
axes[1].set_ylabel("미국 / 한국")
for ax in axes:
    an.policy_marker(ax, label=ax is axes[0])
    ax.set_xlim(m.index.min(), m.index.max() + pd.Timedelta(days=200))
an.end_labels(axes[0], [(m.index[-1], m.kr_export_usd.iloc[-1] / 1e6, "한국 수출(FOB)", C["ink"], 600),
                        (m.index[-1], m.us_import_usd.iloc[-1] / 1e6, "미국 수입(과세가격)", C["primary_dark"], 700)])
an.end_labels(axes[1], [(m.index[-1], m.ratio_value.iloc[-1], "금액 비율", C["ink"], 600),
                        (m.index[-1], m.ratio_weight.iloc[-1], "중량 비율", C["muted"], 600)])
plt.show()
m[["kr_export_usd", "us_import_usd", "ratio_value", "ratio_weight"]].assign(
    kr_export_usd=lambda d: d.kr_export_usd / 1e6, us_import_usd=lambda d: d.us_import_usd / 1e6)"""),

    note("raw"),

    md("""## 3. 주 검정(사전 등록): 정책 후 미국/한국 비율이 올라갔나"""),

    code("""tests = pd.DataFrame({
    "금액 비율": mirror.shift_test(m.log_ratio_value, BREAK),
    "중량 비율": mirror.shift_test(m.log_ratio_weight, BREAK),
}).T
tests[["pre_mean", "shift", "ci_low", "ci_high", "p_value", "pct_change_in_ratio", "placebo_p", "n_candidates", "largest_break"]]"""),

    note("main"),

    md("""## 4. 반전: 같은 흐름, 반대 방향의 가격

금액 비율은 내리고 중량 비율은 올랐다. 그렇다면 같은 물량에 대해 **미국 측 kg당 금액이 한국 측보다 떨어졌다**는 뜻이다.
상대 단가 = (미국 금액 / 미국 해상+항공 중량) / (한국 FOB 금액 / 한국 중량) = 금액 비율 / 중량 비율.
보고서 그림 12는 정책 전후 12개월 합계(HS6 16개)로, 아래 월별 검정은 월 단위 비율로 같은 변화를 본다."""),

    code("""nbviz.show("12_mirror_same_flow_different_price")"""),

    code("""uv = pd.DataFrame({"kr_usd_per_kg": m.kr_export_usd / m.kr_export_kg,
                   "us_usd_per_kg": (m.us_vessel_usd + m.us_air_usd) / m.us_sea_air_kg})
uv["relative_us_over_kr"] = uv.us_usd_per_kg / uv.kr_usd_per_kg
fig, ax = nbviz.figure(title="월별 kg당 신고 금액: 한국 기록과 미국 기록", subtitle="정책 전후 12개월 합계의 비교는 보고서 그림 12")
ax.plot(uv.index, uv.kr_usd_per_kg, color=C["ink"], lw=2)
ax.plot(uv.index, uv.us_usd_per_kg, color=C["primary"], lw=2.4)
an.policy_marker(ax)
ax.set_ylabel("USD / kg")
ax.set_xlim(uv.index.min(), uv.index.max() + pd.Timedelta(days=160))
an.end_labels(ax, [(uv.index[-1], uv.kr_usd_per_kg.iloc[-1], "한국(FOB)", C["ink"], 600),
                   (uv.index[-1], uv.us_usd_per_kg.iloc[-1], "미국(해상+항공)", C["primary_dark"], 700)])
plt.show()
rel = pd.DataFrame({"상대 단가 (미국/한국)": mirror.shift_test(np.log(uv.relative_us_over_kr), BREAK),
                    "한국 USD/kg": mirror.shift_test(np.log(uv.kr_usd_per_kg), BREAK),
                    "미국 USD/kg": mirror.shift_test(np.log(uv.us_usd_per_kg), BREAK)}).T
rel[["pre_mean", "shift", "ci_low", "ci_high", "p_value", "pct_change_in_ratio", "placebo_p", "largest_break"]]"""),

    note("reversal"),

    note("remaining"),

    note("conclusion"),

    md("""---

## 부록

본문 결론을 받치는 확인 작업이다. A2는 사전 등록에 없던 분석이라 판정이 아니라 해석을 돕는 용도로만 쓴다."""),

    md("""### A1. 시차

비율의 **수준**은 1이 아니어도 된다(가격 기준·시차·통관 범위가 다르다). 검정 대상은 비율의 **변화**다. 부산에서 월말에 출항한 화물은 다음 달 미국에 도착한다. 미국 월 t와 한국 월 t−k의 전월 대비 변화율 상관을 본다."""),

    code("""lag = mirror.lag_correlation(m.us_import_usd, m.kr_export_usd)
lag"""),

    note("lag"),

    md("""### A2. 주 검정의 강건성(사전 등록에 없던 분석)

- 시차를 맞춘 비율: 미국 월 t / 한국 월 t−1
- 3개월 합계 비율: 월말 경계 효과를 줄인다
- 항공만 따로: 미국 항공 수입 / 한국 수출 (한국 통계에는 운송수단 구분이 없어 분모는 전체)
- 어느 HS6가 비율 변화를 만드나"""),

    code("""aligned = np.log(m.us_import_usd / m.kr_export_usd.shift(1)).dropna()
rolling3 = np.log(m.us_import_usd.rolling(3).sum() / m.kr_export_usd.rolling(3).sum()).dropna()
air_only = np.log(m.us_air_usd / m.kr_export_usd)
vessel_only = np.log(m.us_vessel_usd / m.kr_export_usd)
aux = pd.DataFrame({
    "시차 맞춤 (미국 t / 한국 t−1)": mirror.shift_test(aligned, BREAK),
    "3개월 합계 비율": mirror.shift_test(rolling3, BREAK),
    "미국 항공 / 한국 전체": mirror.shift_test(air_only, BREAK),
    "미국 해상 / 한국 전체": mirror.shift_test(vessel_only, BREAK),
}).T
aux[["shift", "ci_low", "ci_high", "p_value", "pct_change_in_ratio", "placebo_p", "largest_break"]]"""),

    code("""gap = mirror.hs6_gap_change(mir_raw, PRE_WINDOW, POST_WINDOW)
gap.assign(**{c: gap[c] / 1e6 for c in ["kr_pre", "us_pre", "kr_post", "us_post", "gap_change_usd"]}).head(8)"""),

    note("robust"),

    md("""### A3. 품목 범위와 외부 집계

이 분석은 HS 3303·3304·3305·3307만 쓴다. 비누(3401), 구강용품(3306), 향료 원료 등을 포함하면 한국 측 수출액이 얼마나 커지는지
보관된 관세청 원문에서 직접 계산한다(새 API 호출 없음). 아래 외부 집계 비교에 쓴다."""),

    code("""scopes = {
    "HS 3303·3304·3305·3307 (이 분석)": ("3303", "3304", "3305", "3307"),
    "+ 구강용품 3306": ("3303", "3304", "3305", "3306", "3307"),
    "HS 33 전체 (3301~3307)": ("33",),
    "HS 33 전체 + 비누 3401": ("33", "3401"),
}
raw_scope = {name: mirror.raw_exports(p) for name, p in scopes.items()}
h1_2026 = pd.Series({name: d[d.month.between("2026-01-01", "2026-06-01")].export_usd.sum() / 1e8
                     for name, d in raw_scope.items()}, name="2026 상반기 한국 → 미국 수출 (억 달러)")
h1_2026.to_frame()"""),

    note("scope"),

    md("""**외부 집계와의 차이: 14.5억 달러 vs 10.8억 달러**

프로젝트 기획 단계에서 인용한 "2026년 상반기 대미 화장품 수출 14.5억 달러(식약처)"와 이 분석의 미국 측 수입(같은 기간)을 나란히 놓고,
차이를 품목 범위 · 양국 통계 차이로 나눠 본다. 식약처 집계의 정확한 품목 정의는 원문을 확인하지 못했다."""),

    code("""h1 = m.loc["2026-01-01":"2026-06-01"]
bridge = pd.Series({
    "외부 집계 (식약처 인용, 품목 정의 미확인)": 14.5,
    "한국 수출: HS 33 전체 + 비누 3401": h1_2026["HS 33 전체 + 비누 3401"],
    "한국 수출: HS 33 전체": h1_2026["HS 33 전체 (3301~3307)"],
    "한국 수출: HS 3303·3304·3305·3307": h1.kr_export_usd.sum() / 1e8,
    "미국 수입: HS 3303·3304·3305·3307 (Census)": h1.us_import_usd.sum() / 1e8,
}, name="억 달러")
bridge.to_frame()"""),

    note("bridge"),

    md("""### A4. 일치 확인

이 노트북의 주 검정 결과가 `scripts/run_all.py`가 만든 표(`reports/tables/mirror_shift.csv`)와 같은지 확인한다."""),

    code("""saved = pd.read_csv(TABLES_DIR / "mirror_shift.csv").set_index("metric")
assert np.isclose(saved.loc["log_ratio_value", "shift"], tests.loc["금액 비율", "shift"], atol=1e-12)
assert np.isclose(saved.loc["log_ratio_weight", "shift"], tests.loc["중량 비율", "shift"], atol=1e-12)
assert np.isclose(saved.loc["log_ratio_value", "placebo_p"], tests.loc["금액 비율", "placebo_p"], atol=1e-12)
print("notebook estimates match reports/tables/mirror_shift.csv")"""),

]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
nbf.write(nb, OUT)
print(f"wrote {OUT}")
