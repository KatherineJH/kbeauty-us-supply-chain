"""Write notebooks/06_decision.ipynb (plan M4, M5, Q5, D1, D2). Edit here, then:

    uv run python scripts/build_nb_06_decision.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/06_decision.ipynb

Interpretation text lives in scripts/nb06_interpretation.py and is written after reading the executed outputs.
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "06_decision.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
INTERP = Path(__file__).with_name("nb06_interpretation.py")
interp = {}
if INTERP.exists():
    exec(INTERP.read_text(encoding="utf-8"), interp)
note = lambda key: md(interp.get(key, f"_(해석: 실행 결과를 확인한 뒤 작성 — {key})_"))

cells = [
    md("""# 06. 어디로 들여오고, 언제 바꾸고, 얼마나 노출돼 있나

미국 현지 창고에 재고를 두는 K-뷰티 브랜드의 SCM 담당자에게 이 데이터가 답할 수 있는 질문을 세 개로 줄였다.

| 질문 | 답 | 근거 수준 |
|---|---|---|
| 1. 서부와 동부 중 어디로 들여오나 | 해상 수입 금액의 82%가 LA·롱비치. 해상 구간은 서부 13일, 동부 28일이고 정책 후에도 변하지 않았다 | `OBSERVED` · `ESTIMATED` |
| 2. 어떤 조건에서 선택이 바뀌나 | 동부 수요를 서부 하역 + 내륙 운송으로 채우는 경로는 내륙 운송이 **약 14일 이하**일 때 유리하다. 경계는 내륙 일수의 흔들림에 가장 민감하다 | `SCENARIO` |
| 3. LA·롱비치 중단에 얼마나 노출돼 있나 | 2주 중단이면 해상 수입 약 5,500만 달러(연간의 3%)가 지연된다 | `SCENARIO` |

**근거 수준**: `OBSERVED` 공개 데이터 관측치 · `ESTIMATED` 통계적 검정 · `SCENARIO` 관측되지 않은 가정의 계산(내륙 운송 일수·변동, 수요 변동, 서비스 수준, 중단 기간) · `NOT MEASURED` 이 데이터에 없는 것(운임, 터미널 체류, 통관 시간, 재고 유지 비용).
가정값과 민감도 범위는 `reports/tables/assumptions.csv`, 결정별 권고는 `reports/tables/decision_recommendations.csv`에 있다."""),
    code("""import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kbeauty import decision, report, warehouse
from kbeauty.paths import POLICY_MONTH, POST_WINDOW, PRE_WINDOW, TABLES_DIR
from kbeauty.viz import annotations as an
from kbeauty.viz import notebook as nbviz
from kbeauty.viz.theme import COLORS as C
from kbeauty.report import CV_GRID, INLAND_SD, SERVICE_LEVELS

%matplotlib inline
nbviz.setup()
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
transit = warehouse.load("mart_vessel_transit"); transit["filing_month"] = pd.to_datetime(transit["filing_month"])
conc = warehouse.load("mart_port_concentration_monthly"); conc["month"] = pd.to_datetime(conc["month"])
port_region = warehouse.load("mart_port_region_monthly")
REGIONS = ["west_coast", "gulf_coast", "east_coast"]
KO = {"west_coast": "서부", "gulf_coast": "걸프", "east_coast": "동부"}
print(transit.region.value_counts().to_string())"""),

    md("""## 질문 1. 서부와 동부 중 어디로 들여오나

### 1-1. 물량은 지금 어디로 들어오나 `OBSERVED`"""),
    code("""nbviz.show("06_port_concentration")
window = lambda d, w, col="month": d[d[col].between(*w)]
pd.DataFrame({name: window(conc, w)[["la_lb_share", "west_coast_share", "east_coast_share", "gulf_coast_share", "hhi_vessel"]].mean()
              for name, w in (("정책 전 12개월", PRE_WINDOW), ("정책 후 12개월", POST_WINDOW))}).T"""),
    note("ports"),

    md("""### 1-2. 해상 구간은 얼마나 걸리나 `OBSERVED`"""),
    code("""stats = pd.DataFrame([decision.lead_time_stats(transit, r) for r in REGIONS]).set_index("region")
fig, ax = nbviz.figure(title="권역별 해상 리드타임 분포", subtitle="부산 출항 → 미국 입항 신고 일수. 서부와 동부·걸프는 약 2주 떨어진 두 무리")
for r, c in zip(REGIONS, (C["west"], C["muted"], C["east"])):
    x = transit.loc[transit.region == r, "departure_to_filing_days"]
    ax.hist(x, bins=range(5, 46), alpha=0.75, color=c)
    an.note(ax, x.median(), 1.0, KO[r], coords=("data", "axes fraction"), va="top", ha="center", color=c, weight=700)
ax.set_xlabel("부산 출항 → 미국 입항 신고 (일)")
plt.show()
stats"""),
    note("lead"),

    md("""### 1-3. 정책 후 해상 구간이 달라졌나 `ESTIMATED` (사전 등록 Q5)

판정 기준: 정책 전후 중앙값 차이의 부트스트랩 95% 신뢰구간이 0을 포함하지 않으면 '변화'로 판정한다."""),
    code("""q5 = pd.DataFrame([decision.period_shift(transit, r, POLICY_MONTH) for r in REGIONS]).set_index("region")
monthly = transit.groupby(["filing_month", "region"]).departure_to_filing_days.median().unstack()[REGIONS]
fig, ax = nbviz.figure(title="월별 리드타임 중앙값 (신고 월 기준)")
for r, c in zip(REGIONS, (C["west"], C["muted"], C["east"])):
    ax.plot(monthly.index, monthly[r], color=c, lw=1.8)
    an.note(ax, monthly.index[-1], monthly[r].iloc[-1], KO[r], xytext=(6, 0), color=c, weight=700)
an.policy_marker(ax)
ax.set_ylabel("일")
plt.show()
q5"""),
    note("q5"),

    md("""## 질문 2. 어떤 조건에서 선택이 바뀌나 `SCENARIO`

동부 수요를 **동부 직송**(관측된 해상 구간)으로 채울지, **서부 하역 + 내륙 운송**(관측된 서부 해상 구간 + 가정한 내륙 구간)으로 채울지를 묶이는 재고 일수로 비교한다.

$$\\text{안전재고(일)} = z \\cdot \\sqrt{L \\cdot CV_d^2 + \\sigma_L^2}, \\qquad \\text{묶이는 재고(일)} = L + \\text{안전재고(일)}$$

- 동부 직송: L과 σ_L은 동부 관측치 그대로.
- 서부 하역 + 내륙: L = 서부 관측 평균 + 내륙 일수, σ_L = √(서부 분산 + 내륙 분산).
- **시나리오**(관측 아님): 내륙 일수 5~15일, 내륙 표준편차 2일, 일별 수요 변동계수 CV 0.3/0.5/0.9, 서비스 수준 90/95/99%.
  참고로 LA·롱비치에서 컨테이너가 철도에 실리기까지만 약 4~7일이 걸린다는 보도가 있다. 철도 운송과 최종 배송은 그 위에 더해진다.
- 해상 운임·내륙 운임 등 **비용은 포함하지 않는다.** 재고 일수만 비교한다."""),
    code("""east, west = stats.loc["east_coast"].to_dict(), stats.loc["west_coast"].to_dict()
grid = decision.route_comparison(east, west, range(5, 16), INLAND_SD, CV_GRID, SERVICE_LEVELS)
pivot = grid[grid.service_level == 0.95].pivot(index="inland_days", columns="cv_demand", values="west_minus_east_days")
nbviz.show("14_route_breakeven")
nbviz.show("16_route_sensitivity")
pivot"""),
    code("""be = pd.DataFrame([{"cv_demand": cv, "service_level": sl, "inland_sd": isd,
                    "break_even_inland_days": decision.break_even_inland_days(east, west, isd, cv, sl)}
                   for cv in CV_GRID for sl in SERVICE_LEVELS for isd in (1.0, INLAND_SD, 4.0)])
be.pivot_table(index=["inland_sd", "service_level"], columns="cv_demand", values="break_even_inland_days")"""),
    note("route"),

    md("""## 질문 3. LA·롱비치 중단에 얼마나 노출돼 있나 `SCENARIO`

하루 평균 해상 수입액(관측) × LA·롱비치 비중(관측) × 중단 일수(가정). 중단 중에도 비중과 일평균 금액이 정책 후 12개월 평균과 같다고 둔다."""),
    code("""disrupt = decision.port_disruption(port_region, conc, POST_WINDOW)
reroute_extra_days = east["mean"] - west["mean"]
disrupt.assign(exposed_musd=lambda d: d.exposed_value_usd / 1e6,
               share_of_annual_sea_value=lambda d: d.exposed_value_usd / (d.vessel_value_per_day_usd * 365))[
    ["closure_days", "la_lb_share", "exposed_musd", "share_of_annual_sea_value"]]"""),
    code("""print(f"동부 우회 시 해상 구간 추가 일수: 평균 {reroute_extra_days:.1f}일 (동부 {east['mean']:.1f}일 − 서부 {west['mean']:.1f}일)")
for cv in CV_GRID:
    e = decision.route_inventory(east["mean"], east["sd"], cv, 0.95)
    w = decision.route_inventory(west["mean"], west["sd"], cv, 0.95)
    print(f"수요 CV {cv}: 서부 하역(내륙 제외) {w['total_days']:.1f}일 → 동부 우회 {e['total_days']:.1f}일, 추가로 묶이는 재고 {e['total_days'] - w['total_days']:.1f}일분")"""),
    note("disruption"),

    note("conclusion"),

    md("""---

## 부록: 일치 확인

노트북의 Q5·손익분기·항만 노출 결과가 `scripts/run_all.py`의 표와 같은지 확인한다."""),
    code("""s_q5 = pd.read_csv(TABLES_DIR / "decision_q5_shift.csv").set_index("region")
assert np.allclose(s_q5.loc[REGIONS, ["median_pre", "median_post", "ci_low", "ci_high"]], q5.loc[REGIONS, ["median_pre", "median_post", "ci_low", "ci_high"]])
s_be = pd.read_csv(TABLES_DIR / "decision_breakeven.csv")
mine = be[be.inland_sd == INLAND_SD].merge(s_be, on=["cv_demand", "service_level"], suffixes=("_nb", "_saved"))
assert np.allclose(mine.break_even_inland_days_nb, mine.break_even_inland_days_saved)
s_dis = pd.read_csv(TABLES_DIR / "decision_port_disruption.csv")
assert np.allclose(s_dis.exposed_value_usd, disrupt.exposed_value_usd)
print("notebook results match reports/tables/decision_*.csv")"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
nbf.write(nb, OUT)
print(f"wrote {OUT}")
