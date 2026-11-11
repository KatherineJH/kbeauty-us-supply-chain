"""Write notebooks/05_forecasting.ipynb (plan F1-F4). Edit here, then:

    uv run python scripts/build_nb_05_forecasting.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_forecasting.ipynb

Interpretation text lives in scripts/nb05_interpretation.py and is written after reading the executed outputs.
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "05_forecasting.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
INTERP = Path(__file__).with_name("nb05_interpretation.py")
interp = {}
if INTERP.exists():
    exec(INTERP.read_text(encoding="utf-8"), interp)
note = lambda key: md(interp.get(key, f"_(해석: 실행 결과를 확인한 뒤 작성 — {key})_"))

cells = [
    md("""# 05. 어떤 예측을 언제 쓸 것인가 — 의사결정 기간별 운영 규칙

**SCM 담당자에게 필요한 예측은 기간마다 다르다.**
- **1개월 앞**: 다음 달 입고 흐름을 보고 운영을 조정한다(긴급 항공 보충, 창고 인력).
- **3개월 앞**: 분기 물량과 해상·항공 운송 계획을 잡는다.

이 노트북은 가장 단순한 기준선에서 출발해 통계 모델, 계층 정합, 머신러닝 9종을 **같은 rolling-origin 조건**에서 비교하고, 기간별로 쓸 규칙을 정한다.

- 대상: 미국 측 해상+항공 수입 금액(전체의 98.6%), 품목군 4 × 운송수단 2의 15개 노드 계층. **공개 수입통계의 예측**이며, 기업 내부 판매·재고·발주 데이터를 쓰는 수요예측이 아니다. 발주 판단을 돕는 외부 시장 흐름 지표로 읽는다.
- 예측 기준 시점: 2025-12 ~ 2026-07(학습 24개월 이상), 1~3개월 앞. 모든 모델은 기준 시점 이후 값을 볼 수 없다(`tests/test_forecast.py`, `tests/test_ml.py`).
- 비교 기준: 구매 담당자가 흔히 쓰는 **'작년 같은 달'**(seasonal naive). 설계와 결과 요약은 [모델 카드](../reports/model_card.md)."""),

    code("""import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kbeauty import forecast, hierarchy, ml, report, warehouse
from kbeauty.paths import BACKTEST_FIRST_ORIGIN, POLICY_MONTH, REPO, TABLES_DIR
from kbeauty.viz import annotations as an
from kbeauty.viz import notebook as nbviz
from kbeauty.viz.theme import COLORS as C
from kbeauty.viz.forecasting import MODEL_KO

%matplotlib inline
nbviz.setup()
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
fct = warehouse.load("fct_imports_hs10_mode_monthly")
bottom = hierarchy.bottom_series(fct)
nodes = hierarchy.all_nodes(bottom)
krw = warehouse.load("stg_fed_board__fx_monthly").query("currency == 'KRW'")
krw = krw.assign(month=pd.to_datetime(krw["month"])).set_index("month")["lcu_per_usd"]
FIRST = int(np.flatnonzero(bottom.index == pd.Timestamp(BACKTEST_FIRST_ORIGIN))[0])
print(nodes.shape, "| first origin", bottom.index[FIRST].date(), "| last month", bottom.index[-1].date())"""),

    md("""## 결론 먼저: 운영 규칙

| 의사결정 기간 | 사용할 예측 | 이유 | 근거 수준 |
|---|---|---|---|
| 1개월 앞 | **직전 값** | 합계 WAPE 0.112, 비교한 모델 중 가장 낮음. 순위는 통계적으로 확정되지 않음(DM p 0.17) | `ESTIMATED` |
| 3개월 앞 | **계절 ETS, 품목·운송수단별 합** | 합계 WAPE 0.160, '작년 같은 달' 대비 우위 DM p 0.001(n 6) | `ESTIMATED` |
| 머신러닝 | **보류** | 검증 성적으로 고른 모델이 직전 값과 통계적으로 구분되는 개선을 입증하지 못해 직전 값 기준선 유지(공식 기준 Linux 고정 환경 1개월 0.109 vs 0.112, DM p 0.91. 작성 환경 0.138은 진단 이력), 3개월 앞은 구분 불가 | `ESTIMATED`, 고정 예측 채점 대기 |
| 쓰지 않을 것 | '작년 같은 달' | 정책 이후 수준이 이동해 모든 수준·기간에서 하위권 | `ESTIMATED` |

아래는 이 규칙에 이른 근거다."""),

    md("""## 1. 예측 대상: 15개 노드 계층"""),

    code("""fig, axes = nbviz.figure((14, 6.2), 2, 4, title="하위 8개 시계열: 품목군 × 운송수단 (백만 달러)", subtitle="검은 선 = 해상, 노란 선 = 항공", sharex=True)
for ax, col in zip(axes.ravel(), bottom.columns):
    ax.plot(bottom.index, bottom[col] / 1e6, color=C["primary"] if col.endswith("air") else C["ink"], lw=1.8)
    an.policy_marker(ax, label=False)
    ax.set_title(col, fontsize=10); ax.tick_params(labelsize=8)
plt.show()
share = (nodes.iloc[-12:].sum() / nodes["total"].iloc[-12:].sum()).rename("최근 12개월 비중")
share.to_frame().T"""),

    note("hierarchy"),

    md("""## 2. 가장 단순한 기준선과 통계 모델

| 모델 | 설명 |
|---|---|
| naive | 직전 달 값 |
| seasonal_naive | 작년 같은 달 |
| seasonal_naive_growth | 작년 같은 달 × 최근 12개월 성장률 |
| ets_damped | 감쇠 추세 지수평활(로그) |
| ets_seasonal | 감쇠 추세 + 12개월 계절 지수평활(로그) |
| sarimax_fx_policy | SARIMAX(1,1,0) + 정책 후 더미 + 원/달러(미래 환율은 마지막 값 유지) |

각 노드의 예측은 독립적으로 만든 뒤(base), 상위·하위 합계가 맞도록 정합한다(bottom-up, OLS). 머신러닝 모델은 4절에서 따로 비교한다."""),

    code("""models = hierarchy.standard_models(POLICY_MONTH, krw)
bt = hierarchy.backtest(bottom, models, FIRST)
lw = hierarchy.level_wape(bt)
base = lw[lw.method == "base"].pivot_table(index="model", columns=["level", "horizon"], values="wape")
base[["total", "bottom"]].sort_values(("total", 1))"""),

    code("""fig, axes = nbviz.figure((12, 4.8), 1, 2, title="통계 모델별 WAPE(정합 전, 낮을수록 좋음)", subtitle="노랑 = 1개월 앞, 연회색 = 2개월 앞, 검정 = 3개월 앞", sharey=True)
order = ["naive", "sarimax_fx_policy", "ets_seasonal", "seasonal_naive_growth", "ets_damped", "seasonal_naive"]
for ax, level in zip(axes, ["total", "bottom"]):
    d = lw[(lw.method == "base") & (lw.level == level)].pivot(index="model", columns="horizon", values="wape").reindex(order)
    for i, (h, c) in enumerate(((1, C["primary"]), (2, C["faint"]), (3, C["ink"]))):
        ax.bar(np.arange(len(order)) + (i - 1) * 0.26, d[h], width=0.26, color=c)
    ax.set_xticks(range(len(order)), [MODEL_KO[m] for m in order], rotation=30, ha="right", fontsize=9)
    ax.set_title("합계" if level == "total" else "하위 8개 (합산)")
plt.show()"""),

    note("models"),

    md("""## 3. 계층 정합: 합계가 맞게 고치면 정확도도 좋아지나"""),

    code("""rec = lw[lw.model.isin(["ets_seasonal", "ets_damped", "sarimax_fx_policy", "naive"])].pivot_table(
    index=["model", "level"], columns=["method", "horizon"], values="wape")
rec.loc[(slice(None), ["total", "bottom"]), :]"""),

    note("reconcile"),

    md("""## 4. 머신러닝 9종: 여러 모델을 같은 규칙으로 겨루고, 검증 성적으로 고른다

통계 모델은 시계열마다 따로 맞춘다. 머신러닝 모델은 하위 8개 시계열을 **한 모델로 함께 학습**(global model)해, 시계열 하나당 24~31개월뿐인 데이터 부족을 메울 수 있는지 본다.

| 계열 | 모델 | 조정한 하이퍼파라미터 |
|---|---|---|
| 선형 | Ridge, Elastic Net | 규제 강도 |
| 거리 기반 | kNN | 이웃 수 |
| 트리 앙상블 | Random Forest, Extra Trees | 잎 최소 표본 수 |
| 부스팅 | Gradient Boosting(scikit-learn), LightGBM, XGBoost | 트리 깊이·잎 수 |
| 신경망 | MLP(은닉층 16) | 가중치 규제 |

- **공통 특징**: 기준 시점의 최근 3개월 로그 값, 월, 시계열, 정책 후 여부, 기준 시점의 원/달러. 'full' 특징은 여기에 작년 같은 달을 더한다(대신 학습 행이 12개월 줄어든다).
- **함께 조정한 설계 선택**: 목표(로그 수준 또는 직전 달 대비 로그 변화. 트리는 학습 범위 밖 수준을 외삽하지 못한다)와 특징 묶음(short·full). 모델 9종 × 설정 2~3개 × 목표 2 × 특징 2 = **76개 설정**.
- **선택 규칙(시험 달을 보지 않는다)**: 각 기준 시점에서, 그 시점에 이미 관측된 **마지막 3개월**을 검증 구간으로 삼아 76개 설정을 채점한다(검증 달 v는 v − h 이전 자료로만 학습). 모델마다 가장 좋은 설정으로 기준 시점까지 다시 학습해 예측하고(`ml_<모델>`), 전체에서 가장 좋은 설정을 `ml_selected`로 둔다. `ml_selected`가 "여러 모델을 비교해 가장 좋은 것을 고른다"는 절차의 정직한 표본 외 성적이다.
- 하위 8개 예측을 더해 15개 노드 전체를 만든다(bottom-up). 모든 학습은 단일 스레드·고정 시드라 다시 실행해도 같은 숫자가 나온다(`tests/test_ml.py`)."""),

    code("""mlr = ml.backtest(bottom, krw, FIRST, POLICY_MONTH)   # 약 3~4분
all_bt = pd.concat([bt, mlr.forecasts], ignore_index=True)
lw_all = hierarchy.level_wape(all_bt)
ref = [("naive", "base"), ("ets_seasonal", "bottom_up"), ("sarimax_fx_policy", "base")]
ml_rows = [(m, "bottom_up") for m in sorted(mlr.forecasts.model.unique())]
cmp = lw_all.set_index(["model", "method"]).loc[ref + ml_rows].reset_index()
cmp.pivot_table(index=["model", "method"], columns=["level", "horizon"], values="wape")[["total", "bottom"]].sort_values(("total", 1))"""),

    code("""nbviz.show("15_ml_comparison")  # the report figure, drawn from the same backtest table checked in section 10"""),

    note("ml_models"),

    md("""### 4.1 검증 구간은 어떤 모델을 골랐나

기준 시점·예측 시점마다 `ml_selected`가 고른 설정과 그 검증 WAPE. 고른 모델이 자주 바뀐다면 검증 구간 3개월로는 모델 간 차이를 안정적으로 구분하지 못한다는 뜻이다."""),

    code("""sel = mlr.selection
picked = sel[sel.model == "ml_selected"]
display(picked.pivot(index="origin", columns="horizon", values="chosen"))
picked.groupby(["horizon", "family"]).size().unstack(fill_value=0)"""),

    code("""# 사후 점검(선택에 쓰지 않음): 검증 성적과 실제 시험 성적은 같은 방향인가
# 기준 시점·예측 시점마다 9개 모델의 검증 WAPE 순위와 시험 WAPE 순위의 상관(Spearman)
test_w = (mlr.forecasts.query("method == 'base' and model != 'ml_selected'")
          .assign(err=lambda d: (d.forecast - d.actual).abs())
          .groupby(["origin", "horizon", "model"]).apply(lambda g: g.err.sum() / g.actual.sum(), include_groups=False).rename("test_wape").reset_index())
rank = sel[sel.model != "ml_selected"].merge(test_w, on=["origin", "horizon", "model"])
rho = rank.groupby(["origin", "horizon"]).apply(lambda g: g.inner_wape.rank().corr(g.test_wape.rank()), include_groups=False).rename("spearman")
rho.groupby("horizon").describe()[["count", "mean", "min", "max"]]"""),

    note("ml_selection"),

    md("""### 4.2 무엇이 예측에 쓰였나: 표본 외 순열 중요도

`ml_selected`의 시험 예측에서 특징 하나를 무작위로 섞었을 때 WAPE가 얼마나 나빠지는지(10회 평균). 기준 시점마다 그 시점의 모델로 예측한다. 시험 달은 모두 정책 이후라 '정책 후 여부'는 시험 구간에서 값이 변하지 않아 이 방법으로는 측정할 수 없다."""),

    code("""imp = mlr.importance
imp.pivot(index="feature", columns="horizon", values="wape_rise_mean").sort_values(1, ascending=False)"""),

    note("ml_importance"),

    md("""## 5. 차이가 우연인가: Diebold-Mariano 검정

합계 노드에서 모델 쌍의 절대 오차를 비교한다(HLN 소표본 보정). 기준 시점이 8개(3개월 앞은 6개)뿐이라 검정력이 낮다.
부스팅 모델과 직전 값의 비교는 **결과를 본 뒤 고른 비교**이므로 판정이 아니라 참고로만 읽는다."""),

    code("""def errors(model, method, h, node="total"):
    d = all_bt[(all_bt.model == model) & (all_bt.method == method) & (all_bt.horizon == h) & (all_bt.node == node)].sort_values("origin")
    return (d.forecast - d.actual).to_numpy()

dm = pd.DataFrame([
    {"comparison": "naive vs seasonal_naive", "horizon": 1, **hierarchy.diebold_mariano(errors("naive", "base", 1), errors("seasonal_naive", "base", 1), 1)},
    {"comparison": "ets_seasonal(BU) vs seasonal_naive", "horizon": 3, **hierarchy.diebold_mariano(errors("ets_seasonal", "bottom_up", 3), errors("seasonal_naive", "base", 3), 3)},
    {"comparison": "naive vs ets_seasonal(BU)", "horizon": 1, **hierarchy.diebold_mariano(errors("naive", "base", 1), errors("ets_seasonal", "bottom_up", 1), 1)},
    {"comparison": "ml_selected(BU) vs naive", "horizon": 1, **hierarchy.diebold_mariano(errors("ml_selected", "bottom_up", 1), errors("naive", "base", 1), 1)},
    {"comparison": "ml_selected(BU) vs ets_seasonal(BU)", "horizon": 3, **hierarchy.diebold_mariano(errors("ml_selected", "bottom_up", 3), errors("ets_seasonal", "bottom_up", 3), 3)},
    {"comparison": "[사후] ml_lightgbm(BU) vs naive", "horizon": 1, **hierarchy.diebold_mariano(errors("ml_lightgbm", "bottom_up", 1), errors("naive", "base", 1), 1)},
    {"comparison": "[사후] ml_xgboost(BU) vs naive", "horizon": 1, **hierarchy.diebold_mariano(errors("ml_xgboost", "bottom_up", 1), errors("naive", "base", 1), 1)},
])
dm"""),

    note("dm"),

    md("""## 6. 예측 구간: 얼마나 넓게 잡아야 하나

각 기준 시점의 구간은 **다른 기준 시점들의 오차**로만 만든다(자기 오차로 보정하지 않는다). 15개 노드를 모아 80%·95% 구간의 실제 포함률을 본다."""),

    code("""cov = pd.concat([hierarchy.conformal_coverage(bt.query("model == @m and method == @r")).assign(model=m, method=r)
                 for m, r in (("naive", "base"), ("ets_seasonal", "ols"))], ignore_index=True)
cov.assign(width_pct=lambda d: np.expm1(d.mean_width_log / 2)).pivot_table(
    index=["model", "method", "horizon"], columns="level", values=["coverage", "width_pct", "n_invalid"])"""),

    note("conformal"),

    md("""## 7. 공개 전에 고정한 예측

2026-11-02에 데이터 공개 전 상태로 2026-09~11 예측 30개를 고정했다([사전 등록](../docs/preregistration.md) 5절). 파일 해시를 확인하고, 공개된 달이 있으면 채점한다."""),

    code("""import hashlib
frozen = sorted((REPO / "reports" / "frozen").glob("forecast_*_frozen_*.csv"))[0]
expected = frozen.with_suffix(".sha256").read_text().split()[0]
print(frozen.name, "| hash ok:", hashlib.sha256(frozen.read_bytes()).hexdigest() == expected)
ff = pd.read_csv(frozen)
published = sorted(m for m in ff.target_month.unique() if pd.Timestamp(m) <= nodes.index[-1])
print("published target months:", published or "none yet (Census latest month: %s)" % nodes.index[-1].strftime("%Y-%m"))
ff[ff.series == "total|value_usd"][["target_month", "forecast_primary", "pi80_low", "pi80_high", "backtest_median_actual_to_forecast"]]"""),

    note("frozen"),

    note("conclusion"),

    md("""---

## 부록"""),

    md("""### A1. 구조 변화 확인

정책 이후 수준이 이동했다면 1년 전 패턴에 기대는 모델이 불리하다. 전년 대비 성장률의 평균이 안정적인지 CUSUM 검정으로 본다."""),

    code("""total = nodes["total"]
pd.DataFrame([hierarchy.cusum_break(np.log(total))])"""),

    note("cusum"),

    md("""### A2. 선행 지표: 한국 수출 통계로 미리 알 수 있나

한국 관세청 통계는 미국 Census보다 약 3주 빨리 나온다. 미국 월 t+1을 '한국 월 t+1 × 최근 3개월 미국/한국 비율'로 예측해 본다."""),

    code("""mir = warehouse.load("mart_mirror_monthly")
kr_total = mir.assign(month=pd.to_datetime(mir.month)).groupby("month").kr_export_usd.sum()
us_total = nodes["total"]
nbt = hierarchy.nowcast_backtest(us_total, kr_total.reindex(us_total.index), FIRST)
now = hierarchy.wape(nbt, ["model"])
ratio = (us_total / kr_total.reindex(us_total.index))
fig, ax = nbviz.figure(title="미국 해상+항공 수입 / 한국 수출: 정책 후 비율이 이동했다", subtitle="한국 통계를 선행 지표로 쓰려면 이 비율이 안정적이어야 한다")
ax.plot(ratio.index, ratio, color=C["ink"], lw=2)
an.policy_marker(ax)
plt.show()
now"""),

    note("nowcast"),

    md("""### A3. 일치 확인

노트북의 백테스트 결과가 `scripts/run_all.py`의 표와 같은지 확인한다."""),

    code("""saved = pd.read_csv(TABLES_DIR / "forecast_hierarchy_wape.csv")
merged = saved.merge(lw_all, on=["model", "method", "horizon", "level"], suffixes=("_saved", "_nb"))
assert len(merged) == len(saved) and np.allclose(merged.wape_saved, merged.wape_nb, atol=1e-12)
saved_now = pd.read_csv(TABLES_DIR / "forecast_nowcast.csv").set_index("model").wape
assert np.allclose(saved_now.sort_index(), now.set_index("model").wape.sort_index(), atol=1e-12)
saved_sel = pd.read_csv(TABLES_DIR / "forecast_ml_selection.csv")
assert saved_sel.chosen.tolist() == sel.chosen.tolist() and np.allclose(saved_sel.inner_wape, sel.inner_wape, atol=1e-12)
saved_imp = pd.read_csv(TABLES_DIR / "forecast_ml_importance.csv")
assert np.allclose(saved_imp.wape_rise_mean, imp.wape_rise_mean, atol=1e-12)
print("notebook backtests match reports/tables: forecast_hierarchy_wape, forecast_nowcast, forecast_ml_selection, forecast_ml_importance")"""),

]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
nbf.write(nb, OUT)
print(f"wrote {OUT}")
