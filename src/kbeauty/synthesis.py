"""Final layer: what the analysis claims, what it assumes, and what it recommends (three result tables).

Every number in these tables is read from the other result tables, so the text cannot drift from the analysis.

Evidence levels (column `evidence_level`):
- observed    measured directly in the public statistics, or an accounting identity of measured values
- estimated   a statistical test, model or backtest on observed data (with its sample size and robustness)
- scenario    computed from assumptions that public data do not observe (listed in `assumptions`)
- hypothesis  consistent with the evidence but not testable with these data; no causal claim

Scope: forecasts here are forecasts of public US import statistics (value recorded by Census, by product group
and mode). They are external market signals that can inform planning; they are not demand forecasts built on a
company's own sales, inventory or order data, and the decisions below are rules of thumb, not optimisations.
"""

import numpy as np
import pandas as pd

from .numeric_env import run_label

SCOPE_NOTE = ("공개 수입통계(미국 Census 기록 금액)의 예측이다. 기업 내부 판매·재고·발주 데이터를 쓰는 수요예측이 아니며, "
              "발주 판단을 돕는 외부 시장 흐름 지표로만 쓴다.")


def _pct(x: float, signed: bool = True) -> str:
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def _man(usd: float) -> str:
    """USD in units of 10,000 (만 달러), the unit used in the notebooks and case study."""
    return f"{usd / 1e4:,.0f}만 달러"


REGION_KO = {"west_coast": "서부", "gulf_coast": "걸프", "east_coast": "동부"}


def _row(df: pd.DataFrame, col: str, value) -> pd.Series:
    return df[df[col] == value].iloc[0]


def _wape(lw: pd.DataFrame, model: str, method: str, h: int, level: str = "total") -> float:
    return float(lw[(lw.model == model) & (lw.method == method) & (lw.horizon == h) & (lw.level == level)].wape.iloc[0])


def assumptions(stats: dict, base: dict, ranges: dict, sensitivity: pd.DataFrame,
                port: pd.DataFrame) -> pd.DataFrame:
    """Scenario inputs with their base value, sensitivity range and the effect of that range on the decision.

    base / ranges: inland_sd_days, cv_demand, service_level, closure_days. Break-even effects vary one input over
    its range with the others at their base values."""
    sens = sensitivity

    def be(**kw):
        p = {**{k: base[k] for k in ("inland_sd_days", "cv_demand", "service_level")}, **kw}
        return float(sens[(sens.inland_sd_days == p["inland_sd_days"]) & (sens.cv_demand == p["cv_demand"])
                          & (sens.service_level == p["service_level"])].break_even_inland_days.iloc[0])

    def be_range(name):
        vals = [be(**{name: v}) for v in ranges[name]]
        return f"손익분기 내륙 일수 {min(vals):.1f}~{max(vals):.1f}일"

    exposed = port.set_index("closure_days").exposed_value_usd
    east, west = stats["east_coast"], stats["west_coast"]
    rows = [
        ("inland_days", "서부 항만 → 동부 내륙 운송 일수", "일", "scenario", "가정(공개 데이터에 없음)",
         float(min(sens.inland_days)), float(max(sens.inland_days)), np.nan, "하역 권역 선택",
         f"판단 기준 자체(손익분기와 비교하는 값). 기준 조건 손익분기 {be():.1f}일"),
        ("inland_sd_days", "내륙 운송 일수의 표준편차", "일", "scenario", "가정(운송 업체 자료로 대체 필요)",
         float(min(ranges["inland_sd_days"])), float(max(ranges["inland_sd_days"])), float(base["inland_sd_days"]),
         "하역 권역 선택", be_range("inland_sd_days")),
        ("cv_demand", "일 수요의 변동계수", "비율", "scenario", "가정(기업 판매 데이터로 대체 필요)",
         float(min(ranges["cv_demand"])), float(max(ranges["cv_demand"])), float(base["cv_demand"]),
         "하역 권역 선택, 안전재고", be_range("cv_demand")),
        ("service_level", "주기 서비스 수준", "비율", "scenario", "정책 선택값",
         float(min(ranges["service_level"])), float(max(ranges["service_level"])), float(base["service_level"]),
         "하역 권역 선택, 안전재고", be_range("service_level")),
        ("closure_days", "LA·롱비치 중단 기간", "일", "scenario", "가정(과거 파업·혼잡 사례 수준)",
         float(min(ranges["closure_days"])), float(max(ranges["closure_days"])), float(base["closure_days"]),
         "항만 집중 대비",
         f"노출 금액 {_man(exposed.min())}~{_man(exposed.max())}"),
        ("closure_share_constant", "중단 중에도 하역 비중·일평균 금액이 정책 후 12개월 평균과 같음", "-", "scenario",
         "단순화 가정", np.nan, np.nan, np.nan, "항만 집중 대비", "성수기에는 노출이 더 크다(월별 변동 미반영)"),
        ("east_ocean_leg", "부산 → 동부 입항 신고 일수(평균·표준편차)", "일", "observed",
         f"CF-1400 × PORT-MIS 선박 연결 {east['n']}건", float(east["p10"]), float(east["p90"]), float(east["mean"]),
         "하역 권역 선택", f"표준편차 {east['sd']:.1f}일(관측). 범위는 10·90% 분위"),
        ("west_ocean_leg", "부산 → 서부 입항 신고 일수(평균·표준편차)", "일", "observed",
         f"CF-1400 × PORT-MIS 선박 연결 {west['n']}건", float(west["p10"]), float(west["p90"]), float(west["mean"]),
         "하역 권역 선택", f"표준편차 {west['sd']:.1f}일(관측). 범위는 10·90% 분위"),
        ("excluded_costs", "운임·터미널 체류·통관·창고비", "-", "not_measured", "공개 데이터에 없음",
         np.nan, np.nan, np.nan, "하역 권역 선택", "재고 일수만 비교한다. 비용 비교에는 운임 자료가 필요"),
    ]
    cols = ["assumption_name", "description", "unit", "evidence_level", "source", "sensitivity_low",
            "sensitivity_high", "base_value", "decision_scope", "effect_on_result"]
    return pd.DataFrame(rows, columns=cols)


def claims(r: dict) -> pd.DataFrame:
    """One row per conclusion, with its evidence level, method, sample and robustness. Reads result tables `r`."""
    wt = r["window_totals"].set_index("metric").change_pct
    uv = r["unit_value_decomposition"].set_index("mode")
    did = r["comparison_did"]
    ms = _row(r["mirror_shift"], "metric", "log_ratio_value")
    gran = r["granularity_decomposition"].set_index("series")
    fx = r["fx_decomposition"].set_index("series").fx_share_of_change
    q5 = r["decision_q5_shift"]
    lw = r["forecast_hierarchy_wape"]
    cov = r["forecast_conformal"].query("level == 0.8").coverage
    now = r["forecast_nowcast"].set_index("model").wape
    port = r["decision_port_disruption"].set_index("closure_days")
    be = r["decision_route_sensitivity"]
    be2 = be[be.inland_sd_days == 2].break_even_inland_days
    within = uv.price_share_of_change
    did_ok = did[did.parallel_trends_ok]
    sel = r["forecast_ml_selection"].query("model == 'ml_selected'")

    rows = [
        ("C01", "정책 후 12개월 미국 수입 금액과 중량이 함께 늘었다",
         f"금액 {_pct(wt['value_usd'])}, 해상+항공 중량 {_pct(wt['measured_weight_kg'])}", "observed",
         "window_totals", "정책 전후 12개월 합계 비교", "12개월 × 2", "해당 없음(집계)", False, "01_data_overview"),
        ("C02", "미국 기록 kg당 단가가 떨어졌고, 하락의 대부분은 같은 품목 안의 단가 변화다",
         f"해상 {_pct(uv.loc['vessel', 'change_pct'])}, 항공 {_pct(uv.loc['air', 'change_pct'])}; "
         f"품목 내 몫 {within.min() * 100:.0f}~{within.max() * 100:.0f}%", "observed",
         "unit_value_decomposition", "구성·가격 항등식 분해(HS10)", "HS10 × 12개월 × 2",
         "항등식 검증 통과", False, "02_unit_value"),
        ("C03", "미국 측 변화는 한국만의 변화로 판정되지 않는다",
         f"지표 {len(did)}개 모두 Holm 보정 p ≥ 0.05(최소 {did.p_holm.min():.2f}); 평행 추세 위반 {int((~did.parallel_trends_ok).sum())}개",
         "estimated", "comparison_did", "이중차분 + 시간 × 국가 위약 순위, Holm·BH",
         f"비교국 {did.controls.str.count(',').max() + 1}개 이하, 위약 최대 {did.n_space_time.max()}개",
         f"부분: 평행 추세 충족 지표 {len(did_ok)}개만 해석", False, "03_cross_country"),
        ("C04", "소포가 미국 통계에 새로 잡혔다는 가설은 지지되지 않는다",
         f"미국/한국 금액 비율 {_pct(ms.pct_change_in_ratio)} (HAC p {ms.p_value:.3f})", "estimated",
         "mirror_shift", "미러 비율의 수준 이동, Newey-West HAC", f"정책 전 {ms.n_pre} · 후 {ms.n_post}개월",
         f"부분: 시간 위약 p {ms.placebo_p:.2f}(최대 전환점 {pd.Timestamp(ms.largest_break):%Y-%m})", False, "04_mirror"),
        ("C05", "같은 흐름이 한국에서는 더 비싸게, 미국에서는 더 싸게 기록됐다",
         f"kg당 금액 한국 {_pct(gran.loc['Korea, HS6', 'change_pct'])}, 미국 {_pct(gran.loc['US, HS6 (sea+air)', 'change_pct'])}",
         "observed", "granularity_decomposition", "양국 통계 HS6 합계의 kg당 금액", "HS6 16개 × 12개월 × 2",
         "세분 수준(HS6·HSK10·HS10)과 무관하게 같은 방향", False, "04_mirror"),
        ("C06", "환율은 미국 측 달러 단가 하락의 일부만 설명한다",
         f"해상 {fx['US imports, vessel'] * 100:.0f}%, 항공 {fx['US imports, air'] * 100:.0f}%, 해상+항공 {fx['US imports, sea+air'] * 100:.0f}%",
         "observed", "fx_decomposition", "달러·원화 단가 항등식 분해", "12개월 × 2", "항등식", False, "02_unit_value"),
        ("C07", "남는 설명은 미국 측 신고 가격(과세가격 산정)의 변화다",
         "C04~C06으로 소포·환율·구성 설명을 제거한 뒤 남는 가설", "hypothesis", "mirror_shift, fx_decomposition",
         "대안 설명 제거", "-", "검증 불가(통관 건별 자료 필요)", False, "04_mirror"),
        ("C08", "정책 전후 해상 리드타임은 변하지 않았다",
         "; ".join(f"{REGION_KO[x.region]} {x['diff']:+.0f}일(95% CI {x.ci_low:+.0f}~{x.ci_high:+.0f})" for _, x in q5.iterrows()),
         "estimated", "decision_q5_shift", "중앙값 차이, 부트스트랩", f"선박 {int(q5.n_pre.sum() + q5.n_post.sum())}건",
         "세 권역 모두 CI가 0 포함", False, "06_decision"),
        ("C09", "외부 흐름 예측은 1개월 앞 직전 값, 3개월 앞 계절 ETS(품목·운송수단별 합)가 가장 정확했다",
         f"합계 WAPE 1개월 {_wape(lw, 'naive', 'base', 1):.3f}, 3개월 {_wape(lw, 'ets_seasonal', 'bottom_up', 3):.3f}",
         "estimated", "forecast_hierarchy_wape", "rolling origin 백테스트, Diebold-Mariano",
         "기준 시점 8개(3개월 앞 6개)", "부분: 3개월 앞 우위만 DM p 0.001, 1개월 앞 p 0.17", False, "05_forecasting"),
        ("C10", "머신러닝 9종을 검증 성적으로 고른 모델은 운영 규칙과 통계적으로 구분되는 개선을 입증하지 못해 운영 규칙(1개월 직전 값, 3개월 계절 ETS 합)을 유지했다(ML 채택 보류)",
         f"[{run_label()}] 1개월 {_wape(lw, 'ml_selected', 'bottom_up', 1):.3f}, 3개월 {_wape(lw, 'ml_selected', 'bottom_up', 3):.3f}; "
         f"부스팅 1개월 {min(_wape(lw, m, 'bottom_up', 1) for m in ('ml_lightgbm', 'ml_xgboost', 'ml_gradient_boosting')):.3f}"
         f"~{max(_wape(lw, m, 'bottom_up', 1) for m in ('ml_lightgbm', 'ml_xgboost', 'ml_gradient_boosting')):.3f}(사후 비교)",
         "estimated", "forecast_hierarchy_wape, forecast_ml_selection", "nested rolling origin, 76개 설정",
         f"기준 시점 {sel.origin.nunique()}개", "부분: 모든 차이 DM p > 0.1, 고정 예측(2026-09~11) 채점 대기", False,
         "05_forecasting"),
        ("C11", "80% 예측 구간은 실제보다 좁다",
         f"실제 포함률 {cov.min() * 100:.0f}~{cov.max() * 100:.0f}%", "estimated", "forecast_conformal",
         "leave-one-origin-out split-conformal", "노드 15개 × 기준 시점", "보정 표본이 작아 불안정", False, "05_forecasting"),
        ("C12", "한국 수출 통계는 미국 수입의 선행 지표로 이득이 없다",
         f"WAPE {now['nowcast_kr_ratio']:.3f} vs 직전 값 {now['naive']:.3f}", "estimated", "forecast_nowcast",
         "1개월 앞 백테스트", "기준 시점 8개", "양국 기록 비율이 이동 중", False, "05_forecasting"),
        ("C13", "해상 수입의 대부분이 LA·롱비치로 들어온다",
         f"LA·롱비치 비중 {port.la_lb_share.iloc[0] * 100:.1f}%(정책 후 12개월 평균)", "observed",
         "decision_port_disruption, port_concentration", "하역항별 해상 금액", "12개월", "해당 없음(집계)", False,
         "06_decision"),
        ("C14", f"LA·롱비치가 2주 멈추면 해상 수입 약 {round(port.loc[14, 'exposed_value_usd'] / 1e6) * 100:,}만 달러가 지연된다",
         f"{_man(port.loc[14, 'exposed_value_usd'])}(7일 {_man(port.loc[7, 'exposed_value_usd'])}, "
         f"30일 {_man(port.loc[30, 'exposed_value_usd'])})", "scenario", "decision_port_disruption",
         "일평균 해상 금액 × 비중 × 중단 일수", "-", "가정: 중단 기간, 비중 일정(assumptions)", False, "06_decision"),
        ("C15", "내륙 운송이 손익분기 이하이면 서부 하역이 동부 직송보다 재고를 덜 묶는다",
         f"손익분기 {be.break_even_inland_days.min():.1f}~{be.break_even_inland_days.max():.1f}일"
         f"(내륙 표준편차 2일이면 {be2.min():.1f}~{be2.max():.1f}일)", "scenario", "decision_route_sensitivity",
         "안전재고 공식, 관측 해상 리드타임 + 가정 내륙 구간", "선박 연결 관측치", "가정: 내륙 일수·변동·수요 변동·서비스 수준",
         False, "06_decision"),
    ]
    cols = ["claim_id", "claim", "value", "evidence_level", "source_table", "method", "sample_size",
            "robustness_status", "causal_claim_allowed", "notebook"]
    return pd.DataFrame(rows, columns=cols)


def recommendations(r: dict, cl: pd.DataFrame) -> pd.DataFrame:
    """Operating rules, each tied to claim ids, with its condition, uncertainty, limitation and next data."""
    lw = r["forecast_hierarchy_wape"]
    sens = r["decision_route_sensitivity"]
    be = sens.break_even_inland_days
    be_sd = sens.groupby("inland_sd_days").break_even_inland_days.agg(["min", "max"])
    port = r["decision_port_disruption"].set_index("closure_days")
    level = cl.set_index("claim_id").evidence_level
    rows = [
        ("단가 해석", "미국 kg당 단가가 떨어졌을 때",
         "미국 단가만으로 '상품이 싸졌다'고 판단하지 않는다. 한국 수출 단가와 함께 보고, 두 기록의 격차를 별도 지표로 관리한다",
         "C02, C05, C06, C07", "격차의 원인은 가설(C07)", "통관 건별 자료 없이 원인 미확정", "통관·과세가격 자료"),
        ("1개월 앞 외부 흐름", "다음 달 미국 수입 흐름을 발주 판단의 참고로 볼 때",
         f"직전 값을 기준선으로 쓴다(합계 WAPE {_wape(lw, 'naive', 'base', 1):.3f}). 정책 충격 직후에는 '작년 같은 달'을 쓰지 않는다"
         f"(WAPE {_wape(lw, 'seasonal_naive', 'base', 1):.3f})",
         "C09, C11", "1개월 앞 순위는 미확정(DM p 0.17), 80% 구간 과소 포함", SCOPE_NOTE, "Census 신규 월, 고정 예측 채점"),
        ("3개월 앞 외부 흐름", "분기 물량·운송 계획의 참고로 볼 때",
         f"계절 ETS를 품목·운송수단별로 예측해 더한다(WAPE {_wape(lw, 'ets_seasonal', 'bottom_up', 3):.3f}). "
         "머신러닝은 고정 예측 채점에서 우위가 확인될 때까지 보류한다",
         "C09, C10, C11", f"ML 선택 모델 {_wape(lw, 'ml_selected', 'bottom_up', 3):.3f}와 구분 불가(DM p 0.87)",
         SCOPE_NOTE, "2026-09~11 고정 예측 채점 결과"),
        ("하역 권역", "동부 수요를 서부 하역 + 내륙 운송으로 채울지 판단할 때",
         f"내륙 운송 일수가 손익분기 이하이면 서부 하역, 넘으면 동부 직송. 손익분기는 내륙 일수의 흔들림에 가장 민감하다"
         f"(표준편차 1일 {be_sd.loc[1, 'min']:.1f}~{be_sd.loc[1, 'max']:.1f}일, 4일 {be_sd.loc[4, 'min']:.1f}~{be_sd.loc[4, 'max']:.1f}일)",
         "C08, C15", f"손익분기 전체 범위 {be.min():.1f}~{be.max():.1f}일(가정 조합)",
         "재고 일수만 비교(운임·터미널 체류·통관 제외)", "운송 업체의 내륙 일수 분포, 운임"),
        ("항만 집중", "성수기 전 서부 창고 재고 수준을 정할 때",
         "LA·롱비치 집중도를 별도 리스크 지표로 관리하고, 2주 중단에 대비한 안전재고 또는 대체 항만을 검토한다",
         "C13, C14", f"노출 {_man(port.loc[7, 'exposed_value_usd'])}~{_man(port.loc[30, 'exposed_value_usd'])}(7~30일)",
         "시나리오: 중단 기간·비중 일정 가정, 대체 항만 전환비 미포함", "항만 혼잡·파업 이력, 대체 항만 운임"),
    ]
    out = pd.DataFrame(rows, columns=["decision", "condition", "recommendation", "evidence", "uncertainty",
                                      "limitation", "next_data"])
    out.insert(4, "evidence_level", [", ".join(sorted({level[c.strip()] for c in e.split(",")})) for e in out.evidence])
    return out
