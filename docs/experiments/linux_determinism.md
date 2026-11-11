# Linux 실행 간 결정성 진단

진단 브랜치 `claude/diag-linux-determinism-f7a274` 전용 문서다. main에는 반영하지 않았다.
`reports/frozen/`, 사전 등록, 모델 선택 규칙, 허용 오차, 민감 표 목록은 건드리지 않았다.

## 요약

- **원인은 같은 머신에서의 비결정성이 아니라 runner의 CPU 차이다.** GitHub의 `ubuntu-24.04` runner는 실행마다
  다른 CPU에 배정된다(관측: AMD EPYC 7763, 9V74, 9V45, Intel Xeon Platinum 8370C, 8573C).
  - numpy 2.5 Linux 휠은 AVX-512를 지원하는 CPU에서 exp/log/power 등에 AVX-512(X86_V4) 코드 경로를 쓴다.
  - 같은 CPU에서 OpenBLAS(scipy-openblas)는 SkylakeX 커널을 고른다.
  - 그래서 AVX-512가 없는 EPYC 7763(Haswell 커널)과는 마지막 비트가 다르다.
- **같은 머신에서 두 번 실행하면 예측·ML 단계는 25개 job 모두 비트 단위로 같았다**(입력 테이블 3~4개의 마지막 비트 차이는 아래 6절). 지금 코드의 단일 스레드·고정 시드 설계는 한 머신 안에서는 지켜진다.
- **첫 분기 지점:** 입력·특징 열 해시는 CPU가 달라도 같다. 처음 갈라지는 곳은 통계 모델 단계다.
  - 위치: 첫 기준 시점(2025-12) 합계 시계열의 ETS(감쇠 추세).
  - 과정: statsmodels가 brute-force로 잡은 시작값이 3.6e-15만큼 다르고, 이 차이가 L-BFGS-B를 거치며 커진다.
  - ML 쪽: ridge fold 예측이 학습 데이터 해시가 같은데도 다르다. BLAS/LAPACK 커널 차이다.
  - 전체로는 ETS·SARIMAX·ridge·elastic net·kNN·MLP 예측이 모두 달라지고, ML 선택은 2026-04 1개월에서 처음 뒤집힌다(ridge ↔ MLP).
- **최소 수정안(검증됨):** 아래 5개 환경 변수를 CI job에 고정한다.
  - `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`를 각각 1로 둔다.
  - `OPENBLAS_CORETYPE=Haswell`
  - `NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"`
- **수정안의 증거:**
  - 고정한 실행 10회와 같은 설정의 EPYC 7763 실행 6회를 합쳐 CPU 모델 4종에서 16회가 나왔다.
  - 이 16회는 입력 단계를 뺀 모든 단계 기록(약 4,500개)과 31개 표의 바이트가 전부 같았다.
  - 하나라도 빼면 CPU 종류에 따라 결과가 둘로 갈렸다.
- **추가 검증(8절):** 고정 환경으로 `run_all.py`와 재실행 검사를 runner 3대(7763, 9V74, 9V45)에서 돌렸다. 3대 모두 통과했고 31개 표가 같았다.
- **기존 관측과의 대응:**
  - 고정 후의 예측 표 7개는 `reports/platform/linux_ci_run1/`(Haswell 계열 실행)과 바이트 단위로 같다.
  - `linux_ci_run2/`는 AVX-512 계열이다.
  - Windows 작성 환경은 두 계열 모두와 다르다(1e-4 수준).

## 1. 방법

| 항목 | 내용 |
|---|---|
| 진단 스크립트 | `scripts/diag_determinism.py`: `report.generate_all`(실제 코드 경로)을 한 번 실행하면서 단계별 결과를 SHA-256으로 기록. `src/`는 바꾸지 않고 실행 중 함수만 감싼다 |
| 단계 | 00 커널 프로브 → 10 입력 로드 → 20 시계열 → 30 ML 특징 → 40 통계 모델 호출(시계열·기준 시점별) → 45 통계 백테스트 표 → 50 ML fold(설정×h×예측 월) → 60 ML 검증 선택 → 70 계층 정합 → 80 합성통제 → 90 표 파일 |
| 비교 | `scripts/diag_compare.py`: 실행을 지문(fingerprint)으로 묶고, 기준 실행과 처음 다른 기록을 파이프라인 순서로 찾는다 |
| 반복 | workflow `diag-determinism`, 실행 ID **41993240397**. 5개 설정 × 5회 = 25 job이고, job마다 dbt 빌드부터 파이프라인 전체를 두 번(A, B) 돌렸다. 합계 50회 |
| 비교 축 | 같은 job의 A와 B = 같은 머신의 재실행. 같은 설정의 다른 job = 같은 이미지·다른 VM(CPU가 다를 수 있음) |
| 로컬 점검 | Windows(Intel Family 6 Model 154, AVX2): 진단 hook을 단 상태에서도 31개 표가 커밋된 표와 바이트 단위로 같았다(hook이 결과를 바꾸지 않음) |

설정(누적해서 하나씩 추가했고, numpy만 따로 시험한 설정을 하나 더 두었다):

| 설정 | 환경 |
|---|---|
| `baseline` | 고정 없음(현재 CI) |
| `threads1` | `OMP/OPENBLAS/MKL_NUM_THREADS=1` |
| `threads1-oblas` | + `OPENBLAS_CORETYPE=Haswell` |
| `threads1-oblas-npy` | + `NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"` |
| `npy` | numpy SIMD 제한만 |

## 2. 실행별 환경 기록

공통 환경:
- OS: Linux 7.0.0-1017-azure, glibc 2.39, 이미지 ubuntu24 20261026.354.1, Python 3.12.3. vCPU는 2개(lscpu 기준 1 core × 2 threads).
- 라이브러리: numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, statsmodels 0.15.0, pandas 3.1.0, lightgbm 4.7.0, xgboost 3.4.1, joblib 1.6.0, threadpoolctl 3.7.0, duckdb 2.0.0, dbt-core 1.12.6 / dbt-duckdb 1.11.1.
- BLAS: numpy와 scipy가 각자 `libscipy_openblas`를 하나씩 싣는다(0.3.34.106.0, 0.3.31.dev).
  - 기본 스레드 수는 2다.
  - joblib 작업자 안에서는 `threadpool_limits(1)` 때문에 1이다. 작업자의 threadpoolctl 기록으로 확인했다.
- numpy SIMD: baseline은 X86_V2이고, 런타임 dispatch 후보는 X86_V3, X86_V4, AVX512_ICL, AVX512_SPR이다. Windows 휠의 후보는 X86_V3뿐이다.
- 실행마다 남긴 원자료: `lscpu` 전문, `/proc/cpuinfo` 플래그 해시, `numpy.show_config()`·`show_runtime()`, threadpoolctl(주 프로세스와 작업자), 스레드 환경 변수, 라이브러리 버전. 각 job 산출물 `diag-<설정>-r<n>`의 JSON `env`에 있다.

job별 결과(A와 B의 CPU는 같다):
- "프로브 지문": 커널 프로브 26개 결과.
- "표 지문": 31개 CSV의 바이트.
- "A/B 비입력 차이": 입력 단계를 뺀 단계 기록 가운데 A와 B가 다른 개수.
- "A/B 입력 차이": 입력 테이블 기록 가운데 A와 B가 다른 개수.

| 설정 | 회차 | CPU | AVX-512 노출 | OpenBLAS 커널 | numpy X86_V4 | 프로브 지문 | 표 지문 | A/B 표 | A/B 비입력 차이 | A/B 입력 차이 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | r1 | AMD EPYC 9V74 | yes | SkylakeX | on | `6006b07a4ad1` | `d397fc24c41e` | same | 0 | 1 |
| baseline | r2 | AMD EPYC 7763 | no | Haswell | off | `35d2e84d48aa` | `ebf76fa744cb` | same | 0 | 0 |
| baseline | r3 | Intel Xeon Platinum 8370C | yes | SkylakeX | on | `6006b07a4ad1` | `d397fc24c41e` | same | 0 | 1 |
| baseline | r4 | AMD EPYC 9V74 | yes | SkylakeX | on | `6006b07a4ad1` | `d397fc24c41e` | same | 0 | 2 |
| baseline | r5 | Intel Xeon Platinum 8573C | yes | SkylakeX | on | `6006b07a4ad1` | `d397fc24c41e` | same | 0 | 2 |
| threads1 | r1 | Intel Xeon Platinum 8370C | yes | SkylakeX | on | `51e6dcac0775` | `37a740490a00` | same | 0 | 1 |
| threads1 | r2 | AMD EPYC 9V74 | yes | SkylakeX | on | `51e6dcac0775` | `37a740490a00` | same | 0 | 2 |
| threads1 | r3 | AMD EPYC 7763 | no | Haswell | off | `b41dbaf1f1de` | `4b218a0848ef` | same | 0 | 2 |
| threads1 | r4 | AMD EPYC 7763 | no | Haswell | off | `b41dbaf1f1de` | `4b218a0848ef` | same | 0 | 2 |
| threads1 | r5 | AMD EPYC 7763 | no | Haswell | off | `b41dbaf1f1de` | `4b218a0848ef` | same | 0 | 2 |
| threads1-oblas | r1 | AMD EPYC 9V74 | yes | Haswell | on | `608eaeec9892` | `434cd53b86b1` | same | 0 | 1 |
| threads1-oblas | r2 | AMD EPYC 9V74 | yes | Haswell | on | `608eaeec9892` | `434cd53b86b1` | same | 0 | 1 |
| threads1-oblas | r3 | AMD EPYC 9V45 | yes | Haswell | on | `608eaeec9892` | `434cd53b86b1` | same | 0 | 1 |
| threads1-oblas | r4 | AMD EPYC 9V45 | yes | Haswell | on | `608eaeec9892` | `434cd53b86b1` | same | 0 | 2 |
| threads1-oblas | r5 | AMD EPYC 9V45 | yes | Haswell | on | `608eaeec9892` | `434cd53b86b1` | same | 0 | 2 |
| npy | r1 | Intel Xeon Platinum 8370C | yes | SkylakeX | off | `e41b50f55b7d` | `d397fc24c41e` | same | 0 | 1 |
| npy | r2 | Intel Xeon Platinum 8370C | yes | SkylakeX | off | `e41b50f55b7d` | `d397fc24c41e` | same | 0 | 2 |
| npy | r3 | AMD EPYC 9V45 | yes | SkylakeX | off | `e41b50f55b7d` | `d397fc24c41e` | same | 0 | 2 |
| npy | r4 | AMD EPYC 7763 | no | Haswell | off | `35d2e84d48aa` | `ebf76fa744cb` | same | 0 | 2 |
| npy | r5 | AMD EPYC 7763 | no | Haswell | off | `35d2e84d48aa` | `ebf76fa744cb` | same | 0 | 2 |
| threads1-oblas-npy | r1 | AMD EPYC 9V74 | no | Haswell | off | `b41dbaf1f1de` | `4b218a0848ef` | same | 0 | 1 |
| threads1-oblas-npy | r2 | Intel Xeon Platinum 8370C | yes | Haswell | off | `b41dbaf1f1de` | `4b218a0848ef` | same | 0 | 3 |
| threads1-oblas-npy | r3 | AMD EPYC 9V74 | yes | Haswell | off | `b41dbaf1f1de` | `4b218a0848ef` | same | 0 | 2 |
| threads1-oblas-npy | r4 | Intel Xeon Platinum 8573C | yes | Haswell | off | `b41dbaf1f1de` | `4b218a0848ef` | same | 0 | 2 |
| threads1-oblas-npy | r5 | Intel Xeon Platinum 8370C | yes | Haswell | off | `b41dbaf1f1de` | `4b218a0848ef` | same | 0 | 2 |

참고: CPU 모델명만으로는 계열을 정할 수 없다. 같은 EPYC 9V74도 VM에 따라 AVX-512 플래그 노출 여부가 달랐다(baseline r1은 노출, threads1-oblas-npy r1은 미노출). CI는 모델명이 아니라 실제로 쓰인 커널(threadpoolctl `architecture`, numpy `__cpu_features__`)을 기록해야 한다.

## 3. 고정 전후 재현성 비교

다섯 설정 모두 5회씩 반복했다. 같은 설정 안에서 서로 다른 표 지문이 몇 개 나왔는지 센다.

| 설정 | job | 서로 다른 표 지문 | 지문을 가른 요인 | 결정성 회복 |
|---|---|---|---|---|
| baseline | 5 | **2** (`d397…` AVX-512 계열 4회, `ebf7…` 7763 1회) | CPU 계열 | 아니오 |
| threads1 | 5 | **2** (`37a7…` AVX-512 계열 2회, `4b21…` 7763 3회) | CPU 계열 | 아니오 |
| threads1-oblas | 5 | 1 (`434c…`), 단 5회 모두 AVX-512 노출 머신 | numpy X86_V4가 켜져 있음. 7763 계열(`4b21…`)과 다름 | 아니오(AVX-512 없는 머신과 다름) |
| npy | 5 | **2** (`d397…` 3회, `ebf7…` 2회) | OpenBLAS 커널(SkylakeX ↔ Haswell) | 아니오 |
| **threads1-oblas-npy** | 5 | **1** (`4b21…`, 10회 모두) | 없음 | **예**: 9V74(AVX-512 노출·미노출 각 1대), 8370C 2대, 8573C 1대 |

교차 확인:
- `threads1-oblas-npy` 10회와 `threads1`의 EPYC 7763 6회(원래 AVX-512도 없고 Haswell 커널인 머신)는 프로브 지문 `b41dbaf1f1de`와 표 지문 `4b218a0848ef`가 같다.
- 입력 단계를 뺀 모든 단계 기록도 16회 전부 같다.
- 프로브로 각 변수의 역할이 구분된다.
  - `np.log/exp/log1p/expm1/power`는 numpy X86_V4가 켜졌는지로만 갈린다. 스레드와 OpenBLAS 설정은 무관하다.
  - `np.dot`, `linalg.solve/lstsq`, `scipy.lu_solve`, L-BFGS-B, sklearn MLP는 OpenBLAS 커널로 갈린다.
  - 스레드 고정은 주 프로세스의 BLAS 결과에 영향을 준다. 같은 7763에서도 baseline과 threads1의 `comparison_synth`가 달랐다(SLSQP 5회 호출 모두 다름). 주 프로세스는 기본 2스레드이고, ML 작업자만 1스레드로 제한되어 있다.
- 세 변수는 각각 필요하다. 이번 실행으로 증명한 것은 세 개를 함께 쓴 조합이다. `MKL_NUM_THREADS`와 `OMP_NUM_THREADS`는 따로 떼어 시험하지 않았다. MKL은 쓰이지 않고, LightGBM/XGBoost는 이미 `n_jobs=1`이라 영향이 없을 것으로 보지만 확인하지는 않았다.

## 4. 첫 분기 지점

baseline-r1-A(AMD 9V74, AVX-512 계열)와 baseline-r2-A(AMD 7763, Haswell 계열)를 비교했다. 다른 계열 쌍도 결과가 같다.

| 단계 | 결과 |
|---|---|
| 10 입력 로드 | 예측에 쓰이는 `fct_imports_hs10_mode_monthly`, `mart_monthly_summary`, `stg_fred__fx_monthly`는 같음(6절의 dbt 차이와 별개) |
| 20 시계열 | 합계·8개 하위 시계열·KRW/USD, 순서와 값 모두 같음 |
| 30 특징 | h=1,2,3 특징 프레임의 열 순서(`y, lag0, lag1, lag2, lag_season, post, log_fx, series, month_of_year, node`)·행 순서·값 모두 같음 |
| **40 통계 모델(첫 분기)** | `ets_damped`, 합계 시계열, 기준 시점 2025-12. brute-force 시작값 x0 차이 최대 3.6e-15, 목적함수 0.451260200914778 / …784, 반복 16/16, 적합된 `smoothing_trend` 5.565e-10 / 5.581e-10, 예측 상대 차이 1.2e-10 |
| 40 (확대) | `ets_seasonal`: 같은 시작값 차이(7.1e-15)에서 반복 수가 40/39 … 69/59로 달라지고, 합계 시계열 예측 차이가 2.5e-6에서 1.4e-4로 커진다. 15개 노드 전체에서 계열 간 최대 상대 차이는 ets_seasonal 1.2e-2(기존 관측 "약 1%"와 일치), ets_damped 3.7e-5, SARIMAX 1.5e-5. naive 계열 세 모델은 0 |
| 45 통계 백테스트 표 | `ets_damped`, 2025-12부터 다름 |
| 50 ML fold | 첫 차이는 `h1, ridge[alpha=0.1;level;short], 2025-10`. 학습·시험 데이터 해시는 같고 예측만 다르다(BLAS 커널). 다른 fold 수: ridge 334, MLP 264(전부), elastic net 86, kNN 61. 트리 모델(RF, ExtraTrees, GBM, LightGBM, XGBoost)은 0 |
| 60 ML 선택 | 첫 차이는 2026-04 h1 `ml_selected`: ridge[alpha=0.1;diff;full](inner WAPE 0.13684) ↔ mlp[alpha=1.0;diff;short](0.13377) |
| 70 계층 정합 | ETS·SARIMAX를 쓰는 기록 189개와 naive 계열 OLS 정합 25개가 다름. naive 계열은 base 예측이 같고, OLS 정합 단계의 `np.linalg.solve`가 BLAS 커널을 따라간다 |
| 80 합성통제 | SLSQP 해가 마지막 비트에서 다름(목적함수 표시 자릿수 안에서는 같음) |
| 90 표 | 표 10개가 다름: 예측 표 7개, `claims.csv`, `comparison_synth.csv`, `comparison_synth_weights.csv`. 나머지 21개는 같음 |

결론: MLP만의 문제가 아니다. 출발점은 통계 모델(ETS)과 선형 ML(ridge)의 최하위 비트 차이다. ETS의 brute-force 시작값 탐색과 L-BFGS-B, MLP의 L-BFGS가 이 차이를 키운다. 키운 결과는 표 반올림(12자리)으로도 흡수되지 않는다.

## 5. MLP L-BFGS 수렴 기록

MLP fold는 264개다(설정 8개 × h 3개 × 예측 월). 각 fold의 `n_iter_`, 최종 loss, scipy `OptimizeResult`의 status·message·nfev를 기록했다.

| 계열 | 수렴 실패(최대 반복 2000 도달) | `n_iter_` 중앙값 | 종료 메시지 |
|---|---|---|---|
| AVX-512/SkylakeX 계열(20회) | 29 | 658 | REL_REDUCTION_OF_F 174, PGTOL 61, ITERATIONS REACHED LIMIT 29 |
| Haswell 계열·고정 후(30회) | 30 | 706 | REL_REDUCTION_OF_F 163, PGTOL 71, ITERATIONS REACHED LIMIT 30 |
| Windows 로컬(1회) | 28 | 677 | REL_REDUCTION_OF_F 167, PGTOL 69, ITERATIONS REACHED LIMIT 28 |

- 같은 계열 안에서는 50회 모두 fold 단위로 반복 수·loss·메시지가 같았다. 비결정성은 없었다.
- 계열 간에는 같은 fold의 반복 수 차이가 중앙값 108, 최대 1,151회다.
- 23개 fold는 한쪽 계열에서만 수렴했다(12개는 AVX-512 계열에서만, 11개는 Haswell 계열에서만).
- 즉 MLP는 비트 차이를 가장 크게 키우는 모델이지만 차이의 출발점은 아니다. 약 11%의 fold가 반복 상한에서 멈춘다는 점은 고정 여부와 별개로 남는 품질 정보다.

## 6. 따로 발견한 것: dbt 빌드마다 일부 mart의 마지막 비트가 다르다

- 같은 머신에서 dbt 빌드를 다시 하면 일부 mart의 값이 달라진다. 순서와 무관한 해시로도 다르므로 행 순서가 아니라 값이 다르다.
- 대상은 `mart_country_panel_monthly`(25 job 중 24), `mart_mirror_monthly`, `mart_port_concentration_monthly`, `mart_category_monthly`(1회)이고, 모든 설정에서 나타났다.
- DuckDB가 실수 합계를 병렬로 집계하면서 덧셈 순서가 바뀌는 것으로 보인다(추정이며, 열 단위로 확인하지는 않았다).
- 이 차이는 50회 모두 하위 단계에 전달되지 않았다. 입력 단계를 뺀 기록과 31개 표는 같은 계열 안에서 전부 같았다.
- 지금 결과에는 영향이 없지만, 상류에 숨은 위험이다. 이번 최소 수정 범위에는 넣지 않았다(후속 과제 참고).

## 7. 최소 수정안과 CI 초안

1. **환경 고정**(`.github/workflows/ci.yml`의 `reproduce` job `env`, 진단 브랜치에만 반영):
   ```yaml
   OMP_NUM_THREADS: "1"
   OPENBLAS_NUM_THREADS: "1"
   MKL_NUM_THREADS: "1"
   OPENBLAS_CORETYPE: Haswell
   NPY_DISABLE_CPU_FEATURES: X86_V4 AVX512_ICL AVX512_SPR
   ```
   - 코드와 모델은 바꾸지 않는다.
   - 2절의 모든 runner 계열이 AVX2/FMA를 지원하므로 Haswell 커널과 X86_V3가 공통분모다.
   - AVX-512가 없는 머신에서 numpy가 "not part of the dispatched optimizations" 경고를 낼 수 있지만, 동작에는 문제가 없다(EPYC 7763에서 확인).
   - 비용: AVX-512 머신에서 numpy·BLAS가 느려진다. 이번 실행의 1회 시간은 고정 전후 모두 90~210초였고, 차이는 머신 종류가 좌우했다.
2. **같은 job 재실행 검사**(`scripts/check_rerun_determinism.py`, `ci.yml` 마지막 단계):
   - 파이프라인 뒤에 새 프로세스로 진단 패스를 두 번 더 돌리고, 입력부터 표까지의 단계 해시가 하나라도 다르면 실패한다.
   - 두 패스는 같은 warehouse를 쓴다. 그래서 6절의 dbt 차이는 이 검사 대상이 아니다.
   - 한계: 한 머신 안의 비결정성만 잡고, 다른 runner와의 차이는 잡지 못한다. runner 사이의 차이는 1번의 고정이 막는다.
3. **검사 동작 확인:**
   - 로컬에서 같은 기록 두 개를 넣으면 통과하고(exit 0), 한 기록을 일부러 바꾸면 실패하며 해당 ETS 기록과 최적화기 상태를 출력한다(exit 1).
   - Linux runner 3대에서 고정 환경으로 돌린 결과는 아래 8절에 적는다.

## 8. 고정 환경의 재실행 검사(Linux)

workflow `diag-rerun-check`, 실행 ID **41995368829**.
- 고정 환경에서 `ci.yml`과 같은 순서로 실행했다: `run_all.py`를 돌린 뒤 `check_rerun_determinism.py`를 돌린다.
- 독립 runner 3대에서 반복했다.

| job | CPU | 재실행 검사 | 31개 표 지문 |
|---|---|---|---|
| rerun (1) | AMD EPYC 7763 | OK: 단계 기록 4,531개가 두 패스에서 같음 | `4b218a0848ef` |
| rerun (2) | AMD EPYC 9V74 | OK: 4,531개 같음 | `4b218a0848ef` |
| rerun (3) | AMD EPYC 9V45 | OK: 4,531개 같음 | `4b218a0848ef` |

- 표 지문은 3절의 고정 결과(`4b21…`)와 같다.
- 3절에서는 고정 설정으로 시험하지 못한 9V45도 이번에 같은 지문을 냈다. 그래서 고정 후 같은 결과를 확인한 CPU는 7763, 9V74, 9V45, 8370C, 8573C의 5종이다.
- 이 실행의 예측 표 7개는 `reports/platform/linux_ci_run1/`의 파일과 **바이트 단위로 같다**.
- 고정 후의 주요 값:
  - `ml_selected` 합계 WAPE(bottom-up): 1개월 0.109, 2개월 0.126, 3개월 0.144
  - MLP 3개월: 0.589
  - 계절 ETS(합산 전) 1개월: 0.140
  - DM p값: ML 비교 2건이 0.914와 0.870, 사후 비교는 0.33 이상
- 운영 권고와 "기준선 대비 신뢰할 만한 개선을 입증하지 못했다"는 결론은 그대로다.

참고: 두 패스는 같은 warehouse를 쓰므로 6절의 dbt 차이는 검사에 들어오지 않는다. runner 사이의 단계 지문(pipeline fp)은 `e0e7…`과 `08bb…`로 달랐는데, 이는 runner마다 dbt를 새로 빌드한 입력 단계의 비트 차이 때문이다. 표는 같았다.

## 9. 사용자 결정이 필요한 점

1. **수정안을 main CI에 넣을지.** 넣으면 Linux CI의 기준값은 "Haswell 커널 + numpy X86_V3" 계열의 값이 된다.
   - 이 값의 예측 표 7개는 `reports/platform/linux_ci_run1/`과 바이트 단위로 같다(8절). Windows 작성 환경과는 다르다(1e-4 수준).
   - `compare_tables.py`가 가리키는 canonical 경로(`reports/platform/linux/`, 지금 없음)에 어떤 파일을 둘지는 원래 세션이 정해야 한다. 이 세션은 덱·문서 소유 범위를 건드리지 않았다.
2. **Windows 작성 환경도 같은 계열로 맞출지.**
   - Windows 휠의 numpy는 원래 X86_V3까지만 쓰고, 로컬 OpenBLAS는 Haswell 커널이었다.
   - 그런데도 표 지문이 Linux 고정 계열과 다르다. Windows의 libm·컴파일러 차이로 보이며, 이번 범위에서는 확인하지 않았다.
   - 노트북·덱 숫자의 출처를 Linux 기준으로 옮길지는 문서 소유 세션이 결정할 일이다.
3. **dbt 입력의 비트 차이(6절)를 지금 막을지.** 지금 결과에는 영향이 없다. 막는다면 dbt-duckdb `threads: 1` 또는 mart 반올림이 후보이며, 별도 작업으로 하는 편이 낫다.
4. **MLP 반복 상한 도달 fold 약 11%.** 결정성 문제와는 별개의 품질 정보다. 모델 선택 규칙과 사전 등록은 바꾸지 않았다.

## 재현 방법

```bash
uv run python scripts/diag_determinism.py --build --out diag/run.json --label local-r1-A
uv run python scripts/diag_compare.py diag --md diag/summary.md
uv run python scripts/check_rerun_determinism.py
```

Windows에서는 loky 작업자 풀이 리다이렉트된 표준 핸들 아래에서 멈출 수 있다. 이때는 진단 스크립트에 `--serial`을 붙인다.
