# dbt 빌드마다 mart 마지막 비트가 달라지는 문제: 스레드 실험

`docs/experiments/linux_determinism.md` 6절의 후속이다. 같은 머신에서 dbt를 다시 빌드하면 일부 mart의 값이 마지막 비트에서 달라졌다. 이 차이가 스레드 설정에서 오는지 짧게 확인했다.

## 방법

- workflow `diag-dbt-threads`, 실행 ID **42003050322**. 설정 3개 × runner 5대 = 15 job.
- 각 job은 고정 수치 환경에서 dbt 빌드와 보고서 생성을 두 번(A, B) 했다. 매번 빈 데이터베이스에서 시작했다. 합계 30회.
- 매 빌드마다 기록한 것:
  - 모델 21개의 순서 무관 해시. 실수는 비트 그대로 해시했다.
  - 31개 결과표의 바이트 해시
  - dbt 빌드 시간과 보고서 생성 시간
- 스크립트: `scripts/diag_dbt_threads.py`

| 설정 | 내용 |
|---|---|
| `current` | 커밋된 프로필: dbt `threads: 4`, DuckDB 자체 스레드는 기본값 |
| `dbt1` | dbt `threads: 1`. 모델을 한 번에 하나씩 만든다 |
| `duckdb1` | DuckDB `settings: threads: 1`. 쿼리 안의 병렬 집계를 끈다. dbt threads는 그대로 |

사용자가 요청한 비교는 `current`와 `dbt1`이다. `duckdb1`은 실제 원인 후보(DuckDB 내부 병렬 집계)를 직접 확인하려고 덧붙였다.

## 결과

| 설정 | 빌드 | 서로 다른 mart 지문 | 같은 job A/B가 다른 job | 값이 달라진 모델 | 서로 다른 결과표 지문 | dbt 시간(중앙값) |
|---|---|---|---|---|---|---|
| current | 10 | 4 | 5개 중 3 | dim_port, mart_mirror_monthly, mart_port_concentration_monthly | 1 | 5.2초 |
| dbt1 | 10 | 8 | 5개 중 4 | 위 3개 + mart_category_monthly, mart_country_panel_monthly, mart_port_region_monthly | 1 | 6.2초 |
| **duckdb1** | 10 | **1** | **5개 중 0** | **없음** | 1 | 8.9초 |

- **dbt `threads: 1`은 해결책이 아니다.** 오히려 지문이 더 많이 갈렸다. dbt threads는 동시에 만드는 모델 수일 뿐이다. 쿼리 안에서 DuckDB가 병렬로 더하는 순서는 그대로 남는다.
- **DuckDB `threads: 1`은 해결한다.** 10회 빌드의 mart 지문이 CPU 3종(EPYC 7763, Xeon 8370C 포함)에서 모두 같았다.
- **하위 결과표는 세 설정 모두 30회 전부 같았다**(지문 `1aa93e797157`). mart의 마지막 비트 차이는 지금 결과에 전달되지 않는다. 이 값은 AVX-512 머신을 포함해 고정 수치 환경의 결과가 runner에 무관하다는 추가 증거이기도 하다.
- **비용**: dbt 빌드가 약 5초에서 약 9초로 늘었다(1회 빌드 기준 +4초). 보고서 생성 시간은 runner 성능에 따라 85~170초로 달랐고, 이 설정과는 무관했다.

## 결정

`warehouse/profiles.yml`에 DuckDB `settings: threads: 1`을 적용한다. dbt `threads: 4`는 그대로 둔다.

- 결과표를 바꾸지 않는다. 바꾸지 않는다는 것은 이 실험과 CI의 canonical 비교로 확인한다.
- 대신 숨은 위험을 없앤다. 나중에 mart 값을 반올림하거나 비교하는 분석이 추가되더라도, 빌드마다 결과가 흔들리지 않는다.
- 비용은 빌드당 약 4초다.
