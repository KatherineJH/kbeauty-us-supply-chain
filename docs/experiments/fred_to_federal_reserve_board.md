# 환율 출처 전환: FRED에서 Federal Reserve Board(G.5) 직접 수집으로

## 배경

최초 분석은 FRED를 통해 받은 Federal Reserve Board G.5 월평균 환율 7개 계열을 썼다.
- 수집: `pipeline/fetch_fred_fx.py`, 2026-11-02, `fredgraph.csv`
- 보관: `data/raw/fred/`

공개 전 이용조건을 검토하면서 FRED 이용약관(https://fred.stlouisfed.org/legal/)을 확인했다.
- 다운로드는 "solely for your personal, non-commercial use"로 한정한다.
- 금지 조항 (p)는 저장, 아카이브, 데이터베이스 편입을 금지한다.
- AI·ML 개발과 연결된 사용도 금지한다.

반면 원 제공기관인 연준 이사회는 출처를 밝히면 자유롭게 복사·배포할 수 있다고 밝힌다(https://www.federalreserve.gov/disclaimer.htm): "Unless otherwise indicated, information on Board's website is in the public domain and may be copied and distributed without permission. Please cite to the Board as the source of the information."

그래서 이사회 Data Download Program(DDP)에서 직접 받도록 바꾸기로 했다. 이 문서는 전환에 앞서 두 출처의 값이 같은지 비교한 기록이다.

(1~4절: 전환 전 비교, 5절 이후: 전환 실행과 검증)

## 비교 대상

- **Board:** DDP 사전 구성 패키지 "G.5 Statistical Release - Monthly rates"(csv, 최근 120개 관측치)
  - 받은 날짜: 2026-11-06
  - URL: `https://www.federalreserve.gov/datadownload/Output.aspx?rel=H10&series=c5d6e0edf324b2fb28d73bcacafaaa02&lastobs=120&from=&to=&filetype=csv&label=include&layout=seriescolumn&type=package`
  - 24,877 bytes, SHA-256 앞자리 `06159035f0388750`
  - 기간: 2016-11~2026-10. 최근 공개일은 2026-11-02(G.5/H.10)
- **FRED:** 저장소의 `data/raw/fred/<ID>.csv`
  - 수집 시각: 2026-11-02T16:08 KST(`data/raw/fred/manifest.csv`)

## 계열 대응

| FRED ID | Board 고유 식별자 | Board 설명 | Board 단위·통화 | 방향 | 변환 |
|---|---|---|---|---|---|
| EXKOUS | `H10/H10/RXI_N.M.KO` | KOREA -- SPOT EXCHANGE RATE, WON/US$ | Currency: Per USD, KRW | 달러당 원 | 그대로 |
| EXUSEU | `H10/H10/RXI$US_N.M.EU` | SPOT EXCHANGE RATE - EURO AREA | Currency: Per EUR, USD | **유로당 달러** | 그대로. 저장소가 `1/v`로 달러당 유로 계산 |
| EXCAUS | `H10/H10/RXI_N.M.CA` | CANADA -- SPOT EXCHANGE RATE, CANADIAN $/US$ | Per USD, CAD | 달러당 캐나다달러 | 그대로 |
| EXCHUS | `H10/H10/RXI_N.M.CH` | CHINA -- SPOT EXCHANGE RATE, YUAN/US$ | Per USD, CNY | 달러당 위안 | 그대로 |
| EXMXUS | `H10/H10/RXI_N.M.MX` | MEXICO -- SPOT EXCHANGE RATE, PESOS/US$ | Per USD, MXN | 달러당 페소 | 그대로 |
| EXUSUK | `H10/H10/RXI$US_N.M.UK` | United Kingdom -- Spot Exchange Rate, US$/Pound Sterling Reciprocal of rxi_n.m.uk | Currency: Per GBP, USD | **파운드당 달러** | 그대로. 저장소가 `1/v`로 달러당 파운드 계산 |
| EXJPUS | `H10/H10/RXI_N.M.JA` | JAPAN -- SPOT EXCHANGE RATE, YEA/US$ | Per USD, JPY | 달러당 엔 | 그대로 |

- **역수 처리:** 유로와 파운드는 Board도 FRED와 같은 방향(외화 1단위당 달러, `RXI$US_` 계열)으로 공표한다. 그래서 수집 단계에서 역수를 취하지 않는다. 기존처럼 추출 단계에서만 `1/v`로 바꾼다.
  - 확인: Board 값과 FRED 값의 역수를 비교하면 최대 차이가 0.42(EUR), 0.70(GBP)이다. 역수 관계가 아니라는 뜻이다.
- **파운드 Multiplier 표기:** Board CSV에서 파운드 열(`RXI$US_N.M.UK`)의 `Multiplier`가 `0.00999999978`로 적혀 있다.
  - 실제 값(예: 2024-01 1.2701)은 파운드당 달러이고 FRED와 일치한다.
  - 메타데이터 표기 오류로 보이므로 수집기는 Multiplier를 적용하지 않는다. 변경 시 테스트로 고정한다.

## 비교 결과 (공통 기간 2016-11 ~ 2026-09, 119개월)

| FRED ID | FRED 보관 기간 | 결측(FRED·Board) | 최대 절대 차이 | 최대 상대 차이 | 불일치 월 |
|---|---|---|---|---|---|
| EXKOUS | 1981-04 ~ 2026-09 | 0 · 0 | 0 | 0 | 없음 |
| EXUSEU | 1999-01 ~ 2026-09 | 0 · 0 | 0 | 0 | 없음 |
| EXCAUS | 1971-01 ~ 2026-09 | 0 · 0 | 0 | 0 | 없음 |
| EXCHUS | 1981-01 ~ 2026-09 | 0 · 0 | 0 | 0 | 없음 |
| EXMXUS | 1993-11 ~ 2026-09 | 0 · 0 | 0 | 0 | 없음 |
| EXUSUK | 1971-01 ~ 2026-09 | 0 · 0 | 0 | 0 | 없음 |
| EXJPUS | 1971-01 ~ 2026-09 | 0 · 0 | 0 | 0 | 없음 |

- **소수 자릿수:** Board는 소수 넷째 자리까지 끝자리 0을 그대로 쓴다(예: `1359.2090`). FRED는 끝자리 0을 지운다(예: `1359.209`). 수치는 같고 raw 파일의 바이트만 다르다.
- **개정값:** FRED 수집(2026-11-02)과 Board 수집(2026-11-06) 사이에 공통 기간의 값이 바뀐 월은 없다.
- **기간 차이:** Board 패키지에는 2026-10이 있고, FRED 보관본에는 없다. FRED 수집 시각이 Board의 2026-11-02 공표보다 앞섰기 때문이다.

## 분석 결과에 미치는 영향

저장소의 추출 규칙(2023-01부터, 유로·파운드는 `round(1/v, 6)`)을 Board 값에 그대로 적용했다.

| 추출 범위 | 행 수 | 커밋된 `data/extracted/fx_monthly.csv`와 비교 |
|---|---|---|
| 2023-01 ~ 2026-09 | 315 | **바이트 단위 동일**(SHA-256 앞자리 `6b3624713144`) |
| 2023-01 ~ 2026-10(상한 없음) | 322 | 다름(2026-10 7행 추가) |

- 추출 상한을 현재 분석 종료월(2026-09)에 맞추면 추출 CSV가 같다. 따라서 하류 결과표 31개와 frozen forecast에는 영향이 없을 것으로 본다.
  - 이 판단은 전환 후 파이프라인을 다시 실행하고 canonical 비교로 확인한다.
- 수집기에는 `--end`(기본값: 현재 분석 종료월)를 추가해, 공표된 다음 달 값이 끼어들지 않게 한다.

## 수집 경로 결정: DDP 패키지가 아니라 릴리스 페이지 XML

전환 작업 중 Board의 공식 공지(https://www.federalreserve.gov/data/data-download-fred-information.htm, 2026-07-16)를 확인했다.

> "the Board plans to remove the "Build Your Package" option from the Data Download Program the week of November 9, 2026. Historical data will remain available for download as XML files on statistical release pages."

- 같은 공지에 따르면 사전 구성 패키지도 이후 제거되고, DDP 자체가 폐지될 예정이다.
- 그래서 7개 계열만 받는 custom package 대신, Board가 계속 제공하겠다고 밝힌 **H.10 릴리스 페이지 XML**을 원문으로 쓴다.
  - 파일: `https://www.federalreserve.gov/releases/h10/data/FRB_h10_xml.zip`(약 2.1MB)
  - 이 파일에는 G.5/H.10 전체 계열이 들어 있다. 저장소는 위 7개 월간 계열(FREQ="129")만 쓰고, 그 목록은 `data/raw/federal_reserve_board/series.csv`에 남긴다.
- 같은 날 G.5 릴리스 페이지의 XML 링크(`/releases/g5/data/FRB_g5_xml.zip`)는 404를 반환했다.
- 릴리스 XML의 7개 계열을 FRED 보관본과 다시 비교했다.
  - 공통 전 기간(KRW 1981-04~, EUR 1999-01~, CAD·GBP·JPY 1971-01~, CNY 1981-01~, MXN 1993-11~, 모두 2026-09까지)의 불일치 월: **0**
  - 분석 창(2023-01~2026-09)의 관측치 상태: 모두 `OBS_STATUS="A"`
  - 파운드 계열의 `UNIT_MULT="0.00999999978"`도 XML 속성으로 그대로 들어 있다.

## 전환 실행 (2026-11-06)

| 항목 | 내용 |
|---|---|
| 수집기 | `pipeline/fetch_fed_board_fx.py`. 릴리스 XML zip을 그대로 보관하고, `--offline`이면 보관본만 읽는다 |
| 원문 | `data/raw/federal_reserve_board/FRB_h10_xml.zip`, 2,086,452 bytes, SHA-256 `1c312d2fc1de99f174c26178a2cb1f0f0f6a4a08b6dce25b043603fec9a3f24a`, 수집 2026-11-06T11:50 |
| manifest | `manifest.csv`: URL, 수집 시각, 크기, SHA-256<br>`series.csv`: 7개 계열 ID, 통화, 빈도, 단위, 공표 UNIT_MULT, 적용 UNIT_MULT(1), 방향, 변환식, 이전 FRED ID |
| 기간 고정 | `kbeauty.paths.FX_DATA_START="2023-01"`, `FX_DATA_END="2026-09"`. 보관 XML에 2026-10이 있어도 추출하지 않는다 |
| 파운드 Multiplier | 적용하지 않는다. `tests/test_pipeline.py::test_offline_extraction_is_byte_identical`이 공표 값 `0.00999999978`, 2024-01 값 `1.2701`, 역수 변환, 상한 월을 함께 확인한다 |
| dbt | `stg_fred__fx_monthly` → `stg_fed_board__fx_monthly`(FRED에 종속된 이름이라 변경). 결과표에는 영향이 없고, 데이터 사전과 계보도의 표기만 바뀐다 |
| 삭제 | 새 수집기와 테스트가 통과한 뒤 `pipeline/fetch_fred_fx.py`와 `data/raw/fred/`(7개 CSV, manifest)를 추적 파일에서 제거했다(Git 이력 정리는 별도 계획) |
| 해시 목록 | `data/SHA256SUMS` 93개 → 88개(FRED 8개 제거, Board 3개 추가). `data/extracted/fx_monthly.csv`의 해시는 그대로다 |

## 검증 결과

| 검사 | 결과 |
|---|---|
| `fx_monthly.csv` | 커밋된 파일과 바이트 단위 동일(315행, SHA-256 앞자리 `6b3624713144`) |
| `run_all.py` | 해시 88개 일치, 오프라인 재추출 바이트 일치, dbt 49/49, 표 31·그림 18 생성 |
| 결과표 31개 | HEAD와 바이트 단위 동일(`compare_tables.py --exact`) |
| frozen forecast | 파일 2개와 해시 모두 변경 없음 |
| Linux CI | 전환 커밋 실행 결과로 기록한다(아래) |

### frozen forecast의 출처 기록

2026-11-02 고정 예측(`reports/frozen/`)은 **FRED를 통해 받은 환율 값으로 생성했다**. 그 출처를 Board로 소급해 바꾸지 않는다.
- 생성에 쓴 값(FRED 보관본, 2023-01~2026-09)은 Board 릴리스 XML의 값과 완전히 같다.
- 추출 파일 `fx_monthly.csv`는 바이트 단위로 같다.
- 사전 등록에는 2026-11-06 정정 행으로 기록했다.
