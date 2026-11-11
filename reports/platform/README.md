# 실행 환경별 결과표: 공식 기준과 진단 이력

예측 표 7개와 `claims.csv`(C10이 ML 수치를 인용)는 수치 환경에 따라 값이 달라진다.

원인은 GitHub runner가 실행마다 다른 CPU를 받는 것이었다(`docs/experiments/linux_determinism.md`).
- AVX-512가 있는 CPU에서는 numpy가 AVX-512 경로를, OpenBLAS가 SkylakeX 커널을 쓴다. 그래서 마지막 비트가 달라진다.
- ETS·SARIMAX·ridge·MLP가 그 차이를 키운다.

| 폴더 | 구분 | 내용 |
|---|---|---|
| `linux/` | **공식 기준** | Linux 고정 환경(스레드 1, OpenBLAS Haswell, numpy AVX-512 비활성화)의 31개 결과표 전체. `MANIFEST.json`에 생성 기록이 있다. CI는 같은 환경에서 다시 만든 31개 표가 이 폴더와 바이트 단위로 같아야 통과한다 |
| `windows/` | 진단 이력 | 작성 환경(Windows)의 민감 표 8개. 노트북의 숫자가 이 환경에서 나왔다(`reports/tables/`와 같은 값) |
| `linux_ci_run1/` | 진단 이력 | 고정 전 Linux CI, 커밋 27dde55의 실행(Haswell 계열 CPU). 예측 표 7개는 공식 기준과 바이트 단위로 같다 |
| `linux_ci_run2/` | 진단 이력 | 고정 전 Linux CI, 커밋 fbac8bf의 실행(AVX-512 계열 CPU). 같은 이미지와 같은 `uv.lock`을 썼는데도 1회차와 달랐다 |

`MANIFEST.json`에 남기는 생성 기록:
- 생성 커밋, workflow 실행, 생성 시각
- runner: CPU 모델, AVX-512 노출 여부, 이미지
- 패키지·BLAS·numpy SIMD 버전
- 고정한 환경 변수
- 표마다 SHA-256

## 원칙

- 진단 이력 폴더는 지우거나 고치지 않는다. 공식 기준을 바꿀 때는 이유를 커밋 메시지와 이 문서에 남긴다.
- 공개 전 고정 예측(`reports/frozen/`)과 사전 등록의 기존 행은 고치지 않는다.
- 허용 오차를 넓혀 실행 간 차이를 감추지 않는다. 공식 기준과의 비교는 바이트 단위다.
- 덱·README·모델 카드의 공식 비교값은 `linux/`에서 읽는다.
- 작성 환경 값(예: ML 검증 선택 1개월 앞 0.138)은 진단 이력으로 함께 밝힌다.
