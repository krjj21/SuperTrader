# 2026-04-18 SuperTrader 코드 분석

한 줄 결론: SuperTrader는 팩터 기반 종목 선정, ML/RL 타이밍, KIS 실매매, 리스크 관리, 대시보드를 한 저장소에 통합한 자동매매 시스템이다. 현재 구조는 운영에 필요한 큰 골격을 갖췄지만, 실매매 기준으로는 재현 가능한 검증, 설정/문서 정합성, 시크릿/산출물 관리, 전략-엔진 결합도 개선이 우선순위다.

## 분석 범위

- 진입점: `main.py`
- 설정: `config/settings.yaml`, `src/config.py`
- 데이터/팩터: `src/data/*`, `src/factors/*`
- 전략/타이밍 모델: `src/strategy/*`, `src/timing/*`
- 백테스트: `backtest/*`
- 실매매/리스크/DB: `src/broker/*`, `src/risk/*`, `src/db/*`
- 대시보드: `web/app.py`, `web/templates/*`

이번 분석은 정적 코드 검토 중심이다. 외부 API, KIS 계좌, 데이터 공급자, 학습 모델 실행은 호출하지 않았다.

## 전체 아키텍처

`main.py`가 CLI 진입점이며 `backtest`, `live`, `train`, `retrain`, `status` 다섯 모드를 제공한다. 설정은 YAML 기반 `AppConfig`와 `.env` 기반 `Secrets`로 분리되어 있고, 주요 런타임 파라미터는 `config/settings.yaml`에서 제어된다.

백테스트 경로는 유니버스 구성, OHLCV 일괄 로드, 리밸런싱 날짜 생성, 팩터 기반 종목풀 이력 생성, 모델 준비, 전략 비교 순서로 흐른다. 핵심 엔진인 `PortfolioBacktestEngine`은 날짜별로 전일 주문을 다음 거래일 시가에 체결하고, 리밸런싱, 손절, 타이밍 시그널, 포지션 동기화, 일일 평가금액 기록을 수행한다.

라이브 경로는 KIS 클라이언트, 주문/계좌 관리자, 리스크 관리자, Slack/Notion 리포터, 전략 인스턴스를 초기화한다. 시작 시 팩터 종목풀을 만들고, 장중에는 주기적으로 계좌 조회, 손절, 전략 시그널 생성, LLM 검증, 주문 실행, DB 기록을 수행한다. 대시보드는 같은 SQLite DB와 `data/current_pool.json`을 읽어 계좌 상태, 종목풀, 시그널, 수익률, 학습 로그를 보여준다.

## 핵심 흐름

### 1. 종목 선정

`src/factors/stock_pool.py`의 `build_stock_pool()`이 유니버스, 크로스섹션 팩터, 업종/시총 중립화, 유효 팩터 선택, 복합 점수 산출, 상위 N개 선정을 담당한다. `src/factors/calculator.py`는 설정에 따라 `alpha101`, `alpha158`, 또는 두 모듈 결합을 선택한다.

백테스트도 `main._build_pool_history_factor_based()`를 통해 같은 `build_stock_pool()` 경로를 사용한다. 이전 분석에서 지적된 단순 모멘텀 기반 종목풀 문제는 현재 코드 기준으로는 해소된 상태다.

### 2. 타이밍 모델

`src/timing/trainer.py`는 모델 타입별 생성과 학습을 통합한다. 전통 ML 계열은 `build_features()`와 `generate_labels()`를 사용하고, RL은 `src/timing/rl_trainer.py`의 별도 PPO 학습 파이프라인으로 위임된다.

`src/timing/predictor.py`는 모든 모델을 `predict()` 또는 `predict_with_position()` 인터페이스로 감싼다. `factor_ml`, `factor_rl`, `factor_hybrid` 전략은 이 예측기를 통해 타이밍 판단을 수행한다.

### 3. 전략 계층

모든 전략은 `BaseStrategy.generate_signal()`을 구현하고 `TradeSignal`을 반환한다. `factor_only`는 리밸런싱 중심, `factor_macd`/`factor_kdj`는 기술 지표 중심, `factor_ml`은 분류 모델 중심, `factor_rl`은 PPO RL 중심, `factor_hybrid`는 XGBoost 알파와 RL 리스크 필터를 결합한다.

현재 라이브 전략 선택기는 `factor_hybrid`도 지원한다. 전략 비교도 XGBoost와 RL 모델이 모두 있을 때 하이브리드 전략을 추가한다.

### 4. 실매매와 리스크 관리

`KISClient`는 OAuth 토큰 캐싱, REST 호출, rate limit, 재시도를 담당한다. `OrderManager`는 현금 매수/매도, 취소, 체결 조회를 감싼다. `AccountManager`는 잔고, 보유 포지션, 실현손익을 조회하며 모의투자에서는 DB 기반 fallback을 쓴다.

`RiskManager`는 장 운영 시간, 주말, kill switch, 일일 손실 한도, 종목별 손절, 포지션 사이징, 주문 검증을 담당한다. 라이브 루프는 손절을 LLM 검증 없이 우선 실행하고, BUY/SELL 전략 시그널은 LLM 검증 후 주문한다.

### 5. 운영 가시성

`src/db/models.py`는 거래 로그, 일일 손익, 포지션 스냅샷, 시그널 로그, 런타임 상태, 보유 포지션 매수일을 저장한다. `web/app.py`는 계좌 상태, 파이프라인 상태, 최근 시그널, 누적 수익률, 팩터 종목풀, 지표 차트, 학습 로그를 API로 제공한다.

## 잘된 점

- 백테스트와 라이브 모두 팩터 기반 종목풀 경로를 공유하도록 정리되어 있어 결과 해석 가능성이 좋아졌다.
- 백테스트 엔진이 `T일 판단 -> T+1일 시가 체결` 구조를 사용해 기본적인 룩어헤드 위험을 줄인다.
- PPO RL 전략은 실제 보유 포지션, 미실현 손익, 영업일 기준 보유일을 모델 입력에 반영한다.
- 실매매 루프에 손절, 일일 손실 한도, kill switch, 중복 매수 방지, 런타임 상태 저장이 들어가 있다.
- 대시보드가 DB 우선, 로그 fallback, 인코딩 fallback을 갖고 있어 Windows/WSL 혼합 환경의 운영 이슈를 일부 흡수한다.

## 주요 리스크

### 1. 저장소 위생과 시크릿 노출 위험

`.gitignore`에는 `__pycache__`, `*.pyc`, `logs`, `models`, `data` 성격의 산출물을 막으려는 의도가 있지만, 현재 git 추적 목록에는 `__pycache__`, `logs/supertrader.log`, `models/*.pkl`이 이미 포함되어 있다. 또한 원격 URL에 개인 액세스 토큰이 포함된 형태로 설정되어 있다. 운영 저장소 기준으로는 가장 먼저 정리해야 할 위험이다.

권장 조치:
- 원격 URL에서 토큰 제거 후 credential manager 또는 GitHub CLI 인증으로 전환
- 노출된 토큰은 즉시 폐기/재발급
- 추적 중인 `__pycache__`, 로그, 로컬 모델, DB 산출물 제거
- `.gitignore`에 `logs/`, `data/`, `reports/` 정책을 명확히 반영

### 2. 작업 트리에 운영 산출물과 코드 변경이 섞여 있음

현재 작업 트리에는 코드 변경, 로그, DB, 토큰 캐시, 보고서, pyc 파일이 함께 섞여 있다. 이런 상태에서 `git add .`로 커밋하면 운영 민감 정보와 불필요한 바이너리/로그가 같이 올라갈 수 있다.

권장 조치:
- 커밋 단위를 코드 변경, 문서, 실험 결과로 분리
- 운영 데이터와 모델 파일은 Git LFS 또는 별도 artifact 저장소로 분리
- PR 전 `git status --short`와 `git diff --cached` 확인을 필수화

### 3. 설정 모델과 YAML의 정합성 부족

`config/settings.yaml`에는 `notification.slack.enabled`가 있지만 `src/config.py`의 `AppConfig`에는 `notification` 모델이 없다. Pydantic 기본 동작상 extra 필드 처리가 명시되어 있지 않아 버전/설정에 따라 무시되거나 검증 에러가 될 수 있다.

권장 조치:
- `NotificationConfig`, `SlackConfig`, `NotionConfig`를 명시적으로 추가
- 사용하지 않는 설정은 YAML에서 제거
- 설정 로드 단위 테스트 추가

### 4. 라이브 루프와 전략 구현의 결합도가 높음

`main.py`의 라이브 루프는 전략의 `_pool` 같은 private 속성에 직접 접근한다. 지금은 작동하더라도 새 전략이 같은 내부 필드를 갖지 않으면 실매매 루프가 깨질 수 있다.

권장 조치:
- `BaseStrategy`에 `update_pool()`, `get_pool()`, `sync_positions()` 선택 인터페이스를 공식화
- 라이브 루프는 private 속성 대신 공개 메서드만 사용
- 전략별 capability를 명시적으로 검사

### 5. LLM 검증 모델명과 공급자 의존성

`SignalValidator.DEFAULT_MODEL`은 Anthropic API에 직접 전달되는 문자열이다. 모델명이 실제 API에서 지원되지 않으면 모든 LLM 검증이 실패하고, 현재 코드는 실패 시 시그널을 그대로 실행한다. 이 fallback은 가용성에는 좋지만, "LLM 검증이 켜져 있다"는 운영 기대와 다를 수 있다.

권장 조치:
- 모델명을 설정 파일로 이동
- 시작 시 LLM health check 추가
- LLM 실패 시 `fail-open`/`fail-closed` 정책을 설정으로 선택
- 대시보드에 최근 LLM 실패율 표시

### 6. 재현 가능한 테스트 기반이 약함

현재 자동 테스트는 사실상 없다. 백테스트 체결 시점, 손절/리밸런싱 우선순위, RL 포지션 동기화, DB 기록, 설정 로드는 모두 회귀 위험이 큰 영역이다.

권장 조치:
- `config` 로드 테스트
- `PortfolioBacktestEngine`의 T+1 체결 테스트
- 리밸런싱 퇴출/진입 테스트
- `RiskManager`의 손실 한도/포지션 사이징 테스트
- `FactorRLStrategy.sync_positions()`와 매수일 추적 테스트

## 우선순위 제안

1. 원격 URL 토큰 제거, 토큰 폐기/재발급, 산출물 추적 제거
2. 설정 모델과 YAML 정합성 맞추기
3. 전략 공개 인터페이스 정리로 라이브 루프 결합도 낮추기
4. 백테스트 엔진과 리스크 매니저 최소 단위 테스트 추가
5. LLM 검증 정책을 설정화하고 실패율을 대시보드에 노출
6. 모델/로그/DB artifact 관리 정책 정리

## 실행/검증 메모

- 외부 API 호출, 모델 학습, 백테스트 실행은 수행하지 않았다.
- 이 문서는 코드 정적 검토 결과이며, 성과 수치나 실계좌 동작을 보증하지 않는다.
- 현재 워크트리에 기존 사용자 변경분과 산출물이 많아, 이 문서 커밋 시에는 문서 파일만 선별해서 stage하는 것이 안전하다.
