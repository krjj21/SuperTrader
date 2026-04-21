## 2026-04-13 Codex 분석

한 줄 결론: 저장소 구조는 꽤 잘 나뉘어 있지만, 실제 운영 기준으로 보면 문서와 구현의 불일치, 백테스트와 라이브 경로의 불일치, RL 평가의 비결정성, 테스트/환경 재현성 부족이 핵심 리스크다.

---

## 구조 요약

- `main.py`: `backtest`, `live`, `train`, `retrain`, `status` 진입점
- `src/factors/*`: 유니버스 선정, 팩터 계산, 중립화, 유효성 검증, 복합 점수 계산
- `src/timing/*`: ML/RL 피처, 라벨, 모델 학습, 예측, 재학습
- `src/strategy/*`: `factor_only`, `MACD`, `KDJ`, `ML`, `RL` 전략 래퍼
- `src/broker/*`, `src/risk/*`, `src/db/*`: KIS 연동, 주문/계좌, 리스크, 로컬 DB 기록
- `web/app.py`: Flask 대시보드 API
- `src/notification/*`: Slack/Notion 리포트

---

## 잘된 점

- 모듈 경계가 비교적 명확하다. 데이터, 전략, 브로커, 리스크, DB, 웹이 디렉토리 기준으로 분리되어 있다.
- 라이브 운영에 필요한 안전장치가 기본적으로 들어가 있다. `RiskManager`, 일일 손실 한도, 손절, kill switch, 런타임 상태 저장이 구현돼 있다.
- 백테스트 엔진은 `T일 신호 -> T+1일 시가 체결` 구조를 사용해 기본적인 룩어헤드 문제를 피하려고 설계돼 있다.
- 대시보드와 DB 로그가 붙어 있어 운영 가시성은 나쁘지 않다.

---

## 핵심 발견

### 1. 문서상 `GRPO RL`이지만 실제 구현은 `PPO Actor-Critic`

- `src/timing/rl_agent.py:1-76`은 파일 헤더와 클래스 설명이 모두 PPO Actor-Critic 기준이다.
- `RLTimingModel` 생성자에서 `group_size`, `clip_epsilon_low`, `clip_epsilon_high`는 레거시 호환용으로만 받고 실제 학습은 `clip_epsilon`과 GAE 기반 PPO 업데이트를 사용한다.
- `src/timing/rl_trainer.py:1-68`도 학습 파이프라인을 `PPO 학습 파이프라인`으로 명시한다.
- 영향: Notion 문서와 코드가 같은 알고리즘을 설명하지 않기 때문에 성과 해석과 운영 의사결정이 왜곡될 수 있다.

### 2. 백테스트의 종목 선정 로직이 라이브 팩터 파이프라인과 다름

- `main.py:70-133`의 `_build_pool_history_from_ohlcv()`는 실제로 `6개월 수익률 - 1개월 수익률` 기반 모멘텀 점수만으로 종목풀을 만든다.
- 반면 라이브는 `src/factors/stock_pool.py`에서 `compute_cross_sectional_factors -> neutralize_factor_matrix -> validate_all_factors -> compute_composite_score` 경로를 사용한다.
- 영향: 현재 백테스트 결과는 라이브 팩터 엔진을 검증한 결과가 아니라, 단순 모멘텀 종목선정 + 타이밍 전략 결과에 더 가깝다.

### 3. RL 검증과 재학습 교체 기준이 비결정적

- `src/timing/rl_trainer.py:21-60`의 `evaluate_rl_agent()`는 평가 시 `agent.collect_episode()`를 사용한다.
- `src/timing/rl_agent.py:62-67`의 `sample_action()`은 확률적으로 action을 샘플링한다.
- 즉, 같은 모델이라도 검증 Sharpe가 실행마다 달라질 수 있고, `src/timing/retrain.py`의 RL 교체 기준도 이 노이즈를 그대로 받는다.
- 영향: 주간 재학습에서 모델 교체 여부가 실력 향상보다 샘플링 운에 더 좌우될 수 있다.

### 4. 설정 드리프트 존재

- `config/settings.yaml:144-146`에는 `notification.slack.enabled`가 있다.
- 하지만 `src/config.py:185-195`의 `AppConfig`에는 `notification` 섹션이 없다.
- 영향: 설정 파일에 적힌 값이 실제로는 로드되지 않으며, 운영자가 설정이 적용된다고 오해할 수 있다.

### 5. 테스트와 실행 환경 재현성이 약함

- `tests/` 아래에는 사실상 `tests/__init__.py`만 있고 자동 테스트가 없다.
- 현재 bash 환경에서는 `python3`는 존재하지만, 저장소에 포함된 `venv/Scripts/python.exe`는 WSL에서 `UtilBindVsockAnyPort` 오류로 실행되지 않았다.
- 영향: 코드 변경 후 안정성을 빠르게 검증하기 어렵고, 실행 환경이 Windows/WSL 혼합 상태라 유지보수 비용이 커진다.

---

## 운영 관점 메모

- `main.py`의 라이브 루프는 `strategy._pool` 같은 private 속성에 직접 접근한다. 당장 고장 원인은 아니지만 전략 구현 교체 시 결합도가 높다.
- `weekly_retrain()`의 메시지는 RL에도 `accuracy`, `F1`라는 분류 모델 용어를 그대로 사용한다. RL에서는 사실상 Sharpe를 `new_f1` 자리에 넣고 있어 운영 로그 의미가 흐려진다.
- `src/notification/notion_reporter.py`는 Notion 검색 API로 기존 페이지를 찾고 본문을 전부 삭제 후 다시 추가한다. 데이터량이 커지면 속도와 실패 복구 측면에서 취약할 수 있다.

---

## 우선순위 제안

1. 문서와 구현을 맞추기
현재 코드를 기준으로 문서를 `PPO RL`로 정정할지, 아니면 실제로 GRPO를 구현할지 먼저 결정해야 한다.

2. 백테스트와 라이브의 종목 선정 경로를 통일하기
최소한 백테스트도 `src/factors/stock_pool.py`와 동일한 팩터 파이프라인을 사용해야 결과 해석이 가능하다.

3. RL 평가를 결정적으로 바꾸기
검증 시에는 샘플링 대신 `argmax` 정책 또는 고정 시드 평가를 사용해 Sharpe 비교를 재현 가능하게 만드는 것이 맞다.

4. 최소 테스트를 추가하기
우선순위는 `config 로드`, `백테스트 체결 시점`, `리스크 매니저 주문 검증`, `RL 평가 결정성` 순서가 적절하다.

5. 실행 환경을 표준화하기
Windows 전용 가상환경을 저장소에 두기보다 `requirements.txt` 또는 `pyproject.toml` 기반으로 다시 만들 수 있어야 한다.

---

## 확인한 검증 제약

- `python -m pytest -q`는 현재 쉘에 `python`이 없어 바로 실행되지 않았다.
- 저장소 내 `venv/Scripts/python.exe`는 현재 bash/WSL 환경에서 실행 실패했다.
- 따라서 이번 분석은 코드 정적 검토 중심이며, 자동 테스트 통과 여부까지는 확인하지 못했다.
