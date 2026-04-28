"""SuperTrader 실행 모드 모듈.

main.py 의 4가지 모드를 분리해 유지보수성을 높임:
  · backtest.py — 전략 비교 백테스트
  · live.py     — 실시간 자동매매 (KIS API)
  · training.py — 모델 학습 / 재학습
  · status.py   — 계좌 현황 조회
"""
