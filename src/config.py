"""
설정 관리 모듈
- YAML 파일에서 전략/리스크/팩터/타이밍 설정 로드
- .env 파일에서 API 키/시크릿 로드
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ──────────────────────────────────────────────
# .env 기반 시크릿 설정
# ──────────────────────────────────────────────
class Secrets(BaseSettings):
    kis_app_key: str = Field(default="", alias="KIS_APP_KEY")
    kis_app_secret: str = Field(default="", alias="KIS_APP_SECRET")
    kis_account_no: str = Field(default="", alias="KIS_ACCOUNT_NO")
    kis_hts_id: str = Field(default="", alias="KIS_HTS_ID")
    slack_bot_token: str = Field(default="", alias="SLACK_BOT_TOKEN")
    slack_channel: str = Field(default="#auto-trading", alias="SLACK_CHANNEL")
    slack_trade_channel: str = Field(default="#super_trader_buy_sell", alias="SLACK_TRADE_CHANNEL")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    notion_token: str = Field(default="", alias="NOTION_TOKEN")
    notion_database_id: str = Field(default="", alias="NOTION_DATABASE_ID")

    model_config = {"env_file": "config/.env", "extra": "ignore"}


# ──────────────────────────────────────────────
# YAML 기반 설정 모델
# ──────────────────────────────────────────────
class KISConfig(BaseModel):
    base_url: str = "https://openapivts.koreainvestment.com:29443"
    is_virtual: bool = True
    token_refresh_hours: int = 20


class UniverseConfig(BaseModel):
    market: str = "ALL"
    min_market_cap: int = 100_000_000_000
    min_avg_volume: int = 100_000
    exclude_sectors: list[str] = []
    # Mid-cap rank 필터 (둘 다 >0 일 때만 활성화). 시총 desc rank, 1=최대시총.
    # 예: cap_rank_min=30, cap_rank_max=150 → 시총 30~150위 (120종목)
    # rank 활성화 시 min_market_cap 절대 필터는 비활성 (rank 기반으로 대체)
    cap_rank_min: int = 0
    cap_rank_max: int = 0


class FactorConfig(BaseModel):
    factor_module: str = "alpha101"   # alpha101, alpha158, both
    rebalance_freq: str = "biweekly"  # biweekly, monthly, quarterly
    top_n: int = 30
    composite_method: str = "ic_weighted"
    ic_lookback: int = 12
    min_ir: float = 0.3
    neutralize_industry: bool = True
    neutralize_market_cap: bool = True
    # 투자자 매매 사전 필터 (build_stock_pool 진입 시 적용)
    foreign_filter_enabled: bool = False  # True 시 sappo_investor_trading 활용
    foreign_filter_pct: float = 0.5       # 누적 순매수 상위 N% 통과 (0.5 = 상위 50%)
    foreign_filter_lookback: int = 20     # 누적 일수 (영업일)
    # 모드: foreign(외국인만) | foreign_organ(외국인+기관 합산, B1)
    # 한국 시장 3축(외국인/기관/개인)에서 기관까지 묶으면 신호 표면적이 늘어난다.
    investor_filter_mode: str = "foreign"


class MACDParams(BaseModel):
    fast: int = 12
    slow: int = 26
    signal: int = 9


class KDJParams(BaseModel):
    n: int = 9
    m1: int = 3
    m2: int = 3


class DTParams(BaseModel):
    max_depth: int = 8
    min_samples_leaf: int = 20


class GBParams(BaseModel):
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8


class LSTMParams(BaseModel):
    sequence_length: int = 20
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 64


class TransformerParams(BaseModel):
    sequence_length: int = 30
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 0.0005
    epochs: int = 100
    batch_size: int = 64


class RLParams(BaseModel):
    learning_rate: float = 0.0003
    gamma: float = 0.99
    group_size: int = 8
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float = 0.28
    entropy_coeff: float = 0.03
    epochs_per_update: int = 4
    episodes: int = 200
    hidden_dim: int = 256
    buy_action_threshold: float = 0.13
    sell_action_threshold: float = 0.06
    # SAPPO: sentiment 가중치 (0 = baseline PPO, 논문 권장 0.1)
    sentiment_lambda: float = 0.0
    # SAPPO: sentiment 소스 — off/news/mock/xgb_proxy
    sentiment_source: str = "off"
    # Hybrid: XGB SELL 신뢰도 ≥ threshold 일 때만 RL 보류 무시 매도 허용
    xgb_sell_confidence_threshold: float = 0.60
    # Hybrid: XGB BUY 신뢰도 ≥ threshold 일 때만 진입 허용 (false positive 차단)
    # 잘못된 매수 = 즉시 손실, 잘못된 보류 = 기회 비용에 그쳐 BUY 가 더 보수적이어야 함
    xgb_buy_confidence_threshold: float = 0.55
    # Hybrid (transformer alpha): softmax 분포가 XGB 보다 평탄해 동일 0.55/0.60 임계값
    # 사용 시 980/1059 시그널이 통과해 RL 게이팅 무력화. 이를 보정하기 위한 별도 임계값.
    # 첫 추정치: BUY 0.70, SELL 0.75 (XGB 의 ~5% 통과율 근사 목표)
    transformer_buy_confidence_threshold: float = 0.70
    transformer_sell_confidence_threshold: float = 0.75
    # Ensemble: N개 시드로 학습 후 median Sharpe 모델 채택. 빈 리스트 → 단일 시드(42)
    # GPU 시간 N배 비용으로 시드 변동 ±0.3~0.5 회귀 위험 큰 폭 감소
    ensemble_seeds: list[int] = []
    # Profit-aware adaptive SELL threshold (#3).
    # 보유 중 unrealized_pnl 이 floor 이상이면 ml_sell_threshold 를 선형 인하해 익절 발화율 ↑.
    # XGB p_sell 이 추세장에서 0.50 부근에 갇혀 0.60 게이트를 못 넘는 구조 보정.
    profit_aware_sell_enabled: bool = True
    profit_aware_sell_pnl_floor: float = 0.10      # 이 이하면 변경 없음
    profit_aware_sell_max_discount: float = 0.20   # 최대 인하 비율 (예: 0.20 → 0.60 × 0.80 = 0.48)
    profit_aware_sell_pnl_ceiling: float = 0.30    # 이 이상에서 max_discount 적용


class TimingConfig(BaseModel):
    model: str = "xgboost"
    forward_days: int = 5
    buy_threshold: float = 0.02
    sell_threshold: float = -0.02
    macd: MACDParams = MACDParams()
    kdj: KDJParams = KDJParams()
    decision_tree: DTParams = DTParams()
    gradient_boost: GBParams = GBParams()
    lstm: LSTMParams = LSTMParams()
    transformer: TransformerParams = TransformerParams()
    rl: RLParams = RLParams()


class StrategyParams(BaseModel):
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    atr_multiplier: float = 2.0


class StrategyConfig(BaseModel):
    name: str = "factor_xgb"
    params: StrategyParams = StrategyParams()


class RiskConfig(BaseModel):
    max_position_pct: float = 0.05
    max_total_positions: int = 30
    daily_loss_limit_pct: float = 0.05
    stop_loss_pct: float = 0.07
    max_order_retries: int = 3
    # Confidence-proportional sizing: True 일 때 포지션 슬롯에 strength 를 곱해
    # 강한 시그널은 더 큰 슬롯을 받는다. Hybrid 의 strength 는 0.5~1.0 (XGB
    # predict_proba 기반) 이며 다른 전략은 고정값이므로 활성화 시 의미를 확인할 것.
    # mode:
    #   "clamp" — mult = clamp(strength, min, max)  (기본/하향 사이징)
    #   "scale" — mult = min + (max-min) × ((strength-0.5)/0.5)  (선형 매핑, 상향용)
    # min/max 가 0.5/1.0 일 때 clamp 와 scale 은 동일하게 동작한다(하위 호환).
    confidence_sizing_enabled: bool = False
    confidence_sizing_mode: str = "clamp"
    confidence_sizing_min_mult: float = 0.5
    confidence_sizing_max_mult: float = 1.0
    # E1 변동성 진입 게이트: BUY 신호 ATR/price 가 임계 이상이면 거부.
    # stop_loss(-7%) 이전에 변동성 자체로 진입을 차단해 -7% 손절 빈도를 줄인다.
    # SELL 은 절대 차단하지 않는다 (잘못된 매도 차단 = 손실 누적).
    atr_filter_enabled: bool = False
    atr_filter_max_pct: float = 0.05
    atr_filter_period: int = 14
    # C: 부분 익절 + 트레일링 스탑 (SELL 신호 시 100% 매도 → 수익 보호 분할).
    # 1차: SELL 트리거 시 partial_take_profit_pct 만큼 매도 (잔여 보유 유지).
    # 2차: 잔여는 peak (보유 후 최고가) 대비 trailing_stop_pct 하락 시 전량 매도.
    # OFF 시 기존 동작 (SELL = 100% 매도).
    partial_take_profit_enabled: bool = False
    partial_take_profit_pct: float = 0.50  # 1차 익절 비율 (0.50 = 50%)
    trailing_stop_pct: float = 0.03         # peak 대비 -3% 하락 시 잔여 매도
    # E: LLM SELL 보류 조건 강화 — RSI 단독 강한 모멘텀 보호.
    # 현재 보류 5조건 (MA20위 AND MA5위 AND RSI≥50 AND 1d≥-1% AND 5d≥+1%) 외에,
    # RSI ≥ rsi_strong_momentum_threshold 단독 충족 시 SELL *보류*.
    # OFF 시 기존 5조건 룰만 적용.
    llm_strong_momentum_hold_enabled: bool = False
    llm_strong_momentum_rsi_threshold: float = 80.0


class ScheduleConfig(BaseModel):
    pre_market: str = "08:30"
    market_open: str = "09:00"
    market_close: str = "15:30"
    post_market: str = "15:40"
    check_interval_sec: int = 300
    rebalance_day: int = 1


class BacktestConfig(BaseModel):
    start_date: str = "2018-01-01"
    end_date: str = "2024-12-31"
    initial_capital: int = 100_000_000
    commission_rate: float = 0.00015
    tax_rate: float = 0.0023
    # 슬리피지 (per leg) — mid-cap 시초가 호가 한 단계 ~0.1~0.3% 가정
    slippage_pct: float = 0.0015
    # 손절 갭다운 패널티 — 종가 -7% 트리거 → T+1 시가 평균 추가 갭다운
    # 한국시장 갭다운 평균 ~1.5% (보수적 가정)
    stop_loss_gap_penalty_pct: float = 0.015
    # walk-forward: 백테스트 중에 모델을 새로 학습해야 할 때
    # 전체 OHLCV 의 앞 train_ratio 만 학습에 사용. (1.0 이면 기존처럼 전체)
    train_ratio: float = 0.5


class DatabaseConfig(BaseModel):
    path: str = "data/trading.db"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "logs/supertrader.log"
    rotation: str = "10 MB"
    retention: str = "30 days"


class CodexConfig(BaseModel):
    """Codex CLI 기반 자동 리뷰 설정."""
    enabled: bool = False            # 라이브 데일리/백테스트 완료 시 자동 리뷰 트리거
    daily_review: bool = True        # daily_report 이후 codex daily 실행
    model: str = ""                  # codex 모델 override (빈 문자열이면 기본값)


class RegimeConfig(BaseModel):
    """시장 국면(regime) 감지 설정.
    enabled=False: 매일 라벨 산출/기록만, 매매 영향 없음 (dark-launch).
    lambda=0: enabled여도 가중치/사이즈 모듈레이션 비활성.
    """
    enabled: bool = False
    lambda_: float = Field(default=0.0, alias="lambda")
    hmm_lookback_days: int = 60
    hmm_n_states: int = 3
    llm_override_threshold: float = 0.3
    news_fetch_enabled: bool = True

    model_config = {"populate_by_name": True}


class AppConfig(BaseModel):
    kis: KISConfig = KISConfig()
    universe: UniverseConfig = UniverseConfig()
    factors: FactorConfig = FactorConfig()
    timing: TimingConfig = TimingConfig()
    strategy: StrategyConfig = StrategyConfig()
    risk: RiskConfig = RiskConfig()
    schedule: ScheduleConfig = ScheduleConfig()
    backtest: BacktestConfig = BacktestConfig()
    database: DatabaseConfig = DatabaseConfig()
    logging: LoggingConfig = LoggingConfig()
    codex: CodexConfig = CodexConfig()
    regime: RegimeConfig = RegimeConfig()


# ──────────────────────────────────────────────
# 설정 로더
# ──────────────────────────────────────────────
def load_config(config_path: str = "config/settings.yaml") -> AppConfig:
    global _config
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        _config = AppConfig(**raw)
    else:
        _config = AppConfig()
    return _config


def load_secrets() -> Secrets:
    return Secrets()


_config: AppConfig | None = None
_secrets: Secrets | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_secrets() -> Secrets:
    global _secrets
    if _secrets is None:
        _secrets = load_secrets()
    return _secrets
