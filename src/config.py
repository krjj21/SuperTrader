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


class FactorConfig(BaseModel):
    factor_module: str = "alpha101"   # alpha101, alpha158, both
    rebalance_freq: str = "biweekly"  # biweekly, monthly, quarterly
    top_n: int = 30
    composite_method: str = "ic_weighted"
    ic_lookback: int = 12
    min_ir: float = 0.3
    neutralize_industry: bool = True
    neutralize_market_cap: bool = True


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
