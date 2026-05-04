from __future__ import annotations

import unittest
from unittest.mock import Mock, PropertyMock, patch

import pandas as pd

from backtest.portfolio_engine import PortfolioBacktestEngine
from src.risk.manager import RiskManager
from src.strategy.base import BaseStrategy, Signal, TradeSignal
from src.strategy.factor_hybrid import FactorHybridStrategy


def _make_ohlcv(dates: list[str], opens: list[float], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": [max(o, c) for o, c in zip(opens, closes)],
            "low": [min(o, c) for o, c in zip(opens, closes)],
            "close": closes,
            "volume": [1_000_000] * len(dates),
        }
    )


class FirstDayBuyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="first_day_buy")
        self._pool: set[str] = set()

    def update_pool(self, codes: list[str]) -> None:
        self._pool = set(codes)

    def generate_signal(
        self,
        stock_code: str,
        df: pd.DataFrame,
        stock_name: str = "",
        current_date: str | None = None,
    ) -> TradeSignal:
        price = int(df["close"].iloc[-1])
        if current_date == "2024-01-02":
            return TradeSignal(
                signal=Signal.BUY,
                stock_code=stock_code,
                stock_name=stock_name or stock_code,
                price=price,
                reason="entry",
            )
        return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, stock_name=stock_name, price=price)


class StrategyInvariantTests(unittest.TestCase):
    def test_backtest_executes_signal_on_next_day_open(self) -> None:
        strategy = FirstDayBuyStrategy()
        engine = PortfolioBacktestEngine(initial_capital=1_000_000, max_positions=5, slippage_pct=0.0, gap_penalty_pct=0.0)
        ohlcv = {
            "AAA": _make_ohlcv(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                [100, 105, 110, 120],
                [101, 106, 111, 121],
            )
        }

        result = engine.run(
            strategy=strategy,
            ohlcv_dict=ohlcv,
            pool_history={"2024-01-02": ["AAA"]},
            rebalance_dates=["2024-01-02"],
        )

        self.assertNotIn("error", result)
        self.assertEqual(len(result["trades"]), 1)
        buy = result["trades"][0]
        self.assertEqual(buy.side, "buy")
        self.assertEqual(buy.date, "2024-01-03")
        self.assertEqual(buy.price, 110)

    def test_backtest_force_sells_when_stock_leaves_pool(self) -> None:
        strategy = FirstDayBuyStrategy()
        engine = PortfolioBacktestEngine(initial_capital=1_000_000, max_positions=5, slippage_pct=0.0, gap_penalty_pct=0.0)
        ohlcv = {
            "AAA": _make_ohlcv(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
                [100, 105, 110, 90, 95],
                [101, 106, 111, 91, 96],
            )
        }

        result = engine.run(
            strategy=strategy,
            ohlcv_dict=ohlcv,
            pool_history={
                "2024-01-02": ["AAA"],
                "2024-01-03": [],
            },
            rebalance_dates=["2024-01-02", "2024-01-03"],
        )

        self.assertNotIn("error", result)
        self.assertEqual(len(result["trades"]), 2)
        buy, sell = result["trades"]
        self.assertEqual(buy.side, "buy")
        self.assertEqual(buy.date, "2024-01-03")
        self.assertEqual(sell.side, "sell")
        self.assertEqual(sell.date, "2024-01-04")
        self.assertEqual(sell.price, 90)

    def test_hybrid_buy_requires_both_xgb_and_rl(self) -> None:
        strategy = FactorHybridStrategy.__new__(FactorHybridStrategy)
        strategy.ml_predictor = Mock()
        strategy.rl_predictor = Mock()
        strategy.ml_predictor.predict.return_value = 1
        # Alpha BUY confidence — invariant test 는 신뢰도 통과로 가정 (None = legacy 호환)
        strategy.ml_predictor.predict_proba_last.return_value = None
        strategy.rl_predictor.predict_with_position.return_value = 0
        strategy.rl_predictor.predict_with_position_with_probs.return_value = (0, None)
        strategy._buy_threshold = 0.07
        strategy._sell_threshold = 0.06
        strategy._ml_sell_threshold = 0.60
        strategy._ml_buy_threshold = 0.55
        strategy._ml_label = "XGB"
        strategy._ml_model_type = "xgboost"
        strategy._positions = {}
        strategy._pool = {"AAA"}
        strategy._current_date = None

        df = _make_ohlcv(
            [f"2024-01-{i:02d}" for i in range(1, 62)],
            [100 + i for i in range(61)],
            [100 + i for i in range(61)],
        )

        signal = strategy.generate_signal("AAA", df, stock_name="AAA", current_date="2024-03-01")
        self.assertEqual(signal.signal, Signal.HOLD)
        self.assertEqual(strategy._positions, {})

    def test_hybrid_sell_needs_rl_or_high_xgb_confidence(self) -> None:
        strategy = FactorHybridStrategy.__new__(FactorHybridStrategy)
        strategy.ml_predictor = Mock()
        strategy.rl_predictor = Mock()
        strategy.ml_predictor.predict.return_value = -1
        strategy.ml_predictor.predict_proba_last.return_value = 0.40
        strategy.rl_predictor.predict_with_position.return_value = 0
        strategy.rl_predictor.predict_with_position_with_probs.return_value = (0, None)
        strategy._buy_threshold = 0.07
        strategy._sell_threshold = 0.06
        strategy._ml_sell_threshold = 0.60
        strategy._ml_label = "XGB"
        strategy._ml_model_type = "xgboost"
        strategy._positions = {"AAA": {"entry_price": 100.0, "entry_date": "20240115"}}
        strategy._pool = {"AAA"}
        strategy._current_date = "20240301"

        df = _make_ohlcv(
            [f"2024-01-{i:02d}" for i in range(1, 31)] + [f"2024-02-{i:02d}" for i in range(1, 31)],
            [110] * 60,
            [110] * 60,
        )

        signal = strategy.generate_signal("AAA", df, stock_name="AAA", current_date="2024-03-01")
        self.assertEqual(signal.signal, Signal.HOLD)

        strategy.ml_predictor.predict_proba_last.return_value = 0.85
        signal = strategy.generate_signal("AAA", df, stock_name="AAA", current_date="2024-03-01")
        self.assertEqual(signal.signal, Signal.SELL)

    def test_hybrid_profit_aware_lowers_sell_threshold_at_high_pnl(self) -> None:
        # +30% 보유 + ml_sell_prob=0.50 + RL HOLD: profit-aware off → HOLD, on → SELL.
        # 047040 대우건설 +30% 케이스 회귀 방지 (2026-05-04).
        strategy = FactorHybridStrategy.__new__(FactorHybridStrategy)
        strategy.ml_predictor = Mock()
        strategy.rl_predictor = Mock()
        strategy.ml_predictor.predict.return_value = -1
        strategy.ml_predictor.predict_proba_last.return_value = 0.50
        strategy.rl_predictor.predict_with_position_with_probs.return_value = (0, None)
        strategy._buy_threshold = 0.07
        strategy._sell_threshold = 0.06
        strategy._ml_sell_threshold = 0.60
        strategy._ml_buy_threshold = 0.55
        strategy._ml_label = "XGB"
        strategy._ml_model_type = "xgboost"
        # entry_price=100, df close=130 → unrealized_pnl = +30%
        strategy._positions = {"AAA": {"entry_price": 100.0, "entry_date": "20240115"}}
        strategy._pool = {"AAA"}
        strategy._current_date = "20240301"

        df = _make_ohlcv(
            [f"2024-01-{i:02d}" for i in range(1, 31)] + [f"2024-02-{i:02d}" for i in range(1, 31)],
            [130] * 60,
            [130] * 60,
        )

        # profit-aware off → 0.50 < 0.60 → HOLD
        strategy._profit_aware_enabled = False
        sig_off = strategy.generate_signal("AAA", df, stock_name="AAA", current_date="2024-03-01")
        self.assertEqual(sig_off.signal, Signal.HOLD)

        # profit-aware on (floor=0.10, ceiling=0.30, max_disc=0.20) → +30%면 0.60×0.80=0.48 → SELL
        strategy._profit_aware_enabled = True
        strategy._profit_aware_floor = 0.10
        strategy._profit_aware_ceiling = 0.30
        strategy._profit_aware_max_disc = 0.20
        sig_on = strategy.generate_signal("AAA", df, stock_name="AAA", current_date="2024-03-01")
        self.assertEqual(sig_on.signal, Signal.SELL)
        self.assertLess(strategy._last_diag["effective_sell_threshold"], 0.60)

    def test_position_size_respects_max_position_limit(self) -> None:
        signal = TradeSignal(signal=Signal.BUY, stock_code="AAA", price=10_000)
        risk_mgr = RiskManager(account=Mock())

        with patch.object(RiskManager, "is_trading_allowed", new_callable=PropertyMock, return_value=True):
            qty = risk_mgr.calculate_position_size(
                signal=signal,
                available_cash=1_000_000,
                total_assets=10_000_000,
                current_positions=0,
            )

        # config max_position_pct=5% -> 500,000 won cap -> 50 shares
        self.assertEqual(qty, 50)

    def test_confidence_sizing_scales_with_strength(self) -> None:
        # strength=0.6 → 0.6× slot. 50주 → 30주 기대.
        signal = TradeSignal(signal=Signal.BUY, stock_code="AAA", price=10_000, strength=0.6)
        risk_mgr = RiskManager(account=Mock())
        risk_mgr.config.confidence_sizing_enabled = True
        risk_mgr.config.confidence_sizing_min_mult = 0.5
        risk_mgr.config.confidence_sizing_max_mult = 1.0

        with patch.object(RiskManager, "is_trading_allowed", new_callable=PropertyMock, return_value=True):
            qty = risk_mgr.calculate_position_size(
                signal=signal,
                available_cash=1_000_000,
                total_assets=10_000_000,
                current_positions=0,
            )

        self.assertEqual(qty, 30)

    def test_confidence_sizing_clamps_below_min(self) -> None:
        # strength=0.2 → min 0.5 로 클램프. 50주 → 25주 기대.
        signal = TradeSignal(signal=Signal.BUY, stock_code="AAA", price=10_000, strength=0.2)
        risk_mgr = RiskManager(account=Mock())
        risk_mgr.config.confidence_sizing_enabled = True
        risk_mgr.config.confidence_sizing_min_mult = 0.5
        risk_mgr.config.confidence_sizing_max_mult = 1.0

        with patch.object(RiskManager, "is_trading_allowed", new_callable=PropertyMock, return_value=True):
            qty = risk_mgr.calculate_position_size(
                signal=signal,
                available_cash=1_000_000,
                total_assets=10_000_000,
                current_positions=0,
            )

        self.assertEqual(qty, 25)

    def test_confidence_sizing_disabled_keeps_uniform(self) -> None:
        # 플래그 OFF → strength 무시.
        signal = TradeSignal(signal=Signal.BUY, stock_code="AAA", price=10_000, strength=0.6)
        risk_mgr = RiskManager(account=Mock())
        risk_mgr.config.confidence_sizing_enabled = False

        with patch.object(RiskManager, "is_trading_allowed", new_callable=PropertyMock, return_value=True):
            qty = risk_mgr.calculate_position_size(
                signal=signal,
                available_cash=1_000_000,
                total_assets=10_000_000,
                current_positions=0,
            )

        self.assertEqual(qty, 50)

    # ── E1 변동성 진입 게이트 ─────────────────────────────────────────
    def _build_atr_test_df(self, atr_pct: float, n: int = 30) -> pd.DataFrame:
        # close=100 으로 고정, high-low 폭만 atr_pct × close 가 되게 구성
        # → ATR(14) ≈ atr_pct × 100 = atr_pct × close
        close = 100.0
        spread = atr_pct * close
        opens = [close] * n
        closes = [close] * n
        highs = [close + spread / 2] * n
        lows = [close - spread / 2] * n
        return pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": [1_000_000] * n,
        })

    def test_atr_filter_rejects_high_volatility_buy(self) -> None:
        # ATR/price ≈ 8% > 5% 임계 → reject
        signal = TradeSignal(signal=Signal.BUY, stock_code="AAA", price=100, strength=1.0)
        balance = Mock(total_deposit=10_000_000, total_pnl=0, total_eval=10_000_000)
        risk_mgr = RiskManager(account=Mock())
        risk_mgr.config.atr_filter_enabled = True
        risk_mgr.config.atr_filter_max_pct = 0.05
        risk_mgr.config.atr_filter_period = 14
        df = self._build_atr_test_df(atr_pct=0.08)

        with patch.object(RiskManager, "is_trading_allowed", new_callable=PropertyMock, return_value=True):
            with patch.object(RiskManager, "check_daily_loss_limit", return_value=True):
                valid, reason = risk_mgr.validate_order(signal, quantity=10, balance=balance, df=df)

        self.assertFalse(valid)
        self.assertIn("변동성 초과", reason)

    def test_atr_filter_disabled_passes(self) -> None:
        signal = TradeSignal(signal=Signal.BUY, stock_code="AAA", price=100, strength=1.0)
        balance = Mock(total_deposit=10_000_000, total_pnl=0, total_eval=10_000_000)
        risk_mgr = RiskManager(account=Mock())
        risk_mgr.config.atr_filter_enabled = False
        df = self._build_atr_test_df(atr_pct=0.20)  # 매우 높은 변동성이어도 OFF 면 통과

        with patch.object(RiskManager, "is_trading_allowed", new_callable=PropertyMock, return_value=True):
            with patch.object(RiskManager, "check_daily_loss_limit", return_value=True):
                valid, _ = risk_mgr.validate_order(signal, quantity=10, balance=balance, df=df)

        self.assertTrue(valid)

    def test_atr_filter_does_not_block_sell(self) -> None:
        # SELL 은 절대 변동성 게이트로 차단 X (잘못된 매도 차단 = 손실 누적)
        signal = TradeSignal(signal=Signal.SELL, stock_code="AAA", price=100, strength=1.0)
        balance = Mock(total_deposit=10_000_000, total_pnl=0, total_eval=10_000_000)
        risk_mgr = RiskManager(account=Mock())
        risk_mgr.config.atr_filter_enabled = True
        risk_mgr.config.atr_filter_max_pct = 0.05
        df = self._build_atr_test_df(atr_pct=0.20)  # BUY 였다면 거부됐을 변동성

        with patch.object(RiskManager, "is_trading_allowed", new_callable=PropertyMock, return_value=True):
            valid, _ = risk_mgr.validate_order(signal, quantity=10, balance=balance, df=df)

        self.assertTrue(valid)

    # ── B1 외국인+기관 합산 helper ────────────────────────────────────
    def test_combined_net_buy_helper_handles_missing_organ(self) -> None:
        # organ_net_amount NULL 인 행 1개 + 정상 1개 → foreign 만 합산되는지 검증
        from src.db.sappo_models import get_combined_net_buy_cumulative

        rec_a = Mock(foreign_net_amount=1000, organ_net_amount=None)
        rec_b = Mock(foreign_net_amount=2000, organ_net_amount=500)
        fake_query = Mock()
        fake_query.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [rec_a, rec_b]
        fake_session = Mock()
        fake_session.query.return_value = fake_query
        with patch("src.db.sappo_models.get_sappo_session", return_value=fake_session):
            cum, n = get_combined_net_buy_cumulative("AAA", "20260501", days=20)

        # foreign(1000+2000) + organ(0+500) = 3500
        self.assertEqual(cum, 3500)
        self.assertEqual(n, 2)

    def test_pool_cache_invalidates_on_filter_mode_change(self) -> None:
        # investor_filter_mode 변경 시 cache_key 가 달라져야 함
        from src.config import get_config
        from src.factors import pool_cache

        cfg = get_config()
        cfg.factors.investor_filter_mode = "foreign"
        key_a = pool_cache.cache_key()
        cfg.factors.investor_filter_mode = "foreign_organ"
        key_b = pool_cache.cache_key()

        self.assertNotEqual(key_a, key_b)

    # ── A1+A2 macro overlay ──────────────────────────────────────────
    def test_regime_detector_handles_macro_empty(self) -> None:
        # sappo_macro_features 가 비어있을 때 KOSPI-only 2D fallback 으로 작동
        from src.regime.detector import RegimeDetector
        det = RegimeDetector(lookback_days=60)
        kospi = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=80, freq="D"),
            "open": [100.0] * 80, "high": [101.0] * 80, "low": [99.0] * 80,
            "close": [100.0 + i * 0.1 for i in range(80)],
            "volume": [1_000_000] * 80,
        })
        with patch("src.db.sappo_models.get_macro_features_window", return_value=[]):
            feats = det._build_features(kospi, end_date="20250320")
        # KOSPI 2개 컬럼만 있어야 함 (macro 없음)
        for col in ("usdkrw_log_ret", "vix_z"):
            self.assertNotIn(col, feats.columns)
        self.assertIn("log_ret", feats.columns)
        self.assertIn("roll_vol", feats.columns)

    def test_regime_detector_uses_macro_when_present(self) -> None:
        # macro 데이터 있으면 6D feature 매트릭스 구성
        from src.regime.detector import RegimeDetector
        det = RegimeDetector(lookback_days=60)
        dates = pd.date_range("2025-01-01", periods=80, freq="D")
        kospi = pd.DataFrame({
            "date": dates, "open": [100.0] * 80, "high": [101.0] * 80, "low": [99.0] * 80,
            "close": [100.0 + i * 0.1 for i in range(80)],
            "volume": [1_000_000] * 80,
        })
        macro_rows = [
            Mock(
                date=d.strftime("%Y%m%d"),
                usdkrw_log_ret=0.001 * (i % 3 - 1),
                usdkrw_vol_20d=0.005,
                vix_log_ret=0.02,
                vix_close=15.0 + i * 0.1,
            )
            for i, d in enumerate(dates)
        ]
        with patch("src.db.sappo_models.get_macro_features_window", return_value=macro_rows):
            feats = det._build_features(kospi, end_date="20250320")
        for col in ("usdkrw_log_ret", "usdkrw_vol_20d", "vix_log_ret", "vix_z"):
            self.assertIn(col, feats.columns)


if __name__ == "__main__":
    unittest.main()
