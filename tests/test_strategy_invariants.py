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


if __name__ == "__main__":
    unittest.main()
