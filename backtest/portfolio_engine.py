"""
포트폴리오 백테스트 엔진
- 멀티종목 동시 보유
- 리밸런싱 + 타이밍 시그널
- 거래비용 (수수료 + 매도세) 반영
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_config
from src.strategy.base import BaseStrategy, Signal


@dataclass
class BacktestPosition:
    code: str
    name: str
    quantity: int
    avg_price: float
    entry_date: str


@dataclass
class BacktestTrade:
    code: str
    name: str
    side: str
    quantity: int
    price: float
    date: str
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_days: int = 0


class PortfolioBacktestEngine:
    """포트폴리오 백테스트 엔진"""

    def __init__(
        self,
        initial_capital: int = 100_000_000,
        commission_rate: float = 0.00015,
        tax_rate: float = 0.0023,
        max_positions: int = 30,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self.max_positions = max_positions

        self.cash = float(initial_capital)
        self.positions: dict[str, BacktestPosition] = {}
        self.trades: list[BacktestTrade] = []
        self.equity_history: list[dict] = []

    def _get_portfolio_value(self, prices: dict[str, float]) -> float:
        """포트폴리오 총 가치를 계산합니다."""
        pos_value = sum(
            p.quantity * prices.get(p.code, p.avg_price)
            for p in self.positions.values()
        )
        return self.cash + pos_value

    def _buy(self, code: str, name: str, price: float, date: str, amount: float | None = None) -> bool:
        """매수 실행"""
        if amount is None:
            total_value = self.cash + sum(
                p.quantity * p.avg_price for p in self.positions.values()
            )
            amount = total_value / self.max_positions

        quantity = int(amount / price)
        if quantity <= 0:
            return False

        cost = quantity * price
        commission = cost * self.commission_rate
        total_cost = cost + commission

        if total_cost > self.cash:
            quantity = int((self.cash - 100) / (price * (1 + self.commission_rate)))
            if quantity <= 0:
                return False
            cost = quantity * price
            commission = cost * self.commission_rate
            total_cost = cost + commission

        self.cash -= total_cost

        if code in self.positions:
            pos = self.positions[code]
            total_qty = pos.quantity + quantity
            pos.avg_price = (pos.avg_price * pos.quantity + price * quantity) / total_qty
            pos.quantity = total_qty
        else:
            self.positions[code] = BacktestPosition(
                code=code, name=name, quantity=quantity,
                avg_price=price, entry_date=date,
            )

        self.trades.append(BacktestTrade(
            code=code, name=name, side="buy",
            quantity=quantity, price=price, date=date,
        ))
        return True

    def _sell(self, code: str, price: float, date: str) -> bool:
        """매도 실행 (전량)"""
        if code not in self.positions:
            return False

        pos = self.positions[code]
        revenue = pos.quantity * price
        commission = revenue * self.commission_rate
        tax = revenue * self.tax_rate
        net_revenue = revenue - commission - tax

        self.cash += net_revenue

        pnl = net_revenue - pos.quantity * pos.avg_price
        pnl_pct = (price / pos.avg_price - 1) * 100

        # 보유일수 계산
        try:
            entry = datetime.strptime(pos.entry_date, "%Y-%m-%d")
            exit_date = datetime.strptime(date, "%Y-%m-%d")
            holding_days = (exit_date - entry).days
        except ValueError:
            holding_days = 0

        self.trades.append(BacktestTrade(
            code=code, name=pos.name, side="sell",
            quantity=pos.quantity, price=price, date=date,
            pnl=pnl, pnl_pct=pnl_pct, holding_days=holding_days,
        ))

        del self.positions[code]
        return True

    def run(
        self,
        strategy: BaseStrategy,
        ohlcv_dict: dict[str, pd.DataFrame],
        pool_history: dict[str, list[str]],
        rebalance_dates: list[str],
    ) -> dict:
        """백테스트를 실행합니다.

        Args:
            strategy: 전략 인스턴스
            ohlcv_dict: {종목코드: OHLCV DataFrame} (date 컬럼 필요)
            pool_history: {리밸런싱일: 종목코드 리스트}
            rebalance_dates: 리밸런싱 날짜 리스트 (YYYY-MM-DD)

        Returns:
            백테스트 결과 딕셔너리
        """
        # 모든 거래일 수집
        all_dates = set()
        for df in ohlcv_dict.values():
            if "date" in df.columns:
                all_dates.update(df["date"].astype(str).tolist())
        all_dates = sorted(all_dates)

        rebalance_set = set(rebalance_dates)
        current_pool = []

        for date in all_dates:
            # 리밸런싱일: 종목풀 업데이트
            if date in rebalance_set and date in pool_history:
                new_pool = pool_history[date]

                if hasattr(strategy, "update_pool"):
                    strategy.update_pool(new_pool)

                if hasattr(strategy, "commit_pool"):
                    # factor_only: 종목풀 변경 시 매매
                    old_set = set(current_pool)
                    new_set = set(new_pool)

                    # 퇴출 종목 매도
                    for code in old_set - new_set:
                        if code in self.positions:
                            price = self._get_price(ohlcv_dict, code, date)
                            if price > 0:
                                self._sell(code, price, date)

                    # 신규 편입 종목 매수
                    for code in new_set - old_set:
                        price = self._get_price(ohlcv_dict, code, date)
                        if price > 0:
                            self._buy(code, "", price, date)

                    strategy.commit_pool()

                current_pool = new_pool

            # 일일 타이밍 시그널 (종목풀 내 종목)
            for code in current_pool:
                if code not in ohlcv_dict:
                    continue

                df = ohlcv_dict[code]
                if "date" in df.columns:
                    df_until = df[df["date"].astype(str) <= date]
                else:
                    df_until = df

                if len(df_until) < 2:
                    continue

                signal = strategy.generate_signal(code, df_until)

                if signal.signal == Signal.BUY and code not in self.positions:
                    price = self._get_price(ohlcv_dict, code, date)
                    if price > 0 and len(self.positions) < self.max_positions:
                        self._buy(code, signal.stock_name, price, date)

                elif signal.signal == Signal.SELL and code in self.positions:
                    price = self._get_price(ohlcv_dict, code, date)
                    if price > 0:
                        self._sell(code, price, date)

            # 일일 포트폴리오 가치 기록
            prices = {
                code: self._get_price(ohlcv_dict, code, date)
                for code in self.positions
            }
            portfolio_value = self._get_portfolio_value(prices)
            self.equity_history.append({
                "date": date,
                "equity": portfolio_value,
                "cash": self.cash,
                "n_positions": len(self.positions),
            })

        return self._compile_results()

    def _get_price(self, ohlcv_dict: dict, code: str, date: str) -> float:
        if code not in ohlcv_dict:
            return 0.0
        df = ohlcv_dict[code]
        if "date" in df.columns:
            row = df[df["date"].astype(str) == date]
            if not row.empty:
                return float(row["close"].iloc[0])
        return 0.0

    def _compile_results(self) -> dict:
        equity_df = pd.DataFrame(self.equity_history)
        if equity_df.empty:
            return {"error": "no_trades"}

        equity_curve = equity_df.set_index("date")["equity"]
        daily_returns = equity_curve.pct_change()

        # 라운드트립 거래 (sell만)
        sell_trades = [
            {"pnl": t.pnl, "pnl_pct": t.pnl_pct, "holding_days": t.holding_days}
            for t in self.trades if t.side == "sell"
        ]

        from backtest.metrics import calculate_metrics
        metrics = calculate_metrics(
            equity_curve, daily_returns, sell_trades, self.initial_capital,
        )

        return {
            "metrics": metrics,
            "equity_curve": equity_curve,
            "daily_returns": daily_returns,
            "trades": self.trades,
        }
