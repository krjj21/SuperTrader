"""
포트폴리오 백테스트 엔진
- 멀티종목 동시 보유
- 리밸런싱 + 타이밍 시그널
- 거래비용 (수수료 + 매도세) 반영
"""
from __future__ import annotations

import bisect
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
        stop_loss_pct: float | None = None,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self.max_positions = max_positions
        if stop_loss_pct is None:
            try:
                stop_loss_pct = get_config().risk.stop_loss_pct
            except Exception:
                stop_loss_pct = 0.07
        self.stop_loss_pct = stop_loss_pct

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
        # ── 전처리: 날짜 변환 1회, 가격 캐시, 날짜 인덱스 구축 ──
        close_cache: dict[str, dict[str, float]] = {}
        open_cache: dict[str, dict[str, float]] = {}
        date_lists: dict[str, list[str]] = {}
        all_dates = set()

        for code, df in ohlcv_dict.items():
            if "date" in df.columns:
                date_strs = df["date"].astype(str).tolist()
                df["date_str"] = date_strs
                all_dates.update(date_strs)
                date_lists[code] = date_strs
                close_cache[code] = dict(zip(date_strs, df["close"].astype(float)))
                if "open" in df.columns:
                    open_cache[code] = dict(zip(date_strs, df["open"].astype(float)))

        all_dates = sorted(all_dates)
        self._close_cache = close_cache
        self._open_cache = open_cache

        rebalance_set = set(rebalance_dates)
        current_pool = []

        # T일 시그널 → T+1일 시가 체결을 위한 주문 큐
        pending_orders: list[tuple[str, str, str]] = []  # (code, name, side)

        for i, date in enumerate(all_dates):
            # ── 1. 전일 시그널의 주문을 당일 시가로 체결 ──
            if pending_orders:
                stopped_by_pending: set[str] = set()
                for code, name, side in pending_orders:
                    exec_price = self._get_open_cached(code, date)
                    if exec_price <= 0:
                        continue
                    if side == "buy":
                        if code not in self.positions and len(self.positions) < self.max_positions:
                            self._buy(code, name, exec_price, date)
                    elif side == "sell":
                        if code in self.positions:
                            self._sell(code, exec_price, date)
                            stopped_by_pending.add(code)
                pending_orders = []

            # ── 2. 리밸런싱일: 종목풀 업데이트 → T+1 체결 예약 ──
            if date in rebalance_set and date in pool_history:
                new_pool = pool_history[date]

                if hasattr(strategy, "update_pool"):
                    strategy.update_pool(new_pool)

                if hasattr(strategy, "commit_pool"):
                    old_set = set(current_pool)
                    new_set = set(new_pool)

                    for code in old_set - new_set:
                        if code in self.positions:
                            pending_orders.append((code, "", "sell"))

                    for code in new_set - old_set:
                        pending_orders.append((code, "", "buy"))

                    strategy.commit_pool()

                current_pool = new_pool

            # ── 3. 손절 체크: T일 종가 기준 판단 → T+1 체결 예약 ──
            stop_loss_threshold = -self.stop_loss_pct
            stopped_today: set[str] = set()
            for code in list(self.positions.keys()):
                close_price = self._get_close_cached(code, date)
                if close_price <= 0:
                    continue
                pos = self.positions[code]
                pnl_pct = close_price / pos.avg_price - 1
                if pnl_pct <= stop_loss_threshold:
                    pending_orders.append((code, pos.name, "sell"))
                    stopped_today.add(code)

            # ── 4. 일일 타이밍 시그널: T일 데이터로 판단 → T+1 체결 예약 ──
            for code in current_pool:
                if code not in ohlcv_dict:
                    continue

                dl = date_lists.get(code)
                if not dl:
                    continue

                idx = bisect.bisect_right(dl, date)
                if idx < 2:
                    continue

                df_until = ohlcv_dict[code].iloc[:idx]
                signal = strategy.generate_signal(code, df_until)

                if signal.signal == Signal.BUY and code not in self.positions:
                    if code in stopped_today:
                        continue
                    pending_orders.append((code, signal.stock_name, "buy"))

                elif signal.signal == Signal.SELL and code in self.positions:
                    pending_orders.append((code, "", "sell"))

            # ── 5. 전략-엔진 포지션 동기화 (RL 전략 등) ──
            if hasattr(strategy, "sync_positions"):
                held = set(self.positions.keys())
                prices = {c: self._get_close_cached(c, date) for c in held}
                strategy.sync_positions(held, prices)

            # ── 6. 일일 포트폴리오 가치 기록 (종가 기준) ──
            prices = {
                code: self._get_close_cached(code, date)
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

    def _get_close_cached(self, code: str, date: str) -> float:
        """종가 캐시에서 O(1) 조회"""
        cache = self._close_cache.get(code)
        if cache:
            return cache.get(date, 0.0)
        return 0.0

    def _get_open_cached(self, code: str, date: str) -> float:
        """시가 캐시에서 O(1) 조회 (없으면 종가 fallback)"""
        cache = self._open_cache.get(code)
        if cache:
            price = cache.get(date, 0.0)
            if price > 0:
                return price
        return self._get_close_cached(code, date)

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
