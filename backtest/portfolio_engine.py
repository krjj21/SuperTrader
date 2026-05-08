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
    # C: 부분 익절 + 트레일링 스탑 추적
    peak_price: float = 0.0           # 보유 후 최고가 (매 사이클 max(peak, current) 갱신)
    partial_taken: bool = False       # 1차 부분 익절 실행 여부


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
        slippage_pct: float | None = None,
        gap_penalty_pct: float | None = None,
        llm_validator=None,
        run_id: str = "",
        persist_signals: bool = False,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        # 슬리피지 (per leg, 양수). config.backtest.slippage_pct → 기본 0.0015 (0.15%)
        # mid-cap 시초가 체결 호가 한 단계 ~0.1~0.3% 가정
        if slippage_pct is None:
            try:
                slippage_pct = float(getattr(get_config().backtest, "slippage_pct", 0.0015))
            except Exception:
                slippage_pct = 0.0015
        self.slippage_pct = max(0.0, float(slippage_pct))
        # Stop-loss 갭 보정 (음수 패널티 절댓값). 종가 -7% → T+1 시가 평균 추가 -1.5% 갭다운 가정
        if gap_penalty_pct is None:
            try:
                gap_penalty_pct = float(getattr(get_config().backtest, "stop_loss_gap_penalty_pct", 0.015))
            except Exception:
                gap_penalty_pct = 0.015
        self.gap_penalty_pct = max(0.0, float(gap_penalty_pct))
        self.max_positions = max_positions
        if stop_loss_pct is None:
            try:
                stop_loss_pct = get_config().risk.stop_loss_pct
            except Exception:
                stop_loss_pct = 0.07
        self.stop_loss_pct = stop_loss_pct
        # 선택적 LLM/mock 검증기 — 라이브와 동일한 필터를 백테스트에 적용
        self.llm_validator = llm_validator
        self.run_id = run_id
        self.persist_signals = persist_signals and bool(run_id)

        self.cash = float(initial_capital)
        self.positions: dict[str, BacktestPosition] = {}
        self.trades: list[BacktestTrade] = []
        self.equity_history: list[dict] = []
        # LLM/mock 검증 결정 히스토리 (리포트용)
        # [{date, code, name, signal, confirmed, reason}]
        self.llm_decisions: list[dict] = []
        # C: 부분 익절 + 트레일링 config 캐시
        try:
            risk_cfg = get_config().risk
            self._partial_tp_enabled = bool(getattr(risk_cfg, "partial_take_profit_enabled", False))
            self._partial_tp_pct = float(getattr(risk_cfg, "partial_take_profit_pct", 0.50))
            self._trailing_stop_pct = float(getattr(risk_cfg, "trailing_stop_pct", 0.03))
            # SELL → BUY cooldown (재매수 차단, in-memory 추적)
            self._reentry_cooldown_days = int(getattr(risk_cfg, "reentry_cooldown_days", 0))
        except Exception:
            self._partial_tp_enabled = False
            self._partial_tp_pct = 0.50
            self._trailing_stop_pct = 0.03
            self._reentry_cooldown_days = 0
        # 종목별 마지막 매도일 (cooldown 체크용)
        self._last_sell_date: dict[str, str] = {}
        self._cooldown_blocked_count = 0

    def _get_portfolio_value(self, prices: dict[str, float]) -> float:
        """포트폴리오 총 가치를 계산합니다."""
        pos_value = sum(
            p.quantity * prices.get(p.code, p.avg_price)
            for p in self.positions.values()
        )
        return self.cash + pos_value

    def _buy(self, code: str, name: str, price: float, date: str, amount: float | None = None) -> bool:
        """매수 실행. price = 시가 raw, 실제 체결가 = price × (1 + slippage)."""
        # 슬리피지 적용 — 매수는 호가 한 단계 위로 들어간다고 가정
        effective_price = price * (1.0 + self.slippage_pct)
        if amount is None:
            total_value = self.cash + sum(
                p.quantity * p.avg_price for p in self.positions.values()
            )
            amount = total_value / self.max_positions

        quantity = int(amount / effective_price)
        if quantity <= 0:
            return False

        cost = quantity * effective_price
        commission = cost * self.commission_rate
        total_cost = cost + commission

        if total_cost > self.cash:
            quantity = int((self.cash - 100) / (effective_price * (1 + self.commission_rate)))
            if quantity <= 0:
                return False
            cost = quantity * effective_price
            commission = cost * self.commission_rate
            total_cost = cost + commission

        self.cash -= total_cost

        if code in self.positions:
            pos = self.positions[code]
            total_qty = pos.quantity + quantity
            pos.avg_price = (pos.avg_price * pos.quantity + effective_price * quantity) / total_qty
            pos.quantity = total_qty
            # 추가 매수 시 peak 도 매수가 이상 보장
            pos.peak_price = max(pos.peak_price, effective_price)
        else:
            self.positions[code] = BacktestPosition(
                code=code, name=name, quantity=quantity,
                avg_price=effective_price, entry_date=date,
                peak_price=effective_price, partial_taken=False,
            )

        self.trades.append(BacktestTrade(
            code=code, name=name, side="buy",
            quantity=quantity, price=effective_price, date=date,
        ))
        return True

    def _sell(
        self,
        code: str,
        price: float,
        date: str,
        is_stop_loss: bool = False,
        sell_qty: int | None = None,
    ) -> bool:
        """매도 실행. price = 시가 raw, 실제 체결가 = price × (1 − slippage).

        Args:
            sell_qty: 매도 수량. None 이면 *config 기반 자동 판단* (signal SELL 이고
                     C: 부분 익절 ON 이면 partial_take_profit_pct, 그 외 전량).
                     명시 정수 시 그 수량만 매도.
            is_stop_loss: True 면 갭 패널티 + 부분 매도 강제 비활성 (force pool_exit / stop_loss 도 동일).
        """
        if code not in self.positions:
            return False

        # 슬리피지 — 매도는 호가 한 단계 아래로 빠진다고 가정
        effective_price = price * (1.0 - self.slippage_pct)
        # 손절/force-exit 갭 패널티 — 한국시장 갭다운 평균 추정 (settings.yaml stop_loss_gap_penalty_pct)
        if is_stop_loss:
            effective_price *= (1.0 - self.gap_penalty_pct)

        pos = self.positions[code]
        # sell_qty 결정: 명시 우선, 없으면 config 기반 자동 판단
        if sell_qty is None:
            if (
                not is_stop_loss
                and self._partial_tp_enabled
                and not pos.partial_taken
                and pos.quantity >= 2
            ):
                # C: 1차 부분 익절 (signal SELL 한정)
                sell_qty = max(1, int(pos.quantity * self._partial_tp_pct))
        actual_qty = pos.quantity if sell_qty is None else min(sell_qty, pos.quantity)
        if actual_qty <= 0:
            return False
        revenue = actual_qty * effective_price
        commission = revenue * self.commission_rate
        tax = revenue * self.tax_rate
        net_revenue = revenue - commission - tax

        self.cash += net_revenue

        pnl = net_revenue - actual_qty * pos.avg_price
        pnl_pct = (effective_price / pos.avg_price - 1) * 100

        # 보유일수 계산
        try:
            entry = datetime.strptime(pos.entry_date, "%Y-%m-%d")
            exit_date = datetime.strptime(date, "%Y-%m-%d")
            holding_days = (exit_date - entry).days
        except ValueError:
            holding_days = 0

        self.trades.append(BacktestTrade(
            code=code, name=pos.name, side="sell",
            quantity=actual_qty, price=effective_price, date=date,
            pnl=pnl, pnl_pct=pnl_pct, holding_days=holding_days,
        ))

        # 부분 매도 시 잔여 보유 유지, 전량 매도 시 포지션 제거
        if actual_qty >= pos.quantity:
            del self.positions[code]
            # cooldown: 전량 매도 시 last_sell_date 기록 (부분 매도는 잔여 보유라 skip)
            if self._reentry_cooldown_days > 0:
                self._last_sell_date[code] = date
        else:
            pos.quantity -= actual_qty
            pos.partial_taken = True
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

        # ── A3: TA features 1회 PRECOMPUTE ──
        # build_features 는 동일 df 에 4회 중복 호출 (predict + predict_proba×2 + predict_with_position).
        # 종목별 1회만 빌드하고 일별 루프에서 슬라이스 → 7년 30종목 ≈ 211K → 30회 호출.
        features_full: dict[str, pd.DataFrame] = {}
        try:
            from src.timing.features import build_features as _build_feat
            for code, df in ohlcv_dict.items():
                try:
                    features_full[code] = _build_feat(df)
                except Exception as fe:
                    # 한 종목 실패가 전체 백테스트를 막지 않도록 빈 dict 로 그래시풀
                    features_full[code] = None
        except Exception:
            features_full = {}
        self._features_full = features_full

        rebalance_set = set(rebalance_dates)
        current_pool = []

        # T일 시그널 → T+1일 시가 체결을 위한 주문 큐
        # 5-tuple: (code, name, side, is_stop_loss, strength) — stop_loss 매도 시 갭 패널티,
        # strength 는 confidence-proportional sizing 용 (BUY 만 사용, 그 외 None)
        pending_orders: list[tuple[str, str, str, bool, float | None]] = []
        # confidence-proportional sizing 설정 캐시
        try:
            _risk_cfg = get_config().risk
            self._conf_sizing_enabled = bool(getattr(_risk_cfg, "confidence_sizing_enabled", False))
            self._conf_sizing_mode = str(getattr(_risk_cfg, "confidence_sizing_mode", "clamp"))
            self._conf_sizing_min = float(getattr(_risk_cfg, "confidence_sizing_min_mult", 0.5))
            self._conf_sizing_max = float(getattr(_risk_cfg, "confidence_sizing_max_mult", 1.0))
            # E1 변동성 필터 (BUY 진입 게이트, 라이브 RiskManager.validate_order 와 동일 공식)
            self._atr_filter_enabled = bool(getattr(_risk_cfg, "atr_filter_enabled", False))
            self._atr_filter_max_pct = float(getattr(_risk_cfg, "atr_filter_max_pct", 0.05))
            self._atr_filter_period = int(getattr(_risk_cfg, "atr_filter_period", 14))
        except Exception:
            self._conf_sizing_enabled = False
            self._conf_sizing_mode = "clamp"
            self._conf_sizing_min = 0.5
            self._conf_sizing_max = 1.0
            self._atr_filter_enabled = False
            self._atr_filter_max_pct = 0.05
            self._atr_filter_period = 14

        for i, date in enumerate(all_dates):
            # ── 1. 전일 시그널의 주문을 당일 시가로 체결 ──
            if pending_orders:
                stopped_by_pending: set[str] = set()
                any_executed = False
                for code, name, side, is_stop, strength in pending_orders:
                    exec_price = self._get_open_cached(code, date)
                    if exec_price <= 0:
                        continue
                    if side == "buy":
                        # SELL → BUY cooldown 체크 (같은 종목 재매수 차단)
                        if (
                            self._reentry_cooldown_days > 0
                            and code in self._last_sell_date
                        ):
                            try:
                                last_sell = datetime.strptime(self._last_sell_date[code], "%Y-%m-%d")
                                cur = datetime.strptime(date, "%Y-%m-%d")
                                if (cur - last_sell).days < self._reentry_cooldown_days:
                                    self._cooldown_blocked_count += 1
                                    continue  # cooldown 안 → BUY skip
                            except ValueError:
                                pass
                        if code not in self.positions and len(self.positions) < self.max_positions:
                            buy_amount: float | None = None
                            if (
                                self._conf_sizing_enabled
                                and strength is not None
                                and strength > 0
                            ):
                                total_value = self.cash + sum(
                                    p.quantity * p.avg_price for p in self.positions.values()
                                )
                                base_slot = total_value / self.max_positions
                                if self._conf_sizing_mode == "scale":
                                    # 0.5..1.0 → min..max 선형 매핑 (상향 사이징용)
                                    norm = max(0.0, min(1.0, (float(strength) - 0.5) / 0.5))
                                    mult = self._conf_sizing_min + (
                                        self._conf_sizing_max - self._conf_sizing_min
                                    ) * norm
                                else:
                                    mult = min(max(float(strength), self._conf_sizing_min), self._conf_sizing_max)
                                buy_amount = base_slot * mult
                            if self._buy(code, name, exec_price, date, amount=buy_amount):
                                any_executed = True
                    elif side == "sell":
                        if code in self.positions:
                            if self._sell(code, exec_price, date, is_stop_loss=is_stop):
                                stopped_by_pending.add(code)
                                any_executed = True
                pending_orders = []
                # 포지션 변경 직후 전략 동기화 (주문 체결/강제 매도 포함)
                if any_executed:
                    self._sync_strategy_positions(strategy, date)

            # ── 2. 리밸런싱일: 종목풀 업데이트 → T+1 체결 예약 ──
            if date in rebalance_set and date in pool_history:
                new_pool = pool_history[date]

                if hasattr(strategy, "update_pool"):
                    strategy.update_pool(new_pool)

                old_set = set(current_pool)
                new_set = set(new_pool)

                # 퇴출 종목은 모든 전략에서 T+1 SELL 강제 (풀 diff 기반).
                # is_stop=True 표시: 갭 패널티 적용 + partial 익절 적용 안 함 (전량 매도).
                for code in old_set - new_set:
                    if code in self.positions:
                        pending_orders.append((code, "", "sell", True, None))

                # 신규 편입 자동 BUY 는 commit_pool 기반 전략(factor_only)만 수행.
                # 타이밍 전략은 generate_signal 이 진입을 결정한다.
                if hasattr(strategy, "commit_pool"):
                    for code in new_set - old_set:
                        pending_orders.append((code, "", "buy", False, None))
                    strategy.commit_pool()

                current_pool = new_pool

            # ── 3. 손절 체크 + C: peak 갱신 + 트레일링 스탑 ──
            # T일 종가 기준 판단 → T+1 체결 예약
            stop_loss_threshold = -self.stop_loss_pct
            stopped_today: set[str] = set()
            for code in list(self.positions.keys()):
                close_price = self._get_close_cached(code, date)
                if close_price <= 0:
                    continue
                pos = self.positions[code]
                # C: peak 갱신 — 매 사이클 max(peak, close)
                if close_price > pos.peak_price:
                    pos.peak_price = close_price
                # NOTE: 비율 단위 (예: -0.05). stop_loss_threshold 도 비율 (-0.07).
                pnl_ratio = close_price / pos.avg_price - 1
                if pnl_ratio <= stop_loss_threshold:
                    pending_orders.append((code, pos.name, "sell", True, None))
                    stopped_today.add(code)
                    continue
                # C: 트레일링 스탑 — partial 익절 한 종목이 peak 대비 trailing_stop_pct 하락 시 잔여 매도
                if (
                    self._partial_tp_enabled
                    and pos.partial_taken
                    and pos.peak_price > 0
                    and close_price <= pos.peak_price * (1.0 - self._trailing_stop_pct)
                ):
                    pending_orders.append((code, pos.name, "sell", True, None))
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
                # A3: precomputed features 동일 idx 로 슬라이스 (생성 안 됐거나 None 이면 None thread)
                _feat_full = self._features_full.get(code) if hasattr(self, "_features_full") else None
                feat_until = _feat_full.iloc[:idx] if _feat_full is not None else None
                try:
                    signal = strategy.generate_signal(
                        code, df_until, current_date=date, features=feat_until,
                    )
                except TypeError:
                    try:
                        signal = strategy.generate_signal(code, df_until, current_date=date)
                    except TypeError:
                        signal = strategy.generate_signal(code, df_until)

                if signal.signal == Signal.BUY and code not in self.positions:
                    if code in stopped_today:
                        continue
                    if not self._validate_and_record(
                        code, signal.stock_name, "BUY", signal.reason, df_until, date,
                    ):
                        continue
                    # E1 변동성 게이트 (라이브 RiskManager.validate_order 와 동일)
                    if (
                        self._atr_filter_enabled
                        and len(df_until) >= self._atr_filter_period
                    ):
                        try:
                            from src.data.indicators import calc_atr
                            atr_val = float(calc_atr(
                                df_until["high"], df_until["low"], df_until["close"],
                                period=self._atr_filter_period,
                            ).iloc[-1])
                            close_val = float(df_until["close"].iloc[-1])
                            if close_val > 0 and atr_val / close_val > self._atr_filter_max_pct:
                                continue  # 변동성 초과 — 진입 거부
                        except Exception:
                            pass  # 평가 실패 시 통과 (가용성 우선)
                    pending_orders.append(
                        (code, signal.stock_name, "buy", False, float(signal.strength))
                    )

                elif signal.signal == Signal.SELL and code in self.positions:
                    # SELL 은 검증기에서도 항상 확정되지만 일관성을 위해 호출
                    if not self._validate_and_record(
                        code, signal.stock_name, "SELL", signal.reason, df_until, date,
                    ):
                        continue
                    pending_orders.append((code, "", "sell", False, None))

            # ── 5. 전략-엔진 포지션 동기화 (RL 전략 등) ──
            # 시그널 생성 중 전략이 옵티미스틱하게 추가한 엔트리 정리 +
            # step 1 의 체결 결과 재확인
            self._sync_strategy_positions(strategy, date)

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

    def _sync_strategy_positions(self, strategy: BaseStrategy, date: str) -> None:
        """엔진 보유 포지션으로 전략의 내부 _positions 를 동기화한다.

        주문 체결(step 1) 직후와 일일 신호 처리 종료 시점에 호출돼 전략이 항상
        최신 포지션을 기준으로 판단하도록 한다.
        """
        if not hasattr(strategy, "sync_positions"):
            return
        held = set(self.positions.keys())
        prices = {c: self._get_close_cached(c, date) for c in held}
        avg_prices = {c: p.avg_price for c, p in self.positions.items()}
        entry_dates = {c: p.entry_date.replace("-", "") for c, p in self.positions.items()}
        try:
            strategy.sync_positions(
                held, prices,
                avg_prices=avg_prices,
                entry_dates=entry_dates,
                current_date=date,
            )
        except TypeError:
            strategy.sync_positions(held, prices)

    def _validate_and_record(
        self,
        code: str,
        stock_name: str,
        signal: str,
        ml_reason: str,
        df_until: pd.DataFrame,
        date: str,
    ) -> bool:
        """검증기를 호출하고 결정을 히스토리/DB 에 기록한다. confirmed 여부 반환."""
        if self.llm_validator is None:
            return True
        try:
            confirmed, reason = self.llm_validator.validate_signal(
                code, stock_name, signal, ml_reason, df_until,
            )
        except Exception as e:
            # 검증 실패 시 시그널을 그대로 통과시키되, 결정은 기록한다.
            confirmed = True
            reason = f"검증 오류: {e}"

        decision_text = "확정" if confirmed else "보류"
        self.llm_decisions.append({
            "date": date,
            "code": code,
            "name": stock_name,
            "signal": signal,
            "decision": decision_text,
            "reason": reason,
        })

        if self.persist_signals:
            try:
                from src.db.models import save_signal_log
                save_signal_log(
                    stock_code=code,
                    stock_name=stock_name,
                    signal=signal,
                    decision=decision_text,
                    reason=reason,
                    signal_type="llm_backtest",
                    run_id=self.run_id,
                    signal_date=date,
                )
            except Exception as e:
                logger.debug(f"signal_logs 저장 실패: {e}")

        return confirmed

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
            "llm_decisions": self.llm_decisions,
        }
