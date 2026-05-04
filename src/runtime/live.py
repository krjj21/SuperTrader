"""실시간 자동매매 모드 — main.py 에서 분리됨 (2026-04-28).

  · run_live()                 — KIS API 통한 실시간 자동매매 (BlockingScheduler)
  · _create_live_strategy(cfg) — 설정 기반 전략 인스턴스 생성

스케줄 구성 (휴장일 가드 _skip_if_market_closed 통해 평일+영업일에만 실행):
  · 5분 시그널 체크 (장중)
  · 일일 리포트 (post_market)
  · 일일 SAPPO 수집 (16:00)
  · 일일 외국인 매매 수집 (16:30, foreign_filter_enabled 시)
  · 일일 시장 뉴스 (08:30, regime.enabled 시)
  · 일일 regime detect (08:45, regime.enabled 시)
  · 월간/격주 리밸런싱
  · 토요일 모델 재학습 (06:00)
  · 토요일 SAPPO 주간 리포트 (07:00)
  · 5분 heartbeat
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import get_config


def _trigger_codex_daily_review(model: str = "") -> None:
    """일일 리포트 이후 codex daily 리뷰를 트리거합니다."""
    cmd = [sys.executable, "scripts/codex_review.py", "daily"]
    if model:
        cmd.extend(["--model", model])
    logger.info(f"Codex 일일 리뷰 실행: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        logger.warning("codex CLI 미설치 — 리뷰 단계 건너뜀")
    except Exception as e:
        logger.warning(f"Codex 일일 리뷰 실패: {e}")


def _create_live_strategy(config):
    """설정에 따라 전략을 생성합니다."""
    strategy_name = config.strategy.name

    if strategy_name == "factor_only":
        from src.strategy.factor_only import FactorOnlyStrategy
        return FactorOnlyStrategy(params=config.strategy.params.model_dump())
    elif strategy_name == "factor_macd":
        from src.strategy.factor_macd import FactorMACDStrategy
        return FactorMACDStrategy(params=config.strategy.params.model_dump())
    elif strategy_name == "factor_kdj":
        from src.strategy.factor_kdj import FactorKDJStrategy
        return FactorKDJStrategy(params=config.strategy.params.model_dump())
    elif strategy_name == "factor_rl":
        model_path = "models/rl_timing.pt"
        from src.strategy.factor_rl import FactorRLStrategy
        return FactorRLStrategy(model_path, config.strategy.params.model_dump())
    elif strategy_name == "factor_hybrid":
        from src.strategy.factor_hybrid import FactorHybridStrategy
        return FactorHybridStrategy(
            ml_model_path="models/xgboost_timing.pkl",
            rl_model_path="models/rl_timing.pt",
            params=config.strategy.params.model_dump(),
        )
    elif strategy_name.startswith("factor_"):
        model_type = strategy_name.replace("factor_", "")
        ext = ".pkl" if model_type in ("decision_tree", "xgboost", "lightgbm") else ".pt"
        model_path = f"models/{model_type}_timing{ext}"
        from src.strategy.factor_ml import FactorMLStrategy
        return FactorMLStrategy(model_type, model_path, config.strategy.params.model_dump())
    else:
        raise ValueError(f"알 수 없는 전략: {strategy_name}")


def run_live() -> None:
    """실시간 자동매매를 실행합니다."""
    config = get_config()
    logger.info("=" * 60)
    logger.info("SuperTrader 라이브 매매 시작")
    logger.info("=" * 60)

    from apscheduler.schedulers.blocking import BlockingScheduler
    from src.broker.kis_client import KISClient
    from src.broker.order import OrderManager
    from src.broker.account import AccountManager
    from src.risk.manager import RiskManager
    from src.notification.slack_bot import SlackNotifier
    from src.db.models import (
        init_db, save_runtime_status, save_signal_log, save_daily_pnl,
        save_trade, save_holding, remove_holding,
    )

    # 초기화
    init_db(config.database.path)
    client = KISClient()
    order_mgr = OrderManager(client)
    account_mgr = AccountManager(client)
    risk_mgr = RiskManager(account_mgr)
    notifier = SlackNotifier()

    # 전략 생성
    strategy = _create_live_strategy(config)

    # 팩터 기반 종목풀 구성
    from src.factors.stock_pool import build_stock_pool
    _current_pool = [None]  # mutable container for closure
    _stock_names: dict[str, str] = {}  # 종목코드 → 종목명 캐시
    validator = None

    # ── Regime Detector 컨테이너 (label 문자열 또는 None) ──
    _current_regime: list[str | None] = [None]
    if config.regime.enabled:
        try:
            from src.db.sappo_models import init_sappo_db, get_latest_regime
            init_sappo_db(config.database.path)
            prev = get_latest_regime()
            if prev is not None:
                _current_regime[0] = prev.label
                logger.info(
                    f"[REGIME] 이전 regime 복원: label={prev.label} date={prev.date} "
                    f"(lambda={config.regime.lambda_})"
                )
            else:
                logger.info("[REGIME] 이전 라벨 없음 — 첫 detect_regime_daily 까지 영향 없음")
        except Exception as e:
            logger.warning(f"[REGIME] 시작 시 regime 복원 실패: {e}")

    def _skip_if_market_closed(job_name: str) -> bool:
        """주말/공휴일/임시휴장이면 True 반환 (잡 본문 직전에 호출).
        FDR 조회 실패 시 보수적으로 False 반환 (일을 진행).
        """
        from src.utils.market_calendar import is_market_holiday
        from datetime import datetime as _dt
        now = _dt.now()
        if now.weekday() >= 5 or is_market_holiday(now):
            logger.info(f"[{job_name}] 휴장일/주말 — skip ({now.strftime('%Y-%m-%d %a')})")
            return True
        return False

    def sync_runtime_status() -> None:
        pool = _current_pool[0]
        save_runtime_status(
            strategy=config.strategy.name,
            pool_size=len(pool.codes) if pool else 0,
            llm_enabled=bool(validator and validator.is_enabled),
            kill_switch=risk_mgr.kill_switch_active,
            check_interval=config.schedule.check_interval_sec,
            daily_loss_limit=config.risk.daily_loss_limit_pct * 100,
            max_positions=config.risk.max_total_positions,
        )

    def rebalance_pool():
        """팩터 기반 종목풀을 리밸런싱합니다 (휴장일이면 skip → 다음 사이클 대기)."""
        if _skip_if_market_closed("rebalance_pool"):
            return
        today = datetime.now().strftime("%Y%m%d")
        try:
            pool = build_stock_pool(
                today, previous_pool=_current_pool[0],
                regime_label=_current_regime[0],
            )
            if pool.codes:
                strategy.update_pool(pool.codes)
                _current_pool[0] = pool

                # 종목명 캐시 업데이트 (pykrx)
                from pykrx import stock as krx
                for code in pool.codes:
                    if code not in _stock_names:
                        try:
                            _stock_names[code] = krx.get_market_ticker_name(code)
                        except Exception:
                            _stock_names[code] = ""

                # 종목풀 JSON 저장 (대시보드용)
                import json
                pool_data = [
                    {"code": c, "name": _stock_names.get(c, ""), "score": round(pool.scores.get(c, 0), 4)}
                    for c in pool.codes
                ]
                pool_json = {
                    "date": today,
                    "count": len(pool.codes),
                    "entered": pool.entered,
                    "exited": pool.exited,
                    "stocks": pool_data,
                }
                Path("data").mkdir(exist_ok=True)
                Path("data/current_pool.json").write_text(json.dumps(pool_json, ensure_ascii=False, indent=2))

                logger.info(f"종목풀 업데이트: {len(pool.codes)}종목 (신규 {len(pool.entered)}, 퇴출 {len(pool.exited)})")
                notifier.notify_start()  # 종목풀 갱신 알림
                sync_runtime_status()
            else:
                logger.warning("종목풀이 비어 있습니다 — 기존 풀 유지")
                sync_runtime_status()
        except Exception as e:
            logger.error(f"종목풀 구성 실패: {e}")
            risk_mgr.record_error()
            sync_runtime_status()
            notifier.notify_error(str(e), "종목풀 리밸런싱")

    def _restore_pool_from_disk() -> bool:
        """data/current_pool.json 에서 이전 풀을 복원합니다.

        재시작 시 기존 풀을 유지해 pool-diff 매매 폭증을 방지.
        리밸런싱은 cron 일정(매월 1일)에만 실행.
        성공 시 True, 실패 또는 풀 이상(<10종목) 시 False.
        """
        from src.factors.stock_pool import StockPool
        from src.utils.json_io import load_json_with_fallback
        data = load_json_with_fallback("data/current_pool.json")
        if data is None:
            return False
        try:
            stocks = data.get("stocks", [])
            if len(stocks) < 10:
                logger.warning(f"저장된 풀이 {len(stocks)}종목뿐 — 재빌드")
                return False
            codes = [s["code"] for s in stocks]
            scores = {s["code"]: s.get("score", 0.0) for s in stocks}
            pool = StockPool(
                date=data.get("date", ""),
                codes=codes,
                scores=scores,
                entered=[],
                exited=[],
            )
            strategy.update_pool(pool.codes)
            _current_pool[0] = pool
            for s in stocks:
                if s.get("name"):
                    _stock_names[s["code"]] = s["name"]
            logger.info(
                f"이전 종목풀 복원: {len(pool.codes)}종목 (date={pool.date}) "
                f"— 리밸런싱은 cron 일정에서만 실행"
            )
            sync_runtime_status()
            return True
        except Exception as e:
            logger.warning(f"풀 복원 실패: {e} — 재빌드")
            return False

    # 시작 시: 기존 풀 복원 시도, 실패하면 새로 빌드
    logger.info("종목풀 복원/구성 중...")
    if not _restore_pool_from_disk():
        rebalance_pool()

    # LLM 시그널 검증기
    from src.timing.llm_validator import SignalValidator
    validator = SignalValidator()
    if validator.is_enabled:
        logger.info("LLM 시그널 검증 활성화 (Claude Haiku)")
    else:
        logger.info("LLM 시그널 검증 비활성화 — ANTHROPIC_API_KEY 미설정")

    # Notion 일일 리포터
    from src.notification.notion_reporter import NotionReporter
    notion = NotionReporter()
    if notion.is_enabled:
        logger.info("Notion 일일 리포트 활성화")
    else:
        logger.info("Notion 일일 리포트 비활성화 — NOTION_TOKEN/NOTION_DATABASE_ID 미설정")

    notifier.notify_start()
    sync_runtime_status()

    # 스케줄러
    scheduler = BlockingScheduler()

    def check_signals():
        """주기적 시그널 체크"""
        if not risk_mgr.is_trading_allowed:
            return

        try:
            balance = account_mgr.get_balance()
            risk_mgr.check_daily_loss_limit(balance.total_pnl, balance.total_eval)
            sync_runtime_status()

            # 종목명 캐시 업데이트
            for pos in balance.positions:
                if pos.stock_name and pos.stock_code not in _stock_names:
                    _stock_names[pos.stock_code] = pos.stock_name

            # 포지션 가격 스냅샷 (balance 시점) — sync_positions 에 재사용
            _pos_prices = {p.stock_code: float(p.current_price) for p in balance.positions}
            _pos_avg = {p.stock_code: float(p.avg_price) for p in balance.positions}

            def _sync_strategy_now(held: set[str]) -> None:
                """보유 종목 변경 직후 전략 내부 상태를 즉시 동기화."""
                if not hasattr(strategy, "sync_positions"):
                    return
                try:
                    strategy.sync_positions(held, _pos_prices, avg_prices=_pos_avg)
                except Exception as e:
                    logger.debug(f"sync_positions 실패: {e}")

            # ── 손절 체크 (LLM 검증 없이 즉시 실행) ──
            stop_codes = set(risk_mgr.check_stop_loss(balance.positions))
            # 손절된 종목은 같은 사이클에서 재매수 방지
            held_codes = {
                pos.stock_code for pos in balance.positions
                if pos.stock_code not in stop_codes
            }
            for pos in balance.positions:
                if pos.stock_code not in stop_codes:
                    continue
                try:
                    order = order_mgr.sell(pos.stock_code, pos.quantity, reference_price=int(pos.current_price))
                    if order.order_no:
                        notifier.notify_stop_loss(pos)
                        logger.info(
                            f"손절 매도: {pos.stock_name}({pos.stock_code}) "
                            f"{pos.pnl_pct:.2f}% / {pos.pnl:,}원"
                        )
                        save_trade(
                            stock_code=pos.stock_code,
                            stock_name=pos.stock_name,
                            side="sell",
                            quantity=pos.quantity,
                            price=int(pos.current_price),
                            order_no=order.order_no,
                            strategy=config.strategy.name,
                            signal_reason=f"stop_loss {pos.pnl_pct:.2f}%",
                        )
                        remove_holding(pos.stock_code)
                        held_codes.discard(pos.stock_code)
                        _sync_strategy_now(held_codes)
                except Exception as e:
                    logger.error(f"손절 매도 실패: {pos.stock_code} - {e}")

            # 손절이 한 건도 없었어도 한 번은 동기화 (재시작/편집 후 초기화용)
            if not stop_codes:
                _sync_strategy_now(held_codes)

            # 종목풀 내 종목에 대해 시그널 체크
            pool_codes = list(strategy._pool) if hasattr(strategy, '_pool') else []

            # 풀에서 퇴출된 보유 종목은 강제 매도 (리밸런싱 누락 방지)
            exited_codes = held_codes - set(pool_codes)
            if exited_codes:
                logger.info(f"풀 퇴출 종목 강제 매도: {len(exited_codes)}종목")
                for pos in balance.positions:
                    if pos.stock_code not in exited_codes:
                        continue
                    try:
                        order = order_mgr.sell(pos.stock_code, pos.quantity, reference_price=int(pos.current_price))
                        if order.order_no:
                            notifier.notify_order_filled(order)
                            logger.info(
                                f"풀 퇴출 매도: {pos.stock_name}({pos.stock_code}) "
                                f"{pos.quantity}주"
                            )
                            save_trade(
                                stock_code=pos.stock_code,
                                stock_name=pos.stock_name,
                                side="sell",
                                quantity=pos.quantity,
                                price=int(pos.current_price),
                                order_no=order.order_no,
                                strategy=config.strategy.name,
                                signal_reason="pool_exit",
                            )
                            remove_holding(pos.stock_code)
                            save_signal_log(
                                stock_code=pos.stock_code,
                                stock_name=pos.stock_name,
                                signal="SELL",
                                decision="확정",
                                reason="종목풀 퇴출 강제 매도",
                                signal_type="pool_exit",
                            )
                            held_codes.discard(pos.stock_code)
                            _sync_strategy_now(held_codes)
                    except Exception as e:
                        logger.error(f"풀 퇴출 매도 실패: {pos.stock_code} - {e}")
            signal_summary = {"BUY": 0, "SELL": 0, "HOLD": 0, "error": 0}
            for code in pool_codes:
                try:
                    ohlcv = client.get_daily_ohlcv(code)
                except Exception as e:
                    logger.warning(f"일봉 조회 실패 [{code}] — 스킵: {e}")
                    signal_summary["error"] += 1
                    continue
                if not ohlcv:
                    signal_summary["error"] += 1
                    continue

                df = pd.DataFrame(ohlcv)
                stock_name = _stock_names.get(code, "")
                try:
                    signal = strategy.generate_signal(code, df, stock_name=stock_name)
                except TypeError:
                    signal = strategy.generate_signal(code, df)
                if not signal.stock_name:
                    signal.stock_name = stock_name
                signal_summary[signal.signal.value] = signal_summary.get(signal.signal.value, 0) + 1

                # 모델 추론 로그 (#4) — 전략이 _last_diag 를 노출하면 jsonl 적재
                diag = getattr(strategy, "_last_diag", None)
                if diag:
                    from src.runtime.inference_logger import log_inference
                    log_inference(code, stock_name, signal.signal.value, diag)

                if signal.is_actionable:
                    # LLM 검증: ML 시그널을 기술적 맥락에서 확인
                    confirmed, llm_reason = validator.validate_signal(
                        code, signal.stock_name, signal.signal.value,
                        signal.reason, df,
                    )
                    save_signal_log(
                        stock_code=code,
                        stock_name=signal.stock_name,
                        signal=signal.signal.value,
                        decision="확정" if confirmed else "보류",
                        reason=llm_reason,
                    )
                    if not confirmed:
                        logger.info(f"LLM 보류: {signal.stock_name} {signal.signal.value} — {llm_reason}")
                        continue

                    notifier.notify_signal(signal)

                    if signal.signal.value == "BUY":
                        # 이미 보유 중인 종목은 추가 매수하지 않음
                        if code in held_codes:
                            continue
                        qty = risk_mgr.calculate_position_size(
                            signal, balance.total_deposit,
                            balance.total_eval, len(balance.positions),
                            regime_label=_current_regime[0],
                        )
                        if qty > 0:
                            valid, reason = risk_mgr.validate_order(signal, qty, balance, df=df)
                            if valid:
                                order = order_mgr.buy(code, qty, reference_price=int(signal.price))
                                if order.order_no:
                                    notifier.notify_order_filled(order)
                                    save_trade(
                                        stock_code=code,
                                        stock_name=stock_name,
                                        side="buy",
                                        quantity=qty,
                                        price=int(signal.price),
                                        order_no=order.order_no,
                                        strategy=config.strategy.name,
                                        signal_strength=float(signal.strength),
                                        signal_reason=signal.reason or "",
                                    )
                                    # DB에 매수일 기록
                                    save_holding(
                                        stock_code=code,
                                        stock_name=stock_name,
                                        avg_price=signal.price,
                                        quantity=qty,
                                        buy_date=datetime.now().strftime("%Y%m%d"),
                                    )
                                    held_codes.add(code)
                                    _pos_prices[code] = float(signal.price)
                                    _pos_avg[code] = float(signal.price)
                                    _sync_strategy_now(held_codes)
                    else:
                        # 보유 중이면 전량 매도
                        for pos in balance.positions:
                            if pos.stock_code == code:
                                order = order_mgr.sell(code, pos.quantity, reference_price=int(pos.current_price))
                                if order.order_no:
                                    notifier.notify_order_filled(order)
                                    save_trade(
                                        stock_code=code,
                                        stock_name=pos.stock_name,
                                        side="sell",
                                        quantity=pos.quantity,
                                        price=int(pos.current_price),
                                        order_no=order.order_no,
                                        strategy=config.strategy.name,
                                        signal_strength=float(signal.strength),
                                        signal_reason=signal.reason or "",
                                    )
                                    # DB에서 매도 종목 제거
                                    remove_holding(code)
                                    held_codes.discard(code)
                                    _sync_strategy_now(held_codes)

            # 주문 체결 확인 → 체결된 주문은 확정 알림 재발송
            try:
                filled = order_mgr.check_filled()
                for f_order in filled:
                    notifier.notify_order_filled(f_order)
            except Exception as e:
                logger.warning(f"체결 확인 실패: {e}")

            summary_msg = f"BUY:{signal_summary['BUY']} SELL:{signal_summary['SELL']} HOLD:{signal_summary['HOLD']} 오류:{signal_summary['error']}"
            logger.info(f"시그널 체크 완료: {len(pool_codes)}종목 — {summary_msg}")
            save_signal_log(
                stock_code="",
                stock_name=f"{len(pool_codes)}종목 스캔",
                signal="SCAN",
                decision=f"B:{signal_summary['BUY']} S:{signal_summary['SELL']} H:{signal_summary['HOLD']}",
                reason=f"오류: {signal_summary['error']}" if signal_summary['error'] > 0 else "정상",
                signal_type="summary",
            )
            risk_mgr.reset_error_count()
            sync_runtime_status()

            # 일일 수익 스냅샷 (대시보드 Cumulative Return 차트용)
            save_daily_pnl(
                total_eval=balance.total_eval,
                total_deposit=balance.total_deposit,
                total_pnl=balance.total_pnl,
                total_pnl_pct=balance.total_pnl_pct,
                num_positions=len(balance.positions),
            )

        except Exception as e:
            logger.error(f"시그널 체크 오류: {e}")
            risk_mgr.record_error()
            sync_runtime_status()
            notifier.notify_error(str(e))

    def daily_report():
        """일일 리포트 + LLM 피드백"""
        if _skip_if_market_closed("daily_report"):
            return
        try:
            balance = account_mgr.get_balance()
            notifier.notify_daily_report(balance)
            from src.db.models import save_daily_pnl as _save_daily_pnl, get_today_trades
            _save_daily_pnl(
                balance.total_eval, balance.total_deposit,
                balance.total_pnl, balance.total_pnl_pct,
                len(balance.positions),
            )

            # LLM 일일 매매 피드백
            today_trades = get_today_trades()
            feedback = validator.generate_daily_feedback(
                trades=today_trades,
                positions=balance.positions,
                total_pnl=balance.total_pnl,
                total_pnl_pct=balance.total_pnl_pct,
                total_eval=balance.total_eval,
            )
            if feedback:
                notifier.notify_daily_feedback(feedback)
                logger.info(f"일일 피드백 전송 완료")

            # Notion 일일 리포트
            notion.publish_daily_report(
                balance=balance,
                trades=today_trades,
                feedback=feedback or "",
            )

            # Codex CLI 리뷰 (설정에서 활성화된 경우)
            if config.codex.enabled and config.codex.daily_review:
                _trigger_codex_daily_review(config.codex.model)

        except Exception as e:
            logger.error(f"리포트 오류: {e}")

    def weekly_retrain():
        """주간 모델 재학습 — 최신 데이터로 XGBoost를 재학습하고 성능이 개선되면 교체합니다."""
        from src.data.market_data import get_universe, get_ohlcv_batch
        from src.timing.retrain import retrain_model

        model_type = config.strategy.name.replace("factor_", "")
        ext = ".pkl" if model_type in ("decision_tree", "xgboost", "lightgbm") else ".pt"
        model_path = f"models/{model_type}_timing{ext}"

        logger.info(f"주간 모델 재학습 시작: {model_type}")
        try:
            # 최신 OHLCV 데이터 수집 (최근 5년 — 약세장/regime shift 포함)
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=1825)).strftime("%Y%m%d")
            universe = get_universe()
            codes = universe["code"].tolist()[:100]
            ohlcv_dict = get_ohlcv_batch(codes, start, end)

            result = retrain_model(ohlcv_dict, model_type, model_path)

            if result.get("replaced"):
                # 모델이 교체되었으면 predictor 리로드
                if hasattr(strategy, 'predictor'):
                    strategy.predictor.model.load(model_path)
                    logger.info("라이브 전략에 새 모델 반영 완료")

                msg = (
                    f"모델 재학습 완료 — 교체됨\n"
                    f"• accuracy: {result['new_accuracy']:.3f}\n"
                    f"• F1: {result['new_f1']:.3f}\n"
                    f"• 학습: {result['train_samples']:,}건, 검증: {result['val_samples']:,}건\n"
                    f"• 시그널: BUY {result['signal_dist']['buy']}, "
                    f"SELL {result['signal_dist']['sell']}, "
                    f"HOLD {result['signal_dist']['hold']}"
                )
            else:
                msg = f"모델 재학습 완료 — 기존 모델 유지 (성능 개선 없음)"

            logger.info(msg)
            notifier._send(f"🔄 *{msg}*")

        except Exception as e:
            logger.error(f"모델 재학습 실패: {e}")
            risk_mgr.record_error()
            sync_runtime_status()
            notifier.notify_error(str(e), "주간 모델 재학습")

    # 스케줄 등록
    scheduler.add_job(
        check_signals, "interval",
        seconds=config.schedule.check_interval_sec,
        start_date=f"{datetime.now().strftime('%Y-%m-%d')} {config.schedule.market_open}:00",
    )
    scheduler.add_job(
        daily_report, "cron",
        hour=int(config.schedule.post_market.split(":")[0]),
        minute=int(config.schedule.post_market.split(":")[1]),
    )
    # 월간 리밸런싱 (매월 첫 거래일, 장 시작 전)
    pre_h, pre_m = config.schedule.pre_market.split(":")
    scheduler.add_job(
        rebalance_pool, "cron",
        day=config.schedule.rebalance_day,
        hour=int(pre_h), minute=int(pre_m),
    )

    # 매주 토요일 새벽 모델 재학습
    scheduler.add_job(
        weekly_retrain, "cron",
        day_of_week="sat", hour=6, minute=0,
    )

    # ── SAPPO 파이프라인: 일일 뉴스/sentiment + 주간 리포트 ──
    def _run_sappo_script(script_args: list[str], tag: str, timeout: int) -> None:
        logger.info(f"[{tag}] 실행 시작: {' '.join(script_args)}")
        try:
            result = subprocess.run(
                [sys.executable, *script_args],
                check=False,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
            )
            logger.info(f"[{tag}] 완료 rc={result.returncode}")
            if result.stdout:
                logger.debug(f"[{tag} stdout] {result.stdout[-500:]}")
            if result.returncode != 0 and result.stderr:
                logger.warning(f"[{tag} stderr] {result.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            logger.error(f"[{tag}] {timeout}초 초과 — 강제 종료됨")
        except FileNotFoundError:
            logger.warning(f"[{tag}] 스크립트 없음 — 건너뜀")
        except Exception as e:
            logger.warning(f"[{tag}] 실행 실패: {e}")

    def daily_sappo_fetch():
        if _skip_if_market_closed("SAPPO"):
            return
        _run_sappo_script(["scripts/fetch_daily_news.py"], "SAPPO", timeout=900)

    def daily_market_news_fetch():
        if _skip_if_market_closed("REGIME news"):
            return
        if not config.regime.news_fetch_enabled:
            logger.debug("[REGIME] news_fetch_enabled=False — 시장 뉴스 수집 skip")
            return
        _run_sappo_script(["scripts/fetch_market_news.py"], "REGIME news", timeout=600)

    def daily_foreign_buys_fetch():
        """외국인 매매 데이터 일일 수집 (Mid-cap 풀 외국인 필터 입력).
        장 마감 후 16:30 — 당일 종가 기준 KIS API 데이터 갱신."""
        if _skip_if_market_closed("FOREIGN BUYS"):
            return
        if not config.factors.foreign_filter_enabled:
            logger.debug("[FOREIGN BUYS] foreign_filter_enabled=False — skip")
            return
        _run_sappo_script(
            ["scripts/fetch_foreign_buys.py", "--universe", "all"],
            "FOREIGN BUYS", timeout=900,
        )

    def detect_regime_daily():
        if _skip_if_market_closed("REGIME detect"):
            return
        if not config.regime.enabled:
            logger.debug("[REGIME] enabled=False — detect skip")
            return
        try:
            from src.regime.detector import RegimeDetector
            from src.db.sappo_models import upsert_regime_label, upsert_macro_feature
            today = datetime.now().strftime("%Y%m%d")

            # A1+A2: macro feature(USD/KRW + VIX) 당일치 incremental fetch
            try:
                import FinanceDataReader as fdr
                import numpy as np
                start = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")
                end = datetime.now().strftime("%Y-%m-%d")
                usd = fdr.DataReader("USD/KRW", start, end)
                vix = fdr.DataReader("VIX", start, end)
                if not usd.empty and not vix.empty:
                    usd_close = float(usd["Close"].iloc[-1])
                    usd_prev = float(usd["Close"].iloc[-2]) if len(usd) >= 2 else usd_close
                    vix_close = float(vix["Close"].iloc[-1])
                    vix_prev = float(vix["Close"].iloc[-2]) if len(vix) >= 2 else vix_close
                    upsert_macro_feature(
                        date=today,
                        usdkrw_close=usd_close,
                        usdkrw_log_ret=float(np.log(usd_close / usd_prev)) if usd_prev > 0 else 0.0,
                        usdkrw_vol_20d=float(np.log(usd["Close"]).diff().tail(20).std()),
                        vix_close=vix_close,
                        vix_log_ret=float(np.log(vix_close / vix_prev)) if vix_prev > 0 else 0.0,
                    )
                    logger.info(f"[MACRO] {today} usdkrw={usd_close:.2f} vix={vix_close:.2f} 저장")
            except Exception as me:
                logger.warning(f"[MACRO] daily fetch 실패: {me} — detect 는 KOSPI-only fallback")

            result = RegimeDetector().detect_today(today)
            upsert_regime_label(
                date=today,
                label=result.label,
                hmm_state=result.hmm_state,
                hmm_probs=result.hmm_probs_tuple,
                kospi_return_60d=result.kospi_return_60d,
                kospi_vol_60d=result.kospi_vol_60d,
                llm_score=result.llm_score,
                overridden_by_llm=result.overridden_by_llm,
                notes=result.notes,
            )
            prev_label = _current_regime[0]
            _current_regime[0] = result.label
            try:
                changed_msg = (
                    f" (이전: {prev_label})" if prev_label and prev_label != result.label else ""
                )
                lam_note = (
                    f" λ={config.regime.lambda_}" if config.regime.lambda_ > 0 else " (dark-launch λ=0)"
                )
                notifier.notify_info(
                    f"[REGIME] {today} → *{result.label}*{changed_msg}\n"
                    f"posts={ {k: round(v,2) for k,v in result.hmm_probs.items()} } "
                    f"ret60d={result.kospi_return_60d:+.3f} vol60d={result.kospi_vol_60d:.3f} "
                    f"llm={result.llm_score} override={result.overridden_by_llm}{lam_note}"
                )
            except Exception:
                pass
        except Exception as e:
            logger.error(f"[REGIME] detect_regime_daily 실패: {e}")
            try:
                notifier.notify_error(str(e), "Regime detect")
            except Exception:
                pass

    def weekly_sappo_report():
        _run_sappo_script(
            ["scripts/weekly_sappo_report.py", "--slack"], "SAPPO 주간", timeout=600
        )

    # 일일 SAPPO 수집 — post_market 이후 (16:00)
    scheduler.add_job(
        daily_sappo_fetch, "cron",
        hour=16, minute=0,
    )
    # 일일 외국인 매매 수집 — SAPPO 다음 (16:30) — Mid-cap 풀 외국인 필터 입력
    if config.factors.foreign_filter_enabled:
        scheduler.add_job(
            daily_foreign_buys_fetch, "cron",
            hour=16, minute=30, id="foreign_buys_fetch",
        )
        logger.info("[FOREIGN BUYS] 스케줄 등록: 매일 16:30")
    # ── Regime Detector — 매일 장 시작 전 (08:30 시장 뉴스, 08:45 detect) ──
    if config.regime.enabled:
        scheduler.add_job(
            daily_market_news_fetch, "cron",
            hour=8, minute=30, id="regime_news_fetch",
        )
        scheduler.add_job(
            detect_regime_daily, "cron",
            hour=8, minute=45, id="regime_detect",
        )
        logger.info("[REGIME] 스케줄 등록: 시장 뉴스 08:30, regime detect 08:45")
    # 주간 SAPPO 리포트 — 매주 토요일 오전 (weekly_retrain 직후 07:00)
    scheduler.add_job(
        weekly_sappo_report, "cron",
        day_of_week="sat", hour=7, minute=0,
    )

    scheduler.add_job(
        lambda: logger.info("[HEARTBEAT] scheduler alive"),
        "interval", minutes=5, id="heartbeat",
    )

    from apscheduler.events import (
        EVENT_JOB_ERROR, EVENT_JOB_MISSED, EVENT_JOB_MAX_INSTANCES,
    )

    _EVENT_META = {
        EVENT_JOB_ERROR: ("error", "잡 실행 실패", "scheduler job"),
        EVENT_JOB_MISSED: ("warning", "잡 실행 누락 (스케줄 시간 경과)", "scheduler miss"),
        EVENT_JOB_MAX_INSTANCES: ("warning", "잡 최대 동시 실행 초과 (이전 실행 미완료)", "scheduler max_instances"),
    }

    def _scheduler_event_listener(event):
        level, desc, ctx = _EVENT_META.get(event.code, ("warning", "알 수 없는 이벤트", "scheduler"))
        job_id = getattr(event, "job_id", "?")
        exc = getattr(event, "exception", None)
        msg = f"{desc}: {job_id}" + (f" — {exc}" if exc else "")
        getattr(logger, level)(f"[SCHEDULER] {msg}")
        notifier.notify_error(str(exc) if exc else msg, f"{ctx}: {job_id}")

    scheduler.add_listener(
        _scheduler_event_listener,
        EVENT_JOB_ERROR | EVENT_JOB_MISSED | EVENT_JOB_MAX_INSTANCES,
    )

    logger.info(
        f"스케줄 시작: {config.schedule.check_interval_sec}초 간격, "
        f"리밸런싱 매월 {config.schedule.rebalance_day}일, "
        f"재학습 매주 토요일, SAPPO 수집 매일 16:00, "
        f"SAPPO 리포트 매주 토요일 07:00, heartbeat 5분"
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("스케줄러 정상 종료")
    except Exception as e:
        logger.critical(f"스케줄러 비정상 종료: {e}")
        sync_runtime_status()
        notifier.notify_error(str(e), "스케줄러 크래시")
    finally:
        sync_runtime_status()
        scheduler.shutdown(wait=False)
