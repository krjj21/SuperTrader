"""
계좌 조회 모듈
- 잔고/보유종목 조회
- 수익률 조회
- 예수금 조회
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from src.broker.kis_client import KISClient
from src.config import get_config


@dataclass
class Position:
    stock_code: str
    stock_name: str
    quantity: int
    avg_price: int
    current_price: int
    pnl: int              # 평가손익
    pnl_pct: float         # 수익률 (%)
    eval_amount: int       # 평가금액


@dataclass
class AccountSummary:
    total_eval: int        # 총 평가금액 (순자산)
    total_deposit: int     # 예수금 (D+2 결제 전 실제 현금)
    total_pnl: int         # 총 평가손익 (보유종목 평가손익 합)
    total_pnl_pct: float   # 총 수익률
    positions: list[Position]
    available_cash: int = 0       # 주문 가능 금액 (매도 대금 포함)
    asset_change: int = 0         # 자산 증감액
    asset_change_pct: float = 0.0 # 자산 수익률 (%)


class AccountManager:
    """계좌 조회 관리"""

    def __init__(self, client: KISClient | None = None):
        self.client = client or KISClient()
        self.config = get_config()

    def get_balance(self) -> AccountSummary:
        """계좌 잔고를 조회합니다."""
        tr_id = "VTTC8434R" if self.config.kis.is_virtual else "TTTC8434R"

        params = {
            "CANO": self.client.secrets.kis_account_no.split("-")[0],
            "ACNT_PRDT_CD": self.client.secrets.kis_account_no.split("-")[1],
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        data = self.client.get(
            "/uapi/domestic-stock/v1/trading/inquire-balance",
            tr_id,
            params,
        )

        positions = []
        for item in data.get("output1", []):
            qty = int(item.get("hldg_qty", 0))
            if qty <= 0:
                continue

            avg_price = int(float(item.get("pchs_avg_pric", 0)))
            cur_price = int(item.get("prpr", 0))
            eval_amt = int(item.get("evlu_amt", 0))
            pnl = int(item.get("evlu_pfls_amt", 0))
            pnl_pct = float(item.get("evlu_pfls_rt", 0))

            positions.append(Position(
                stock_code=item.get("pdno", ""),
                stock_name=item.get("prdt_name", ""),
                quantity=qty,
                avg_price=avg_price,
                current_price=cur_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                eval_amount=eval_amt,
            ))

        output2 = data.get("output2", [{}])
        summary_data = output2[0] if output2 else {}

        total_pnl = int(summary_data.get("evlu_pfls_smtl_amt", 0))
        pnl_pct_api = float(summary_data.get("evlu_pfls_smtl_rt", 0)) if summary_data.get("evlu_pfls_smtl_rt") else 0.0
        # KIS 모의투자는 evlu_pfls_smtl_rt 가 0.0 으로 반환되는 경우가 있어 수동 계산 fallback
        if pnl_pct_api == 0.0 and total_pnl != 0:
            total_invested = sum(p.avg_price * p.quantity for p in positions)
            if total_invested > 0:
                pnl_pct_api = total_pnl / total_invested * 100.0

        summary = AccountSummary(
            total_eval=int(summary_data.get("tot_evlu_amt", 0)),
            total_deposit=int(summary_data.get("dnca_tot_amt", 0)),
            total_pnl=total_pnl,
            total_pnl_pct=pnl_pct_api,
            positions=positions,
            available_cash=int(summary_data.get("prvs_rcdl_excc_amt", 0)),
            asset_change=int(summary_data.get("asst_icdc_amt", 0)),
            asset_change_pct=float(summary_data.get("asst_icdc_erng_rt", 0)) if summary_data.get("asst_icdc_erng_rt") else 0.0,
        )

        logger.info(
            f"잔고 조회: 총평가 {summary.total_eval:,}원, "
            f"손익 {summary.total_pnl:,}원 ({summary.total_pnl_pct:.2f}%), "
            f"보유 {len(positions)}종목"
        )
        return summary

    def get_available_cash(self) -> int:
        """주문 가능 예수금을 조회합니다."""
        summary = self.get_balance()
        return summary.total_deposit

    def get_realized_pnl(self) -> dict:
        """당일 실현손익을 조회합니다."""
        # 모의투자는 실현손익 API 미지원 → DB fallback 직행
        if self.config.kis.is_virtual:
            return self._get_realized_pnl_from_db()

        tr_id = "TTTC8715R"

        today = datetime.now().strftime("%Y%m%d")
        params = {
            "CANO": self.client.secrets.kis_account_no.split("-")[0],
            "ACNT_PRDT_CD": self.client.secrets.kis_account_no.split("-")[1],
            "INQR_STRT_DT": today,
            "INQR_END_DT": today,
            "SLL_BUY_DVSN_CD": "00",
            "INQR_DVSN": "00",
            "PDNO": "",
            "CCLD_DVSN": "00",
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "INQR_DVSN_3": "00",
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        try:
            data = self.client.get(
                "/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
                tr_id,
                params,
            )

            output2 = data.get("output2", {})
            if isinstance(output2, list):
                output2 = output2[0] if output2 else {}

            realized_pnl = int(output2.get("rlzt_pfls", 0))
            total_sell = int(output2.get("sll_amt_smtl", 0))
            total_buy = int(output2.get("buy_amt_smtl", 0))
            n_trades = int(output2.get("tot_ccld_qty", 0))

            logger.info(f"실현손익 조회: {realized_pnl:,}원, 매도 {total_sell:,}원, 매수 {total_buy:,}원, 체결 {n_trades}건")
            return {
                "realized_pnl": realized_pnl,
                "total_sell": total_sell,
                "total_buy": total_buy,
                "n_trades": n_trades,
            }
        except Exception as e:
            logger.warning(f"실현손익 조회 실패: {e}")
            # DB fallback: trade_logs에서 당일 매매 집계
            return self._get_realized_pnl_from_db()

    @staticmethod
    def _get_realized_pnl_from_db() -> dict:
        """DB trade_logs에서 당일 매매 금액을 집계합니다 (KIS API fallback)."""
        try:
            from src.db.models import init_db, get_session, TradeLog
            init_db()
            session = get_session()
            today = datetime.now().strftime("%Y%m%d")
            cutoff = datetime.strptime(today, "%Y%m%d")
            trades = (
                session.query(TradeLog)
                .filter(TradeLog.created_at >= cutoff, TradeLog.status == "filled")
                .all()
            )
            session.close()

            total_buy = sum(t.amount for t in trades if t.side == "buy")
            total_sell = sum(t.amount for t in trades if t.side == "sell")
            n_trades = len(trades)
            realized_pnl = total_sell - total_buy

            return {
                "realized_pnl": realized_pnl,
                "total_sell": total_sell,
                "total_buy": total_buy,
                "n_trades": n_trades,
            }
        except Exception:
            return {"realized_pnl": 0, "total_sell": 0, "total_buy": 0, "n_trades": 0}
