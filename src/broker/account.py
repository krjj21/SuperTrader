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
    total_eval: int        # 총 평가금액
    total_deposit: int     # 예수금
    total_pnl: int         # 총 평가손익
    total_pnl_pct: float   # 총 수익률
    positions: list[Position]


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

        summary = AccountSummary(
            total_eval=int(summary_data.get("tot_evlu_amt", 0)),
            total_deposit=int(summary_data.get("dnca_tot_amt", 0)),
            total_pnl=int(summary_data.get("evlu_pfls_smtl_amt", 0)),
            total_pnl_pct=float(summary_data.get("evlu_pfls_smtl_rt", 0)) if summary_data.get("evlu_pfls_smtl_rt") else 0.0,
            positions=positions,
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
