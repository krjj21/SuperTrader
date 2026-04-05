"""
주문 모듈
- 매수/매도 주문 실행
- 주문 조회/취소
- 체결 확인
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from loguru import logger

from src.broker.kis_client import KISClient
from src.config import get_config


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "01"     # 시장가
    LIMIT = "00"      # 지정가


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class Order:
    stock_code: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    price: int = 0                          # 지정가 주문 시 가격
    status: OrderStatus = OrderStatus.PENDING
    order_no: str = ""                      # KIS 주문번호
    filled_qty: int = 0
    filled_price: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    error_msg: str = ""


class OrderManager:
    """주문 실행 및 관리"""

    def __init__(self, client: KISClient | None = None):
        self.client = client or KISClient()
        self.config = get_config()
        self._pending_orders: dict[str, Order] = {}

    def place_order(self, order: Order) -> Order:
        """주문을 실행합니다."""
        try:
            if order.side == OrderSide.BUY:
                result = self._place_buy(order)
            else:
                result = self._place_sell(order)

            order.order_no = result.get("output", {}).get("ODNO", "")
            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.now()

            self._pending_orders[order.order_no] = order
            logger.info(
                f"주문 제출 완료: {order.side.value} {order.stock_code} "
                f"{order.quantity}주 (주문번호: {order.order_no})"
            )

        except Exception as e:
            order.status = OrderStatus.FAILED
            order.error_msg = str(e)
            order.updated_at = datetime.now()
            logger.error(f"주문 실패: {order.stock_code} - {e}")

        return order

    def _place_buy(self, order: Order) -> dict:
        """매수 주문을 실행합니다."""
        # 모의투자 vs 실전투자 tr_id가 다름
        tr_id = "VTTC0802U" if self.config.kis.is_virtual else "TTTC0802U"

        payload = {
            "CANO": self.client.secrets.kis_account_no.split("-")[0],
            "ACNT_PRDT_CD": self.client.secrets.kis_account_no.split("-")[1],
            "PDNO": order.stock_code,
            "ORD_DVSN": order.order_type.value,
            "ORD_QTY": str(order.quantity),
            "ORD_UNPR": str(order.price) if order.order_type == OrderType.LIMIT else "0",
        }

        return self.client.post(
            "/uapi/domestic-stock/v1/trading/order-cash",
            tr_id,
            payload,
        )

    def _place_sell(self, order: Order) -> dict:
        """매도 주문을 실행합니다."""
        tr_id = "VTTC0801U" if self.config.kis.is_virtual else "TTTC0801U"

        payload = {
            "CANO": self.client.secrets.kis_account_no.split("-")[0],
            "ACNT_PRDT_CD": self.client.secrets.kis_account_no.split("-")[1],
            "PDNO": order.stock_code,
            "ORD_DVSN": order.order_type.value,
            "ORD_QTY": str(order.quantity),
            "ORD_UNPR": str(order.price) if order.order_type == OrderType.LIMIT else "0",
        }

        return self.client.post(
            "/uapi/domestic-stock/v1/trading/order-cash",
            tr_id,
            payload,
        )

    def cancel_order(self, order_no: str, stock_code: str, quantity: int) -> dict:
        """주문을 취소합니다."""
        tr_id = "VTTC0803U" if self.config.kis.is_virtual else "TTTC0803U"

        payload = {
            "CANO": self.client.secrets.kis_account_no.split("-")[0],
            "ACNT_PRDT_CD": self.client.secrets.kis_account_no.split("-")[1],
            "KRX_FWDG_ORD_ORGNO": "",
            "ORGN_ODNO": order_no,
            "ORD_DVSN": "00",
            "RVSE_CNCL_DVSN_CD": "02",  # 취소
            "ORD_QTY": str(quantity),
            "ORD_UNPR": "0",
            "QTY_ALL_ORD_YN": "Y",
        }

        result = self.client.post(
            "/uapi/domestic-stock/v1/trading/order-rvsecncl",
            tr_id,
            payload,
        )

        if order_no in self._pending_orders:
            self._pending_orders[order_no].status = OrderStatus.CANCELLED
            self._pending_orders[order_no].updated_at = datetime.now()

        logger.info(f"주문 취소 완료: {order_no}")
        return result

    def check_filled(self) -> list[Order]:
        """미체결 주문의 체결 상태를 확인합니다."""
        tr_id = "VTTC8001R" if self.config.kis.is_virtual else "TTTC8001R"

        params = {
            "CANO": self.client.secrets.kis_account_no.split("-")[0],
            "ACNT_PRDT_CD": self.client.secrets.kis_account_no.split("-")[1],
            "INQR_STRT_DT": datetime.now().strftime("%Y%m%d"),
            "INQR_END_DT": datetime.now().strftime("%Y%m%d"),
            "SLL_BUY_DVSN_CD": "00",  # 전체
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

        data = self.client.get(
            "/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
            tr_id,
            params,
        )

        filled_orders = []
        for item in data.get("output1", []):
            order_no = item.get("odno", "")
            if order_no in self._pending_orders:
                order = self._pending_orders[order_no]
                filled_qty = int(item.get("tot_ccld_qty", 0))
                filled_price = int(float(item.get("avg_prvs", 0)))

                if filled_qty > 0:
                    order.filled_qty = filled_qty
                    order.filled_price = filled_price
                    order.status = (
                        OrderStatus.FILLED
                        if filled_qty >= order.quantity
                        else OrderStatus.PARTIAL
                    )
                    order.updated_at = datetime.now()
                    filled_orders.append(order)
                    logger.info(
                        f"체결 확인: {order.stock_code} {filled_qty}주 @ {filled_price:,}원"
                    )

        return filled_orders

    def buy(
        self,
        stock_code: str,
        quantity: int,
        price: int = 0,
        order_type: OrderType = OrderType.MARKET,
    ) -> Order:
        """매수 주문 헬퍼"""
        order = Order(
            stock_code=stock_code,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=order_type,
            price=price,
        )
        return self.place_order(order)

    def sell(
        self,
        stock_code: str,
        quantity: int,
        price: int = 0,
        order_type: OrderType = OrderType.MARKET,
    ) -> Order:
        """매도 주문 헬퍼"""
        order = Order(
            stock_code=stock_code,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=order_type,
            price=price,
        )
        return self.place_order(order)
