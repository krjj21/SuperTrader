"""
Notion 일일 매매 리포트 모듈
- 장 마감 후 당일 실적을 Notion 데이터베이스에 자동 기록
- 보유 종목 상세, 매매 내역을 페이지 본문에 포함
"""
from __future__ import annotations

from datetime import datetime

from loguru import logger

from src.config import get_secrets


class NotionReporter:
    """Notion Daily Trading Log 기록"""

    def __init__(self):
        secrets = get_secrets()
        self.token = secrets.notion_token
        self.database_id = secrets.notion_database_id
        self._client = None

    @property
    def is_enabled(self) -> bool:
        return bool(self.token and self.database_id)

    @property
    def client(self):
        if self._client is None:
            from notion_client import Client
            self._client = Client(auth=self.token)
        return self._client

    def publish_daily_report(
        self,
        balance,
        trades: list = None,
        feedback: str = "",
    ) -> bool:
        """일일 리포트를 Notion에 기록합니다."""
        if not self.is_enabled:
            logger.debug("[Notion 미설정] 일일 리포트 건너뜀")
            return False

        today = datetime.now().strftime("%Y-%m-%d")
        trades = trades or []

        buy_count = sum(1 for t in trades if t.side == "buy")
        sell_count = sum(1 for t in trades if t.side == "sell")

        # 시스템 상태 판단
        status = "정상"
        if balance.total_pnl_pct <= -3.0:
            status = "손실제한"

        try:
            # 본문 생성
            children = self._build_content(balance, trades, feedback)

            # 기존 페이지 확인 (같은 날짜 중복 방지)
            existing = self._find_existing_page(today)

            if existing:
                # 기존 페이지 업데이트
                page_id = existing
                self.client.pages.update(
                    page_id=page_id,
                    properties=self._build_properties(
                        today, balance, buy_count, sell_count, status,
                    ),
                )
                self._clear_page_content(page_id)
                for block in children:
                    self.client.blocks.children.append(
                        block_id=page_id, children=[block],
                    )
                logger.info(f"Notion 일일 리포트 업데이트: {today}")
            else:
                # 새 페이지 생성
                self.client.pages.create(
                    parent={"database_id": self.database_id},
                    properties=self._build_properties(
                        today, balance, buy_count, sell_count, status,
                    ),
                    children=children,
                )
                logger.info(f"Notion 일일 리포트 생성: {today}")

            return True

        except Exception as e:
            logger.error(f"Notion 리포트 실패: {e}")
            return False

    def _build_properties(
        self, date: str, balance, buy_count: int, sell_count: int, status: str,
    ) -> dict:
        """Notion 페이지 속성을 생성합니다."""
        return {
            "날짜": {"title": [{"text": {"content": date}}]},
            "총평가": {"number": balance.total_eval},
            "예수금": {"number": balance.total_deposit},
            "평가손익": {"number": balance.total_pnl},
            "수익률": {"number": round(balance.total_pnl_pct / 100, 4)},
            "보유종목수": {"number": len(balance.positions)},
            "매수": {"number": buy_count},
            "매도": {"number": sell_count},
            "상태": {"select": {"name": status}},
        }

    def _build_content(self, balance, trades: list, feedback: str) -> list:
        """페이지 본문 블록을 생성합니다."""
        blocks = []

        # 계좌 요약
        blocks.append(self._heading("계좌 요약", level=2))
        pnl_sign = "+" if balance.total_pnl >= 0 else ""
        blocks.append(self._bulleted(f"총 평가금액: {balance.total_eval:,}원"))
        blocks.append(self._bulleted(f"예수금: {balance.total_deposit:,}원"))
        blocks.append(self._bulleted(
            f"평가손익: {pnl_sign}{balance.total_pnl:,}원 "
            f"({pnl_sign}{balance.total_pnl_pct:.2f}%)"
        ))
        blocks.append(self._bulleted(f"보유 종목: {len(balance.positions)}개"))
        blocks.append(self._divider())

        # 보유 종목 상세
        if balance.positions:
            blocks.append(self._heading("보유 종목", level=2))
            # 테이블 생성
            header = ["종목", "수량", "매입가", "현재가", "손익", "수익률"]
            rows = []
            for pos in sorted(balance.positions, key=lambda p: p.pnl_pct, reverse=True):
                pnl_s = f"{'+' if pos.pnl >= 0 else ''}{pos.pnl:,}"
                pct_s = f"{'+' if pos.pnl_pct >= 0 else ''}{pos.pnl_pct:.1f}%"
                rows.append([
                    f"{pos.stock_name} ({pos.stock_code})",
                    f"{pos.quantity:,}",
                    f"{pos.avg_price:,}",
                    f"{pos.current_price:,}",
                    pnl_s,
                    pct_s,
                ])
            blocks.extend(self._table(header, rows))
            blocks.append(self._divider())

        # 매매 내역
        if trades:
            blocks.append(self._heading("당일 매매", level=2))
            for t in trades:
                side = "매수" if t.side == "buy" else "매도"
                blocks.append(self._bulleted(
                    f"[{side}] {t.stock_name or t.stock_code} "
                    f"{t.quantity:,}주 @ {t.price:,}원 "
                    f"({t.signal_reason[:50]})"
                ))
            blocks.append(self._divider())

        # AI 피드백
        if feedback:
            blocks.append(self._heading("AI 매매 피드백", level=2))
            for paragraph in feedback.split("\n\n"):
                if paragraph.strip():
                    blocks.append(self._paragraph(paragraph.strip()))

        return blocks

    def _find_existing_page(self, date: str) -> str | None:
        """같은 날짜의 기존 페이지 ID를 반환합니다."""
        try:
            results = self.client.search(
                query=date,
                filter={"property": "object", "value": "page"},
                page_size=5,
            )
            for page in results.get("results", []):
                # database_id가 일치하는 페이지만 확인
                parent = page.get("parent", {})
                if parent.get("database_id", "").replace("-", "") != self.database_id.replace("-", ""):
                    continue
                title_prop = page.get("properties", {}).get("날짜", {})
                title_texts = title_prop.get("title", [])
                if title_texts and title_texts[0].get("plain_text") == date:
                    return page["id"]
        except Exception:
            pass
        return None

    def _clear_page_content(self, page_id: str):
        """페이지 본문의 기존 블록을 삭제합니다."""
        try:
            children = self.client.blocks.children.list(block_id=page_id)
            for block in children["results"]:
                self.client.blocks.delete(block_id=block["id"])
        except Exception:
            pass

    # ── Notion 블록 헬퍼 ──

    @staticmethod
    def _heading(text: str, level: int = 2) -> dict:
        return {
            f"heading_{level}": {
                "rich_text": [{"text": {"content": text}}],
            },
            "type": f"heading_{level}",
        }

    @staticmethod
    def _paragraph(text: str) -> dict:
        return {
            "paragraph": {
                "rich_text": [{"text": {"content": text[:2000]}}],
            },
            "type": "paragraph",
        }

    @staticmethod
    def _bulleted(text: str) -> dict:
        return {
            "bulleted_list_item": {
                "rich_text": [{"text": {"content": text[:2000]}}],
            },
            "type": "bulleted_list_item",
        }

    @staticmethod
    def _divider() -> dict:
        return {"type": "divider", "divider": {}}

    @staticmethod
    def _table(header: list[str], rows: list[list[str]]) -> list[dict]:
        """테이블 블록을 생성합니다."""
        width = len(header)

        def _row_cells(cells: list[str]) -> list:
            return [
                [{"type": "text", "text": {"content": c}}] for c in cells
            ]

        table_rows = [
            {"type": "table_row", "table_row": {"cells": _row_cells(header)}}
        ]
        for row in rows:
            table_rows.append(
                {"type": "table_row", "table_row": {"cells": _row_cells(row)}}
            )

        return [{
            "type": "table",
            "table": {
                "table_width": width,
                "has_column_header": True,
                "has_row_header": False,
                "children": table_rows,
            },
        }]
