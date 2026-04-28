"""계좌 현황 모드 — main.py 에서 분리됨 (2026-04-28).

  · run_status() — KIS 모의/실전 계좌의 평가금액·예수금·보유종목 출력
"""
from __future__ import annotations

from src.config import get_config


def run_status() -> None:
    """모의투자 계좌 현황을 간편하게 출력합니다."""
    config = get_config()

    from src.broker.kis_client import KISClient
    from src.broker.account import AccountManager

    print()
    env_label = "🏦 모의투자" if config.kis.is_virtual else "🏦 실전투자"
    print(f"  {env_label} 계좌 현황")
    print(f"  {'─' * 50}")

    try:
        client = KISClient()
        account_mgr = AccountManager(client)
        summary = account_mgr.get_balance()
    except Exception as e:
        print(f"  ❌ 계좌 조회 실패: {e}")
        return

    pnl_sign = "+" if summary.total_pnl >= 0 else ""
    pnl_color = "\033[91m" if summary.total_pnl < 0 else "\033[92m"
    reset = "\033[0m"

    print(f"  총 평가금액  : {summary.total_eval:>15,}원")
    print(f"  예수금       : {summary.total_deposit:>15,}원")
    print(
        f"  평가손익     : {pnl_color}{pnl_sign}{summary.total_pnl:>14,}원 "
        f"({pnl_sign}{summary.total_pnl_pct:.2f}%){reset}"
    )
    print(f"  보유 종목    : {len(summary.positions)}개")
    print()

    if summary.positions:
        print(f"  {'종목명':<12} {'수량':>6} {'평균가':>10} {'현재가':>10} {'손익':>12} {'수익률':>8}")
        print(f"  {'─' * 62}")

        for pos in summary.positions:
            ps = "+" if pos.pnl >= 0 else ""
            pc = "\033[91m" if pos.pnl < 0 else "\033[92m"
            print(
                f"  {pos.stock_name:<12} {pos.quantity:>5}주 "
                f"{pos.avg_price:>9,}  {pos.current_price:>9,}  "
                f"{pc}{ps}{pos.pnl:>10,}  {ps}{pos.pnl_pct:>6.1f}%{reset}"
            )

        print()
    else:
        print("  보유 종목 없음")
        print()
