"""
한국투자증권 KIS Open API 클라이언트
- OAuth 토큰 발급/갱신/캐싱
- REST API 호출 (rate limiting 포함)
- WebSocket 실시간 시세 수신
"""
from __future__ import annotations

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import websockets
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_config, get_secrets


# ──────────────────────────────────────────────
# 토큰 매니저
# ──────────────────────────────────────────────
class TokenManager:
    """Access Token 발급/갱신/캐싱을 관리합니다."""

    TOKEN_CACHE_FILE = "config/.token_cache.json"

    def __init__(self):
        self._token: str = ""
        self._expires_at: datetime = datetime.min
        self._lock = threading.Lock()

    @property
    def is_valid(self) -> bool:
        return bool(self._token) and datetime.now() < self._expires_at

    def get_token(self) -> str:
        """유효한 토큰을 반환합니다. 만료 시 자동 갱신."""
        with self._lock:
            if self.is_valid:
                return self._token

            # 캐시 파일에서 로드 시도
            if self._load_from_cache():
                return self._token

            # 새로 발급
            self._issue_token()
            return self._token

    def _issue_token(self) -> None:
        """KIS API에서 새 토큰을 발급받습니다."""
        secrets = get_secrets()
        config = get_config()

        url = f"{config.kis.base_url}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": secrets.kis_app_key,
            "appsecret": secrets.kis_app_secret,
        }

        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        self._token = data["access_token"]
        # KIS 토큰은 24시간 유효, 여유를 두고 20시간 후 만료 처리
        expires_str = data.get("access_token_token_expired", "")
        if expires_str:
            self._expires_at = datetime.strptime(expires_str, "%Y-%m-%d %H:%M:%S")
        else:
            self._expires_at = datetime.now() + timedelta(hours=config.kis.token_refresh_hours)

        self._save_to_cache()
        logger.info(f"토큰 발급 완료 (만료: {self._expires_at})")

    def _load_from_cache(self) -> bool:
        """캐시 파일에서 토큰을 로드합니다."""
        cache_path = Path(self.TOKEN_CACHE_FILE)
        if not cache_path.exists():
            return False

        try:
            data = json.loads(cache_path.read_text())
            expires_at = datetime.strptime(data["expires_at"], "%Y-%m-%d %H:%M:%S")
            if datetime.now() < expires_at:
                self._token = data["access_token"]
                self._expires_at = expires_at
                logger.info("캐시에서 토큰 로드 완료")
                return True
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return False

    def _save_to_cache(self) -> None:
        """토큰을 캐시 파일에 저장합니다."""
        cache_path = Path(self.TOKEN_CACHE_FILE)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "access_token": self._token,
            "expires_at": self._expires_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
        cache_path.write_text(json.dumps(data, indent=2))


# ──────────────────────────────────────────────
# REST API 클라이언트
# ──────────────────────────────────────────────
class KISClient:
    """KIS REST API 클라이언트 (rate limiting + 자동 재시도)"""

    # KIS API 초당 호출 제한
    RATE_LIMIT_PER_SEC = 10
    MIN_INTERVAL = 1.0 / RATE_LIMIT_PER_SEC

    def __init__(self):
        self.config = get_config()
        self.secrets = get_secrets()
        self.token_manager = TokenManager()
        self._last_call_time = 0.0
        self._rate_lock = threading.Lock()

    def _get_headers(self, tr_id: str) -> dict[str, str]:
        """API 요청 헤더를 생성합니다."""
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.token_manager.get_token()}",
            "appkey": self.secrets.kis_app_key,
            "appsecret": self.secrets.kis_app_secret,
            "tr_id": tr_id,
            "custtype": "P",  # 개인
        }

    def _rate_limit(self) -> None:
        """초당 호출 제한을 준수합니다."""
        with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            if elapsed < self.MIN_INTERVAL:
                time.sleep(self.MIN_INTERVAL - elapsed)
            self._last_call_time = time.monotonic()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def get(self, path: str, tr_id: str, params: dict[str, Any] | None = None) -> dict:
        """GET 요청을 실행합니다."""
        self._rate_limit()
        url = f"{self.config.kis.base_url}{path}"
        headers = self._get_headers(tr_id)

        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url, headers=headers, params=params or {})
            resp.raise_for_status()
            data = resp.json()

        if data.get("rt_cd") != "0":
            msg = data.get("msg1", "Unknown error")
            logger.error(f"KIS API 오류 [{tr_id}]: {msg}")
            raise KISAPIError(data.get("msg_cd", ""), msg)

        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def post(self, path: str, tr_id: str, payload: dict[str, Any] | None = None) -> dict:
        """POST 요청을 실행합니다."""
        self._rate_limit()
        url = f"{self.config.kis.base_url}{path}"
        headers = self._get_headers(tr_id)

        with httpx.Client(timeout=15.0) as client:
            resp = client.post(url, headers=headers, json=payload or {})
            resp.raise_for_status()
            data = resp.json()

        if data.get("rt_cd") != "0":
            msg = data.get("msg1", "Unknown error")
            logger.error(f"KIS API 오류 [{tr_id}]: {msg}")
            raise KISAPIError(data.get("msg_cd", ""), msg)

        return data

    # ──────────────────────────────────────────
    # 시세 조회 헬퍼
    # ──────────────────────────────────────────
    def get_current_price(self, stock_code: str) -> dict:
        """현재가를 조회합니다."""
        tr_id = "FHKST01010100" if not self.config.kis.is_virtual else "FHKST01010100"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code,
        }
        data = self.get("/uapi/domestic-stock/v1/quotations/inquire-price", tr_id, params)
        output = data.get("output", {})
        return {
            "code": stock_code,
            "price": int(output.get("stck_prpr", 0)),
            "change": int(output.get("prdy_vrss", 0)),
            "change_pct": float(output.get("prdy_ctrt", 0)),
            "volume": int(output.get("acml_vol", 0)),
            "high": int(output.get("stck_hgpr", 0)),
            "low": int(output.get("stck_lwpr", 0)),
            "open": int(output.get("stck_oprc", 0)),
        }

    def get_daily_ohlcv(
        self,
        stock_code: str,
        start_date: str = "",
        end_date: str = "",
        period: str = "D",
    ) -> list[dict]:
        """일봉/주봉/월봉 OHLCV 데이터를 조회합니다.

        Args:
            stock_code: 종목코드 (예: "005930")
            start_date: 시작일 (YYYYMMDD), 기본값은 100일 전
            end_date: 종료일 (YYYYMMDD), 기본값은 오늘
            period: "D"=일봉, "W"=주봉, "M"=월봉
        """
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=100)).strftime("%Y%m%d")

        tr_id = "FHKST03010100"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code,
            "FID_INPUT_DATE_1": start_date,
            "FID_INPUT_DATE_2": end_date,
            "FID_PERIOD_DIV_CODE": period,
            "FID_ORG_ADJ_PRC": "0",  # 수정주가
        }

        data = self.get(
            "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            tr_id,
            params,
        )

        records = []
        for item in data.get("output2", []):
            if not item.get("stck_bsop_date"):
                continue
            records.append({
                "date": item["stck_bsop_date"],
                "open": int(item.get("stck_oprc", 0)),
                "high": int(item.get("stck_hgpr", 0)),
                "low": int(item.get("stck_lwpr", 0)),
                "close": int(item.get("stck_clpr", 0)),
                "volume": int(item.get("acml_vol", 0)),
            })

        # 날짜 오름차순 정렬
        records.sort(key=lambda x: x["date"])
        return records

    def get_investor_trading(self, stock_code: str) -> list[dict]:
        """종목별 투자자 매매동향 (최근 30일).

        TR_ID: FHKST01010900 (종목별 투자자 매매동향).
        Returns:
            [{date, close, foreign_net_qty, foreign_net_amount, organ_net_qty, person_net_qty}, ...]
            (날짜 오름차순, 가장 최근일 1건은 장중에는 빈 값일 수 있음)
        """
        tr_id = "FHKST01010900"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code,
        }
        try:
            data = self.get(
                "/uapi/domestic-stock/v1/quotations/inquire-investor",
                tr_id, params,
            )
        except Exception as e:
            logger.warning(f"외국인 매매 조회 실패 [{stock_code}]: {e}")
            return []

        records = []
        for item in data.get("output", []):
            date = item.get("stck_bsop_date", "")
            if not date:
                continue
            # 장중 당일은 frgn_ntby_qty 가 빈 문자열일 수 있음 → 0 으로 처리
            def _to_int(v):
                try:
                    return int(v) if v not in (None, "", " ") else 0
                except (ValueError, TypeError):
                    return 0
            records.append({
                "date": date,
                "close": _to_int(item.get("stck_clpr")),
                "foreign_net_qty": _to_int(item.get("frgn_ntby_qty")),
                "foreign_net_amount": _to_int(item.get("frgn_ntby_tr_pbmn")),
                "organ_net_qty": _to_int(item.get("orgn_ntby_qty")),
                "organ_net_amount": _to_int(item.get("orgn_ntby_tr_pbmn")),
                "person_net_qty": _to_int(item.get("prsn_ntby_qty")),
            })

        records.sort(key=lambda x: x["date"])
        return records


# ──────────────────────────────────────────────
# WebSocket 실시간 시세
# ──────────────────────────────────────────────
class KISWebSocket:
    """KIS WebSocket 실시간 시세 수신"""

    WS_URL_REAL = "ws://ops.koreainvestment.com:21000"
    WS_URL_VIRTUAL = "ws://ops.koreainvestment.com:31000"

    def __init__(self, on_tick: Callable[[dict], None] | None = None):
        self.config = get_config()
        self.secrets = get_secrets()
        self._on_tick = on_tick
        self._ws_key: str = ""
        self._subscriptions: set[str] = set()
        self._running = False

    def _get_ws_url(self) -> str:
        return self.WS_URL_VIRTUAL if self.config.kis.is_virtual else self.WS_URL_REAL

    def _get_approval_key(self) -> str:
        """WebSocket 접속키를 발급받습니다."""
        url = f"{self.config.kis.base_url}/oauth2/Approval"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.secrets.kis_app_key,
            "secretkey": self.secrets.kis_app_secret,
        }
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        self._ws_key = data["approval_key"]
        logger.info("WebSocket 접속키 발급 완료")
        return self._ws_key

    def _build_subscribe_msg(self, stock_code: str, tr_type: str = "1") -> str:
        """실시간 구독 메시지를 생성합니다."""
        return json.dumps({
            "header": {
                "approval_key": self._ws_key,
                "custtype": "P",
                "tr_type": tr_type,  # 1=등록, 2=해제
                "content-type": "utf-8",
            },
            "body": {
                "input": {
                    "tr_id": "H0STCNT0",  # 실시간 체결가
                    "tr_key": stock_code,
                }
            },
        })

    def _parse_tick(self, raw: str) -> dict | None:
        """실시간 체결 데이터를 파싱합니다."""
        try:
            parts = raw.split("|")
            if len(parts) < 4:
                return None
            tr_id = parts[1]
            if tr_id != "H0STCNT0":
                return None

            fields = parts[3].split("^")
            if len(fields) < 20:
                return None

            return {
                "code": fields[0],
                "time": fields[1],
                "price": int(fields[2]),
                "change": int(fields[4]),
                "change_pct": float(fields[5]),
                "volume": int(fields[12]),
                "bid": int(fields[7]) if len(fields) > 7 else 0,
                "ask": int(fields[6]) if len(fields) > 6 else 0,
            }
        except (IndexError, ValueError) as e:
            logger.debug(f"체결 데이터 파싱 실패: {e}")
            return None

    async def _connect_and_listen(self, stock_codes: list[str]) -> None:
        """WebSocket에 연결하고 데이터를 수신합니다."""
        if not self._ws_key:
            self._get_approval_key()

        ws_url = self._get_ws_url()
        self._running = True

        while self._running:
            try:
                async with websockets.connect(ws_url, ping_interval=30) as ws:
                    logger.info(f"WebSocket 연결됨: {ws_url}")

                    # 종목 구독
                    for code in stock_codes:
                        await ws.send(self._build_subscribe_msg(code))
                        self._subscriptions.add(code)
                        logger.info(f"실시간 시세 구독: {code}")

                    # 메시지 수신 루프
                    async for message in ws:
                        if isinstance(message, bytes):
                            message = message.decode("utf-8")

                        # JSON 응답 (구독 확인 등)
                        if message.startswith("{"):
                            data = json.loads(message)
                            logger.debug(f"WS 응답: {data.get('header', {}).get('tr_id')}")
                            continue

                        # 실시간 데이터
                        tick = self._parse_tick(message)
                        if tick and self._on_tick:
                            self._on_tick(tick)

            except websockets.exceptions.ConnectionClosed:
                if self._running:
                    logger.warning("WebSocket 연결 끊김, 5초 후 재연결...")
                    await asyncio.sleep(5)
            except Exception as e:
                if self._running:
                    logger.error(f"WebSocket 오류: {e}, 10초 후 재연결...")
                    await asyncio.sleep(10)

    def start(self, stock_codes: list[str]) -> None:
        """별도 스레드에서 WebSocket 수신을 시작합니다."""
        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._connect_and_listen(stock_codes))

        thread = threading.Thread(target=_run, daemon=True, name="ws-listener")
        thread.start()
        logger.info("WebSocket 리스너 스레드 시작")

    def stop(self) -> None:
        """WebSocket 수신을 중지합니다."""
        self._running = False
        logger.info("WebSocket 리스너 중지 요청")


# ──────────────────────────────────────────────
# 예외
# ──────────────────────────────────────────────
class KISAPIError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")
