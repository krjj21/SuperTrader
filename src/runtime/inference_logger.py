"""모델 추론 로그 (#4, 2026-05-04).

라이브 사이클에서 generate_signal 호출 직후 호출. JSONL 한 줄 = 한 추론.
RL 재학습 시 raw 데이터로 활용 (peak +30% 시점에 모델이 어떻게 반응했는지 등).

DB 저장 대비 IO 부담 ↓ (sqlite WAL write 회피), pandas.read_json(lines=True) 으로 즉시 로드.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import threading

from loguru import logger

_LOG_PATH = Path("logs/model_inference.jsonl")
_LOCK = threading.Lock()


def log_inference(
    stock_code: str,
    stock_name: str,
    final_signal: str,
    diag: dict,
) -> None:
    """추론 1건을 jsonl 에 append. 실패해도 라이브 흐름은 막지 않는다."""
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "stock_code": stock_code,
        "stock_name": stock_name,
        "final_signal": final_signal,
    }
    record.update(diag or {})
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _LOCK:
            with _LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        logger.debug(f"inference log 기록 실패: {e}")
