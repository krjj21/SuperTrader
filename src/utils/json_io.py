"""
JSON 파일 읽기/쓰기 유틸 — 한국어 인코딩 fallback 포함.

Windows Python 이 cp949/EUC-KR 로 파일을 쓰는 환경에서 WSL/Flask 에서 읽을 때
여러 인코딩을 순차 시도해 성공하는 첫 번째를 사용.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

DEFAULT_ENCODINGS: tuple[str, ...] = ("utf-8", "cp949", "euc-kr")


def load_json_with_fallback(
    path: str | Path,
    encodings: Sequence[str] = DEFAULT_ENCODINGS,
    default: Any = None,
) -> Any:
    """파일을 여러 인코딩으로 순차 시도해 JSON 파싱. 실패 시 default 반환."""
    p = Path(path)
    if not p.exists():
        return default
    for enc in encodings:
        try:
            raw = p.read_text(encoding=enc)
            return json.loads(raw)
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError:
            return default
    return default
