"""buy_action_threshold sweep → optimal 선택 → config 갱신 → RL retrain → 최종 Slack

순서:
  1. threshold_sweep.py 실행 (subprocess, ~2h)
  2. data/threshold_sweep_results.json 읽기 → Sharpe 기준 최적 threshold 선정
  3. config/settings.yaml 의 timing.rl.buy_action_threshold 갱신
  4. main.py retrain --model rl 실행 (F1 개선 시에만 모델 교체)
  5. Slack 으로 종합 요약 전송 (sweep 결과 + 선정 threshold + retrain 결과)

이전 결과:
  · 오전 sweep [0.15..0.05] 완료 → 0.05가 가장 우수 (Sharpe +0.40)
  · 추세상 더 낮은 영역(0.03~0.00)이 더 나을 가능성

Usage:
    /mnt/e/SuperTrader/venv/Scripts/python.exe scripts/sweep_and_retrain.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

logger.add("logs/sweep_and_retrain.log", rotation="10 MB")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(PROJECT_ROOT / "venv" / "Scripts" / "python.exe")
SLACK_SWEEP_CHANNEL = "C0AT0BM1AHF"   # threshold sweep 결과 채널
RESULTS_JSON = PROJECT_ROOT / "data" / "threshold_sweep_results.json"
SETTINGS_YAML = PROJECT_ROOT / "config" / "settings.yaml"


def run_sweep() -> dict[float, dict]:
    logger.info("=== Phase 1/3: threshold sweep 시작 ===")
    t0 = time.time()
    rc = subprocess.call(
        [PYTHON, "-X", "utf8", "scripts/threshold_sweep.py"],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = (time.time() - t0) / 60
    logger.info(f"sweep 종료 rc={rc} ({elapsed:.0f}분)")
    if rc != 0:
        raise RuntimeError(f"threshold_sweep.py exit={rc}")
    if not RESULTS_JSON.exists():
        raise RuntimeError(f"결과 JSON 없음: {RESULTS_JSON}")
    raw = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    return {float(k): v for k, v in raw.items()}


def pick_best(results: dict[float, dict]) -> tuple[float, dict]:
    """Sharpe 최댓값 threshold 선택. Sharpe 동률 시 return 우선."""
    def key(item):
        th, m = item
        return (m.get("sharpe_ratio", -999), m.get("total_return", -999))
    best_th, best_m = max(results.items(), key=key)
    logger.info(
        f"=== 최적 threshold = {best_th} "
        f"(return={best_m.get('total_return',0):+.1f}%, "
        f"sharpe={best_m.get('sharpe_ratio',0):.2f}, "
        f"mdd={best_m.get('max_drawdown',0):.1f}%) ==="
    )
    return best_th, best_m


def update_settings_yaml(new_threshold: float) -> float:
    """timing.rl.buy_action_threshold 값을 in-place 갱신. 이전 값 반환."""
    text = SETTINGS_YAML.read_text(encoding="utf-8")
    pattern = re.compile(r"^(\s*buy_action_threshold:\s*)([+-]?\d+\.?\d*)\s*(#.*)?$", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        raise RuntimeError("buy_action_threshold 라인을 settings.yaml 에서 찾지 못했습니다")
    old = float(m.group(2))
    suffix = m.group(3) or ""
    new_line = f"{m.group(1)}{new_threshold}    {suffix}".rstrip()
    new_text = pattern.sub(new_line, text, count=1)
    SETTINGS_YAML.write_text(new_text, encoding="utf-8")
    logger.info(f"settings.yaml: buy_action_threshold {old} → {new_threshold}")
    return old


def run_retrain() -> tuple[int, str]:
    """main.py retrain --model rl 실행. 표준출력 + return code 반환."""
    logger.info("=== Phase 3/3: RL retrain 시작 ===")
    t0 = time.time()
    proc = subprocess.run(
        [PYTHON, "-X", "utf8", "main.py", "retrain", "--model", "rl"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = (time.time() - t0) / 60
    logger.info(f"retrain 종료 rc={proc.returncode} ({elapsed:.0f}분)")
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or "")


def send_final_slack(
    results: dict[float, dict],
    best_th: float,
    best_m: dict,
    old_threshold: float,
    retrain_rc: int,
    retrain_output: str,
) -> None:
    """전체 파이프라인 결과를 Slack 으로 전송."""
    try:
        from slack_sdk import WebClient
        from src.config import get_secrets, load_config
        load_config(str(SETTINGS_YAML))
        client = WebClient(token=get_secrets().slack_bot_token)

        # 결과 표
        header = f"{'Threshold':>10} {'Return':>10} {'Sharpe':>8} {'MDD':>8} {'WinRate':>8} {'Trades':>7}"
        lines = ["```", header, "-" * 60]
        for th in sorted(results.keys(), reverse=True):
            m = results[th]
            star = "  ←" if th == best_th else ""
            lines.append(
                f"{th:>10.2f} "
                f"{m.get('total_return',0):>9.1f}% "
                f"{m.get('sharpe_ratio',0):>8.2f} "
                f"{m.get('max_drawdown',0):>7.1f}% "
                f"{m.get('win_rate',0):>7.1f}% "
                f"{m.get('total_trades',0):>7.0f}{star}"
            )
        lines.append("```")
        table = "\n".join(lines)

        # retrain 결과 추출
        replaced = "교체됨" if "성능 개선 — 모델 교체" in retrain_output or "replaced" in retrain_output.lower() else \
                   "성능 동일/저하 — 기존 모델 유지" if retrain_rc == 0 else \
                   f"실패 (rc={retrain_rc})"

        # F1 / 백업 정보 추출 (best-effort)
        f1_match = re.search(r"F1[: ]+([0-9.]+).*?(?:→|->)\s*([0-9.]+)", retrain_output)
        f1_note = f" — F1 {f1_match.group(1)} → {f1_match.group(2)}" if f1_match else ""

        msg = (
            f"🎯 *buy_action_threshold sweep + RL retrain 완료* — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"*Sweep 결과* (`[0.05, 0.03, 0.01, 0.00]`)\n"
            f"{table}\n\n"
            f"*최적 threshold*: `{best_th}` "
            f"(return {best_m.get('total_return',0):+.1f}%, "
            f"Sharpe {best_m.get('sharpe_ratio',0):.2f}, "
            f"MDD {best_m.get('max_drawdown',0):.1f}%)\n"
            f"*config 갱신*: `{old_threshold}` → `{best_th}` (settings.yaml)\n"
            f"*RL retrain*: {replaced}{f1_note}\n\n"
            f"_라이브에 적용하려면 main.py live 재시작 필요._"
        )

        client.chat_postMessage(channel=SLACK_SWEEP_CHANNEL, text=msg)
        logger.info("Slack 최종 요약 전송 완료")

        # 시스템 채널에도 짧게
        sys_channel = get_secrets().slack_channel
        client.chat_postMessage(
            channel=sys_channel,
            text=(
                f"✅ buy_action_threshold sweep + RL retrain 완료. "
                f"최적 `{best_th}` (Sharpe {best_m.get('sharpe_ratio',0):.2f}). "
                f"라이브 재시작 시 반영. 자세한 결과는 sweep 채널 참고."
            ),
        )
    except Exception as e:
        logger.error(f"Slack 전송 실패: {e}")


def main() -> int:
    try:
        results = run_sweep()
    except Exception as e:
        logger.error(f"sweep 실패: {e}")
        try:
            from slack_sdk import WebClient
            from src.config import get_secrets, load_config
            load_config(str(SETTINGS_YAML))
            WebClient(token=get_secrets().slack_bot_token).chat_postMessage(
                channel=SLACK_SWEEP_CHANNEL,
                text=f"❌ threshold sweep 실패: {e}",
            )
        except Exception:
            pass
        return 1

    best_th, best_m = pick_best(results)
    try:
        old_threshold = update_settings_yaml(best_th)
    except Exception as e:
        logger.error(f"settings.yaml 갱신 실패: {e}")
        old_threshold = -1.0

    rc, output = run_retrain()
    send_final_slack(results, best_th, best_m, old_threshold, rc, output)
    return 0 if rc == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
