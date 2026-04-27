#!/usr/bin/env bash
# SuperTrader live watchdog
#
# 동작:
#   - logs/supertrader.log 의 마지막 수정 시각을 STALE_MINUTES 마다 확인
#   - STALE_THRESHOLD_MIN 초과 미갱신 시 Windows python.exe live 프로세스를 강제 종료하고 재기동
#   - heartbeat 로그가 5분마다 찍히므로 정상 상태에서는 평균 mtime < 5분
#
# 배포:
#   cron (crontab -e): 매 2분
#     */2 * * * * /mnt/e/SuperTrader/scripts/watchdog.sh >> /mnt/e/SuperTrader/logs/watchdog.log 2>&1
#
#   또는 별도 tmux/screen 세션에서 loop:
#     while true; do /mnt/e/SuperTrader/scripts/watchdog.sh; sleep 120; done

set -u

PROJECT_DIR="/mnt/e/SuperTrader"
LOG_FILE="${PROJECT_DIR}/logs/supertrader.log"
PYTHON_EXE="${PROJECT_DIR}/venv/Scripts/python.exe"
STALE_THRESHOLD_MIN=15
NOW=$(date +%s)

if [ ! -f "${LOG_FILE}" ]; then
    echo "$(date '+%F %T') [watchdog] 로그 파일 없음 — 처음 실행일 수 있음, 대기"
    exit 0
fi

LOG_MTIME=$(stat -c %Y "${LOG_FILE}")
AGE_SEC=$(( NOW - LOG_MTIME ))
AGE_MIN=$(( AGE_SEC / 60 ))

if [ "${AGE_MIN}" -lt "${STALE_THRESHOLD_MIN}" ]; then
    echo "$(date '+%F %T') [watchdog] OK — 로그 갱신 ${AGE_MIN}분 전"
    exit 0
fi

echo "$(date '+%F %T') [watchdog] STALE — 로그 ${AGE_MIN}분간 갱신 없음, 재시작 시도"

# Windows python.exe live 프로세스 후보: Services 세션, 메모리 300MB 이상
TARGET_PID=$(/mnt/c/Windows/System32/tasklist.exe /FI "IMAGENAME eq python.exe" /FO CSV 2>/dev/null \
    | awk -F'","' 'NR>1 && $3 ~ /Services/ {
        gsub(/[",K ]/, "", $5);
        if ($5+0 > 300000) {
            gsub(/"/, "", $2);
            print $2; exit
        }
    }')

if [ -n "${TARGET_PID}" ]; then
    echo "$(date '+%F %T') [watchdog] live 프로세스 PID=${TARGET_PID} 종료"
    /mnt/c/Windows/System32/taskkill.exe /PID "${TARGET_PID}" /F >/dev/null 2>&1
    sleep 3
fi

# 재기동 (nohup 으로 detach)
cd "${PROJECT_DIR}" || exit 1
nohup "${PYTHON_EXE}" -X utf8 main.py live \
    >> "${PROJECT_DIR}/logs/live_restart.log" 2>&1 &

RESTART_PID=$!
echo "$(date '+%F %T') [watchdog] 재기동 완료 (bash PID=${RESTART_PID})"
