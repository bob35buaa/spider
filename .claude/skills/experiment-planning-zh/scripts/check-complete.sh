#!/bin/bash
# 检查 EXPERIMENT_TRACKER.md 中实验完成状态
# 由 Stop 钩子调用，始终以退出码 0 结束

EXP_WS="${EXPERIMENT_WORKSPACE:-workspace/v2}"
TRACKER="$EXP_WS/EXPERIMENT_TRACKER.md"

if [ ! -f "$TRACKER" ]; then
    echo "[experiment-planning] 未找到 EXPERIMENT_TRACKER.md — 没有进行中的实验项目。"
    exit 0
fi

DONE=$(grep -c "| DONE |" "$TRACKER" 2>/dev/null || echo "0")
SKIP=$(grep -c "| SKIP |" "$TRACKER" 2>/dev/null || echo "0")
TODO=$(grep -c "| TODO |" "$TRACKER" 2>/dev/null || echo "0")

: "${DONE:=0}"
: "${SKIP:=0}"
: "${TODO:=0}"

TOTAL=$((DONE + SKIP + TODO))

if [ "$TODO" -eq 0 ] && [ "$TOTAL" -gt 0 ]; then
    echo "[experiment-planning] 所有实验已完成（$DONE done, $SKIP skipped, $TOTAL total）。"
else
    echo "[experiment-planning] 实验进行中（$DONE/$TOTAL done, $TODO 待完成）。"
    if [ "$TODO" -gt 0 ]; then
        echo "[experiment-planning] 未完成的实验："
        grep "| TODO |" "$TRACKER" 2>/dev/null | head -5
    fi
    echo "[experiment-planning] 停止前请确认 progress.md 和 EXPERIMENT_TRACKER.md 已更新。"
fi
exit 0
