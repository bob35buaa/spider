#!/bin/bash
# 初始化新实验的 plan 和 log 文件
# 用法：./init-experiment.sh <version> <topic> [run_id]
# 示例：./init-experiment.sh v5.0 new_reward
#        ./init-experiment.sh v5.0 new_reward R027

set -e

VERSION="${1:?用法: $0 <version> <topic> [run_id]}"
TOPIC="${2:?用法: $0 <version> <topic> [run_id]}"
RUN_ID="${3:-}"

EXP_NAME="${EXPERIMENT_NAME:-core4d}"
EXP_WS="${EXPERIMENT_WORKSPACE:-workspace/$EXP_NAME}"
EXP_NAME="${EXPERIMENT_NAME:-$(basename "$EXP_WS")}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE_DIR="$SCRIPT_DIR/../templates"

# 确保目录存在
mkdir -p "$EXP_WS/plan" "$EXP_WS/log"

# 自动检测下一个编号 (plan/ 和 log/ 取最大值 + 1)
PLAN_MAX=$(ls -1 "$EXP_WS/plan/"*.md 2>/dev/null | sed 's|.*/||' | grep -oP '^\d+' | sort -n | tail -1 || echo "0")
LOG_MAX=$(ls -1 "$EXP_WS/log/"*.md 2>/dev/null | sed 's|.*/||' | grep -oP '^\d+' | sort -n | tail -1 || echo "0")
: "${PLAN_MAX:=0}"
: "${LOG_MAX:=0}"
# 去掉前导零避免 bash 八进制解析
PLAN_MAX=$((10#$PLAN_MAX))
LOG_MAX=$((10#$LOG_MAX))
NEXT_NN=$(( PLAN_MAX > LOG_MAX ? PLAN_MAX + 1 : LOG_MAX + 1 ))
NN=$(printf "%02d" "$NEXT_NN")

# 自动检测下一个 Run ID
if [ -z "$RUN_ID" ]; then
    LAST_RUN=$(grep -oP 'R\d+' "$EXP_WS/EXPERIMENT_TRACKER.md" 2>/dev/null | sed 's/R//' | sort -n | tail -1 || echo "0")
    : "${LAST_RUN:=0}"
    # 去掉前导零避免 bash 八进制解析
    LAST_RUN=$((10#$LAST_RUN))
    NEXT_RUN=$((LAST_RUN + 1))
    RUN_ID="R$(printf "%03d" "$NEXT_RUN")"
fi

DATE=$(date +%Y-%m-%d)

# --- 创建 plan 文件 ---
PLAN_FILE="$EXP_WS/plan/${NN}_${VERSION}_${TOPIC}_plan.md"
if [ ! -f "$PLAN_FILE" ]; then
    if [ -f "$TEMPLATE_DIR/experiment_plan.md" ]; then
        sed -e "s/{XXX}/${RUN_ID#R}/g" \
            -e "s/{X\.X}/${VERSION#v}/g" \
            -e "s/{Version}/$VERSION/g" \
            -e "s/{version}/$VERSION/g" \
            -e "s/{exp_name}/$EXP_NAME/g" \
            -e "s/{topic}/$TOPIC/g" \
            -e "s/{标题}/[待填写]/g" \
            "$TEMPLATE_DIR/experiment_plan.md" > "$PLAN_FILE"
    else
        cat > "$PLAN_FILE" << EOF
# $RUN_ID (V${VERSION#v}) 实验计划：[待填写]

## Context

[前置实验结果 + 根因分析]

## Claims

| Claim | 最低证据 |
|-------|---------|
|       |         |

## 改动

### 1. [改动描述]

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|

## 训练命令

\`\`\`bash
bash $EXP_WS/scripts/train/train_${EXP_NAME}_$VERSION.sh $RUN_ID 0
\`\`\`

## 成功标准

| 指标 | 前次结果 | 本次目标 |
|------|---------|---------|
EOF
    fi
    echo "已创建 plan: $PLAN_FILE"
else
    echo "plan 已存在，跳过: $PLAN_FILE"
fi

# --- 创建 log 文件 ---
LOG_FILE="$EXP_WS/log/${NN}_${VERSION}_${TOPIC}.md"
if [ ! -f "$LOG_FILE" ]; then
    if [ -f "$TEMPLATE_DIR/experiment_log.md" ]; then
        sed -e "s/{XXX}/${RUN_ID#R}/g" \
            -e "s/{X\.X}/${VERSION#v}/g" \
            -e "s/{Version}/$VERSION/g" \
            -e "s/{version}/$VERSION/g" \
            -e "s/{exp_name}/$EXP_NAME/g" \
            -e "s/{topic}/$TOPIC/g" \
            -e "s/{YYYY-MM-DD}/$DATE/g" \
            -e "s/{NN}/$NN/g" \
            "$TEMPLATE_DIR/experiment_log.md" > "$LOG_FILE"
    else
        cat > "$LOG_FILE" << EOF
# V${VERSION#v} 实验结果

**日期**: $DATE
**实验域 (exp_name)**: \`$EXP_NAME\`
**对应Plan**: \`$EXP_WS/plan/${NN}_${VERSION}_${TOPIC}_plan.md\`

## 1. 背景

[简述实验目的]

## 2. $RUN_ID (V${VERSION#v}): [实验名]

### 训练指标

| 指标 | 前次 | **本次** | 变化 |
|------|------|---------|------|

### 分析

## Claims 验证

| Claim | 结果 |
|-------|------|

## 下一步
EOF
    fi
    echo "已创建 log: $LOG_FILE"
else
    echo "log 已存在，跳过: $LOG_FILE"
fi

# --- 创建 progress.md ---
PROGRESS_FILE="$EXP_WS/progress.md"
if [ ! -f "$PROGRESS_FILE" ]; then
    if [ -f "$TEMPLATE_DIR/progress.md" ]; then
        sed -e "s/{XXX}/${RUN_ID#R}/g" \
            -e "s/{X\.X}/${VERSION#v}/g" \
            -e "s/{YYYY-MM-DD}/$DATE/g" \
            "$TEMPLATE_DIR/progress.md" > "$PROGRESS_FILE"
    else
        cat > "$PROGRESS_FILE" << EOF
# 实验进度日志

## 会话：$DATE

### 当前实验
- **Run ID**: $RUN_ID
- **Version**: V${VERSION#v}
- **阶段**: Plan
EOF
    fi
    echo "已创建 progress: $PROGRESS_FILE"
else
    echo "progress.md 已存在，跳过"
fi

echo ""
echo "=== 初始化完成 ==="
echo "Run ID:  $RUN_ID"
echo "Version: V${VERSION#v}"
echo "Plan:    $PLAN_FILE"
echo "Log:     $LOG_FILE"
echo "Progress: $PROGRESS_FILE"
