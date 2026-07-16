# 远程实验执行指南

## 概述

当本地 GPU 不足以并行运行 RL 实验时，可以使用两台远程机器：

- `spider-remote`：常用两卡 A6000/RTX 6000 Ada 机器。
- `61.172.170.106:30409`：8 卡 A100 机器，启动时动态选择空闲 GPU；空闲定义为显存占用 `<5000MB`，最多使用 4 张。

本文档是 agent 启动远程实验前必须遵循的标准流程。目标是：

1. 远程代码与本地当前分支、当前 commit 严格一致。
2. `.gitignore` 忽略的训练输入只同步本次实验需要的文件。
3. 不同步历史 `logs/`、历史 `workspace/*/results/` 或无关大文件。
4. 启动前用 preflight 验证远程输入完整。

## 远程机器配置

### Profile A：常用两卡机

| 字段 | 值 |
|------|------|
| Profile | `a6000-2gpu` |
| SSH alias | `spider-remote` |
| Host | 10.100.71.70 |
| Port | 58122 |
| User | xiayb |
| GPUs | 2x A6000 / RTX 6000 Ada 级别 GPU |
| 默认可用 GPU | `0,1` |
| 当前 holosoma worktree | `/home/xiayb/pHRI_workspace/holosoma_r053` |
| SSH config | `~/.ssh/config` 中配置为 `spider-remote` |

注意：`/home/xiayb/pHRI_workspace/spider` 是 spider 仓库/旧 worktree，不是当前 holosoma RL 实验的默认执行目录。除非用户明确指定，不要在该目录执行 holosoma 训练。

### Profile B：8 卡机

| 字段 | 值 |
|------|------|
| Profile | `A100-8gpu` |
| SSH alias | `61.172.170.106` 或本地自定义 alias |
| HostName | `61.172.170.106` |
| Port | `30409` |
| User | `batchcom` |
| IdentityFile | `~/.ssh/id_rsa_tianyiyun` |
| GPUs | 8x A100，实际型号以 `nvidia-smi` 为准 |
| 默认可用 GPU | 启动时动态选择，显存占用 `<5000MB` 的 GPU，最多 4 张 |
| 当前 holosoma worktree | `/home/dataset-assist-0/xiayb/workspace/holosoma` |
| HOLOSOMA_DEPS_DIR | `/home/dataset-assist-0/xiayb/.holosoma_deps` |

推荐 SSH config：

```sshconfig
Host tianyiyun-A100
    HostName 61.172.170.106
    Port 30409
    User batchcom
    IdentityFile ~/.ssh/id_rsa_tianyiyun
```

如果本地没有配置 alias，也可以直接用 `ssh -p 30409 -i ~/.ssh/id_rsa_tianyiyun batchcom@61.172.170.106`。

**硬约束**：8 卡机不再固定后四张卡。启动前必须用 `nvidia-smi` 查询 0-7 号 GPU 的显存占用，选择所有显存占用 `<5000MB` 的 GPU，按 GPU index 升序最多取 4 张。示例：如果 GPU `2,3,6,7` 空闲，则本轮使用 `2,3,6,7`。如果没有空闲 GPU，不启动 A100 任务；不要 kill 其他用户进程。

### 远程 profile 选择

启动远程任务前必须先明确使用哪个 profile，并设置统一变量：

```bash
# 常用两卡机
REMOTE_PROFILE=a6000-2gpu
REMOTE_HOST=spider-remote
REMOTE_ROOT=/home/xiayb/pHRI_workspace/holosoma_r053
ALLOWED_GPUS="0 1"

# 8 卡机
REMOTE_PROFILE=A100-8gpu
REMOTE_HOST=tianyiyun-A100
REMOTE_ROOT=/home/dataset-assist-0/xiayb/workspace/holosoma
HOLOSOMA_DEPS_DIR=/home/dataset-assist-0/xiayb/.holosoma_deps
GPU_MEMORY_USED_MAX_MB=5000
MAX_A100_GPUS=4
ALLOWED_GPUS="$(ssh "$REMOTE_HOST" "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v limit=$GPU_MEMORY_USED_MAX_MB '{gsub(/ /,\"\",\$1); gsub(/ /,\"\",\$2); if (\$2+0 < limit) print \$1}' | head -n $MAX_A100_GPUS | paste -sd' ' -")"
test -n "$ALLOWED_GPUS"
```

如果用户只说“远程跑”但没有指定机器：

1. 默认继续使用 `spider-remote` 两卡机。
2. 如果本地/两卡机资源不够，询问或明确切到 8 卡机。
3. 切到 8 卡机时，按显存占用 `<5000MB` 动态选择空闲卡，最多使用 4 张。

## GPU 使用规则

| Profile | 默认 GPU | 规则 |
|---|---|---|
| `a6000-2gpu` | `0,1` | 每张 GPU 同时最多 1 个 IsaacSim/RL 训练；多任务同卡串行 |
| `A100-8gpu` | 动态空闲卡 | 空闲定义为显存占用 `<5000MB`；哪个 GPU 空闲就用哪个，最多 4 张；同卡串行 |

启动任何远程 session 前，必须检查目标 GPU 是否在当前 profile 的允许列表中：

```bash
if [ "$REMOTE_PROFILE" = "A100-8gpu" ]; then
  GPU_MEMORY_USED_MAX_MB="${GPU_MEMORY_USED_MAX_MB:-5000}"
  MAX_A100_GPUS="${MAX_A100_GPUS:-4}"
  ALLOWED_GPUS="$(ssh "$REMOTE_HOST" "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v limit=$GPU_MEMORY_USED_MAX_MB '{gsub(/ /,\"\",\$1); gsub(/ /,\"\",\$2); if (\$2+0 < limit) print \$1}' | head -n $MAX_A100_GPUS | paste -sd' ' -")"
  test -n "$ALLOWED_GPUS"
fi

case " $ALLOWED_GPUS " in
  *" $GPU "*) ;;
  *) echo "GPU $GPU is not allowed for $REMOTE_PROFILE; allowed: $ALLOWED_GPUS"; exit 2 ;;
esac
```

## 标准流程

### 1. 确认本次实验输入清单

启动前先从训练脚本和 plan 中列出本次实验需要的输入，不要同步整个仓库的大文件目录。

常见输入包括：

| 类型 | 示例 |
|------|------|
| 轨迹 npz | `workspace/data/spider_best_E018b_E022_E025_for_rl_rename/<TAG>_v2_mj_w_obj_w_partner.npz` |
| 对象模型 | `src/holosoma_retargeting/holosoma_retargeting/models/<OBJ>/` |
| base checkpoint | `logs/<base_project>/<base_run>/model_XXXXX.pt` |
| 训练/eval config | 已提交的 config 走 git；未提交或生成的 `holosoma_config.yaml` 等走 rsync |
| 本次 eval 输入 | checkpoint、`holosoma_config.yaml`、本次 run 目录下必要文件 |

禁止为了省事同步：

- 整个 `logs/`
- 整个 `tmp/`
- 整个 `workspace/v2/results/`
- 整个 `workspace/v2/artifacts/`
- 与本次实验无关的历史 checkpoint、视频、metrics

### 2. Git 同步

远程必须跟随本地当前分支，而不是写死分支名。

```bash
# 先按“远程 profile 选择”设置 REMOTE_HOST / REMOTE_ROOT / ALLOWED_GPUS

BRANCH="$(git branch --show-current)"
LOCAL_HEAD="$(git rev-parse HEAD)"

# 本地先确认要跑的代码已经提交。
git status --short

# 如果本次实验依赖代码/config 改动，先提交并推送。
git push origin "$BRANCH"

# 远程只允许 fast-forward 到当前分支，避免 merge commit 或误覆盖 dirty worktree。
ssh "$REMOTE_HOST" "
  set -e
  cd '$REMOTE_ROOT'
  echo 'remote branch:' \$(git branch --show-current)
  echo 'remote head before:' \$(git rev-parse --short HEAD)
  git status --short
  git fetch origin '$BRANCH'
  if git show-ref --verify --quiet 'refs/heads/$BRANCH'; then
    git checkout '$BRANCH'
  else
    git checkout -b '$BRANCH' 'origin/$BRANCH'
  fi
  git merge --ff-only 'origin/$BRANCH'
  test \"\$(git rev-parse HEAD)\" = '$LOCAL_HEAD'
  echo 'remote head after:' \$(git rev-parse --short HEAD)
"
```

如果 `git status --short` 显示远程有未提交代码改动，先停下来判断来源，不要直接 `git pull` 或覆盖。被 `.gitignore` 忽略的运行产物不影响 git 状态。

### 3. Rsync 本次实验需要的忽略文件

`.gitignore` 会忽略 npz、模型目录、logs/checkpoints、视频等大文件。它们不会通过 git 到远程，必须按本次实验清单同步。

推荐从 repo root 使用 `rsync -avR`，保留相对路径：

```bash
# 先按“远程 profile 选择”设置 REMOTE_HOST / REMOTE_ROOT / ALLOWED_GPUS

TAG=20231011-048-person2-Box025
OBJ=Box025

MOTION_REL="workspace/data/spider_best_E018b_E022_E025_for_rl_rename/${TAG}_v2_mj_w_obj_w_partner.npz"
MODEL_DIR_REL="src/holosoma_retargeting/holosoma_retargeting/models/${OBJ}"
BASE_CKPT_REL="logs/core4d_stage1_body_only_20231011-048-person2-Box025/20260522_134654-stage1_body_only_R048-locomotion/model_04000.pt"

rsync -avR "$MOTION_REL" "$REMOTE_HOST:$REMOTE_ROOT/"
rsync -avR "$MODEL_DIR_REL" "$REMOTE_HOST:$REMOTE_ROOT/"
rsync -avR "$BASE_CKPT_REL" "$REMOTE_HOST:$REMOTE_ROOT/"
```

如果本次实验使用新生成或未提交的 config，也只同步对应 config 文件：

```bash
CONFIG_REL="logs/<project>/<run>/holosoma_config.yaml"
rsync -avR "$CONFIG_REL" "$REMOTE_HOST:$REMOTE_ROOT/"
```

如果是本地 checkpoint 触发远程 eval，只同步该 checkpoint 和该 run 的 config，不要同步整个历史 run：

```bash
LOCAL_RUN="logs/<project>/<run>"
REMOTE_RUN="$REMOTE_ROOT/logs/<project>/<run>"
CKPT_FILE="model_00200.pt"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_RUN'"
rsync -av "$LOCAL_RUN/$CKPT_FILE" "$LOCAL_RUN/holosoma_config.yaml" \
  "$REMOTE_HOST:$REMOTE_RUN/"
```

### 4. 远程 preflight

启动训练前必须在远程验证本次输入都存在，并确认 GPU 使用符合当前 profile 的允许范围。

```bash
GPU="${GPU:?set by execution manifest}"

case " $ALLOWED_GPUS " in
  *" $GPU "*) ;;
  *) echo "GPU $GPU is not allowed for $REMOTE_PROFILE; allowed: $ALLOWED_GPUS"; exit 2 ;;
esac

ssh "$REMOTE_HOST" "
  set -e
  cd '$REMOTE_ROOT'
  test -f '$MOTION_REL'
  test -f 'src/holosoma_retargeting/holosoma_retargeting/models/$OBJ/$OBJ.urdf'
  test -f '$BASE_CKPT_REL'
  nvidia-smi -i '$GPU'
  echo 'preflight ok'
"
```

如果训练不需要 base checkpoint，可以删掉对应检查。若脚本中还有额外输入，例如 resume checkpoint、offline eval video source、手写 override config，也必须加入 preflight。

### 5. 部署并启动 tmux

使用 `tmux -c "$REMOTE_ROOT"` 固定工作目录，并显式设置 `HOLOSOMA_ROOT`，避免远端 shell 中残留旧路径。

```bash
SESSION=r085_train
SCRIPT=workspace/v2/scripts/train/train_core4d_r084_r086_box023_handbox_stagec.sh
GPU=4

case " $ALLOWED_GPUS " in
  *" $GPU "*) ;;
  *) echo "GPU $GPU is not allowed for $REMOTE_PROFILE; allowed: $ALLOWED_GPUS"; exit 2 ;;
esac

ssh "$REMOTE_HOST" "
  cd '$REMOTE_ROOT'
  tmux kill-session -t '$SESSION' 2>/dev/null || true
  tmux new-session -d -s '$SESSION' -c '$REMOTE_ROOT' \
    \"HOLOSOMA_ROOT='$REMOTE_ROOT' HOLOSOMA_DEPS_DIR='${HOLOSOMA_DEPS_DIR:-}' bash '$SCRIPT' R085 '$GPU' 801 8192\"
"
```

同一张 GPU 上不要并行跑多个 MuJoCo/IsaacSim RL 训练；需要多个实验时，每张 GPU 内串行排队。

### 6. 监控进度

```bash
# 查看 tmux 输出
ssh "$REMOTE_HOST" "tmux capture-pane -t '$SESSION' -p | tail -40"

# 查看 GPU
ssh "$REMOTE_HOST" "nvidia-smi"

# 查看当前训练 log
ssh "$REMOTE_HOST" "tail -40 '$REMOTE_ROOT/logs/<project>/<run_name>_train.log'"

# 查看 checkpoint
ssh "$REMOTE_HOST" "find '$REMOTE_ROOT/logs/<project>' -name 'model_*.pt' | sort | tail"
```

如果 tmux session 很快退出，先查 train log 和 tmux pane，不要直接重复启动同一实验。

### 7. 回收结果

只回收本次实验的 run/project 输出，不要把远端整个 `logs/` 拉回本地。

```bash
REMOTE_RUN="$REMOTE_ROOT/logs/<project>/<run>"
LOCAL_RUN="logs/<project>/<run>"
mkdir -p "$LOCAL_RUN"

rsync -av "$REMOTE_HOST:$REMOTE_RUN/" "$LOCAL_RUN/"
```

如果只需要 eval 结果，优先同步 metrics/video/log 子目录：

```bash
rsync -av "$REMOTE_HOST:$REMOTE_RUN/eval_metrics_<name>/" "$LOCAL_RUN/eval_metrics_<name>/"
rsync -av "$REMOTE_HOST:$REMOTE_RUN/eval_videos_<name>/" "$LOCAL_RUN/eval_videos_<name>/"
rsync -av "$REMOTE_HOST:$REMOTE_RUN/"'eval_*_model_*.log' "$LOCAL_RUN/" || true
```

## 并行策略

| 场景 | GPU 分配 | 原则 |
|------|---------|------|
| 两卡机 2 个训练 | `GPU0/GPU1` 每卡 1 个 | 并行 |
| 两卡机 4 个训练 | `GPU0/GPU1` 每卡 2 个 | 每 GPU 内串行 |
| 8 卡机 4 个以内训练 | 启动时空闲 GPU，例如 `2,3,6,7` | 每卡 1 个，并行 |
| 8 卡机超过 4 个训练 | 启动时空闲 GPU，最多 4 张 | 每 GPU 内串行；未被选中的 GPU 不使用 |
| 本地训练 + 远程 eval | eval 等 checkpoint 出现后启动 | 只 rsync checkpoint + config |
| 多 checkpoint eval | 每个 checkpoint 独立 session | 避免覆盖 video/log 路径 |

## 关键约束

1. 远程 profile 必须明确；两卡机默认 `spider-remote`，8 卡机默认 `tianyiyun-A100` 或 `61.172.170.106`。
2. 两卡机执行目录默认是 `/home/xiayb/pHRI_workspace/holosoma_r053`；8 卡机执行目录默认是 `/home/dataset-assist-0/xiayb/workspace/holosoma`。
3. 8 卡机必须按显存占用 `<5000MB` 动态选择空闲 GPU，最多 4 张；示例空闲 GPU `2,3,6,7` 时就使用 `2,3,6,7`。
4. 8 卡机启动训练/eval 时必须显式设置 `HOLOSOMA_DEPS_DIR=/home/dataset-assist-0/xiayb/.holosoma_deps`，否则 `scripts/source_isaacsim_setup.sh` 会找错 conda 路径。
5. 远程代码必须 fast-forward 到本地当前分支的当前 commit。
6. git 只同步代码和已提交 config；`.gitignore` 忽略的大输入必须 rsync。
7. rsync 只同步本次实验 manifest 里的 motion/model/config/checkpoint。
8. 不要同步整个历史 `logs/` 或 `workspace/v2/results/`。
9. 启动前必须运行远程 preflight，包括 GPU allowlist 检查。
10. 每个实验的视频、metrics、log 输出路径必须唯一，避免互相覆盖。

## 故障排除

| 问题 | 解决 |
|------|------|
| SSH connection refused | 检查 VPN/网络，`ssh -v spider-remote` |
| 8 卡机 SSH 失败 | 检查 `ssh -p 30409 -i ~/.ssh/id_rsa_tianyiyun batchcom@61.172.170.106` 或 `ssh -v tianyiyun-A100` |
| Permission denied | 检查 ssh config 和 key |
| 远程 git 非 fast-forward | 不要 merge，先确认远端是否有未推送改动 |
| 远程 dirty worktree | 先 `git status --short` 判断来源，不要覆盖 |
| 找不到 motion npz | 用 `rsync -avR "$MOTION_REL" "$REMOTE_HOST:$REMOTE_ROOT/"` 同步 |
| 找不到 URDF/OBJ | 同步 `models/<OBJ>/` 整个目录 |
| 找不到 base checkpoint | 只同步本次需要的 `model_XXXXX.pt` |
| `HOLOSOMA_ROOT` 指向旧目录 | tmux 命令里显式 `HOLOSOMA_ROOT=$REMOTE_ROOT` |
| A100 source conda 失败 | 设置 `HOLOSOMA_DEPS_DIR=/home/dataset-assist-0/xiayb/.holosoma_deps` 后再 source `scripts/source_isaacsim_setup.sh` |
| CUDA OOM | 降低 `num_envs` 或等待 GPU 空闲 |
| 8 卡机任务跑到未选中的 GPU | 立即停止对应 session；重新按显存占用 `<5000MB` 选择空闲 GPU，并只在 `ALLOWED_GPUS` 内启动 |
| tmux session 找不到 | `ssh "$REMOTE_HOST" "tmux list-sessions"` |
| 脚本快速退出 | 查看 train log 和 `tmux capture-pane`，不要盲目重启 |
