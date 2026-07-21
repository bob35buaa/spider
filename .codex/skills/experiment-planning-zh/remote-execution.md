# 远程实验执行指南

## 概述

当本地 GPU 资源不足以并行运行多个实验时，使用远程多卡机器并行执行。
本文档定义了标准流程，确保代码同步、日志收集、结果回传的可复现性。

## 远程机器配置

### 常用两卡机

| 字段 | 值 |
|------|------|
| Profile | `a6000-2gpu` |
| SSH alias | `spider-remote` |
| Host | 10.100.71.70 |
| Port | 58122 |
| User | xiayb |
| GPUs | 2x NVIDIA RTX 6000 Ada (48GB) |
| 默认可用 GPU | `0,1` |
| 项目路径 | `/home/xiayb/pHRI_workspace/spider` |
| SSH config | `~/.ssh/config` 中配置为 `spider-remote` |

### A100 8 卡机

| 字段 | 值 |
|------|------|
| Profile | `A100-8gpu` |
| 推荐 SSH alias | `tianyiyun-A100` |
| HostName | `61.172.170.106` |
| Port | `30409` |
| User | `batchcom` |
| IdentityFile | `~/.ssh/id_rsa_tianyiyun` |
| GPUs | 8x A100，实际型号以 `nvidia-smi` 为准 |
| 默认可用 GPU | 启动时动态选择，最多 4 张 |
| 项目路径 | `/home/dataset-assist-0/xiayb/workspace/spider` |

推荐 SSH config：

```sshconfig
Host tianyiyun-A100
    HostName 61.172.170.106
    Port 30409
    User batchcom
    IdentityFile ~/.ssh/id_rsa_tianyiyun
```

未配置 alias 时可直接连接：

```bash
ssh -p 30409 -i ~/.ssh/id_rsa_tianyiyun \
  batchcom@61.172.170.106
```

#### A100 动态选卡

A100 不固定使用前四张或后四张卡。启动时必须：

1. 查询全部 0-7 号 GPU，显存占用必须 `<5000MB`。
2. 排除存在其他用户计算任务或不符合机器预约规则的 GPU。
3. 按 GPU index 升序最多选择 4 张。
4. selection 和 tmux 启动之间再次检查；状态变化时重建 worker pool。
5. 没有合格 GPU 时不启动，不 kill 或抢占其他进程。

```bash
REMOTE_PROFILE=A100-8gpu
REMOTE_HOST=tianyiyun-A100
REMOTE_ROOT=/home/dataset-assist-0/xiayb/workspace/spider
A100_GPU_MEM_USED_LIMIT_MB=5000
A100_MAX_GPUS=4

A100_LOW_MEM_GPUS="$(
  ssh "$REMOTE_HOST" \
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits" |
    awk -F, -v limit="$A100_GPU_MEM_USED_LIMIT_MB" '
      {
        gsub(/ /, "", $1)
        gsub(/ /, "", $2)
        if ($2 + 0 < limit) print $1
      }
    '
)"

ssh "$REMOTE_HOST" \
  "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
   --format=csv,noheader,nounits"
```

最终 `ALLOWED_GPUS` 必须是低显存候选集与机器预约/所有者允许集合的交集。推荐由启动脚本调用 Phase 0 已确认的只读检查入口：

```bash
: "${A100_AVAILABILITY_CMD:?confirm availability command in Phase 0}"

A100_POLICY_GPUS="$(ssh "$REMOTE_HOST" "$A100_AVAILABILITY_CMD")"
ALLOWED_GPUS="$(
  comm -12 \
    <(printf '%s\n' "$A100_LOW_MEM_GPUS" | tr ', ' '\n' | sed '/^$/d' | sort -n) \
    <(printf '%s\n' "$A100_POLICY_GPUS" | tr ', ' '\n' | sed '/^$/d' | sort -n) |
    head -n "$A100_MAX_GPUS" |
    paste -sd' ' -
)"
test -n "$ALLOWED_GPUS"
```

如果机器没有统一预约工具，必须人工核对 `nvidia-smi` 和任务归属后显式提供允许集合，并在 experiment environment manifest 中保存确认时间、GPU snapshot 和最终 `ALLOWED_GPUS`。不能只凭 `<5000MB` 判断可抢占。

示例：低显存候选和预约允许集合的交集为 `2,3,6,7` 时，本轮使用 `2,3,6,7`；如果只剩 `3,7`，就只创建两个 worker。

## 标准流程

以下原有命令默认使用常用两卡机。使用 A100 时，将主机和项目路径替换为上面的 `REMOTE_HOST` / `REMOTE_ROOT`，GPU id 必须来自本轮 `ALLOWED_GPUS`。

### 1. 代码同步 (本地 → 远程)

```bash
# 本地提交并推送
git add <files> && git commit -m "..." && git push

# 远程拉取
ssh spider-remote "cd /home/xiayb/pHRI_workspace/spider && git pull"
```

**注意**: gitignore 的文件 (npz 结果、视频) 用 scp 传输。

### 2. 编写远程运行脚本

在 `workspace/{exp_name}/scripts/launch/active/` 下创建真实脚本。
如果需要保留旧命令，再在 `workspace/{exp_name}/scripts/` 根目录放 thin wrapper。
CORE4D 新实验不要把真实 `run_E*.sh` / `pull_E*.sh` 实现直接写在 `scripts/` 根目录。

模板如下：

```bash
#!/bin/bash
# E{NNN}: <实验描述>
# GPU0: <任务列表> | GPU1: <任务列表>
set -e

OVERRIDE=core4d_e0XX
RESULTS_DIR=workspace/core4d/results/E0XX
LOGS_DIR=logs/E0XX
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

run_experiment() {
    local name=$1 gpu=$2 task=$3
    # ... 其余参数 ...
    echo "[$(date '+%H:%M:%S')] Starting $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl uv run examples/run_mjwp.py \
        +override=$OVERRIDE task=$task \
        video_output_path="$RESULTS_DIR/${name}.mp4" \
        > "$LOGS_DIR/${name}.log" 2>&1
    cp "<output_dir>/trajectory_mjwp_act.npz" "$RESULTS_DIR/${name}.npz"
    echo "[$(date '+%H:%M:%S')] Finished $name"
}

# GPU 0: 实验 A (sequential)
( run_experiment "A1" 0 task1; run_experiment "A2" 0 task2 ) &
PID0=$!

# GPU 1: 实验 B (sequential)
( run_experiment "B1" 1 task1; run_experiment "B2" 1 task2 ) &
PID1=$!

echo "=== Launched: GPU0 PID=$PID0 | GPU1 PID=$PID1 ==="
wait $PID0; echo "GPU0 done"
wait $PID1; echo "GPU1 done"
echo "=== All complete ==="
```

#### A100 worker queue

A100 脚本不能写死 `0,1` 或 `4,5,6,7`。它必须读取启动时生成的 execution manifest/`ALLOWED_GPUS`，为每张被选中的 GPU 创建一个串行 worker queue：

```bash
worker_pids=()
for gpu in $ALLOWED_GPUS; do
  (
    while IFS=$'\t' read -r run_id task; do
      run_experiment "$run_id" "$gpu" "$task"
    done < "$QUEUE_ROOT/gpu${gpu}.tsv"
  ) &
  worker_pids+=("$!")
done

for pid in "${worker_pids[@]}"; do
  wait "$pid"
done
```

每条 run 的 NPZ、outdir、MP4 和 log 路径必须唯一。每张 GPU 内严格串行，不同 GPU 才并行。

### 3. 部署并启动 (tmux)

```bash
# 同步代码
ssh spider-remote "cd /home/xiayb/pHRI_workspace/spider && git pull"

# 在 tmux 中启动 (不会因 SSH 断开而终止)
ssh spider-remote "cd /home/xiayb/pHRI_workspace/spider && \
    tmux new-session -d -s <session_name> && \
    tmux send-keys -t <session_name> 'bash <script_path>' Enter"
```

A100 使用：

```bash
ssh "$REMOTE_HOST" "cd '$REMOTE_ROOT' && git pull --ff-only"
ssh "$REMOTE_HOST" "
  cd '$REMOTE_ROOT'
  tmux new-session -d -s <session_name> -c '$REMOTE_ROOT' \
    'bash <script_path>'
"
```

A100 启动 tmux 前必须重新执行动态选卡；如果某张卡不再满足条件，重建 worker queues，不能沿用旧 `ALLOWED_GPUS`。

### 4. 监控进度

```bash
# 查看 tmux 输出
ssh spider-remote "tmux capture-pane -t <session_name> -p | tail -10"

# 检查已完成的结果数量
ssh spider-remote "ls /home/xiayb/pHRI_workspace/spider/<results_dir>/*.npz | wc -l"

# 查看某个实验的进度
ssh spider-remote "tail -1 /home/xiayb/pHRI_workspace/spider/<logs_dir>/<name>.log"
```

A100 监控命令使用 `REMOTE_HOST` / `REMOTE_ROOT`，并同时检查 `nvidia-smi` 和各 worker queue：

```bash
ssh "$REMOTE_HOST" "nvidia-smi"
ssh "$REMOTE_HOST" "tmux capture-pane -t <session_name> -p | tail -20"
```

### 5. 回收结果

```bash
# 优先固化为 pull 脚本
bash workspace/core4d/scripts/launch/active/pull_E###_remote_results.sh <stage>

# 如需兼容历史命令，可通过根目录 wrapper 调用
bash workspace/core4d/scripts/pull_E###_remote_results.sh <stage>
```

同时使用两台远程机器时，分别固化 pull 脚本，避免主机和结果目录混用：

```bash
bash workspace/core4d/scripts/launch/active/pull_E###_remote_a6000_results.sh <stage>
bash workspace/core4d/scripts/launch/active/pull_E###_remote_a100_results.sh <stage>
```

pull 只回收本次 execution manifest 登记的结果；回收后按 worker 汇总 NPZ、outdir、config、MP4、log 的数量、大小和 SHA。

### 6. 本地评估

```bash
# 使用当前 eval 结构
python workspace/core4d/scripts/eval/runners/eval_E###_<topic>.py <stage>

# 若已保留兼容 wrapper，也可使用旧根路径
python workspace/core4d/scripts/eval/eval_E###_<topic>.py <stage>
```

### 7. 离线可视化 (不需要重跑实验)

```bash
MUJOCO_GL=egl python workspace/hdmi_reproduce/scripts/render_trajectory_video.py \
    --scene <scene.xml> \
    --kin <kinematic.npz> \
    --phys <result.npz> \
    --output <output.mp4> --fps 30
```

## 自动化回收 (watch_and_pull)

手动轮询远程实验是否完成、拉取结果、校验产物数量、启动 eval 是重复性劳动。
`watch_and_pull_template.sh` 将此流程自动化为一个后台脚本。

### 模板位置

```
workspace/core4d/scripts/templates/watch_and_pull_template.sh
```

### 核心逻辑

1. **循环轮询**: 每 `POLL_INTERVAL` 秒检查本地 tmux session 和远程 tmux session 是否存在
2. **Hardened SSH**: SSH 不通时保守假定远程仍在运行，避免误判完成
3. **双重确认**: 连续 2 次检测到双端完成后才视为真正完成
4. **自动 pull**: 调用 pull 脚本将远程产物同步回本地
5. **产物校验**: `find $RESULT_ROOT -name '*.npz' | wc -l` 与预期数量比对
6. **自动 eval**: 产物数量达标后自动调用 eval 脚本
7. **日志记录**: 全程输出带时间戳写入 `logs/{EXP_ID}/monitor/` 下

同时使用 A6000/A100 时，watcher 必须分别检查两个远程 tmux session、分别调用 pull 脚本；任一 SSH 不通都按“仍在运行”处理。只有所有 manifest row 均有产物或明确 failure 后才启动 eval。

### 占位符说明

| 占位符 | 含义 | 默认值 |
|--------|------|--------|
| `{{EXP_ID}}` | 实验编号 | — |
| `{{REMOTE_HOST}}` | 远程 SSH alias | `spider-remote` |
| `{{LOCAL_TMUX}}` | 本地 tmux session 名 | `{exp_id_lower}_local` |
| `{{REMOTE_TMUX}}` | 远程 tmux session 名 | `{exp_id_lower}_remote` |
| `{{EXPECTED_NPZ_COUNT}}` | 预期 NPZ 产物数 | splits 数量 |
| `{{PULL_SCRIPT}}` | pull 脚本路径 (repo-relative) | — |
| `{{EVAL_SCRIPT}}` | eval 脚本路径 (repo-relative) | — |
| `{{RESULT_ROOT}}` | 结果目录 (repo-relative) | — |
| `{{POLL_INTERVAL}}` | 轮询间隔秒数 | `600` |

### 使用方式一: gen_experiment.py --with-watcher

最简方式，自动填充模板：

```bash
python workspace/core4d/scripts/gen_experiment.py \
    --exp-id E153 \
    --description "my experiment" \
    --splits "local-gpu0,remote-gpu0,remote-gpu1" \
    --with-watcher \
    --poll-interval 600 \
    --expected-npz-count 20
```

这会额外生成 watcher。新脚本应优先放在 `workspace/core4d/scripts/launch/active/`；
若 generator 仍输出到根目录，后续需要迁移真实实现到 `launch/active/` 并保留根 wrapper。

可选参数：
- `--remote-host` 覆盖远程主机 (默认 `spider-remote`)
- `--poll-interval` 轮询间隔秒数 (默认 600)
- `--expected-npz-count` 预期产物数 (默认从 splits 数量推算)

### 使用方式二: 手动实例化模板

```bash
# 复制模板并替换占位符
cp workspace/core4d/scripts/templates/watch_and_pull_template.sh \
   workspace/core4d/scripts/launch/active/watch_and_pull_e153.sh
sed -i 's/{{EXP_ID}}/E153/g; s/{{REMOTE_HOST}}/spider-remote/g; ...' \
   workspace/core4d/scripts/launch/active/watch_and_pull_e153.sh
chmod +x workspace/core4d/scripts/launch/active/watch_and_pull_e153.sh
```

### 启动 watcher

```bash
# 在后台 tmux 中运行 (不怕终端断开)
tmux new-session -d -s e153_watcher \
  "bash workspace/core4d/scripts/launch/active/watch_and_pull_e153.sh"

# 或直接前台运行
bash workspace/core4d/scripts/launch/active/watch_and_pull_e153.sh
```

### 环境变量覆盖

运行时可通过环境变量覆盖模板默认值：

```bash
INTERVAL_SECONDS=300 EXPECTED_NPZ_COUNT=30 \
  bash workspace/core4d/scripts/launch/active/watch_and_pull_e153.sh
```

## 并行策略

| 场景 | GPU 分配 | 预期时间/原则 |
|------|---------|---------|
| 2 个 case, 2 GPU | 每 GPU 1 个 | ~6 min |
| 4 个 case, 2 GPU | 每 GPU 2 个 (串行) | ~12 min |
| 6 个 case, 2 GPU | 每 GPU 3 个 (串行) | ~18 min |
| Sweep 5 configs, 1 case | GPU0: 2+GPU1: 3 | ~18 min |
| A100 不超过 4 个 case | 动态选择最多 4 张 | 每 GPU 1 个，并行 |
| A100 超过 4 个 case | 动态选择最多 4 张 | 按长度均衡，每 GPU 内串行 |
| 本地 + A6000 + A100 | 每个 profile 独立 worker pool | 路径、session、pull 脚本唯一 |

## 关键约束

1. **同一 GPU 上的实验必须串行** — MuJoCo Warp 占满显存
2. **video_output_path 必须不同** — 否则后跑的覆盖先跑的视频
3. **trajectory npz 也会互相覆盖** — 每个实验完成后立即 cp 到结果目录
4. **tmux session 不会主动结束** — 所有实验完成后手动 `tmux kill-session`
5. **A100 不固定 GPU id** — 每次 launch 动态选择，最多 4 张
6. **低显存不等于可抢占** — 还必须确认 compute process、所有者和预约规则
7. **没有可用 GPU 时不启动** — 不 kill 或抢占其他任务
8. **不同 profile 不共享输出路径** — run id、worker id、log 和 pull 目标必须唯一

## 故障排除

| 问题 | 解决 |
|------|------|
| SSH connection refused | 检查 VPN/网络, `ssh -v spider-remote` |
| A100 SSH 失败 | 检查 `ssh -v tianyiyun-A100` 或直连端口/key |
| Permission denied | 重新 `ssh-copy-id` 或检查 A100 IdentityFile |
| CUDA OOM | 减少 num_samples (1024→512) |
| EGL error on exit | 无害,忽略 |
| A100 没有候选 GPU | 不启动 A100；等待资源或只使用本地/A6000 |
| A100 候选卡存在其他任务 | 从 `ALLOWED_GPUS` 移除并重建 worker pool |
| A100 selection 后被占用 | 重新查询、重建 queue 和 tmux，不沿用旧选择 |
| 任务跑到未选中的 GPU | 停止本实验对应 session，修复 allowlist 后重启，不影响其他任务 |
| tmux session 找不到 | `ssh spider-remote "tmux list-sessions"` |
| 脚本在远程报错 | 查看 `<logs_dir>/<name>.log` |
