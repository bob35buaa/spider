# Phase 13: E048-E049 碰撞盒修复后重跑 + 小箱子 HDMI 对比

## Context

碰撞盒模板 Bug 修复后 (详见 `log/54_collision_box_bug_fix.md`), 所有 21 case 的 scene.xml + scene_act.xml 已更新。
历史 baseline (E041c) 指标不再准确, 需要重跑。同时新增 box023 HDMI 对比实验。

## 实验矩阵

| 实验 | Case | 算法 | 目的 | GPU |
|------|------|------|------|-----|
| E048-base | box025_person1 | E041c | 碰撞盒修复后 baseline | 远程 GPU0 |
| E048-base | bucket010_person1 | E041c | 碰撞盒修复后 baseline | 远程 GPU0 |
| E048-base | desk005_person2 | E041c | 碰撞盒修复后 baseline | 远程 GPU1 |
| E048b | box023_person1 | E041c | 小箱子 MJWP | 远程 GPU1 |
| E048c | box001_person1 | E041c | 大箱子泛化 | 远程 GPU0 |
| E048d | box024_person1 | E041c | 大箱子泛化 | 远程 GPU1 |
| E048a | box023_person1 | HDMI | 小箱子 HDMI 对比 | 本地 |

## Claims

- C1: 碰撞盒修复后 box025 Stability ≥ 95%, Contact 变化 < 15%
- C2: box023 (小箱子) Contact<10cm ≥ 50% (单人可及)
- C3: box001/box024 Stability ≥ 90%
- C4 (HDMI 对比): 明确判定 data vs algorithm 问题

## 执行计划

### Track A: 远程并行 E041c (6 runs, ~36min 总耗时)

**脚本**: `workspace/core4d/scripts/run_E048_remote.sh`

```
GPU0 (串行): box025_person1 → bucket010_person1 → box001_person1  (~18min)
GPU1 (串行): desk005_person2 → box023_person1 → box024_person1    (~18min)
```

### Track B: 本地 HDMI 对比 (E048a)

1. 写 `convert_core4d_to_hdmi.py` — CORE4D qpos → HDMI motion.npz (FK + 上采样 30→50fps)
2. 创建 HDMI task YAML `move_box023.yaml`
3. 创建 HDMI scene XML (rename object→suitcase)
4. 运行 `run_hdmi.py task=move_box023`

### 时间线

```
T+0:   git push, ssh 远程启动 Track A
T+0:   本地开始写 convert_core4d_to_hdmi.py (Track B)
T+30m: Track A 远程结果回收 (scp)
T+60m: Track B HDMI 数据转换完成
T+90m: Track B HDMI 实验完成
T+120m: 全部评估 + 写日志
```

## 评估标准

| 指标 | 定义 | 旧 E041c Baseline |
|------|------|-------------------|
| MPKPE | Mean Per-Keypoint Position Error (cm) | 1.4 cm |
| Contact<10cm | 双手距物体 <10cm 帧占比 | 66% (box025) |
| Stability>0.6 | pelvis z > 0.60m 帧占比 | 100% |

**碰撞盒修复后的新 baseline 将由本次实验建立。**

## 关键文件

| 文件 | 操作 |
|------|------|
| `workspace/core4d/scripts/run_E048_remote.sh` | 新建 — 远程 2-GPU 并行脚本 |
| `workspace/core4d/scripts/convert/convert_core4d_to_hdmi.py` | 新建 — CORE4D→HDMI 数据转换 |
| `/home/ubuntu/Workspace/HDMI/cfg/task/G1/hdmi/move_box023.yaml` | 新建 — HDMI task config |
| `workspace/core4d/results/E048/` | 结果目录 |
