# E058: bucket005_s2 + Snap Warmstart 喂入 CEM (Path B-CEM 首跑) — 结果

## 状态: ❌ Claims 2/6 通过 (流水线 OK, 内容失败 — 双方 CEM 都摔倒)

## 实验配置

| 项 | 值 |
|---|---|
| 实验类型 | CEM rollout (1024 samples × 32 iter) on MJWP scene_act |
| Case | `bucket005_s2_person1` (E057 推荐, 唯一对侧握姿) |
| Reward stack | E041c (additive ori 0.3, dynamic target, contact_hdmi gain=5.0, body track local-frame) |
| Warmstart | E057 `warmstart_qpos.npz` 喂入 hook, intent 内 88f×ref_steps(2)=176 帧 ref+ctrl 替换 |
| GPU | parallel: GPU 0 baseline, GPU 1 warm |
| Wall time | baseline **36 min**; warm **3h54min** ⚠️ (intent 内 plan time 14s → 99s, 7x 慢) |

## 训练命令

```bash
bash workspace/core4d/scripts/train/train_E058.sh parallel 0 1
# 等价: +override=core4d_e041c task=bucket005_s2_person1 +use_torch_compile=false
#       output_dir=<per-run> [+warmstart_qpos_path=...]
```

## 输出文件

| 路径 | 内容 |
|---|---|
| `workspace/core4d/results/E058/E058_baseline.{npz,mp4}` | baseline CEM rollout |
| `workspace/core4d/results/E058/E058_warm.{npz,mp4}` | warm CEM rollout |
| `workspace/core4d/results/E058/eval_summary.csv` | 2 行: contact% / stable% / face / pelvis |
| `workspace/core4d/results/E058/face_dist_E058.png` | 2×2 face dist 时序 (BASELINE vs WARM × L/R) |
| `workspace/core4d/results/E058/keyframes/E058_*_t*.jpg` | 10 帧 (baseline + warm) × 5 时间戳 |
| `logs/E058/E058_{baseline,warm}.log` | 训练日志 |

## 数值结果 (eval_summary.csv)

| 指标 | baseline | warm | Δ | 期望 |
|------|----------|------|---|------|
| L palm contact % | 36.4% | 35.2% | **-1.1pp** | warm > baseline +10pp |
| R palm contact % | 37.5% | 20.5% | **-17.0pp** ❌ | 同上 |
| both palm contact % | 12.5% | 9.1% | -3.4pp | 同上 |
| stable % (full episode) | 38.5% | 35.1% | -3.4pp | warm ≥ 80% |
| **stable % (intent only)** | 12.5% | **33.0%** | **+20.5pp** ⭐ | — |
| pelvis_min (m) | 0.110 | 0.109 | ≈ 0 | ≥ 0.50 (站立) |
| pelvis_min_intent (m) | 0.110 | 0.109 | ≈ 0 | 同上 |
| L main face | None | None | — | -yz |
| R main face | None | None | — | +yz |

**核心发现**: **两个 run 的 pelvis_min ≈ 0.11m = 倒地状态**。warmstart 没机会"帮助" — baseline 本身在 bucket005_s2 上就站不住。

## Claims 验证

| ID | 描述 | 量化标准 | 实际 | 通过 |
|---|---|---|---|---|
| C1 | warmstart hook 跑通, CEM 收敛无 NaN | npz + mp4 + 32 iter 全部正常 | ✅ 两 run 都到 sim_step 296/296 | ✅ |
| C2 | warm contact 比 baseline +10pp | both_contact_pct +10pp | both -3.4pp, R -17pp | ❌ |
| C3 | warm stability ≥ 80% | stable_pct ≥ 0.80 | 35.1% (摔了) | ❌ |
| C4 | warm main face = -yz/+yz | L=-yz AND R=+yz | 都 None (没稳定贴面) | ❌ |
| C5 | 视频 warm ≥ 3/5 视觉优势 | A/B 5 帧目检 | 双方都摔, 不可比, 仅 t=2.10s warm 单手仍按桶 | ❌ |
| C6 | 一键脚本可复现 | run_E058 端到端 | ✅ train + eval + keyframes | ✅ |

**2/6 通过**。流水线 OK，但 warmstart 在当前 reward / sigma / case 组合下无法挽救基础不稳定。

## C5 视觉详情 (5 keyframe × 2 traj = 10 帧目检)

| 时间 | baseline | warm |
|---|---|---|
| 0.40s (pre) | 弯腰对桶, 单手贴边 | 同 baseline |
| 0.70s (intent start) | 开始倾斜 | 弯腰更深, 双手向桶靠 |
| 2.10s (intent mid) | **侧倒, 物体压顶** | **侧倒, 单手仍按桶顶** ← 唯一 warm 视觉略好 |
| 3.55s (intent end) | 跪在地上 | 跪在地上 |
| 4.20s (post) | 趴着 | 趴着 |

**两者都摔, 没有"成功搬运"**。warm 唯一边际优势: intent mid 时手仍在桶上 (baseline 完全脱手)。

## 关键观察

### 1. warm 在 intent 内 plan time 7x 慢 (15s → 99s @ sim_step 14)

sim_step 14 ≈ horizon 前进到 source frame 19 ≈ intent 起点 20。从那一刻起 CEM rollout 包含了 warmstart 帧 (双手贴桶 5cm 内), 接触约束爆炸 → MuJoCo 求解器迭代次数飙升。**这本身证明 warmstart 几何"在物理层起作用"**, 但 CEM 没有相应的 reward 把这股 contact 转为搬运。

### 2. stable_intent +20pp 是真信号

baseline 12.5% vs warm 33.0% (intent 88 帧内站立帧比例)。warmstart 让 intent 早期机器人"撑得久一点", 但还是在 intent 末段倒下。**这暗示 warmstart 起手好但 CEM 后续控制顶不住**。

### 3. R hand contact -17pp 是不对称失败

L 持平 (-1pp), R 暴跌 (-17pp)。可能 CEM 在被 warmstart 锁住的姿态下, 为了维持身体平衡只能"放弃"R hand 跟踪。E057 snap 已验证 R 在 +yz 100% close (snap 后), 但 CEM rollout 中 R 散到 26.66 cm (远超 7cm 阈值)。

### 4. 双方都摔 → 反向印证 ref qpos 与 sim 物理脱节严重

baseline (无 warmstart) 也 pelvis_min=0.11m。这是 53 个 CEM 实验的老问题:
- mocap ref qpos 用 OmniRetarget 生成, 不一定满足 G1 物理可行性
- E041c reward stack 在 box023 上调 (那里 ref 更接近物理可行); bucket005_s2 是新 case, ref 的 dynamic feasibility 未知
- CEM 1024×32 没有发现稳定的搬运策略

## 教训

1. **warmstart 不是"万能补丁"**: 它把"几何不可能"修成"几何可行", 但**物理可行性 + CEM optimizer 能力**还是两个独立瓶颈。E057 snap 通过几何验证, 不等于 CEM 能跑出来。

2. **要分阶段 baseline**: E058 直接对比 "baseline vs warm" 在一个**未验证**的 case (bucket005_s2 + E041c reward) 上, 没法分清失败原因。**正确做法**: 先确认 case + reward stack 的 baseline 站得稳, 再加 warmstart 测增量。

3. **CEM 反馈循环时间长 (warm 4 小时)**: 即使一次能跑通, 调参代价高。需要先用更小预算 (256 samples × 16 iter) 做 reward 扫描确认稳定基线, 再上满预算。

4. **stable_intent 信号值得追**: +20pp 不是噪声, 是 warmstart 真效果。问题是 CEM 没能"放大"这点优势, 反而被身体不稳压垮。

## 下一步路径分析 (供决策)

### A. 修 baseline (推荐先做)
**先在 bucket005_s2 + E041c 上加 stability_penalty (E034 引入但 E041c 关闭) + 更小 sigma**, 让 baseline 不摔, 再上 warmstart。预计 2 个 case × 2 配置 = 4 runs (~4 h)。

### B. Path B 改为 SDF-anchor
warmstart 几何对了 → 改 reward 锁住"双手停在 ±yz 5cm 内"。CEM 不跑 mocap tracking, 只跑 stability + face anchor。可能更直接但脱离 ref。

### C. 切到 dual-robot
bucket005_s2 是 person1, 但 CORE4D 数据是双人协作。**单人物理上可能从一开始就不可解** (类似 box025 E001-E009 的结论)。E054 列了 dual-robot=2 case, bucket005_s2 不在那 2 个里 — 但仍可能需要 partner force 配合。

### D. Drop bucket005_s2, 试其他 valid case
E056 列了其他 valid case (box023 / bucket007 / desk021 / bucket001)。E055 已在 box023 上做过 snap 但没接 CEM。**box023 baseline 摔倒概率比 bucket005_s2 低吗?** 不知道, 但可以先在 box023 上重做 E058, 看是否同样摔。

## 改动文件

| 文件 | 改动 |
|---|---|
| `spider/config.py` | +1 字段 `warmstart_qpos_path: str = ""`; 修 output_dir 不强制覆盖 (尊重 CLI) |
| `examples/run_mjwp.py` | +warmstart hook ~30 行 (load + interp + replace qpos_ref slices) |
| `workspace/core4d/scripts/train/train_E058.sh` | 新建, 并行/串行模式 |
| `workspace/core4d/scripts/eval/eval_E058.py` | 新建 ~200 行 (face dist + contact% + stable% + 2×2 png) |
| `workspace/core4d/scripts/eval/extract_E058_keyframes.sh` | 新建 |
| `workspace/core4d/scripts/run_E058.sh` | 新建 一键 |
| `workspace/core4d/results/E058/` | 新生成 (2 npz + 2 mp4 + 10 jpg + csv + png + 2 outdir) |
| `workspace/core4d/plan/68_E058_*.md` | 新建 |
| `workspace/core4d/log/68_E058_*.md` | 本文件 |
| `workspace/core4d/EXPERIMENT_TRACKER.md` | 待更新 E058 行 |

## 环境维护副产品 (本次发现并修)

1. `uv run` 触发 re-resolve 升级 torch 2.8.0 → 2.11.0 (与 nccl 2.27 不兼容); 按 env.md 装回 2.8.0
2. spider/interp.py 在 nearest 模式传 align_corners 给 torch (后者拒绝); 用 `repeat_interleave` 绕过
3. 缺 python3.12-dev → triton 编译失败; 改用 `+use_torch_compile=false`, CEM 慢 ~2x 但能跑
4. 未来 train script 改用 `.venv/bin/python`, 避免 uv re-resolve
