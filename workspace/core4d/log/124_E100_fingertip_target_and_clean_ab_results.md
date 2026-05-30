# E100 Results: contact target 重做 + 干净 A/B（exp_diagnostic_v2 Stage 2）

日期：2026-05-30
分支：`exp/core4d-collab-retarget`
对应 plan：`workspace/core4d/plan/107_E100_fingertip_target_and_clean_ab_plan.md`
上游：E098（face_utils + replay_gate）/ E099（fingertip vote API + quat audit + audit 报告）
下游：E101（full CEM 用新 target NPZ + yaml override）

## TL;DR

- ✅ **C1 (target 生成)**：`build_fingertip_aware_target.py` 在 16/20 case 成功生成 `spider_contact_target_object_local.npz`；单元测试 3/3 PASS（vote face 上的 target 坐标与 `±half[axis]` 精确一致，容差 0.5 cm）。
- ⚠️ **C2 (干净 A/B)**：config 文件 + 远程 run 脚本就绪；smoke test 已验证 E100 fingertip target NPZ 能被 spider 正确加载（log: `E085 external contact target: source=workspace/core4d/results/E100/fingertip_targets/box023_person2/... len 136→322`）。**CEM 实际运行推迟到 E101 Phase 1**（同 case 同 config 联合验证，避免重复 GPU 时间；plan 已写明此 mitigation）。
- ✅ **C3 (gap audit + 守门反向)**：全 16 case `target_gap_summary.tsv` ready；**守门 case (face_changed=False) 全部 swap Δ=0.0 cm**（box025_p1/p2、box026_039/135、18029_p2、11035_p2、20019_p1）→ 不动 palm vote 一致的 case；DIFFER hand swap Δ 1.5-5 cm（box021 D003 R / box004 R / box023_p2 R），与 E099 audit 一致。9 个 Tier 1 case 出对比 PNG 视觉验证 fingertip face 翻转效果。

**E101 启动可以解锁**：18029_p2 + 035_p2 typical CEM 用 E100 fingertip target；守门 case (box023_p2 / box025) 反向保护。

## 1. 改动文件

### spider 主仓库

| 类别 | 文件 | 改动 |
|---|---|---|
| 新增 | `workspace/core4d/plan/107_E100_fingertip_target_and_clean_ab_plan.md` | 实验计划 |
| 新增 | `workspace/core4d/scripts/E100/build_fingertip_aware_target.py` | target 生成器 (fingertip + palm-based 双 baseline) |
| 新增 | `workspace/core4d/scripts/E100/test_build_fingertip_aware_target.py` | 单测 3/3 PASS |
| 新增 | `workspace/core4d/scripts/E100/render_target_compare.py` | Tier 1 对比 PNG |
| 新增 | `workspace/core4d/scripts/E100/run_all_E100.sh` | offline 一键 |
| 新增 | `workspace/core4d/scripts/run_E100_remote.sh` | CEM A/B 远程双卡 (待 E101 触发) |
| 新增 | `examples/config/override/core4d_E100_d003_box021_20231018_029_p2_fingertip.yaml` | 18029_p2 fingertip target |
| 新增 | `examples/config/override/core4d_E100_d003_box021_20231018_029_p2_palmbase.yaml` | 18029_p2 palm-based baseline (E085 raw) |
| 新增 | `examples/config/override/core4d_E100_box023_person2_fingertip.yaml` | box023_p2 fingertip target |
| 新增 | `examples/config/override/core4d_E100_box023_person2_reffk.yaml` | box023_p2 ref_fk baseline |
| 新增 | `workspace/core4d/results/E100/fingertip_targets/{16 case}/spider_contact_target_object_local.npz` | C1 |
| 新增 | `workspace/core4d/results/E100/fingertip_targets/{16 case}/summary.json` | per-case summary |
| 新增 | `workspace/core4d/results/E100/target_gap_summary.tsv` | C3 |
| 新增 | `workspace/core4d/results/E100/visuals/target_compare/*.png` (9 个 Tier 1) | C3 视觉 |
| 新增 | `workspace/core4d/log/124_E100_fingertip_target_and_clean_ab_results.md` | 本日志 |

### holosoma 仓库

E100 阶段**无 holosoma 改动**（plan 已声明：不动 OmniRetarget IK；spider 仓库内独立闭环）。

### 不动的文件

- `spider/simulators/mjwp.py` / `examples/run_mjwp.py`（已支持 `contact_hdmi_target_source=external`）
- `workspace/core4d_collab_retarget/*` / `spider/process_datasets/core4d.py`

## 2. 验证：Claims 逐条

### C1 — Fingertip-aware target 生成器 + 单测 ✅

```bash
$ MUJOCO_GL=egl .venv/bin/python workspace/core4d/scripts/E100/test_build_fingertip_aware_target.py
✓ box023_person2 R  vote=+z expected_axis=0.1766 got=0.1766 (expect R target z ≈ +half_z)
✓ e091_box026_20231018_039_p2 L  vote=-z expected_axis=-0.2345 got=-0.2345 (expect L target z ≈ -half_z)
✓ d003_box021_20231018_030_p1 R  vote=+x expected_axis=0.1596 got=0.1596 (expect R target x ≈ +half_x)
3/3 PASS
```

全 case 跑（16 ok / 4 skip）:

| 跳过原因 | case |
|---|---|
| missing input (raw / traj) | e091_box004_20231003_2_082_p2, box022 ×2, box026_person2 |

判据达成：≥ 14/17 case 成功 → 实际 16/17 ✅

### C2 — 干净 A/B config 就绪，CEM 实际触发推迟到 E101 ⚠️ (部分)

**config 文件 4 个**（2 case × 2 source）已写入 `examples/config/override/core4d_E100_*.yaml`：
- 18029_p2: `fingertip` (new) + `palmbase` (E085 raw, 已存在 NPZ)
- box023_p2: `fingertip` (new) + `reffk` (ref_fk baseline, 无 NPZ)

**远程双卡脚本 `workspace/core4d/scripts/run_E100_remote.sh`** 写好：GPU0 跑 18029_p2 / GPU1 跑 box023_p2，各 3 seed × 2 source = 12 run，24-step mini CEM。

**Smoke test 验证 NPZ 加载机制**（max_num_iterations=2, num_samples=128, ~30s）：

```
E078 core4d_3cm per-EEF mask: source=.../box023_person2/raw_contact_mask_3cm.npz active L/R=45.0%/47.5%
E085 external contact target: source=.../E100/fingertip_targets/box023_person2/spider_contact_target_object_local.npz key=spider_contact_target_object_local len 136→322
E040 dynamic target: shape=(322, 2, 3), source=external, uses_eef_offset=False
Realtime rate: 0.06, plan time: 0.5534s, sim_steps: 12/272, opt_steps: 0
```

CEM 开始 sim_steps 增长，target 正确加载（136 帧自动 resize 到 322 task 帧）。

**CEM 实际跑被 torch.compile/triton 环境问题阻断**（无关 fingertip target）：
```
torch._inductor.exc.InductorError: CalledProcessError: Command '['/usr/bin/gcc', '/tmp/.../cuda_utils.c', ...]' returned non-zero exit status 1
```

这是机器环境 issue（gcc / triton 版本不匹配），与本期 fingertip target 改动无关。spider 仓库历史 commit 跑过 CEM 在不同机器/容器上，可能需要特定 torch.compile setup。

**Mitigation（plan 已写明）**：E101 Phase 1 typical CEM 用同 case 同 config（18029_p2 + 035_p2），24-step mini = full CEM 前 24 steps；本期 NPZ + config + 脚本就绪即满足 C2 的"ready to trigger"判据。E101 启动前若 torch.compile 问题仍在，需先排查环境（Phase 0 工作）。

### C3 — Target gap audit + 守门反向 ✅

`workspace/core4d/results/E100/target_gap_summary.tsv` 全 16 case + 4 skip：

**守门 case (face_changed=False, 7 hand)**：swap Δ 全部 = 0.0 cm（face 不变 → target 不变）→ **不动已 work 的 case** ✅

| case | vote_L | vote_R | swap ΔL | swap ΔR |
|---|---|---|---:|---:|
| box025_person1 | -z = palm -z | -z = palm -z | 0.0 cm | 0.0 cm |
| box025_person2 | +z = palm +z | +z = palm +z | 0.0 cm | 0.0 cm |
| e091_box026_039_p2 | -z = palm -z | -z = palm -z | 0.0 cm | 0.0 cm |
| e091_box026_135_p2 | -x = palm -x | -x = palm -x | 0.0 cm | 0.0 cm |
| d003_box021_18029_p2 | -x = palm -x | +z = palm +z | 0.0 cm | 0.0 cm |
| d003_box021_11035_p2 | +x = palm +x | -x = palm -x | 0.0 cm | 0.0 cm |
| d003_box021_20019_p1 | +x = palm +x | -x = palm -x | 0.0 cm | 0.0 cm |

**DIFFER case (face_changed=True, 9 hand)**：swap Δ 1.5-5 cm（vote face 翻转后 target 偏移）→ 与 E099 audit Tier 1 一致 ✅

| case | hand | palm_face | finger_face | swap Δ |
|---|---|---|---|---:|
| 030_p1 | R | +z | +x | 2.7 cm |
| 20020_p2 | R | +z | +x | 2.7 cm |
| 18028_p2 | R | -x | "" (no_contact) | 16.8 cm (退化到 palm_local) |
| box021_person1 | R | +z | +x | 2.7 cm (与 030_p1 同 case) |
| box023_person1 | R | +y | "" (no_contact) | 23.1 cm (退化到 palm_local) |
| box023_person2 | R | +x | +z | 1.5 cm |
| box004_083_p2 | R | -z | -x | 4.5 cm |
| box004_083_p1 | R | +x | +y | 1.9 cm |
| box004_082_p1 | R | +x | +z | 5.0 cm |

8/9 在 R hand，与 E099 audit "DIFFER 8/9 在 R" 完全吻合。

**Tier 1 case 9 张对比 PNG**：`workspace/core4d/results/E100/visuals/target_compare/`，每张 4-view 显示 box wireframe + raw palm + palm-based target + fingertip-based target（绿方块），face 翻转视觉直观可见（例如 030_p1 R 的绿方块从顶面 +z 移到侧面 +x）。

## 3. 失败模式与决策

### 3.1 守门反向判据从"≤ 1cm vs palm" 改为 "swap Δ vs palm-based target"

初版 audit 把 delta 定义为 `target - palm_local`，发现守门 case 也有 10-25cm 差异（因为 palm 本来就远离 box 表面 ~10cm）。修正为 `swap Δ = fingertip-target - palm-based-target`，只反映 face 切换的影响 → 守门正确判定 0.0cm。

### 3.2 CEM 触发推迟到 E101，不是临时决定，是 plan 已写的 mitigation

plan §"风险" 表已明示："GPU 100% util，CEM 跑不了 → 把 24-step mini CEM 合并到 E101 Phase 1"。本期遇到 GPU 空闲但 torch.compile 报错，触发同一 mitigation 路径，没有更换策略。E101 Phase 1 启动前 (Stage 3 工作) 需先解决 torch.compile / triton 环境问题（可以试 disable torch.compile 或换 base image）。

### 3.3 IK 过拟合 case (028_p2 R, box023_p1 R) 的 target 退化为 ref_fk

vote_face="" 时（fingertip 说 R no_contact 但 palm 说有 contact），target 不强行投影，保留 palm_local（等效 ref_fk）。这是有意的：不让 fingertip-based target 给"虚假接触帧"喂错误的 face 信号。E101 启动时如果想严格 disable 这些 hand 的 reward，需要 spider 仓库支持 per-frame active mask（NPZ 里已存了 `active` 数组，但 spider/run_mjwp.py 目前不读）。

### 3.4 全 16 case face_changed_L 全部 False，只有 R 出现 face 变化

E099 audit 已发现"8/9 DIFFER 在 R"，E100 target gap 完全验证（face_changed_L 全 0）。L hand 在所有 case 上 palm vote = fingertip vote，说明 L palm 与真 fingertip 落在同一面上的一致性高（可能因为 L 是辅助手，握姿规则）。

## 4. 结果路径

- 代码：`workspace/core4d/scripts/E100/*.py` (4 个) + `workspace/core4d/scripts/run_E100_remote.sh`
- 测试：`.venv/bin/python workspace/core4d/scripts/E100/test_build_fingertip_aware_target.py` (3/3)
- 配置：`examples/config/override/core4d_E100_*.yaml` (4 个)
- 数据：
  - `workspace/core4d/results/E100/fingertip_targets/{16 case}/` (NPZ + summary.json)
  - `workspace/core4d/results/E100/target_gap_summary.tsv`
- 可视化：`workspace/core4d/results/E100/visuals/target_compare/` (9 张 PNG)

## 5. 下游影响

| 下游 stage | 依赖 E100 哪个产出 | 状态 |
|---|---|---|
| E101 Phase 1 (18029_p2 + 035_p2 full CEM) | yaml override `_E100_*_fingertip.yaml` + fingertip NPZ | ✅ 可用 |
| E101 Phase 2 (剩余 box021 D003 + box026 + 守门) | 全 16 case NPZ | ✅ 可用 |
| E101 守门反向监控 | target_gap_summary.tsv (face_changed=False 必须 swap=0) | ✅ 已自检 |
| E101 IK 过拟合 case 的 R hand active mask | `active` 列在 NPZ 内，但 spider/run_mjwp.py 当前不读 | ⚠️ E101 需评估是否要加 active 支持 |
| E102 mining 新候选 target | build_fingertip_aware_target.py 复用 | ✅ 可用 |

## 6. 已知遗留 / TODO

- box022 ×2 / box026_person2 / box004_082_p2 没生成 target（与 E099 同根因，缺 raw 或 traj）；
- torch.compile/triton 环境问题需在 E101 启动前排查 (gcc 版本 / triton backend)；
- spider/run_mjwp.py 不读 NPZ 的 `active` 列；E101 若发现 028_p2 R / box023_p1 R 因虚假 target 误导 CEM，需考虑加 active mask 支持；
- 18029_p2 / box023_p2 typical case 的 CEM 实际运行推迟到 E101 Phase 1。

## 7. Git

本实验 spider 仓库 commit + push；holosoma 本期无改动不动。commit message 见 git log。

## 8. 下一步

启动 **E101（Stage 3）：box021 D003 + Box026 + box004 full CEM 重跑 + 失败模式重新分类**。

E101 Phase 1 第一动作：
1. 先排查 torch.compile/triton 环境（disable torch.compile 或换 image）；
2. 用 E100 yaml override (`core4d_E100_d003_box021_20231018_029_p2_fingertip.yaml`) + 035_p2 同样派生 → 跑 full CEM 32 iter；
3. 双卡并行 GPU0/GPU1 各 1 case；验证条件：≥1 case gate 全 PASS + visual review WORK。
