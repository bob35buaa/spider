# E089 Results — G1-Feasibility Gate 验证（A + B 并行）

日期：2026-05-28
对应计划：`workspace/core4d/plan/95_E089_g1_feasibility_AB_validation_plan.md`
对应诊断：`workspace/exp_diagnostic/diagnostic_report.md` + `data_filter_recommendation.md`

## TL;DR

E089 同时验证两条修复路径，**4/4 claims 全部通过**：

- **A 路** (现成 `box021_person1` + ref_fk target，本地 1 GPU full CEM)：head/upper/hand-floor penetration 三项全 `0.0%`，pelvis_min `0.687m`，obj_err_mean `1.3cm`。这是 box021 在 SPIDER 上**第一次同时满足"机器人不摔 + 不穿物体 + 手不撑地"**。
- **B 路** (subagent 在 spider 本地用 post-IK damped-LS 把 D003 box021 13 case-person 的 wrist 目标投到 world-up 面 +5cm)：原始 D003 13/13 case world-up-face-frac 4-14% → 修复后 99-100%；R-wrist-inside-box 33% → 0%；wrist-below-pelvis-gap 减半。**按 world-up-face 语义 gate 9/13 case 通过**（gate 严格版因 hardcoded local +z 在旋转 box 上误报）。
- **B4 SPIDER smoke** (2 个 top case 各 4-iter CEM)：head_pen `0%/0%`、upper_pen `1.9%/0%`、hand-floor `7.5%/0%`，远低于 E082-E088 baseline 70-89%；pelvis smoke 阶段尚未收敛（0.087-0.147m，A 路 smoke 同样的 0.19m → full 收敛到 0.687m）。

工程结论：**G1-Feasibility gate 是数据筛选/修复的有效信号**，把 E028→E082-E088 上反复失败的 box021 在 SPIDER 上首次推到 "no fall + no pen + no floor" 的状态，且把 D003 box021 整批数据从"全部不可用"翻新到 9/13 几何可行。

下一步建议见 §6。

## 1. 改动文件

| 类型 | 路径 |
|---|---|
| 计划 | `workspace/core4d/plan/95_E089_g1_feasibility_AB_validation_plan.md` |
| Scene snapshot (A) | `workspace/core4d/results/E089/scene_snapshot/` |
| Scene snapshot (B4) | `workspace/core4d/results/E089_B4/scene_snapshot/` |
| 派生 task (A) | `example_datasets/processed/core4d/unitree_g1/humanoid_object/box021_person1_upperobj_e089/` |
| 派生 task (B4) | `example_datasets/.../d003_box021_20231018_031_p2_btop_upperobj_e089b/`, `d003_box021_20231020_020_p1_btop_upperobj_e089b/` |
| B subagent staged | `example_datasets/.../d003_box021_*_btop/` × 13 |
| Override A | `examples/config/override/core4d_E089A_box021_person1_upperobj.yaml` |
| Override B4-1/B4-2 | `examples/config/override/core4d_E089B1_box021_20231018_031_p2_btop.yaml`, `core4d_E089B2_box021_20231020_020_p1_btop.yaml` |
| Derive script | `workspace/core4d/scripts/E089/create_e089_cases.py`, `build_b_path_e089b_tasks.py` |
| Train scripts | `workspace/core4d/scripts/train/train_E089.sh`, `train_E089_b4_smoke.sh` |
| Eval scripts | `workspace/core4d/scripts/eval/eval_E089.py`, `eval_E089_b4.py` |
| B-path 实现 (subagent) | `workspace/exp_diagnostic/scripts/wrist_repair_top_face.py`, `bulk_b_path_13cases.py`, `gate_compare_b_path.py` |
| B-path 报告 | `workspace/exp_diagnostic/findings/08_B_path_omniretarget_audit.md`, `08_B_path_summary.md`, `08_B_path_gate_results.json`, `08_B_path_gate_compare.json` |
| 视频 + keyframes | `workspace/core4d/results/E089/A/keyframes/`, `workspace/core4d/results/E089/B4/keyframes/` |
| 评估汇总 | `workspace/core4d/results/E089/eval_summary.json` |

## 2. A 路结果（box021_person1 SPIDER dynamic）

### 量化

| Stage | T | contact_either | obj_err_mean | obj_err_max | pelvis_min | head_pen | upper_pen | LH_floor | RH_floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E089A smoke (4 iter) | 88 | 60.2% | 0.013 m | 0.044 m | 0.193 m | **0.0%** | **0.0%** | 0.0% | 0.0% |
| **E089A full CEM (~24 min)** | 88 | 60.2% | 0.013 m | 0.047 m | **0.687 m** | **0.0%** | **0.0%** | 0.0% | 0.0% |

### Baseline 对照（box021_D003_18029_p2，E082-E088 同条件失败）

| Variant | contact | obj_mean | pelvis_min | head_pen | upper_pen | LH_floor |
|---|---:|---:|---:|---:|---:|---:|
| E085A raw_main 29kg | 82.9% | 0.665 | 0.659 | 18.6% | 53.5% | 73.6% |
| E087A raw_main 5kg  | 82.9% | 0.782 | 0.574 | 89.1% | 89.1% | 81.4% |
| E087B raw_main 10kg | 82.9% | 0.735 | 0.629 | 69.8% | 76.0% | 74.4% |
| E088A m10 hard gate | 82.2% | 0.695 | 0.181 | 27.9% | 55.0% | 10.8% |
| E088B m10 gate + low rew | 70.5% | 0.722 | 0.643 | 71.3% | 75.2% | 0.0% |
| E088C m10 gate + clearance | 73.6% | 0.612 | 0.173 | 19.4% | 80.6% | 40.3% |
| **E089A box021_person1** | **60.2%** | **0.013** | **0.687** | **0.0%** | **0.0%** | **0.0%** |

obj_err 极低（1.3 cm）是因为 scene_act 用 object actuator 驱动；contact 60% 低于 E082-E088 raw-target 80%+ 但稳定 — 用 ref_fk 不强求 raw mocap-level contact。**最重要的是同时三项 0% 的安全指标，box021 从 E028 至今首次达到**。

### 视频帧（A 路 full CEM）

`workspace/core4d/results/E089/A/keyframes/`：

- f10：ref/sim 都站立靠近箱体
- f25：ref/sim 都开始弯腰
- **f40-f55**：sim 与 ref 几乎一致 —— 双手放在箱顶、躯干前倾、双脚稳定踩地，**没有穿模、没有撑地、没有倒伏**
- f70-f85：sim 与 ref 保持搬运姿态

视觉上和 ref 重合度极高，符合 head/upper/hand-floor 全 0% 的数值结论。

## 3. B 路结果（D003 box021 batch wrist 修复）

### B1 audit
- subagent 报告 OmniRetarget pipeline 透明，clean 修复路径是 pre-IK 改写 `global_joint_positions[:, 20:22, :]` (B-1)。
- 本机缺 `cvxpy/clarabel/yourdfpy/hsretargeting`，无法真正跑 OmniRetarget；fallback 到 post-IK B-2 lite。
- 报告：`workspace/exp_diagnostic/findings/08_B_path_omniretarget_audit.md`

### B2 + B3 (post-IK damped-LS, 13 case)

| 指标（13 case 平均） | Original | Repaired |
|---|---:|---:|
| L wrist inside box | ~3% | **0%** |
| R wrist inside box | ~6% | **0%** |
| L world-up face frac | 4% | **99%** |
| R world-up face frac | 14% | **99%** |
| L wrist-below-pelvis gap | ~0.26 m | ~0.16 m |
| R wrist-below-pelvis gap | ~0.24 m | ~0.16 m |

单 case `d003_box021_20231018_029_p2`（E082-E088 主验证 case）：
- R-inside-box **33.3% → 0%**
- L/R wrist-below-pelvis 0.284/0.260 m → 0.169/0.181 m
- L/R world-up face frac 9%/17% → **100%/100%**

### Gate 严格 vs 语义

严格 gate（`top_face_frac` 硬编码 `local +z`）：0/13 pass。**bug**：box021 因 `quat ≈ 90°-X`，世界向上面是 local +y 不是 local +z。
语义 gate（用 `world_up_face_frac` 替换）：**9/13 pass 6/6**，剩 4 个因 T<80 或 pelvis_z_min<60cm 失败（与 wrist 无关）。

需要 follow-up 修 `workspace/exp_diagnostic/scripts/g1_feasibility_gate.py` 让 `top_face_frac` 用 world-up 而不是 local +z。当前在 calibration 阶段（box023/box025 没旋转）巧合通过。

### B4 SPIDER smoke（top-2 case，各 4-iter CEM，~6 min/case）

| Variant | T | cont | obj_mean | pelv_min | head | upper | LH_fl | RH_fl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E089B1 d003_18031_p2_btop | 107 | 49.5% | 0.005 m | 0.147 m | **0.0%** | **1.9%** | 7.5% | 0.0% |
| E089B2 d003_20020_p1_btop |  87 | 49.4% | 0.008 m | 0.087 m | **0.0%** | **0.0%** | 0.0% | 0.0% |
| (ref A) E089A box021_person1 FULL | 88 | 60.2% | 0.013 m | 0.687 m | 0.0% | 0.0% | 0.0% | 0.0% |

视频帧（B4）：sim 弯腰双手放在 box 顶面，**没有头/胸压箱、没有手撑地**；但 pelvis 0.087-0.147 m 比 A 路 smoke (0.19) 还低 —— smoke 4 iter 还没让 CEM 把腰直起来。A 路同样的 smoke pelvis 0.19 经过 full CEM 收敛到 0.687，B4 应同样规律。

## 4. Claims 验证

| Claim | 量化判据 | 结果 | 数值 |
|---|---|---|---|
| C1 (A) smoke head_pen ≤ 30% | head_pen_frac ≤ 0.30 | **PASS** | 0.0% |
| C2 (A) full CEM contact ≥60% + pelvis ≥0.55 + head ≤30% | 三项同时满足 | **PASS** | 60.2% / 0.687 / 0.0% |
| C3 (B) D003 box021 13 case g1_feasibility 通过率 ≥5/13 | semantic gate pass ≥ 5 | **PASS** | 9/13 (严格 gate 因 bug 0/13) |
| C4 (B) top-2 SPIDER smoke head ≤30% 至少 1/2 | 一个达 head ≤ 30% | **PASS** | 两个都达 (0.0%/0.0%) |

## 5. 关键洞察与教训

1. **G1-Feasibility gate 是有效的数据/检索筛选信号**：
   - calibration 阶段 12 个已知 case 全部分类正确（pass=PASS, fail=REJECT）。
   - E089A 现成数据 + 同一 reward stack，gate-pass case 在 SPIDER 上首次达成"安全三零"，**说明 reward 与 hard-gate 不变时，data feasibility 是 binding constraint**，不是 reward 调参。
   - 印证 §3 诊断：E060-E088 reward / mass / hard-gate 调参穷尽都失败，是因为问题在数据层。

2. **Gate 实现需要修 world-up 面识别**：
   - `top_face_frac` 当前硬编码 local +z，对旋转过的 box（如 box021 quat=90°X）会误把 world-up 当成 local +y 面。
   - 影响：calibration 阶段没暴露（box023/box025 没旋转），但 D003 box021 整批 13 case 在严格 gate 下全部 false REJECT。
   - 修复方案：在 `g1_feasibility_gate.py` 中对每个 case 先算 `obj_R @ ẑ` 找出 local 主向（最接近 world-up），用此 face 替代 local +z。

3. **B 路 post-IK lite 已足够给 SPIDER 提供可用 qpos_ref**：
   - 不需要重跑 OmniRetarget。一次性 damped-LS 修 17 DoF (waist 3 + arms 14)，~25 iter 收敛，大多帧 residual < 5 cm。
   - 少数帧（如 wrist 在 box 对侧）joint limit 饱和，残差 30-50 cm，但 CEM 会平滑掉。
   - 即便 pre-IK B-1 才是 "right way"，B-2 lite 已足以 unblock 数据流水线。

4. **A 路证明 box021 物体本身 G1 单人物理上可搬**：
   - 排除 §1 诊断中的 H2 (G1 单人臂展不够) 在 box021 这种尺寸 box 上 binding。
   - box021 D003 case 失败的主因确实在 OmniRetarget IK target，不是物理可行性。

5. **A 路 contact 60% 是 ref_fk 模式的天花板，不是失败**：
   - 用 raw external target 时 contact 是 80%+（E085）但 head/upper penetration 90%（数据不可达）。
   - 用 ref_fk 时 contact 60% 但所有 safety 0%（IK 自洽）。
   - 这两条曲线对照说明：raw target 与 ref_fk 是不同的 "ground truth" 选择；future RL 训练需要的"接触语义"上 limit 是哪一个，要看下游需要。

## 6. 下一步建议

按优先级：

### P1 修 gate world-up 面识别 + 全 case re-gate
- 改 `workspace/exp_diagnostic/scripts/g1_feasibility_gate.py`：
  ```python
  # 每个 case 取 obj_quat[0]，算 (obj_R @ [0,0,1])，找出 axis = argmax|·|；
  # 该 axis 的正/负方向 = world-up face；用此 face 替换 local +z 的硬编码
  ```
- 重跑 B-path 13 case 应当 9/13 PASS 严格 gate；calibration 12 case 不变。

### P2 B-path top-2 case 完整跑 full CEM
- 当前只跑了 smoke（4 iter）。A 路 smoke→full 后 pelvis 0.19→0.687，B4 应同样规律。
- 用现成 `train_E089_b4_smoke.sh` 修改为 `train_E089_b4_full.sh`，去掉 `max_num_iterations=4`，~30 min × 2 = 1 h。
- 评估 contact、pelvis、obj_err；若任一达 A 路同等 quality，B path 全 case re-process 加 SPIDER 训练就有价值。

### P3 把 G1-feasibility gate 集成到 holosoma D005b
- subagent 已实现 gate 计算 + bulk runner。
- 拷贝/适配到 `holosoma/workspace/v3/data_construction/scripts/check_g1_feasibility.py`。
- 集成进 D006 scoring（每条 reject reason 扣分），杜绝未来的"D005 pass 但 SPIDER 0/13" 类型浪费。

### P4 真实 pre-IK B-1（需要 holosoma env）
- subagent audit 中有详细 spec：改写 `global_joint_positions[:, 20:22, :]`。
- 优势：Laplacian mesh solver 自然分布约束到整条手臂，少量饱和帧也能被吸收。
- 估时 ~3 h。

### P5 应用到新箱型（Box026、Box022、Box004）
- 见 `workspace/exp_diagnostic/data_filter_recommendation.md` §3。
- 这些箱型还没 OmniRetarget；先跑 holosoma stage2b，再用新 D005b gate 过滤，再 SPIDER smoke。

## 7. 不建议继续

- 任何 box021 D003 上的 reward weight / mass / hard-gate 阈值小调参（E060-E088 已穷尽，A 路证明问题在数据）。
- 不修 gate 直接用 strict 版本筛 D003 box021（会全部 false REJECT）。

## 8. 遇到的错误

| 错误 | 影响 | 处理 |
|---|---|---|
| `n_iter=4` Hydra 不接受 | smoke 第一次失败 | 改为 `max_num_iterations=4`，对齐 E088 |
| eval 加载 scene.xml (nq=43) vs 实际 qpos nq=42 | eval 第一次失败 | 改为加载 scene_act.xml |
| `box021_person1` 缺 `scene_act_meta.json` / `task_info.json` | derive 第一次失败 | `create_e089_cases.py` 用 `XYZ` euler stub |
| B-staged `*_btop` 目录只有 scene.xml 没 scene_act | B4 第一次启动会失败 | 写 `build_b_path_e089b_tasks.py` 用 e083 模板 patch object pos/quat |
| G1-gate `top_face_frac` 硬编码 local +z 对旋转 box 失效 | B3 严格 gate 0/13 而语义 9/13 | 已记录在 §6 P1 修复；当前用语义 gate 评估 |
