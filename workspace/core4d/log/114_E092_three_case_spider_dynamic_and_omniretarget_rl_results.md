# E092 三 case SPIDER dynamic + direct OmniRetarget smoke results

日期：2026-05-29

计划：`workspace/core4d/plan/98_E092_three_case_spider_dynamic_and_omniretarget_rl_plan.md`

状态：`FAIL / smoke-gated stop`

## 0. 执行说明

本轮按计划对三条 E091 case 做两条路线的 smoke 对照：

- Stage A `spider_dyn`: 三条 case 均跑 SPIDER/MJWP dynamic smoke。
- Stage B `rl_from_spider`: 计划要求只对 Stage A full `WORK` 序列执行；本轮 Stage A smoke 无 `PASS/REVIEW+`，因此无 full、无 `WORK` 输入，Stage B 按计划跳过。
- Stage C `rl_from_omni`: 三条 case 均跑 direct OmniRetarget smoke。

重要限制：当前仓库可见训练入口只有 `examples/run_mjwp.py` / MJWP 动态优化栈，未发现独立 PPO/RL trainer。因此本轮 `rl_from_omni` / `rl_from_spider` 脚本先接入同一 MJWP 训练/评估栈，保留计划中的 route 命名，但不把结果解释为 PPO main-RL 收敛结果。

## 1. 三卡并行与回收

| 阶段 | 本地 GPU0 | 远程 GPU0 | 远程 GPU1 | 回收 |
|---|---|---|---|---|
| Stage A `spider_dyn smoke` | C1 `box004_083_p2` | C2 `box026_039_p2` | C3 `box026_135_p2` | 已回收 |
| Stage C `rl_from_omni smoke` | C1 `box004_083_p2` | C2 `box026_039_p2` | C3 `box026_135_p2` | 已回收 |

远程执行使用 clean clone：

```text
spider-remote:/home/xiayb/pHRI_workspace/spider_e092_run
```

原远程 repo `/home/xiayb/pHRI_workspace/spider` 有未提交本地改动，未 reset，按远程指南改用独立 clean clone 并软链接旧 `.venv`。

## 2. 结果路径

| 类型 | 路径 |
|---|---|
| variants | `workspace/core4d/scripts/E092/variants.tsv` |
| scene snapshots | `workspace/core4d/results/E092/scene_snapshot/` |
| Stage A results | `workspace/core4d/results/E092/spider_dyn/smoke/` |
| Stage A logs | `logs/E092/spider_dyn/smoke/` |
| Stage C results | `workspace/core4d/results/E092/rl_from_omni/smoke/` |
| Stage C logs | `logs/E092/rl_from_omni/smoke/` |
| comparison | `workspace/core4d/results/E092/comparison/` |
| visual review | `workspace/core4d/results/E092/visual_review/` |
| global eval summary | `workspace/core4d/results/E092/eval_summary.json` |

## 3. Stage A: SPIDER dynamic smoke

| variant | case | T | contact | obj mean | obj max | pelvis min | head | upper | LH floor | RH floor | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `E092D1_box004_083_p2_dyn` | C1 | 105 | 3.8% | 0.006m | 0.016m | 0.074m | 0.0% | 0.0% | 0.0% | 1.0% | FAIL |
| `E092D2_box026_039_p2_dyn` | C2 | 123 | 58.5% | 0.008m | 0.062m | 0.191m | 0.0% | 0.0% | 0.0% | 0.0% | FAIL |
| `E092D3_box026_135_p2_dyn` | C3 | 82 | 69.5% | 0.006m | 0.020m | 0.186m | 0.0% | 0.0% | 0.0% | 39.0% | FAIL |

判定：三条均远低于 dynamic `WORK` pelvis 阈值 `0.55m`；C3 另有明显 RH floor shortcut。没有 smoke `PASS/REVIEW+`，因此不跑 Stage A full。

## 4. Stage C: direct OmniRetarget smoke

| variant | case | T | contact | obj mean | pelvis min | head | upper | LH floor | RH floor | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `E092O1_box004_083_p2_omni` | C1 | 105 | 3.8% | 0.006m | 0.073m | 0.0% | 0.0% | 0.0% | 1.0% | FAIL |
| `E092O2_box026_039_p2_omni` | C2 | 123 | 57.7% | 0.008m | 0.135m | 0.0% | 0.0% | 0.0% | 0.0% | FAIL |
| `E092O3_box026_135_p2_omni` | C3 | 82 | 69.5% | 0.006m | 0.189m | 0.0% | 0.0% | 0.0% | 40.2% | FAIL |

判定：三条均远低于 RL smoke pelvis 阈值 `0.45m`；C3 右手贴地与 Stage A 一致。因此不跑 Stage C main。

## 5. 可视化复核

产物：

- 6 个 mp4 均存在，时长约 `3.28-4.92s`。
- 60 张 keyframe jpg 非空。
- 已生成 6 张 contact sheet：`workspace/core4d/results/E092/visual_review/contact_sheets/`。
- high subagent 复核报告：`workspace/core4d/results/E092/visual_review/high_subagent_visual_review.md`，agent `019e6fc7-1fa9-7442-af87-b56c8f376236`。

视觉结论：

- Contact sheets 支持量化 FAIL：三个 case 都存在真实姿态失稳，C1/C3 尤其明显。
- C1 box004 两条路线几乎同轨迹：早期弯腰，随后 pelvis/hip collapse 并后仰倒地；物体稳定但 hand-object contact 语义失败。
- C2 Box026 039 没有完全摔平，但多帧低髋、半跪/压箱，direct Omni 路线后段更低，符合 pelvis `0.191m -> 0.135m`。
- C3 Box026 135 两条路线均中后段倒地，右手/右臂贴地明显，支持 RH floor `39-40%`。
- 未观察到明确 head 或 upper-body 穿箱，head/upper `0%` 与视觉一致。
- 物体 tracking 指标好，但人体接触和姿态语义失败；C3 还有箱体大角度翻转/抬起，交互不可信。

## 6. Claims 验证

| Claim | 结果 | 证据 |
|---|---|---|
| C1: 至少一条 E091 case 可通过 SPIDER dynamic 得到可用动态序列 | 不通过 | Stage A 三条 smoke 均 FAIL，pelvis min `0.074/0.191/0.186m` |
| C2: `rl_from_spider` 优于 `rl_from_omni` | 不可验证 | 无 Stage A full `WORK`，按计划跳过 Stage B |
| C3: Box026 near-pass 失败类型可被动态重定向区分 | 部分验证为失败 | C2/C3 两条路线都 fail；C3 wrist-inside near-reject 对应 RH floor/collapse 风险更明显 |
| C4: E091 box004 pelvis collapse 是否只是数据筛选失败 | 倾向不是数据筛选单点问题 | C1 D005b pass 但两条 smoke 都 collapse，object/head/upper 安全好，瓶颈集中在 pelvis/upright/contact dynamics |

## 7. 决策

- 不跑 Stage A full：smoke 没有 `PASS/REVIEW+`，且视觉确认 collapse。
- 不跑 Stage B `rl_from_spider`：没有 Stage A full `WORK` 序列，不能构建合格输入。
- 不跑 Stage C main：三条 direct Omni smoke 均 fail，继续 main 只会消耗算力。
- 不扩大 medium-box 数据：当前瓶颈不是更多 case，而是 pelvis/upright/contact/floor 约束。

## 8. 下一步建议

开一个小范围 E093 姿态稳定实验，固定 C1 box004 作为正种子，只改动态优化约束，不改上游数据：

1. 加 `pelvis_z` / upright elite gate 或 hard termination，避免 CEM 接受低髋局部解。
2. 加 hand-floor shortcut penalty / gate，尤其覆盖 C3 类右手贴地。
3. 将 object tracking 与 hand-object support/contact continuity 绑定，避免物体 tracking 好但人体交互无效。
4. 若 C1 修复有效，再回到 C2/C3；否则不扩 Box026 或更多 medium boxes。
