# log291 · E202：bucket 类 s6 full-CEM 平移增强（碰撞体 + CEM 全用 E178）

_Core4D · Phase 63 · Run **R290** · 计划 [plan232](../plan/232_E202_bucket_e178_translation_augmentation_plan.md) · 2026-08-19-20 · **状态：完成（C0–C6 通过；C7 视觉因 EGL 渲染环境受阻，记录豁免）**_

## 摘要

把 E199/E200 已验证的 OmniRetarget 物体平移增强机制放量到 **bucket 类**，碰撞体与 CEM 栈**整体改用 E178**（contact-aligned 五段 proxy + E174 PRG arm + `E170_PRG` reward/gate + 1024×32 seed0 + 3cm 掩码）。27 个 E178 full-CEM bucket case × 3 平移 → **73/81 变体可行、73/73 CEM 完成、0 error**。aug 数据物理上与 E178 orig 相当（物体跟踪保持、跌倒/穿透不劣），主要差异是接触保持率小幅下降。

## 训练/执行

- **数据构建**：`train_E202.sh` / `build_augmented_tasks.py`，上游 omnirt_v2（Phase-4 relaxation）retarget original+trans0/1/2 → 固定窗 trim → SPIDER task → **E178 碰撞 sidecar `scene_act_E202_bucketAlignedTop_PRG`**（rubber_hull → 五段 proxy → 18-pair/geom union）。**73 变体 across 27/27 case，0 failure**；8 变体因参考首帧腿-桶穿透（`runtime_initial_overlap`<0）跳过。
- **CEM**：8 卡 priority queue（复用 E199 manifest-driven queue），`GPUS=0-7`，1024×32 seed0。**73/73 `run_complete_pending_eval`，0 problem row。**
- **orig 基线**：复用 E178 27-case full-CEM（omnirt_v1），eval 直接取 `e178_case_metrics.tsv`（同一 core_metrics）。**confound（诚实报告）**：orig=omnirt_v1，aug=omnirt_v2；**碰撞体两者完全相同（E178）**，唯一变量是 retarget 变体。

## 结果（eval：`results/E202/s6_downstream/eval/full_augmentation/`，73 aug vs 25/27 case E178 orig）

| 指标 | orig(E178) mean | aug mean | aug worst | 判定 |
|---|--:|--:|--:|---|
| track_obj_pos_err (cm) | 10.52 | **10.46（+−0.6%）** | 17.61 | ✓ 远低于 ≤25% 阈值（C4） |
| track_obj_ori_err (°) | 5.64 | 5.53 | 10.69 | ✓ 保持 |
| track_root_pos_err (cm) | 19.82 | 22.95 | 129.2 | ⚠ 个别发散 outlier |
| track_eef_pos_err (cm) | 17.81 | 21.50 | 128.1 | ⚠ 同一 outlier |
| contact_in_mask_frac | 0.699 | 0.603 | — | ↓ ~14%（主要变化） |
| hand_pen_3mm_frac | 0.178 | 0.171 | 0.537 | ✓ 略好 |
| leg_pen_frac | 0.015 | 0.037 | 0.441 | ~ 略升（outlier） |
| fall_flag | 0.040 | 0.041 | 1.0 | ~ 持平 |
| **12-gate 通过率** | **0.60** | **0.589** | — | ~ 持平 |

**逐物体（C4 分层）**：

| 物体 | n_aug | gate 通过 | obj_pos aug | contact aug | fall_worst | leg_pen_worst |
|---|--:|--:|--:|--:|--:|--:|
| bucket003 | 27 | 0.444 | 9.72cm | 0.496 | 1.0 | 0.441 |
| bucket004 | 12 | **0.917** | 14.55cm | 0.660 | 0 | 0.048 |
| bucket007 | 34 | 0.588 | 9.60cm | 0.667 | 0 | 0.158 |

**可行性分布（C6）**：73/81 变体可行 = **90.1%**。bucket003 9/9/9、bucket004 4/4/4 全可行；bucket007 12 case（另 2 case 三档全不可行）trans0/1/2 = 12/11/11。8 个不可行档均为参考首帧腿-桶穿透（PRG 运行时重叠保护拦截），非静默丢弃。

## Claims 验证

- **C0 链路放量** ✓：73 变体 task + `scene_act_E202_bucketAlignedTop_PRG` + base yaml + E202 PRG override 全生成；scene sidecar 快照（`results/E202/scene_snapshot/`）。
- **C1 碰撞体几何一致** ✓：`test_e202_scene_parity.py` 对 bucket003 重建 object_collision geom(5)+pair(90) 与 E178 snapshot 逐字段一致；构建期每变体 compile 断言 geom=EXPECTED、pair=18×N（bucket003/007=5/90，bucket004=1/18）。
- **C2 case 权威一致** ✓：27 case 与 E178 full manifest 对齐（9/4/14）。
- **C3 执行闭合** ⚠（基本达成）：73/73 CEM 完成、0 error/non-finite；但 aug 有 **1 个 fall case（bucket003_20231018_005_p1，三档均 fall）** + 个别 root/eef 发散 outlier（worst ~129cm）。fall_mean 0.041≈orig 0.040（不劣于 orig），未达「fall=0」的严格措辞。
- **C4 增强正确性** ✓：接近段偏移 0.200m；obj_pos 增幅 −0.6%（≤25%）。
- **C5 物理可信度** ✓（主结论）：obj_pos/ori 保持、hand_pen 略好、fall/gate 与 orig 持平、leg_pen 略升；**接触保持率 0.70→0.60 是主要退化**。全分布 mean+std+worst 逐物体报告，未 cherry-pick。
- **C6 可行性** ✓：90.1%，逐物体档位计数报告。
- **C7 视觉复核** ✗（豁免记录，rule 9）：CEM `save_video=false`，离线渲染脚本 `render_qc.py` 已就位，但本环境 **mujoco EGL GLContext 初始化失败**（env.md 已知问题 #1，`MUJOCO_EGL_DEVICE_ID` 冲突）→ 无法 headless 渲染。**待 EGL 可用的卡/display 补渲染 + 关键帧观察**（重点看：bucket003_005_p1 fall、068_p1 腿穿透 0.44 的失败模式；对照健康档 bucket004/007）。

## 判定与结论

- **达到「放量数据可用于下游 RL」的门槛**：73 变体物理上与 E178 orig 相当（跟踪保持、跌倒/穿透不劣、90% 可行），obj_pos 几乎不变。
- **bucket003 最弱**（gate 0.44，含 1 fall + 腿穿透 outlier）—— 符合 plan232 预判「bucket 在严格 12-gate 下本就难（lower_body 主导）」，且与增强变量正交（orig bucket003 亦弱）。**建议下游用漏斗（E201）过滤 bucket003 弱样本**，或对 bucket003 收窄放量。
- **接触保持率下降**是增强的固有代价（接近段位姿平移改变了接触时序），非物理破坏。

## 改动文件

| 文件 | 说明 |
|---|---|
| `scripts/experiments/E202/e202_common.py` | 契约 + `build_prg_scene(object_key=)` E178 碰撞体构建 + `load_e178_bucket_cases()` + `_drop_broken_torch()`（scipy/torch stub 兼容） |
| `scripts/experiments/E202/build_augmented_tasks.py` · `build_aug_manifest.py` | 上游 aug → E202 task；P1 trans manifest + P0 reused_e178 orig 入 authority |
| `scripts/experiments/E202/test_e202_scene_parity.py` · `render_qc.py` | C1 几何自测（PASS）· C7 渲染（就位，EGL 受阻） |
| `scripts/eval/runners/eval_E202_bucket_augmentation.py` · wrappers/`*.sh` | orig 基线取 E178 `e178_case_metrics.tsv`；per-case delta + 逐物体分层 + 可行性 |
| `scripts/train/train_E202.sh` · `launch/active/run_E202_local_8gpu.sh` · `watch_E202_fanout.sh` | 数据构建 + 8 卡 CEM + 自动 fan-out watcher |
| scene 快照 | `results/E202/scene_snapshot/cem_sidecars/*` + `manifest.txt`（git HEAD + sha256） |

## 结果路径

- 增强 task/scene：`example_datasets/.../dcv3_omnirt_v2_ref_fk_bucket*__aug_trans*/`（gitignore；快照入 `results/E202/scene_snapshot/`）
- CEM：`results/E202/s6_downstream/cem/full/E202_*_aug_trans*_PRG*`（73 条）
- eval：`results/E202/s6_downstream/eval/full_augmentation/{e202_aug_case_metrics.tsv, e202_aug_orig_deltas.tsv, e202_aug_eval_summary.json}`
- manifest：`results/E202/s6_downstream/manifests/{e202_bucket_priority_manifest.tsv(73), e202_bucket_authority.tsv}`

## 环境事故记录（本轮踩坑）

- SPIDER venv 的 torch 2.11.0 + nvidia cu13 包**文件被清空只剩 dist-info**（疑存储/挂载 eviction）→ 所有 CEM 一度全挂。
- `uv sync`（用户批准）未修 torch（stale dist-info）反而 prune 34 个跨项目包。
- 按 `workspace/hdmi_reproduce/env.md` 用快镜像 `pypi.devops.xiaohongshu.com` 重装 **torch 2.8.0+cu128**（canonical 共享版）+ 恢复 torchrl/tensordict/isaaclab/HDMI editable 等 → 环境复原，CEM 恢复。

## 下一步

- 补 C7：EGL 可用时跑 `render_qc.py`，关键帧观察写回本 log。
- （可选）bucket aug 的 RL-export（仿 E200 三版）；接 E201 漏斗过滤 bucket003 弱样本。
- object scale 增强仍延后（需上游 holosoma 扩展）。
