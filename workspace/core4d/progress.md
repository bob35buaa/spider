# CORE4D 当前进度

> 完整备份：
> [E167–E179 初始恢复](progress_archive/E167_E179_20260724_full_backup.md) ·
> [E179 执行过程](progress_archive/E179_20260724_execution_full_backup.md) ·
> [E179 最终收尾](progress_archive/E179_20260725_closeout_full_backup.md)
>
> 本文件只保留最近完成工作的可靠结论与下一会话入口。

## 2026-07-26：E180 RL38 指标线性可分性审计（完成）

- 计划：`plan/198_E180_rl38_metric_linear_separability_audit_plan.md`。
- Authority：38 条 `32 success / 6 fail`；Box021/004/001/024/023 为
  `11/4/13/3/7`。两面板为 reference-input `38×67` 与同口径 metrics
  `36×111`；另有 42 条未进 RL proxy negative。
- 单指标无完美分离。训练内 sparse SVM 可完美分开 RL-only：
  reference 3 指标、standardized 2 指标，但不能当作泛化 gate。
- reference SVM：LOOCV failure recall=`4/6`，按对象留出=`5/6`，但误拦
  success=`10/32`；standardized SVM：LOOCV=`5/6`，按对象仅 Box024
  `028_p2` 命中（`1/6`）。
- 冻结 standardized RL separator 只拒绝 proxy `4/42=9.5%`；将 proxy
  合并拟合时，5 指标模型仍接受 `6/48` failure-like，故联合 hard gate
  当前不成立。
- 报告：`analysis/RL38_metric_linear_separability_audit_20260726.md`；
  日志：`log/243_E180_rl38_metric_linear_separability_results.md`；artifact：
  `results/E180/rl_metric_separability/`。
- 4 项直接单测、py_compile、链接、标题与 diff-check 通过；pytest 环境缺失。

## 2026-07-26：E168–E178 可用率与下游 RL 诊断（含 Box023）

### 最终状态

- 任务：`exp_analysis_0726.md`。
- 报告：
  `analysis/E168_E178_availability_and_RL_diagnosis_20260726.md`。
- 视觉证据：
  `analysis/assets/exp_analysis_0726/video_frames/`。
- 本轮只分析存量数据、人工标签与 rollout，未启动新训练；不更新 Tracker。
- Box023 新增 S6 后验：7 条 `RL_EXPORT_READY` 中 `040_p2`、`042_p2`
  训练失败，其余 5 条成功；两条失败均为人工 `MINOR_ACCEPTABLE`，
  且 E173 六门、十二门均通过。联合集合为 `32/38` RL 成功。

### 核心结论

- 低 yield 是 raw/action/contact、retarget 可解性、template/proxy、
  SPIDER 联合可行域和 gate/RL 接口的分层损失，不是单一算法故障。
- E170 PRG 将 Box021 人工可用从 `13/28→18/28`，但 10 条未恢复 case
  leg penetration `10/10` 改善、3mm hand contact `10/10` 下降。
- E170–E178 candidate gate-health 连续为 `0/N`；combined-valid 约束和
  `least_violation` fallback contract 是最高置信的 SPIDER 共性瓶颈。
- E174 非 box `5/39=13%`，修正 contact-aligned proxy 后 E178 physics
  `16/27=59%`、人工 `12/23=52%`；旧 proxy 是主要可修复损失源。
- 用户给出的下游集合为 `32/38` RL 成功。36 条可直接对齐六门：
  PASS 中 `20 success / 5 fail`，FAIL 中 `10 success / 1 fail`；
  旧六门既非 RL 成功充分条件，也非必要条件。
- Box004 `082_p1/p2` 和 Box024 `028_p2` 是六门漏检。`082` 对象轨迹的
  `94.236rad/s` 角速度峰值在 pre-Omni converted NPZ 已存在；
  `028` 的 pre-Omni 对象线速度也显著高于成功对照。
- converted→retargeted 的对象 quaternion 逐帧一致，故上述对象异常不能
  归因于 Omni solver；raw root 未挂载，仍需区分 raw 标注与 converter。
- Box023 两个失败从前段直立交互转为约中段多环境同步倒地；其 pre-Omni
  对象动态不能与成功对照分离，说明 `082/028` 式病态只解释失败子类，
  还需 phase-stratified 闭环 probe。
- E109 已确认历史 tracking 多为 self-ref/method-ref；没有 raw-GT 统一表前，
  不能宣称 OmniRetarget 整体质量已被证明差。

### 后续入口

1. P0：全库 pre-Omni 对象 SE(3) 连续性审计，先定位 raw/converter 边界。
2. P0：修复 combined-valid 选择；空交集必须定向重采样或拒绝。
3. P0：对失败与成功对照做早/中/末 phase PD、`partner_off/on` 短 probe。
4. P1：补 raw-GT OmniRetarget benchmark、full-3D/support/terminal gates，
   并按 object/date/sequence held-out 校准两级发布策略。

## E179：box023 / E167A no-PRG / Full CEM

### 最终状态与结果

- 同一 16 条 paired denominator；Full budget=`seed 0, 1024×32`；
  本地 4 条、A100 12 条，完成 `16/16`，terminal failure=`0`。
- E167A profile parity、no-PRG audit、视频与 completion audit 全部通过。

| 口径 | E173 PRG | E179 no-PRG |
|---|---:|---:|
| Physics 六门 | `13/16` | `9/16` |
| Physics + tracking 十二门 | `7/16` | `4/16` |

- 十二门迁移=`P→P 3 / P→F 4 / F→P 1 / F→F 8`，McNemar
  exact `p=0.375`。
- lower-body=`14→10/16`；leg penetration paired mean
  `Δ=+0.07363`，95% CI=`[+0.03704,+0.11427]`。
- 16 条 paired 视频复核支持 no-PRG 下肢退化；最终裁决
  `PRG_BETTER`，Box023 保留 E170 PRG。

### Canonical evidence

- 评测：`results/E179/s6_downstream/eval/full/E179_vs_E173_report.md`。
- Summary：`results/E179/s6_downstream/eval/full/e179_eval_summary.json`。

## 2026-07-30：E178 非凸物体 MuJoCo 碰撞表示调研（完成）

- 正式报告：
  `log/244_E178_nonconvex_mujoco_collision_review.md`。
- 结论分层：非凸刚体的引擎表示已经解决；自动获得“少组件、高保真、保任务
  空腔、1024-world 高吞吐”的 collider 尚未完全解决。
- MuJoCo 官方推荐 compound convex，并点名 CoACD。CoACD 是当前最佳开源
  production baseline；V-HACD 已 EOL。Navigation-driven ACD、RL-ACD 和
  CPD 是更前沿的任务/性能方向，但暂不能作为可复现 MuJoCo production 依赖。
- 普通 mesh geom 的碰撞仍是 convex hull。rigid flex 和 mesh-backed SDF
  能表达真实非凸面；E178 锁定的 MJWarp 3.7 rigid-flex 无 broadphase，且
  不支持与 rubber-hand mesh 的 geom-flex narrow phase，不宜进 Full。
  MJWarp v3.11 已增加 flex broadphase，但 flex 仍标为 experimental。
- 本地最小检查确认 `geom type="sdf" mesh="..."` 在 MuJoCo CPU 和 MJWarp
  3.7 均可编译；SDF 因 Halton 多起点与迭代成本，建议只作低批量 oracle。
- SPIDER 上游虽有 CoACD 入口，但 `coacd` 未进入项目依赖，入口未接 Core4D；
  当前 PRG/penalty/gate 又全部绑定 box-union SDF，不能只替换 XML。
- E178 推荐路线：先用 CoACD box mode 做兼容 bridge；目标架构为
  CoACD convex-union physics + 同源 GPU grid-SDF，并加入 cavity/
  navigability/contact-aware gates；原 mesh SDF 作为 fidelity oracle。
- 不把 collider 当作唯一瓶颈：E178 六门 `16/27`、十二门 `10/27`、人工
  `12/23 USE`，combined-valid `0/N` 与 tracking 仍需在 paired ablation 中
  同时报告。

## 2026-07-31：E178 CoACD / canonical SDF 路线审查

- 主路线可行：CoACD 生成 production collision set `C`，P 使用同一组 convex
  hull geoms；从 `C` 烘焙 object-local grid-SDF `D_C` 供 R/G 使用；原始
  mesh SDF `D_M` 只用于 fidelity 与 task-space 审计。
- `D_M→C` 不能只检查全局 Chamfer/Hausdorff；bucket 必须增加 cavity
  must-stay-free、inner/outer/rim must-cover、raw-contact/swept-volume 与
  contact-normal 等 task-aware gates，否则仍会放过“全局误差小但填腔”的
  collider。
- 当前实现缺口明确：`config.py` 的 union mode fail-closed 拒绝 non-box，
  `mjwp.py` 的 reward/penalty/CEM gate 全部绑定 box-union SDF；已有
  `preprocess/decompose.py` CoACD 原型参数固定、未接 Core4D，且 `coacd`
  尚未进入项目依赖。
- 推荐按 `oracle/asset gate → grid-SDF backend contract → convex physics
  canary → paired CEM` 分阶段推进；在 `D_C` 与 MuJoCo hull union 的
  occupancy/zero-surface/坐标系一致性验证通过前，不替换 E178 production。
- `D_M` 前置条件尚需实测：原始 mesh 必须有可靠的 scale、winding、闭合实体
  或显式 generalized-winding/sign 策略；系统 Python 的首轮 `trimesh`
  topology probe 因 `ModuleNotFoundError` 未启动，不能先验假定三个 bucket
  mesh 都是 watertight。
- 改用项目 `uv --frozen` 环境后拓扑 probe 通过：bucket003/004/007 主 mesh
  均为 watertight、winding-consistent，scene scale=`1 1 1`；但 bucket004
  含一个 10-face 零体积组件，bucket007 含 2-face/4-face 零体积组件。
  `D_M` 应来自带 manifest 的最小清洗 oracle mesh，而非未经审计的原 OBJ。

## 2026-07-31：E181 CoACD canonical geometry（计划完成）

- 用户批准以 `cleaned M* → D_M oracle → CoACD C → canonical D_C → P/R/G`
  为主路线；详细计划已写入
  `plan/199_E181_coacd_canonical_geometry_plan.md`。
- `C0–C10` Claims 与 `S0–S6` stop/go 阶段均已冻结；Gate 0–E 未通过前
  不允许启动 Full 27。
- 本次只完成规划文档：未修改实现代码、未构建 collision assets、未启动
  CEM/RL；分阶段 stop/go gates、脚本入口、scene snapshot、可视化和结果
  路径已在计划内冻结。
- 历史基线要求计划拆开四类判据：`C↔D_M` fidelity、`D_C↔C` consistency、
  convex physics/contact 与 `64×4/1024×32` throughput、paired CEM 的
  candidate health/downstream。E175/E176 的高 geom 慢速和
  E170–E178 `combined-valid=0/N` 均不得被合并成一个总分掩盖。
- 编号冻结为 `E181` / `plan/199_E181_coacd_canonical_geometry_plan.md`；
  E178 的 27 条只作为最终 paired authority，首轮必须先用三 bucket
  representative probes 通过离线资产、`D_C↔C`、接触和 `64×4` 吞吐门。
- 计划将 object `mass/inertia/friction/solref/condim` 设为 collider 外的冻结
  contract，避免 compound geoms 改变动力学后把收益误归因于几何。
- 吞吐采用每条 dev3 在其分配设备上同轮交错 A/B；canary 覆盖本地单卡、
  Ada GPU 0 与 Ada GPU 1。保留逐 case `≤3.0s` strict gate，同时报告同设备
  E181/E178 ratio；任何 `3.0s` 以上运行只能走既有显式 waiver，不能改写
  strict 结果。
- CoACD selection 采用 hard gates 后按最小 hull 数词典序选优，不用加权
  总分；当前 E178 canary 三条冻结为 dev3，其余 24 条 contact fidelity 为
  held-out，最终 Full 仍报告 27 与 heldout24 两种口径。
- E181 v1 只接入 canonical `solid_sdf` 并保持既有 R/G 公式、权重和阈值；
  inner/outer/rim 仅作离线审计分层，避免把 collider、backend 与 semantic
  reward 三个变量混成一次实验。
- 用户补充 Full 执行合同：正式 Full CEM 必须复用 E178 的同一 27 条 case，
  方便逐 case paired 对比；计算资源固定为本地单卡与远程 Ada 6000 两卡，
  共三卡并行。远程 profile 已确认是 `spider-remote` 的
  `2× NVIDIA RTX 6000 Ada 48GB`。E178 原始 `ordinal` 存在跳号，不能直接
  取模；计划现已按 canonical manifest 物理行序生成连续
  `authority_row_index=1..27`，再轮转得到确定性 `9/9/9` 分片。计划已同步
  补齐三 worker 的 GPU 归属、环境 SHA parity、远程回收与 27-row merge
  audit；正式 Full 只能由三卡 orchestrator 从冻结 allocation 启动。
- 更新后校验通过：E178 source manifest SHA=`de9a3d...f022a8`，rows/case
  uniqueness=`27/27`，物理行序分片=`9/9/9`；计划中无 A100/旧 E181
  launcher 残留，Markdown 标题、链接、空白和 `git diff --check` 均 PASS。
