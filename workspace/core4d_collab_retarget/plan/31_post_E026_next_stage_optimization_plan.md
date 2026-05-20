# Post-E026 下阶段优化总计划：E027-E031 robot-side hard constraints

日期：2026-05-21

## Context

E026 已把 OmniRetarget / Holosoma kinematic、E081 baseline full rerun、E018b，以及 E022-E025 优化变体放到统一 13case 账本中。关键结论是：

- `spider_best_E018b_E022_E025` 是当前 best dynamic：P1 13case object `5.33cm`、orientation `4.70deg`、contact `55.87%`、deep penetration `26.37%`、fall `3`、strict `1/13`。
- P0 9case 排除 `desk021_p1`、`box021_p1`、`box021_p2`、`bucket001_p1` 后，best dynamic object `4.97cm`、contact `65.81%`、deep penetration `30.52%`、fall `0`、strict `1/9`。
- E081 full rerun 已补齐 paper metrics，但 object/contact/fall 明显差于 best dynamic；它的 strict 仍是 legacy leg-object proxy，不作为当前优化主线。
- E022-E025 已证明普通 reward sweep 的边际收益很低：mask 修正、lower-body geometry shrink、stability/contact gain、soft penetration penalty 都只产生局部信号，没有把 strict success 打开。

因此下一阶段目标不是继续提高 object tracking 均值。object-side canonical support proxy 已基本成立，后续主线应转向 robot-side 的接触时序、硬表面可行性、下肢稳定/几何和最终 best-selection 复评。

## Scope

### 优先级

| 优先级 | Case | E026 best dynamic 信号 | 下一阶段归类 |
|---|---|---|---|
| P0 guard | `box025_p2` | strict pass；contact `86.93%`；deep pen `0%` | 不优化，只做回归保护 |
| P0 high | `box023_p1` | contact `25.30%`；no fall；deep pen `0%` | E027 contact timing / target phase |
| P0 high | `box023_p2` | contact `28.57%`；no fall；deep pen `3.33%` | E027 contact timing / target phase |
| P0 high | `box025_p1` | contact `66.07%`；deep pen `5.06%`；ref leg/object interference 高 | E027 + E030 near-pass 修复 |
| P0 high | `bucket001_p2` | contact `77.53%`；deep pen `59.60%` | E028 hard no-penetration |
| P0 high | `bucket005_s2_p1` | contact `97.59%`；deep pen `88.15%` | E028 hard no-penetration |
| P0 high | `bucket005_s2_p2` | contact `96.42%`；deep pen `64.53%` | E028 hard no-penetration |
| P0 high | `bucket007_p1` | contact `84.13%`；deep pen `35.57%` | E028 hard no-penetration |
| P0 high | `bucket007_p2` | contact `29.75%`；deep pen `18.42%`；ref interference 高 | E027 + E030 |
| P1 diagnostic | `bucket001_p1` | fall；contact `0%` | E029 stability/control，可选进入 P1 改善 |
| P1 diagnostic | `box021_p1/p2` | fall / 数据问题；暂不进 P0 | 只做数据/稳定性诊断，不作为主成功标准 |
| P1 diagnostic | `desk021_p1` | OmniRetarget SOCP infeasible；SPIDER contact 低 | 只保留 P1 报告，不优先优化 |

### P0/P1 口径

- **P0 主目标**：暂沿用 E026 9case，排除 `desk021_p1`、`box021_p1`、`box021_p2`、`bucket001_p1`。这是下一阶段优化是否有效的主判断，但 E027 会重新做数据/retarget 可用性审计；如果多方面证据表明某些 P0 case 数据质量或前置运动学重定向质量过差，可以从优化分母中降级为 caveat case。
- **P1 完整账本**：继续 13case 全评估，保留 caveat；P1 不用于否定 P0 优化，但必须报告，避免选择性呈现。
- **不再混用指标**：SPIDER-side contact 用 E019/E026 的 5cm contact；OmniRetarget 28cm contact 只作为 kinematic 复现口径，不作为 physical success gate。

### 数据可用性判定原则

CORE4D 的原始动捕质量一般，且当前 pipeline 先经过 OmniRetarget / Holosoma 运动学重定向，再进入 SPIDER-style 物理重定向。失败可能来自两层误差叠加：

1. raw CORE4D motion / object / contact annotation 本身质量差；
2. OmniRetarget 前置运动学重定向无法给出可物理执行的 robot reference；
3. SPIDER 物理控制/约束机制不足。

E027 必须把这三类原因分开。弃用 case 不能只凭单个指标决定；至少需要两类独立证据同时指向数据或前置 retarget 不可靠，并且视频/关键帧审核与指标一致。建议标签：

| 标签 | 含义 | 后续处理 |
|---|---|---|
| `usable_algorithmic_failure` | 数据/kinematic reference 基本可用，失败主要来自当前物理控制 | 留在 P0/P1 优化分母 |
| `usable_with_caveat` | 有数据/retarget 瑕疵，但仍能支持算法对比 | 留在 P1；是否留在 P0 需说明 |
| `retarget_questionable` | OmniRetarget infeasible、高残差、高 jitter、reference 自碰/穿物等证据明显 | 不作为算法失败主证据，先做 retarget/data caveat |
| `raw_data_questionable` | 原始 mocap/object/contact 证据不可信，如动作跳变、物体轨迹异常、contact annotation 与视觉冲突 | 从优化目标中降级；P1 报告保留 |
| `discard_from_success_denominator` | raw 与 retarget 两层均有强证据不可用 | 从 P0 和 strict success 主分母移除，但必须在 appendix/caveat 表中列出 |

弃用协议的最低证据要求：

- 至少 `2/4` 类证据失败：raw motion quality、object/contact annotation、OmniRetarget/kinematic feasibility、SPIDER-independent visual audit。
- 不能把 E027/E028/E029 的算法失败反向解释成数据差；必须有算法运行前或与算法无关的证据。
- 被弃用 case 仍需保留在 P1 full table 中，列明 `discard_reason` 和证据路径。

## Stage Claims

| Claim | 最低证据 |
|---|---|
| C0: 下一阶段必须先区分算法失败、raw 数据问题和前置 kinematic retarget 问题 | E027 对 13case 输出 data/retarget quality audit，给出 `usable_algorithmic_failure` / `usable_with_caveat` / `retarget_questionable` / `raw_data_questionable` / `discard_from_success_denominator` 标签和证据路径 |
| C1: 低接触 case 的主因可以从 timing/target phase、数据/retarget 质量、几何/控制不足中分离出来 | E027 对 `box023_p1/p2`、`bucket007_p2` 等 low-contact case 输出 per-frame timing panel 与 data-quality panel；至少能解释 contact miss frames 的主导来源，并通过小规模 shift/hold variant 产生或排除 `>=15pp` contact 改善 |
| C2: bucket 系列的主要失败不是 object tracking，而是 high-contact penetration shortcut | E028 在 `bucket005_s2_p1/p2`、`bucket007_p1`、`bucket001_p2` 上把 deep penetration 相比 E026 best 降低 `>=25pp`，且 object error 不超过 `8cm` |
| C3: `bucket001_p1` 需要独立 stability/control 机制，不应和 contact/collision sweep 混在一起 | E029 要么使 `bucket001_p1` no-fall 且 pelvis min `>=0.45m`，要么用 reference/posture diagnostics 证明当前 ref/control 不可行 |
| C4: lower-body geometry/control 是 `box025_p1`、`bucket007_p2` 的必要辅助线，但不能用删除 collision pair 伪造通过 | E030 将 ref leg/object interference 降到 `<15%`，并保持 sim leg artifact 不上升；任何 `legpair_off` 类结果只能作为反例 |
| C5: 下一阶段有效性的最终标准是 strict success 增长，而非单项均值好看 | E031 full eval 中 P0 strict 从 `1/9` 提升到 `>=4/9`；stretch 目标 `>=6/9`；P0 object mean 保持 `<=6.5cm`，fall 保持 `0` |

## 总体成功标准

| 指标 | E026 best dynamic | 阶段最低目标 | Stretch 目标 |
|---|---:|---:|---:|
| P0 strict success | `1/9` | `>=4/9` | `>=6/9` |
| P0 object pos mean | `4.97cm` | `<=6.5cm` | `<=5.5cm` |
| P0 contact 5cm mean | `65.81%` | `>=70%` | `>=78%` |
| P0 deep penetration mean | `30.52%` | `<=18%` | `<=12%` |
| P0 falls | `0` | `0` | `0` |
| P1 strict success | `1/13` | `>=4/13` with caveat | `>=6/13` if P1 diagnostic cases improve |

对单 case 的 strict target 使用更保守门槛：

- object pos `<=10cm`，目标 `<=8cm`
- object ori `<=25deg`
- contact 5cm `>=70%`
- robot-object deep penetration `<=20%`，优化目标 `<=15%`
- max penetration `<=5cm`
- no fall；pelvis min `>=0.45m`，目标 `>=0.55m`

## Roadmap

### E027: Contact Timing + Data / Retarget Quality Diagnosis

**目标 case**：

- data / retarget quality audit：13case 全覆盖。
- contact timing diagnosis：优先 `box023_p1`、`box023_p2`、`bucket007_p2`、`box025_p1`；如果 data audit 发现其他 case 的 contact annotation 或 kinematic reference 可疑，也纳入 timing panel。

**动机**：E022 修掉 `box023_p1` mask overclaim 后 contact 只从 `24.50%` 到 `25.30%`；E025 高 contact gain 也只把 `box023_p1` 推到 `28.51%`，`box023_p2` 提升 contact 时反而 fall / penetration 变坏。这说明问题不是简单 mask 或 reward 权重，而可能是 contact target 的相位、手-物相对速度、support proxy 时间基准或 near-field 姿态 gating。

同时，CORE4D 原始动捕、object pose、contact annotation 质量有限；当前方法还依赖 OmniRetarget / Holosoma 的前置运动学重定向。raw 数据误差和 kinematic retarget 误差叠加后，某些 case 可能本身不适合作为物理重定向优化目标。E027 必须把“算法还没做好”和“这个 case 的数据/前置 reference 不可信”分开。

**改动**：

1. 新增离线诊断脚本，逐帧输出：
   - raw/demo contact mask、runtime contact mask、robot 5cm contact
   - EEF 到 object surface/center 的距离、SDF、相对速度
   - object/support proxy pose lag、target frame index、mask active window
   - contact miss frames 的分类：too early、too late、too far、wrong side、orientation mismatch、unstable posture
2. 新增 data / retarget quality audit，逐 case 输出四类证据：
   - raw CORE4D motion：pelvis/root/hand/object pose jitter、速度/加速度尖峰、缺帧、动作跳变、明显视觉不合理帧。
   - object/contact annotation：demo hand-object distance、raw 3cm/5cm contact frames、processed contact overclaim/mismatch、contact window 是否与视频一致。
   - OmniRetarget / kinematic reference：SOCP infeasible、qpos discontinuity、smoothness outlier、28cm contact preservation、kinematic robot-object penetration、ref leg/object interference、EEF 是否可达 demo contact phase。
   - SPIDER-independent visual audit：关键帧/视频是否支持“数据差”判断，而不是只支持“算法失败”。
3. 新增 case-level 判定表：
   - `quality_label`
   - `discard_from_p0`
   - `discard_from_success_denominator`
   - `primary_evidence`
   - `secondary_evidence`
   - `counter_evidence`
   - `decision_rationale`
4. 新增小规模 full variants：
   - contact mask / target phase shift：`-4/-2/+2/+4` frames，先 quick smoke，再只保留有诊断依据的 full
   - contact hold window：在 raw contact 前后扩 `2-4` frames，但不改 mask semantics
   - surface-aware contact target：把 EEF target 投到 object surface 外侧，避免把手目标设到物体内部或错误侧
   - near-field orientation gate：只有 EEF 在近场且 palm normal 合理时加 contact reward

**候选文件**：

| 文件 | 改动 |
|---|---|
| `workspace/core4d_collab_retarget/scripts/E027/generate_e027_overrides.py` | 生成 timing/hold/surface variants |
| `workspace/core4d_collab_retarget/scripts/eval/diagnose_contact_timing.py` | 新增 per-frame timing diagnosis |
| `workspace/core4d_collab_retarget/scripts/eval/audit_case_data_quality.py` | 新增 raw / contact / kinematic / visual 多证据 case audit |
| `workspace/core4d_collab_retarget/scripts/eval/eval_E027.py` | 汇总 contact timing、paper metrics、strict gates |
| `workspace/core4d_collab_retarget/scripts/train/train_E027.sh` | 固化训练/rollout 命令 |
| `workspace/core4d_collab_retarget/scripts/run_E027_remote.sh` | 如果 full variants `>=3`，按 2-GPU 远程队列执行 |
| `spider/config.py` | 如需新增 phase shift / surface target 参数，在 config 显式声明 |
| `spider/simulators/mjwp.py` | 如需 runtime contact target shift / surface projection，在 reward 计算中实现 |

**数据质量证据输出**：

| 产物 | 内容 |
|---|---|
| `results/E027/data_quality/case_quality_audit.csv` | 13case 标签、弃用标记、证据字段 |
| `results/E027/data_quality/case_quality_audit.md` | 面向报告的逐 case 解释 |
| `results/E027/data_quality/per_case/*.csv` | 逐帧 raw/retarget/contact/timing 指标 |
| `results/E027/data_quality/keyframes/` | 支撑弃用或 caveat 的关键帧 |
| `results/E027/timing_panels/*.md` | low-contact case timing 分析面板 |

**成功标准**：

| 指标 | 目标 |
|---|---|
| data audit coverage | 13/13 case 都有 quality label 和证据路径 |
| discard protocol | 任何弃用 case 都满足至少两类独立证据失败，且视觉/关键帧支持 |
| `box023_p1/p2` contact | 至少一个 case 达 `>=70%`，另一个提升 `>=15pp` |
| `bucket007_p2` contact | `29.75% -> >=60%`，目标 `>=70%` |
| `box025_p1` near-pass | contact `66.07% -> >=70%` 且 no regression |
| object/fall/penetration | object `<=8cm`；no fall；deep pen `<=15%` |
| 诊断产物 | 每个 case 有 CSV + Markdown panel；被 caveat/弃用 case 必须有关键帧/视频核对 |

**停止条件**：

- 如果某 case 被 E027 标为 `discard_from_success_denominator`，后续 E028-E031 不再把它作为优化成功分母，只在 P1/caveat 表保留。
- 如果所有 phase/hold variants 的 contact 改善 `<10pp`，且 data/retarget audit 没有支持弃用，则停止 timing sweep，转 E030 geometry/control；不要继续加 contact gain。
- 如果 timing 失败但 data/retarget audit 显示 reference 本身不可达，先回到 OmniRetarget/data 修复或弃用判断，不进入 SPIDER reward sweep。

### E028: Hard No-Penetration / Surface Feasibility

**目标 case**：`bucket005_s2_p1`、`bucket005_s2_p2`、`bucket007_p1`、`bucket001_p2`；`box025_p2` 做 guard。

**动机**：E025 soft penalty 有信号但不够硬。`bucket007_p1` deep pen 从 `51.01%` 降到 `35.57%`，但仍远高于 `<15%`；`bucket005_s2_p1` deep pen 甚至保持在 `88.15-92.89%`。当前 CEM 仍会选择 high contact + inside-object 的低成本轨迹。

**改动**：

1. 将 E025 的 soft hinge penalty 升级为 hard feasibility：
   - SDF barrier 从 surface 外侧 margin 开始惩罚，而不是只惩罚 deep threshold 后的内部点。
   - inside-object samples 直接 rejection 或强降权，避免 contact reward 抵消 penetration。
   - 使用 exponential / squared barrier 对 `SDF < margin` 的路径给非线性惩罚。
2. Surface target：
   - contact target 投影到 object surface 外法线方向。
   - EEF target 保持非负 SDF margin，例如 `1-3cm`。
   - 对 bucket 开口/侧壁需要区分可接触面与不可穿透体积。
3. 分阶段 objective：
   - Stage A：先 object + posture + no-penetration。
   - Stage B：再打开 contact preservation。
   - 避免初始阶段为了追 contact 直接钻入物体。

**候选文件**：

| 文件 | 改动 |
|---|---|
| `spider/config.py` | 新增 `robot_object_barrier_*`、`cem_reject_penetrating_samples`、`surface_contact_target_*` |
| `spider/simulators/mjwp.py` | 实现 barrier / rejection / surface projection / staged objective |
| `workspace/core4d_collab_retarget/scripts/E028/generate_e028_overrides.py` | 生成 hard-barrier variants |
| `workspace/core4d_collab_retarget/scripts/eval/eval_E028.py` | 汇总 deep pen、max pen、contact、object/fall guard |
| `workspace/core4d_collab_retarget/scripts/train/train_E028.sh` | 固化训练命令 |
| `workspace/core4d_collab_retarget/scripts/run_E028_remote.sh` | 远程并行运行 |

**建议 variants**：

| Variant | 含义 |
|---|---|
| `barrier_surf_m02` | surface margin `2cm`，非线性 barrier，无 rejection |
| `reject_deep_m01` | SDF `< -1cm` 直接 rejection / score cap |
| `projection_surf_m02` | contact target 投影到 surface 外 `2cm` |
| `staged_barrier_contact` | 先 no-penetration/posture，再逐步打开 contact reward |

**成功标准**：

| 指标 | 目标 |
|---|---|
| deep penetration | 每个目标 case `<=15%`；至少 `2/4` case 达 `<=10%` |
| max penetration | `<=5cm`，目标 `<=3cm` |
| contact | 保持 `>=70%`；若 barrier 降 contact，必须解释为真实接触不可行或 target 需修 |
| object | object pos `<=8cm`，不允许为了 no-penetration 丢掉 transport |
| strict | 目标 `>=2/4` strict pass |

**停止条件**：如果 hard barrier 使 object error `>10cm` 且 contact 同时下降 `>20pp`，暂停 scale sweep，先做 surface target/可接触面建模，不继续加 barrier scale。

### E029: Bucket001 Stability / Control

**目标 case**：`bucket001_p1` 主，`bucket001_p2` guard；`box021_p1/p2` 只做 P1 diagnostic。

**动机**：E024 证明 `bucket001_p2` 可以站稳，但仍 deep penetration；`bucket001_p1` 三个 variants 全 fall，contact `0%`。p1 不像普通 contact/stability gain 可解，需要控制机制和 reference/posture 诊断。

**改动**：

1. Upright/root terminal：
   - 对 pelvis height、root tilt、root angular velocity、terminal upright 加硬 gate 或强惩罚。
   - fall 早期直接降低 CEM sample score，避免用摔倒姿态换 object tracking。
2. Foot support / lower-body control：
   - 加 foot-ground support proxy 或脚底接触稳定性指标。
   - 增强 lower-body tracking regularizer，但不牺牲 hand/object 接触。
3. Staged contact：
   - 在 posture invalid 时弱化 contact reward。
   - posture valid 后再打开 contact/transport reward。
4. Reference feasibility audit：
   - 输出 ref pelvis height、root tilt、foot support、hand-object distance，同 E018b/E024 rollout 对齐。

**候选文件**：

| 文件 | 改动 |
|---|---|
| `spider/config.py` | 新增 upright/root terminal、posture-valid contact gate 参数 |
| `spider/simulators/mjwp.py` | 实现 stability hard score / staged contact |
| `workspace/core4d_collab_retarget/scripts/E029/generate_e029_overrides.py` | 生成 stability/control variants |
| `workspace/core4d_collab_retarget/scripts/eval/eval_E029.py` | 汇总 pelvis/upright/foot/contact/deep pen |
| `workspace/core4d_collab_retarget/scripts/train/train_E029.sh` | 固化训练命令 |

**成功标准**：

| 指标 | `bucket001_p1` 目标 | `bucket001_p2` guard |
|---|---:|---:|
| fall | false | false |
| pelvis min | `>=0.45m`，目标 `>=0.55m` | `>=0.55m` |
| contact 5cm | `>=50%`，目标 `>=70%` | `>=70%` |
| deep penetration | `<=20%`，目标 `<=15%` | 不回退，目标跟 E028 合并到 `<=15%` |
| object pos | `<=10cm` | `<=8cm` |

**停止条件**：如果 p1 在 3 个机制 variants 后 pelvis min 仍 `<0.25m` 或 contact 仍 `0%`，将 p1 标为 reference/control infeasible candidate，进入数据/参考修复而不是继续 reward sweep。

### E030: Lower-Body Geometry / Control Repair

**目标 case**：`box025_p1`、`bucket007_p2`；`bucket005_s2_p1` 作为 leg shortcut guard。

**动机**：E023 `lowerbody_proxy_min` 将 ref interference 明显降低，但仍未达 `<15%`；`legpair_off` 会制造 artifact shortcut，不是可接受修复。E030 应同时处理真实 geometry、collision pair 和 lower-body control，而不是只 shrink 或只删碰撞。

**改动**：

1. Geometry refresh：
   - 复查 lower-body/foot geoms 的 capsule/box 尺寸和局部坐标。
   - 对明显过粗 geoms 做 case-independent shrink 或替换，而不是只对目标 case hack。
   - 活跃 scene XML 必须 snapshot，必要时 `git add -f` 纳入版本。
2. Collision semantics：
   - 保留必要 collision pair，不用 `legpair_off` 作为修复。
   - 对不可避免的 ref-side overlap，记录为 ref infeasible，而不是让 sim 穿透。
3. Control regularization：
   - lower-body tracking / leg-object min-distance barrier 与 E028 surface feasibility 合并。
   - 防止 legs/feet 参与推物体 shortcut。

**候选文件**：

| 文件 | 改动 |
|---|---|
| `workspace/core4d_collab_retarget/scripts/E030/*` | 生成 geometry/control variants 与 snapshots |
| `workspace/core4d_collab_retarget/scripts/eval/eval_E030.py` | 复用 E023 S2 + E026 paper metrics，输出 ref/sim interference |
| `workspace/core4d_collab_retarget/scripts/train/train_E030.sh` | 固化训练命令 |
| `example_datasets/processed/.../*.xml` | 如确需改 scene XML，必须 snapshot 并强制纳入 git |
| `spider/simulators/mjwp.py` | 如需 lower-body min-distance barrier，复用 E028 barrier path |

**成功标准**：

| 指标 | 目标 |
|---|---|
| ref leg/object interference | `box025_p1`、`bucket007_p2` 均 `<15%` |
| sim leg artifact | 不高于 E026 best，目标 `<10%` |
| contact | `box025_p1 >=70%`；`bucket007_p2 >=60%`，目标 `>=70%` |
| object/fall | object `<=8cm`；no fall |
| penetration | robot-object deep pen `<=15%`，max pen `<=5cm` |

**停止条件**：如果 geometry 改动降低 ref interference 但 sim penetration/contact 无改善，E030 只作为 data/geometry caveat，不继续做 case-specific XML 微调。

### E031: Best Dynamic Assembly + Full Evaluation

**目标**：把 E027-E030 的有效 variants 纳入 E026-style full eval，形成下一版 paper-facing 账本。

**改动**：

1. 复用 `eval_E026_full_eval.py` 的 schema，扩展 candidate pool：
   - E018b baseline
   - E022-E025 validated variants
   - E027-E030 新 variants
2. Best selection 规则保持 deterministic：
   - strict success 第一优先
   - object/transport/no-fall gate
   - contact 高、deep penetration 低
   - smoothness 作为 regression guard，不压过 physical strict
3. 输出：
   - `summary_9case.md`
   - `summary_13case.md`
   - `method_case_metrics.csv`
   - `best_dynamic_selection.csv`
   - `visual_metric_audit.md`
   - 必要时生成 xlsx 供报告使用

**候选文件**：

| 文件 | 改动 |
|---|---|
| `workspace/core4d_collab_retarget/scripts/eval/eval_E031_full_eval.py` | E026 full eval 的后续版本 |
| `workspace/core4d_collab_retarget/results/E031_full_eval/` | 全量评估输出 |
| `workspace/core4d_collab_retarget/log/31_E031_full_eval_results.md` | 记录 P0/P1 和视觉核对 |

**成功标准**：

| 指标 | 目标 |
|---|---|
| P0 strict | `>=4/9`，stretch `>=6/9` |
| P0 object | `<=6.5cm` |
| P0 deep pen | mean `<=18%`，没有单 case `>50%` |
| P0 fall | `0` |
| P1 report | 13case 完整输出，并明确 data/problem cases caveat |
| visual audit | 不允许出现“指标过但视频明显不过”的反例 |

## 统一执行规范

| 项 | 规则 |
|---|---|
| Plan first | 每个子实验 E027-E031 仍需单独 plan 文件，不能直接改代码开跑 |
| Scripts | 每个实验必须有 `scripts/train/train_E0NN.sh`、`scripts/eval/eval_E0NN.py`；独立 variants `>=3` 时必须有 remote wrapper |
| Results | 训练/rollout 输出到 `workspace/core4d_collab_retarget/results/E0NN/`；最终 full eval 输出到 `results/E031_full_eval/` |
| Scene XML | 改 XML 必须 snapshot；活跃 case XML 必须按现有规则强制纳入 git |
| Metrics | 使用 `paper_metrics_version=2026-05-21-P3` 或后续显式版本；contact 使用 5cm physical contact |
| Visual | 每个目标 case 至少保留 online MP4；失败或边界 case 用 video-frames 抽帧写入 log |
| Remote | remote tmux 默认带 `RUN_TIMEOUT_SECONDS` / `RUN_STALL_TIMEOUT_SECONDS`；pull 前确认不是 smoke `T=2` |
| Regression guards | `box025_p2` 必须保持 strict pass；P0 fall 必须保持 `0`；object mean 不允许为了 contact/penetration 明显退化 |

## 执行顺序

1. **E027 first**：先做 13case data / retarget quality audit，再做 timing diagnosis；它决定 low-contact case 是继续优化、降级为 caveat，还是从成功分母中弃用。
2. **E028 parallel after E027 setup**：hard no-penetration 与 timing 基本独立，可在 E027 诊断完成后并行推进 bucket cases。
3. **E029 after E028 first signal**：`bucket001_p2` 可能被 E028 修；`bucket001_p1` 单独进入 stability/control，不和 bucket penetration 混跑。
4. **E030 after E027/E028 evidence**：只有在 timing/barrier 不能解释 `box025_p1`、`bucket007_p2` 时，才做 geometry/control 修复，避免过早 XML 微调。
5. **E031 last**：所有子实验按 log 验证后再组装，不把未验证 smoke/partial results 混入 best selection。

## 第一批落地任务

| 顺序 | 任务 | 产物 |
|---|---|---|
| 1 | 写 E027 详细 plan | `plan/32_E027_contact_timing_data_quality_plan.md` |
| 2 | 实现 E027 13case data / retarget quality audit，不跑 full 先判定 case 可用性 | `results/E027/data_quality/case_quality_audit.md/csv` |
| 3 | 实现 E027 离线 timing panel，优先 low-contact 且未弃用 case | `results/E027/timing_panels/*.md/csv` |
| 4 | 根据 E027 audit + timing panel 选择不超过 6 个 full variants | `scripts/E027/variants.tsv` |
| 5 | 同步写 E028 详细 plan，但等 E027 panel 出来后再定 final variants | `plan/33_E028_hard_penetration_feasibility_plan.md` |
| 6 | E027/E028 跑完后再决定 E029/E030 的 exact scope | 对应 plan 文件 |

## 风险与应对

| 风险 | 应对 |
|---|---|
| Hard barrier 让 contact/object tracking 崩掉 | 用 staged objective 和 surface projection 替代继续加 scale |
| Timing shift 只提升单 case，泛化差 | 只把它作为 case-specific evidence，不写成通用方法；优先找相位/速度共性 |
| 过度弃用困难 case，导致结果看起来变好但证据不扎实 | 弃用必须满足多证据协议；P1 full table 保留所有 case；报告同时给 raw/retarget evidence 和 counter-evidence |
| XML geometry 改动不可复现 | scene snapshot + tracked XML + log 中记录 diff 和 active cases |
| Best selection 过拟合 | P0/P1 同时报告；视觉 audit 必须逐 case 对齐 |
| 远程不稳定 | 远程只跑非阻塞队列，本地保留关键 variant；所有 wrapper 带 stall guard |

## 最终判断

E026 后的优化重点应从“能不能把 object 带到位”转向“数据/reference 是否可信，以及机器人是否以物理可信的方式接触并稳定完成动作”。下一阶段只把能同时改善 strict contact、penetration、fall 且不牺牲 object tracking 的机制纳入 best dynamic；对多证据确认的数据差 case，应明确弃用或降级为 caveat，而不是把不可用数据强行算作算法失败。普通 contact gain、mask dilation、soft penalty scale 或删除 collision pair 不再作为主线。
