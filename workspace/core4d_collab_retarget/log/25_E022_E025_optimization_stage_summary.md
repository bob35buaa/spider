# E022-E025 阶段总结：post-E020 robot-side optimization

日期：2026-05-20

## 背景与范围

E020 对 E018b 的 13-case canonical support proxy 泛化做了 failure attribution。用户随后明确：

- `desk021_p1` 暂不优化。
- `box021_p1/p2` 先视为数据问题暂不优化。
- E021 已被 Holosoma RL export plan 占用，post-E020 优化实验从 E022 开始编号。

因此 E022-E025 的目标不是重做 anchor/object-side support proxy，而是按剩余 root cause 分线处理：

| Root cause | Cases | 实验 |
|---|---|---|
| `contact_mask` | `box023_p1` | E022 |
| `retarget_kinematic` | `box025_p1`, `bucket007_p2` | E023 |
| `algo_stability` | `bucket001_p1`, `bucket001_p2` | E024 |
| `algo_contact` + E022 unresolved | `box023_p1/p2`, `bucket005_s2_p1/p2`, `bucket007_p1` | E025 |

## 总览结论

E022-E025 没有产生 strict full-retargeting success，但阶段性价值很明确：

1. **object-side canonical support proxy 已经站住**。E022-E025 的 full variants 几乎全部 object no-regression pass，说明 E018b 的 anchor / support proxy 不是当前主瓶颈。
2. **失败被进一步拆细**：mask semantics、lower-body reference geometry、stability、contact closure、deep penetration shortcut 已经分开验证，不再混成“泛化失败”。
3. **单纯 reward 加权不够**：E024 的 stability/contact gain 对 p1 无效，E025 的 high contact reward 和 soft penetration penalty 也不能消除 artifact shortcut。
4. **下一步应该从 reward sweep 转为约束/时序/几何机制改造**：dynamic target timing、surface-contact feasibility、harder SDF barrier、CEM projection/rejection、lower-body geometry/control regularization。

## 阶段基线对比总表

读表规则：`↑` 越高越好，`↓` 越低越好；`Δ = 实验结果 - baseline`，百分比指标的 Δ 单位是 pp。`baseline` 优先使用进入该实验前的最近有效结果：E022/E024/E025 多数是 E018b，E025 的 `box023_p1` 使用 E022 mask 修复后的 best-contact 结果。

| 实验 | Case / 指标 | 方向 | Baseline | 最好/代表结果 | Δ | 阶段判断 |
|---|---|---|---:|---:|---:|---|
| E022 | `box023_p1` mask overclaim | ↓ | `54.41%` | `0.00%` | `-54.41pp` | 明显成功，mask bug 修掉 |
| E022 | `box023_p1` contact 5cm | ↑ | `24.50%` | `25.30%` | `+0.80pp` | 几乎没提升，contact 主问题仍在 |
| E023 | `box025_p1` full ref leg/object interference | ↓ | `66.53%` | `25.40%` | `-41.13pp` | 明显改善，但未达 `<15%` |
| E023 | `bucket007_p2` full ref leg/object interference | ↓ | `66.32%` | `45.26%` | `-21.05pp` | 有改善但仍很高 |
| E024 | `bucket001_p1` best pelvis min | ↑ | `0.145m` | `0.156m` | `+0.011m` | 小幅改善但仍 fall，基本失败 |
| E024 | `bucket001_p2` best pelvis min | ↑ | `0.439m` | `0.726m` | `+0.287m` | stability 明显修复 |
| E024 | `bucket001_p2` deep penetration | ↓ | `64.65%` | `59.60%` | `-5.05pp` | 只小幅改善，仍严重超标 |
| E025 | `box023_p1` contact 5cm | ↑ | `25.30%` | `28.51%` | `+3.21pp` | 小幅改善但远未达标 |
| E025 | `box023_p2` contact 5cm | ↑ | `28.57%` | `52.38%` | `+23.81pp` | 接触提高，但 fall/deep pen 变坏 |
| E025 | `bucket005_s2_p1` deep penetration | ↓ | `88.15%` | `92.89%` | `+4.74pp` | 变坏，penalty 没压住 |
| E025 | `bucket005_s2_p2` deep penetration | ↓ | `74.38%` | `64.53%` | `-9.85pp` | 有改善但远未达标 |
| E025 | `bucket007_p1` deep penetration | ↓ | `68.46%` | `35.57%` | `-32.89pp` | 最明显改善，但仍高于 `<15%` |

从这张表看，E022-E025 的正向结果主要是“诊断和局部改善”，不是 strict success：mask 修得很干净，E023 geometry 有实质下降，E024 p2 站稳了，E025 stronger penalty 对部分 bucket case 有信号；但核心指标没有同时过线。

## 实验逐项总结

### E022: contact mask semantics repair

目标：验证 `box023_p1` 的低 contact 是否主要来自 processed `contact` 字段 overclaim / time-axis mismatch。

结果：

- 4/4 full 完成。
- `num_mask_semantics_pass=3/4`，baseline control intentional fail。
- patched variants 将 mask overclaim/mismatch 从 `54.41%` 降到 `0-0.44%`。
- object no-regression `4/4`，artifact no-regression `4/4`。
- contact goal `0/4`，best strict contact only `25.30%`。

结论：

- mask bug 是真实 bug，E022 成功修正了 mask semantics。
- 但 contact preservation 没有随 mask 修复闭合，`box023_p1` 的主要瓶颈不是 mask 本身，而是 robot-side target/reward/timing。
- 后续不应继续做 mask dilation / axis sweep；`box023_p1` 正确转入 E025 contact-control 分线。

### E023: retarget kinematic geometry repair

目标：处理 E020 的 `box025_p1`、`bucket007_p2` retarget kinematic reference 里已有 lower-body/object interference 的问题。

结果：

- 6/6 full 完成。
- object no-regression `6/6`。
- ref geometry repair `0/6`，success `0/6`。
- `lowerbody_proxy_min` 有帮助但不足：
  - `box025_p1`: ref interference `66.53% -> 25.40%`
  - `bucket007_p2`: ref interference `66.32% -> 45.26%`
- `legpair_off` 不是解：会把 collision pair 删除成 artifact shortcut，box025 虽有 `72.86%` contact，但 deep penetration / sim leg artifact 明显变坏。

结论：

- lower-body collision geometry 是一部分原因，但仅 shrink proxy 不足以过 `<15%` repair gate。
- 删除 collision pairs 只是在绕约束，不是物理修复。
- 后续要么做更真实的 lower-body geometry建模，要么把 E025-style contact/collision penalty 升级为真正的 feasibility constraint。

### E024: bucket001 stability repair

目标：修复 `bucket001_p1/p2` 的 `algo_stability`，保持 object-side fields 不变。

结果：

- 5 个关键 full variants 完成。
- object no-regression `5/5`。
- strict success `0/5`。
- `bucket001_p1` 三个 variants 全部 fall，contact `0%`：
  - best pelvis min only `0.1556m`
- `bucket001_p2` 两个 variants no-fall，5cm contact `77.53-79.78%`，pelvis min `0.7128-0.7257m`。
- 但 p2 deep penetration 仍 `59.60-64.65%`，max pen `8cm+`。

结论：

- p2 的 stability 可以被 root/contact/stability 参数修住，但 contact 质量失败，属于 penetration shortcut。
- p1 不是同类 stability/contact-gain sweep 可解；height-only stability 反而更差。
- 下一步 p1 应做更强的 upright/root terminal、foot support、lower-body control regularizer，或回查 reference/support timing。

### E025: robot-side contact closure + collision penalty

目标：把 E022 unresolved 的 `box023_p1` 和 E020 `algo_contact` cases 放到统一 contact/collision penalty 框架里。

结果：

- 8/8 full 完成。
- object no-regression `8/8`。
- no-fall `7/8`。
- contact closure pass `6/8`，但 strict success `0/8`。
- penetration guard only `1/8`。
- low-contact box023:
  - `box023_p1`: strict contact `28.51%`，penetration/object/stability pass，但 contact fail。
  - `box023_p2`: strict contact `52.38%`，但 fall + deep pen `20.67%`。
- bucket penetration:
  - `bucket005_s2_p1`: hand deep pen `92.89%`，leg guard 只把 leg pen `17.54% -> 14.22%`，hand penetration 不动。
  - `bucket005_s2_p2`: s4 比 lite 有改善，deep pen `77.83% -> 64.53%`，但仍远超 `<15%`。
  - `bucket007_p1`: s4 比 lite 有明显改善，deep pen `51.01% -> 35.57%`，max pen `10.55cm -> 5.16cm`，但仍 fail。

结论：

- E025 的工程实现成功：默认关闭不影响旧实验，E025 override 正确解析 geoms，8/8 full 可跑。
- 但 soft penalty 不够强。当前 CEM 仍能找到高 contact + inside-object 的低成本路径。
- stronger scale (`4.0`) 有方向性改善，证明 penalty 信号不是完全无效，但量级/形式不足以达到 strict physical contact。

## 成功点

| 成功点 | 证据 | 意义 |
|---|---|---|
| E022 修正 mask semantics | overclaim/mismatch `54.41% -> 0-0.44%` | 后续 contact eval 不再被 all-on contact 字段污染 |
| object-side support proxy 稳定 | E022 `4/4`、E023 `6/6`、E024 `5/5`、E025 `8/8` object no-regression | anchor/support proxy 不是当前主要瓶颈 |
| E023 缩小 lower-body geometry 问题 | ref interference best 降到 `25.40%` | 证明 geometry 是因子之一，但不是完整解 |
| E024 修复 p2 stability | p2 pelvis min `>0.71m`，no-fall | stability 与 penetration 可以分开处理 |
| E025 penalty wiring 成功 | hand geoms 2、leg geoms 16 resolved，8/8 full | 后续可复用 reward path 做更强 barrier |
| E025 s4 有方向性改善 | bucket007 deep pen `51.01% -> 35.57%` | penalty 口径有信号，但需要机制升级 |

## 失败点与根因判断

| 失败点 | 证据 | 当前判断 |
|---|---|---|
| `box023_p1` contact closure | E022 best `25.30%`，E025 `28.51%` | 不是 mask 问题，也不是简单 gain 问题；更像 dynamic target / timing / contact phase 问题 |
| `box023_p2` contact closure | E025 contact `52.38%` 但 fall/deep pen | high contact reward 会用不稳定姿态换接触 |
| `bucket001_p1` stability | E024 三个 variants 全 fall | 需要更强 posture/control mechanism，不是参数微调 |
| bucket deep penetration | E025 bucket deep pen `35.57-92.89%` | soft reward penalty 太弱，缺少 hard feasibility |
| lower-body geometry | E023 best ref intf `25.40%`，E025 leg guard only local help | 需要真实 geometry / collision pair / lower-body policy 共同处理 |
| remote reliability | 多次 SSH reset、stale、stall | 远程可用但不稳定；长 rollout 需要 stall guard 和本地 fallback |

## 下一步方向

### 1. Box023 contact timing / dynamic target diagnosis

适用：`box023_p1/p2`

不要继续做 mask sweep 或简单加 contact gain。建议新增一个 E026-style 诊断实验：

- 对齐 ref contact active window、EEF target time、object support proxy time。
- 输出 per-frame target distance / SDF / velocity / contact mask panel。
- 测试 delayed/advanced contact mask、target hold window、near-field orientation gating 是否改善 strict 5cm preservation。
- 成功标准应是 `box023_p1/p2` strict contact `>=70%`，且 no-fall / penetration guard 不回退。

### 2. Penetration feasibility constraint，而不是 soft reward

适用：`bucket005_s2_p1/p2`、`bucket007_p1`、`bucket001_p2`

E025 说明 soft penalty scale 2/4 不够。下一步应做更硬的机制：

- SDF barrier 使用更陡 hinge 或 exponential barrier，并把 inside-object threshold 从 `2cm` 改为从 surface 开始惩罚。
- CEM sample rejection：若 hand/object SDF 过深，直接丢弃样本或强行降权，不让高 contact reward 抵消。
- Surface projection：contact target 不只是靠近 object，而是投影到 object surface，并沿法线保持非负 SDF。
- 分阶段 objective：先 no-penetration + posture，再开启 contact preservation，避免一开始就钻入物体。

### 3. Lower-body geometry/control 分线

适用：`box025_p1`、`bucket007_p2`、`bucket005_s2_p1`

- E023 shrink proxy 有帮助但不够，说明几何建模仍需改。
- E025 leg guard 对 `bucket005_s2_p1` 有小幅帮助，可以保留为辅助项。
- 下一步应同步改 scene collision geometry、leg-object pair、lower-body control regularizer，而不是只改 reward。

### 4. Bucket001 p1 stability/control

适用：`bucket001_p1`

- E024 已证明 height/root/contact-gain sweep 不够。
- 建议新增 upright/root terminal constraint、foot support penalty、pelvis velocity/tilt guard、或 staged contact where contact reward is disabled until posture is valid。
- 需要重新看 p1 reference motion 是否本身要求机器人进入不可行姿态。

### 5. 远程执行工程

远程仍有价值，但必须带防护：

- 所有 remote tmux 默认带 `RUN_TIMEOUT_SECONDS` / `RUN_STALL_TIMEOUT_SECONDS`。
- 每次 pull 前确认是否为 full NPZ，而不是 smoke `T=2`。
- 同名 variant 不要本地和远程同时跑，除非先定好结果命名，避免覆盖。

## 建议优先级

1. **E026 box023 timing diagnosis**：最可能快速判断 low-contact 是否来自 target timing，而不是继续 reward sweep。
2. **E027 hard no-penetration / surface projection**：针对 bucket 系列和 `bucket001_p2` 的主要瓶颈。
3. **E028 bucket001_p1 stability/control**：独立处理，不和 contact/collision 混在一起。
4. **E029 lower-body geometry refresh**：结合 E023/E025 的 leg evidence，修真实 collision geometry 与 control regularization。

## 最终判断

E022-E025 阶段没有把 E018b 的失败 case 变成 strict success，但完成了必要的排雷：

- mask bug 已修，但不是最终瓶颈；
- object-side support proxy 稳定，不应继续优先调 anchor；
- lower-body geometry、stability、contact timing、penetration shortcut 是四条不同问题；
- soft reward sweep 已经到达边界，下一阶段要转向时序诊断、硬约束和几何机制。
