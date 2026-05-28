# E086 Plan: raw-target CEM failure iteration

日期：2026-05-28

## Context

E085 修复了两类明确的数据/target 问题：

- 旧 `ref_fk` / G1 wrist pseudo target 和 raw CORE4D hand-object surface contact 平均偏差约 `27cm`；
- raw visual target 投影到 collision box 时，旧 face 选择会把部分离 `+y` 侧面更近的点误投到 `-z` 面，已改为 nearest-face projection。

E085 main full CEM 结果：

- contact 提升到 `82.95%`；
- hand-floor 为 `0%/0%`，pelvis 没有倒地；
- 但 head-object penetration `18.60%`，upperbody-object penetration `53.49%`，first head penetration frame `35`；
- 关键帧显示 sim 为了贴低位 hand target，长期弯腰压箱，头/肩/肘和箱体干涉。

target-selection audit 进一步说明：

- main left `broad_projected` vfrac mean/median `0.057/0.060`；
- `tip_best_projected` `0.052/0.044`，`tip_mean_projected` `0.063/0.068`；
- active frame 平均 `4.55/5` 个 fingertip 在 3cm 内，tip min dist mean `0.76cm`；
- 因此 main left 低位不是 contact mask 单独错误，也不是 broad centroid 单独错误，而是 raw SMPL-X/object 接触几何本身对 G1 形态很难直接满足。

E086 的目标不是继续调旧 wrist target，而是区分：

1. CEM 是否只是利用穿透漏洞；
2. raw low target 是否对 G1 形态不可达；
3. 是否需要转为 COLA/support-body 风格，让 object support 不再完全依赖手部低位目标。

## Claims

| Claim | 验证方式 | 成功标准 |
|---|---|---|
| C1 E085 失败主因是穿透接触漏洞 | 加 hand-deep penalty / stricter upperbody penalty | hand/head/elbow penetration 显著下降；若 contact 同时崩掉，说明 raw target 不可达 |
| C2 低位 raw target 是 G1 形态瓶颈 | vfrac floor / high-close target variant | 抬高左手 target 后 upperbody penetration 明显下降，即使 contact/object tracking 有 tradeoff |
| C3 需要 support-body seed | support-body/COLA-style route | object 不再靠头/肘/手穿透来获得支撑，body 姿态更接近 upright carry |

## 实验组

### E086A: raw target + strict penetration penalties

目的：保留 E085 nearest-face raw target，不改数据，只堵住 reward loophole。

改动：

- 新增 hand-object deep penetration penalty，和 upperbody penalty 分开；
- hand penalty 只惩罚深穿透，不惩罚正常接触附近：
  - `hand_object_deep_penalty_scale: 8-12`
  - `hand_object_deep_penalty_threshold_m: 0.01`
  - geoms: `["lh", "rh"]`
- upperbody penalty 加强：
  - `robot_object_penalty_scale: 8`
  - margin 保持 `0.02m`
- contact gain 从 `5.0` 降到 `3.0`，防止 contact reward 压过 collision safety。

判定：

- 若 hand/head/elbow penetration 大幅下降，但 contact 掉到 `<40%` 或 object error 变差，说明 low target 对当前 G1 姿态不可达；
- 若 penetration/contact 同时可接受，E086A 进入 guard。

### E086B: raw target with minimum vertical fraction

目的：检验“把左手从低侧/下沿抬到侧面中低位”是否能避免弯腰压箱。

改动：

- 在 target 生成阶段对 active target 做 per-frame world-vertical clamp：
  - main left min vfrac `0.15` 或 `0.20`；
  - right 保持 raw；
  - guard 不先改，避免破坏 positive case。
- target 仍投影到 nearest collision face；
- reward 使用 E085 base 或 E086A 的弱版 safety。

判定：

- 若 upperbody penetration 明显下降且接触不完全崩，说明 E085 低位 raw target 不适合直接作为 G1 contact target；
- 若抬高后仍压箱，说明问题不仅是 target height，还包括 object support / CEM dynamics。

### E086C: support-body/COLA seed route

目的：回到用户指出的 COLA support body 思路：让物体由 support body/6-DoF connector 提供可控支撑，手部 contact 不再承担全部 object stabilization。

初步路线：

- 复用已有 `support_proxy_*` / `support_dynamic_*` 代码，而不是继续调 anchor；
- object 保持自由或 scene_act actuated，support body 追踪 object/support point；
- hand contact reward 降权，body/upright + object tracking 升权；
- 若 E086A/B 都证明 raw low target 不可达，优先实现 E086C。

判定：

- 不要求 contact 第一轮最高，但要求 head/torso/upperbody penetration 明显低于 E085；
- object 不翻、不长期贴地；
- 视觉上不再靠头/肘压箱。

## 运行顺序

1. 等 E085 remote guard 完成并回收，确认 raw target 是否破坏 box023 guard。
2. 先跑 E086A main 本地 smoke/full：
   - 若 penetration 降但 contact 崩，直接转 E086B；
   - 若 penetration/contact 都改善，再远程 guard。
3. E086B 只跑 main，验证 low-target 不可达假设。
4. E086A/B 都不能解决时，进入 E086C support-body seed。

## 成功标准

E086 main 至少需要：

- head-object penetration `<5%`；
- upperbody-object penetration `<15%`；
- hand deep penetration 明显低于 E085；
- hand-floor `<5%`；
- contact `>=50%`；
- object error mean 不劣于 E085 超过 `10cm`；
- 视频中不能再出现头/肘压箱作为主要支撑。

## 失败后不重复规则

- 如果 E086A 只是让 contact 消失，不继续加 penalty；
- 如果 E086B 抬高 target 后仍压箱，不继续手调 vfrac；
- 如果 E086C 仍需要 anchor 人工选择，则回到 COLA support body 的 6-DoF connector 设计，不走 support proxy anchor sweep。
