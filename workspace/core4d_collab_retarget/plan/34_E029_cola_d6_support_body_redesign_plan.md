# E029 Plan: COLA-style D6 support body redesign for D003 Box021

日期：2026-05-27

## Context

本计划回应当前问题：E018/E028 的 support proxy 是否已经是 COLA 的 support body 设计，以及 D003 Box021 后续是否应继续调 anchor。

结论先写清楚：**当前 E018/E028 不是 COLA 原文的 dynamic support body + 6-DoF joint 设计**。当前主线是 `support_proxy_mode=mocap_pad`：

- scene 里创建 `mocap=true` 的 `support_weld_anchor`；
- object 与该 mocap body 通过 MuJoCo equality weld 连接；
- runtime 根据 `support_proxy_point_local` 从 object ref 推出 mocap 位姿；
- `mocap_pad` 下不记录真实 support reaction force，且 `connector_kp/force_clamp=0`。

这更准确地说是 **kinematic support anchor + soft weld scaffold**，它在 E014/E018 已经证明 object-side tracking 有效，但它把“人类端 held/support point”压缩成一个单点 `support_proxy_point_local`。D003 Box021 上这个点来自 face/height heuristic，因此泛化很差。

E015 曾有更接近 dynamic support 的分支：非 mocap `support_dynamic_anchor`，3 slide + 3 hinge 6 个标量自由度，通过 `qfrc_applied` 做 support PD，并用 soft weld 连到 object。但 E015 失败于 support target lag、force/torque clamp、NaN，并不是当前 E018/E028 在用的基线。

COLA 论文的关键接口不同：closed-loop training environment 包含 humanoid、模拟 human carrier 的 supporting base body、carried object；support body 和 object 通过 6-DoF joint 连接，support body 由 velocity / yaw torque / height force command 控制，joint 的 friction/damping/limits 把 support body dynamics 传到 object。

Holosoma r051-r056 给了直接证据：

- r051：free-contact support proxy 无法形成 load path；direct object wrench 可以动 object，但 kinematic/finite-force free-contact support 都失败。
- r052：support body 与 object 建 fixed joint 后，finite-force support 可以稳定 lift/carry。
- r052b：D6 locked joint 也通过。
- r052c：有限 limit + DriveAPI 的 compliant D6 也通过，drift 很小。
- r053-r055：无效，根因是 object local axis 用错，不应作为 D6 路线失败证据。
- r056：axis 修正后 sanity/smoke 恢复，说明 D6 support load path 不是阻塞项。

因此 E029 不再沿 E028b 继续“找更好 anchor”。E029 的目标是把 SPIDER 的 support scaffold 改成 COLA/Holosoma 语义的 **dynamic support body + D6-equivalent support-object constraint**，并先用不训练 sanity 验证 load path。后续只使用 `workspace/core4d_collab_retarget/results/E028/candidates.json` 的 5 条 case。

## Claims

| Claim | 验证方式 |
|---|---|
| C1 当前 E018/E028 不是 COLA dynamic support body | audit 输出 scene/config/runtime 证据：`mocap_pad`、mocap body、equality weld、no support dynamics |
| C2 E029 support body 是动态 body，不是 mocap/kinematic target | scene 编译检查 support body 非 mocap，存在 6DoF qvel 控制通道，object 仍 true-freejoint，`nu=29`，no object actuator |
| C3 support-object 连接语义对齐 COLA/Holosoma D6 | support body 与 object 通过 finite stiffness/limit 的 6D equality/D6-equivalent 约束连接，记录 local frames、drift、relative pose |
| C4 不再默认 `canonical_z=0.62` 或固定 object local 轴语义 | 每个候选先做 axis/contact preflight：local axes、contact cloud、support side、endpoint confidence |
| C5 不训练 sanity 先证明 load path | robot 隔离或弱干扰下 scripted support command 可 lift/carry object，drift `<5cm`，support saturation `<15%`，no NaN |
| C6 full retarget 只在 sanity 通过后跑 5 个 candidates | 仍分离 object tracking、robot upright、contact preservation、deep penetration、leg/floor artifact |
| C7 可视化不可省略 | axis/contact panel、D6 support sanity video、full online video 均需写入 log；视频分析交给 high/xhigh subagent |

## Non-goals

- 不把 E018/E028 的 `mocap_pad` 结果继续包装成 COLA 原版 support body。
- 不把 D003 的失败简单归因于预处理。预处理确实有问题，但当前方法本身也依赖 brittle 单点 anchor。
- 不在 sanity 失败前跑 5 条 full CEM。
- 不用更硬的 weld 掩盖 robot fall/contact artifact。

## Design

### 1. Audit current semantics

新增只读 audit：

- `scripts/E029/audit_support_semantics.py`

输出：

- `results/E029/audit/current_support_semantics.md`
- `results/E029/audit/e028_candidate_modes.csv`

检查项：

- override 是否 `support_proxy_mode=mocap_pad`；
- scene 是否有 `support_weld_anchor mocap=true`；
- support body 是否有 qpos/qvel/dof；
- object 是否仍 last freejoint；
- 是否使用 object direct `xfrc_applied`；
- `support_proxy_force` 在 mocap 模式下是否只是占位零量。

### 2. Axis/contact preflight

新增：

- `scripts/E029/preflight_axis_contact.py`

输入固定为：

- `workspace/core4d_collab_retarget/results/E028/candidates.json`

输出：

- `results/E029/preflight/axis_contact_summary.csv`
- `results/E029/preflight/*_axis_contact_panel.jpg`
- `results/E029/preflight/preflight_report.md`

每个候选记录：

- object local x/y/z 到 world 主方向的时间均值；
- selected-person / counterpart-person contact cloud 的 object-local mean、trimmed centroid、dominant side；
- support endpoint 的 confidence；
- anchor-to-contact-cloud distance；
- 是否触发 Holosoma r053-r055 风格 axis ambiguity；
- 是否可安全进入 D6 sanity。

决策规则：

- 如果 axis/contact preflight 不能给出 stable support side，不做 full retarget，只保留为 data caveat。
- 如果只是不确定高度，但 side 稳定，可以进入 D6 sanity，用 robust contact centroid 的自由轴初始化，而不是 `0.62*half_z`。

### 3. D6-equivalent scene path

MuJoCo 没有 IsaacSim 那种 USD D6 joint API；SPIDER 侧采用 D6-equivalent 语义：

- dynamic `support_dynamic_anchor` 非 mocap；
- support body 插在 robot qpos 与 object qpos 之间，保持 object 仍为最后 7 qpos；
- support body 初版使用 E015 兼容的 `3 slide + 3 hinge` 6DoF 标量关节；
- object 与 support body 通过 finite compliance equality weld/6D constraint 连接；
- equality local frame 显式写出 support body local frame 与 object support endpoint；
- 所有 direct object wrench 禁用；
- support command 只通过 support body generalized force 写入。

新增/修改：

- `scripts/E029/generate_e029_d6_assets.py`
- `scripts/E029/generate_e029_overrides.py`
- `scripts/run_E029_preprocess.sh`
- `scripts/train/train_E029.sh`
- `scripts/eval/eval_E029.py`

如果 3 slide + 3 hinge 版本再次暴露 Euler/axis 数值问题，再升级为 freejoint support body，并同时修 `mjwp.py` 的 support qpos/quaternion target handling。第一轮不直接跳 freejoint，避免把 qpos layout 和 D6 语义两个风险耦合在一起。

### 4. No-training sanity before CEM

新增：

- `scripts/E029/check_d6_support_load_path.py`

模式：

| Mode | 目的 | Gate |
|---|---|---|
| `direct-object-wrench-control` | 复核 object physics/metrics | object 可 lift/carry |
| `mocap-weld-current` | 当前 E028 scaffold 对照 | 记录 object tracking 与 weld residual，不算 dynamic support |
| `d6-locked-support` | D6-equivalent upper bound | lift/carry pass，drift `<3cm` |
| `d6-compliant-support` | COLA/Holosoma r052c 对齐版本 | lift/carry pass，drift `<5cm`，saturation `<15%` |

sanity 只先跑 2 条代表：

- `20231018_029_p2`：E028 里 face 最清晰；
- `20231020_020_p1`：E028 top review bank 里 robot upright / object pass 的边界样本。

2 条都过后，再扩到 candidates 5/5。

停止条件：

- compliant D6 support 不能 lift/carry；
- support target gap 均值 `>5cm` 或 max `>12cm`；
- support force saturation `>15%`；
- 出现 NaN/black frames；
- axis/contact preflight 标记 support endpoint 不可信。

### 5. Candidate full retarget

只有 C1-C5 都通过后，才跑 full CEM：

- denominator 固定为 `results/E028/candidates.json` 5 条；
- 每条只跑一个主配置，必要时最多一个 low-gain fallback；
- 不使用 E028b 的 leftover smoke/full 结果作为结论；
- online video 必须生成并索引。

建议主配置：

- `support_proxy_mode=dynamic_weld` 或新 `dynamic_d6`；
- support mass `2kg` 起步；
- support PD ramp `0.5-1.0s`；
- pos kp 从 E015 的 `500` 下调到 `150-300`；
- rot kp 从 E015 的 `80` 下调到 `10-30`；
- force clamp 初始 `120N`，最多对照 `250N`；
- equality compliance 按 r052c 思路先允许有限 drift，而不是 E014/E028 的硬跟踪。

## Success Criteria

### Sanity success

- candidates preflight 覆盖 `5/5`；
- representative D6 sanity `2/2` pass；
- candidate D6 sanity `>=4/5` pass；
- no direct object wrench；
- support body non-mocap；
- support drift mean `<5cm`；
- support force saturation `<15%`；
- no NaN。

### Full success

本轮先定义为机制验证，不要求 5/5 strict clean：

- full result `5/5` 完整落盘；
- object transport pass 不低于 E028 candidate baseline；
- robot fall 不高于 E028 candidate baseline；
- contact preservation / deep penetration 不因 D6 变差；
- 至少 `1` 条从 E028/E028b 失败视觉提升为 candidate clean 或 near-clean；
- log 明确说明失败是 D6 load path、endpoint selection、还是 robot-side artifact。

## Scripts

```bash
cd /home/ubuntu/Workspace/spider

# 1. audit + preflight
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/audit_support_semantics.py
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/preflight_axis_contact.py \
  --candidates workspace/core4d_collab_retarget/results/E028/candidates.json

# 2. preprocess D6 scenes
bash workspace/core4d_collab_retarget/scripts/run_E029_preprocess.sh --force

# 3. no-training sanity
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py \
  --subset representative --mode d6-compliant-support

# 4. full only after sanity pass
bash workspace/core4d_collab_retarget/scripts/train/train_E029.sh local 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E029.py --all
```

## Expected diagnosis

当前后续失败大概率不是单一预处理错误：

- E028 的 face/height anchor preflight 失败说明 D003 preprocessing/endpoint inference 确实不可靠；
- E018/E028 的 `mocap_pad` 单点 scaffold 与 COLA dynamic D6 support body 不同，方法本身不具备 anchor-free 泛化；
- E015 表明 naive dynamic support PD 会 lag/clamp/NaN，因此必须先做 D6 load-path sanity 和低带宽 compliance；
- 即使 D6 support 成功，E018b/E028 已显示 robot fall/contact artifact 是独立瓶颈，不能用 object Epos/Erot 当完整成功。

E029 的价值是把这几个因素拆开：先证明 support-object load path，再看 endpoint selection，再看 robot-side artifact。
