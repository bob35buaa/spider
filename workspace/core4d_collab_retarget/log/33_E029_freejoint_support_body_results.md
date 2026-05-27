# E029 freejoint support body results

日期：2026-05-27

## 目标

`log/32` 已经证明 scalar `3 slide + 3 hinge` D6 路线不能进入 full：

- high translational gain 只能让 representative pass；
- 5-case compliant 仅 `1/5`；
- 提高 rotational gain 触发 `QACC huge`；
- 因此按 `plan/34_E029_cola_d6_support_body_redesign_plan.md` 的条件升级到 freejoint support body + quaternion target handling。

本轮目标：验证 freejoint support body 是否能消除 scalar hinge/Euler 姿态问题，并让 no-training D6 sanity 达到 candidate gate。

## 代码变更

新增：

- `scripts/E029/generate_e029_freejoint_assets.py`
  - 输出 `results/E029/freejoint/manifest.tsv`；
  - 生成 `free_locked` / `free_compliant` 两个 profile；
  - support body 使用 MuJoCo `freejoint`，qpos layout `[pos(3), quat(4)]`；
  - object 仍保持最后 7 个 qpos；
  - 编译维度为 `nq/nv/nu=50/47/29`；
  - 数据 qpos layout：robot `0:36`，support `36:43`，object `43:50`；
  - 数据 qvel layout：robot `0:35`，support `35:41`，object `41:47`。

修改：

- `scripts/E029/check_d6_support_load_path.py`
  - 支持 `free-locked-support` / `free-compliant-support`；
  - 自动识别 support body 是 scalar6 还是 freejoint；
  - freejoint rotation error 使用 quaternion log/rotvec；
  - freejoint force/torque 使用 `mujoco.mj_applyFT` 投影到 generalized force；
  - 新增 `--pos-kd-override` / `--rot-kd-override`；
  - 输出 `support_joint_layout`、`pos_kd`、`rot_kd`。

静态检查：

```bash
python -m py_compile \
  workspace/core4d_collab_retarget/scripts/E029/generate_e029_freejoint_assets.py \
  workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py

git diff --check -- \
  workspace/core4d_collab_retarget/scripts/E029/generate_e029_freejoint_assets.py \
  workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py
```

均通过。

## Asset generation

命令：

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/generate_e029_freejoint_assets.py --force
```

输出：

- `results/E029/freejoint/manifest.tsv`
- `results/E029/freejoint/data/*/trajectory_kinematic.npz`
- derived tasks: `<source_task>_freejoint_legobj_e029free`
- scenes: `scene_e029_free_locked_*` / `scene_e029_free_compliant_*`

结果：`5 candidates × 2 profiles = 10` 条全部生成并编译通过。

代表 endpoint 仍使用 phase-1 axis/contact preflight 的 local-y remap：

| source | endpoint local |
|---|---|
| `20231011_034_p1` | `[0.1596, 0.095043, 0.230027]` |
| `20231011_035_p1` | `[0.1596, 0.103787, 0.230595]` |
| `20231011_035_p2` | `[-0.1596, 0.13234, -0.186216]` |
| `20231018_029_p2` | `[-0.1596, 0.198455, -0.226954]` |
| `20231020_019_p1` | `[-0.1596, 0.067995, -0.248675]` |

## Representative sanity

初版直接写 freejoint `qfrc_applied[dadr:dadr+6]` 时，姿态误差偏大：

| profile | object mean/max | support target mean/max | drift mean/max | quat mean/max | pass |
|---|---:|---:|---:|---:|---|
| free_locked | `0.1730/0.3777m` | `0.0515/0.0879m` | `0.0068/0.0187m` | `29.7/81.0deg` | false |
| free_compliant | `0.1359/0.2927m` | `0.1006/0.2296m` | `0.0180/0.0373m` | `11.5/38.2deg` | false |

改为 `mj_applyFT` 后：

| profile | object mean/max | support target mean/max | drift mean/max | force sat | pass |
|---|---:|---:|---:|---:|---|
| free_locked | `0.1339/0.2821m` | `0.0650/0.1405m` | `0.0089/0.0199m` | `0` | false |
| free_compliant | `0.1160/0.2513m` | `0.0815/0.1861m` | `0.0150/0.0309m` | `0.0135` | true |

结论：`mj_applyFT` 是 freejoint sanity 必需路径；representative compliant 能通过，但 locked upper-bound 仍略高于 `0.12m` object mean gate。

视频输出：

- `results/E029/freejoint/sanity/E029_d003_box021_20231018_029_p2_free_locked_rep_raw_applyft_sanity.mp4`
- `results/E029/freejoint/sanity/E029_d003_box021_20231018_029_p2_free_compliant_rep_raw_applyft_sanity.mp4`
- 抽帧：`results/E029/freejoint/sanity/E029_d003_box021_20231018_029_p2_free_locked_rep_raw_t1p20.jpg` / `...free_compliant_rep_raw_t1p20.jpg`

## Candidate sanity

### raw + mj_applyFT

命令：

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py \
  --manifest workspace/core4d_collab_retarget/results/E029/freejoint/manifest.tsv \
  --out-dir workspace/core4d_collab_retarget/results/E029/freejoint/sanity \
  --subset candidates --mode free-compliant-support --target-mode raw_ref \
  --tag candidates_raw_applyft

.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py \
  --manifest workspace/core4d_collab_retarget/results/E029/freejoint/manifest.tsv \
  --out-dir workspace/core4d_collab_retarget/results/E029/freejoint/sanity \
  --subset candidates --mode free-locked-support --target-mode raw_ref \
  --tag candidates_raw_applyft
```

free compliant: `1/5` pass。

| variant | object mean/max | support target mean | drift mean | force sat | pass |
|---|---:|---:|---:|---:|---|
| `20231011_034_p1` | `0.1993/0.3014m` | `0.0962m` | `0.0149m` | `0.0070` | false |
| `20231011_035_p1` | `0.1996/0.3375m` | `0.1032m` | `0.0139m` | `0` | false |
| `20231011_035_p2` | `0.1641/0.3208m` | `0.0818m` | `0.0127m` | `0.0076` | false |
| `20231018_029_p2` | `0.1160/0.2513m` | `0.0815m` | `0.0150m` | `0.0135` | true |
| `20231020_019_p1` | `0.1385/0.5857m` | `0.0811m` | `0.0113m` | `0.0515` | false |

free locked: `3/5` pass。

| variant | object mean/max | support target mean | drift mean | pass |
|---|---:|---:|---:|---|
| `20231011_034_p1` | `0.1190/0.2322m` | `0.0456m` | `0.0057m` | true |
| `20231011_035_p1` | `0.1288/0.2342m` | `0.0515m` | `0.0051m` | false |
| `20231011_035_p2` | `0.1151/0.2151m` | `0.0502m` | `0.0062m` | true |
| `20231018_029_p2` | `0.1339/0.2821m` | `0.0650m` | `0.0089m` | false |
| `20231020_019_p1` | `0.1104/0.3185m` | `0.0623m` | `0.0082m` | true |

locked fail case 不是 support-object 断开：drift mean 都 `<1cm`。主要剩余误差来自 orientation tracking：free locked object quat mean 约 `14-20deg`。

### rotational tuning

更高 rotational gain 不可用：

| setting | result |
|---|---|
| free locked `rot_kp=800` | `0/5` pass，`nan_count=70`，5/5 `QACC huge` |
| free compliant `pos_kp=4000, force=2400, rot_kp=800, torque=800` | `0/5` pass，`nan_count=80`，5/5 `QACC huge` |
| free locked `rot_kp=300, rot_kd=20` | 5/5 `QACC huge` |
| free locked `rot_kp=400, rot_kd=40` | 5/5 `QACC huge` |

去掉 angular velocity damping/feedforward 也没有达到 gate：

| setting | pass | notes |
|---|---:|---|
| free locked `rot_kp=200, rot_kd=0` | `2/5` | stable，但 `034_p1/035_p2` 从略过变成略不过 |
| free compliant `pos_kp=2400, force=1200, rot_kp=120, rot_kd=0` | `0/5` | stable，但 representative 从 pass 退到 object mean `0.1224m` |

## 诊断

1. freejoint 修掉了 scalar hinge/Euler 的一部分问题，但不是充分解。
   - 默认 freejoint 没有 scalar high-rot gain 那种立即爆炸；
   - 但 candidate compliant 仍 `1/5`，locked upper-bound 也只有 `3/5`；
   - 继续加 rotational gain 会再次 `QACC huge`。

2. 当前主要断点不是 anchor 脱开。
   - locked/freejoint 下 support-object drift mean 约 `0.5-0.9cm`；
   - compliant/freejoint 下 drift mean 约 `1.1-1.5cm`；
   - endpoint local-y remap 后，load path 在几何上是闭合的。

3. 剩余误差主要来自“全姿态焊接 + support body 姿态 target”的方法假设。
   - 我们现在用 equality weld/D6-equivalent 把 object 与 support body 的 6D relative pose 约束住；
   - support body 又试图追 object reference 的完整 quaternion；
   - 对 Box021 这种支撑点离 COM 较远的 case，`15-20deg` 姿态误差就足以带来 `10cm+` COM error；
   - 这和 COLA 的 supporting base body 并不完全等价：COLA 更像用 support body 的低维命令和 D6 joint limits/friction/damping 形成支撑，而不是强行给 support body 跟完整 object quaternion reference。

4. 因此，后续失败不是单一预处理错误。
   - 预处理确实有问题：E028 local `z` anchor 规则已证明错，Box021 height axis 是 local `y`；
   - 但 freejoint/sanity 中 support drift 很小，说明当前 corrected endpoint 已不是主断点；
   - 更核心的是当前 SPIDER 迁移版 D6-equivalent 方法仍把 COLA support body 简化成“一个被 PD 追踪完整 pose 的 support body + weld”，泛化性仍不足。

## 决策

不跑 5-case full CEM。

原因：

- sanity success 要求 candidate D6 sanity `>=4/5`、no huge、saturation `<15%`；
- freejoint locked raw 只有 `3/5`；
- freejoint compliant raw 只有 `1/5`；
- rotational gain tuning 触发 huge acceleration。

下一步不应继续普通 gain sweep。建议把方法改成更接近 COLA 的低维 support actuation / compliant joint：

1. support body 不追完整 object quaternion。
   - 只给 translation/yaw/height 低维 command；
   - roll/pitch 通过 joint compliance/limits 和 robot/contact 自然决定。

2. 不再把 equality weld 当作唯一 D6 joint。
   - 保留 finite compliance，但显式记录和限制 relative rotation/translation residual；
   - 如果 MuJoCo equality 仍不够表达 D6 limits/friction，需要转成多 site/constraint 或 tendon/soft body frame 的近似。

3. 在 full 之前新增一个 stricter reference-consistency audit。
   - 对每个 case 计算 support point offset、orientation error 对 COM error 的贡献；
   - 如果 locked upper-bound 都只能 `~0.13m`，说明 full CEM 不可能靠 robot control 修好。

