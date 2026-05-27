# E029 compliant support sweep results

日期：2026-05-27

## 目标

沿 `plan/34_E029_cola_d6_support_body_redesign_plan.md` 的 phase-2 继续验证 D6-equivalent support body。上一轮结果显示：

- `d6_locked` representative raw-ref upper bound 通过；
- `d6_compliant` 默认有限力配置失败，object mean/max `0.584/1.130m`，force saturation `64.9%`；
- 因此本轮不进入 full CEM，而是先做 target shaping、force clamp、support PD gain 的 no-training sanity。

本轮仍只使用 `results/E028/candidates.json` 中 5 个 Box021 case。

## 代码变更

新增/修改：

- `scripts/E029/sweep_d6_support_sanity.py`
  - 新增 compliant sanity sweep；
  - 支持 `--force-clamps` / `--pos-kps`；
  - `MUJOCO_GL=egl` 在 import MuJoCo 前设置，修复 best-video render 的 OpenGL 初始化顺序；
  - 输出 aggregate CSV/MD 与 best video。
- `scripts/E029/check_d6_support_load_path.py`
  - 新增 `--target-lowpass-tau`、`--target-ramp-time`；
  - 新增 force/torque clamp override、pos/rot kp override、`--tag`；
  - timeseries/video 文件带 tag，避免 sweep 覆盖；
  - `nan_count` 现在也计入 huge `qacc/qvel/qpos`，避免 MuJoCo `QACC huge` 被误报为稳定。

静态检查：

```bash
python -m py_compile workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py workspace/core4d_collab_retarget/scripts/E029/sweep_d6_support_sanity.py
git diff --check -- workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py workspace/core4d_collab_retarget/scripts/E029/sweep_d6_support_sanity.py
```

均通过。

## Representative sweep

### 默认 compliant sweep

命令：

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/sweep_d6_support_sanity.py \
  --subset representative --render-best-video
```

输出：

- `results/E029/d6/sanity_sweep/representative_compliant_sweep.csv`
- `results/E029/d6/sanity_sweep/representative_compliant_sweep.md`
- `results/E029/d6/sanity_sweep/E029_d003_box021_20231018_029_p2_d6_compliant_best_raw_ref_tau0p25_ramp0p5_f250_kp200_sanity.mp4`

结果：`0/24` pass。

best setting：

| target | tau | ramp | clamp | kp | drift mean/max | object mean/max | force sat | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| raw_ref | 0.25 | 0.5 | 250 | 200 | `0.0072/0.0141m` | `0.4023/1.0089m` | `0` | false |

视频帧：

- `results/E029/d6/sanity_sweep/E029_d003_box021_20231018_029_p2_d6_compliant_best_t0p60.jpg`
- `results/E029/d6/sanity_sweep/E029_d003_box021_20231018_029_p2_d6_compliant_best_t1p20.jpg`

视觉观察：support sphere 基本贴着箱体支撑点，box/support 整体明显落后 reference；失败不是 anchor 脱开，而是 support/object 系统追不上 target。

### 扩展 pos gain / force clamp sweep

命令：

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/sweep_d6_support_sanity.py \
  --subset representative \
  --force-clamps 120 250 400 800 1200 \
  --pos-kps 200 400 800 1200 \
  --render-best-video
```

结果：`0/160` pass，但误差显著下降。

best setting：

| target | tau | ramp | clamp | kp | drift mean/max | object mean/max | force sat | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| raw_ref | 0.25 | 0.5 | 800 | 1200 | `0.0127/0.0267m` | `0.1748/0.3406m` | `0` | false |

进一步单点高 gain 诊断：

| subset | target | tau | ramp | force | pos kp | drift mean/max | target mean/max | object mean/max | pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| representative | raw_ref | 0.25 | 0.5 | 1200 | 2400 | `0.0173/0.0382m` | `0.1060/0.2621m` | `0.1178/0.2450m` | true |
| representative | raw_ref | 0.25 | 0.5 | 2400 | 4000 | `0.0196/0.0434m` | `0.0747/0.2005m` | `0.1071/0.2360m` | true |

结论：representative 不是完全不可行；默认 compliant 失败主要是 support actuator 太软。但代表 case 能过，不代表 5-case 泛化。

## 5-case candidate sanity

### 高平移 gain，保留低旋转 gain

命令：

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py \
  --subset candidates --mode d6-compliant-support --target-mode raw_ref \
  --target-lowpass-tau 0.25 --target-ramp-time 0.5 \
  --force-clamp-override 1200 --pos-kp-override 2400 \
  --tag candidates_kp2400_f1200

.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py \
  --subset candidates --mode d6-compliant-support --target-mode raw_ref \
  --target-lowpass-tau 0.25 --target-ramp-time 0.5 \
  --force-clamp-override 2400 --pos-kp-override 4000 \
  --tag candidates_kp4000_f2400
```

结果均为 `1/5` pass。

`kp=2400/f=1200`：

| variant | object mean/max | drift mean/max | force sat | pass |
|---|---:|---:|---:|---|
| `20231011_034_p1` | `0.1601/0.2830m` | `0.0125/0.0199m` | `0` | false |
| `20231011_035_p1` | `0.3088/0.5581m` | `0.0219/0.0390m` | `0` | false |
| `20231011_035_p2` | `0.1891/0.3513m` | `0.0140/0.0251m` | `0` | false |
| `20231018_029_p2` | `0.1178/0.2450m` | `0.0173/0.0382m` | `0` | true |
| `20231020_019_p1` | `0.1388/0.2655m` | `0.0116/0.0273m` | `0` | false |

`kp=4000/f=2400`：

| variant | object mean/max | drift mean/max | force sat | pass |
|---|---:|---:|---:|---|
| `20231011_034_p1` | `0.1627/0.3059m` | `0.0137/0.0276m` | `0` | false |
| `20231011_035_p1` | `0.3308/0.6038m` | `0.0291/0.0551m` | `0` | false |
| `20231011_035_p2` | `0.1794/0.3202m` | `0.0164/0.0330m` | `0` | false |
| `20231018_029_p2` | `0.1071/0.2360m` | `0.0196/0.0434m` | `0` | true |
| `20231020_019_p1` | `0.1551/0.3118m` | `0.0139/0.0265m` | `0` | false |

### locked upper-bound 对照

命令：

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py \
  --subset candidates --mode d6-locked-support --target-mode raw_ref \
  --tag candidates_locked_raw
```

结果：`3/5` pass。

| variant | target mean | object mean/max | drift mean/max | pass |
|---|---:|---:|---:|---|
| `20231011_034_p1` | `0.0461m` | `0.1235/0.2368m` | `0.0058/0.0156m` | false |
| `20231011_035_p1` | `0.0517m` | `0.1274/0.2255m` | `0.0052/0.0091m` | false |
| `20231011_035_p2` | `0.0496m` | `0.1131/0.2247m` | `0.0062/0.0124m` | true |
| `20231018_029_p2` | `0.0734m` | `0.1112/0.2500m` | `0.0098/0.0232m` | true |
| `20231020_019_p1` | `0.0662m` | `0.1056/0.3210m` | `0.0090/0.0262m` | true |

低通/ramp 对 locked 不一定有益。`tau=0.25/ramp=0.5` 时 locked 变为 `2/5` pass，`20231011_035_p1` 反而恶化到 object mean/max `0.3434/0.6716m`。因此后续不能默认用 target shaping，它会破坏部分 case 的 reference consistency。

### rotational gain 对照

命令：

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py \
  --subset candidates --mode d6-compliant-support --target-mode raw_ref \
  --force-clamp-override 2400 --pos-kp-override 4000 \
  --rot-kp-override 200 --torque-clamp-override 300 \
  --tag candidates_raw_kp4000_f2400_rkp200_t300
```

结果：`2/5` pass，但出现 scalar hinge/Euler 数值爆炸。

| variant | object mean/max | drift mean/max | force sat | torque sat | huge/nan count | pass |
|---|---:|---:|---:|---:|---:|---|
| `20231011_034_p1` | `4.6814/571.3886m` | `64.2378/9032.9703m` | `0.2324` | `0.2535` | `4` | false |
| `20231011_035_p1` | `0.1662/0.2806m` | `0.0167/0.0325m` | `0` | `0` | `0` | false |
| `20231011_035_p2` | `0.1482/0.2843m` | `0.0161/0.0243m` | `0` | `0` | `0` | false |
| `20231018_029_p2` | `0.0957/0.2995m` | `0.0186/0.0384m` | `0` | `0` | `0` | true |
| `20231020_019_p1` | `0.1166/0.4599m` | `0.0156/0.0642m` | `0` | `0` | `0` | true |

更高 rotational setting `rot_kp=500/torque=800` 5/5 爆炸，不能作为训练配置。

## 诊断

1. 当前 E029 scalar D6 load path 比 E018/E028 的 mocap anchor 更接近 COLA，但仍不是理想实现。
   - 它是 dynamic support body + equality weld；
   - 但 support pose 使用 `3 slide + 3 hinge` 6 个标量 qpos，姿态 target 走 Euler；
   - 高 rotational drive 会触发 `QACC huge`，这是计划里预期的 Euler/axis 数值风险。

2. 失败不是单纯的数据预处理问题。
   - Phase-1 已确认 Box021 old anchor 的 local `z` 规则错误，preprocessing/endpoint inference 确实有问题；
   - 但本轮 support drift 多数很小，说明 anchor/endpoint 不是这轮主要断点；
   - 5-case candidate 失败来自 D6 scalar support target tracking、姿态 servo、有限 compliance 与 reference consistency 的组合问题。

3. 不应进入 5-case full CEM。
   - compliant high translational gain 只 `1/5` pass；
   - locked upper-bound 也只有 `3/5` raw pass，且两个 fail 只是略过 `0.12m` mean object gate；
   - rotational gain 可让部分 case 变好，但引入数值爆炸；
   - 这不满足计划的 sanity success：candidate D6 sanity `>=4/5` pass、no NaN/huge、saturation `<15%`。

4. target low-pass/ramp 不是通用修复。
   - 对 representative 有帮助；
   - 对 `20231011_035_p1` locked upper-bound 会严重恶化；
   - 后续如果使用 shaping，必须 per-case gate，而不是作为全局默认。

## 决策

触发 plan 中的升级条件：**停止 scalar `3 slide + 3 hinge` D6 full 路线，进入 freejoint support body + quaternion target handling**。

下一步建议：

1. 新建 E029 freejoint variant，而不是覆盖当前 scalar D6 结果。
   - support body 用 MuJoCo `freejoint`；
   - qpos 从 `49` 变为 `50`，qvel 保持 `47`；
   - support qpos layout 为 `[pos(3), quat(4)]`，避免 Euler target；
   - object 仍保持 last freejoint。

2. 改 `check_d6_support_load_path.py` 支持 freejoint support。
   - translation error 仍按 support body world pos；
   - rotation error 用 quaternion log/rotvec；
   - target qvel 的 angular velocity 从 quaternion finite difference 计算；
   - no-training gate 先跑 locked upper-bound raw-ref 5/5，再跑 compliant。

3. 只有 freejoint locked `>=4/5` 且 compliant `>=4/5`，才生成 full overrides 并跑 CEM。

4. 若 freejoint locked 仍不能达到 `>=4/5`，说明 remaining issue 是 reference/support endpoint/data consistency，而不是 support actuator；届时再回到 per-case preprocessing/data filter。

