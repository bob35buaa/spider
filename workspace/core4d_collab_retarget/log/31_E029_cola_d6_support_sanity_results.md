# E029 结果：COLA D6 support body phase-2 sanity

日期：2026-05-27

## 目标

在 phase-1 audit/preflight 后，按 `plan/34_E029_cola_d6_support_body_redesign_plan.md` 实现 D6-equivalent support body assets，并先跑 no-training support load-path sanity。分母仍固定为 `workspace/core4d_collab_retarget/results/E028/candidates.json` 的 5 条 case。

## 新增/修改脚本

```text
workspace/core4d_collab_retarget/scripts/E029/generate_e029_d6_assets.py
workspace/core4d_collab_retarget/scripts/E029/generate_e029_overrides.py
workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py
workspace/core4d_collab_retarget/scripts/run_E029_preprocess.sh
workspace/core4d_collab_retarget/scripts/train/train_E029.sh
```

静态检查：

```bash
python -m py_compile workspace/core4d_collab_retarget/scripts/E029/*.py
bash -n workspace/core4d_collab_retarget/scripts/run_E029_preprocess.sh
bash -n workspace/core4d_collab_retarget/scripts/train/train_E029.sh
git diff --check -- workspace/core4d_collab_retarget/scripts/E029 \
  workspace/core4d_collab_retarget/scripts/run_E029_preprocess.sh \
  workspace/core4d_collab_retarget/scripts/train/train_E029.sh
```

结果：通过。

## Preprocess

命令：

```bash
bash workspace/core4d_collab_retarget/scripts/run_E029_preprocess.sh --force
```

产物：

| 产物 | 路径 |
|---|---|
| D6 manifest | `workspace/core4d_collab_retarget/results/E029/d6/manifest.tsv` |
| D6 augmented data | `workspace/core4d_collab_retarget/results/E029/d6/data/*/trajectory_kinematic.npz` |
| overrides | `examples/config/override/core4d_collab_E029_*_d6_{locked,compliant}.yaml` |
| contact masks | `workspace/core4d_collab_retarget/results/E029/d6/contact_masks/` |

结果：

| 项 | 结果 |
|---|---:|
| manifest rows | 10 |
| candidates | 5 |
| profiles per candidate | `d6_locked`, `d6_compliant` |
| derived tasks | 5 |
| scene compile dims | `nq/nv/nu=49/47/29` |
| `nmocap` | 0 |
| support body | non-mocap, 6 scalar joints |
| support q/d addr | 36 / 35 |
| object q/d addr | 42 / 41 |
| object still last freejoint | true |

Endpoint policy:

- `support_endpoint_policy=axis_remap_selected_side_centroid`
- side axis: selected `+x/-x` face surface；
- height axis: local `y`；
- free axes: selected-side robust centroid clipped to object half extent。

代表 case `20231018_029_p2` 的 endpoint 从旧 E028 `[-0.1596, 0, 0.164114]` 改为 `[-0.1596, 0.198455, -0.226954]`，即 local-`y` 高度轴重映射后的 selected-side endpoint。

## No-training sanity

命令：

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py \
  --subset representative --mode d6-locked-support --target-mode raw_ref --render-video

.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py \
  --subset representative --mode d6-compliant-support --target-mode raw_ref --render-video
```

代表 case：

- `E028_d003_box021_20231018_029_p2_canonical_t02`

Sanity 结果：

| Mode | target | drift mean/max | support target err mean/max | object pos err mean/max | force sat | NaN | gate |
|---|---|---:|---:|---:|---:|---:|---|
| `d6_locked` | raw ref | `0.0098 / 0.0232m` | `0.0734 / 0.1679m` | `0.1112 / 0.2500m` | `0.000` | 0 | pass |
| `d6_compliant` | raw ref | `0.0057 / 0.0082m` | `0.5736 / 1.1124m` | `0.5836 / 1.1296m` | `0.6486` | 0 | fail |

补充诊断：

- `smooth_final` target 下 locked support 的 support-point tracking 也能保持较小误差（support target err `0.0388 / 0.1089m`，drift `0.0061 / 0.0154m`），但 object COM 相对 smooth COM target 仍 `0.1846 / 0.2682m`，说明单 endpoint + 3 hinge 姿态闭合仍不理想。
- locked upper-bound high gain 过硬时出现过 MuJoCo `QACC huge`（`kp=20000, rkp=2000` 与 `rkp=800` 都触发过），最终稳定 sanity 使用 position upper gain `kp=8000`、rot upper gain `rkp=200`，只用于 no-training upper bound，不写入 full override。
- compliant mode 失败模式与 E015 一致：D6/equality drift 很小，说明 support-object 连接传力路径存在；真正失败来自有限 support force 无法追上 target，force saturation 约 `65%`。

## Runtime smoke

命令：

```bash
bash workspace/core4d_collab_retarget/scripts/train/train_E029.sh smoke 0
```

结果：

| 项 | 结果 |
|---|---:|
| compliant variants | 5/5 |
| root NPZ | 5/5 |
| outdir trajectory | 5/5 |
| qpos shape | `(2, 2, 49)` for 5/5 |
| qvel shape | `(2, 2, 47)` for 5/5 |
| ctrl shape | `(2, 2, 29)` for 5/5 |
| support diagnostics keys | present for 5/5 |

代表 log 显示 runtime 已加载 dynamic support reference：

```text
dynamic support: qadr=36, dadr=35, ref_qpos=(200, 6), pos_kp=200, pos_kd=40.0, rot_kp=20, ...
support proxy: ref_pos=(200, 3), point_local=[-0.1596, 0.198455, -0.226954], dt=0.0166667
```

Smoke 只证明 49/47 维 D6 scene + `dynamic_weld` runtime wiring 可加载，不声明 full 动力学成功。

## 可视化

视频：

- `workspace/core4d_collab_retarget/results/E029/d6/sanity/E029_d003_box021_20231018_029_p2_d6_locked_sanity.mp4`
- `workspace/core4d_collab_retarget/results/E029/d6/sanity/E029_d003_box021_20231018_029_p2_d6_compliant_sanity.mp4`

抽帧：

- `workspace/core4d_collab_retarget/results/E029/d6/sanity/E029_d003_box021_20231018_029_p2_d6_locked_raw_t1.jpg`
- `workspace/core4d_collab_retarget/results/E029/d6/sanity/E029_d003_box021_20231018_029_p2_d6_compliant_raw_t1.jpg`

实际观察：

- locked raw-ref：箱体在 1s 抽帧中已明显被 support body 带到机器人前方并离地/倾斜，视觉上确认 support-object load path 能带动物体。
- compliant raw-ref：1s 抽帧中只看到机器人与蓝色 support sphere，箱体没有跟上相机中心，和 CSV 中 target lag / force saturation 一致。

## Claims 当前状态

| Claim | 当前证据 | 状态 |
|---|---|---|
| C2 support body 是动态 body，不是 mocap | compiled scene `nmocap=0`、support 非 mocap、6 scalar joints、object last freejoint | pass |
| C3 D6-equivalent support-object 连接存在 | equality weld + locked sanity drift mean `9.8mm` | partial pass |
| C4 不再默认 fixed local-`z` | endpoint 来自 phase-1 axis remap + selected-side centroid | pass |
| C5 no-training sanity 先证明 load path | locked upper bound pass；compliant finite-force fail | partial / not enough for full |

## 决策

1. 不进入 5-case full retarget。`d6_compliant` 代表 sanity 未过，force saturation `~65%`。
2. D6 load path 本身不是 blocker：locked upper-bound 已能带动物体，drift `<3cm`。
3. 下一个最小动作不是重复 full，而是修 compliant support command：
   - 增加 support target ramp / low-pass，而不是 raw ref 逐点追踪；
   - 对 force clamp 做 `120N -> 250N/400N` 的 no-training sanity sweep；
   - 若 3 scalar hinge 姿态闭合仍造成 object COM/rotation gap，再升级 freejoint support body 并同步修改 `mjwp.py` 的 quaternion target handling。

