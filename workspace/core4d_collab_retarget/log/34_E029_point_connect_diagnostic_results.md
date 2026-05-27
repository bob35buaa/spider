# E029 point-connect diagnostic results

日期：2026-05-27

## 目标

`log/33` 显示 freejoint + full weld/D6-equivalent 仍未过 5-case sanity：

- free compliant raw `1/5` pass；
- free locked raw `3/5` pass；
- support-object drift 很小，但 object orientation error 造成 COM error；
- 提高 rotational gain 会触发 `QACC huge`。

本轮做一个更窄的机制诊断：把“支撑点平移闭合”和“完整 6D 姿态焊接”拆开。新增 point-connect 版本只把 support body origin 和 object support contact body 做 equality `connect`，不锁完整姿态。它不是最终 COLA 实现，只用于判断 full-pose weld 是否是唯一瓶颈。

## 代码变更

新增：

- `scripts/E029/generate_e029_connect_assets.py`
  - 输出 `results/E029/connect/manifest.tsv`；
  - 生成 5 个 `connect_compliant` 诊断 scene/data；
  - support body 仍是 freejoint，qpos layout 与 freejoint 分支一致：robot `0:36`，support `[pos,quat] = 36:43`，object `43:50`；
  - object 内新增 massless child body `e029_object_support_contact`，位置为 corrected support endpoint；
  - equality 从 full weld 改为 `connect`：`support_dynamic_anchor` ↔ `e029_object_support_contact`；
  - 编译维度仍为 `nq/nv/nu=50/47/29`。

修改：

- `scripts/E029/check_d6_support_load_path.py`
  - 新增 mode `connect-compliant-support`，用于读取 `connect_compliant` profile。

静态检查：

```bash
python -m py_compile \
  workspace/core4d_collab_retarget/scripts/E029/generate_e029_connect_assets.py \
  workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py

git diff --check -- \
  workspace/core4d_collab_retarget/scripts/E029/generate_e029_connect_assets.py \
  workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py
```

均通过。

## Asset generation

命令：

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/generate_e029_connect_assets.py --force
```

结果：`5/5` 生成并编译通过。

初始一致性检查：

- 在 representative qpos_ref[0] 下，`support_dynamic_anchor` 与 `e029_object_support_contact` 的 world position 距离为 `1.1e-16m`；
- 说明 XML `connect` anchor 初始化没有明显错误。

## Sanity results

### upper-bound point force

命令：

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py \
  --manifest workspace/core4d_collab_retarget/results/E029/connect/manifest.tsv \
  --out-dir workspace/core4d_collab_retarget/results/E029/connect/sanity \
  --subset candidates --mode connect-compliant-support --target-mode raw_ref \
  --force-clamp-override 0 --torque-clamp-override 0 \
  --pos-kp-override 8000 --pos-kd-override 250 \
  --rot-kp-override 0 --rot-kd-override 0 \
  --tag candidates_point_upper
```

结果：`0/5` pass，5/5 出现 `QACC huge`，不作为有效物理配置。由于初始 residual 为零，爆炸主要来自无限/过硬 point constraint drive，而不是 XML anchor 初始化错位。

### finite-force point-connect

命令组：

```bash
# kp=400, force=120
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py \
  --manifest workspace/core4d_collab_retarget/results/E029/connect/manifest.tsv \
  --out-dir workspace/core4d_collab_retarget/results/E029/connect/sanity \
  --subset candidates --mode connect-compliant-support --target-mode raw_ref \
  --force-clamp-override 120 --torque-clamp-override 0 \
  --pos-kp-override 400 --pos-kd-override 60 \
  --rot-kp-override 0 --rot-kd-override 0 \
  --tag candidates_point_kp400_f120

# kp=800, force=250
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py \
  --manifest workspace/core4d_collab_retarget/results/E029/connect/manifest.tsv \
  --out-dir workspace/core4d_collab_retarget/results/E029/connect/sanity \
  --subset candidates --mode connect-compliant-support --target-mode raw_ref \
  --force-clamp-override 250 --torque-clamp-override 0 \
  --pos-kp-override 800 --pos-kd-override 100 \
  --rot-kp-override 0 --rot-kd-override 0 \
  --tag candidates_point_kp800_f250

# kp=1200, force=400
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py \
  --manifest workspace/core4d_collab_retarget/results/E029/connect/manifest.tsv \
  --out-dir workspace/core4d_collab_retarget/results/E029/connect/sanity \
  --subset candidates --mode connect-compliant-support --target-mode raw_ref \
  --force-clamp-override 400 --torque-clamp-override 0 \
  --pos-kp-override 1200 --pos-kd-override 140 \
  --rot-kp-override 0 --rot-kd-override 0 \
  --tag candidates_point_kp1200_f400
```

结果：所有有限力 point-connect 配置均 `0/5` pass，无 NaN，但 support-object point drift 很大。

| setting | pass | representative object mean/max | representative drift mean/max | saturation |
|---|---:|---:|---:|---:|
| `kp400_f120` | `0/5` | `0.641/1.254m` | `0.507/1.418m` | `0.541` |
| `kp800_f250` | `0/5` | `0.672/1.280m` | `0.832/1.652m` | `0.135` |
| `kp1200_f400` | `0/5` | `0.662/1.261m` | `0.837/1.635m` | `0.095` |

Best-ish case in this diagnostic was not representative: `20231011_035_p1` at `kp1200_f400` reached object mean/max `0.165/0.260m`, but drift mean was still `0.296m`, so it does not prove a useful load path.

## Visualization

Rendered representative finite-force point-connect video:

- `results/E029/connect/sanity/E029_d003_box021_20231018_029_p2_connect_compliant_representative_point_kp1200_f400_sanity.mp4`
- frame: `results/E029/connect/sanity/E029_d003_box021_20231018_029_p2_connect_point_kp1200_f400_t1p20.jpg`

Actual observation from the extracted frame: robot is bent near the support sphere, but the box has already moved out of view / is not being transported with the reference. This matches the metrics: single point-connect has large support-object drift and cannot replace full D6/weld load path.

## Diagnosis

1. Full weld is not the only issue.
   - Removing orientation lock and keeping only point translation does not improve sanity;
   - point-connect loses support-object closure by `0.3-1.0m` mean drift under finite force;
   - therefore a single point support constraint is too weak for this Box021 transport reference.

2. The previous full-weld/freejoint result remains the strongest current load path.
   - freejoint full-weld locked raw gave `3/5` pass with drift `<1cm`;
   - point-connect gives `0/5` and much larger drift;
   - so the next method should not downgrade to single connect.

3. The missing piece is a true compliant 6D/limited joint behavior, not more anchor tuning.
   - E018/E028 anchor/local-z preprocessing was wrong, but corrected endpoint already closes the point geometry;
   - full weld over-constrains orientation and needs too much rotational drive;
   - point-connect under-constrains the object and cannot carry it;
   - COLA-aligned next step should be a finite-compliance 6D joint approximation with limited relative rotation / damping / friction, or a multi-constraint approximation that allows bounded roll/pitch without dropping the load path.

## Decision

Do not run full CEM from point-connect.

Next valid branch:

- keep freejoint support body and corrected endpoint;
- replace binary full weld vs single point connect with a bounded 6D approximation:
  - translation constraint at support point remains stiff enough to keep drift `<5cm`;
  - relative roll/pitch compliance is finite, not fully locked;
  - yaw/height/translation can be commanded low-dimensionally;
  - sanity gate remains `>=4/5`, no huge, saturation `<15%`.

