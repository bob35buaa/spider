# E029 multi-connect bounded 6D diagnostic results

日期：2026-05-27

## 目标

`log/34` 的 point-connect 诊断说明 single point support 欠约束：初始 anchor residual 正常，但有限力下 support-object drift 达到 `0.3-1.0m`，object 不能被稳定运输。

本轮继续验证一个更接近 bounded/compliant 6D 的 MuJoCo 近似：在 object support endpoint 附近放置多组 non-collinear `connect` 约束，用 triad/cross 点集提供有限姿态约束，同时 support body 仍是 dynamic freejoint。目标是判断“多点 bounded 6D approximation”能否替代 full weld，又避免 point-connect 的欠约束。

## 代码变更

新增：

- `scripts/E029/generate_e029_multiconnect_assets.py`
  - 输出 `results/E029/multiconnect/manifest.tsv`；
  - 固定只用 `results/E028/candidates.json` 的 5 个 Box021 case；
  - support body 为 non-mocap freejoint；
  - object 内新增 support endpoint 周围的 child bodies；
  - support body 内新增同 offset 的 anchor child bodies；
  - 每对 child bodies 通过 equality `connect` 连接；
  - object 仍保持最后 7 qpos，scene 编译维度为 `nq/nv/nu=50/47/29`。

修改：

- `scripts/E029/check_d6_support_load_path.py`
  - 新增 modes：`multi-triad-support`、`multi-cross-support`、`multi-cross-stiff-support`、`multi-cross-ultra-support`；
  - 复用 freejoint support wrench projection 与 existing pass gates。

静态检查：

```bash
python -m py_compile \
  workspace/core4d_collab_retarget/scripts/E029/generate_e029_multiconnect_assets.py \
  workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py

git diff --check -- \
  workspace/core4d_collab_retarget/scripts/E029/generate_e029_multiconnect_assets.py \
  workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py
```

均通过。

## Assets

生成命令：

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/generate_e029_multiconnect_assets.py --force
```

结果：

- manifest: `results/E029/multiconnect/manifest.tsv`
- profiles: `multi_triad`, `multi_cross`, `multi_cross_stiff`, `multi_cross_ultra`
- rows: `5 candidates x 4 profiles = 20`

Profile 语义：

| profile | connect pattern | support force clamp | pos kp/kd | rot kp/kd | constraint |
|---|---|---:|---:|---:|---|
| `multi_triad` | origin, +u, +v | 400 | 1200 / 140 | 40 / 4 | soft |
| `multi_cross` | origin, +/-u, +/-v | 400 | 1200 / 140 | 40 / 4 | soft |
| `multi_cross_stiff` | origin, +/-u, +/-v | 1200 | 2400 / 160 | 80 / 6 | stiffer |
| `multi_cross_ultra` | origin, +/-u, +/-v | 2400 | 4000 / 220 | 160 / 8 | stiffest finite run |

## Sanity results

All results use:

- subset: `candidates`
- target mode: `raw_ref`
- denominator: 5 E028 candidates only
- pass gate: existing E029 load-path sanity gate, including object error, support drift, saturation, and no huge qacc

| mode | pass | representative object mean/max | representative drift mean/max | notes |
|---|---:|---:|---:|---|
| `multi-triad-support` | `0/5` | `0.493/1.013m` | `0.559/1.229m` | triad still too weak |
| `multi-cross-support` | `0/5` | `0.429/0.913m` | `0.434/1.045m` | cross improves over triad but fails |
| `multi-cross-stiff-support` | `0/5` | `0.204/0.484m` | `0.177/0.529m` | stronger but still large drift |
| `multi-cross-ultra-support` | `0/5` | `0.106/0.228m` | `0.062/0.152m` | best finite multi-connect, still fails drift/object gate |

`multi_cross_ultra` per-case results:

| variant | drift mean/max | object mean/max | force sat | torque sat | pass |
|---|---:|---:|---:|---:|---|
| `20231011_034_p1` | `0.0778/0.1297m` | `0.2062/0.2974m` | `0.000` | `0.000` | false |
| `20231011_035_p1` | `0.1181/0.2539m` | `0.1681/0.2719m` | `0.492` | `0.000` | false |
| `20231011_035_p2` | `0.0978/0.2315m` | `0.1624/0.3385m` | `0.0076` | `0.000` | false |
| `20231018_029_p2` | `0.0623/0.1519m` | `0.1062/0.2282m` | `0.000` | `0.000` | false |
| `20231020_019_p1` | `0.3900/1.0839m` | `0.2988/0.9574m` | `0.773` | `0.0309` | false |

Best representative command:

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py \
  --manifest workspace/core4d_collab_retarget/results/E029/multiconnect/manifest.tsv \
  --out-dir workspace/core4d_collab_retarget/results/E029/multiconnect/sanity \
  --subset representative --mode multi-cross-ultra-support --target-mode raw_ref \
  --tag representative_default --render-video
```

Best representative summary:

- `representative_multi-cross-ultra-support_raw_ref_representative_default_summary.md`
- object mean/max: `0.1062/0.2282m`
- support drift mean/max: `0.0623/0.1519m`
- force saturation: `0`
- pass: `0/1`

## Visualization

Rendered video:

- `results/E029/multiconnect/sanity/E029_d003_box021_20231018_029_p2_multi_cross_ultra_representative_default_sanity.mp4`

Extracted frame:

- `results/E029/multiconnect/sanity/E029_d003_box021_20231018_029_p2_multi_cross_ultra_t1p20.jpg`

Actual observation from the extracted frame: the box is visible and is partly transported near the robot. The support spheres remain near the intended top/support area, so this is visibly much better than the single point-connect diagnostic. However the support body still lags the object support frame, and the robot posture is already bent forward. This matches the metrics: object mean error is close to the gate, but support drift mean `6.2cm` and max `15.2cm` still fail the candidate sanity requirement.

## Diagnosis

1. The old anchor/preprocessing bug is real but no longer the main explanation for this branch.
   - Phase-1 already showed Box021 height axis is local `y`, so E018/E028 local-`z` canonical anchor was wrong.
   - The corrected endpoint is used in D6/freejoint/connect/multiconnect assets.
   - Point-connect initial residual was `1e-16m`, so XML support endpoint alignment itself is not broken.

2. Full weld remains the strongest current MuJoCo load path, but it is method-limited.
   - Freejoint full-weld locked raw reached `3/5` pass and drift below about `1cm`.
   - It fails mainly through full-pose orientation coupling and becomes unstable when rotational gains are increased.
   - This is better than point-connect and multi-connect for strict object transport, but still not enough to justify full CEM.

3. Single point-connect is too weak, and multi-connect only partially fixes it.
   - Single point-connect finite-force result was `0/5`, representative drift `0.837/1.635m`.
   - Multi-cross-ultra improves representative drift to `0.062/0.152m`, but still fails `>=4/5` candidate gate.
   - The monotonic improvement with stiffness means the implementation is doing something meaningful, but the finite MuJoCo `connect` approximation is still not a robust COLA D6 replacement.

4. This is a method issue more than a remaining data preprocessing issue.
   - Data preprocessing caused the earlier anchor/axis mistake.
   - After correcting endpoint semantics, the core failure pattern is load-path semantics: full weld over-constrains orientation, point connect under-constrains, multi-connect remains too compliant or too stiff depending on the profile.
   - Repeating anchor/gain sweeps will not make the method generalize.

## Decision

Do not run E029 full CEM from multi-connect.

The E029 candidate sanity criterion was `>=4/5` pass before full; the best bounded 6D approximation is still `0/5`. The strongest upper-bound remains freejoint full-weld locked raw at `3/5`, which also falls short.

Next valid direction is not another support anchor heuristic. It should be one of:

1. Implement a true per-axis D6-style joint/constraint backend, with explicit translational locks and bounded roll/pitch/yaw compliance/friction rather than equality-connect approximations.
2. If staying in MuJoCo, formulate a dedicated support-object constraint controller that controls relative pose with per-axis impedance and force limits, then validate it against the same 5-case no-training sanity gate.
3. If neither can reach `>=4/5` sanity, stop the dynamic support-body branch for SPIDER and treat COLA-style support as requiring a simulator/constraint model closer to Holosoma/Isaac D6 than the current MuJoCo equality scaffolds.

