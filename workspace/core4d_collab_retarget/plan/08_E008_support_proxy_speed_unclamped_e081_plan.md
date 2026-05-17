# E008 计划：support proxy 高/不限速与 E081 搬运验收

日期：2026-05-18

## Context

E006/E007 的视频和 NPZ 都显示 `box025` 在 true-freejoint 下没有形成参考所需的水平搬运，而是“平移不足、旋转过量”：

- 参考 object 水平净位移约 `1.57m`，起终旋转约 `2deg`；
- E006 main 实际水平净位移约 `0.21-0.50m`，旋转约 `20.6-47.6deg`；
- E007 修正了 E006 的 `support_proxy_ref_dt` 时间基准，但首个 full `E007_box025_p2_yneg_k20_simdt` 仍只有 object xy ratio `0.291`、旋转 `41.6deg`；
- E007 的 proxy 自身只走完约 `69%` 参考水平位移，原因是 `support_proxy_max_xy_speed=0.8` 截断了快速段。

因此 E008 不继续扫同一限速配置，而是先验证 support proxy target 是否能完整跟随参考 support point。只有当 proxy 本身完整平移后，才判断 force-only off-COM support proxy 是否仍然导致物体绕支撑点旋转。

## Claims

| Claim | 验证方式 |
|-------|----------|
| C1 取消或提高 `support_proxy_max_xy_speed` 后，proxy target 能完整走完参考 support point | 新 eval 输出 `proxy_xy_disp / ref_support_xy_disp >= 0.95` 且 final proxy-to-ref-support gap 小 |
| C2 如果 E007 主要失败来自限速，E008 main 应显著提升 object xy transport 并降低旋转 | 对比 E007：object xy ratio、object rotation、obj mean/max、floor contact |
| C3 如果 proxy 完整平移但 object 仍只旋转不搬运，则 force-only off-COM wrench 路线可判定不足 | `E008_proxy_support_tracking_ok=true` 但 `E008_reaches_E081_transport_proxy=false` |
| C4 E008 eval 必须继续对齐 E081，而不是 E005 | 输出 `E008_vs_E081_*`、`E008_e081_majority_score`、`E008_reaches_E081_transport_proxy` |
| C5 true-freejoint parity 不被破坏 | `contact_guidance=false`、`scene_name=""`、`nu=29`、`nq_obj=7`、object actuator empty |

## 实现改动

1. 新增 E008 脚本与 overrides：
   - `workspace/core4d_collab_retarget/scripts/E008/variants.tsv`
   - `workspace/core4d_collab_retarget/scripts/E008/generate_e008_overrides.py`
   - `workspace/core4d_collab_retarget/scripts/run_E008_preprocess.sh`
   - `workspace/core4d_collab_retarget/scripts/train/train_E008.sh`
   - `workspace/core4d_collab_retarget/scripts/train/train_E008_remote_tmux.sh`
   - `workspace/core4d_collab_retarget/scripts/run_E008_remote.sh`
   - `workspace/core4d_collab_retarget/scripts/pull_E008_remote_results.sh`
   - `workspace/core4d_collab_retarget/scripts/eval/eval_E008.py`

2. E008 eval 在 E007 基础上新增 support-reference 诊断：
   - `E008_ref_support_xy_disp_m`
   - `E008_proxy_xy_disp_ratio_vs_ref_support`
   - `E008_proxy_final_gap_to_ref_support_m`
   - `E008_proxy_support_tracking_ok`

3. 本轮不改 `spider/simulators/mjwp.py` 的施力公式。仅把 support proxy 日志文案从 E006 专名改成通用名称，避免后续日志误读。E008 先隔离限速变量；如果失败，再进入 E009 的 contact pad / robot-side support reward。

## Variants

| Variant | Role | Queue | 目的 |
|---------|------|-------|------|
| `E008_box025_p2_yneg_k20_vmax0` | main | local | E007 首个失败变体的不限速对照 |
| `E008_box025_p2_yneg_k40_vmax0` | main | local | 不限速 + 更强 connector，检查 gap 是否缩小 |
| `E008_box025_p2_ypos_k20_vmax0` | main | remote_gpu0 | E006/E007 较优 side 的不限速对照 |
| `E008_box025_p2_yneg_k20_vmax2` | main | remote_gpu0 | 高限速 `2.0m/s`，避免完全不限速带来的冲击 |
| `E008_box025_p2_ypos_k20_vmax2` | main | remote_gpu0 | side 对照 + 高限速 |
| `E008_box023_p2_xneg_k10_vmax0` | guard | remote_gpu1 | guard 方向 A，不限速是否摔倒/干涉 |
| `E008_box023_p2_xpos_k10_vmax0` | guard | remote_gpu1 | guard 方向 B，不限速是否稳定 |

## E081 对齐成功标准

E081 baseline 仍取 `workspace/core4d/results/E081/comparison.csv`：

- main `E081_box025_p2_legobj`: obj mean/max `0.143/0.271m`，hand contact `89.0%`，leg intf `7.5%`，floor `59.5%`；
- guard `E081_box023_p2_legobj`: obj mean/max `0.164/0.317m`，floor `34.7%`。

E008 先过 proxy gate，再过 transport gate：

1. `E008_proxy_support_tracking_ok`
   - `E008_proxy_xy_disp_ratio_vs_ref_support >= 0.95`
   - `E008_proxy_final_gap_to_ref_support_m <= 0.08`

2. `E008_reaches_E081_transport_proxy`
   - role 为 main；
   - true-freejoint parity ok；
   - proxy support tracking ok；
   - `case_window_obj_err_mean_m <= 0.20`
   - `case_window_obj_err_max_m <= 0.40`
   - `case_window_sim_contact_frames_pct >= 80%`
   - `case_window_sim_object_floor_contact_frames_pct <= 75%`
   - `E008_object_xy_disp_ratio >= 0.75`
   - `E008_object_rot_deg <= 15deg`

3. `E008_beats_or_matches_E081_majority`
   - 与 E081 main 比较 obj mean/max、hand contact、floor contact、leg intf、xy transport 六项，至少四项达标。

## 预授权命令

```bash
bash workspace/core4d_collab_retarget/scripts/train/train_E008.sh __codex_auth_probe__ 0
bash workspace/core4d_collab_retarget/scripts/run_E008_remote.sh __codex_auth_probe__
bash workspace/core4d_collab_retarget/scripts/pull_E008_remote_results.sh __codex_auth_probe__
```

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E008_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E008.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E008.py --all

# full
bash workspace/core4d_collab_retarget/scripts/run_E008_remote.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E008.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/pull_E008_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E008.py --all
```

## 决策规则

- 若 proxy gate 不通过：继续排查 proxy trajectory 生成/索引/速度限制，不能进入 contact pad 结论。
- 若 proxy gate 通过但 object 仍只旋转不平移：E009 转向 XML-level mocap contact pad 或 robot-side hand support reward；不再继续只扫 `kp`。
- 若某个 main 接近 E081 transport gate：围绕该 side / clamp / contact reward 做 E009 局部加密。
