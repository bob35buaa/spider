# E007 计划：support proxy 时间基准修正与 E081 对齐验收

日期：2026-05-18

## Context

E006 证明 COLA-style support proxy 可以工程接入 true-freejoint MJWarp 管线，但主任务 `box025_person2_freejoint_legobj` 没有形成搬运。后验读 NPZ 发现视频中的“物体不平移、主要旋转”量化成立：

- 参考 object 水平净位移约 `1.57m`、起终旋转约 `2deg`；
- E006 main 实际水平净位移只有 `0.21-0.50m`，但旋转约 `20.6-47.6deg`；
- proxy 目标自身也只走了约 `0.79-0.84m`，约为参考位移的一半。

代码层面原因是：`_load_support_proxy()` 接收的 `qpos_ref` 已由 `spider/io.py::load_data()` 插值到 `sim_dt`，但 E006 override 显式设置 `support_proxy_ref_dt=0.0333`，导致 `idx=int(t / dt)` 以原始 ref dt 索引插值后的参考轨迹，proxy 只走半段。

用户明确要求本轮不能再以 E005 为验收基线，而要对齐 `workspace/core4d` 的 E081：work 的含义是基本实现搬运，量化和可视化需与 E081 对照。

## Claims

| Claim | 验证方式 |
|-------|----------|
| C1 修正 support proxy 时间基准后，proxy 目标能走完整参考水平位移 | E007 eval 统计 `proxy_xy_disp / ref_obj_xy_disp >= 0.90` |
| C2 若 E006 主要失败来自半速 proxy，则 dt-corrected variants 应显著提升 object transport | 与 E006 比：`box025` obj mean/max、floor contact、object xy displacement ratio、rotation error |
| C3 E007 必须使用 E081 对齐指标，而不是 E005 support-site proxy | eval 输出 `E007_vs_E081_*` 指标和 aggregate |
| C4 true-freejoint parity 不能破坏 | `contact_guidance=false`、`nu=29`、`nq_obj=7`、object actuator empty |
| C5 guard 不能以摔倒换 tracking | `box023` guard pelvis min `>=0.55m` 且 leg intf 不高于 E081 guard + margin |

## 实现改动

1. `spider/config.py`
   - `support_proxy_ref_dt <= 0` 的默认语义改为使用 `config.sim_dt`。

2. `spider/simulators/mjwp.py`
   - `_load_support_proxy()` 预计算 proxy velocity 时默认用插值后参考的 `sim_dt`；
   - `_apply_support_proxy_force()` 索引 proxy trajectory 时默认用同一 `sim_dt`。
   - 历史 E006 overrides 仍显式 `support_proxy_ref_dt=0.0333`，不会改变旧结果解释；E007 variants 使用 `-1` 触发新默认。

3. E007 脚本
   - `scripts/E007/variants.tsv`
   - `scripts/E007/generate_e007_overrides.py`
   - `scripts/run_E007_preprocess.sh`
   - `scripts/train/train_E007.sh`
   - `scripts/train/train_E007_remote_tmux.sh`
   - `scripts/run_E007_remote.sh`
   - `scripts/pull_E007_remote_results.sh`
   - `scripts/eval/eval_E007.py`

## Variants

第一轮只测试“时间基准修正是否足够恢复运输”，不引入 XML contact pad：

| Variant | Role | Queue | 目的 |
|---------|------|-------|------|
| `E007_box025_p2_yneg_k20_simdt` | main | local | E006 yneg k20 的最小 dt-corrected 对照 |
| `E007_box025_p2_ypos_k20_simdt` | main | remote_gpu0 | E006 best side 的 dt-corrected 对照 |
| `E007_box025_p2_yneg_k40_simdt` | main | remote_gpu0 | 更强 connector 是否改善 tracking |
| `E007_box025_p2_ypos_k40_simdt` | main | remote_gpu0 | best side + 更强 connector |
| `E007_box025_p2_yneg_k20_hc_simdt` | main | local | dt fix + hold-contact 是否恢复手端闭环 |
| `E007_box023_p2_xneg_k10_simdt` | guard | remote_gpu1 | guard 方向 A，防摔/运输检查 |
| `E007_box023_p2_xpos_k10_simdt` | guard | remote_gpu1 | guard 方向 B，稳定性检查 |

## E081 对齐成功标准

E081 baseline 取 `workspace/core4d/results/E081/comparison.csv`：

- `box025` main: obj mean/max `0.143/0.271m`，hand contact `89.0%`，leg intf `7.5%`，floor `59.5%`，bottom mean `-0.075m`。
- `box023` guard: obj mean/max `0.164/0.317m`，pelvis stable，leg intf `2.7%`，floor `34.7%`。

E007 的验收分两档：

1. `E007_reaches_E081_transport_proxy`
   - main `case_window_obj_err_mean_m <= 0.20`
   - main `case_window_obj_err_max_m <= 0.40`
   - main `case_window_sim_contact_frames_pct >= 80%`
   - main `case_window_sim_object_floor_contact_frames_pct <= 75%`
   - main `E007_object_xy_disp_ratio >= 0.75`
   - main `E007_object_rot_deg <= 15deg`
   - true-freejoint parity ok

2. `E007_beats_or_matches_E081_majority`
   - 与 E081 main 比较 obj mean/max、hand contact、floor contact、leg intf、bottom mean、visual transport 六项，至少四项不弱于 E081 margin：
     - obj mean <= E081 + `0.05m`
     - obj max <= E081 + `0.10m`
     - hand contact >= E081 - `10pp`
     - floor contact <= E081 + `15pp`
     - leg intf <= E081 + `5pp`
     - xy displacement ratio >= `0.75`

如果第一轮 dt-corrected E007 仍失败，下一轮不继续只扫 `kp`，转向 XML-level mocap/contact pad 或 robot-side contact/support reward。

## 预授权命令

本轮第一步先预授权本地 CUDA 与远程脚本入口：

```bash
bash workspace/core4d_collab_retarget/scripts/train/train_E007.sh __codex_auth_probe__ 0
bash workspace/core4d_collab_retarget/scripts/run_E007_remote.sh __codex_auth_probe__
bash workspace/core4d_collab_retarget/scripts/pull_E007_remote_results.sh __codex_auth_probe__
```

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E007_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E007.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E007.py --all

# full
bash workspace/core4d_collab_retarget/scripts/run_E007_remote.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E007.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/pull_E007_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E007.py --all
```
