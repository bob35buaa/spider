# E007 partial 结果：support proxy 时间基准修正与 E081 对齐验收

日期：2026-05-18

## 状态

E007 已完成 plan、实现、脚本、预授权、preprocess、smoke，并完成一个本地 full 变体：

- completed: `E007_box025_p2_yneg_k20_simdt`
- stopped: 其余本地/远程 E007 队列

停止原因：首个 full 已足够暴露下一层瓶颈。继续运行同一 `support_proxy_max_xy_speed=0.8` 配置的其它 E007 变体预计不会解决运输问题，因此提前切换到 E008 高/不限速 proxy。

## 执行记录

```bash
bash workspace/core4d_collab_retarget/scripts/run_E007_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E007.sh __codex_auth_probe__ 0
bash workspace/core4d_collab_retarget/scripts/run_E007_remote.sh __codex_auth_probe__
bash workspace/core4d_collab_retarget/scripts/pull_E007_remote_results.sh __codex_auth_probe__
bash workspace/core4d_collab_retarget/scripts/train/train_E007.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E007.py --all
bash workspace/core4d_collab_retarget/scripts/run_E007_remote.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E007.sh local_wave 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E007.py E007_box025_p2_yneg_k20_simdt
```

## 结果路径

| 产物 | 路径 |
|------|------|
| Plan | `workspace/core4d_collab_retarget/plan/07_E007_support_proxy_timebase_e081_plan.md` |
| Results | `workspace/core4d_collab_retarget/results/E007/` |
| Completed NPZ | `workspace/core4d_collab_retarget/results/E007/E007_box025_p2_yneg_k20_simdt.npz` |
| Completed video | `workspace/core4d_collab_retarget/results/E007/E007_box025_p2_yneg_k20_simdt.mp4` |
| Keyframes | `workspace/core4d_collab_retarget/results/E007/keyframes/E007_box025_p2_yneg_k20_simdt/` |
| Eval summary | `workspace/core4d_collab_retarget/results/E007/eval_summary_E007_box025_p2_yneg_k20_simdt.json` |
| Logs | `logs/core4d_collab_retarget/E007/` |

## Smoke

Smoke 只验证 wiring，不作为任务结论：

```json
{
  "num_results": 7,
  "num_freejoint_parity_ok": 7,
  "num_support_proxy_metrics_present": 7
}
```

## Full partial 指标

E081 main baseline：`E081_box025_p2_legobj` case-window obj mean/max `0.143/0.271m`，hand contact `89.0%`，leg intf `7.5%`，floor `59.5%`。

`E007_box025_p2_yneg_k20_simdt`:

| 指标 | 值 |
|------|----|
| obj mean/max | `0.625 / 1.205m` |
| hand contact | `61.3%` |
| floor contact | `93.1%` |
| leg intf | `0.0%` |
| object xy displacement ratio | `0.291` |
| case-window object xy ratio | `0.247` |
| object start/end rotation | `41.6deg` |
| proxy xy ratio vs ref object | `0.693` |
| connector gap mean/max | `0.311 / 0.553m` |
| force mean/max | `25.1 / 40.9N` |
| torque mean/max | `10.6 / 19.3Nm` |
| E081 majority score | `1/6` |
| reaches E081 transport proxy | `false` |

## 关键诊断

E007 修正了 E006 发现的时间基准错误：运行日志中 `_load_support_proxy` 打印 `dt=0.0166667`，说明 support proxy 已按插值后的 `sim_dt` 索引，而不是 E006 的原始 `ref_dt=0.0333`。

但 dt 修正后出现下一层问题：`support_proxy_max_xy_speed=0.8` 变成新的限速瓶颈。`box025` 的 reference support point 在快速段超过 `0.8m/s`，proxy 被限速后最终只走完约 `69%` 的参考水平位移。object 本身只走完约 `29%`，同时旋转 `41.6deg`，仍然是“旋转替代平移”。

所以 E007 的结论不是“dt fix 无效”，而是：

1. E006 的半速问题被修正；
2. 但 E007 仍被 proxy max-speed clamp 截断；
3. 在 proxy target 本身没有完整平移前，不能判断 support proxy 范式是否彻底失败。

## 下一步

E008 不再沿用 `support_proxy_max_xy_speed=0.8`。优先测试：

- `support_proxy_max_xy_speed=0.0`（不限制）
- `support_proxy_max_xy_speed=2.0`
- 必要时提高 force clamp / torque clamp，但先保持 effort 在可解释范围内

E008 的第一验收门槛是 `proxy_xy_disp_ratio_vs_ref_obj >= 0.95`；如果 proxy 能完整平移但 object 仍不运输，再切到 mocap contact pad 或 robot-side contact/support reward。
