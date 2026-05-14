# E069 Results — First-tick ref-control warmup 验证失败

**日期**: 2026-05-14
**实验域 (exp_name)**: `core4d`
**对应 Plan**: `workspace/core4d/plan/74_E069_first_tick_warmup_plan.md`
**前置**: `workspace/core4d/log/88_E068_mjwp_init_drift_results.md`

## 1. 背景

E068 发现真实 MJWP 轨迹的最早期 drift 出现在 first committed step 后，且旧结果首帧 robot ctrl 相对 ref 有 1.56 rad 偏差。因此 E069 验证一个最小假设：

> 如果前 0.2s / 0.5s 强制使用 `ctrl_ref`，不让 CEM 在 t=0 覆盖控制，那么 early yaw drift 与 pre-contact lunge 应明显下降。

## 2. 运行配置

| 变体 | YAML | warmup | 运行命令 |
|------|------|--------|----------|
| E069-W02 | `examples/config/override/core4d_e069w02_box023.yaml` | 0.20s | `bash workspace/core4d/scripts/train/train_E069.sh parallel 0 0` |
| E069-W05 | `examples/config/override/core4d_e069w05_box023.yaml` | 0.50s | 同上 |

运行说明：

- 本机 sandbox 内看不到 CUDA；非 sandbox 命令确认 RTX 5090 可用。
- W02/W05 均跑到 `sim_steps: 272/272`。
- 首轮运行在保存阶段失败：`info_list` 的 `improvement` 字段 warmup tick 是 scalar，CEM tick 是 `(32,)`，`np.stack` shape mismatch。
- 已修复 `examples/run_mjwp.py`：核心轨迹字段正常保存，shape 不稳定的诊断字段跳过并 warning。

## 3. 结果路径

| 类型 | 路径 |
|------|------|
| W02 NPZ | `workspace/core4d/results/E069/E069W02_box023.npz` |
| W05 NPZ | `workspace/core4d/results/E069/E069W05_box023.npz` |
| W02 video | `workspace/core4d/results/E069/E069W02_box023.mp4` |
| W05 video | `workspace/core4d/results/E069/E069W05_box023.mp4` |
| Eval summary | `workspace/core4d/results/E069/eval_summary.csv` |
| Keyframes | `workspace/core4d/results/E069/keyframes/` |
| Scene snapshot | `workspace/core4d/results/E069/scene_snapshot/manifest.txt` |
| Logs | `logs/E069/E069W02_box023.log`, `logs/E069/E069W05_box023.log` |

## 4. 量化结果

| 指标 | E062/E063 参考 | E069-W02 | E069-W05 | 目标 | 结论 |
|------|----------------|----------|----------|------|------|
| t=0.017s yaw err | 12.34 deg | 12.40 deg | 12.40 deg | < 5 deg | FAIL |
| t=0.033s yaw err | 22.00 deg | 22.16 deg | 22.16 deg | < 5 deg | FAIL |
| robot ctrl max diff during warmup | 1.56 rad | 0.00 rad | 0.00 rad | <= 0.05 rad | PASS |
| object ctrl max diff during warmup | ~0.01 | 0.00 | 0.00 | small | PASS |
| B1 max foot z [0,2s] | 0.481m | 0.222m | 0.428m | <= 0.10m | FAIL |
| B2 single-foot runs | high | 4 | 2 | 0 | FAIL |
| pelvis_min_intent | E063 0.192m | 0.578m | 0.197m | >= 0.50m | W02 PASS / W05 FAIL |
| final object pos err | - | 0.1048m | 0.1012m | lower is better | similar |

### 4.1 评估脚本修正

`run_mjwp.py` 在 contact guidance 且 ctrl dim 不匹配时会先执行：

```python
ctrl_ref = qpos_ref[:, : config.nu]
```

然后再做 scene_act 转换。`eval_E069.py` 初版直接使用原始 29-dim `ctrl_ref` 补齐到 35-dim，误报 warmup ctrl diff 为 1.57/1.88 rad。已修正为 mirror run 口径，重算后 W02/W05 warmup ctrl diff 均为 0。

## 5. 可视化

关键帧：

| 变体 | 0.20s | 0.60s | 0.80s |
|------|-------|-------|-------|
| W02 | `keyframes/E069W02_t0.20s.jpg` | `keyframes/E069W02_t0.60s.jpg` | `keyframes/E069W02_t0.80s.jpg` |
| W05 | `keyframes/E069W05_t0.20s.jpg` | `keyframes/E069W05_t0.60s.jpg` | `keyframes/E069W05_t0.80s.jpg` |

**实际观察**:

- W02/W05 在 t=0.20s 已经相对 ref 明显转身，sim 右腿抬起并进入单脚支撑模式；这发生在 warmup 内，说明不是 CEM 覆盖造成。
- W02 t=0.80s 是前倾长步接近 box，姿态仍非稳定双脚弯腰。
- W05 t=0.80s 更差，sim 与 box 拉开，转身并进入明显单脚/侧向模式。
- W02 的 `pelvis_min_intent=0.578m` 数值看起来通过，但视频仍不是可接受搬运；这是又一个 pelvis_z 单指标误导案例。

## 6. Claims 验证

| Claim | 结果 |
|-------|------|
| C1: warmup 能压住 early yaw drift | **失败** — yaw err 12.40/22.16 deg，与旧结果几乎相同 |
| C2: warmup 能压住 pre-contact foot lift | **失败** — B1 0.222m / 0.428m，均 > 0.10m |
| C3: box023 body stability 改善 | **部分** — W02 pelvis_min_intent 0.578m 通过，但视觉仍前倾长步；W05 失败 |
| C4: first ctrl 不再极端偏离 ref | **通过** — warmup 窗口内 robot/object ctrl diff 均 0 |
| C5: 视频实际姿态改善 | **失败** — 0.2s 即偏航，0.8s 仍 lunge / 单脚模式 |

## 7. 结论

E069 推翻了 E068 的最强假设：

> early yaw drift 不是 first-tick CEM override 单独造成。即使 warmup 期间实际提交的 ctrl 与 ref 完全一致，MJWarp commit step 仍在头两帧产生 12/22 deg yaw drift。

新的根因方向：

1. **MJWarp commit dynamics vs MuJoCo CPU step 不一致**：E068 的 CPU `mj_step(ctrl_ref)` 只偏 0.22 deg；E069 的 MJWarp `step_env(ctrl_ref)` 一步后偏 12.4 deg。
2. **需要做 ref-control parity test**：同一 `qpos_ref[0] / qvel_ref[0] / ctrl_ref[0:k]`，逐步比较 MuJoCo `mj_step` 和 MJWarp `step_env` 的 qpos/qvel/contact/qfrc。
3. 暂停 trust-region / delta-clamp 方向：warmup 已证明控制没有偏 ref，继续限制 CEM 不会解决最早期 drift。

## 8. 下一步

E070: **MJWarp ref-control commit parity 诊断**。

最小设计：

- 固定 `qpos_ref[0]`, `qvel_ref[0]`, `ctrl_ref[0:12]`。
- 分别跑 CPU MuJoCo `mj_step` 与 MJWarp `step_env`。
- 每个 substep dump pelvis quat/yaw、foot z、qpos diff、qvel diff、ctrl、contact count、object/robot actuator force。
- 若 MJWarp 复现 12/22 deg 而 CPU 不复现，定位到 MJWarp step / actuator / contact 参数路径。
- 若两者一致，则回查 E068 CPU 诊断口径。

## 9. 改动文件

| 类型 | 路径 |
|------|------|
| 保存修复 | `examples/run_mjwp.py` |
| W02 config | `examples/config/override/core4d_e069w02_box023.yaml` |
| W05 config | `examples/config/override/core4d_e069w05_box023.yaml` |
| 训练脚本 | `workspace/core4d/scripts/train/train_E069.sh` |
| 评估脚本 | `workspace/core4d/scripts/eval/eval_E069.py` |
| 结果 log | `workspace/core4d/log/89_E069_first_tick_warmup_results.md` |
