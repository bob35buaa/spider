# E194 结果：物体重力补偿 2×2（机制 (b)）——G1 重力补偿成立，G2/G3 硬伺服发散

_Core4D · Phase 57 · 承接 [E191](266_E191_object_support_offline_audit_results.md) · 计划 [plan/220](../plan/220_E194_object_gravity_compensation_plan.md)_

## TL;DR

- **G1（`gravcomp="1"`，kp 保持 500）成立且安全**：修好平移下垂、object tracking 变好、承重接触保住、穿透下降；远端倾斜如预测**修不好**（质心施力无力矩）。
- **G2/G3（`init_pos_actuator_gain=2500`，5×）在 15/15 例全部数值发散**：物体被甩飞（位置误差均值 233/237 cm，最坏 373 cm），承重接触塌到 0.06。其"穿透下降"是假象（物体没了自然不穿透）。命中计划预登记的 stop-loss 担忧（失重物体+硬弹簧发散风险）。
- **C6 主判别（穿透来自"位置错"还是"伺服顶"）无法回答**：2×2 需要一个"修好位置又不发散"的 G2，但 G2 发散了。硬伺服 kp=2500 路线**不可行**；若要判 C6 需更温和的 kp（如 1000）作后续实验。

## 设计（冻结不变量见 plan/220）

- A0 基线复用 E172(box004×6)/E173(box024×9) 落盘；G1/G2/G3 各 15 例，共 **45 条 Full CEM**（1024×32，seed 0），本机 GPU **4-7** 并行。
- 四臂仅差 run_mjwp CLI：G1 `scene_name=scene_act_E194_rubberHull_PRG_gravcomp`；G2 `init_pos_actuator_gain=2500`；G3 两者。`init_rot_actuator_gain` 三臂全保持 50（C3 要求）。PRG 全程开启。
- G1/G3 的 gravcomp sidecar = base PRG sidecar 仅在 object body 注入 `gravcomp="1"`（`tree_signature` 逐字段校验 + 编译后 model 除 `body_gravcomp[object]` 外全等 A0）。审计 45/45 通过。

## 结果（逐物体，禁止合并）

| 指标 | A0 | **G1 gravcomp** | G2 kp2500 | G3 两者 |
|---|---:|---:|---:|---:|
| object 位置误差 cm (mean, n=15) | 13.0 | **11.2** | 232.7 | 236.6 |
| object 位置误差 cm (max) | 23.3 | **16.9** | 373.4 | 375.7 |
| 抬起帧 z 误差 m (box024) | −0.101 | **−0.022** | 0.688 | 0.774 |
| 3mm 承重接触占比 (mean) | 0.365 | **0.423** | 0.062 | 0.067 |
| 3mm 穿透帧占比 (mean) | 0.285 | **0.232** | 0.138* | 0.147* |
| obj_side_z_asym cm (box024, G1) | 7.33(基线) | 1.74 | −24.8* | −24.7* |
| qpos_jerk_l2_p95 (box024) | 3037 | 2228 | 2523 | 2405 |

\* G2/G3 已发散，这些数值是物体飞离后的假象，不可采信。

### 预注册 claim 判定（仅 G1 有效臂）
- **C1（下垂可修）**：G1 抬起帧 |z|=0.022m ≈ 0 ✅（G2/G3 因发散不适用）
- **C2（下垂只是误差一小部分，复核 E191 H1）**：G1 pos_err 降幅 box024 15.5%、box004 11.5%，落在 z-share≈0.35 预期内 ✅ —— 印证"修好伺服≠解决 object tracking"
- **C3（质心施力修不好远端倾斜）**：G1 asym 仍 ~1.7cm ✅（倾斜基本保留）
- **C4（承重接触不消失，核心风险）**：G1 3mm 接触 0.397(box024)/0.463(box004) ≥ 基线 ✅ —— 重力补偿**没有**变成"虚扶"
- **C6/C7（主判别 / 交互项）**：❌ **无法判定** —— G2 发散，2×2 塌成"G1 vs 发散"
- **C9（数值稳定）**：G1 无新增发散、jerk 下降 ✅；G2/G3 ❌ 全发散

判决：`SAG_FIXABLE_TILT_NOT`（G1 侧成立）+ 硬伺服路线 `STIFF_SERVO_DIVERGES`。

## 视觉确认（强制）

`results/E194/.../render/full/keyframes/box024_20231011_026_p1_A0_G1_G2_G3_4cell.png`：
- A0 行：箱子抬起时明显倾斜/下栽；G1 行：箱子全程保持水平、贴合参考；**G2/G3 行：lift 之后箱子翻滚→飞到画面右上角→消失**（发散实锤）。

## 复现入口

```bash
.venv/bin/python workspace/core4d/scripts/experiments/E194/build_gravcomp_manifest.py --apply --snapshot
.venv/bin/python workspace/core4d/scripts/experiments/E194/audit_gravcomp_scenes.py --require-all   # 45/45
MODE=full bash workspace/core4d/scripts/launch/active/run_E194_local_8gpu.sh                        # GPU 4-7
bash workspace/core4d/scripts/eval/wrappers/eval_E194_gravcomp_arms.sh full
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E194_arm_comparison.py full
bash workspace/core4d/scripts/launch/active/run_E194_render_all.sh
```

## 踩坑记录（已修入脚本）
1. `workspace/core4d/results` 是跨挂载软链接 → `repo_path().resolve()` 逃出 REPO 致 `rel()` 抛错；改 `repo_path` 不 resolve、`rel` 优雅回退。
2. eval/report 的 `sys.path` 用了 `parents[3]` 应为 `parents[2]`（`import e194_common` 失败）。
3. `import mujoco` 在 `MUJOCO_GL` 未设时于本机死锁；eval wrapper 改 `export MUJOCO_GL=osmesa` + 线程上限（避免 192 核 BLAS 127 线程抖动）。

## 产物路径（**结果目录 gitignored + 软链接外挂，未随 git 提交**）
- `results/E194/s6_downstream/cem/full/` 45 条 CEM
- `results/E194/s6_downstream/eval/full/{e194_arm_case_metrics.tsv, e194_paired_deltas.tsv, e194_arm_diff_summary.json, E194_arm_comparison.md}`
- `results/E194/s6_downstream/render/full/` 45 MP4 + `keyframes/*_4cell.png`
- `results/E194/scene_snapshot/` 15 例 base+gravcomp XML 快照（Safeguard 2）
- git 内已 force-add 15 个 `scene_act_E194_rubberHull_PRG_gravcomp.xml`（Safeguard 1）

## 后续
- 若要回答 C6 机制判别：跑一个温和 kp（~1000）臂，"修位置又不发散"，才能把"位置错 vs 伺服顶"分开。
- 报表脚本可加发散 guard（object 误差 > 50cm 判 DIVERGED 并从 C6 剔除），避免发散指标污染 claim 判定。
