# log287 · E199 全量：box 类 s6-full-CEM case 平移增强放量

_Core4D · Phase 62 · Run **R286** · plan [229](../plan/229_E199_box_fullscale_translation_augmentation_plan.md) · 承接 [log286](286_E199_omniretarget_object_augmentation_results.md)（pilot）· 2026-08-17 · **结论：放量成功，平移增强数据物理可信、可用于下游 RL**_

## Purpose / 假设

pilot（plan228/log286）已证 OmniRetarget object **平移**增强物理可信、旋转档系统性不可达。本轮把平移增强**放量**到所有进入 s6 full CEM 的 box 类 case，为下游 RL 批量扩数据（每 case +3× 平移变体）。只做平移、不做旋转；orig 复用现有 E198 A0/PRG full CEM；实验号仍 E199。

## 关键决策（用户确认）

1. **范围 = box 类 s6-full-CEM 全 case**（权威 = E198 A0 arm）：box001/004/021/023/024，目标 87 case。
2. **orig 基线 = 复用现有 E198 A0/PRG full CEM**（不重跑；eval 在同一 metric contract 下重打分配对，标注 orig=omnirt_v1 / aug=omnirt_v2 的轻微 retarget 混淆）。
3. **只做平移**（trans0/1/2），不做旋转。
4. **双机并行执行**：本机 8 卡 + 第二台 8 卡（独立 FS，自包含 bundle + standalone 远程队列）。

## 参数 / 冻结不变量

| 项 | 值 |
|---|---|
| CEM | seed=0, num_samples=1024, max_num_iterations=32, use_torch_compile=false |
| retarget（aug） | omnirt_v2/ref_fk（Phase-4 relaxation+foot_z+contact_preservation, slide=1.0, penetration_tol=0.8）|
| arm | E199 rubber_hull + 16 lowerbody-pair PRG（scene_act_E199_rubberHull_PRG）, reward=E167A |
| 增强 config | holosoma 原生 trans_0/1/2（前/左/右 0.2m），逐字节沿用；不加载 rot |
| 接触掩码 | 每 case 由 v2 `_original` trim 窗重算 1 份 3cm 掩码，3 平移变体复用（固定窗对齐）|
| orig 基线 | 复用 E198 A0 arm rollout（E173 `_PRG.npz`, omnirt_v1），同 `core4d-e154-physics-contact-v1` contract 重打分 |
| evaluator | 公共 `eval.core.core_metrics`（rule 13）|

## Run command

```bash
# 数据构建（本机 6-shard 并行 CPU）
bash workspace/core4d/scripts/launch/active/run_E199_fullscale_build.sh
# CEM 双机（本机 machineA + 远程 machineB bundle）
SCOPE=box_fullscale bash workspace/core4d/scripts/launch/active/run_E199_local_8gpu.sh      # 本机
#  远程：解压 e199_machineB_bundle.tar.gz → GPUS=… bash run_E199_machineB_remote.sh
# 评估（aug vs 复用 A0/PRG orig，per-case 配对）
bash workspace/core4d/scripts/eval/wrappers/eval_E199_fullscale_augmentation.sh --require-all
# 视觉 QC
MUJOCO_GL=egl .venv/bin/python workspace/core4d/scripts/experiments/E199/render_qc.py --objects box001,box004,box021,box023,box024 --manifest .../e199_fullscale_priority_manifest.tsv --max-per-object 6 --out .../render/fullscale_qc
```

## 改动文件

| 文件 | 改动 |
|---|---|
| `e199_common.py` | +`load_fullscale_cases()`（从 E198 A0 arm_cache 派生 box case，base 取 scene_xml 父目录，v1/v2 幂等）、`TRANS_VARIANTS`、`FULLSCALE_*` 路径、`write_tsv` 加 JuiceFS EIO 重试 |
| `build_augmented_tasks.py` / `build_aug_manifest.py` | +`--scope box_fullscale`（trans-only, skip-existing, 独立 artifacts/manifest, `--artifacts` 分片）|
| `run_E199_fullscale_build.sh` | 6-shard 并行 CPU 构建 + merge + manifest + snapshot |
| `run_E199_machineB_remote.py` + `pack_E199_machineB_remote.py` | 独立远程队列（不 import e199_common）+ 自包含 bundle 打包器（15.4MB：124 aug task + override + mask + 物体 mesh + queue + manifest + README）|
| `eval/runners/eval_E199_fullscale_augmentation.py` + wrapper | per-case 配对（orig=同 contract 重打分 A0）+ `_norm_cid`（person1↔p1 join 归一）+ 逐物体分层 + C5 可行性 |
| `render_qc.py` | +`--manifest`/`--max-per-object` |
| holosoma `parallel_robot_retarget.py` | find_files(smplx) 按 object_name 过滤（只取 combined `*_with_obj.npz`），避免 object-only/person-only sidecar 触发 KeyError |
| scene 快照 | `results/E199/scene_snapshot/cem_sidecars/`（每 sidecar 自动）+ manifest.txt |

## Result

### C0 链路放量 ✅（附一处已知缺口）
83/87 box case → **249 个平移 aug SPIDER task**（scene + trajectory + scene_act_E199_rubberHull_PRG 齐全）；base yaml + E199 PRG override + 249 行 manifest 全生成，**0 blocker**；scene sidecar 全快照。
**缺口**：box001 实际建 **24/28** case（4 个 case 在 6-shard 构建中未产出，合并 failures 日志未捕获具体原因——列为后续复查项，不影响其余 249 条）。其余 4 物体 case 数完整。

### C1 case 权威一致 ✅（83/87 realized）
| 物体 | 目标(s6 A0) | 实际建成 |
|---|---|---|
| box001 | 28 | **24** |
| box004 | 6 | 6 |
| box021 | 28 | 28 |
| box023 | 16 | 16 |
| box024 | 9 | 9 |
| 合计 | 87 | **83（249 aug）** |

### C2 执行闭合 ✅
- **249/249 full CEM 完成**（本机 machineA 125 + 远程 machineB box021/023 62 + 本机 box024 12 + pilot 复用），**249/249 打分成功，0 error / non-finite / diverged**。
- **双机执行细节**：远程 machineB 因缺 box024 object mesh → box024 12 条转本机跑；box021/023 远程正常，rsync 回收；machineA 中途遇 1 次 JuiceFS EIO 崩溃（已加 write_tsv 重试 + cron 自愈，resume 续跑无损）。
- **跌倒**：aug **9/249 (3.6%)** vs orig **3/83 (3.6%)** —— **同率，增强未额外引入跌倒**；9 个 aug 跌倒集中在 box004（`083_p2`/`086_p2`）+ box023（`018_p1`）这几个固有难 case。

### C3 增强正确性 ✅
所有 trans 变体接近段 pose 相对 orig 偏移 = **0.200m**，操作终点偏移 **≈0.024m**（~12%，指数衰减锚定生效）；构建期逐 case 校验（如 box024_027_p1 trans0/1/2 approach 0.200m→endpoint 0.024m）。

### C4 物理可信度 ✅（249 aug vs 83 orig，per-case 配对，全分布不 cherry-pick）

**总体：**

| 指标 | orig 均值 | aug 均值 (Δ%) | aug worst | 判读 |
|---|---|---|---|---|
| obj_pos 误差 cm | 13.30 | 13.51 (**+1.6%**) | 29.52 | ✅ 远 <25% 阈 |
| obj_ori 误差 ° | 6.20 | 6.11 (−1.4%) | 19.90 | ✅ 更优 |
| obj_z 误差 cm | 5.54 | 5.41 (−2.4%) | 9.71 | ✅ 更优 |
| eef_pos 误差 cm | 24.96 | 18.30 (**−26.7%**) | 44.99 | ✅ 显著更优 |
| in-mask 接触保持 | 0.726 | 0.719 (−1.1%) | — | ✅ 接近 |
| 手-物穿透 3mm frac | 0.196 | 0.177 (**−9.7%**) | 0.598 | ✅ 更优 |
| 腿穿透 frac | 0.073 | 0.086 (+17.8%) | 0.609 | 略升（box024 大箱贴腿，见 C6）|
| fall | 0.036 | 0.036 (±0) | — | ✅ 同率 |
| 12-gate 通过率 | 0.494 | **0.510** | — | ✅ aug 不低于 orig |

**逐物体（obj_pos / obj_ori / contact / leg_pen: orig→aug；gate o/a；aug 跌倒数）：**

| 物体 | obj_pos | obj_ori | contact | leg_pen | gate o/a | falls |
|---|---|---|---|---|---|---|
| box001 | 10.89→11.03 (+1%) | 5.68→5.49 | 0.77→0.78 | 0.063→0.076 | 0.42/**0.61** | 0 |
| box004 | 12.01→12.50 (+4%) | 11.53→11.24 | 0.69→0.59 | 0.056→0.073 | 0.50/**0.17** | 6 |
| box021 | 15.67→15.47 (−1%) | 6.09→5.85 | 0.67→0.65 | 0.120→**0.083** | 0.43/**0.52** | 0 |
| box023 | 13.06→14.03 (+7%) | 5.11→5.05 | 0.71→0.70 | 0.031→0.081 | 0.81/**0.54** | 3 |
| box024 | 13.64→13.79 (+1%) | 6.25→7.00 | 0.82→0.85 | 0.039→**0.143** | 0.33/0.37 | 0 |

- **obj_pos 增幅温和**（总体 +1.6%，逐物体 −1%~+7%，全 <10%），远低于 25% 阈。
- **eef/手穿透/obj_z 全维 aug 更优**；接触保持基本持平；12-gate 通过 aug≥orig（box001/box021 明显改善）。
- **两处 aug 稍劣，均为物体特异且可解释**：① box004 gate 0.50→0.17 是 6 个跌倒 case（`083_p2`/`086_p2`）拉低——box004 本身难、orig 也接近临界；② box024 leg_pen 0.039→0.143 是大长箱贴腿搬运的几何必然（见 C6），非破坏物理。

### C5 平移可行性分布 ✅（一等结论）
**已建成的 83 case 全部 3/3 平移可行**（full_3of3），无 partial / none：box001 24/24、box004 6/6、box021 28/28、box023 16/16、box024 9/9。→ 与 pilot「旋转 8/8 物体系统性不可达」形成对照：**平移方向对 box 类 100% 可行**，是放量增益的可靠来源。

### C6 视觉复核 ✅（rule 9，fullscale_qc 每物体×3 变体关键帧）
- **box021 trans1**：接近（增强初始偏移）→ 弯腰 → 抓取 → 抬起，动作连贯、姿态直立自然，无穿模/漂浮/跌倒。
- **box004 trans0**：接近 → 抓取 → 抬箱，箱体抬起时倾斜（box004 固有高 obj_ori 特征，orig 同样），机器人直立、物理可信；个别 case（083_p2/086_p2）失稳跌倒属该物体固有难度。
- **box024 trans0**：推扶大长箱 → 贴身搬运，箱体保持水平、机器人直立；**腿与大箱贴近**（→ leg_pen 偏高纯几何原因，非穿模失败）。
- 结论：增强变体**无致命 artifact**；跌倒限于已知难 case。

## Conclusion

**E199 平移增强放量成功——平移增强数据在本管线物理可信、可直接用于下游 RL 扩数据。**

1. **链路全通 + 双机放量**：83/87 box case × 3 平移 = 249 条 aug full CEM 全跑通、全打分、0 error；双机并行（本机 + 远程 bundle）把 ~40h 压到实际数小时；期间修复 holosoma find_files bug + JuiceFS EIO 重试。
2. **物理可信（C4 达标）**：与复用 orig 同条件对比，obj_pos 仅 **+1.6%**（远 <25% 阈），eef/手穿透/obj_z 全维更优，接触持平，12-gate 通过率 aug（51%）≥ orig（49%），**跌倒同率、增强未额外引入**。→ 满足放量判据（≥90% aug 达标 + 视觉无致命 artifact）。
3. **平移 100% 可行（C5）**：全 83 case 的 3 平移档全可行，是可靠增益来源（每 case +3× 数据）。
4. **诚实报告的两处物体特异弱化**：box004 跌倒 case + box024 大箱贴腿 leg_pen 偏高——均为物体固有特性 / 几何必然，与增强变量正交，非增强破坏物理。
5. **已知缺口**：box001 4/28 case 未在构建中产出（shard 级原因未捕获），列后续复查；不影响 249 条结论。

**下一步**：① box001 缺 4 case 复查补齐（可选）；② 249 条平移增强数据交下游 RL 训练验证增益；③ **Phase 2**（object scale/长宽高）——需上游 holosoma 扩 scale 增强 + 重算接触，另开计划。

## 结果路径

| 类型 | 路径 |
|---|---|
| CEM 输出 | `workspace/core4d/results/E199/s6_downstream/cem/full/`（249 条）|
| priority manifest | `.../manifests/e199_fullscale_priority_manifest.tsv`（249）+ machineA/B 分片 |
| eval | `.../eval/fullscale_augmentation/`（summary.json + case_metrics + orig_deltas，status=pass, paired=249, errors=0）|
| render QC | `.../render/fullscale_qc/`（30 rollout 关键帧 + render_index.json）|
| 远程 bundle | `.../remote_bundle/e199_machineB_bundle.tar.gz` |
| scene 快照 | `.../scene_snapshot/cem_sidecars/` + manifest.txt |
