# E214c — Part-wise Penetration 阈值统一消融（贡献2「分部位阈值」必要性）

- **Run**: R303  **Plan slot**: 248  **Log slot**: 307  **Phase**: 72（E214 核心方法消融系列延续）
- **分支**: `experiment/E199-omniretarget-object-augmentation`（core4d 方向，无需新分支）
- **日期**: 2026-09-18 起
- **前置**: [plan/245 E214](245_E214_core_method_four_ablation_plan.md) · [log/303 E214](../log/303_E214_core_method_four_ablation.md) · `report/0908/core4d_technical_summary.md` §2

## Context / 目的

E214 用留一消融验证了方法两大贡献的四支（A1 去 surface_band、A2 去 contact_hdmi、A3 去**全部**硬门、A4 去**全部**软惩罚），全部 5 条 claim 成立。但 A3/A4 是把穿透约束**整体拆掉**，**没有触碰贡献2 真正的「part-wise 实质」——分部位阈值本身**（`technical_summary` §2.2：「阈值本身编码部位差异：leg 严禁碰、hand 允许贴」）。

本实验补**两条新消融臂**，直接检验「三个部位用不同阈值」是否必要：把 body/hand/leg 三个 SDF 硬门的阈值**统一**成同一口径，两个方向各一条：

- **A5 uniform_hand**：三部位全用**手**阈值（最宽松，允许接触级浅穿）→ 预期下肢/身体穿透失控。
- **A6 uniform_leg**：三部位全用**腿**阈值（最严格，必须净空 +5mm）→ 预期手被排斥、无法贴物、接触崩塌。

若 part-wise 必要，则统一任一方向都应在对应轴上显著退化，而 full（分部位）同时守住「手能贴」与「腿/身净空」两轴。

### 关键发现（决定本实验设计，已与用户确认）

对 50 case 全量核查基线 `config_act.yaml`（11 个 object 组各采样，跨 E170/E173/E198/E178/E207/E212 源实验）：

1. **软惩罚本就部位统一**：`robot_object`/`leg_object`/`hand_floor` 全部 `scale=2.0, margin=2cm`，`hand_object_deep=0.0(off)`；hand 对**物体**无软惩罚（只有 hand↔floor）。→ **「统一软阈值」是 no-op**。part-wise 差异**只在硬门**。**用户决定：软层保持不变，本实验只动硬门。**
2. **硬门确为分部位、非对称**（这是 part-wise 实质）：

   | 部位(门) | min_sdf θ | hard_floor | max_vio% | min_valid_frac / fallback |
   |---|---|---|---|---|
   | body (`cem_safety_gate`) | **−5mm** | NaN(=θ) | 0% | 0.02 / least_violation |
   | hand (`cem_hand_gate`) | **−10mm** | **−15mm**¹ / **−20mm**² | 5%¹ / 10%² | （无该字段，走全局） |
   | leg (`cem_leg_gate`) | **+5mm** | **−5mm** | 2% | 0.02 / least_violation |

   ¹ E198 源（box001/box004/box024）；² 其余源（box021/box023/bucket003/bucket007/chair006/desk007/desk021/desk023）。**hand 的 floor/vio 因源实验而异 → A5 必须 per-case 拷贝各自 hand 值，不能用全局定值。** leg/body 阈值跨 case 完全一致。
3. **评估器已支持逐部位穿透**：`eval.core.core_metrics` 已含 `leg_penetration_frac`、`body_penetration_frac`、`upper_body_penetration_frac`、`hand_geom_penetration_2mm_frac`、`hand_object_physics_contact_3mm_frac` 等 → part-wise claim 可直接量化，无需新增 metric。

## Claims（可证伪 + 量化）

统计基础：50 case 为样本分布，报 mean/std/**worst**（禁 cherry-pick）；对 full 做配对 Wilcoxon（p<0.05 显著）+ per-object 分层。协议与 E214 完全一致。

| Claim | 最低证据 |
|---|---|
| **C1（A5 uniform_hand → 下肢/身体穿透失控）** | body/leg 门放宽到手级后：`leg_penetration_frac`↑ **且/或** `body_penetration_frac`/`upper_body_penetration_frac`↑、`geom penetration@2mm`↑（p<.05 或 mean+worst 明确）；可伴 `body-z err p95`↑ / `fall`↑。**手接触不变**：`hand_object_physics_contact_3mm_frac` ≈ full（hand 门未动，作为内部对照）。 |
| **C2（A6 uniform_leg → 手接触崩塌）** | body/hand 门收紧到 leg 级(+5mm)后：`hand_object_physics_contact_3mm_frac`↓↓、`raw contact`↓、`eef pos err`↑（手贴不上/被排斥，类 A2「够不到」但经由**过严硬门**）；`hand_geom_penetration`↓（手不穿是因没接触，非改善）。body/leg 穿透**不升**（门更严）。 |
| **C3（part-wise 必要性 / 综合）** | full 在「手接触」与「下肢/身体穿透控制」**两轴同时**优于任一统一口径：A5 牺牲腿/身穿透控制（手接触无收益），A6 牺牲手接触（腿/身穿透本就被软层压住、无收益）。→ 统一阈值无法同时满足手的「贴」与腿/身的「净空」，**分部位阈值必要**。 |
| **C4（与 A3/A4 正交互补）** | A5/A6 提供 A3(去全部硬门)/A4(去全部软惩罚)未覆盖的信息：A5 隔离「腿/身门**严格度**」、A6 隔离「手门**宽松度**」；方向相反、失败模式不同（A5 穿透失控 vs A6 接触崩塌），共同支撑「每部位阈值按其角色调谐」。 |

**成功判据**：C1–C4 各在其目标轴出现预期方向、统计显著（或 mean+worst 明确）的退化 → part-wise 阈值贡献成立。**诚实负结果同样有价值**：若 A5 腿/身穿透不升（说明软层 2cm 推离已足够、硬门严格度对腿/身冗余），或 A6 手接触不崩（说明 +5mm 门未真正束缚手），记为方法边界，写入 log 与论文 limitation。

## 设计（用户已定）

- **case 集 = 50**：直接复用 `tmp/E214_case_id.txt`（= 论文 52 − 2 个 box021 E167A）。与现有 A1–A4 消融表**同集**，可并入对比。
- **100 CEM 重跑**（50×2），**基线/full 不重跑**：full 列复用 E214 的 full（论文缓存 43 + E173 重算 7）。
- **只动 3 个 SDF 硬门的阈值**：`cem_safety_gate`(body) / `cem_hand_gate` / `cem_leg_gate` 的 `min_sdf_m` + `hard_floor_m` + `max_violation_pct`（**统一全部三参数**，用户定）。
- **不动**：所有 `*_gate_enabled`（保持 true）、`min_valid_frac`/`fallback`（门机制非阈值，保持基线）、posture 门、e167 body-z/ground-z、peak_margin(off)、全部软惩罚、接触奖励、物体驱动。
- **零核心代码改动**：A5/A6 只翻转既有 config 键值（门逻辑已实现），`examples/run_mjwp.py` **不变**（比 E214 更干净——E214 为 A2 改过 mask 预计算）。

### 两消融（per-case 计算 toggle，叠加在 loaded 基线之上）

**A5 uniform_hand**（body/leg ← 该 case 自己的 hand 阈值；hand 门不动）：

| toggle key | 值（来源＝该 case 基线 hand 门） |
|---|---|
| `cem_safety_gate_min_sdf_m` | = hand min_sdf = **−0.01** |
| `cem_safety_gate_hard_floor_m` | = hand hard_floor = **−0.02**² / **−0.015**¹ |
| `cem_safety_gate_max_violation_pct` | = hand max_vio = **0.1**² / **0.05**¹ |
| `cem_leg_gate_min_sdf_m` | = hand min_sdf = **−0.01** |
| `cem_leg_gate_hard_floor_m` | = hand hard_floor = **−0.02**² / **−0.015**¹ |
| `cem_leg_gate_max_violation_pct` | = hand max_vio = **0.1**² / **0.05**¹ |

**A6 uniform_leg**（body/hand ← leg 阈值；leg 门不动。leg 值跨 case 一致）：

| toggle key | 值（来源＝leg 门，全 case 相同） |
|---|---|
| `cem_safety_gate_min_sdf_m` | = leg min_sdf = **+0.005** |
| `cem_safety_gate_hard_floor_m` | = leg hard_floor = **−0.005** |
| `cem_safety_gate_max_violation_pct` | = leg max_vio = **0.02** |
| `cem_hand_gate_min_sdf_m` | = leg min_sdf = **+0.005** |
| `cem_hand_gate_hard_floor_m` | = leg hard_floor = **−0.005** |
| `cem_hand_gate_max_violation_pct` | = leg max_vio = **0.02** |

> **per-case 自洽**：A5 的 hand 阈值随源实验有 floor/vio 两变体，故 toggle 值由 `build_manifest` **逐 case 读取该 case 基线 config 后计算**（非 E214 式固定 dict）。A6 leg 值全 case 相同，但同走 per-case 路径以统一审计。

### 机制（复用 E214）

`run_mjwp.py load_config_path=<基线 config_act.yaml>` 加载该 case 完整已解析配置，CLI 覆写 6 个目标键叠加其上，另覆写 `output_dir`(→E214c 树) 与 `model_path/data_path/contact_hdmi_mask_path`(基线存别机绝对路径，重解析到本地)。baseline 解析直接 import `e214_common.baseline_paths`（与 E214 同源，保证同基线）。

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `tmp/E214c_case_id.txt` | 新增＝`cp tmp/E214_case_id.txt`；contract test 断言两文件逐行相同 |
| 2 | `workspace/core4d/scripts/experiments/E214c/e214c_common.py` | 新增。import `e214_common`（baseline 解析/hydra_token/路径）；定义 `ABLATIONS={A5_uniform_hand,A6_uniform_leg}`；核心新函数 `per_case_toggles(case_id, ablation)`：读该 case 基线 config 的 hand/leg 门三参数，返回 6 个 toggle |
| 3 | `.../E214c/build_manifest.py` | 新增。逐 (case,ablation) 解析基线 config + 计算 per-case toggle，写 manifest（含 toggle 值 + 基线 hand/leg 门值供审计） |
| 4 | `.../E214c/run_ablation_cem.py` | 新增。从 manifest 读 per-case toggle 派发 CEM（复用 E214 runner 模式：flock 单实例、per-task timeout、cem_ok 判定） |
| 5 | `.../E214c/audit_single_variable.py` | 新增。断言每个 A5/A6 config 相对基线**只**改了 6 个目标键，其余逐字一致 |
| 6 | `.../E214c/snapshot_E214c_scenes.sh` | 新增。50 case 基线 scene XML 快照 + manifest（git HEAD + sha256，rule 7） |
| 7 | `.../E214c/test_e214c_contract.py` | 新增。50 case baseline 全可解析、per-case toggle 计算正确、case 集＝E214、smoke 确认 safety 门接受非 NaN hard_floor |
| 8 | `workspace/core4d/scripts/eval/runners/eval_E214c_ablation.py` | 新增。`eval.core.core_metrics` 评估 100 新 npz；full 列复用 E214 eval 缓存 |
| 9 | `workspace/core4d/scripts/eval/wrappers/eval_E214c_ablation.sh` | 新增。eval 入口 |
| 10 | `workspace/core4d/scripts/eval/reports/gen_E214c_ablation_table.py` | 新增。出 6 臂表 `full｜A1｜A2｜A3｜A4｜A5-uniform-hand｜A6-uniform-leg`（mean/std/worst + 配对 Wilcoxon，md/tsv/xlsx），**突出逐部位穿透**(leg/body/upper_body_penetration_frac, hand_geom_penetration_2mm_frac) + 接触(hand_object_physics_contact_3mm_frac) + eef pos err + fall + body-z |
| 11 | `workspace/core4d/scripts/launch/active/run_E214c_local_8gpu.sh` | 新增。本机 8 卡入口（snapshot→manifest→dispatch→audit） |
| 12 | `workspace/core4d/results/E214c/` | 产出树：cem/ manifests/ eval/ reports/ scene_snapshot/ preflight/ |

> **不改** `examples/run_mjwp.py`、不改任何基线 config/scene/npz、不改 `spider/config.py`。纯 additive。

## 训练命令

```bash
# 0) 跑 CEM 前先 kill GPU 占卡脚本（见 memory: gpu-keepalive-script），跑完挂回
# 1) 全流程（本机 8 卡，后台）
mkdir -p logs/E214c
nohup bash workspace/core4d/scripts/launch/active/run_E214c_local_8gpu.sh > logs/E214c/run.log 2>&1 &
# 2) 评估 + 出表
bash workspace/core4d/scripts/eval/wrappers/eval_E214c_ablation.sh
# 3) 可视化（rule 9）：代表 case 渲染 + /video-frames 抽帧
python workspace/core4d/scripts/experiments/E214c/render_E214c.py   # 复用 E214 render 模式
```

**计算预算**：100 CEM 重跑（50×2）。E214 200 任务用时 19.35h（本机 8 卡，wall median 43min）→ 预计 **~10h**（8 卡）。可选：本机 + 远程 2 卡分担（`run_E214c_remote.sh`，按 GPU 均分、同 GPU 串行）。

## 评估（rule 9 可视化强制 + rule 13 公共评测）

- **量化**：`eval.core.core_metrics.evaluate_sequence` + `motion_health.run_health` + `fixed_reference_z_metrics`，与 E214 同定义。**重点逐部位指标**：
  - C1(A5)：`leg_penetration_frac`、`body_penetration_frac`、`upper_body_penetration_frac`、`geom penetration@2mm`、`body-z err p95`、`fall`；对照 `hand_object_physics_contact_3mm_frac`(应≈full)。
  - C2(A6)：`hand_object_physics_contact_3mm_frac`、`raw contact`、`eef pos err`、`hand_geom_penetration_2mm_frac`；对照 `leg/body_penetration_frac`(应不升)。
- **可视化**（强制）：A5/A6 各渲代表 case（box021 抱箱搬运 + 一个 bucket/desk），`/video-frames` 抽关键帧。
  - A5 查：下肢/躯干是否**扎进箱体**（leg/body 穿透）、是否失稳摔倒。
  - A6 查：手是否**被推开、贴不上物体**、抓取失败（类 A2 够不到）。
  - 与 full + ref 侧边对比；关键帧写 log「实际观察」，不得留空。

## 成功标准

| 指标 | full (E214 基线) | A5 uniform_hand 目标 | A6 uniform_leg 目标 |
|------|------------------|----------------------|---------------------|
| `leg_penetration_frac` / `body_penetration_frac` | 低（净空守住） | **↑ 显著**（腿/身可穿） | ≤ full（门更严） |
| `hand_object_physics_contact_3mm_frac` | **58.8%**（E214） | ≈ full（手门未动） | **↓↓ 显著**（手被排斥） |
| `eef pos err (cm)` | 13.07（E214） | ≈ full | **↑ 显著**（贴不上） |
| `fall rate` | 0.0%（E214） | 可 ↑（腿扎箱失稳） | 0.0%（门更严） |
| GUI A5 | — | **腿/躯干扎进箱体** | — |
| GUI A6 | — | — | **手离开物体、抓取失败** |
| 单变量审计 | — | 100/100 ALL PASS（只改 6 键） | 同 |
| CEM 完成 | — | 50/50 `cem_ok`（失败记 failure_mode 不静默丢） | 同 |

## Scene 快照（rule 7 双重保障）

训练脚本第一步调 `snapshot_E214c_scenes.sh`，把 50 case 基线 scene XML 复制到 `results/E214c/scene_snapshot/` + `manifest.txt`(git HEAD + 每文件 sha256)。scene 与 E214 基线相同（本实验不改 scene），但仍独立快照随 log commit。log「改动文件」表列出 `scene_snapshot/` 路径。

## 已知 caveat（写入 log）

1. **基线协议不对称**：full 列复用 E214 的 full（论文人工择优缓存 43 + E173 重算 7），A5/A6 为单次重跑；同 config/scene，结论以「消融相对 full 的退化方向 + 配对显著性」为准。
2. **代码版本 confound**：基线 rollout 旧代码生成，A5/A6 当前代码；`object_distance_backend=legacy_box` 复现旧语义，confound 二阶。
3. **per-case hand 阈值差异**：A5 在 E198(box001/004/024) 用 floor−15mm/vio5%，其余用 floor−20mm/vio10%；per-case 自洽拷贝各自 hand 值，保证与该 case 基线手门一致（不用全局定值）。
4. **A6 手门过严风险**：hand min_sdf=+5mm 与接触奖励（surface_band 奖励 SDF∈[−1,+3]mm）直接冲突 → 候选可能大面积被否决 → `least_violation` 回退兜底（永不出空精英）；若某 case `cem` 失败，按 E214 方式记 `failure_mode`，不静默丢弃。这本身是 C2 的预期证据（手被门排斥）。
5. **safety 门 hard_floor 由 NaN→实值**：A5/A6 给 safety 门设真实 hard_floor（基线为 NaN=legacy）；contract smoke 先确认 GatedElite 对 safety 门接受非 NaN floor 无 schema/逻辑错（hand/leg 门已用实值 floor，路径同源）。

## 结果

（见 log/307，训练完成后填写：6 臂指标表、C1–C4 逐条验证、A5/A6 可视化关键帧观察、与 A3/A4 的对照结论、caveat 复盘）
