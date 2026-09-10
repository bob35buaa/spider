# E214 — 核心方法四消融实验（Multi-Scale Contact + Part-wise Penetration）

- **Run**: R300  **Plan slot**: 245  **Log slot**: 303  **Phase**: 72
- **分支**: `experiment/E199-omniretarget-object-augmentation`（core4d 方向，无需新分支）
- **日期**: 2026-09-09 起

## Context / 目的

`report/0908/core4d_technical_summary.md` 提炼方法两大算法贡献：
1. **Multi-Scale Contact Reward** = 粗项 `contact_hdmi`（锚点欧氏吸引）+ 细项 `surface_band`（SDF 薄带贴合），同时叠加。
2. **Part-wise Penetration Constraints** = 同一 per-part 净空约束的两层执行：软层可微惩罚（reward）+ 硬层候选门（精英选择前剔除）。

E214 用**留一消融**验证每一支的必要性，数据集为论文 case 集。

## Claims（可证伪 + 量化）

统计基础：50 case 为样本分布，报 mean/std/**worst**（禁 cherry-pick）；对 full 做配对 Wilcoxon（p<0.05 显著）+ per-object 分层。

- **C1（A1 去细项 surface_band）**：接触贴合变差 / 几何穿透上升 —— `phys contact@3mm`↓ 且/或 `geom penetration@2mm`/`phys penetration@3mm`↑。
- **C2（A2 去粗项 contact_hdmi）**：够物能力下降 —— `eef pos err`↑ 且 `phys contact@3mm`↓（幅度应 > A1）。
- **C3（A3 去全部硬门）**：穿透显著失控 —— `penetration@3mm`/`geom@2mm`/`body-z p95`/`fall`↑（验证软层单独被 tracking 淹没）。
- **C4（A4 去全部软惩罚）**：收敛质量下降（tracking/contact 变差）但硬门仍压住极端穿透（penetration 不比 A3 更差）。
- **C5（综合）**：full 不被任一消融在全轴上超过。

**成功判据**：C1–C4 各在其目标轴出现预期方向、统计显著（或 mean+worst 明确）的退化 → 组件贡献成立；C5 成立 → 两贡献互补。反例记为诚实负结果。

## 设计（用户已定）

- **case 集 = 50**：论文 52 − 2 个 E167A box021（`box021_20231018_029_p2`、`box021_20231011_035_p1`）。box023 改用**全栈 E173**（`results/E173/.../box023_user_approved`，不重跑）→ 基线全为全栈。
- **200 CEM 重跑**（50×4），**基线不重跑**：full 列 = 论文缓存（43 non-box023）+ E173 全栈重算（7 box023）。
- **只新增**：不改任何基线 config/scene/npz。
- **A3 = 去全部硬候选剔除**（safety+hand+leg+posture 门；peak_margin 本就 off）。

### 四消融（叠加在 loaded 基线配置之上，单变量组）

| 消融 | toggle |
|---|---|
| A1 仅 contact_hdmi | `surface_band_rew_scale=0` `surface_band_penalty_scale=0` |
| A2 仅 surface_band | `contact_hdmi_gain=0` `contact_hdmi_ori_weight=0` |
| A3 仅软惩罚 | `cem_{safety,hand,leg,posture}_gate_enabled=false` `cem_peak_margin_enabled=false` |
| A4 仅硬候选门 | `robot_object_penalty_scale=0` `leg_object_penalty_scale=0` `hand_floor_penalty_scale=0` `e167_body_z_enabled=false` `e167_ground_z_enabled=false` |

### 机制（关键实现选择）

`run_mjwp.py load_config_path=<基线 config_act.yaml>` 加载基线**完整已解析配置**，CLI 覆写叠加其上 —— 无需跨 6 实验恢复 override_id，无需新建 override 文件，纯 additive。命令另覆写 `output_dir`（→E214 树）与 `model_path/data_path/contact_hdmi_mask_path`（基线 config 存的是别机绝对路径，重解析到本地）。已验证：加载无 schema 错（基线字段 ⊆ 当前 Config，无删除字段）；24 个 None→默认新字段均惰性/`legacy_box` 复现旧语义。

## 改动文件

| 文件 | 说明 |
|---|---|
| `tmp/E214_case_id.txt` | 50 case（新增） |
| `examples/run_mjwp.py` | **核心改动（隔离/可逆/guarded）**：接触 mask 预计算条件 `gain>0` → `gain>0 OR surface_band 需 contact_mask 门`。仅 A2（gain=0）生效，所有 gain>0 run 字节不变。 |
| `workspace/core4d/scripts/experiments/E214/` | e214_common / build_manifest / run_ablation_cem / audit_single_variable / snapshot_E214_scenes.sh / test_e214_contract |
| `workspace/core4d/scripts/eval/runners/eval_E214_ablation.py` | 15 指标评估（复用 eval.core.core_metrics + motion_health + fixed_reference_z_metrics） |
| `workspace/core4d/scripts/eval/wrappers/eval_E214_ablation.sh` | eval 入口 |
| `workspace/core4d/scripts/eval/reports/gen_E214_ablation_table.py` | full vs A1–A4 表（mean/std/worst + Wilcoxon，md/tsv/xlsx） |
| `workspace/core4d/scripts/launch/active/run_E214_local_8gpu.sh` | 本机 8 卡入口（snapshot→manifest→dispatch→audit） |
| `workspace/core4d/results/E214/scene_snapshot/` | 50 case 基线 scene 快照 + manifest（rule 7） |

## 运行命令

```bash
# 全流程（本机 8 卡，后台）
nohup bash workspace/core4d/scripts/launch/active/run_E214_local_8gpu.sh > logs/E214/run.log 2>&1 &
# 评估 + 出表
bash workspace/core4d/scripts/eval/wrappers/eval_E214_ablation.sh
```

## 已知 caveat（写入 log）

- **基线协议不对称**：full 列为论文人工择优 rollout（复用缓存），消融为该选定 config 的单次重跑；同 config/scene，结论以"消融相对 full 的退化方向"为准。
- **代码版本 confound**：基线 rollout 由旧代码生成，消融由当前代码；`object_distance_backend=legacy_box` 默认复现旧语义，confound 二阶。
- **A2 核心改动**：已用隔离 guarded 改动使 A2 用回精确 3cm mask（否则退化为 30cm 几何 mask）。

## 结果

（见 log/303，训练完成后填写：指标表、Claims 验证、可视化观察、结论）
