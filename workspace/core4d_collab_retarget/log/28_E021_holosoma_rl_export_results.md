# E021 Holosoma RL export results

日期：2026-05-21

## 目标

按 E026 full eval 的 best dynamic selection `spider_best_E018b_E022_E025`，把 13 个 Spider 动力学轨迹导出为 Holosoma RL 可直接加载的 `.npz` 序列。

参考计划：`workspace/core4d_collab_retarget/plan/22_E021_holosoma_rl_export_plan.md`。

## 输入

- Selector：`workspace/core4d_collab_retarget/results/E026_full_eval/best_dynamic_selection.csv`
- Manifest：`workspace/core4d_collab_retarget/scripts/export/manifest_rl.tsv`
- Source NPZ：`workspace/core4d_collab_retarget/results/{E018b,E022,E024,E025}/{selected_variant}_outdir/trajectory_mjwp.npz`
- Spider input fps：`60`
- Holosoma output fps：`50`

## 输出

| 类型 | 路径 |
|---|---|
| Holosoma RL NPZ（原始前缀命名） | `/home/ubuntu/Workspace/holosoma/workspace/data/spider_best_E018b_E022_E025_for_rl_rename/` |
| Spider shim inputs | `workspace/core4d_collab_retarget/results/E021_rl_export_manifest_rename/shim_inputs/` |
| Conversion log | `workspace/core4d_collab_retarget/results/E021_rl_export_manifest_rename/conversion_log.csv` |
| Partner log | `workspace/core4d_collab_retarget/results/E021_rl_export_manifest_rename/partner_log.csv` |
| Load verification CSV | `workspace/core4d_collab_retarget/results/E021_rl_export_manifest_rename/verify_load.csv` |
| Load verification JSON | `workspace/core4d_collab_retarget/results/E021_rl_export_manifest_rename/verify_load.json` |

## 结果

13/13 base 转换成功，13/13 load-level 验证通过。已额外生成 12 个双向 `_w_partner` 文件；`desk021_p1` 当前 best selection 没有对应 person2 源，因此不生成 partner 版。Holosoma 输出目录当前包含 25 个 `.npz`，总大小约 `30M`。

| Case | Output filename | Selected variant | Recommend | Frames |
|---|---|---|---|---:|
| box021_p1 | `20231018-030-person1-Box021_v2_mj_w_obj.npz` | E018b_box021_p1_canonical_t02 | no | 146 |
| box021_p2 | `20231018-030-person2-Box021_v2_mj_w_obj.npz` | E018b_box021_p2_canonical_t02 | no | 125 |
| box023_p1 | `20231008-045-person1-Box023_v2_mj_w_obj.npz` | E022_box023_p1_raw3_eval_axis | maybe | 226 |
| box023_p2 | `20231008-045-person2-Box023_v2_mj_w_obj.npz` | E018b_box023_p2_canonical_t02 | yes | 226 |
| box025_p1 | `20231011-048-person1-Box025_v2_mj_w_obj.npz` | E018b_box025_p1_canonical_t02 | maybe | 206 |
| box025_p2 | `20231011-048-person2-Box025_v2_mj_w_obj.npz` | E018b_box025_p2_canonical_t02 | yes | 206 |
| bucket001_p1 | `20231030-094-person1-bucket001_v2_mj_w_obj.npz` | E024_bucket001_p1_root025_gain2_stab_t065 | no | 178 |
| bucket001_p2 | `20231030-094-person2-bucket001_v2_mj_w_obj.npz` | E024_bucket001_p2_root025_gain2_stab_t065 | maybe | 203 |
| bucket005_s2_p1 | `20231002-004-person1-bucket005_v2_mj_w_obj.npz` | E018b_bucket005_s2_p1_canonical_t02 | no | 246 |
| bucket005_s2_p2 | `20231002-004-person2-bucket005_v2_mj_w_obj.npz` | E025_bucket005_s2_p2_penalty_s4_hc1 | no | 246 |
| bucket007_p1 | `20231020-055-person1-Bucket007_v2_mj_w_obj.npz` | E025_bucket007_p1_penalty_s4_hc1 | maybe | 201 |
| bucket007_p2 | `20231020-055-person2-Bucket007_v2_mj_w_obj.npz` | E018b_bucket007_p2_canonical_t02 | no | 158 |
| desk021_p1 | `20231008-007-person1-Desk021_v2_mj_w_obj.npz` | E018b_desk021_p1_canonical_t02 | maybe | 223 |

### Partner files

使用 `/home/ubuntu/Workspace/holosoma/workspace/v2/scripts/add_partner_hands_to_motion.py` 从 person2 的 `left_wrist_yaw_link` / `right_wrist_yaw_link` 提取 partner hand trajectory，并写入 person1 npz：

| Case | Partner output | Frames | Partner field shape |
|---|---|---:|---|
| box021_p1 | `20231018-030-person1-Box021_v2_mj_w_obj_w_partner.npz` | 146 | `(146, 2, 3)` |
| box021_p2 | `20231018-030-person2-Box021_v2_mj_w_obj_w_partner.npz` | 125 | `(125, 2, 3)` |
| box023_p1 | `20231008-045-person1-Box023_v2_mj_w_obj_w_partner.npz` | 226 | `(226, 2, 3)` |
| box023_p2 | `20231008-045-person2-Box023_v2_mj_w_obj_w_partner.npz` | 226 | `(226, 2, 3)` |
| box025_p1 | `20231011-048-person1-Box025_v2_mj_w_obj_w_partner.npz` | 206 | `(206, 2, 3)` |
| box025_p2 | `20231011-048-person2-Box025_v2_mj_w_obj_w_partner.npz` | 206 | `(206, 2, 3)` |
| bucket001_p1 | `20231030-094-person1-bucket001_v2_mj_w_obj_w_partner.npz` | 178 | `(178, 2, 3)` |
| bucket001_p2 | `20231030-094-person2-bucket001_v2_mj_w_obj_w_partner.npz` | 203 | `(203, 2, 3)` |
| bucket005_s2_p1 | `20231002-004-person1-bucket005_v2_mj_w_obj_w_partner.npz` | 246 | `(246, 2, 3)` |
| bucket005_s2_p2 | `20231002-004-person2-bucket005_v2_mj_w_obj_w_partner.npz` | 246 | `(246, 2, 3)` |
| bucket007_p1 | `20231020-055-person1-Bucket007_v2_mj_w_obj_w_partner.npz` | 201 | `(201, 2, 3)` |
| bucket007_p2 | `20231020-055-person2-Bucket007_v2_mj_w_obj_w_partner.npz` | 158 | `(158, 2, 3)` |

`desk021_p1` 未生成 partner 文件：E026 best selection 只包含 `desk021_p1`，没有同源 `desk021_p2`。

## 与 Holosoma v2 trimmed 目录差异

- Schema：base npz 与 v2 base 文件字段一致；`_w_partner` 文件比 base 多 `partner_hand_pos_w (T, 2, 3)` 和 `partner_hand_quat_w (T, 2, 4)`。
- 命名：v2 训练脚本常用 `${TAG}_v2_trimmed_mj_w_obj_w_partner.npz`；当前 spider 专用导出生成 `${TAG}_v2_mj_w_obj_w_partner.npz`，与 `/home/ubuntu/Workspace/holosoma/workspace/v2/scripts/train/train_core4d_v4.3-spider.sh` 当前查找路径一致。该 spider 训练脚本原有 `MOTION=...` 末尾多余引号已修复，`bash -n` 通过。
- 覆盖范围：v2 trimmed 目录只含少量 baseline case；当前目录是 E026 best dynamic selection 的 13 条，并额外给 6 对成对 person 生成双向 partner 版。
- 直接重名 overlap 的 Box025：当前修正后 `20231011-048-person1-Box025_v2_mj_w_obj.npz` 为 206 帧，v2 同名 base 为 205 帧；字段完全一致。数值不应期望相同，因为当前文件来自 Spider dynamics best trajectory，而 v2 是原 Holosoma conversion baseline。
- 原半帧问题已修复：Spider `trajectory_mjwp.npz` 的 `qpos` 形状是 `(control_ticks, ctrl_steps, 43)`，旧 shim 误把第二维当 env index 只取第 0 个子步，导致导出帧数约减半；现在改为 flatten 两个维度后再按 `60Hz -> 50Hz` 转换。

## 验证

命令：

```bash
MUJOCO_GL=egl .venv/bin/python workspace/core4d_collab_retarget/scripts/export/export_spider_best_to_rl.py --force
.venv/bin/python workspace/core4d_collab_retarget/scripts/export/verify_rl_load.py
.venv/bin/python workspace/core4d_collab_retarget/scripts/export/add_partner_hands_to_spider_rl.py --force
bash -n /home/ubuntu/Workspace/holosoma/workspace/v2/scripts/train/train_core4d_v4.3-spider.sh
```

`verify_rl_load.py` 检查通过：

- `fps == [50]`
- `joint_pos.shape == (T, 36)`
- `joint_vel.shape == (T, 35)`
- `object_pos_w.shape == (T, 3)`
- `object_quat_w.shape == (T, 4)`
- `body_pos_w.shape == (T, 52, 3)` for all current object models
- `joint_names` length `29`
- required RL fields complete

## 注意事项

- 当前导出默认保留 13 个 best-selection case；`manifest_rl.tsv` 中仍标注 `recommended_for_rl`，后续训练可用 `--recommended-only` 只导出 `yes/maybe`。
- `manifest_rl.tsv` 的 `original_prefix` 来自 CORE4D 原始 case 表 / Holosoma v2 replace-batch 数据；`bucket005_s2` 对应 `20231002-004-*`，不是 v2 baseline 示例中的 `20231002-003-*`。
- Holosoma converter 当前实际重采样依赖 CLI 的 `--input_fps`，因此 batch driver 显式传入 `--input_fps 60`；shim 仍写入 `qpos + fps` 两字段以保留输入元数据。
- 当前 shim 默认 `--layout flatten`，适配 MJWP 保存的 `(control_ticks, ctrl_steps, 43)` qpos；不要再用旧的 `env_index` 方式导出这批数据。
- 本轮只做 NPZ load-level 验证，未启动 Isaac/Holosoma RL trainer。
