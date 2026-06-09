# E150 — 路线A：接触锚点 eef_offset 扫参（橡胶手 CEM）

## 背景与动机

分析报告 `report/contact_anchor_misalignment_analysis.md`（memory `project_contact_anchor_misalignment`）定位了 E147-E149 "换橡胶手后穿透↓但接触不涨" 的根因：

`contact_hdmi` 接触奖励的锚点 = `wrist_yaw_link 原点 + R(wrist_quat)·contact_hdmi_eef_offset`，默认 `eef_offset=[0.05,0,0]`（`config.py:195-197`，注释 `# wrist→palm`）。这个点**埋在橡胶手根部**（wrist 局部系 x：锚点 0.05 < 掌心 site 0.08 < 球心 0.10 < 橡胶手 mesh bbox 末端 0.173）。CEM 把这个埋在手内部的点往箱面拽 → 真实掌/指必然插进箱子（穿透↑）；而奖励**全程不读手面**（不碰任何 geom/mesh/SDF，`mjwp.py:1031-1044`）→ 接触保真度上不去。

**关键杠杆**：参考侧（`run_mjwp.py:1016-1038`，`target_uses_eef_offset=true`）和机器人侧（`mjwp.py:1031-1044`）**共享同一个 `config.contact_hdmi_eef_offset` 字段**，改这一个值两侧自动同步平移、对称性天然保持。

**路线A** = 纯 config 把锚点从 0.05 沿手指轴前移到橡胶手实际接触面，验证 near-band 接触能否上升、穿透能否同时下降。**不改 SPIDER reward/算法代码**，offset 走 Hydra CLI override。

## 已定决策（用户拍板 2026-06-09）

1. **扫值只跑 0.08 / 0.11**（0.05 不重跑）。0.11 仍在 mesh bbox(0.173)内，安全；不扫 0.14+。
2. **0.05 基线复用 E148 rubber-0.05 结果**（`rubber_outdir_npz`），不同批次、CEM 随机，评测时明确标注混淆风险。
3. **benchmark = E149 `relaxed8_valid_like` 8 case**。
4. → 新 CEM 跑量 = 8 case × 2 offset = **16 runs**。

## 注入点（已核实）

单 CEM 命令（照 `train_E147_rubber_hand_collision.sh:113-117`）：
```bash
CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
  +override=<rubber_hull override> task=<task> +use_torch_compile=false video_camera=auto \
  output_dir=<out> video_output_path=<out>.mp4 \
  'contact_hdmi_eef_offset=[0.08,0.0,0.0]'        # ← 本实验新增（已存在字段不加 +，引号防 shell glob）
```
橡胶 scene 由 override YAML 内 `scene_name: scene_act_E14{7,8}_rubber_hull` 选定（不走 CLI）。8 case 的 rubber_hull override **全部已存在**，未设 eef_offset（继承默认 0.05），`contact_hdmi_target_uses_eef_offset: true`（基链已设）→ 扫参只需 CLI 覆盖，**不改任何 override YAML**。

8 case → 现有 rubber override + 任务 + 0.05 baseline npz（取自 `workspace/core4d/scripts/E148/variants.tsv`）：

| case | derived_task | override | rubber-0.05 baseline npz (rubber_outdir_npz) |
|---|---|---|---|
| box021_035_p1 | d003_box021_20231011_035_p1_e107_clean | core4d_E147_d003_box021_20231011_035_p1_rubber_hull | E148 rubber_outdir |
| box021_035_p2 | d003_box021_20231011_035_p2_e107_clean | core4d_E147_d003_box021_20231011_035_p2_rubber_hull | E148/E147 rubber_outdir |
| box021_029_p2 | d003_box021_20231018_029_p2_e107_clean | core4d_E148_box021_029_p2_rubber_hull | E148 rubber_outdir |
| box004_082_p1 | e091_box004_20231003_2_082_p1_e096b_mask_cem | core4d_E147_e091_box004_20231003_2_082_p1_rubber_hull | E148/E147 rubber_outdir |
| box004_083_p1 | e091_box004_20231003_2_083_p1_e096b_mask_cem | core4d_E147_e091_box004_20231003_2_083_p1_rubber_hull | E148/E147 rubber_outdir |
| box004_083_p2 | e091_box004_20231003_2_083_p2_e092_dyn | core4d_E148_box004_083_p2_rubber_hull | E148 rubber_outdir |
| box023_person2 | box023_person2_legobj | core4d_E147_box023_person2_rubber_hull | E148/E147 rubber_outdir |
| box026_139_p1 | e091_box026_20231023_139_p1_e106_clean | core4d_E148_box026_139_p1_rubber_hull | E148 rubber_outdir |

（rubber-0.05 baseline 的具体 npz 路径由 manifest builder 从 E148 variants.tsv 的 `rubber_outdir_npz`/`rubber_npz` 列解析填入。）

## 实现（照 E147/E148 四件套，新建 E150，限 8 case + offset 维度）

### A. Manifest + scene 快照 — `scripts/E150/build_eef_offset_sweep_manifest.py`
照 `E147/build_rubber_hand_collision_manifest.py` 结构，但**直接从 `scripts/E148/variants.tsv` 取 8 行**（override/task/object_key/rubber_scene_act/rubber baseline npz 都现成，无需重新 patch scene）：
- 笛卡尔展开 offset ∈ {0.08, 0.11} → `variants.tsv` **16 run 行** + 8 行 baseline-ref（offset=0.05，标 `reuse_e148`，不跑只评测）。
- variant 命名 `E150_<case>_off{08,11}`；列含 `eef_offset_x`、`override`、`derived_task`、`object_key`、`split`（gpu0/gpu1 各 8）、`rubber_scene_act`、`scene_name`、`baseline_npz`(rubber-0.05)。
- scene 复用既有 rubber_hull sidecar（**不重新 patch**）；snapshot 进 `results/E150/scene_snapshot/` + `manifest.txt`（git HEAD + sha256），满足 experiment.md §7。

### B. Train 脚本 — `scripts/train/train_E150_eef_offset_sweep.sh`
照 E147：modes `list|single|remote-gpu0|remote-gpu1`，stage `smoke|full`。在 run_mjwp 命令尾注入 `'contact_hdmi_eef_offset=[${eef_offset_x},0.0,0.0]'`；out_dir/npz/mp4 名带 `_off{08,11}`；first step 调 snapshot helper。

### C. 远端执行 + 回收 — `scripts/run_E150_remote.sh` / `scripts/pull_E150_remote_results.sh`
照 E147：rsync override（已存在）+ task dir + object asset + rubber scene 到 `spider-remote`；tmux 2 卡并行（每卡 8 runs）；回收 npz/mp4/log。

### D. 评测 — `scripts/eval/eval_E150_eef_offset_sweep.py`
复用 `eval/eval_E147_rubber_hand_collision.py` 的度量逻辑，但**关键修正**：
- **参数化 `EEF_OFFSET`**（现 `eval_E147...py:23,363` 硬编码 `[0.05,0,0]`）：EEF near-band 指标须用各 run 训练时的 offset，否则量错。hand-geom SDF 指标 offset-无关，保持。
- 三方对比每 case：baseline(rubber-0.05) vs off-0.08 vs off-0.11。
- 指标（照 E147 METRIC_FIELDS）：接触 `hand_geom_near_{3,5,8,10}cm_frac` + `eef_near_*`（各自 offset）+ `hand_object_physics_contact_frac`；穿透 `hand_geom_penetration_frac`/`deep_2cm`；稳定 pelvis/fall、obj_err、leg_pen、object_floor。
- 输出 `results/E150/.../e150_method_metrics.tsv` + `e150_offset_delta.tsv`（off−baseline）+ summary.md + xlsx（逐 case + 均值 + 按物体分组）。

### E. 记录
- `log/<N>_E150_eef_offset_sweep_results.md`：三方(0.05/0.08/0.11)×8 case 双指标表、按物体分组、视频、结论。
- `EXPERIMENT_TRACKER.md` 加 E150 行。
- memory `project_contact_anchor_misalignment`：从"机制假设"升级到"扫参实测增益"。

## 成功判据（事前定义，照 experiment.md §5）

相对 0.05 baseline，存在某 offset 使 **near-band 5cm 接触 ↑ 显著（≥+0.03）且 hand 穿透不升（≤baseline）**，pelvis 不摔、obj_err 不恶化、leg_pen 不显著升。
报 mean+std+worst，禁 cherry-pick；A/B 同 clip 并排视频（grasp/contact 关键帧）必看。

## 风险与对策

- **参考也穿透**：OmniRetarget 参考本身 hand-pen 0.596，offset 前移可能把 target 推到箱内 → 接触假高。对策：用 clean relaxed8；评测同看 hand-geom 穿透（target 入箱会暴露）；视频复核。
- **eval EEF_OFFSET 不参数化 → 量错**：D 已强制修复。
- **baseline 批次混淆**：0.05 复用 E148、0.08/0.11 新批，CEM 随机。对策：log 标注；若 off 版指标与 E148 量级矛盾则复核；必要时补跑 0.05 同批（不在本轮）。

## 范围约束

- 不改 SPIDER reward/算法代码（路线A 纯 config；唯一代码改动是 eval 脚本 EEF_OFFSET 参数化，属评测工具）。
- 不改 8 case 的 override YAML（offset 走 CLI）。
- 不重跑 0.05（复用 E148）。
- 不碰路线B（mesh 表面接触）、object actuator、partner、物体侧 collision_policy。
- 所有 scene/manifest git 追踪 + per-exp snapshot。

## 验证（端到端）

1. 烟测 1 case（box021_035_p1）off=0.08 smoke：日志确认 `contact_hdmi_eef_offset=[0.08,0.0,0.0]` 吃进 config；出 `trajectory_mjwp_act.npz`。
2. 全 16 runs 回收 npz/mp4 齐全（missing=0）。
3. eval 产出三方 TSV/xlsx，EEF 指标用对应 offset；与 E148 rubber-0.05 baseline 行可对齐。
4. 视觉：off-0.08/0.11 关键帧手面更贴箱、穿透不恶化（对比 0.05）。
5. 结论明确回答：前移锚点是否让 near-band 接触↑且穿透不升——验证/证伪锚点错位假设。
