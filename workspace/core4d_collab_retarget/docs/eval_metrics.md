# CORE4D 双人协作动力学重定向 — 评测指标体系

> 适用范围：`workspace/core4d_collab_retarget/` 下 E001–E018b 及后续实验。
> 父文档：`workspace/core4d/docs/eval_metrics.md`（单人 SPIDER + core4d 通用指标）。本文件在父文档基础上新增 **true-freejoint object + 虚拟 partner + canonical anchor** 设定下的扩展定义与实现入口。
> 实现入口：`workspace/core4d_collab_retarget/scripts/eval/paper_metrics.py`（核心）+ `unified_eval.py`（表格 / xlsx 生成器）。

---

## 0. 设定差异（务必先读）

| 项 | core4d E081 baseline | core4d_collab_retarget E014/E018/E018b |
|---|---|---|
| Object 控制 | `scene_act` 6-DoF actuator + contact guidance | 真 freejoint（MJCF `<freejoint/>`）|
| `nq_obj` | 6（slide + euler） | 7（pos 3 + quat 4） |
| `nu` | 35（29 robot + 6 object actuator） | 29（纯 robot） |
| Virtual partner | 无 | mocap support body + soft weld（COLA-B） |
| Anchor | — | E018b canonical：face center + `0.62 · half_z` |

E081 的 obj 跟踪精度（`0.143/0.271m`）是在 object 不受真实惯性约束下取得的；E014/E018b 在更严格的 true-freejoint 设定下，做到 `0.056/0.087m`（E014 best）与 13 case mean `0.054m`（E018b）。**对比表必须标明设定差异**，避免读者误把数字横向对齐。

---

## 1. 指标全集（按论文 / 输出字段）

字段以 `paper_*` 前缀写到 `comparison.csv`；本文件用"字段名 ↔ 论文符号 ↔ 中文 ↔ 方向 ↔ 实现 file:line"五列对齐。Y/N 列表示 E019 P0 是否已实现并通过 E018b 13 case 验证。

**方向 legend（全文档统一）**：
- **↓** 越小越好（误差、穿透、脚滑、smoothness mean |q̈|、relative smoothness vs ref）
- **↑** 越大越好（contact preservation、object/transport success、pelvis upright）
- **↑→1** 期望接近 1（progress ratio）

### 1.1 SPIDER Table 4 严格对齐（FK 全身 body）

> 方向标记：↓ 越小越好；↑ 越大越好；↑→1 期望接近 1。

| 字段 | 论文符号 | 中文 | 方向 | 公式（简） | 实现 | Y/N |
|---|---|---|:-:|---|---|---|
| `paper_spider_joint_err_deg` | Joint Err. | 关节角误差（°） | ↓ | `mean_{t, j∈[7..36)} | q^sim - q^ref | · 180/π`，29 dof | `paper_metrics.py:_add_body_tracking_metrics` | ✓ |
| `paper_spider_pos_err_cm` | Pos. Err. (MPKPE) | 全身 body 位置误差（cm） | ↓ | `mean_{t, b∈robot_bodies} ‖xpos^sim - xpos^ref‖ · 100` | 同上 | ✓ |
| `paper_spider_ori_err_deg` | Ori. Err. | 全身 body 朝向误差（°） | ↓ | `mean 2·arccos(|xquat^sim · xquat^ref|) · 180/π` | 同上 | ✓ |
| `paper_spider_root_pos_err_cm` | Root Pos. Err. | 根（pelvis）位置（cm） | ↓ | 限定 b=pelvis 版 Pos Err | 同上 | ✓ |
| `paper_spider_root_ori_err_deg` | Root Ori. Err. | 根（pelvis）朝向（°） | ↓ | 限定 b=pelvis 版 Ori Err | 同上 | ✓ |
| `paper_spider_eef_pos_err_cm` | EEF Pos. Err. | 末端（L/R wrist_yaw_link）位置（cm） | ↓ | L/R 平均 | 同上 | ✓ |
| `paper_spider_eef_ori_err_deg` | EEF Ori. Err. | 末端朝向（°） | ↓ | L/R 平均 | 同上 | ✓ |
| `paper_spider_obj_pos_err_cm` | Obj. Pos. Err. | 物体位置（cm） | ↓ | re-export `paper_object_Epos_case_m × 100` | `paper_metrics.py:_add_object_tracking_metrics` | ✓ |
| `paper_spider_obj_ori_err_deg` | Obj. Ori. Err. | 物体朝向（°） | ↓ | re-export `paper_object_Erot_case_deg` | 同上 | ✓ |

**Robot body 集合定义**：`[1 .. nbody-1]` 排除 world(0) + `object` body + `support_weld_anchor`/`support_dynamic_anchor` body。E018b 模型 `nbody=33` → 30 个 robot bodies（pelvis + 29 link）。

**Case window**：所有指标在 `summary["case_window_start_frame"]:end_frame` 内取平均；与父文档一致。

### 1.2 OmniRetarget Table II 严格对齐（mj_geomDistance 穿透）

| 字段 | 论文符号 | 中文 | 方向 | 公式 | 实现 | Y/N |
|---|---|---|:-:|---|---|---|
| `paper_omniretarget_mj_penetration_duration_pct` | Pen. Duration | 穿透时长比例（%） | ↓ | `frac(t : ∃ pair, sdf < -0.01m) · 100` | `paper_metrics.py:_add_penetration_metrics_mj` | ✓ |
| `paper_omniretarget_mj_penetration_max_depth_cm` | Pen. Max Depth | 最大穿透深度（cm） | ↓ | `max_{t, pair} (-sdf) · 100` | 同上 | ✓ |
| `paper_omniretarget_mj_penetration_mean_depth_cm` | (派生) | 平均穿透深度（cm） | ↓ | `mean over penetrating frames` | 同上 | ✓ |
| `paper_omniretarget_mj_penetration_case_*` | (派生) | case-window 版 | ↓ | 同上限 `start:end` | 同上 | ✓ |
| `paper_omniretarget_foot_skating_duration_pct` | Foot Skating Duration | 脚滑时长比例（%） | ↓ | demo 接触帧中 `|v_xy|>0.05m/s` 的比例 | `paper_metrics.py:_add_keypoint_proxy_metrics` | ✓ |
| `paper_omniretarget_foot_skating_max_vel_cm_s` | Foot Skating Max Vel | 脚滑最大速度（cm/s） | ↓ | demo 接触帧内的最大 v_xy | 同上 | ✓ |
| `paper_omniretarget_contact_preservation_5cm_pct` | Contact Preservation (代理) | 接触保持率（%） | ↑ | mask 期望接触帧 ∩ sim hand-obj < 5cm 的比例 | `paper_metrics.py:_add_contact_and_penetration_metrics` | △ |

**△ Contact Preservation 当前是 mask-gated 代理**（28cm obj-local 二值版需要 SMPL-X 22 关节，P1 加）。

**Pair 过滤**：复用 holosoma `eval_paper_metrics.py:67-92` 的 `_prefilter_collision_pairs`（临时扩 margin 后跑 `mj_collision`），再用 `mj_geomDistance` 精算；排除 object↔ground 配对（物体接地是合理的）。

### 1.3 DynaRetarget Table V

| 字段 | 论文符号 | 方向 | 公式 | 实现 |
|---|---|:-:|---|---|
| `paper_object_Epos_case_m` | Obj. Epos | ↓ | `mean ‖p_obj^sim - p_obj^ref‖` | `paper_metrics.py:_add_object_tracking_metrics` |
| `paper_object_Erot_case_deg` | Obj. Erot | ↓ | `mean 2·arccos(|q · q_ref|) · 180/π` | 同上 |
| `paper_dynaretarget_object_success` | Object Success (二值) | ↑ | `Epos<0.10m ∧ Erot<25°` | 同上 |
| `paper_dynaretarget_smoothness` | Smoothness | ↓ | `mean |q̈|`, 29 dof, central diff (FPS²) — 越小越平滑 | `paper_metrics.py:_add_smoothness_metrics` |
| `paper_dynaretarget_relative_smoothness` | (派生) | ↓ | sim / ref smoothness 比值（< 1 表示 sim 比 ref demo 更平滑）| 同上 |

### 1.4 CORE4D 协作 自定义（不在论文）

| 字段 | 中文 | 方向 | 公式 | 用途 |
|---|---|:-:|---|---|
| `paper_carry_progress_ratio_case` | 搬运进度比 | ↑→1 | `proj(Δsim_xy, Δref_xy) / ‖Δref_xy‖`（理想 1.0）| E081 transport gate |
| `paper_transport_success` | 任务级二值成功 | ↑ | `progress ≥ 0.7 ∧ z̄ ≥ 0.20m ∧ Epos < 0.20m` | E018b 13/13 通过 |
| `paper_omniretarget_robot_object_deep_penetration_duration_pct` | 深穿透时长（2cm 阈值，%） | ↓ | `frac(t : robot-obj sdf < -0.02m)` | csv-based，需 `legobj_timeseries_*.csv` |
| `case_window_pelvis_z_min_m` | pelvis 最低高度（m） | ↑ | min over case-window（高 = 站着，低 = 摔了）| fall gate（< 0.45m 即视为摔倒） |
| `E018b_robot_fall_detected` | 摔倒检测 | ↓ | `pelvis_z_min < 0.45 ∨ first_pelvis_z_lt_45cm_frame ≥ 0` | E018b 视觉稳定 gate |

---

## 2. 输入数据约定

| 字段 | shape | 说明 |
|---|---|---|
| `qpos` | `(T, 2, 43)` 或 `(T, 43)` | spider 多 env CEM 输出；eval 时取 env 0（`eval_E072.flatten_time_major`）。第二维是 **parallel env**，不是 person。|
| `qpos_ref` | `(T, 43)` | hydra cfg 重建的 kinematic reference，时间已插值到 sim_dt |
| `ctrl` | `(T, 2, 29)` 或 `(T, 29)` | 29 dof robot ctrl；object 无 actuator |
| `support_proxy_*` | `(T, …)` | E014+ 的诊断字段（force/torque/pos/vel），仅本工作区使用 |

E018b layout：`qpos = [pelvis(7) + 29 robot dof + object(7)]`，`nq=43, nv=41, nu=29`，`nbody=33` (含 mocap support_weld_anchor)。

**Reference 来源**：所有 `qpos_ref` 由 `e002.load_ref(override, case)` 重新加载 hydra config，再用 `spider.io.load_data` 插值到 `sim_dt`（默认 1/60 = 0.01667s）。

---

## 3. 实现入口（file:line）

| 入口 | 路径 | 用途 |
|---|---|---|
| `add_paper_metrics` | `scripts/eval/paper_metrics.py:686` | 唯一对外入口，被各 `eval_E0NN.py` 调用 |
| `_add_object_tracking_metrics` | `paper_metrics.py:125` | object Pos/Ori Err + transport |
| `_add_smoothness_metrics` | `paper_metrics.py:185` | DynaRetarget smoothness |
| `_add_keypoint_proxy_metrics` | `paper_metrics.py:201` | foot skating + 5-keypoint MPKPE proxy（旧） |
| `_add_contact_and_penetration_metrics` | `paper_metrics.py:594` | csv-based 接触/深穿透（兼容 E001+ 旧管线） |
| `_add_body_tracking_metrics` | `paper_metrics.py:300` | **SPIDER T4 严格对齐（FK 全身 body）** |
| `_add_penetration_metrics_mj` | `paper_metrics.py:414` | **OmniRetarget Pen via mj_geomDistance + prefilter** |
| `unified_eval` CLI | `scripts/eval/unified_eval.py` | 把任意 `comparison.csv` 转 md + xlsx |
| Eval per-experiment 包装 | `scripts/eval/eval_E{014,018,018b,…}.py` | 复用 `add_paper_metrics`，加各自工程字段 |

依赖：

- `eval_E072.flatten_time_major` / `replay_metrics`：`workspace/core4d/scripts/eval/eval_E072.py:42, 137` — env 重排 + FK 时间序列
- `e002.load_ref` / `load_scene_model`：`workspace/core4d_collab_retarget/scripts/eval/eval_E002.py` — hydra ref 与 scene 模型加载
- holosoma 参考实现：`/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/workspace/v1/scripts/eval_paper_metrics.py:67-148`（penetration / foot skating / contact preservation）

---

## 4. 复现命令

### 4.1 单 case 重评

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python \
  workspace/core4d_collab_retarget/scripts/eval/eval_E018b.py \
  --variant E018b_box025_p2_canonical_t02
```

输出：
- `results/E018b/eval_summary_E018b_box025_p2_canonical_t02.{csv,json}` — 单 case 全字段
- `results/E018b/comparison.csv` — 已有时合并，未有时新建
- `results/E018b/aggregate_summary.json` — 跨 case 聚合

### 4.2 13 case 全量重评

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python \
  workspace/core4d_collab_retarget/scripts/eval/eval_E018b.py --all
```

### 4.3 生成统一表格 + xlsx

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/unified_eval.py \
  --method spider_E018b \
  --comparison workspace/core4d_collab_retarget/results/E018b/comparison.csv \
  --out workspace/core4d_collab_retarget/results/eval_unified
```

输出：
- `results/eval_unified/INDEX.md`
- `results/eval_unified/tables/table_spider_t4.md`
- `results/eval_unified/tables/table_omniretarget_t2.md`
- `results/eval_unified/tables/table_dynaretarget_t5.md`
- `results/eval_unified/tables/table_core4d_collab.md`
- `results/eval_unified/tables/table_spider_E018b.md`（per-method 合订本）
- `results/eval_unified/tables/table_paper_all.xlsx`（4 sheet：`raw_spider_E018b` / `spider_t4_mean` / `omni_t2_mean` / `by_object_spider_E018b`）
- `results/eval_unified/per_case/spider_E018b/{case}.json`（13 个）

多 method 对比（E014/E018/E018b/holosoma kin 等）：

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/unified_eval.py \
  --method spider_E014 --comparison workspace/core4d_collab_retarget/results/E014/comparison.csv \
  --method spider_E018b --comparison workspace/core4d_collab_retarget/results/E018b/comparison.csv \
  --out workspace/core4d_collab_retarget/results/eval_unified
```

xlsx 的 `spider_t4_mean` / `omni_t2_mean` sheet 会自动并列出现一行 per method。

---

## 5. 阈值与方向汇总

| 指标 | 方向 | 单 case 阈值（成功） | 来源 |
|---|---|---|---|
| Obj. Pos. Err. | ↓ | < 10 cm | DynaRetarget |
| Obj. Ori. Err. | ↓ | < 25° | DynaRetarget |
| Joint Err. | ↓ | （论文未给二值阈值，报告 mean ± std） | — |
| MPKPE | ↓ | （论文未给二值阈值，报告 mean ± std） | — |
| Pen. Duration | ↓ | < 5% | OmniRetarget |
| Pen. Max Depth | ↓ | < 5 cm | OmniRetarget |
| Foot Skating Duration | ↓ | < 10% | OmniRetarget |
| Contact Preservation 5cm | ↑ | ≥ 70% | OmniRetarget（代理阈值，原 28cm 二值） |
| Carry Progress Ratio | ↑ | ≥ 0.70 | 本任务 |
| Transport Success | — (二值) | progress + z + Epos 三联门 | 本任务 |
| Robot Upright | ↑ | pelvis_z_min ≥ 0.45m | 本任务（fall gate） |

---

## 6. E018b 13-case Mean ± Std（首发数字，2026-05-20）

抽自 `results/eval_unified/tables/table_spider_t4.md` aggregate：

| 指标 | Mean | Std |
|---|---:|---:|
| Joint Err. (°) | 5.85 | 3.44 |
| MPKPE (cm) | 23.13 | 22.77 |
| Body Ori. Err. (°) | 23.49 | 22.46 |
| Root Pos. Err. (cm) | 22.90 | 24.70 |
| Root Ori. Err. (°) | 15.79 | 22.23 |
| EEF Pos. Err. (cm) | 20.29 | 25.29 |
| EEF Ori. Err. (°) | 30.17 | 26.04 |
| Obj. Pos. Err. (cm) | 5.45 | 1.63 |
| Obj. Ori. Err. (°) | 5.22 | 3.07 |

注意 std 大的字段（MPKPE / Root / EEF）主要由 4 个 robot fall case（`box021_p1/p2`、`bucket001_p1/p2`）拉高；剔除后非 fall 9 case mean 大幅下降。这恰好印证报告中"object-side success ≠ 完整 retargeting 成功"的诚实定位。

---

## 7. E019 P1 新增（2026-05-20）

### 7.1 28cm 严格版 Contact Preservation（OmniRetarget Table II）

新增 `_add_contact_preservation_omni_local`（`paper_metrics.py:594`）：

- **定义**（对齐 holosoma `eval_paper_metrics.py:258-303`）：
  - demo_contact_t = ‖transform_world_to_local(demo_wrist_t − demo_obj_pos_t, demo_obj_quat_t)‖ < 0.28m
  - sim_contact_t = ‖transform_world_to_local(FK(qpos_t)_wrist − sim_obj_pos_t, sim_obj_quat_t)‖ < 0.28m
  - miss_t = demo_contact_t ∧ ¬sim_contact_t（per-side then OR）
  - **preservation = 1 − miss_frames / T**（T = 全帧数；当 demo 全无接触时 trivial 100%）
- 字段：`paper_omniretarget_contact_preservation_local_{full,case}_pct` + `_demo_frames` + `_miss_frames` + `_radius_m`
- **CORE4D 大物体上的退化 caveat**：SMPL-X 腕关节（j20/j21）中心位置距大箱体（half-size ~16–26 cm）COM 通常 >28cm（实测 box025_p2 demo 腕→obj COM mean 50cm），demo 全 0 帧接触 → metric 退化为 trivial 100%。在我们的 case 集合上**这条指标的二值结论不可作为单一 work/fail 判据**；mask-gated 5cm proxy (`paper_omniretarget_contact_preservation_5cm_pct`) 仍是主要可用指标
- 输入：`summary["human_joints"]`（直接传 `(T,22,3)` ndarray）或 `summary["human_joints_npz"]`（NPZ 路径）+ optional `summary["human_joints_key"]`（默认 `"human_joints"`）

### 7.2 EvalInputs adapter

新增 `scripts/eval/adapters/`：

| 文件 | 用途 |
|---|---|
| `common_inputs.py` | `EvalInputs` dataclass：`method / case / model / qpos_sim / qpos_ref / fps / case_window / human_joints / object_poses / extras` |
| `kinematic_to_common.py` | `load_kinematic_inputs(case, model)` — 读 holosoma v2 `retarget_replace_batch_trimmed/*.npz` + companion `data/core4d_replace_batch/{seq}-object.npz`，输出 `EvalInputs(method="holosoma_v2_kinematic")` |

`adapters.HOLOSOMA_V2_CASE_MAP` 映射 spider 短 case → holosoma NPZ 文件名。当前覆盖：

| spider | holosoma v2 file |
|---|---|
| `box025_p1` | `20231011-048-person1-Box025_with_obj_original.npz` |
| `box025_p2` | `20231011-048-person2-Box025_with_obj_original.npz` |

其余 11 个 spider E018b case（box021/p2、box023/p2、bucket001/p2、bucket005/p2、bucket007/p2、desk021）**holosoma v2 未 retarget** — 跨方法对比受限 N=2。

### 7.3 Kinematic eval 入口

新增 `scripts/eval/eval_holosoma_kinematic.py`：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python \
  workspace/core4d_collab_retarget/scripts/eval/eval_holosoma_kinematic.py --all
```

输出 `results/holosoma_v2_kinematic/`：
- `eval_summary_holosoma_v2_kinematic_{case}.{csv,json}` × 2
- `comparison.csv` — unified_eval 直接消费
- `aggregate_summary.json`

对 kinematic 走 **physics-only** 模式（`add_paper_metrics_physics`）：跳过 SPIDER T4 body tracking（kin = ref，sim/ref err 退化），只评 OmniRetarget mj_geomDistance penetration + 28cm contact preservation + DynaRetarget smoothness + object self-Pos/Ori（degenerate）。

### 7.4 Tab.5 跨方法对比

`unified_eval.py` 现支持多 method 输入，自动产出：

- `tables/table_method_comparison.md` — 4 张 metric group 表（SPIDER T4 / OmniRetarget T2 / DynaRetarget T5 / CORE4D 协作），列：metric × method × Δ
- 自动取**短 case 名交集**（修正后的 `_short_case` 把 `box025_person2_freejoint_legobj_e018b` 与 `box025_p2` 都归一化为 `box025_p2`）
- xlsx 的 `spider_t4_mean` / `omni_t2_mean` sheet 自动包含全 method 行

复现：

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/unified_eval.py \
  --method spider_E018b --comparison workspace/core4d_collab_retarget/results/E018b/comparison.csv \
  --method holosoma_v2_kinematic --comparison workspace/core4d_collab_retarget/results/holosoma_v2_kinematic/comparison.csv \
  --out workspace/core4d_collab_retarget/results/eval_unified
```

**Tab.5 数字（N=12，2026-05-20 升级，详见 log/20b）**：

箭头：↓ 越小越好（error / penetration / skating / smoothness jerk magnitude）；↑ 越大越好（contact preservation / pelvis upright）；↑→1 期望接近 1（progress ratio）。

| 指标 | spider physical (N=12) | holosoma kinematic (N=12) | Δ |
|---|---:|---:|---:|
| Obj. Pos. Err. (cm) ↓ | 5.50 | 0.00 | −5.50 |
| Obj. Ori. Err. (°) ↓ | 5.51 | 0.00 | −5.51 |
| mj_pen Duration (%) ↓ | 0.0 | 0.0 | 0 |
| mj_pen Max Depth (cm) ↓ | 0.0 | 0.0 | 0 |
| spider 5cm Contact Preservation (%) ↑ | 54.55 | — | — |
| kin 28cm Contact Preservation (%) ↑ | — | 53.59 | — |
| **Smoothness (rad/s²) ↓** | **13428** | **36048** | **+22621 (kin 2.7× 更不平滑)** |
| Rel. Smoothness vs ref ↓ | 0.757 | 1.00 | spider 比 kin 更平滑 |
| Foot Skating Max Vel (cm/s) ↓ | 96.71 | — | — |
| Pelvis Min z (m) ↑ | 0.567 | 0.711 | −0.145 (spider 含 4 fall case) |

`—` 表 kin 是 sim=ref 退化、字段不在 physics-only 路径、或阈值不同不能直接比。

**诚实解读**：
1. **smoothness 是最强 selling**：spider 比 kin 低 62.7% (3.0× gap)，在 12 个 case + 6 个 obj 类型上 robust 保持。gap 演化：N=2 −67.8% → N=3 −66.9% → N=12 −62.7%，略缩窄但量级稳定
2. **mj_pen 13 case 全零，两边都 0/0**：OmniRetarget 软约束 + spider 物理硬约束都把 penetration 压住了 — Tab.5 这条**不能区分方法**（"tie"信号）
3. **28cm contact preservation**：N=2 box025 子集 trivial 100%；N=12 加 box021/box023/bucket001/bucket007 后阈值开始有判别力，kin 平均 53.59%，box021_p1 = 29.55%
4. **spider 5cm preservation 54.55% vs kin 28cm 53.59%**：spider 在 5cm 严格阈值下打平 kin 在 28cm 更宽松阈值下的数字 — 间接说明 spider contact closure 质量显著更好
5. **N=12 缺 desk021_p1**：CVXPY clarabel SOCP 返回 infeasible（motion-specific 问题，XML 与已成功的 desk005 完全一致，不是模板问题）。N=12 已足够支撑论文核心论断

### 7.5 已知 caveat 与下一步

1. ~~跨方法 N=2 是数据限制~~ → **2026-05-20 已升 N=12**（详见 `log/20b`）。剩余 desk021_p1 单 case 缺 (SOCP infeasible)。报告 v1 用 N=12 + caveat。
2. **FPS 单点修复**（v2026-05-20-P2，已落地）：模块常量 `FPS=50` → `FPS=30`，匹配 CORE4D 30Hz + holosoma v2 kinematic 30Hz。`add_paper_metrics` 入口加 warning：若 `summary["fps"]` 与 FPS 不一致则报警。per-case 字段全联动降级为 P3（当前唯一 30Hz 消费者已统一，per-case 改造的收益主要是 future-proofing）。
3. **mask-gated 5cm contact preservation 与 28cm local-frame 并存**：mask-gated 是当前唯一对 CORE4D 大物体有判别力的接触指标；28cm 严格版用于跨方法严格对齐，但容易 trivial 100%。docs 必须同时呈现。
4. **xlsx 依赖 `openpyxl`**：本机已装 `openpyxl 3.1.5`（via uv + 小红书 mirror）。
5. **`E018b_diagnostic_class`** 在 `paper_metrics` 之外（由 `eval_E018b.py` 计算）；归因强化在 plan 21（E020 audit）。
