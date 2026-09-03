# plan236 · E206：desk+chair 走 dcv3 全流程 + lowgeom 非凸碰撞代理 + noPRG/PRG 双 arm

_Core4D · Phase 66 · Run **R291**(noPRG) + **R292**(PRG) · 承接 [E174 bucket/desk move2 非凸](../log/234_E174_bucket_desk_move2_nonbox_results.md) / [E176 lowgeom 代理](../log/235_E176_lowgeom_proxy_canary_results.md) / [plan234 双 arm 口径](234_E204_E205_bucket_noprg_g1a2_arm_ablation_plan.md) · 2026-09-02 · 分支 `feat/E206-desk-chair-move2-dcv3-arms` · **状态：计划已批准，待实现**_

## Context / 目标

CORE4D v1 的 6 大类里，真正被重定向 + 物理筛选落地的只有 **box / bucket** 两类。剩下四类：

- **board / stick 本轮不做**：在 dcv3 S1 `build_inventory.py::hard_reject_reason` 的 AABB 体积/尺寸硬门被终止（stick 114 行全部 `reject_too_small_bucket001_or_smaller`；board020 126 行因 2.7cm 薄板的 mid/max extent ≥ box025 触发 `reject_too_large_box025_or_larger` —— **薄板的真实误判**）。0 场景、0 模板审批。另立后续实验，且需先修 S1 尺寸门。
- **desk / chair = 本实验目标**：有 mesh、E145/E174 有过 `approve_clean`、已有 43 个 dcv3 任务目录。但物理证据极差 —— **desk CEM 历史 2/14 通过（E145 phase2 0/5 + E174 desk007 2/9）；chair 从未跑过一次 CEM**。

**新诊断（本实验最强假设）**：E174 desk 失败**很可能不是代理精度问题，而是碰撞对配错**。机器人 geom 在 `<default class="g1">` 下 `contype=0 conaffinity=0`，只能通过显式 `<pair>` 接触。实测 `dcv3_omnirt_v1_ref_fk_desk007_20231023_047_p1/scene_act.xml` 只有 **2 条** robot↔object pair（`lh/rh ↔ object_collision`），而 desk007 的体素代理有 **41 个 box** → **40/41 个碰撞盒对机器人物理不可见**。这完整解释了 E174 的 `hand_object_contact_in_mask=0.379` 与 `leg_penetration_frac=0.165`。E175 修过（`replace_robot_object_pairs` → 18×N）但从未跑到 full CEM。**E206 是第一次在正确配对的多 geom 非凸物体上跑完整流程。**

**硬约束**：`spider/config.py:69-79` 的 `object_collision_sdf_mode='union'` **fail-closed 只接受 box geom**，`mjwp.py:181/213/361` 全链路断言 `mjGEOM_BOX` → 碰撞代理**必须是轴对齐 box**，不能用 mesh/CoACD。而现网 desk/chair 场景是 26-cell 体素草稿：desk007 41 / desk023 37 / desk005 74 / desk021 73 / chair005 64 / **chair021 127** 个 box（127 → 2286 对，落在 E175 已被否决的 52–66h 区间）。

**目标**：(1) 打通 desk+chair 的 dcv3 全流程；(2) 建立一套可复审、可度量的 **≤N_MAX box 非凸碰撞代理方法**；(3) 在同一批 case 上产出 noPRG / PRG 两 arm 数据并配对对比。

## 已锁定的口径（用户 2026-09-02 确认 5 项）

| 项 | 决定 |
|---|---|
| 物体 | **desk + chair 全部可用实例 = 10 key**：desk005/007/020/021/023 + chair005/006/020/021/022（desk001 6 行被 S1 尺寸门硬拒）。board/stick 不做 |
| 碰撞代理 | **E176 lowgeom 自适应粗体素（≤N_MAX box）+ 强制人工 overlay 复审** |
| 数据版本 | **core4d v1**（`SPIDER_DATASET=core4d`），不用 core4d_v2 |
| 动作 | **move2_\*（obs0+obs1+obs3）全要**，与 bucket 线（E174→E178→E202→E204/E205）一致 |
| arm | **恰好两个：noPRG 与 PRG**。无 G1（gravcomp）、无 A2（hand-gate retune） |
| **接触掩码** | **3cm 单一口径**，与 box/bucket 全线一致，无桥接层。<br>_（口径先定 5cm，P1 实测后改回：见下「P1 实测修正」）_ |
| **N_MAX** | **16**（用户授权从 E176 冻结的 9 抬上来）；P3 实测不可负担则按 A3 阶梯回落到 9 |
| **noPRG 定义** | **≡ E167A**（reward arm）：`leg_object_penalty_scale=0` / `geom_names=[]` / `cem_leg_gate_enabled=false`，base reward `E167A_zOnlyBody`，无 G1 无 A2。**场景仍用共享 rubber-hull** |

## P1 实测修正（2026-09-02，S1 跑完后）

计划期锁的是「5cm 主口径 + 3cm 桥接层」，理由是担心 3cm 样本不够。**S1 实测把这个前提证伪了**：

| 口径 | 落地 case | 覆盖物体 | 说明 |
|---|---:|---:|---|
| 3cm | **74** | 9/10 | 与 box/bucket 全线 + E174 同尺 |
| 5cm | 76 | 9/10 | 只多 2 个（desk007 +1、desk020 +1） |

放宽接触阈值**只救回 2 个 case**，却要付出与整条历史线口径断裂 + 维护桥接层（独立 S3、独立 task dir、C5a 特殊处理）的代价。用户据此改回 **3cm 单一口径**。收益：R1b 口径断裂风险消失、C5a 可直接与 E174 desk007 同尺对比、S3 少跑一轮、CEM 少 18 条。

**衰减真因（本实验第一个 finding）**：瓶颈不是接触阈值，是 **`object_rotation >= 45°` 运动硬门，卡掉 56/164（34%）**；`object_lift <= 0.30m` 只卡掉 8 个（5%），接触质量不足 25 个（15%）。双人搬桌/椅本来就常要转向（绕门、调头），而 45° 门是为 box 搬运设的。**E206 不动此门**（保持与 bucket 线 E174/E178/E202/E204/E205 同一 S1 口径，结论可横向比），作为后续实验方向记入 log。

**S1 规模**：74 case / 9 物体。`desk005` 落地 0 case（6 个候选全被旋转门/接触质量拒），**退出 E206**。

## P2.3b 修正（2026-09-03，人工重摆碰撞体后）

计划期的碰撞代理是「自动体素 ≤N_MAX box + 人工 overlay 复审（只能删）」。实测下来删 box 救不了最差的椅子（chair005/chair022 腔体过填 0.41，不存在正确的 box 子集），故追加**自由编辑界面**（`edit_proxy_3d.py`），由用户逐物体重摆 box，存绝对几何于 `manual_boxes.json`，优先级高于 voxel/semantic。详见 [log295 F9–F12](../log/295_E206_desk_chair_move2_dcv3_noprg_prg.md)。

三项口径变更：

| 项 | 变更 |
|---|---|
| **物体** | **chair021 弃用**（几何不支持值得跑的 ≤16-box 代理）→ 实际物体数 **8** |
| **规模** | **65 case × 2 arm = 130 条 CEM**。逐物体：desk007 9 / desk020 2 / desk021 17 / desk023 14 / chair005 2 / chair006 13 / chair020 1 / chair022 7 |
| **新门 G10** | **支撑面共面**：承重 box（底面距 mesh 地面 ≤2cm）的 `|bottom − floor_y| ≤ 5mm`。见下 P2.2 门表 |

**G6 豁免需求归零**：手编后最高过填是 chair020 的 0.25（< 0.30 告警线）。R3 与 C1 里关于「椅子座下被填实、腿无法摆过」的结论边界声明**全部撤销**。

> **样本量诚实声明**：chair020 只有 1 例、chair005/desk020 各 2 例 —— 这三个物体**不做 per-object arm 推荐**，只并入总体统计并在 log 中标注 n。

## Plan 阶段已实测的关键事实（非假设）

**M1 · ≤9 box 对 10 个物体全部可行**（`merge_occupied_voxels(max_boxes=9)`，`left_unmerged==0`）：

| 物体 | best tc | boxes | mesh→proxy p90 | proxy→mesh p90 | 腔体过填(>5cm) |
|---|---:|---:|---:|---:|---:|
| desk005 | 5 | 9 | 0.059 | 0.093 | 0.26 |
| desk007 | 7 | 9 | 0.042 | 0.095 | 0.34 |
| desk020 | 6 | 9 | 0.028 | 0.070 | 0.08 |
| desk021 | 12 | 7 | 0.023 | 0.028 | **0.00** |
| desk023 | 9 | 9 | 0.075 | 0.079 | 0.22 |
| chair005 | 5 | 8 | 0.038 | 0.055 | 0.08 |
| chair006 | 4 | 9 | 0.057 | 0.123 | **0.45** |
| chair020 | 5 | 9 | 0.037 | 0.078 | 0.23 |
| chair021 | 4 | 8 | 0.042 | 0.124 | **0.42** |
| chair022 | 4 | 8 | 0.064 | 0.127 | **0.44** |

全部满足 E176 的 `mesh→proxy p90 ≤ 0.08` 门。`desk007 tc=5→9 boxes` 与 E176 冻结值逐位一致（自洽性校验通过）。

**M2 · N_MAX=16 能显著修好差的椅子**：`max_boxes=64` 时 chair021 tc6→11、chair006 tc6→15、chair022 tc6→14、chair005 tc6→14、desk007 tc8→12 → **约 1.5× 体素分辨率，正好作用在 42–45% 过填的物体上**。

**M3 · E176 的 `center_inside_count` 断言会卡死全部 5 把椅子**（`lowgeom_proxy.py:105-113`）。这是为空心 bucket/desk 写的启发式，椅面天然占据 AABB 中心。**必须换成直接腔体度量，不能简单删掉**（删了就失去 E176 C4/C5 建立的腔体保证）。

**M4 · E176 的 ≤3.0s/record 吞吐门必须作废重推**。其自身 log235 §4.3 写明残余超时"不能归因于 object geom 数"（6 geom 的 bucket004 失败、9 geom 的 desk007 通过），且测于 `64×4` 小预算 + `torch.compile` 关闭。本机 `Python.h` 存在 → `torch.compile` 可用；E203 已确立 `1024×32` 全预算 ~46min/task 为接受常态。

**M5 · 源模板缺失/污染**：`chair006_person1/2`、`chair020_person1/2`、`chair022_person2` **完全不存在**；`desk005_person2`、`chair022_person1` 只有 1 个遗留单 AABB box（椅子当实心块 → 所有接触指标失真）。

**M6 · P5 会覆盖 43 个已有任务目录**（`run_stage2b.py:278` `target_task = f"dcv3_{run_slug}_{object_key}_{date}_{seq}_{pshort}"` + `SPIDER_DATASET=core4d` 锁定）→ 必须先做覆盖前快照。

## Arm 精确定义（两 arm 对照）

| Arm | scene_act | 手 geom | 手↔物对 | 腿↔物对 | leg penalty / gate | gravcomp | hand-gate | Run |
|---|---|---|---:|---:|---|---:|---|---|
| **noPRG (≡E167A)** | `scene_act_E206_lowgeom_noPRG` | rubber_hull mesh | 2N | **0** | **0 / false** | 0 | E163 默认 | R291 |
| **PRG** | `scene_act_E206_lowgeom_PRG` | rubber_hull mesh | 2N | **16N** | **2.0 / true** | 0 | E163 默认 | R292 |

> **noPRG ≡ E167A 的精确落地**。`E167A` 是 **reward arm 名**，不是场景名 —— plan230 的口径表里 "reward base（三 arm 同）= `E167A_zOnlyBody`"，两 arm 共用同一 reward 基座，差别只在腿约束三件套。
> 场景侧按 **E200 的实现**而非 plan230 的文字表：`e200_common.py` 里 `noprg -> scene_name=scene_act_E199_rubberHull`（**rubber-hull 手**）；plan230 正文写的 "aug `scene_act.xml`（base，无碰撞对）" 与代码不符，**以代码为准**。
> 理由：base `scene_act.xml` 是 **5cm 球手**，PRG 场景是 **rubber_hull mesh 凸包手**。若 noPRG 用球手，两 arm 就差了「腿约束 + 手部几何」两个变量，C4 单变量断言不成立、整个 arm 对比作废。
> **结论：两 arm 共享 rubber-hull + lowgeom 中间态，唯一差异 = 16×N 腿↔物碰撞对 + `{leg_object_penalty_scale, leg_object_penalty_geom_names, cem_leg_gate_enabled, cem_leg_gate_geom_names, scene_name}` 5 键。**

## 执行阶段（P0–P10）

记号：`$E=workspace/core4d/results/E206`，`$DCV3=workspace/core4d/scripts/data_construction_v3`，`$ED=workspace/core4d/scripts/experiments/E206`，`PY=.venv/bin/python`。**绝不运行 `uv sync`**。

### P0 · 冻结与覆盖前快照
grep 43 个 `dcv3_*_{desk,chair}*` 目录的下游消费者 → `$E/preflight/task_dir_consumers.txt`；`snapshot_scenes.sh E206_pre <43 目标目录 + 16 源模板>` + `git add -f`。
**退出检查**：`results/E206_pre/scene_snapshot/manifest.txt` 含 git HEAD + 每个 `scene*.xml` 的 sha256；消费者冲突清空或 log 显式豁免。

### P1 · S0 + S1：案例集推导  ✅ 已完成
三层过滤逐层单独计数：inventory（desk/chair 652 行）→ `action ∈ move2_{obs0,obs1,obs3}` 且无 `hard_reject_reason`（**164**）→ `run_raw_contact.py --queue object-key --object-keys <10 keys> --thresholds-m 0.03,0.05`。
两个阈值都打分（一次运行），**3cm 为唯一口径**，5cm 列仅作为口径选择的依据保留。
`$ED/report_s1_funnel.py` → `$E/s1_raw_contact/e206_s1_funnel.{tsv,md,json}` + `e206_s1_case_terminal.tsv`（逐 case 两阈值终态 + 原文拒因）+ 案例权威 `raw_contact/raw_contact_pass_3cm_move2only.tsv`。
**退出检查（已过）**：164 候选每个在两阈值下各有唯一终态；落地 **74 case / 9 物体**；停机线 12 → PASS。

### P2 · 碰撞代理（方法核心）
**P2.1 `$ED/lowgeom_proxy_v2.py`** — import `E176/lowgeom_proxy.py`（`load_mesh`/`ProxyBox`/`proxy_xml`/`point_to_proxy_surface_distance`/`fidelity_metrics`）与 `build_or_audit_templates.py` 的 `merge_occupied_voxels`/`geom_box_xml`，**不 fork**。新增：
- `sweep_target_cells(mesh, n_max, lo=2, hi=16)`；选择规则 = **预算内取最大可行 target_cells（最细体素）**，确定性。
- 沿用 E176 内缩 `shrink=min(pitch*0.12, half*0.25)`、`half=max(half-shrink, pitch*0.22)` → E206 是 E176 的严格推广而非新配方。
- **`cavity_metrics()` 替代 `center_inside_count`（M3）**：`interior_overfill_frac_5cm`（代理并集内体积采样 6000 点，`trimesh.proximity.closest_point_naive` 距 mesh >5cm 比例）、`interior_overfill_frac_pitch`、`proxy_volume_over_mesh_aabb`。
- `union_geoms_are_boxes(scene_xml)` — 断言全部编译为 `mjGEOM_BOX`。

**P2.2 `$ED/audit_lowgeom_contract.py`** → `$E/s2_proxy/lowgeom_contract.{tsv,md}`：

| # | 检查 | 门 | 依据 |
|---|---|---|---|
| G1 | `left_unmerged==0` 且 `1≤n≤N_MAX` | 硬 | M1 已验证 10/10 可行 |
| G2 | 全部 `mjGEOM_BOX` | 硬 | `spider/config.py:69-79` fail-closed |
| G3 | `mesh→proxy p90 ≤ 0.08 m` | 硬 | E176 原门；实测最差 0.075 |
| G4 | `mesh→proxy max ≤ 0.16 m` | 硬 | 实测最差 0.133 |
| G5 | `proxy→mesh p90 ≤ 0.14 m` | 硬 | 实测最差 0.127（外凸上限） |
| G6 | `interior_overfill_frac_5cm` | **报告+豁免**，>0.30 告警 | N=9 时 chair006/022/021、desk007 会触发 |
| G7 | `interior_overfill_frac_pitch ≤ 0.02` | 硬 | 实测最差 0.004 |
| G8 | ref-FK 接触目标→代理表面 `p90 ≤ 0.08 m`（逐物体） | 硬，P6 执行 | E176 C6 原门，那里 6/6 达成 |
| G9 | 相对现有 26-cell 草稿，`mesh→proxy p90` 退化 ≤ 0.04 m | 硬 | 证明减 geom 不是保真崩塌 |
| **G10** | **承重 box 的 `\|bottom − floor_y\| ≤ 5 mm`**（承重 = 底面距 mesh 地面 ≤2cm） | **硬** | **P2.3b 新增。桌椅的 3/4 条腿必须同时着地；一高一低 → 物体只站最低那条腿，倾斜由代理捏造；穿地 → 物体浮空。G3/G4/G5 是聚合距离，一条腿差 12mm 对 p90 影响为零，此前无门可见。实测手编后 8/8 全部不合格（高低差 0.4–16.0 mm），`align_supports.py` 对齐后 8/8 归零** |

G6 的诚实表述：N_MAX=16 下椅子过填预期从 42–45% 明显下降（M2），但**必须逐物体实测重报，不能拿 N=9 数字外推**。若 P3 被迫回落到 9，粗椅子会填掉座下空间（腿无法从椅下摆过）—— 有物理后果的近似，**必须显式声明+复审+豁免**，不能静默通过。

**代理不达标回退阶梯**：① 重扫到最粗可行 `tc`；② 手工壳代理（比照 `object_collision_geoms` 的 `bucket_wall_proxy_aabb` 分支），box 数硬封顶 N_MAX；③ 剔除该物体，写 `$E/s2_proxy/lowgeom_dropped.tsv` 附扫描表。**绝不回退到单 mesh-AABB box**（椅子会变实心块，所有接触指标作废）。

**P2.3 overlay 复审**（`docs/data_construction_v3/04_scene_template_policy.md:63-77` 强制：MuJoCo load + render 通过**不足以**放行）
`E175/render_nonbox_proxy_overlay.py`（Agg，无需 GL）+ `render_template_mesh_collision_review_package.py --object-only`（osmesa）双通道；新建 `$ED/build_nonbox_template_review.py`（改自 `E174/build_nonbox_template_review.py`，已有 `--render`/`--approve`），每行附带 P2.2 契约列。逐 `<obj>_person{1,2}`（≈20 行）人工看三点：代理是否堵住真实开口（桌下空间/椅面到地）？碰撞体是否有处明显大于 mesh？搬运真正接触的顶面是否覆盖？
**退出检查**：G1–G5、G7 对 10/10 物体 pass，G6 要么 ≤0.30 要么有 `waived_checks`；`nonbox_template_review.tsv` 每个在范围行 `review_decision=approve_clean` + 非空 reviewer/时间戳。

### P3 · 吞吐实测与 N_MAX 准入（专设阶段，作废 E176 旧门）
**N_MAX=16 已授权为目标值**；本阶段任务从"要不要抬"变成"**16 花多少、8 卡跑不跑得完**"及回落路径。
`$ED/measure_throughput.py`：desk021（最干净，0% 过填）+ chair021（最差，42%）各一 case，构造 **N ∈ {1,5,9,16}** 探针场景（pair 按 18N 缩放：N=16 → **288 对**，其中腿↔物 256 对），**真实预算 1024×32**，`torch.compile` 开/关各一遍，输出到 `--stage throughput` 隔离目录。用 `E176/validate_canary_throughput.py:27-29` 正则解析 per-record，**外加进程总墙钟**，拟合 `total_wall_min ≈ a + b·N`。

**预注册准入判据**：
- **A1（单任务）** N=16 + compile 开，中位总墙钟 **≤ 120 min/task**。依据：E203 确立 N=1 时 ~46min 为接受常态；N=16 是 16× geom / 288× pair，120min ≈ 2.6× 容差。
- **A2（队列）** `65 case × 2 arm = 130 条 ÷ 8 GPU ≤ 48 h`（P2.3b 后：chair021 弃用，148→130）→ 中位墙钟须 ≤ **177 min/task**；A1 的 120min 门比它更紧，故 A1 成立即 A2 成立。driver 是 slot 池 + rollout-npz 为键的 skip-already-done，**全程可中断可续跑**。
- **A2b（优先级排队）** 队列按 **① desk007 全部 9 例（C5a 命脉，与 E174 同 case）→ ② 每物体 round-robin → ③ 余量** 排序。**腰斩时也保证每物体有样本、C5a 一定有数据**。
- **A3（回落）** A1/A2 在 16 不成立时：开 `object_collision_sdf_batch_groups=true`（E176 实测 −2~4%，C10 已验证数值等价）→ 回落 **N_MAX=9**（M1 已验证 10/10 可行，椅子过填按 G6 走豁免）→ 缩物体范围 → 最后才对**两 arm 对称**降 `num_samples`。**每步记录，绝不静默超预算。**

**退出检查**：`e206_throughput_curve.md` ≥16 行实测；`admission_decision.json` 冻结 `N_MAX / use_torch_compile / num_samples / max_num_iterations / 队列优先级序` 与 A1/A2/A2b/A3 各自裁决；**先贴进 log 再发 CEM 队列**。

### P4 · S2：把冻结代理装进源模板 + 补建缺失模板
把 lowgeom 写进 **source template**（而非事后打 CEM sidecar），使 S2→S6 看到同一碰撞体。
- **修改** `$DCV3/stages/s2_templates/build_or_audit_templates.py`：新增 `lowgeom_collision_geoms()`（挨着 `surface_voxel_collision_geoms`，line 360）+ CLI `--nonbox-collision-policy {surface_voxel_draft,lowgeom}`，**默认 `surface_voxel_draft`** 保证既有实验逐字节复现；仅在 `lowgeom` 时于 `object_collision_geoms`（line 402）分派。
- **新建** `$ED/install_lowgeom_templates.py`：只重写 `<body name="object">` 内的 `object_collision*`，复用 `E175/build_nonbox_multigeom_production.py::replace_bucket_proxy` + `stripped_signature` 断言碰撞块之外零变化。**顺带修掉 M5 的两个遗留单 AABB 模板**。
- `--apply-build --nonbox-collision-policy lowgeom` 补建 chair006/chair020/chair022_person2。
**退出检查**：每个源 `scene.xml` 满足 `1≤n(object_collision*)≤N_MAX` 且全 box、MuJoCo 可加载、`nq/nv/nu=43/41/29`、无 `29.632` 惯量污染；`template_backlog.tsv` 全部 `clean_reviewed`；`stripped_signature` 100% 未变。

### P5 · S3：重定向（omnirt_v1 主 + omnirt_v2 rescue）
**重建，不复用现有 43 个目录**。理由：① 它们带超预算草稿代理，`scene.xml`/`scene_act.xml` 至少要重生；② 上游 `converted/retargeted/trimmed` npz 在 **E145 的 RESULT_ROOT** 下，新 run root 命不中 `pipeline.sh` 的存在性守卫（311/346/398/454 行），无论如何会重解；③ 要求"全流程"。
**代价缓解**：19 个重叠 case 变成**可复现性交叉校验** —— 比对 E206 `trimmed_npz["qpos"]` 与既有 `0/trajectory_kinematic.npz` 最大绝对偏差 → `$E/s3_retarget/reproducibility_vs_e145.tsv`。相同输入 + 相同 omnirt_v1 参数应当复现；**不符即为值得单独上报的发现**（管线有隐藏非确定性）。
- `KEEP_GOING=1` + `run_stage2b.py --raw-contact-tsv .../raw_contact_pass_5cm_move2only.tsv --retarget-variant-id omnirt_v1 --target-variant-id ref_fk --execute --allow-legacy-stage2b-wrapper`（经 `E174/run_stage2b_queue.py` 并行）；v2 rescue **只对本轮 v1 的 `omniretarget_infeasible` 集**（docs 08 §rescue），不用历史标签、不靠文件名猜。
**退出检查**：manifest 每行有终态（`stage2b_ok` 或 `06_failure_taxonomy.md` 编码）；rescue 集精确等于本轮 v1 infeasible 集；每个产出目录 `scene.xml`+`scene_act.xml` 的 `n(object_collision*) ≤ N_MAX`。

### P6 · S4 目标门 + 视觉 QC + 接触保真 G8
`run_target_gate.py` → `make_visual_qc.py` → `render_visual_qc_package.py`（osmesa）。
G8 复用 `scripts/eval/runners/eval_E176_contact_fidelity.py`（其 `load_reference_qpos` 在轨迹存在时不碰 rollout npz → **可 pre-CEM 运行**）；**需改一处**：`eval_E176_contact_fidelity.py:172-176` 的 `> 9` 硬编码参数化为 `--max-object-geoms`（默认 9，向后兼容）。新建 `$ED/build_proxy_manifest.py` 产出 manifest 列（`case_id, object_key, scene_act, trajectory, contact_mask, object_geom_count, compiled_robot_object_pair_count, source_e206_config_act`）。
**退出检查**：`target_gate_status=pass ≥ 90%` 且每个失败带 taxonomy 码；接触保真 **每个物体组 p90 ≤ 0.08 m**；每个过门 case 有回放 MP4 + 关键帧。

### P7 · S5 handoff + 双 arm 场景/override（单变量）
场景链（`$ED/build_arm_scenes.py`，仿 `E204_E205/build_arm_scenes.py`）：
1. dcv3 `scene_act.xml`（球手 + lowgeom）→ `patch_hand_collision.py::patch_scene(hand_collision_variant_id="rubber_hull")` → `scene_act_E206_lowgeom_rubberHull.xml`
2. → 重建 robot↔object pair 为 **2N 手对**（condim 4, friction "2 1"）**+ 0 腿对**，用 `E175/build_nonbox_multigeom_production.py::is_robot_object_pair`(line 256) 精确剥离 → **`scene_act_E206_lowgeom_noPRG.xml`**
3. → 复制 + 加 **16N 腿对**（condim 1, margin 0, gap 0, solref "0.008 1"）→ **`scene_act_E206_lowgeom_PRG.xml`**

**单变量断言（逐 case，零容忍）**：忽略 pair 后 `stripped_signature(noPRG) == stripped_signature(PRG)`；`pair_count` noPRG`==2N` / PRG`==18N`；noPRG 零残留腿↔物对；两 arm `lh/rh` 均 `mjGEOM_MESH`；两 arm `object` body `gravcomp==0`（无 G1）；全部 `object_collision*` 为 `mjGEOM_BOX`。

**Override**（`$ED/build_overrides.py`，base reward `E167A_zOnlyBody`）：

```yaml
# core4d_E206_<case>_lowgeom_noPRG.yaml
defaults: [core4d_dcv3_omnirt_v1_ref_fk_<case>, _self_]
scene_name: scene_act_E206_lowgeom_noPRG
object_collision_sdf_mode: union            # 不继承，必须显式重声明
object_collision_sdf_batch_groups: true
leg_object_penalty_scale: 0
leg_object_penalty_geom_names: []
cem_leg_gate_enabled: false
```
```yaml
# core4d_E206_<case>_lowgeom_PRG.yaml
defaults: [core4d_dcv3_omnirt_v1_ref_fk_<case>, _self_]
scene_name: scene_act_E206_lowgeom_PRG
object_collision_sdf_mode: union
object_collision_sdf_batch_groups: true
leg_object_penalty_scale: 2.0
leg_object_penalty_geom_names: [<16 个 LOWER_BODY_GEOMS>]
cem_leg_gate_enabled: true
cem_leg_gate_geom_names: [<同上 16>]
```

**Hydra compose 审计**（无需 GPU，`_compose()` from `E204_E205/build_overrides.py:63`）：两 arm hand-gate 均须 == `E163_HAND_GATE`（`min_sdf −0.010 / max_viol 0.10 / hard_floor −0.020`，即**无 A2**）；**跨 arm diff 审计** —— composed config 差异恰为上述 5 键（复用 `E176/build_lowgeom_production.py:144-153` 白名单 diff）；CEM 后再用 `E190/audit_38case_noprg_rl.py::prg_negative_ok`(line 49) 对 `config_act.yaml` 做独立事后确认。

**快照（rule 7 / 10b）**：`build_arm_scenes.py` 调 `snapshot_scenes.sh E206 <tasks>` 并对活跃 case 的 `scene.xml/scene_act*.xml/scene_act_meta.json/task_info.json` `git add -f`；启动脚本**第一步**就调它。

### P8 · CEM：2 arm × N case，8 卡
`$ED/run_e206_cem.py`（改自 `E204_E205/run_e204e205_cem.py`）：GPU slot 池、**逐 arm 独立 `output_dir`**、以该 arm rollout npz 为键 skip-already-done、`--stage {full,smoke,throughput}` 防串档、任一 `cem_fail` 非零退出。
两处相对 E204/E205 的有据偏离：`+use_torch_compile` 由 `admission_decision.json` 决定（**不再硬编码 false**，本机 `Python.h` 存在）；`object_collision_sdf_batch_groups: true` 两 arm 都开（E176 已验证数值等价），**因此不是 arm 变量**。
启动器 `scripts/launch/active/run_E206_2arm_8gpu.sh`。**先跑 2 case × 2 arm 的 smoke 并 diff 两份 `config_act.yaml`，再按 A2b 优先级序发全量队列。**
**退出检查**：`e206_cem_summary.json` 中 `cem_ok` 覆盖 100% 作业（2×n_cases），`cem_fail_rc*` 为 0；抽检 ≥4 份 `config_act.yaml` 确认运行时六字段 arm 契约。

### P9 · 评测：配对 noPRG vs PRG + 强制视觉复核
`scripts/eval/runners/eval_E206_arm_ablation.py`（仿 `eval_E204E205_arm_ablation.py`）—— **关键改进：两 arm 都从新鲜 rollout 同尺打分**（E204 的 PRG 行是读 TSV，存在基线不对称）。
- **E201 14-gate 漏斗**（`E201/funnel_config.py`，先 `assert_monotonic()`）+ 冻结 **6-gate**（`docs/EVAL_METRICS_12GATE.md`）双口径。全部 `variant=orig`、单 arm family → narrow-pass ⇒ `L3_auto`。
- **配对统计**：同一 case 集；逐门 **mean / std / worst**（rule 5）+ 配对 delta `PRG − noPRG` 带符号 + 逐物体拆分（5 desk + 5 chair）。
- **基线列**：desk007 的 9 例与 E174 同尺（同 3cm 掩码、同 case）并列，让 F7 配对 bug 的修复直接可见。
- **样本量诚实标注**：逐物体表必须带 n；chair020(n=1)、chair005/desk020(n=2) **不出 per-object arm 推荐**。
- 报表 `gen_E206_two_arm_workbook.py` → `e206_two_arm.xlsx`（`funnel_summary / per_gate / per_object / paired_delta / vs_E174_desk007`）。
- **视觉复核（rule 5 §Visual Evaluation 强制）**：viser 并排 `review_player.sh E206ARM` + `$ED/render_qc.py` 关键帧。覆盖 = **每物体 × 两 arm**，外加**全部 L3 case**（防 reward hacking）与**全部单门 L1 case**（防指标假象）；逐 case `USE/DO_NOT_USE` + 失败模式 → `user_manual_review_filled.tsv`；**数值与视觉矛盾必须逐条显式列出**。
- **顺带做掉 log234 §9 的廉价根因项**：**D1** 接触目标落在无代理区域的比例（= G8 产物，§9 item 1）；**D2** 逐物体局部漏检图（按 object-local Z 分桶：台面 vs 桌腿 / 椅面 vs 靠背，§9 item 4）；**D3** 参考轨迹自身的腿-物距离（复用 `E175/build_nonbox_multigeom_production.py:441` 的 `reference_first5_diagnostic`，§9 item 3）—— 判断 `leg_pen` 是数据属性还是控制失败。

### P10 · 收尾
写 `log/295_E206_desk_chair_move2_dcv3_noprg_prg.md`（结构照 log234：一句话结论 / 目的与假设 / 冻结配置 / 运行命令 / Funnel / 结果 / 视觉核验 / Claims / 分析 / Yield 与下一步 / 结果路径 / 改动文件含 `scene_snapshot/`）；`EXPERIMENT_TRACKER.md` 加 R291/R292；更新 `progress.md`；`build_log_index.py` 重建 `log/INDEX.md`。提交 `exp(core4d): E206 desk/chair move2 dcv3 全流程 + lowgeom 碰撞代理 + noPRG/PRG 双arm`。

## Claims 与量化成功标准

| ID | Claim | 数值门 |
|---|---|---|
| **C0** | dcv3 S0→S6 在 desk+chair move2_* 上闭合 | ✅ S1 部分已达成：164 候选在两阈值下各 100% 有终态，落地 74 case / 9 物体（P2.3b 人工弃用 chair021 后 **65 case / 8 物体**）。余下：Stage2b 成功 ≥ **90%** of `clean_reviewed`；S4 机器门 pass ≥ **90%** of Stage2b 成功；**0** 条无法解释的丢失 |
| **C1** | 每个在范围物体得到满足契约的 ≤N_MAX box 代理 | ✅ **已达成（8/8，`all_pass=true`）**：G1,G2,G3,G4,G5,G7,**G10** 全过，G6 零豁免。余下 G8 逐物体 `p90 ≤ 0.08 m`（P6 执行） |
| **C2** | lowgeom 相对现有草稿是严格改进 | geom 数 `37–127 → ≤N_MAX` 覆盖 100% 物体，**且** `mesh→proxy p90` 相对 26-cell 草稿退化 ≤ **0.04 m** |
| **C3** | 吞吐被重新实测并给出准入 | `admission_decision.json` ≥16 行实测；N_MAX=16 下中位全预算墙钟 ≤ **120 min/task**；按 P1 实际 case 数投影的双 arm 队列 ≤ **48 h**；队列优先级序已冻结。E176 的 3.0s 门显式作废并记录理由 |
| **C4** | 两 arm 严格单变量 | **100%，零容忍**：语义 diff 恰为 16N 腿对；pair 2N/18N；两 arm 均 rubber-hull mesh 手且 `gravcomp=0`；composed config 恰差 5 键；两 arm hand-gate 均 == `E163_HAND_GATE` |
| **C5a** | **F7 配对修复带来实质接触改善**（核心科学看点） | 在 **desk007 的 9 例**上（与 E174 同掩码 3cm、同 case）：`hand_object_physics_contact_in_mask_frac` 均值相对 E174 desk007 基线 **提升 ≥ +0.15 绝对值**（E174 非 box 总体均值 0.379，desk007 分组值以 `results/E174` 实际 TSV 为准） |
| **C5b** | arm 对比 | 仅当 `leg_pen` narrow 通过率 delta ≥ **+10 pp** 且 L3 数 delta ≥ 0 才宣称 "PRG 胜"；否则按实测符号宣称 "无分离" 或 "noPRG 胜"。**14 门全部带 mean/std/worst** |
| **C5c** | **物理通过率只报告，不设门** | 逐 arm 逐物体公布 L1/L2/L3 与 6-gate 计数 + mean/std/**worst**。**无 pass/fail 门** |
| **C6** | 强制视觉复核 | 覆盖集 100% 有 `USE/DO_NOT_USE` + 失败模式；数值↔视觉矛盾逐条列出 |
| **C7** | 可复现 | `results/E206/scene_snapshot/manifest.txt`（git HEAD + sha256）在**首条 CEM 之前**写好；活跃 case scene XML `git add -f`；`results/E206_pre/scene_snapshot` 保留覆盖前状态 |

**总判定**
- **SUCCESS** = C0,C1,C2,C3,C4,C6,C7 全过 **且** C5a 过。
- **PARTIAL SUCCESS** = C0,C1,C2,C3,C4,C6,C7 过但 **C5a 不过**。仍交付：闭合的 desk/chair dcv3 路径 + 可度量的 ≤N_MAX 代理方法 + 重新议定的吞吐政策 + 干净的双 arm 数据集。**C5a 不过即证伪 log234 §9 的"代理保真是瓶颈"假设，本身是有价值的结论。**
- **FAIL** = C0/C1/C4/C7 任一不过；或 C3 不过却未记录缩范围就发了全量队列。
- **显式不作为判据**：desk/chair 的 CEM 通过率。**0/N 且 C0–C4+C6+C7 全绿 = PARTIAL SUCCESS，不是失败**（desk 历史 2/14，chair 无先例，设通过率门就是拍脑袋）。

## 风险 / 缓解

| # | 风险 | 缓解 | 触发点 |
|---|---|---|---|
| R1 | ~~S1 落地 case 太少~~ **已消解**：实测 74 case / 9 物体，远高于停机线 12 | — | P1 已过 |
| **R1b** | ~~改 5cm 造成口径断裂~~ **已消解**：实测 5cm 只多 2 个 case，遂改回 3cm 单一口径，与全线同尺，桥接层取消 | — | P1 已过 |
| **R1c** | 队列规模：P2.3b 后 **65×2 = 130 条**（原 148），最大 box 数 12（原 16）→ 压力已降 | A2 = 48h 上限（slot 池 + skip-already-done，可中断续跑）；**A2b 优先级排队**保证腰斩时每物体有样本、C5a 一定有数据；再不够走 A3 | P3 / P8 |
| **R1d** | **chair020 n=1、chair005/desk020 n=2** —— 这三个物体没有 per-object 统计力 | 只并入总体，**不出 per-object arm 推荐**，log 中逐物体标 n；C5b 的 arm 结论以总体 74 例为准 | P9 |
| R2 | E176 `center_inside_count` 卡死全部椅子（**已实测必然发生**） | P2.1 用 `cavity_metrics` 替换；**不是简单删断言**，保留腔体保证 | P2.1 |
| R3 | ~~N=9 椅子腔体过填 42–45%~~ **已消解**：P2.3b 人工重摆 box 后最高过填 0.25（chair020），**G6 零豁免**，无需任何结论边界声明。实装最大 box 数 12 | — | P2.3b 已过 |
| R4 | P5 覆盖 43 个既有任务目录（**已实测必然发生**） | P0 覆盖前快照 + `git add -f` + 消费者 grep | P0 |
| R5 | OmniRetarget 全量重解耗时 + CVXPY infeasible | `KEEP_GOING=1` 逐 case 隔离（`pipeline.sh:19-22`）；`run_stage2b_queue.py` 并行；v2 Phase-4 rescue；重定向墙钟单独上报 | P5 |
| R6 | N=16 吞吐超预算 | A3 阶梯：`batch_groups` → 回落 N_MAX=9 → 缩物体 → **两 arm 对称**降预算，每步记录 | P3 |
| R7 | ~~手部几何造成 arm 混淆~~ **设计已消解**（noPRG≡E167A 为 reward arm 定义，场景按 E200 实现用 rubber-hull） | 共享 rubber-hull 中间态 + `stripped_signature` 断言 + 两 arm `lh/rh==mjGEOM_MESH` | P7 |
| R8 | union fail-closed 遇非 box geom 直接崩 | G2 在代理构建 / 模板安装 / arm 场景构建三处独立断言，**全在 GPU 时间之前** | P2/P4/P7 |
| R9 | 只有 geom 0 有调过的 `object_floor` pair | P7 按同 solref/friction 补齐 N 条地面 pair，并记为 diff | P7 |
| R10 | `eval_E176_contact_fidelity.py` 硬编码 `>9`（**N_MAX=16 时必然崩**） | 参数化 `--max-object-geoms`（默认 9，向后兼容） | P6 |
| R11 | 新物体类上的 reward hacking / 指标假象 | C6 覆盖集刻意包含**全部 L3** 与**全部单门 L1** | P9 |
| R12 | GPU 争用（E203 踩过：外部压测让 MPC step 涨到 211s） | driver slot 池 + 可续跑；`GPUS=` 限卡；`--stage` 隔离 | P8 |

## 改动文件

**新建**（全部在 E206 命名空间）：本 plan、`log/295_*`、`$ED/{e206_common, lowgeom_proxy_v2, audit_lowgeom_contract, install_lowgeom_templates, build_nonbox_template_review, report_s1_funnel, measure_throughput, build_proxy_manifest, build_arm_scenes, build_overrides, run_e206_cem, build_arm_review_tsv, render_qc}.py`、`scripts/eval/runners/eval_E206_arm_ablation.py`、`scripts/eval/reports/gen_E206_two_arm_workbook.py`、`scripts/launch/active/run_E206_{data_pipeline,2arm_8gpu}.sh`、`examples/config/override/core4d_E206_<case>_lowgeom_{noPRG,PRG}.yaml`。

**修改（3 个，全部增量且默认关）**：

| 路径 | 改动 | 动 SPIDER core？ |
|---|---|---|
| `$DCV3/stages/s2_templates/build_or_audit_templates.py` | 加 `lowgeom_collision_geoms()` + `--nonbox-collision-policy`，**默认维持既有行为** | 否（dcv3 管线代码，正是本实验对象） |
| `scripts/eval/runners/eval_E176_contact_fidelity.py` | `--max-object-geoms`（默认 9）替代硬编码 `>9` | 否 |
| `scripts/eval/wrappers/review_player.sh` | 注册 `E206ARM` | 否 |

**SPIDER core（`spider/`）零改动**。`spider/config.py:69-79` 的 union fail-closed 被当作**必须绕开设计的硬约束**（全 box 代理），而非可放松项。若中途确需改 core，须先提出、隔离到单一 commit、实验结束回滚、并写进 log 的改动文件表。

**被覆盖的数据产物**（非 git，有快照）：16 个源模板 `example_datasets/.../{desk,chair}*_person{1,2}/{scene.xml,task_info.json}` + ~43 个 `dcv3_omnirt_v1_ref_fk_{desk,chair}*/` —— 由 P0 的 `results/E206_pre/scene_snapshot` 与 P7 的 `results/E206/scene_snapshot` 双快照保护。

## 下一步

P0 → P1 → P2（人工复审卡点）→ P3（吞吐冻结）→ P4–P8 → P9 → log295 + tracker。
下游：两 arm × desk/chair 数据交 RL 训练对比择 arm（与 bucket 线 E204/E205 的 arm 结论横向印证）。
