# Clean Contact Benchmark v1 + 恒定差距机制（诊断 v3）

日期：2026-06-05
依据：`reverify_v3_report.md` 的数据驱动复查 + 本文对 SPIDER reference 来源的代码级核查。
冻结产物：
```
workspace/exp_diagnostic_v3/results/benchmark/
├── clean_benchmark.tsv            ← 全 24 case 的 tier/decision/reason + 指标
└── clean_benchmark_manifest.tsv   ← 入选 10 case 的可复跑路径（scene/baseline/omni/mask/override）
脚本：workspace/exp_diagnostic_v3/scripts/select_clean_benchmark.py（阈值改这里）
```

---

## Q1 — 恒定 −2.9pp 差距来源于什么？

**答：是的，根因正是"SPIDER 把 OmniRetarget 的输出当作 reference 来跟踪优化"，因此在 5cm 这类"接近度"指标上它结构性地**无法超过、只能逼近** OmniRetarget。** 这是代码级证实的，不是推测：

### 证据链（file:line）

1. **评测表里的 "OmniRetarget" 方法 = `trajectory_kinematic.npz`**
   `variants.tsv` 的 `omni_qpos_path` 字段就是 `example_datasets/.../{task}/0/trajectory_kinematic.npz`（E143 metric 审计已确认 OmniRetarget 行从此文件取 qpos）。

2. **SPIDER 把同一个 `trajectory_kinematic.npz` 作为跟踪 reference 载入**
   - `spider/preprocess/hand_snap_ik.py:20`、`spider/postprocess/get_humanoid_tracking_err.py:48-60` 都从 `trajectory_kinematic.npz` 取 `qpos` 作为 reference。
   - E143 override config（`examples/config/override/core4d_E143_*.yaml`）`task: ...` 指向同一 processed 目录，SPIDER 由此加载该 kinematic 轨迹作为 `qpos_ref`。

3. **SPIDER 的 reward 全程跟踪这个 ref**
   `spider/simulators/mjwp.py:638-682`：reward 解包 `qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref, body_xpos_ref, ...` —— **全部来自加载的 kinematic ref**。`mjwp.py:681` 起的 `qpos tracking`、`contact_hdmi_rew`(976+) 都以 ref 的位姿/接触点为目标。

4. **接触目标 = ref 的手部 FK（OmniRetarget 的手位置）**
   config `contact_hdmi_target_source: ref_fk`（`spider/config.py:211-212`：`"ref_fk" | "external"`）。`ref_fk` 意为**接触目标 = 参考轨迹的 end-effector 前向运动学位置 + offset**（`mjwp.py:989-1024` 用 `hand_approach_body_ids` 的 wrist + `contact_hdmi_eef_offset` 构造目标）。也就是说，**SPIDER 想把手放到 OmniRetarget 把手放的地方**。

### 机制解释为什么是"恒定且必然低"

- OmniRetarget 是**无物理的运动学解**，它的手为了贴合常常**插进箱子**（v3 已证：它的"接触"≈穿透，深穿透 29.4%）。
- SPIDER 以 OmniRetarget 的手位置为**目标**，但在**物理仿真**里优化：它不能穿透（有 `hand_object_deep_penalty`、`robot_object_penalty`、CEM gate）。于是它只能把手停在 OmniRetarget 目标点的**外侧一层**——逼近但不越界。
- 结果：在"距离 ≤ 5cm 帧占比"这种以 ref 为锚的接近度指标上，SPIDER = "OmniRetarget 目标 − 一层物理可行的薄壳"，**永远略低于把目标当成可达的 OmniRetarget 本身**。这个"薄壳"厚度由物理约束/接触 sigma 决定，**对所有 case 基本一致 → 恒定 −2.9pp**。
- 这也解释了为何差距**与数据质量解耦**：无论 case 干净还是脏，SPIDER 都在跟同一个 OmniRetarget 参考、扣同样的物理薄壳。脏数据让**两条曲线一起降**（绝对接触低），但**薄壳厚度不变**（gap 恒定）。

### 重要推论
- **在"以 OmniRetarget 为锚的接近度指标"上要求 SPIDER 超过 OmniRetarget，是逻辑上不自洽的**——它在跟踪 OmniRetarget，且被物理约束扣一层。这正是 E142/E143 报"0/24 超过"的根本原因，不是 SPIDER 弱。
- 要让 SPIDER "赢"，必须换一个**不以 OmniRetarget 为锚、且奖励物理真实接触**的指标：
  - `frac(SDF ∈ [0, 2cm])`（无穿透贴合），SPIDER 该反超（OmniRetarget 大量是 SDF<0）；
  - 或以 **GT 动捕接触**（`raw_min_dist_m`）为锚，而不是以 OmniRetarget 的手位置为锚。
- 若想真正抬高 SPIDER 的 5cm 数：得**改 reference 本身**（ref repair / 用 GT 指尖目标替代 OmniRetarget wrist 目标，即 v2 报告 B6 路线），而不是调 SPIDER reward 权重。

---

## Q2 — 按阈值选数据

采用阈值（写在 `select_clean_benchmark.py` 顶部，可调）：
- **物体总旋转 < 45°**（你的建议；数据上 30–45° 之间只有 box026_139_p1=34、139_p2=45 两个边界，45° 把它们纳入边界层）
- **物体抬升 > 0.30 m**（你的建议）
- **GT 动捕接触 ≥ 0.30**（附加守门：人本身确实在接触箱子，排除传递/空抓）
- **显式排除传递动作**（box026_141_p1/p2）

24 case → **10 入选 / 14 排除**。排除明细见 `clean_benchmark.tsv`，主因：抬升不足 6 个、旋转过大 4 个、传递 2 个、无物体位姿(bucket缺scene) 2 个。

---

## Q3 — Clean Benchmark v1（冻结）

分两层。**Tier-1 是算法优化的主基准**；Tier-2 是带 caveat 的扩展集，单独报告、不混入主均值。

### Tier-1 core（6 case）— 主基准

| case | object | rot° | lift m | gtC | Omni 5cm | Spider 5cm | Δ(S−O) |
|---|---|---:|---:|---:|---:|---:|---:|
| box021_035_p1 | box021 | 2.1 | 0.350 | 0.34 | 0.82 | 0.79 | −0.031 |
| box021_035_p2 | box021 | 4.4 | 0.371 | 0.55 | 0.80 | 0.77 | −0.030 |
| box021_029_p2 | box021 | 1.0 | 0.365 | 0.41 | 0.76 | 0.71 | −0.053 |
| box023_person2 | box023 | 5.2 | 0.447 | 0.36 | 0.51 | 0.49 | −0.022 |
| box004_083_p1 | box004 | 23.8 | 0.321 | 0.52 | 0.65 | 0.60 | −0.049 |
| box004_082_p1 | box004 | 24.2 | 0.428 | 0.43 | 0.60 | 0.56 | −0.037 |

- 覆盖 3 个 object（box021×3, box023×1, box004×2）。
- box023_person2 注"箱子偏小"；box004_082_p1 注"物体动捕抖动"（已量化坐实，但抬升最好、RL 过）。
- Tier-1 均值：Spider−Omni 5cm = **−0.037**（这是要打的靶子）。

### Tier-2 conditional（4 case）— 扩展，单独报告

| case | rot° | lift | gtC | Δ5cm | caveat |
|---|---:|---:|---:|---:|---|
| box026_039_p2 | 1.3 | 0.315 | 0.71 | −0.041 | walk-up + ref 初始接触解算错 → **先 ref repair** |
| box026_039_p1 | 7.2 | 0.302 | 0.67 | −0.025 | walk-up + 单手 + p1 抬前抖动 |
| box026_139_p1 | 33.7 | 0.339 | 0.54 | −0.014 | 旋转 34° + ref 腿接触；mixed |
| box026_20231023_139_p2 | 45.0 | 0.372 | 0.54 | −0.050 | 旋转边界 45°(max69)；曾被误标~180 |

### 使用约定
1. **算法优化以 Tier-1 为唯一主基准**，报 mean+std+worst（按实验规范，禁止 cherry-pick）。
2. 评测指标用 **无穿透贴合 `frac(SDF∈[0,2cm])` + 5cm 近场 + 穿透惩罚 + `object_floor_contact_frac` 跃升探测器**；**不要**用 "physics contact"（=穿透）。
3. **不以"超过 OmniRetarget 5cm"为成功判据**（见 Q1，逻辑不自洽）；成功判据应是"无穿透贴合 ↑ 且 腿穿透不退化 且 不下沉摔箱"。
4. Tier-2 仅用于鲁棒性/扩展报告；box026_039 两个要先做 ref repair 才能进主比较。
5. 复跑路径见 `clean_benchmark_manifest.tsv`（每 case 的 scene_xml / baseline / omni_qpos / mask / override 齐全）。

---

## 附：数据可复现
- 阈值与分层逻辑：`scripts/select_clean_benchmark.py`（改 `ROT_MAX_DEG`/`LIFT_MIN_M`/`GT_CONTACT_MIN`/`TIER2_CAVEAT` 即可重选）。
- 全量逐 case 指标：`results/trajectory_analysis/object_trajectory_metrics.tsv`。
- 旋转/抬升/差距图：`results/figures/reverify_*.png`。
