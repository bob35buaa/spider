# E170 实验计划：Box021 全 28 Case PRG 泛化验证

日期：2026-07-18

实验方向：`core4d`

Phase：33

状态：执行中；A100 主 full 与 qpos0-only recovery full 正在运行

---

## 0. 决策摘要

用户复核 E169 指标和视频后，认为 `PRG` 整体质量可以接受，决定将同一固定配置扩展到 E168 的全部 28 条 Box021 person-case。E170 是单配置泛化验证，不再做 `P/R/G` 因子消融：

```text
28 Box021 cases
= 4 条 E169 PRG full 结果直接复用
+ 24 条 E170 PRG new full CEM
```

E168 的 28 条 `E167A_zOnlyBody` 结果作为逐 case frozen baseline，只读复用，不重跑。E170 重新以统一 evaluator 评测 28 条 PRG，并生成 E168 B0 与 PRG 的 paired delta、视频对照和新一轮人工审查。

责任边界已经固定：Codex 负责 28/28 指标核验、分层视觉抽查、异常清单和机器建议；用户负责查看完整 paired review package、给出 28/28 最终人工标签，并对是否推广 PRG 作最终拍板。Codex 的视觉抽查不得写入或覆盖用户的 `manual_*` 字段。

E169 的 gate-health 结论仍然有效：当时 G-on 为 `0/16` health pass。E170 可以验证 PRG 输出的实际可用性，但不能因为视频可用就宣称 hard gate 已健康。轨迹质量和 gate 机制健康度必须分开裁决。

执行资源沿用当前约定：只使用远程 A100 `GPU 0,1,2,3` 四卡并行；不使用本地 GPU、A6000 或 A100 `4-7`。

---

## 1. Context

### 1.1 E168 全量基线

E168 已完成 Box021 全 28 case 的 CEM、量化评测与人工审查：

- `13 USE / 15 DO_NOT_USE`；
- 15 条失败中，`14/15` 含 lower-body/object 非法接触或非法支撑症状；
- 28 条覆盖 `25 omnirt_v1 + 3 omnirt_v2`；
- p1/p2 继续作为独立 person-case，不合并为 sequence-level 单条结果；
- frozen baseline 表为 `results/E168/s6_downstream/cem/eval/box021_all28_reviewed/e168_case_metrics.tsv`。

E170 不改变 E168 retarget route。每条 case 继续使用其 E168 已选中的 `retarget_variant_id`、`target_variant_id=ref_fk`、raw contact mask 和 rubber-hull hand collision scene；三条 v2 case 不回退到 v1。

### 1.2 E169 选择依据与边界

E169 在 4 个代表 case 上完成 `2^3` factorial。PRG 的平均 `leg_penetration_frac=0.089`，用户当前复核认为整体指标与视频可接受，因此将其作为 E170 唯一候选。

同时保留以下已知边界：

- E169 自动 gate-health 未通过，PRG 可能在 combined body/hand/leg gate 不足时走 `least_violation` fallback；
- E169 旧人工标签由当时较严格的逐 arm 审查产生，不能直接代替用户当前判断或 E170 的全量 paired review；
- `leg_object_physics_contact_frac` 在 P-off scene 中结构性为 0，在 P-on 中才会产生 contact，因此不能把 E168 的 0 与 PRG 的非零值直接解释为回归；
- PRG 是否可作为 Box021 默认配置，必须由 28-case retention、recovery 和视觉结果共同决定。

### 1.3 E170 不回答的问题

E170 不重新分解 P/R/G 因果，不调 reward/gate/pair 参数，不修复 gate selector contract，也不扩展到 Box004/Bucket004。它只回答一个问题：

> 冻结的 E169 PRG 配置在 Box021 全 28 person-case 上，能否稳定提高可用率，并保住 E168 已可用轨迹？

### 1.4 双轨裁决与人工 authority

E170 同时保留两套不互相覆盖的结果：

```text
manual_operational_use = 用户最终 manual_use_decision == USE
strict_release_usable = 数值硬门全部通过 AND 用户最终 manual_use_decision == USE
```

用户可以把存在数值告警但视觉可接受的轨迹判为 `manual_operational_use=true`；该判断不会删除数值失败，也不会把该轨迹改写成 `strict_release_usable=true`。全量 strong/partial/fail 的机器建议以 `strict_release_usable` 为主，人工 operational use 率并列报告，最终推广裁决由用户作出。

---

## 2. Claims

| Claim | 最低证据 |
|---|---|
| C0：28-case provenance 完整 | 28 条均绑定 E168 source row、retarget variant、trajectory/contact-mask/scene SHA；4 条 reuse 额外绑定 E169 artifact SHA |
| C1：执行完整且无重复计算 | 分析集精确为 `4 reused E169 PRG + 24 new E170 PRG`；E168 baseline 和 E169 reuse 均不重跑 |
| C2：PRG 改善 E168 失败集 | 对 15 条 E168 `DO_NOT_USE` 报告逐 case paired delta 和用户 fresh review；至少 `9/15 strict_release_usable` 才支持强泛化，人工 operational recovery 另报 |
| C3：PRG 不破坏 E168 可用集 | 13 条 E168 `USE` 中至少 `12/13 strict_release_usable`，且无新增 fall、明显 object kick、踩箱或借箱支撑；人工 retention 另报 |
| C4：核心量化质量可接受 | 每条报告 body-z、raw/clean3 contact、release-3mm、hand penetration、lower-body、tracking、fall 和 motion-health 完整指标 |
| C5：视觉改善不是指标假象 | Codex 完成分层视觉抽查和异常扩查，用户对 28 条 B0-vs-PRG paired video 给出最终标签；无明显穿箱、踩箱、借箱支撑、爆姿或箱体被踢飞 |
| C6：gate 输出可诊断 | 24 条新结果和4条 reuse 均有 leg valid/selected/fallback/SDF diagnostics；质量结论与 gate-health 结论分列 |
| C7：结论覆盖 person/variant 分层 | 分别报告 p1/p2、E168 USE/DNU、omnirt_v1/v2、sequence/action/obstacle 分组，不用 overall mean 掩盖小组失败 |
| C8：结果可复现 | checkpoint commit、dirty-state audit、同步文件 SHA、effective config、scene semantic diff、root/outdir NPZ、日志、视频、评测表和人工标签均有路径与 SHA |

---

## 3. 固定 Case 集与复用清单

### 3.1 E169 PRG 直接复用的 4 条

| Case | E168 人工基线 | E170 执行方式 | Frozen artifact |
|---|---|---|---|
| `box021_20231018_032_p1` | DO_NOT_USE | reuse E169 PRG full | `results/E169/cem/full/E169_box021_20231018_032_p1_PRG.npz` |
| `box021_20231018_033_p1` | DO_NOT_USE | reuse E169 PRG full | `results/E169/cem/full/E169_box021_20231018_033_p1_PRG.npz` |
| `box021_20231020_020_p2` | DO_NOT_USE | reuse E169 PRG full | `results/E169/cem/full/E169_box021_20231020_020_p2_PRG.npz` |
| `box021_20231020_023_p2` | USE | reuse E169 PRG full | `results/E169/cem/full/E169_box021_20231020_023_p2_PRG.npz` |

复用采用引用加 SHA 的方式，不复制或改写 E169 原始结果。E170 manifest 必须包含：

```text
execution_source = E169
reused_full = true
source_variant = E169_<case>_PRG
source_result_npz / source_outdir_npz / source_config / source_video
source_*_sha256
```

复用前重新检查 root/outdir qpos 一致、qpos finite、scene SHA、16 pairs、reward diagnostics 和 gate diagnostics。任何 SHA 漂移都使对应 row 进入 `reuse_blocked`，不得静默重跑后仍标记为复用。

### 3.2 E170 新跑的 24 条

| Case | Retarget | E168 人工基线 |
|---|---|---|
| `box021_20231011_034_p1` | omnirt_v1 | USE |
| `box021_20231011_034_p2` | omnirt_v2 | USE |
| `box021_20231011_036_p1` | omnirt_v1 | DO_NOT_USE |
| `box021_20231011_036_p2` | omnirt_v1 | USE |
| `box021_20231011_037_p1` | omnirt_v1 | USE |
| `box021_20231011_037_p2` | omnirt_v1 | USE |
| `box021_20231011_038_p1` | omnirt_v1 | USE |
| `box021_20231011_038_p2` | omnirt_v1 | USE |
| `box021_20231018_028_p1` | omnirt_v2 | DO_NOT_USE |
| `box021_20231018_028_p2` | omnirt_v1 | DO_NOT_USE |
| `box021_20231018_029_p1` | omnirt_v1 | DO_NOT_USE |
| `box021_20231018_030_p1` | omnirt_v1 | DO_NOT_USE |
| `box021_20231018_030_p2` | omnirt_v1 | DO_NOT_USE |
| `box021_20231018_031_p2` | omnirt_v1 | DO_NOT_USE |
| `box021_20231018_032_p2` | omnirt_v1 | DO_NOT_USE |
| `box021_20231018_033_p2` | omnirt_v1 | DO_NOT_USE |
| `box021_20231018_034_p2` | omnirt_v1 | DO_NOT_USE |
| `box021_20231018_035_p2` | omnirt_v1 | DO_NOT_USE |
| `box021_20231020_019_p1` | omnirt_v2 | DO_NOT_USE |
| `box021_20231020_019_p2` | omnirt_v1 | USE |
| `box021_20231020_020_p1` | omnirt_v1 | USE |
| `box021_20231020_022_p1` | omnirt_v1 | USE |
| `box021_20231020_022_p2` | omnirt_v1 | USE |
| `box021_20231020_023_p1` | omnirt_v1 | USE |

清单 authority 是 E168 `box021_all28_reviewed/evaluated_manifest_snapshot.tsv`，不是目录扫描结果。builder 必须断言总数28、case_id唯一、reuse交集4、new差集24、v1/v2为25/3。

---

## 4. 冻结 PRG 配置

E170 不产生新的算法 arm。24 条新 full 必须与 E169 PRG 的 effective config 字段级一致，只有 case-specific source path、scene name/path、contact mask 和输出路径可以变化。

### 4.1 P：下肢-物体显式物理碰撞

复用 E169 的16个 lower-body geom 集，为每个新 case 从 E168 rubber-hull scene 生成 E170-scoped sidecar。只新增16个到 `object_collision` 的 pair：

```xml
<pair geom1="<lower_body_geom>"
      geom2="object_collision"
      solref="0.008 1"
      margin="0"
      gap="0"
      condim="1" />
```

不得修改共享 E168 scene，不得调整 friction、contype/conaffinity、object mass 或 actuator gains。XML semantic diff 除16个 pair 外必须为空。

### 4.2 R：固定 lower-body soft penalty

```yaml
leg_object_penalty_scale: 2.0
leg_object_penalty_margin_m: 0.02
leg_object_penalty_geom_names: <E169 frozen 16 geoms>
leg_object_penalty_geom_ids: []
leg_object_penalty_gate_source: always
leg_object_penalty_start_eval_time: 0.0
leg_object_penalty_end_eval_time: 999.0
```

### 4.3 G：固定 CEM lower-body candidate gate

```yaml
cem_leg_gate_enabled: true
cem_leg_gate_geom_names: <E169 frozen 16 geoms>
cem_leg_gate_geom_ids: []
cem_leg_gate_min_sdf_m: 0.005
cem_leg_gate_max_violation_pct: 0.02
cem_leg_gate_hard_floor_m: -0.005
cem_leg_gate_min_valid_frac: 0.02
cem_leg_gate_fallback: least_violation
```

候选有效性仍为：

```text
min_sdf >= -0.005m
AND fraction[sdf < +0.005m] <= 0.02
```

### 4.4 其他冻结项

```text
spider_method_id = E167A_zOnlyBody
target_variant_id = ref_fk
hand_collision_variant_id = rubber_hull
seed = 0
num_samples = 1024
opt_steps = 32
foot_slip = off
foot_ground = off
cem_smooth = off
reference retiming = off
```

retarget variant 按每条 E168 source row 冻结，不能为了让 E170 通过而切换 v1/v2。CEM horizon、action、body-z、hand/object reward、现有 body/hand gate 与 E168/E169 保持一致，以 effective config audit 为最终 authority。

---

## 5. 实现范围

计划新增以下 E170-scoped 文件；不修改 simulator/sampling 数学逻辑：

| 文件 | 作用 |
|---|---|
| `scripts/experiments/E170/build_box021_prg_manifest.py` | 从 E168 28-row snapshot 生成4 reuse + 24 new manifest、override、scene audit和SHA |
| `scripts/experiments/E170/variants.tsv` | 28-row 分析 authority，显式记录 execution source/reuse |
| `scripts/experiments/E170/render_box021_prg_results.py` | 24条新结果离线渲染，并生成28条 B0-vs-PRG paired montage |
| `scripts/launch/active/run_E170_remote_a100.sh` | A100 0-3 四 worker full launcher |
| `scripts/launch/active/pull_E170_remote_a100_results.sh` | 只回收24条 manifest new rows |
| `scripts/launch/active/watch_E170_remote_a100.sh` | 持续监控、增量回收、严格完成判定 |
| `scripts/launch/active/watch_E170_postprocess_after_remote.sh` | 等待主/recovery session均自然结束，强制最终24-row pull后执行strict postprocess |
| `scripts/launch/active/postprocess_E170_after_full.sh` | 24/24 strict 后触发 render/eval/xlsx |
| `scripts/eval/runners/eval_E170_box021_prg.py` | 统一评测28条PRG并与28条E168 baseline做 paired delta |
| `scripts/eval/wrappers/eval_E170_box021_prg.sh` | 固化评测入口 |
| `scripts/eval/reports/gen_E170_box021_prg_xlsx.py` | 生成完整指标、分组、delta、gate-health和人工审查 workbook |

允许复用 E169 的通用 lower-body geom/pair helper，但不得让 E170 builder 改写 E169 manifest、override、scene 或结果。若复用需要把 helper 从 experiment-specific 文件抽出，只做无行为变化的机械移动，并用 E169 4-case snapshot regression test 证明输出 SHA/语义不变。

---

## 6. Preflight 与 Canary 决策

### 6.0 代码冻结与远程同步

实现与本地验证通过后、任何 canary/full 启动前，创建并 push 一个 E169/E170 foundation checkpoint。只提交 E169/E170 相关实现、配置、计划和记录，不夹带无关工作树文件。E170 `s0_environment/` 必须保存：

```text
checkpoint git commit
local git status / diff-stat
remote git commit
实际 rsync 文件清单及逐文件 local/remote SHA256
Python/CUDA/MuJoCo/torch 版本
```

远程是共享 workspace，不为了 E170 强制切换或重置远程分支；launcher 仍按 manifest 精确 rsync，但 canary 前必须证明远程实际消费文件 SHA 与本地 checkpoint 内容一致。只记录本地或远程 git HEAD、却不记录 dirty/sync 内容，不算可复现。

### 6.1 全 28-row 静态 preflight

GPU 启动前必须完成：

1. E168 authority 精确为28条，4/24复用拆分无交叉或遗漏；
2. 每条 retarget trajectory、contact mask、base scene 和 E168 baseline artifact 存在且 SHA 固化；
3. 24条新 physical sidecar 均可被 MuJoCo compile，精确包含16个唯一 pair；
4. sidecar 与 base scene 的 semantic diff 只包含这16个 pair；
5. 实际 runtime 初始化所用的 reference 前5帧没有深于5mm的 lower-body/object 初始重叠；XML `model.qpos0` 同项仅作诊断，若单独失败则该 row 必须先通过 case-specific runtime smoke 才能进入 full recovery；
6. 24个 override 与 E169 PRG 字段级 parity 通过；
7. 4条 E169 reuse 通过 root/outdir、finite、config、scene和diagnostics审计；
8. command plan 覆盖24条唯一 full 输出；实际 dry-run 命令数必须精确等于 `READY_FOR_FULL` new rows，local blocker 只进入 recovery manifest，所有输出路径互不覆盖。

Preflight 采用分级阻断，不把单条数据问题扩大成全批停机：

| 级别 | 典型问题 | 状态与动作 |
|---|---|---|
| 全局 contract blocker | E168 authority 不是28条、case重复或4/24拆分错误；公共 config/schema 与 E169 PRG 不一致；helper regression 改变 E169 语义；远程公共代码 SHA 不一致 | 标记 `global_contract_blocked`，阻断全部 canary/full；修复公共契约并重跑完整28-row preflight |
| 单 case local blocker | 某条 trajectory/contact mask/base scene/baseline artifact 缺失或 SHA 异常；该 case scene compile 失败、reference 前5帧初始深重叠、override parity 失败；某条 E169 reuse artifact 漂移 | 只把该 row 标记为 `preflight_blocked_local`，从当前 canary/full queue 排除；其余通过行标记 `READY_FOR_FULL` 并可继续执行 |
| qpos0-only 诊断告警 | XML `model.qpos0` 深重叠，但 runtime 实际消费的 reference 前5帧通过硬门 | 标记 `READY_FOR_RECOVERY_SMOKE`，不直接进入 full；使用原 scene/trajectory/config 做 case-specific smoke，产物契约全部通过后才升级为 `READY_FOR_FULL` 并进入 recovery full |

单 case 被隔离时必须保存 `blocker_type/blocker_detail/evidence_path/first_seen_at/recovery_status`，立即列入 recovery manifest；不得把它改成无 P scene、关闭 pair、切换 retarget variant 或修改初始姿态后冒充同一配置。修复后只对 blocked row 重跑 preflight，并在原实验 ID 下定向补跑，不重复计算已经完成的 rows。

`qpos0-only` 告警的降级依据必须可审计：`run_mjwp.py`/`setup_env` 明确从 `qpos_ref[0]` seed simulator；scene audit 同时保存 qpos0 与 reference-first5 clearance；专用 smoke 仍须通过 root/outdir qpos exact match、finite、effective config/scene SHA、PRG reward/gate diagnostics 和无 NaN/OOM/Traceback。它不是对真实 runtime 初始重叠硬门的豁免。

分级阻断只改变执行顺序，不降低 E170 完成标准：允许先取得例如 `23/24 new full` 的阶段性结果并开展 allow-missing 核验，但在全部 local blocker 修复、24/24 new + 4/4 reuse 完整前，不得生成最终 strict conclusion、用户终审包或把 E170 标为完成。

### 6.2 强制双 variant Canary

E169 的 v1 runtime 证据保留为历史参考，但 E170 含 3 条未被 E169 PRG runtime 覆盖的 `omnirt_v2` case，因此不再条件式跳过 canary。全局 contract preflight 通过、且 v1/v2 各至少有一条 `READY_FOR_FULL` row 后，必须在 A100 上执行：

```text
1 omnirt_v1 smoke
1 omnirt_v2 smoke
```

两条 smoke 都必须验证启动、MuJoCo contact、16 pairs、reward/gate diagnostics、root/outdir qpos 一致、NaN/OOM和产物写盘。Canary 不计算质量门槛，也不复用为 full。任意一条失败都阻断 24 条 full；诊断修复后必须重新跑完整双 variant canary，不能只补失败的一条。

若 simulator/sampling/config schema 有行为变化、helper regression 不一致或 A100 runtime 漂移，则在上述双 variant canary 之外先修复相应 preflight，不通过时不得用 smoke 结果绕过静态 gate。

---

## 7. A100 Full 执行计划

### 7.1 资源契约

```text
host = tianyiyun-A100
remote_root = /home/dataset-assist-0/xiayb/workspace/spider
allowed_gpus = 0,1,2,3
parallel_workers = 4
per_gpu_concurrency = 1
remote_render = false
```

启动前保存 `nvidia-smi`、compute process、git SHA、Python/CUDA/MuJoCo版本和用户允许集合。沿用用户已给出的0-3叠加许可，但不 kill、暂停或修改其他任务。每张卡内严格串行，四张卡之间并行。

### 7.2 固定平衡分片

按 E168 `qpos_frames` 做 greedy balance，每卡6条，总帧数约 `627-638`：

| GPU | New full queue | Frames |
|---:|---|---:|
| 0 | `box021_20231011_034_p2`, `box021_20231011_037_p1`, `box021_20231018_031_p2`, `box021_20231020_019_p2`, `box021_20231018_030_p1`, `box021_20231020_020_p1` | 638 |
| 1 | `box021_20231011_036_p2`, `box021_20231011_038_p1`, `box021_20231020_023_p1`, `box021_20231018_033_p2`, `box021_20231018_035_p2`, `box021_20231018_028_p1` | 630 |
| 2 | `box021_20231011_034_p1`, `box021_20231011_037_p2`, `box021_20231020_022_p1`, `box021_20231020_019_p1`, `box021_20231018_030_p2`, `box021_20231018_029_p1` | 627 |
| 3 | `box021_20231011_036_p1`, `box021_20231011_038_p2`, `box021_20231020_022_p2`, `box021_20231018_034_p2`, `box021_20231018_028_p2`, `box021_20231018_032_p2` | 631 |

GPU0-3各自 queue manifest 必须保存完整 case_id、variant、source SHA、override SHA和预期输出路径。正常情况下仍按上表每卡6条；若存在 `preflight_blocked_local`，launcher 只装载 `READY_FOR_FULL` rows，保留其他 case 的原 assigned GPU 和相对顺序，不因缺一条而拒绝整个 shard。recovery manifest 仅包含修复后的 blocked rows，可在 A100 0-3 任一允许 GPU 上定向补跑，并继续使用唯一原输出路径。

### 7.3 计划命令

```bash
# build + static preflight
python workspace/core4d/scripts/experiments/E170/build_box021_prg_manifest.py --preflight

# mandatory 1 v1 + 1 v2 runtime canary
bash workspace/core4d/scripts/launch/active/run_E170_remote_a100.sh canary

# remote A100 0-3: 24 new full
bash workspace/core4d/scripts/launch/active/run_E170_remote_a100.sh full

# qpos0-only warning rows: isolated smoke, then targeted canonical full recovery
bash workspace/core4d/scripts/launch/active/run_E170_remote_a100.sh recovery_smoke
bash workspace/core4d/scripts/launch/active/pull_E170_remote_a100_results.sh recovery_smoke
python workspace/core4d/scripts/experiments/E170/build_box021_prg_manifest.py --preflight
bash workspace/core4d/scripts/launch/active/run_E170_remote_a100.sh recovery_full

# local persistent monitor + strict pull/postprocess
bash workspace/core4d/scripts/launch/active/watch_E170_remote_a100.sh

# manual strict pull/recovery entry
bash workspace/core4d/scripts/launch/active/pull_E170_remote_a100_results.sh full

# unified 28-row evaluation
bash workspace/core4d/scripts/eval/wrappers/eval_E170_box021_prg.sh full

# after Codex fills codex_verification.tsv: rebuild the user review package
bash workspace/core4d/scripts/launch/active/refresh_E170_review_package.sh pre_user

# after the user fills fresh 28/28 labels: validate labels and rebuild metrics/xlsx
bash workspace/core4d/scripts/launch/active/refresh_E170_review_package.sh final
```

launcher、pull、watcher、postprocess和review refresh在实现后先做 `bash -n` 与 dry-run/negative gate test，不在相应前置证据未满足时绕过 gate 直接执行。三阶段 review-package audit 持久保存到 `s6_downstream/evidence/completion/review_package_audit_{postprocess,pre_user,final}.json`；`postprocess` 通过只表示自动包齐全，`pre_user` 通过表示 Codex 核验完成，`final` 通过表示用户 28/28 标签已验证，三者都不单独等同于 E170 最终推广裁决。

---

## 8. Artifact 回收与统一评测

### 8.1 严格完整性

24条 new row 每条至少回收：

```text
root result NPZ
outdir trajectory_mjwp_act.npz
outdir config_act.yaml
run log
```

远程 A100 不渲染。严格 postprocess 仅在以下条件同时成立后启动：

```text
new_complete_rows = 24/24
reused_rows_audited = 4/4
incomplete = []
qpos_finite = 28/28
root_outdir_qpos_match = 28/28
config_scene_sha_pass = 28/28
```

### 8.2 E168 对齐核心指标

每条 PRG 都计算 E168 完整字段，并显式展示以下主指标：

| 类别 | 指标 | 口径 |
|---|---|---|
| Body-z | `body_z_err_p95_m` | `<=0.20m`；peak只诊断 |
| Terminal z | `track_pelvis_z_err_terminal_m` | 保留诊断，不替代 body-z p95 主 gate |
| Fall | `fall_flag` | `false` |
| Raw contact | `hand_object_physics_contact_in_mask_frac` | `>=0.50` |
| Clean3 contact | `hand_object_physics_contact_3mm_in_mask_frac` | 越高越好；报告 raw-clean3 gap |
| Release 3mm | `hand_object_release_false_contact_3mm_frac` | 有 release window 时 `<=0.30`；无窗口为 N/A，不伪填0 |
| Hand penetration | `hand_object_physics_penetration_3mm_frame_frac` | `<=0.30` |
| Lower body | `leg_penetration_frac` | `<=0.10`，主下肢指标 |
| Near object | `leg_near_2cm_frac` | 诊断持续贴箱/踩箱 |
| Leg physics | `leg_object_physics_contact_frac` | P-on内部越低越好；不得与P-off的结构性0直接比较 |
| Tracking | root/EEF/object pos+ori error | 报绝对值和相对E168 B0 delta |
| Motion | accel/jerk、object speed、foot slip | 诊断碰撞冲量和激烈纠错，不临时拟合新阈值 |

### 8.3 Gate-health 单列

沿用 E169 口径：

```text
cem_leg_gate_fallback_used_mean <= 0.10
cem_leg_gate_valid_frac_last_iter_mean >= 0.05
selected leg-valid invariant = true
```

同时报告 combined body/hand/leg fallback，避免把 leg-valid pool 足够误写成 combined pool 足够。gate-health fail 不自动覆盖人工 `USE`，但这种轨迹不能用于证明 G 是可靠 hard constraint。

### 8.4 Paired baseline 分析

每条 E170 PRG 与同 case 的 E168 B0 做一对一 delta。重点报告：

```text
delta leg_penetration
delta leg_near_2cm
delta raw/clean3 hand contact
delta release-3mm false contact
delta >3mm hand penetration
delta body-z p95
delta root/EEF/object tracking
delta motion-health
E168 manual -> E170 fresh manual transition
```

不做跨 case 伪显著性检验；输出 overall、E168 USE/DNU、p1/p2、v1/v2和sequence分层的 count、mean、median、p95及worst cases。

### 8.5 双轨状态输出

每条 row 必须同时输出且不得互相覆盖：

```text
numeric_release_pass
codex_metric_verification_status
codex_visual_spotcheck_status
codex_visual_findings
user_manual_review_status
manual_use_decision
manual_quality_label
manual_failure_taxonomy
manual_reviewer
manual_reviewed_at
manual_operational_use
strict_release_usable
```

Codex 在用户标签回填前只能生成 `PENDING_USER_REVIEW` 的机器建议，不得把自己的抽查结论写成 `manual_use_decision`。用户标签回填后 evaluator 必须重跑，并保存回填前后 manifest SHA。

---

## 9. 视频与人工审查

本地离线生成：

- 24条 new PRG ref/sim side-by-side 视频；
- 4条 reuse 直接引用 E169 PRG full 视频，不重复编码；
- 28条 E168 B0 vs E170 PRG paired montage；
- 每条接触前、最大 lower-body penetration、最大 hand penetration、最大 object speed和末帧关键帧表。

审查分两层执行。

Codex 负责 28/28 指标核验，并按以下 fail-open-to-expand 规则做视觉抽查：

1. 必查所有 numeric fail、阈值上下 10% 的边界 case、fall/object-kick/systemic alarm 和各指标 worst case 的并集；
2. 从剩余 numeric pass 中分层抽取至少 8 条，覆盖 E168 USE/DNU、p1/p2、omnirt_v1/v2、主要 sequence/date；
3. 若分层样本出现未被指标捕获的 `MAJOR_ISSUE`，扩查同 strata 全部 case；同类异常累计达到 2 条时，Codex 视觉检查扩展为 28/28；
4. 所有抽查使用 video-frames 提取关键帧并写具体观察，不把空白或“待补充”当作完成。

用户负责最终人工审查 authority。系统向用户交付 28 条 paired montage、关键帧表、完整指标、Codex 抽查发现和待填 label table；用户对 28/28 新建 E170 标签，不直接复制 E168 或 E169 标签：

```text
manual_use_decision = USE / DO_NOT_USE
manual_quality_label = NO_ISSUE / MINOR_ACCEPTABLE / MAJOR_ISSUE
failure_taxonomy = lower_body / illegal_support / hand_penetration /
                   contact_loss / release_contact / body_z / tracking /
                   object_kick / fall / jitter / other
manual_reviewer = user
```

主观察项：

1. 是否仍有下肢穿箱、踩箱或借箱支撑；
2. P 是否把穿箱变成真实踩箱或把箱体踢飞；
3. R/G 是否通过丢手、身体远离reference或异常姿态换取clearance；
4. 手部是否深穿、接触丢失或release后仍粘箱；
5. p1/p2配对中是否出现一侧明显退化。

Codex 在用户标签前可报告“数值核验完成”和“视觉抽查发现”，但不能宣布 PRG strong/partial/fail；最终实验裁决必须等用户标签回填后产生。

---

## 10. 成功标准与决策规则

### 10.1 单条轨迹双轨状态

`manual_operational_use` 只由用户最终标签决定：

```text
manual_operational_use = (manual_use_decision == USE)
```

单条 case 只有同时满足以下条件才计为 `strict_release_usable`：

```text
fall_flag = false
body_z_err_p95_m <= 0.20
raw hand contact in mask >= 0.50
release false contact 3mm <= 0.30, when applicable
hand physics penetration >3mm frame frac <= 0.30
leg_penetration_frac <= 0.10
manual_use_decision = USE
无明显下肢穿箱/踩箱/借支撑/object kick/爆姿
```

Clean3 contact、root/EEF/object tracking delta和motion-health用于解释边界 case；不得因人工可接受就从表中删除数值失败或把 `strict_release_usable` 改为 true，也不得因 gate-health fail 自动把用户认为 operationally usable 的轨迹改为 DO_NOT_USE。

### 10.2 全量 PRG 裁决

下表产生机器建议，最终推广决定由用户拍板：

| 结果 | 完备条件 | 机器建议 |
|---|---|---|
| 强泛化通过 | overall `>=22/28 strict_release_usable` AND E168失败恢复 `>=9/15` AND E168可用保留 `>=12/13` AND 无 catastrophic/systemic regression | PRG 可作为 Box021 默认候选，下一步再做跨物体验证/人工冻结 |
| 部分通过 | 未达强通过，但 overall `>=18/28` AND 恢复 `>=6/15` AND 保留 `>=10/13` AND 无 catastrophic/systemic regression | PRG 保留为 per-case rescue，不直接成为统一默认 |
| 失败 | 不满足以上两档任一完备条件，或出现 catastrophic/systemic regression | 不推广；按失败分层回到 P/R/G 或 gate calibration |

定义：

- `systemic regression`：相同的新增严重 failure taxonomy 在至少 2 条 case 上出现；
- `catastrophic regression`：任意 1 条新增 fall、明显箱体被踢飞、真实踩箱/借箱支撑导致失稳或爆姿；
- 上述新增均相对同 case E168 B0 判断，历史已有问题与 PRG 新增问题分列。

若 strict 质量强泛化通过但 gate-health 仍普遍失败，机器结论必须写成“PRG strict quality generalization supported，G hard-constraint claim unsupported”。若仅人工 operational use 较高，则只能写“用户认为 operationally usable”，不能提升为 strict strong pass。只有 gate-health 和 strict quality 同时通过，才能进一步声称 frozen PRG 的 gate 机制也具备全量稳定性。

### 10.3 RL 边界

E170 本身不自动修改 E168 已冻结的13条 RL export，也不生成新的 RL export。只有 E170 全量人工审查完成后，才另行冻结可用 source rows，并按 sequence 补齐 partner OmniRetarget 信息。p1/p2仍作为独立轨迹处理，任何新 RL export 都必须满足配对完整性。

---

## 11. 风险与缓解

| 风险 | 影响 | 控制 |
|---|---|---|
| E169 gate-health 已失败 | PRG好看但G并非可靠hard gate | 质量与机制双轨裁决，保留全部valid/fallback diagnostics |
| P-on physics contact天然上升 | 把启用碰撞误判为穿透回归 | 主比较用SDF `leg_penetration_frac`；leg physics contact只在P-on内部解释 |
| runtime 初始深重叠 | 首步爆炸冲量或case无法运行 | 全24 scene reference前5帧硬门；隔离该 row、其余继续，修复后定向补跑；未被 runtime 消费的 XML qpos0 单独作为诊断并强制专用 smoke |
| 公共契约错误 | 全批结果不可比或远程运行错误代码 | `global_contract_blocked` 阻断全部 canary/full，修复后重跑完整 preflight |
| 单 case 输入/scene 错误 | 一条数据拖延整批或被静默跳过 | `preflight_blocked_local` 隔离并进入 recovery manifest，其余 READY rows 继续 |
| v2三条未在E169覆盖 | scene/override路径差异导致运行失败 | v2单独静态审计；强制执行1条v1+1条v2 canary，不切回v1 |
| 复用结果与新结果口径漂移 | 28-row统计不可比 | E170统一重评，记录source experiment和SHA，不复用旧汇总值 |
| E168 USE回归 | 提高失败集同时破坏好case | 13条control retention单列，paired video逐条复核 |
| 手部质量被lower-body改善掩盖 | 视觉站姿改善但接触/穿透变差 | raw/clean3/release3mm/penetration四项同时展示 |
| A100远程不可渲染 | 无法即时视觉判断 | compute-only，strict pull后本地统一渲染 |
| 24条队列中途失败 | 重复运行或覆盖产物 | manifest-driven唯一输出、逐row原子完成、恢复时只补missing rows |
| 人工结论受E168标签先验影响 | confirmation bias | E170 fresh review列与E168 baseline label分列，先看paired video再填标签 |

---

## 12. 产物与完成定义

```text
workspace/core4d/results/E170/
  s0_environment/
  s1_raw_contact/imported_e168_snapshot/
  s2_templates/imported_e168_snapshot/
  s3_retarget/imported_e168_snapshot/
  s4_gate_visual_qc/
  s5_handoff/
  scene_snapshot/
  registries/
  s6_downstream/
    manifests/
    cem/canary/
    cem/full/
    artifacts/canary/
    artifacts/full/
    eval/full/
    render/full/
    evidence/
  execution_manifest.json
  artifact_manifest.json
```

E170 不重跑 S1-S5，但必须用小型 imported snapshot 明确引用 E168 authority、retarget route、raw contact、template、rubber-hull handoff 和相应 SHA；不能通过目录扫描隐式继承。E170 新 CEM/eval/render 都属于 S6，不再新建 E169 风格的根级 `cem/`、`eval/`、`render/` 正式目录。

E170 完成必须同时满足：

1. 28-row manifest 唯一且拆分为4 reuse + 24 new；
2. 24/24 new full 严格回收，4/4 E169 reuse SHA审计通过；
3. 28/28 unified evaluation 无 error/not-ready；
4. E168完整指标、paired deltas、group summary、worst cases和gate-health齐全；
5. 24条新视频、4条reuse引用、28条paired montage、Codex 指标核验/视觉抽查证据和用户 28/28 fresh人工标签完成；
6. xlsx 经 LibreOffice 重算，formula error为0，`xlsx_recalc_validation.json` 与相应阶段 review-package audit 均持久化且通过；
7. 结果 log 明确裁决 C0-C8、manual operational/strict release 双轨、机器 strong/partial/fail 建议、用户最终裁决和 gate-health独立结论；
8. 更新 `EXPERIMENT_TRACKER.md` 与 `progress.md`。

当前已完成实现、静态 preflight、双 variant canary、qpos0-only case-specific smoke 与首批 full 严格回收；A100 主 full 和两条 `036_p1/p2` 定向 recovery full 正在运行。最终完成仍须满足上述 1-8 项，不因阶段性结果提前生成结论。
