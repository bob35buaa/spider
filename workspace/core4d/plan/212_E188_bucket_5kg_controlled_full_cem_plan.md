# E188 实验计划：Bucket 2→5 kg 的 E187 15-case 受控 Full CEM 复跑

_CORE4D Phase 51 · 2026-08-04 · 修订：跳过7条5→5kg · 状态：PLAN_COMPLETE / NOT_STARTED_

## 1. Context

E187 在三类 bucket 的 22 条冻结 case 上完成了 Full CEM，并使用统一的
`1024 samples × 32 iterations × seed 0`、canonical distance continuation reward、
object-specific CoACD compound collision 与 grid-SDF。E187 22 条结果均已完成并通过
artifact 完整性检查，但冻结 12 门的 numeric pass 为 `6/22`，主要失败项是 lower-body
penetration。

本次质量审计发现，E187 的 bucket 物体质量并不统一：

| Object | E187 case 数 | E187 质量分布 | E188运行 | E188跳过 |
|---|---:|---|---:|---:|
| bucket003 | 5 | 2条 `2.0 kg`；3条 `5.0 kg` | 2条 `2→5` | 3条 `5→5` |
| bucket004 | 4 | 4条 `5.0 kg` | 0 | 4条 `5→5` |
| bucket007 | 13 | 13条 `2.0 kg` | 13条 `2→5` | 0 |
| **总计** | **22** | **15条2kg；7条5kg** | **15条** | **7条** |

E188 只运行真正发生质量变化的 15 条：bucket003 的2条2kg case和bucket007的13条2kg
case。已经是5kg的 bucket003 3条与 bucket004 4条不生成E188 scene、不生成override、
不进入canary/Full/eval/video。E188 的唯一科学变量是 object inertial：15条均从2kg改成
5kg，并同步按质量比缩放 `diaginertia`。E187 的scene、结果和历史记录不得覆盖。

2026-08-04执行修订（用户明确确认）：A100主机缺NVIDIA EGL且历史已验证osmesa/glfw
同样不可用。远程11条改为compute-only `save_video=false`，回收NPZ/config/log后在本机
离线渲染；本机4条仍`save_video=true`。这是执行/渲染路径差异，不改变CEM科学输入、预算
或seed；最终15条E188视频和15条paired视频仍是C9硬门。

### 1.1 冻结上游 authority

| Authority | 路径 | SHA256 |
|---|---|---|
| E187 keep22 | `workspace/core4d/results/E187/s0_environment/keep22_protocol_manifest.tsv` | `b1122a3a99d9760ff2bcd22b90783ca93999b17f620923e8ecef44801d6a4924` |
| E187 Full queue | `workspace/core4d/results/E187/s5_full/queue/queue_manifest.json` | `836b5388420f968e9c88968349a87799ae1ea3bd43d6276e631847c7956dcd92` |
| E187 evaluation manifest | `workspace/core4d/results/E187/s6_downstream/manifests/e187_full_evaluation_manifest.tsv` | `507c3a79766e61b49d5e298432bd0a766d9566f931ec9a2382f1a5b6e82c6fc7` |
| E187 reward/grid lock | `workspace/core4d/results/E187/s2_canonical_grid_sdf/reward_grid_lock.json` | `2cc949e19cb313e58883fd5d0446872220188ea9ecce06cb23d39e2a6b42194f` |
| E187 production integration lock | `workspace/core4d/results/E187/s3_prg_audit/production_integration_lock.json` | `400c98b422ac4458eccffb589eecc1d776fe5cd3968eb855cd89a485bb7d5441` |
| E187 metrics baseline | `workspace/core4d/results/E187/s6_downstream/eval/full/e187_case_metrics.tsv` | `77405ebd72a02c7135d3b28945ffcf65e1fa2c0e2394bbba687e3c699000cb06` |

E188 复用上述 scientific payload 与 case authority，但不继承 E187 lock 中的旧设备
治理字段 `use_a100=false`，也不改写 E187 的 C9 `FAIL / USER_WAIVED` 历史事实。E188 的
设备映射由本计划和新的 execution manifest 单独授权。

### 1.2 E187 主要指标基线

下表是 E188 选中15条在E187中的配对基线均值。接触越高越好，其余八项越低越好。

| 指标 | E187 mean | 冻结 gate（如适用） |
|---|---:|---:|
| 手接触：`hand_object_physics_contact_in_mask_frac` | 0.73976 | `≥0.50` |
| 手穿透：`hand_object_physics_penetration_3mm_frame_frac` | 0.22674 | `≤0.30` |
| 下肢穿透：`leg_penetration_frac` | 0.15848 | `≤0.10` |
| Body/root position error | 17.1703 cm | `≤20 cm` |
| Body/root orientation error | 9.6985° | `≤20°` |
| Body/hand position error | 16.1721 cm | `≤20 cm` |
| Body/hand orientation error | 22.6287° | `≤20°` |
| Object position error | 9.8231 cm | `≤20 cm` |
| Object orientation error | 5.8030° | `≤10°` |

这15条在E187中为 numeric pass `3/15`、lower-body pass `4/15`。本次不运行7条5kg
控制，因此不再使用A/A漂移门或difference-in-differences分析。

## 2. 实验问题与关键假设

核心问题不是“5 kg 一定更好”，而是：在 E187 其余条件冻结时，将 2 kg bucket 提高到
5 kg 是否会系统性改变 CEM 解，并能否缓解 E187 的 lower-body penetration，而不牺牲
接触与 tracking。

物理上只修改 `mass` 而不修改 inertia 会产生不一致的刚体动力学，因此 E188 把
`mass + diaginertia` 视为一个不可拆分的 object inertial intervention：

```text
scale = 5.0 / old_mass
new_mass = 5.0
new_diaginertia = old_diaginertia * scale
```

几何、碰撞、摩擦、关节、初始位姿、轨迹、contact mask、reward、grid、CEM 预算与 seed
均不得改变。

## 3. Claims

| Claim | 最低证据与判定 |
|---|---|
| C0 Authority 完整 | 15个case精确等于E187 evaluation manifest中`old_mass=2kg`子集；bucket003=2、bucket007=13、bucket004=0；duplicate/missing/unexpected均为0；上游authority SHA全部匹配 |
| C1 质量变体正确 | 15/15 object mass=`5.0 kg`且 inertia ratio=`2.5`；robot inertial零改动；7条原5kg case零进入E188执行manifest |
| C2 单变量合同成立 | XML canonical diff 仅允许 object `<inertial mass,diaginertia>`；E188 override 相对 E187 只允许 `scene_name` 改变；trajectory/contact/collision/grid/reward/budget/seed SHA 或值全部冻结 |
| C3 执行闭合 | canary `3/3`、Full `15/15`；finite、config、scene、result、log与row manifest均完整；本机inline video可读，远程row显式`DEFERRED_LOCAL_RENDER`；missing/error=0 |
| C4 设备边界如实 | 保留4条RTX5090同设备配对；速度重排后执行分片为本机7条/A100 8条，但其余11条仍按E187→E188实际设备变化单列；不以跨设备裸delta单独证明质量因果 |
| C5 下肢改善 | 15条 lower-body pass 从 `4/15` 提升到至少 `7/15`，paired mean `leg_penetration` improvement ≥0.05；至少10/15 case非退化；同设备local-4方向单列 |
| C6 接触/手穿透非劣 | 15条 mean contact 不低于E187超过0.03；mean hand penetration不高于E187超过0.03 |
| C7 Tracking 非劣 | 15条四个Body Tracking paired mean regression：位置各≤2cm、角度各≤2°；两个Object Tracking regression：位置≤2cm、角度≤2° |
| C8 总门改善 | E188 numeric pass至少`5/15`，且numeric PASS→FAIL≤1 |
| C9 可视化与报告闭合 | 15条E187-left/E188-right视频可读；xlsx 15-row paired、质量/设备provenance与gate transition完整；人工观察不留空 |

C0–C4/C9是技术与解释边界；C5–C8是效果Claims。即使C5–C8失败，只要C0–C4/C9
通过，实验仍可作为“5kg无效或有害”的完整负结果收口。由于删掉A/A控制，远程11条
不能单独支持纯质量因果；最强证据来自保持RTX5090的local-4同设备配对。

## 4. 单变量冻结矩阵

| 变量 | E187 | E188 | 规则 |
|---|---|---|---|
| case set/order | keep22 | 其中old mass=2kg的15条，保持相对ordinal | 精确过滤；7条5kg零进入 |
| retarget variant | 每case冻结 | 同 E187 | SHA不变 |
| target task/trajectory | 每case冻结 | 同 E187 | trajectory SHA不变 |
| contact mask | E174 3cm raw mask | 同 E187 | SHA不变 |
| collision geometry | object-specific CoACD compound | 同 E187 | asset/manifest SHA不变 |
| grid-SDF | bucket003 5mm；004/007 2.5mm | 同 E187 | manifest、epsilon、asset SHA不变 |
| reward | distance continuation `0.25/0.75` far/near | 同 E187 | A1 scientific payload不变 |
| PRG / lower-body gate | E187 production integration | 同 E187 | 不回调 P/R/G |
| object mass | 2kg或5kg | **全部5kg** | 唯一主动变量 |
| object `diaginertia` | 与旧mass配套 | 按质量比同步缩放 | 唯一必要伴随变化 |
| object geometry/friction/pose | E187 | 同 E187 | XML diff禁止变化 |
| CEM samples/iterations/seed | `1024/32/0` | `1024/32/0` | 不降预算、不换seed |
| query tape | off | off | `query_tape_enabled=false` |
| video | on | 本机on；A100 compute-only后本机离线补齐 | 用户确认的EGL环境修订；每case最终唯一路径 |
| evaluator | `core4d-e154-physics-contact-v1` | 同标准 | 同12门阈值 |
| 本地设备 | RTX5090 GPU0 | RTX5090 GPU0 | 原 local shard保持 |
| 远程设备 | RTX6000 Ada GPU0/1 | A100 GPU4/5 | 显式混杂；分设备报告，不做A/A校准 |

## 5. Scene 质量派生设计

### 5.1 新 scene 文件

15个选中E187 case的 `scene_act_E186_coacd_compound.xml` 派生为同task目录中的：

```text
scene_act_E188_mass5kg.xml
```

不得原地修改或覆盖任何E186/E187 scene。15个新XML在launch前必须：

1. 用 builder 自动生成，禁止手工逐文件编辑；
2. 经XML parser和MuJoCo load `15/15`；
3. 用 `git add -f` 纳入主 git 的活跃 scene 追踪；
4. 复制到 `workspace/core4d/results/E188/scene_snapshot/`；
5. 在 manifest 中记录 source/target path、old/new inertial、source/target SHA256 与 canonical diff。

### 5.2 预期 inertial

| Object / group | old mass | old `diaginertia` | scale | expected new `diaginertia` |
|---|---:|---|---:|---|
| bucket003 p1（2条） | 2.0 | `0.13313547 0.08489633 0.14572285` | 2.5 | `0.332838675 0.212240825 0.364307125` |
| bucket007（13条） | 2.0 | `0.10901105 0.10390275 0.10468122` | 2.5 | `0.272527625 0.259756875 0.261703050` |

浮点写出保留足够有效位；审计使用高精度数值比较，不依赖字符串完全相等。

### 5.3 XML diff 白名单

对 canonicalized XML tree，只允许以下 XPath 属性变化：

```text
/mujoco/worldbody/body[@name='object']/inertial/@mass
/mujoco/worldbody/body[@name='object']/inertial/@diaginertia
```

每个scene必须恰有一个`body[name=object]`和一个目标inertial；禁止误改robot/link
inertial。builder必须拒绝old mass不等于2kg的输入，确保7条原5kg case不会被静默纳入。

## 6. Override 与运行合同

自动生成15个E188 override：

```yaml
# @package _global_
defaults:
  - core4d_E187_<case>_distanceContinuation
  - _self_
scene_name: scene_act_E188_mass5kg
```

所有 effective config 做结构化 diff。相对 E187 production override，plan-time 只允许
`scene_name` 改变；runtime `config_act.yaml` 另允许 output/video/device 等执行路径字段变化。

Full 命令保持：

```text
num_samples=1024
max_num_iterations=32
seed=0
local-0: save_video=true
a100-4/a100-5: save_video=false  # 用户确认；本机离线渲染为硬性后处理
save_info=true
query_tape_enabled=false
query_tape_record_geometry_state=false
```

## 7. 三卡并行分片

E188从E187三个worker中删除7条5kg case，保留剩余15条的相对顺序；`local-0`继续本机
GPU0，`remote-0/1`分别映射到A100 GPU4/5。保留local原分片可使其中4条维持与E187
相同RTX5090设备。

| Worker | 物理设备 | case 数 | `2→5` | `5→5` | 可晋级 canary |
|---|---|---:|---:|---:|---|
| `local-0` | 本机 RTX5090 GPU0 | 4 | 4 | 0 | `bucket003_20231018_003_p1` |
| `a100-4` | 远程 A100 GPU4 | 6 | 6 | 0 | `bucket007_20231003_2_021_p1` |
| `a100-5` | 远程 A100 GPU5 | 5 | 5 | 0 | `bucket007_20231020_055_p1` |
| **总计** | 3卡 | **15** | **15** | **0** | **3** |

### 7.1 初始 `local-0` 队列（remaining已由§7.4取代）

| Pos | Case | old→new mass | 角色 |
|---:|---|---:|---|
| 1 | `bucket003_20231018_003_p1` | 2→5 | canary / treatment |
| 2 | `bucket007_20231018_021_p2` | 2→5 | treatment |
| 3 | `bucket007_20231018_021_p1` | 2→5 | treatment |
| 4 | `bucket007_20231018_019_p2` | 2→5 | treatment |

### 7.2 初始 `a100-4` 队列（remaining已由§7.4取代）

| Pos | Case | old→new mass | 角色 |
|---:|---|---:|---|
| 1 | `bucket007_20231003_2_021_p1` | 2→5 | canary / treatment |
| 2 | `bucket007_20231023_075_p2` | 2→5 | treatment |
| 3 | `bucket007_20231003_1_021_p2` | 2→5 | treatment |
| 4 | `bucket003_20231020_064_p1` | 2→5 | treatment |
| 5 | `bucket007_20231003_1_021_p1` | 2→5 | treatment |
| 6 | `bucket007_20231003_2_023_p1` | 2→5 | treatment |

### 7.3 初始 `a100-5` 队列（remaining已由§7.4取代）

| Pos | Case | old→new mass | 角色 |
|---:|---|---:|---|
| 1 | `bucket007_20231020_055_p1` | 2→5 | canary / treatment |
| 2 | `bucket007_20231023_073_p1` | 2→5 | treatment |
| 3 | `bucket007_20231023_075_p1` | 2→5 | treatment |
| 4 | `bucket007_20231020_059_p1` | 2→5 | treatment |
| 5 | `bucket007_20231018_019_p1` | 2→5 | treatment |

按E187对应15条的wall time求和，三队列约为`2.94h / 5.47h / 4.16h`；A100实际吞吐
以E188 canary为准。约25秒plan time仅做监控，不作为停止门，也不得为提速降低CEM预算。

### 7.4 2026-08-05 用户批准的速度重排（最终执行authority）

Canary实测每个plan step中位数为local=`27.9845s`、A100-4=`73.0582s`、
A100-5=`69.1504s`。E188/E187三条canary的plan_time_count完全一致，因此以12条E187
plan_time_count为工作量，最小化三机makespan，并硬约束原local-4全部保留在本机。

| Worker | Remaining case（plan steps） | 数量 | 预测总时长 |
|---|---|---:|---:|
| local-0 | `20231023_073_p1(135)`, `20231018_021_p2(120)`, `bucket003_20231020_064_p1(111)`, `20231023_075_p2(106)`, `20231018_021_p1(94)`, `20231018_019_p2(88)` | 6 | 5.084h |
| a100-4 | `20231023_075_p1(105)`, `20231018_019_p1(88)`, `20231003_2_023_p1(59)` | 3 | 5.114h |
| a100-5 | `20231003_1_021_p2(98)`, `20231020_059_p1(93)`, `20231003_1_021_p1(76)` | 3 | 5.129h |

最终含canary分片为`7/4/4`；v1队列因已被canary manifest引用而保持不可变，正式remaining
authority为additive `queue_speed_rebalanced_v2`。预测未计外部程序叠加造成的吞吐波动。

## 8. A100 GPU4/5 安全与授权门

### 8.0 2026-08-05 用户授权覆盖

用户明确要求GPU0/4/5允许与其他程序叠加运行，并撤销memory/compute-process启动门。
因此下述原始空闲门不再适用于remaining12；启动只固定物理GPU编号、记录telemetry，不查询
空闲交集、不因已有compute process阻断、不抢占/kill其他任务，也不fallback到其他GPU。

用户指定远程 A100 GPU4/5，但 worker 代码不能无条件写死并直接占卡。实现要求：

1. `requested_gpu_set=[4,5]` 写入待冻结 execution manifest；
2. 通过预约工具或显式 `A100_POLICY_GPUS=4,5` 获取 owner/policy 许可；
3. 查询全部 GPU 的 index/UUID/name/memory，并查询 compute apps；
4. GPU4、5 均须 `memory.used < 5000 MiB`，且对应 UUID 无未授权 compute process；
5. policy、显存、compute process 三者交集必须精确得到 `ALLOWED_GPUS=4 5`；
6. queue 与 tmux 启动之间再次检查，任一卡状态变化则不启动并重建 execution manifest；
7. worker 从 manifest 读取 `ALLOWED_GPUS`，物理卡通过 `CUDA_VISIBLE_DEVICES=4/5` 映射，进程内部使用 `device=cuda:0`；
8. 不 fallback 到其他卡、不 kill/暂停/抢占外部任务、不允许 compute overlap；
9. 若 GPU4 或5不满足门，E188 保持 `REMOTE_PREFLIGHT_BLOCKED`，等待用户或外部资源变化。

远程运行使用隔离目录：

```text
/home/dataset-assist-0/xiayb/workspace/e188_spider_runs/<execution_id>/spider
```

通过manifest-driven rsync部署代码、15个E188 scene、override、trajectory、contact mask、
grid 和必要 asset；部署前后逐文件 SHA 校验。不得在远程共享 checkout 原地修改 scene。

## 9. 执行阶段

### Phase 0：Plan review

- 本文件、Tracker 和 progress 落盘；
- 状态保持 `NOT_STARTED`；
- 用户确认后才进入实现与 GPU launch。

### Phase 1：实现与静态测试

1. 实现 mass5 scene builder、authority freezer、queue builder 和 runner；
2. 自动生成15个E188 override；
3. 单元测试覆盖XML唯一目标、inertia ratio、拒绝old mass≠2kg、diff白名单与不可覆盖；
4. runner dry-run必须解析15/15唯一命令与输出路径；
5. evaluator 直接使用 `eval.core.core_metrics`，不 dynamic-import 历史 evaluator；
6. `py_compile`、targeted tests、`git diff --check` 全部通过。

### Phase 2：Authority、scene 和快照冻结

产出：

```text
workspace/core4d/results/E188/s0_environment/authority_manifest.tsv
workspace/core4d/results/E188/s0_environment/mass_audit.tsv
workspace/core4d/results/E188/s0_environment/invariant_diff.tsv
workspace/core4d/results/E188/s0_environment/execution_manifest.json
workspace/core4d/results/E188/scene_snapshot/
```

硬门：15/15 mass=5kg、15/15 scale=2.5、15/15 MuJoCo load、7条原5kg零进入、上游SHA
闭合。任一失败禁止 canary。

### Phase 3：三卡 full-budget canary

三条 canary 同时启动，使用正式 `1024×32 seed0`，输出进入各自独立 row 目录。canary
晋级条件：

- scene/config/grid/reward/trajectory/contact SHA 与 authority 一致；
- 运行成功，无 OOM，所有 numeric array finite；
- `trajectory_mjwp_act.npz`、`config_act.yaml`、log均存在且SHA已登记；本机inline video同时登记；
- A100 row必须登记`DEFERRED_LOCAL_RENDER`，不得伪造inline video；最终离线video须由ffprobe解码且时长>0；
- recorder/query tape 关闭，无意外 raw chunks；
- A100 卡仍是 execution manifest 中的 physical GPU4/5；
- 约25秒 plan time 只记录，不阻断。

`3/3` 通过后，canary 原子登记为 Full 的前三条 completed row，禁止重复计算或覆盖。

### Phase 4：剩余12条Full CEM

- 每卡内部严格串行，三卡之间并行；
- resume 只运行 manifest 中 `NOT_RUN`/明确可恢复的缺失条目；
- complete row 的 artifact SHA 不一致时 fail closed，不原地覆盖；
- 每条结束立即写 immutable row manifest；
- 最终closure必须`completed + terminal_failed = 15`，正常目标`completed=15`。

### Phase 5：远程回收与完整性审计

- pull 只读取本次 execution manifest 登记的远程路径；
- 校验15份NPZ/config/log/row manifest及本机inline video的数量、大小、SHA；远程video在本机离线渲染后另行登记SHA；
- 保留远程 GPU、compute process、tmux、部署 inventory 与环境快照；
- 不因本地已有同名文件而静默覆盖。

### Phase 6：E188 vs E187 配对评测

从E187 frozen 22-row metrics精确过滤相同15条作为唯一主baseline，并保留E178仅作历史背景。输出：

```text
workspace/core4d/results/E188/s6_downstream/manifests/e188_full_evaluation_manifest.tsv
workspace/core4d/results/E188/s6_downstream/eval/full/e188_case_metrics.tsv
workspace/core4d/results/E188/s6_downstream/eval/full/e188_vs_e187_paired_deltas.tsv
workspace/core4d/results/E188/s6_downstream/eval/full/e188_device_stratified_summary.tsv
workspace/core4d/results/E188/s6_downstream/eval/full/e188_gate_transitions.tsv
workspace/core4d/results/E188/s6_downstream/eval/full/summary.json
```

paired improvement 的统一方向：

```text
contact improvement = E188 - E187
penetration/tracking error improvement = E187 - E188
```

对15条paired delta做10,000次case bootstrap，报告point estimate与95% CI；同时按执行设备
严格分层：`local-0 n=4`、`a100-4 n=6`、`a100-5 n=5`。不再进行A/A drift adjustment或
difference-in-differences。阈值在launch前冻结，不得看结果后改门。

解释边界：

- local-0的4条保持E187/E188均为RTX5090，是质量效应的最强配对证据，但仍是单seed；
- 远程11条从E187 RTX6000 Ada切换到E188 A100，mass与device同时变化；
- bucket003仅运行2条历史2kg行，3条历史5kg行不重跑；bucket004完全不进入E188；
- pooled-15与remote-11只能称为“5kg版本相对E187的结果变化”；纯质量因果主要看local-4，且不得超范围推广。

### Phase 7：xlsx、左右视频与人工观察

工作簿至少包含：

1. `Overview`；
2. `Case Mass Audit`；
3. `E188 Metrics`；
4. `E187 Baseline`；
5. `Paired Comparison`；
6. `Device Stratified`；
7. `Gate Transitions`；
8. `Object Summary`；
9. `Best Improvements`；
10. `Worst Regressions`；
11. `Artifact Provenance`。

生成15个左右视频：左侧E187 2kg，右侧E188 5kg；case、old/new mass、worker/device、关键
delta 叠字。视频写入：

```text
workspace/core4d/results/E188/s6_downstream/render/full/paired_e187_vs_e188/
```

并注册到 `workspace/core4d/scripts/eval/wrappers/review_player.sh` 的 E188 review 入口。视频
生成后使用`video-frames`抽取改善、退化、同设备local与跨设备remote四类代表帧，并把具体
观察写入 E188 log，不得写“待补充”。

### Phase 8：Log、Tracker 与收口

- 新建 E188 results log，记录所有 Claims、失败、视频观察、artifact SHA 与路径；
- 更新 `log/INDEX.md`、`EXPERIMENT_TRACKER.md`、`progress.md`；
- 只有 Claims 按实际证据收口后才决定是否 commit/push；
- 不自动进入 RL export/training；需要另行用户授权与计划。

## 10. 需要创建或修改的文件

| # | 文件/目录 | 计划改动 |
|---:|---|---|
| 1 | `workspace/core4d/scripts/experiments/E188/build_mass5_scenes.py` | 15条scene派生、inertia scale、diff与mass audit |
| 2 | `workspace/core4d/scripts/experiments/E188/freeze_authority.py` | 冻结E187 SHA、15-case过滤与单变量合同 |
| 3 | `workspace/core4d/scripts/experiments/E188/build_overrides.py` | 生成仅改`scene_name`的15个override |
| 4 | `workspace/core4d/scripts/experiments/E188/build_full_queues.py` | 过滤7条5kg并保持剩余相对顺序，映射到local/A100 4/5 |
| 5 | `workspace/core4d/scripts/experiments/E188/run_full_queue.py` | dry-run/canary/promotion/full/resume/closure |
| 6 | `workspace/core4d/scripts/experiments/E188/deploy_remote_a100.py` | isolated snapshot、execution manifest、两次GPU门与SHA部署 |
| 7 | `workspace/core4d/scripts/experiments/E188/test_*.py` | mass、authority、queue、runner、deployment与eval合同测试 |
| 8 | `examples/config/override/core4d_E188_*_mass5kg.yaml` | 15个自动生成override |
| 9 | `example_datasets/.../<task>/scene_act_E188_mass5kg.xml` | 15个强制tracked质量scene |
| 10 | `workspace/core4d/scripts/launch/active/run_E188_local.sh` | 本机GPU0 preflight/canary/full入口 |
| 11 | `workspace/core4d/scripts/launch/active/run_E188_remote_a100.sh` | A100 GPU4/5安全部署与并行worker入口 |
| 12 | `workspace/core4d/scripts/launch/active/pull_E188_remote_a100_results.sh` | manifest-driven回收与SHA审计 |
| 13 | `workspace/core4d/scripts/launch/active/watch_and_pull_E188.sh` | 本地+远程状态、自动pull、closure后eval |
| 14 | `workspace/core4d/scripts/eval/runners/eval_E188_vs_E187_full.py` | 公共metrics、paired delta与device-stratified summary |
| 15 | `workspace/core4d/scripts/eval/wrappers/eval_E188_vs_E187_full.sh` | canonical评测入口 |
| 16 | `workspace/core4d/scripts/eval/reports/gen_E188_vs_E187_xlsx.py` | 11-sheet工作簿 |
| 17 | `workspace/core4d/scripts/experiments/E188/render_paired_videos.py` | 15条左右视频与manifest |
| 18 | `workspace/core4d/scripts/eval/wrappers/review_player.sh` | 注册E188 paired review入口 |

## 11. Canonical commands（实现后）

### 11.1 构建与冻结

```bash
.venv/bin/python workspace/core4d/scripts/experiments/E188/build_mass5_scenes.py --preflight
.venv/bin/python workspace/core4d/scripts/experiments/E188/build_mass5_scenes.py --freeze
.venv/bin/python workspace/core4d/scripts/experiments/E188/freeze_authority.py
.venv/bin/python workspace/core4d/scripts/experiments/E188/build_overrides.py --freeze
.venv/bin/python workspace/core4d/scripts/experiments/E188/build_full_queues.py --freeze
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E188 <15 target tasks>
```

### 11.2 本地与远程 preflight

```bash
bash workspace/core4d/scripts/launch/active/run_E188_local.sh preflight
A100_POLICY_GPUS=4,5 \
  bash workspace/core4d/scripts/launch/active/run_E188_remote_a100.sh preflight
```

### 11.3 Canary 与 Full

```bash
bash workspace/core4d/scripts/launch/active/run_E188_local.sh canary
A100_POLICY_GPUS=4,5 \
  bash workspace/core4d/scripts/launch/active/run_E188_remote_a100.sh canary

bash workspace/core4d/scripts/launch/active/run_E188_local.sh full
A100_POLICY_GPUS=4,5 \
  bash workspace/core4d/scripts/launch/active/run_E188_remote_a100.sh full
```

### 11.4 回收、评测与可视化

```bash
bash workspace/core4d/scripts/launch/active/pull_E188_remote_a100_results.sh full
bash workspace/core4d/scripts/eval/wrappers/eval_E188_vs_E187_full.sh preflight
bash workspace/core4d/scripts/eval/wrappers/eval_E188_vs_E187_full.sh run
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E188_vs_E187_xlsx.py
bash workspace/core4d/scripts/eval/wrappers/render_E188_vs_E187_paired.sh
bash workspace/core4d/scripts/eval/wrappers/review_player.sh E188
```

## 12. 成功分层

| 层级 | 条件 | 允许结论 |
|---|---|---|
| T0 技术失败 | C0–C3任一失败 | 不比较效果；修复合同后只补缺失条目 |
| T1 技术完成、设备受限 | C0–C3/C9通过，但C4未被遵守 | 可交付5kg结果；禁止质量因果表述 |
| T2 可解释负/中性结果 | C0–C4/C9通过，C5–C8未达 | 5kg版本在本设置下无明确收益或存在trade-off |
| T3 方向性改善 | C0–C9通过，但local-4与remote-11方向不一致或关键CI含0 | 5kg版本有局部收益，证据强度有限 |
| T4 强支持 | C0–C9通过，local-4同设备方向一致且pooled-15/remote-11不反向 | 5kg对主要失败模式有较强支持，仍需独立seed复验 |

单 seed 的 T4 仍不能推广为一般物理规律；如要升级默认质量，应另开多 seed 或独立 case
确认实验，不在 E188 内追加 post-hoc sweep。

## 13. Stop rules

- E187 authority、trajectory、contact mask、collision/grid/reward SHA 任一漂移：停止；
- XML diff 超出 object inertial 白名单、object mass非5kg、robot inertial被改：停止；
- 15条任一MuJoCo load失败：停止canary；
- GPU0/4/5已有外部compute process或高显存：按用户授权允许叠加，不作为停止条件；
- 仍固定使用物理GPU0/4/5，不fallback、不抢占、不kill其他任务；
- OOM、non-finite、本机inline video不可读、远程deferred标记缺失、最终离线video不可读或artifact路径冲突：停止对应worker，保留失败证据；
- 已完成 row 的 SHA 不一致：fail closed，禁止覆盖；
- 不降低 samples/iterations、不换seed、不改reward/grid/P/G来“救”失败case；
- remote-11与local-4方向冲突：继续完成技术报告，但不得用pooled结果宣称纯质量收益；
- 不 kill、暂停、抢占其他任务，不自动改用A100其他卡；
- E188 未收口前不启动 RL export 或训练。

## 14. 结果路径

| 内容 | 路径 |
|---|---|
| scene/authority/mass audit | `workspace/core4d/results/E188/s0_environment/`、`scene_snapshot/` |
| canary | `workspace/core4d/results/E188/s4_canary/` |
| Full NPZ/config/log/row manifest；本机inline与远程deferred video证据 | `workspace/core4d/results/E188/s5_full/` |
| evaluator/paired/device strata | `workspace/core4d/results/E188/s6_downstream/eval/full/` |
| paired videos | `workspace/core4d/results/E188/s6_downstream/render/full/paired_e187_vs_e188/` |
| launcher/worker/monitor logs | `logs/E188/` |
| 正式结果记录 | `workspace/core4d/log/<next>_E188_bucket_5kg_controlled_full_cem_results.md` |

## 15. 当前边界

本计划仅完成实验设计。当前没有创建 E188 scene/override/script/result，没有启动本地或远程
GPU，也没有改变 E187/E178 artifact。进入实现与 launch 前需先由用户确认本计划。
