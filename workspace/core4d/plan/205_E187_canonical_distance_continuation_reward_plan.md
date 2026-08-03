# E187 实验计划：Canonical Distance Continuation Reward 与 E178-compatible Full CEM

_Core4D Phase 50 · 2026-08-02 · 继承 E186 keep22 / object-specific CoACD C_

前置证据：

- [E178 主计划](194_E178_bucket_contact_aligned_top_segment_plan.md)
- [E178 Full 与评测结果](../log/239_E178_local_5090_hybrid_rebalance.md)
- [E178 12 门修订](../log/240_E178_tracking_error_numeric_gates.md)
- [E186 计划](204_E186_22case_object_specific_prg_full_plan.md)
- [E186 P/R/G blocker 结果](../log/255_E186_production_prg_shadow_blocker_results.md)

## 1. Context

E186 已证明当前路线中的 P 和 G 不是主要阻塞：

| 维度 | E186 结果 | E187 决策 |
|---|---|---|
| P：physics | keep22 CPU MuJoCo `22/22`、MJWarp `22/22` PASS | 冻结三个 object-specific CoACD compound collider，不重选 C/K/case |
| G：candidate gate | formal shadow false-safe/reject=`0/0` | 保持基于同源 `D_C-epsilon_grid` 的保守 gate |
| R：reward | bucket003 无捕获域；bucket007 grid/exact 排序失真 | 只重做 R，并在 Full 前冻结 reward 与 grid |

两个 R blocker 的证据不同：

1. bucket003 在 formal `1024×32` 下 posture/combined valid 都是
   `1024/1024`，但 hand 到 C 表面的最小距离仍约 `45.8–48.6mm`。旧 R 仅在
   `[-1,3]mm` 内非零，因此所有候选没有接近表面的引导。
2. bucket007 的 geometry active/valid 都是 `1024/1024`，G 也无 false-safe，
   但 5mm grid 与 `sigma=1.5mm`、4mm hard band 尺度不匹配，导致
   total rho=`0.80238`、top-k overlap=`0.5784`、selected0 mismatch。

因此 E187 的目标不是再次修改碰撞体，而是：

```text
Frozen CoACD C
  ├─ P: compound-convex physics                    (unchanged)
  ├─ D_C: coarsest fidelity-qualified grid-SDF
  │    ├─ R: wide-to-near continuous reward        (new, opt-in)
  │    └─ G: D_C - epsilon_grid conservative gate  (unchanged formula)
  └─ exact C / original-mesh D_M
       ├─ exact C: R/G fidelity authority only
       └─ D_M: C-vs-original-mesh fidelity only
```

E187 仍是 E178 的 paired replacement experiment。正式 Full 必须使用 E178 相同的
case、target、trajectory、contact mask、`1024×32`、seed0 和评测口径；唯一科学变量是
冻结 CoACD P/D_C 与新的连续 R。E178 的 scene、override、results、manifest 和日志均为
只读历史，不允许覆盖。

## 2. Frozen authority 与控制变量

### 2.1 E178 authority

```text
workspace/core4d/results/E178/s6_downstream/manifests/
  semantic_bucket_full_manifest.tsv
sha256 = de9a3d165301318049da208aad52b1f5bc4d3ed671631f0097958734a0f022a8
```

- E178 Full27：27 行，`1024 samples × 32 opt steps × seed0`；
- E187 authority：E186 已冻结的 keep22，顺序为 E178 Full27 删除 drop5 后的投影；
- E178 在 keep22 上的冻结 12 门 baseline：`8/22 PASS`；
- Full 后只允许按 case_id 与这 22 行做 inner join，禁止补入 drop5 或扫描目录发现 case。

### 2.2 keep22 与 collider

| Object | Frozen C | Max hull budget | keep cases |
|---|---|---:|---:|
| bucket003 | `E181__bucket003__t020_k16_v032` | 16 | 5/9 |
| bucket004 | `E181__bucket004__t005_k08_v064` | 8 | 4/4 |
| bucket007 | `E181__bucket007__t020_k08_v064` | 8 | 13/14 |
| Total | 三个 object-specific C | — | 22/27 |

这里的 max hulls 只是 CoACD 上限；runtime 必须读取冻结 manifest 中的实际 part 数和
ordered part SHA。E187 不测试 K=4，不减少 hull，不根据效率或 Full 结果重选 C。若最终
证明 hull 数导致不可接受的性能开销，应新开后续实验做 C-complexity ablation，并重新过
P、D_C、R、G 全部门，不能在 E187 内偷偷降精度。

### 2.3 科学控制变量

| Axis | Frozen value |
|---|---|
| retarget / target / hand | `omnirt_v1 / ref_fk / rubber_hull` |
| case authority | keep22 exact order |
| CEM | `samples=1024, opt_steps=32, seed=0` |
| P | E186 object-specific compound-convex C |
| G | existing body/hand/leg/posture thresholds；object distance用`D_C-epsilon_grid` |
| surface geom group | E178/E186 的 `['lh','rh']`，保持 non-bimanual min-distance 语义 |
| reward temporal gate | `contact_mask` |
| reward scale / tail | `1.5 / decay_frac=0.15` |
| other rewards | 与对应 E178 row exact，不启用 hand-approach/contact-mask额外 reward |
| physics/config | timestep、mass/inertia、friction、solver、joint、target和mask均与E186/E178 authority闭合 |
| Full resources | local GPU0 + `spider-remote` RTX 6000 Ada GPU0/1 |
| coexistence | 与已有程序叠加；禁止 kill、暂停、抢占、改动已有进程 |

允许变化只有：E187 method/output ID、连续 surface score 模式、冻结后的 D_C grid manifest、
worker owner 和运行时间戳。

## 3. E178 backward-compatibility contract（Gate 0）

E187 的实现必须先证明 E178 legacy 路径仍可复现；Gate 0 未通过前禁止 E187 GPU
shadow、canary 或 Full。

### 3.1 实现隔离

1. 新字段全部有关闭新行为的默认值；建议新增
   `surface_band_score_mode=distance_continuation` 所需参数，但历史默认仍为现有
   `one_sided`，E178 override 继续显式解析为 `symmetric_abs`。
2. `object_distance_backend=legacy_box` 与
   `surface_band_score_mode in {one_sided,symmetric_abs}` 必须进入原 legacy 分支；新纯函数
   只在 E187 override 显式选择 continuation 时调用。
3. 不修改任何 `core4d_E178_*.yaml`、`scene_act_E178_*.xml`、E178 manifest/result/log。
4. query-tape、transform recorder 和 diagnostic 字段保持 default-off、
   observational-only；关闭时不得改变随机数消费、CEM 选择或输出 schema 的必要字段。
5. 新配置字段即使出现在新生成的 `config_act.yaml` 中，也必须是无行为影响的默认值；
   historical config diff 只允许新增 default-only key、设备/输出路径/时间戳，不允许现有
   reward、gate、scene、seed、budget 值发生变化。

### 3.2 Gate 0 证据

| 检查 | 硬门 |
|---|---|
| E178 authority reconstruction | Full27 行数/顺序、输入/override/scene SHA、budget、seed全部 exact；manifest SHA仍为上述值 |
| source isolation | E178 override/scene/result/log的实现前后SHA不变；E187不得写入`results/E178/` |
| effective config whitelist diff | 三物体代表 case 的旧/新 config 除默认新字段、device、output、timestamp外无差异 |
| legacy pure reward | 固定 qpos/transform tape 上旧 surface、body/leg reward max abs error `≤1e-7` |
| legacy G | body/hand/leg/posture valid mask、fallback、elite/selected index逐值一致 |
| recorder no-op | recorder off 时 qpos/reward/selected index逐值一致；无 tape artifact |
| E178 replay | 三个代表 case 使用独立 `E187/e178_compat/` 输出；runtime finite，12门决策与历史一致 |
| same-device golden | 至少一条历史在本地5090执行的代表 row 做同设备 replay；关键query qpos/reward max abs `≤1e-5`，valid/selected一致 |
| semantic tolerance | 跨GPU代表 replay仅允许：tracking位置均值差`≤0.5cm`、姿态均值差`≤0.5°`、contact fraction差`≤0.02`、penetration fraction差`≤0.01` |

E178 replay 产物必须使用新 variant/output root，绝不覆盖历史 NPZ。若跨硬件数值超过上述
门，先区分硬件数值漂移与 legacy 行为变化；无法闭合时 E187 停止，不得用“新方法更好”
跳过兼容性问题。

## 4. Reward 设计与预注册参数

### 4.1 旧 reward

E178/E186 的 surface reward 为：

```text
R_old(d) = 1.5 * exp(-|d| / 1.5mm)
           * 1[-1mm <= d <= 3mm]
           * contact_gate * tail_decay
```

它在 3mm 处发生硬跳变，并且 3mm 捕获域小于 5mm grid 尺度。

### 4.2 E187 主 reward：双尺度 distance continuation

保持相同 hand geom group、min-distance reduce、scale、contact gate 和 tail decay，只替换
surface score：

```text
smooth_abs(d; delta) = sqrt(d^2 + delta^2) - delta

S_cont(d) = 0.25 * exp(-smooth_abs(d; 1mm) / 50mm)
          + 0.75 * exp(-smooth_abs(d; 1mm) / 15mm)

R_E187(d,t) = 1.5 * S_cont(d) * contact_gate(t) * tail_decay(t)
```

预注册解释：

- `50mm` 远场尺度覆盖 bucket003 已观测的约 48.6mm gap；该处归一化 score 约
  `0.12`，不再是零梯度区；
- `15mm` 近场尺度保留贴近表面的偏好，同时显著大于 1.5mm，降低 grid 小误差对排序的
  放大；
- `1mm` smooth-abs 避免 `|d|` 在零点的尖角；CEM 不要求梯度，但平滑函数便于建立
  grid-error 的 Lipschitz 上界；
- score 对正/负距离对称衰减，深穿透不会获得高 reward；最终安全仍由冻结 G 与真实 P
  保证；
- 不新增 penetration reward/penalty，不改变 bimanual 语义，不启用 E025
  hand-approach，从而保持单一科学变量。

本实验不在 Full 后调 `0.25/0.75/50/15/1mm`。旧 hard-band 和 broad-only score只用于
离线机制图/ablation，不是可根据结果回选的生产候选。若主 reward 未通过预注册门，
E187 以 R FAIL 结束，再新开实验；不得在同一 Full 结果上扫权重。

## 5. D_C grid 选择协议

P 的 C 不变；每个 object 从同一个 frozen C 烘焙多个不可覆盖的候选 grid。生产使用
“满足全部 R/G/效率门的最粗 grid”，以控制 Full 成本：

```text
h=5.0mm -> 若失败才测 h=2.5mm -> 若仍失败才测 h=1.25mm
```

每个 object 独立执行这个预注册的 coarse-to-fine 决策树；一旦某分辨率通过就冻结，不再
为该 object 查看更细 grid。padding 固定 120mm，sign、trilinear interpolation、
outside-grid rule 与 E186 v4 相同。每个候选仍做 50k smoke + 1M formal exact-C 验证，
冻结 grid payload/source/manifest SHA 和 `epsilon_grid`。

### 5.1 Grid/Reward/G gates

| 指标 | 硬门 |
|---|---:|
| finite / known sign | 100% / 100% |
| CPU/CUDA same-query max abs | `≤1e-5m` |
| Minkowski query+reward support padding | actual minimum `≥110mm` |
| G false-safe accept | `0` |
| selected exact-valid | `100%` |
| continuation component grid-vs-exact p99 abs | `≤0.05` reward unit |
| total reward Spearman | `≥0.99`，每个代表 object单独通过 |
| CEM elite top-k overlap | `≥0.90`，k使用正式elite count |
| grid-selected exact regret | exact top reward相对差 `≤0.5%`，且处于exact top 1% |
| R/G query throughput | 相对同 tape 的 E186 v4 median `≤1.25×` |
| process incremental peak GPU memory | `≤6GiB`，且无OOM |

`epsilon_grid`只进入 G 的保守下界，不从 reward 距离中减去；否则 reward 会产生系统性
偏移。exact-C 是 R/G fidelity authority，D_M 只继续验证 C 对 original mesh 的几何
保真度，不进入 runtime reward/gate。若 1.25mm 仍不能同时满足 fidelity 和效率，E187
停止；analytic convex-union runtime只能记为诊断/后续方向，不能在本实验临时替换
canonical grid-SDF。

## 6. Claims

| Claim | 最低证据 |
|---|---|
| C0 authority | E178 manifest SHA exact；keep/drop=`22/5`、无交并错误；三collider/parts SHA与E186 lock exact |
| C1 E178 compatibility | Gate 0全部通过；E178文件SHA不变；legacy reward/G/recorder no-op与代表 replay闭合 |
| C2 reward definition | continuation公式、参数、geom reduce、temporal gate与实现逐值一致；旧模式回归不变 |
| C3 canonical D_C | 每object按预注册决策树冻结唯一grid；exact-C、CPU/CUDA、padding、SHA和error bound闭合 |
| C4 P/G invariance | 22/22 CPU/MJWarp compile；pair=`18×K_actual`；P参数与G阈值不变；false-safe=0 |
| C5 bucket003 capture | formal tape/CEM shadow不再是trivial zero：score>0.05候选占比`≥50%`，reward finite，p95-p05`≥0.01` |
| C6 bucket007 fidelity | formal shadow total rho`≥0.99`、top-k overlap`≥0.90`、selected exact regret`≤0.5%` |
| C7 all-object R/G | 三代表 formal `1024×32` 均过C3–C6相关门，combined-valid恢复且selected exact-valid=100% |
| C8 production canary | 三物体各一条完整轨迹`1024×32 seed0`，runtime validator 3/3、无OOM/NaN/覆盖、视频可读 |
| C9 efficiency | same-hardware E187/E178 median plan-time ratio`≤1.50`，R/G kernel ratio`≤1.25`，报告GPU-hours与共存负载 |
| C10 Full closure | keep22满足`completed + terminal_failed = 22`且missing=0；正常目标22/22 completed |
| C11 paired result | 与E178同22行做12门、连续指标、bootstrap、效率和迁移分析；不把Full结果用于回调R/grid/C |
| C12 visual/handoff | 22/22 playable；3D视频与2D时序图人工观察完整；S6 evidence显式生成，不扫描目录 |
| C13 isolation | E178/E181–E186 artifacts不覆盖；不使用A100；不kill/暂停/抢占既有本地/远程进程 |

C0–C10 是兼容性、机制、运行和闭合 claim，不预设 E187 一定优于 E178。E187 是否改善
轨迹可用度，只能由 C11/C12 的 paired result 决定。

## 7. 实验阶段与 stop/go gates

### S0：Authority freeze + E178 Gate 0

1. 创建 E187 protocol manifest，引用而不复制修改 E178 Full27、E186 keep22、collider
   lock、compound scene 和 ordered parts；
2. 对 27 行 E178 输入及 22 行 E187 投影做路径/SHA/顺序/budget/seed closure；
3. 生成 E178 artifact SHA inventory；
4. 实现 default-off config 和 legacy golden tests；
5. 在独立 root 完成三代表 E178 replay，运行同一公共 evaluator；
6. 记录本地/远程依赖、GPU和既有process快照。

**Gate S0**：C0/C1 任一失败即停止。不得先跑 E187 再回头补 E178 compatibility。

### S1：Reward pure-function 与离线机制审计

1. 把 legacy surface score 和 continuation score做成可单测纯函数；
2. 在 synthetic signed distances `[-100,100]mm` 上验证 finite、对称、零点峰值、
   单调远离表面、连续性和解析误差上界；
3. 在 E186 reward-aligned transform tapes 上同时计算 legacy、continuation-grid、
   continuation-exact-C；
4. 生成 bucket003 远场支持曲线和 bucket007 hard-band跳变/连续score对照图；
5. 旧 reward仅用于解释机制，不参与生产参数选择。

**Gate S1**：公式/实现 max abs `≤1e-7`；003 score support与007连续性满足C5/C6的离线
前置门。失败时只修实现 bug；科学公式失败则结束 E187，不现场调权重。

### S2：Coarse-to-fine D_C freeze

按 §5 的 `5→2.5→1.25mm` 决策树逐 object 烘焙、验证并冻结最粗通过 grid。已有 E186
grid可作为5/2.5mm候选输入，但 E187 manifest必须引用其SHA并记录是否直接复用；新 grid
写入 E187 新目录，不覆盖 v4。

三代表 formal shadow固定为：

```text
bucket003_20231018_003_p1  record sim step 16
bucket004_20231002_021_p1  production-fixed contact query
bucket007_20231020_055_p1  record sim step 22
```

003/007可只读复用E186 formal reward-aligned transforms；004补一条同口径
`1024×32` bounded tape，所有新结果写 E187 root。参数选择只看这一阶段的 exact-C fidelity
和效率，不看任何完整轨迹 Full 结果。

**Gate S2**：C2/C3/C5/C6/C7全部通过后生成不可变 `reward_grid_lock.json`；SHA冻结后
禁止根据 canary/Full 改 reward/grid。

### S3：P/G regression 与 production integration

1. E187 override只在E186 override上追加E187 method、grid lock和continuation字段；
2. 重新验证22/22 CPU MuJoCo、22/22 MJWarp、pair count和compound asset SHA；
3. 44条reference/E178-final query复核G false-safe=0；
4. query tape保持off做 deterministic no-op；打开时仅用于bounded shadow；
5. 用公共 evaluator接口组织后续结果，禁止importlib加载其他实验 evaluator。

**Gate S3**：P/G invariance、override whitelist diff和22-row authority全过，否则禁止
production canary。

### S4：三卡 production canary + 效率 gate

三物体各一条keep case运行完整轨迹、正式`1024×32 seed0`：

| Worker | Device | Case policy |
|---|---|---|
| local-0 | local GPU0 | bucket003 representative |
| remote-0 | RTX 6000 Ada GPU0 | bucket004 representative |
| remote-1 | RTX 6000 Ada GPU1 | bucket007 representative |

三卡并行、每卡内部串行；允许与现有任务叠加，启动前后仅被动记录
`nvidia-smi`/process snapshot，不等待idle，不kill/暂停/抢占。远程使用独立 immutable
source snapshot和独立run root，不在远程脏worktree中merge/reset/checkout。

canary同时生成：

- result NPZ、outdir NPZ、config、row manifest、log；
- 3D E178-vs-E187 side-by-side MP4；
- 2D distance/reward/valid/fallback/plan-time时序图；
- per-record wall、R/G kernel、P step、峰值显存和共存process快照。

若三条 scientific config 与最终 Full lock SHA 完全一致，canary通过后可原子登记为 Full
前三行，避免重复计算；一旦修改任何 scientific field，旧canary不得promote，必须新root。

**Gate S4**：C8/C9全过才启动其余19行。若end-to-end ratio在`1.50–2.00`，只允许一次
不改变数值的kernel/cache优化并重过S0–S4；`>2.00`或OOM则停止Full。不得用减hull、减
samples或减iterations救效率。

### S5：keep22 Full CEM

根据S4实测cost做LPT，生成local/remote0/remote1三张互斥queue。每个worker只消费
execution manifest登记的rows，卡内串行、三卡并行：

```text
local GPU0
spider-remote RTX 6000 Ada GPU0
spider-remote RTX 6000 Ada GPU1
```

执行规则：

- 只用Ada两卡，不使用A100；
- 正式Full recorder-off；scene/config/lock SHA fail-closed；
- 每row输出路径唯一，complete artifact自动skip，禁止覆盖running/complete row；
- 单case失败先分类；只有阻塞条件发生真实变化时允许一次相同科学配置recover；
- 外部共存任务导致OOM时不操作该任务，也不降低CEM budget；无法安全恢复则记
  terminal_failed；
- 正常目标22/22 completed，最低closure为completed+terminal_failed=22且missing=0。

远程执行不依赖未满足claims时的git commit/push：先制作带文件清单和SHA的immutable
source snapshot并rsync到独立工作目录；只有E187全部claims收口后才按实验规则
commit/push。

### S6：Paired evaluation、3D/2D visual 与结论

只join keep22，E178 baseline固定为12门`8/22`：

1. 公共物理/跟踪 evaluator：原6门、tracking 6门、12门交集；
2. 连续指标：root/hand/object位置姿态、in-mask contact、release false contact、
   3/5mm penetration、lower-body contact、smoothness、fall/body-z；
3. 逐object和总体迁移：fail→pass、pass→fail、win/tie/loss；
4. 对连续paired delta做10k bootstrap CI；总体采用case bootstrap，并补object-stratified
   sensitivity；
5. efficiency：同硬件canary做因果比较；异构Full只报告逐卡wall、GPU-hours和负载，
   不把5090/Ada差异误归因算法；
6. 22/22视频与固定关键帧；重点观察穿透、phantom contact、手是否被引向真实bucket
   表面、spider重定向轨迹是否可用；
7. 生成盲化E178/E187 paired review workbook，用户人工裁决与numeric gate分开记录；
8. 明确区分`DOWNSTREAM_CEM_PASS`与RL成功，不自动宣称RL可用。

Full结果禁止反向调reward/grid/C。若结果不佳，E187仍作为完整负结果收口并新开实验。

## 8. 成功标准与决策

### 8.1 运行/机制成功

- Gate S0–S4全过；
- Full 22/22 completed；
- 三对象均有非平凡continuation support；
- G false-safe=0、selected exact-valid=100%；
- grid/exact total rho≥0.99、top-k overlap≥0.90；
- same-hardware plan-time ratio≤1.50；
- 22/22 playable且实际视觉观察完成。

### 8.2 相对 E178 的结果分级

| 结论 | 预注册口径 |
|---|---|
| `E187_BETTER` | 12门PASS `>8/22`，且E178原PASS无pass→fail；主要contact/penetration指标方向一致，人工可用度不下降 |
| `E187_NONINFERIOR` | 12门PASS `≥8/22`，pass→fail=0，关键连续指标的object-stratified CI无明确退化 |
| `MIXED` | 数字门或连续指标互有得失，或人工/numeric结论不一致 |
| `E178_BETTER` | 12门PASS下降、出现不可接受pass→fail，或人工可用度明确下降 |
| `INCONCLUSIVE` | terminal_failed>0、视频/配对缺失、authority/compatibility不闭合 |

“可复现 E178”与“E187 优于 E178”是两个独立结论：前者由Gate 0证明，后者只能由S6
证明。

## 9. 需要修改/新增的文件（实现阶段）

| # | 文件 | 计划改动 |
|---:|---|---|
| 1 | `spider/config.py` | 新增default-off continuation参数与fail-closed validation；legacy默认行为不变 |
| 2 | `spider/simulators/mjwp.py` | 独立continuation score分支；旧one-sided/symmetric_abs分支保持回归 |
| 3 | `spider/optimizers/sampling.py` | 仅在query-tape显式开启时记录新score诊断；off时no-op |
| 4 | `workspace/core4d/scripts/experiments/E187/` | authority、grid选择、reward shadow、builder、queue和全部static tests |
| 5 | `examples/config/override/core4d_E187_*.yaml` | 22个E187 opt-in override，不修改E178/E186 override |
| 6 | `workspace/core4d/scripts/eval/core/core_metrics.py` | 只有确属公共的新指标才参数化加入；不得放实验特有逻辑 |
| 7 | `workspace/core4d/scripts/eval/runners/eval_E187_*.py` | E178 compatibility、R/G fidelity、Full paired evaluator |
| 8 | `workspace/core4d/scripts/eval/wrappers/eval_E187_*.sh` | 固化所有评测入口 |
| 9 | `workspace/core4d/scripts/eval/reports/gen_E187_*.py` | paired表、2D诊断和review workbook |
| 10 | `workspace/core4d/scripts/train/train_core4d_E187.sh` | production CEM canonical入口；首步scene snapshot |
| 11 | `workspace/core4d/scripts/launch/active/run_E187_local.sh` | local GPU0串行worker |
| 12 | `workspace/core4d/scripts/launch/active/run_E187_remote_a6000.sh` | Ada GPU0/1两队列、tmux、immutable snapshot |
| 13 | `workspace/core4d/scripts/launch/active/pull_E187_remote_a6000_results.sh` | 按execution manifest回收与SHA验证 |
| 14 | `workspace/core4d/scripts/launch/active/watch_E187_full.sh` | hardened monitor/pull/eval，不误杀进程 |

实现前先用 `gen_experiment.py --dry-run` 预览标准脚本骨架；实验特有builder仍手写。任何
真实命令必须固化在上述脚本中，log不记录无法复用的裸命令。

## 10. 计划执行入口

以下是实现后应存在的canonical入口；本计划创建时它们尚未生成，不得据此误报可运行：

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E187_e178_compat.sh
bash workspace/core4d/scripts/eval/wrappers/eval_E187_reward_grid_fidelity.sh
bash workspace/core4d/scripts/eval/wrappers/eval_E187_compound_regression.sh

bash workspace/core4d/scripts/launch/active/run_E187_local.sh canary
bash workspace/core4d/scripts/launch/active/run_E187_remote_a6000.sh canary
bash workspace/core4d/scripts/launch/active/pull_E187_remote_a6000_results.sh canary

bash workspace/core4d/scripts/launch/active/run_E187_local.sh full
bash workspace/core4d/scripts/launch/active/run_E187_remote_a6000.sh full
bash workspace/core4d/scripts/launch/active/pull_E187_remote_a6000_results.sh full

bash workspace/core4d/scripts/eval/wrappers/eval_E187_paired_full.sh --require-all
```

## 11. Artifacts、快照与复现

```text
workspace/core4d/results/E187/
├── s0_environment/
│   ├── protocol_manifest.json
│   ├── e178_authority_sha_inventory.tsv
│   ├── e178_compat/
│   └── execution_manifests/
├── s1_reward_design/
├── s2_canonical_grid_sdf/
├── s3_prg_audit/
├── s4_canary/
├── s5_handoff/
├── s6_downstream/
│   ├── cem/full/
│   ├── eval/full/
│   └── render/full/
├── scene_snapshot/
└── registries/
```

训练/Full前必须调用scene snapshot脚本并记录git HEAD、scene/override/grid/collider/input SHA。
活跃的22条scene XML必须确认已被主git跟踪；若尚未tracked，按规则`git add -f`，但不
改文件内容。results本身不进入git，plan、实现、tests、launchers、log、Tracker和progress
进入git。Claims全部通过前不commit/push；远程靠immutable source snapshot复现。

## 12. Stop rules 与三次失败协议

1. Gate 0失败：先定位legacy兼容；不得继续新reward。
2. 同一问题第1次失败：诊断并修bug，使用新versioned root；
3. 第2次同类失败：执行计划内替代路径，如下一档grid分辨率，不能重复同配置；
4. 第3次仍失败：停止E187并向用户汇报，不继续扩大搜索；
5. 任一Full产物出现后，reward/grid/C/keep22全部不可改；
6. 不通过的case不能像P阶段那样在Full后再丢弃；22行是E187 frozen authority；
7. 不kill、不暂停、不抢占现有程序；不使用A100；
8. 可视化有条件时必须执行，MP4必须用`video-frames`抽帧并把实际观察写入结果log；
9. 若Claims未全过，不commit/push；若全部通过，按规则commit、push并在最终log记录SHA。
