# CORE4D 当前进度

> 完整备份：
> [E167–E179 初始恢复](progress_archive/E167_E179_20260724_full_backup.md) ·
> [E179 执行过程](progress_archive/E179_20260724_execution_full_backup.md) ·
> [E179 最终收尾](progress_archive/E179_20260725_closeout_full_backup.md) ·
> [E180–E181 完整执行](progress_archive/E180_E181_20260731_full_backup.md) ·
> [E182 Gate0–S1 canary](progress_archive/E182_gate0_s1_canary_20260801_full_backup.md) ·
> [E182 S1 continuation](progress_archive/E182_s1_continuation_20260801_full_backup.md) ·
> [E182 S2 v2–v8](progress_archive/E182_s2_v2_v8_20260801_full_backup.md) ·
> [E182 v9 完整过程](progress_archive/E182_v9_20260801_full_backup.md)
>
> 本文件只保留 E182 当前权威状态与下一入口；v9详细结果见
> [log250](log/250_E182_bucket003_double_plane_preseg_v9_results.md)。

## 2026-08-01：不可变执行口径

- 权威计划：`plan/200_E182_task_conditioned_coacd_full_cem_plan.md`。
- Full 与 E178 exact：27 cases、`seed0, 1024×32`；不测试K4。
- Full资源只允许本机单卡 + `spider-remote` RTX 6000 Ada GPU0/1叠加运行；禁止kill、
  暂停、抢占或修改既有进程，不使用A100。
- heldout24 在production SHA冻结前selection-forbidden，禁止按heldout/Full反选碰撞体。
- bucket003完整P/R/G launch floor未闭合前，grid-SDF、heldout和Full全部禁止。

## E182 已完成阶段

- Gate0 authority/preflight PASS；Gate1真实P/R/G query tape PASS，见log247。
- v5 global threshold=`0/9`；v6 per-segment hybrid=`0/6561` each K；
  v7 Bell(4) partition=`0/15` each K，见log248。
- v8 single-plane family 12/12构建、static P=`0/12`；最佳balance=`TP18/phantom7`，
  TP19最少phantom=12；3D/2D diagnostic确认rim convex envelope问题，见log249。

## v9 simultaneous double-plane：完整负结果

- 用户批准并冻结同时使用两条既有plane：
  `-0.14311002844145554/-0.11878629238288715m`；7 unchanged+3 children=10 segments。
- family固定为`threshold {5,10,20mm} × K {16,32}=6`；K8结构不可行，K4禁止；
  static gate=`TP>=19 AND phantom<=8`。
- pre-freeze functional matrix=`55/55 PASS`；ruff/format/compileall/diff-check PASS；
  protocol SHA=`5ea0245bd0cbe7808af3c3d0875d18067b6d1b33f35aaac3d73b546439acde62`。
- 3-child exact geometry PASS，volume closure delta=`3.584199e-11m³`。
- 只运行3个middle CoACD，hulls=`4/4/2`；21个v4 unchanged与6个v8 outer manifest
  只读复用；composite totals=`40/36/24`。
- 6/6 candidates BUILD_PASS；actual hulls=`16/32/16/32/16/24`，全部10-segment、
  no-cross merge、actual`<=K`、max vertices`<=256`。
- static P confusion（TP/phantom/missed）：
  `22/24/5,22/21/5,19/17/8,19/14/8,15/10/12,15/7/12`；结果=`0/6 PASS`。
- aggregate SHA=`87c401182e082c54de4dec5b0b4c4797c0ef2b5c552bb24b60b417d84b5a642c`；
  `selected_candidate_ids=[]`、`full_prg_eligible=false`。

## 当前权威状态

```text
V9_STOPPED_COMPLETE_NEGATIVE_RESULT
STOP_V9_NO_MORE_PLANES_OR_FLOOR_CHANGES
P_LAUNCH_FLOOR_NOT_CLOSED
FULL_NOT_AUTHORIZED
HELDOUT_NOT_ACCESSED
```

- V9-C1/C2/C3/C4/C6 PASS；V9-C5 FAIL。
- heldout/grid/GPU/Full访问均为0，本地/远程既有进程未被操作。
- 按预注册stop，不移动/增加plane、不降低0.70 floor、不追加K或threshold、不运行
  完整P/R/G、physics replay、v9 visual、grid或Full。
- 若继续E182，必须由用户批准实质不同的方法自由度，例如rim-aware非平面primitive或
  MuJoCo SDF/plugin高保真局部碰撞；不能继续给CoACD平面切分加手工规则。

## 归档闭环

- log250已创建；Tracker E182已更新并链接log250；log INDEX已包含250。
- v9正式结果根共175文件，位于results symlink外；protocol/segment/base/build/static
  validators均已二次PASS。
- git归档范围：plan200 amendment、v9 builder/tests、log250、Tracker、INDEX、progress
  与本完整备份；结果artifact沿用results外置存储策略。
- progress已完整备份并精简；最终ruff/format/py_compile/diff-check、文档链接与
  Tracker/INDEX一致性均PASS。E182 v9 closure已提交，未push；worktree待终检。

## 2026-08-01：static-P 全 case 扩展前只读性能盘点

- 用户要求后续 static-P 覆盖 E178 全部27 case，而不是只使用
  `bucket003_20231018_001_p1` 的 reference/E178-final 两段轨迹。
- E178 Full authority 的物体分布为 bucket003/004/007=`9/4/14`；当前v9候选仅属于
  bucket003，禁止把该几何直接用于 bucket004/007，扩展需三物体各自候选与query。
- 当前 Open3D `RaycastingScene.compute_signed_distance` 未指定CUDA device，static-P和
  CoACD均按CPU任务处理；下一步只读实测单case拆分耗时及1/2/4/8 CPU进程吞吐，不写正式
  v9 artifact、不访问heldout E182 evidence、不启动Full或GPU任务。
- 本机 Ryzen 9 9950X3D（16C/32T）、已有任务叠加状态下，bucket003现有882-pose case：
  query/oracle准备=`0.191s`，6个candidate核心总计=`6.385s`，含逐candidate清理的外层
  case wall=`7.378s`；每candidate total=`1.042–1.092s`，其中scene build仅
  `0.010–0.024s`，主要成本是signed-distance query。
- 8个相同case-equivalent的CPU多进程实测（worker=`1/2/4/8`）wall=
  `56.979/37.251/28.903/24.417s`，相对加速=`1.00/1.53/1.97/2.33x`；全部指标exact。
  单worker累计CPU/wall约`560.3/57.0=9.8 cores`，说明Open3D单进程已内部并行，外层
  进程可并行但强烈次线性；8 worker单任务膨胀至`23.94s`，不应按worker数线性估算。
- E178 27 case 的reference frame总数为`3298`，E178-final同为`3298`；现有tape合同下
  每case static pose数=`(2T+50)+2T=4T+50`，所以全量预计`14542 poses`，约等于当前
  882-pose case的`16.49x`，不是简单的27倍。按当前bucket003几何线性外推：单一冻结
  candidate全27 case核心评分约`21s`串行；6 candidates约`1.8–2.2min`串行，4/8 worker
  预计约`55–65s/47–55s`（不同物体hull复杂度与尚未生成的static tape会带来偏差）。
- static-P的reference/E178-final query可由`build_prg_query_tape.py`在CPU上从冻结qpos
  materialize；只有扩展CEM sample的R/G `on_a` tape才需要MJWarp GPU replay（历史dev
  canary约`106s/case`）。因此本次“全case P碰撞校验”应走CPU池，GPU保留给Full CEM或
  R/G query capture。

## 2026-08-01：E183 Full27 static-P coverage audit启动

- 用户批准按上述口径执行。新建plan201，E183作为独立full27 coverage audit，不改写
  已收口E182-v9；本轮将有意访问27个已知E178 case，但结果标记evaluation-only，不能
  回头修改plane/K/threshold或再声称heldout独立。
- 候选在score前冻结为E181标准CoACD 54个（每物体18）+ bucket003 E182-v9 6个，合计
  60个；只做object-matched评分，预期candidate-case row=`540`。query固定为27 case
  reference+E178-final，预计`14542 poses`；4 CPU worker，不使用GPU。
- E183 evaluator首版已实现，formal result root确认ABSENT；并行任务按candidate分组，
  每次加载candidate scene后连续评相应object全部case，以复用加速结构。首次静态检查中
  `.venv/bin/ruff`不存在（py_compile与diff-check无报错），这是工具路径问题、未执行正式
  protocol/query/score；下一步定位repo实际ruff入口后继续，不安装或改动环境。
- 已定位权威lint入口为`uv run ruff`；首版经doc/import修正与机械format后ruff、format、
  py_compile、diff-check全部PASS。只读preflight确认authority=`27`且物体`9/4/14`、
  candidate=`60`且物体`24/18/18`、来源E181/v9=`54/6`，三oracle manifest/cleaned mesh
  SHA均闭合；formal root仍ABSENT，尚未访问full27 score。
- 临时目录dev回归3/3 PASS：重建query=`882 poses/27 oracle contacts`，v9六行
  TP/phantom/missed与log250 exact，篡改candidate result会被pooled-confusion validator
  拒绝；测试未写formal artifact。测试/runner ruff与format全PASS，已固化CPU-only wrapper，
  下一步是freeze正式protocol并执行4-worker full27。
- 正式pre-freeze matrix（bash-n/ruff/format/compileall/diff-check/root-empty）PASS；已在空root
  冻结protocol，candidate=`60`，protocol SHA=
  `7ed82ada0dc2675dac17e3cc435de45fae79495b86a5e8ed06661c8e807d172f`。从此runner及
  authority/candidate/oracle任一SHA变化都会拒绝resume；下一步正式4-worker query build。
- 正式query build COMPLETE：27/27 case、`14542 poses`、oracle contact总数=`2198`，
  4-worker wall=`4.820s`、max worker RSS=`2038.6MiB`、tape=`275MiB/81 files`。发现
  bucket007有2个case oracle contact=`0`、另3个仅`1/4/3`，按冻结的E182 metric其
  per-case recall会为0；不事后改protocol，后续同时解释pooled/macro与零阳性case限制。
- 正式score/aggregate COMPLETE：60/60 candidates、540/540 candidate-case rows，4-worker
  score wall=`51.267s`（sum candidate wall=`187.215s`，max RSS=`993.6MiB`），GPU访问0；
  v9六行dev回归6/6 exact。pooled/macro/all-case coverage PASS=`23/19/3`。
- 分物体结论：bucket003 `0`个all-case PASS，最佳E181`t020_k16_v032`仅`5/9`且
  pooled P/R=`0.645/0.863`；bucket004有3个`4/4` PASS，最佳`t005_k08_v064`
  macro=`0.859/0.853`、pooled=`0.874/0.876`；bucket007最佳`t005_k16_v064`
  `11/14`且macro/pooled均PASS，但all-case因3个case失败（含零阳性case）为FAIL。
- visual/validation COMPLETE且validator PASS。实际观察：bucket003最佳在reference早段和
  E178-final后段产生长串phantom，final接触切换处另有少量missed，符合precision主导失败；
  bucket004最佳总体跟随oracle，reference中有零散missed、final中有离散phantom但worst
  case仍P/R=`0.767/0.752`；bucket007零oracle worst case中candidate在reference/final两段
  均产生密集接触（166 phantom），因此不只是“零阳性recall定义”为0，而是真实严重过碰撞。
- E183 log251已创建并完整记录object结果、零阳性case解释、效率、可视化观察、Claims与
  artifact SHA；Tracker新增Phase46索引。最终科学决策：bucket003/004/007全case可用
  candidate=`0/3/0`，Full不启动；bucket004 K8可用，bucket007 mixed，bucket003仍需新方法。
- log INDEX已重建并新增Phase46；最终ruff/format/bash-n/compileall/diff-check、formal
  validator与standalone tests 3/3均PASS。正式result共277MiB沿results外置策略不入git；
  下一步只提交plan/runner/tests/log/Tracker/INDEX/progress，本轮不push、不启动Full/GPU。

## 2026-08-01：E184 static-P gate 0.65 sensitivity启动

- 用户要求TP覆盖放宽到65%，phantom对应放宽。新建plan202，正例case冻结为recall和
  precision各`>=0.65`，即`phantom<=floor(TP*0.35/0.65)`；oracle零接触case单列为
  `phantom==0`才PASS，并保留E183 legacy口径对照，避免混淆threshold与零阳性语义收益。
- E184只读E183已冻结540行confusion重新聚合，不重跑SDF/CoACD，不访问GPU/Full，
  E183 0.70 artifact与结论不改。下一步实现pre-score protocol与边界测试。
- 已核对E183权威表头与聚合实现：case表直接保留TP/phantom/missed/TN，足以无损重算；
  candidate表提供0.70 legacy exact对照。E184将同时输出legacy与zero-aware两套all-case，
  并把positive-only macro独立列出，避免零oracle语义变化污染0.70→0.65阈值迁移。
- E184 evaluator、4项standalone tests与CPU-only wrapper首版已实现。runner覆盖SHA锁定、
  protocol先冻结、case/candidate双表、legacy/zero-aware分解、0.70回归、0.65迁移、对比图与
  isolation validator；尚未运行测试或formal protocol，下一步先做静态检查和临时根回归。
- 首轮临时回归的三项边界/SHA测试PASS，end-to-end在source结果目录为外置symlink时暴露
  lexical path与resolve path不一致；这是artifact路径记录问题，不涉及指标。已改为保留repo
  lexical路径，并同步修复ruff提示；formal root仍未创建，待复跑4/4测试。
- 临时根end-to-end现4/4 PASS；ruff、format、bash-n、py_compile均PASS，E183 legacy
  pooled/macro/all-case=`23/19/3` exact复现。formal E184 protocol仍未冻结；下一步确认正式root
  为空、diff无越界改动后执行CPU-only wrapper。
- 正式pre-freeze检查PASS：E184 root ABSENT，E183 case/summary SHA exact，git diff-check
  无报错，改动仅E184 plan/runner/tests/wrapper/progress。现在允许冻结formal protocol并执行
  offline aggregate/visual/validate；GPU、SDF、CoACD与Full访问继续为0。
- formal wrapper首次调用在进入Python前失败：复用的旧wrapper目录层级多退了一层，导致
  在`/home/ubuntu/Workspace`查找`.venv`；protocol和任何result均未创建。已把E184 wrapper
  repo-root跳转修正为5层，需重新做bash-n/root-empty检查后再执行。
- wrapper修复后root-empty、4/4 tests与diff-check再次PASS；formal protocol已冻结，随后
  aggregate/visual/validate全部COMPLETE/PASS，总wall=`0.84s`，zero-aware all-case 0.65初值
  为`4/60`。下一步读取分object迁移、v9与失败case，并实际查看对比图。
- 正式聚合读取完成并实际查看`threshold_comparison.png`：图例、分组、数值标签与4 panel均
  清晰无异常。0.65相对0.70仅新增1个zero-aware all-case候选，即bucket004
  `t005_k16_v064`；bucket003/bucket007仍为0个全覆盖。下一步补齐best/failure与零语义分解。
- 深入分解完成：bucket003 v9 K16虽pooled/macro+过线但仅5/9；bucket007 K8的13/14来自
  两个clean zero-oracle case按正确语义PASS，剩余case为TP0/missed3；新增bucket004 K16
  的边界case为TP37/phantom19/missed3、precision=0.661。log252与Tracker Phase47已写入，
  下一步重建INDEX并做最终validation/commit。
- log INDEX已自动重建，新增Phase47/E184且总日志251份；diff-check无报错。归档文件现仅
  E184 plan/runner/tests/wrapper/log252、Tracker/INDEX/progress，正式result沿既有外置策略
  不进入git。下一步执行最终ruff/tests/formal validator/文档链接检查后本地commit，不push。
- 最终ruff/format/bash-n/compileall、4/4 standalone tests、formal validator、doc links与
  diff-check全部PASS；protocol对当前runner/wrapper仍exact。下一步清点untracked目录，排除
  compileall缓存后提交E184归档。

## 2026-08-01：E185 static-P gate 0.60 sensitivity启动

- 用户要求继续检查60%口径。已完整恢复Tracker、E184 plan/log/progress并新建plan203；
  positive-case冻结为precision/recall各`>=0.60`，即
  `phantom<=floor(TP*0.40/0.60)=floor(2TP/3)`，TP18/19/22对应12/12/14。
- zero-oracle继续使用`phantom==0`，primary comparison冻结为0.70→0.65→0.60；E185直接
  读取E183两张权威TSV，不从E184计数派生，不访问SDF/CoACD/GPU/Full。下一步在formal
  root创建前实现runner/tests/wrapper并做临时根回归。
- E185 runner、4项standalone tests与CPU-only wrapper首版已实现；runner独立从E183计数
  重算三阈值，使用E184 aggregate SHA作0.70/0.65 regression authority，并生成三阈值
  case/candidate表、4-panel对比图和isolation validator。尚未运行测试或创建formal root。
- 临时根end-to-end 4/4 PASS；ruff、format、bash-n与py_compile均PASS，E184 primary
  overall的0.70/0.65 pooled、macro+、all-case六项exact复现。formal E185 root尚未创建，
  下一步执行source SHA/root-empty/diff-check后冻结protocol。
- 正式pre-freeze matrix PASS：E185 root ABSENT，E183两表与E184 aggregate SHA exact，
  diff-check无报错，worktree只有E185 plan/runner/tests/wrapper/progress。允许执行CPU-only
  formal wrapper；SDF/CoACD/GPU/Full访问继续为0。
- formal protocol已冻结，aggregate/visual/validate全部COMPLETE/PASS，总wall=`0.83s`；
  zero-aware all-case在0.60下初值为`7/60`（0.65为4/60）。下一步读取分object迁移、v9、
  hull效率和remaining failure，并实际查看三阈值对比图。
- aggregate/validation读取确认0.60新增的3个all-case candidate全部属于bucket004；
  bucket003/bucket007 all-case仍为0，最佳coverage仍5/9与13/14。首次只读诊断打印在拼接
  int/str时TypeError中止，正式artifact不受影响；后续修正诊断表达式继续读取。
- 已实际查看三阈值4-panel PNG：0.70/0.65/0.60图例、柱高、标签和coverage均清晰；视觉
  显示0.60只扩大bucket004 all-case 4→7，bucket003/007 pooled增加但all-case与最佳coverage
  均不动。下一步补齐新晋级候选、best failure与v9明细。
- 修正只读诊断表达式后明细闭合：0.60新增bucket004 K16/K8/K16三个candidate；bucket003
  v9 K16仍有P=0.478、R=0.378、全missed和R=0.586四个失败case；bucket007 K8剩余失败
  为TP0/missed3。bucket004共有7个4/4，其中3个K8，原`t005_k08_v064`仍为裕量首选。
- log253与Tracker Phase48已写入，明确all-case 0.70→0.65→0.60=`3→4→7`且新增全属
  bucket004；下一步重建log INDEX并执行最终validation/commit。
- log INDEX已自动重建，新增Phase48/E185且总日志252份；diff-check无报错。归档范围仅
  E185 plan/runner/tests/wrapper/log253、Tracker/INDEX/progress，正式result继续外置不入git。
  下一步执行ruff/tests/formal validator/doc-link检查，排除缓存后本地commit，不push。
- 最终ruff/format/bash-n/compileall、4/4 standalone tests、formal validator、doc links与
  diff-check全部PASS；protocol对当前runner/wrapper和三份authority SHA仍exact。下一步
  删除untracked compile缓存并提交E185归档。
- untracked清点只发现两个compileall生成的E185 `.pyc`及预期归档文件；将显式删除该缓存，
  不触碰正式result或用户文件，然后按已列明范围提交。
- E185归档已本地提交且未push；提交包含plan203、runner/tests/wrapper、log253、
  Tracker/INDEX/progress，正式result沿外置策略保留。E185 closure完成，worktree待最终确认。

## 2026-08-01：E186 22-case object-specific collision R/G + Full启动

- 用户明确修订production口径：不要求每个物体全部case通过；固定一个object-specific
  碰撞体后，只保留该碰撞体通过P gate的case，失败case允许永久丢弃。E185历史log不改写，
  E186作为新selection/handoff authority。
- 当前冻结候选拟定为bucket003 E181`t020_k16_v032`、bucket004 E181`t005_k08_v064`、
  bucket007 E181`t020_k08_v064`。在zero-aware P>=0.70下保留`5+4+13=22/27`，且该22数量
  在0.65/0.60不增加，因此production继续使用更严格0.70，不使用降阈值收益。
- 资源合同继承用户已批准口径：本机单卡 + `spider-remote` RTX 6000 Ada GPU0/1并行，
  可与既有任务叠加，禁止kill/暂停/抢占现有进程。下一步只读审计E178 R/G/Full入口、
  data_construction_v3 S5/S6合同和本地/远程进程，再创建plan204，尚未启动GPU任务。
- 已完整读取`experiment-planning-zh`与`data-construction-v3-zh`技能并开始核对相关v3文档。
  E186必须使用`workspace/core4d/results/E186/`持久根、显式S5 handoff与S6 downstream
  evidence；target route保持`ref_fk`，碰撞体方法写`spider_method_id`而非target variant。
- Full结果不能被直接目录扫描冒充handoff；必须保留scene/trajectory/contact mask/result、
  source exp和method身份。当前仍处于只读设计审计，未创建plan204、未改代码、未触碰GPU。
- 已定位E178主计划194、hybrid计划195及E181 canonical plan199。E186控制变量应保持E178
  `seed0,1024×32`、`omnirt_v1/ref_fk/rubber_hull`、reward/gate/target exact；允许差异仅为
  object collider、同源R/G backend、22-case投影、输出和worker owner。
- E178 Full authority SHA=`de9a3d...022a8`，E182 authority保留逐行输入SHA；E181计划已明确
  三卡合同为本地1卡+Ada GPU0/1且每卡约9条。下一步核对已实现nonbox physics/grid-SDF
  runtime、E182 R/G query状态与现有CEM runner，不能把仅P静态候选直接误当Full-ready。
- runtime审计确认production `spider/config.py`目前只接受`primary/union`，且`union`强制
  object collision geoms为box；`spider/simulators/mjwp.py`的R/G查询也是box-union SDF。
  E181/E182尚未提供可直接用于Full的canonical grid-SDF或MJWarp compound-convex backend，
  因此E186必须先完成同源P与R/G runtime/parity，再进入canary与Full，禁止用E178旧box
  backend重跑后冒充CoACD实验。
- 已按用户批准口径创建plan204：E186唯一production authority是固定三collider在
  zero-aware P>=0.70下的keep22；drop5永久标记P_REJECTED，禁止进入R/G、canary、Full。
  三collider、grid与22-case在Full前冻结，Full结果不得反向选K/C/grid。计划同时冻结
  E178 exact控制变量、本地GPU0+远程Ada0/1叠加资源、S5/S6合同和分阶段stop/go gates。
- authority实现输入已定位：E183原始case计数表SHA=`28a3c811...a949c`、candidate表
  SHA=`08b8927c...6093`、protocol SHA=`7ed82ada...72f`；E181三个manifest均含
  candidate asset SHA、actual hull count和ordered part path/SHA，可构成fail-closed
  collider lock。E178 manifest字段足以直接投影22行并保留trajectory/mask/scene SHA。
- E186 S0 authority freezer、4项standalone tests与canonical wrapper首版已实现。freezer
  从E178/E183 frozen SHA重算zero-aware P gate，验证三manifest与32个ordered convex
  parts（16/8/8）SHA，输出immutable protocol/collider lock/evidence/keep22/drop5；正式root
  尚未写入，下一步先做ruff/format/bash/临时根回归，任何失败先修代码再freeze正式authority。
- 首次静态检查在执行正式protocol前被ruff D103阻止：runner的`parse_args/main`与4个
  standalone test缺docstring；这是文档lint问题，formal E186 root仍ABSENT。已补docstring，
  按“不重复失败”规则下一次执行完整静态矩阵与临时根测试，而不是忽略lint直接freeze。
- D103修复后ruff、bash-n、py_compile、diff-check与formal-root-empty均PASS；format-check
  仅要求对freezer做机械排版，未发现语义问题。下一步先运行ruff formatter，再复跑静态矩阵
  和4项仅写临时目录的end-to-end tests；正式root继续保持为空直到这些合同全部通过。
- formatter后ruff/format-check与4/4临时根contracts全部PASS：边界、formal source/collider、
  tamper rejection、idempotent freeze均闭合，且正式E186 root仍ABSENT。现在满足plan204
  Gate0的pre-freeze条件，允许通过canonical wrapper写入正式S0 authority；该操作CPU-only，
  不访问SDF/MuJoCo/GPU/Full。
- 正式E186 Gate0 authority已冻结并validate PASS：keep/drop=`22/5`，分object keep=
  `5/4/13`；protocol SHA=`e74ca890...95795`、collider lock SHA=`6a20df7c...66065`、
  collider-set SHA=`1bc7efe5...22e62`、keep22 SHA=`35d028f3...eb535`、drop5 SHA=
  `baa2dc11...a604`。正式结果仅写`results/E186`且不入git；下一门为S1 canonical grid-SDF。
- S1首版已实现新的`spider.geometry.CanonicalGridSDF`与E186 baker/tests：float32
  object-local trilinear grid、candidate/grid SHA fail-closed、`D_C-epsilon`保守gate查询，
  grid外使用object AABB距离下界避免false-safe accept。builder从32个frozen parts做
  manifold exact union/Open3D signed distance，5mm/50mm、1M点/object验证；尚未正式bake。
- 首次S1静态检查仅发现2个import-sort和3个format差异；py_compile/diff-check未报语义
  错误。进一步代码审查同时修正surface gate只评0h顶点（不误把±1..3h当surface），并
  增加existing frozen manifest的只读幂等resume，防止重跑用wall time覆盖正式manifest。
- S1 import-sort与format已机械修复，随后ruff/format-check全PASS。下一步运行4项grid
  runtime unit contracts（affine interpolation、AABB OOB lower bound、identity/payload
  tamper、grid alignment）及py_compile/diff-check；正式S1目录仍未创建。
- S1 runtime unit contracts 4/4 PASS，py_compile/diff-check PASS，formal S1 root仍ABSENT。
  现在进入临时根三物体真实bake smoke：保持5mm/50mm，只把validation降至50k/object，
  用于验证manifold union、Open3D sign、grid内存、实际error gate与CUDA parity；smoke结果
  不作为正式authority，失败将先按根因修builder或gate，不重复同配置直接formal。
- 首次真实smoke在bucket003 grid计算完成后的final manifest序列化失败：validation gates
  中至少一个值为`numpy.bool_`，stdlib JSON拒绝；正式S1仍ABSENT，临时root保留PENDING
  半成品且不复用。已将全部gate显式归一为Python bool并新增JSON序列化contract，下一步
  先跑5项unit/static，再换全新临时root复跑，不重复使用失败root。
- bool归一修复后ruff、5/5 grid contracts、py_compile、diff-check与formal-root-empty均
  PASS。现在使用全新临时root重跑同一5mm/50k三物体smoke；旧PENDING root不再消费。
- 新临时根5mm/50k三物体真实smoke PASS：grid shape bucket003/004/007=
  `130x176x115 / 87x114x82 / 129x137x135`，payload约`10.0/3.1/9.1MiB`；p99 error=
  `1.428/1.246/1.131mm`，sign disagreement outside2h均0，CUDA parity max约
  `0.07–0.12um`。但bucket004单点max=`12.73mm`导致全局epsilon过保守，暂不freeze 5mm。
- 下一步按E181/E186 coarse-to-fine合同做2.5mm临时敏感性，重点观察bucket004 max与payload；
  不用Full/R/G结果选grid，也不写正式S1。
- 2.5mm/50k三物体smoke PASS：max error bucket003/004/007=`1.554/1.386/1.230mm`，
  p99=`0.566/0.602/0.475mm`，但payload=`78.2/24.0/71.5MiB`。据此在任何R/G/Full
  结果前冻结mixed grid：bucket003/004/007=`5/2.5/5mm`，只对5mm存在12.73mm outlier
  的bucket004细化，兼顾epsilon与cache/memory。
- builder已加入object resolution map、max-error point provenance和PENDING安全resume；
  正式S1仍未运行。下一步静态/6项unit后，用mixed map先做50k临时exact回归，再以
  1M/object正式freeze。
- mixed-map改动经ruff、6/6 unit、py_compile、diff-check和formal-root-empty全部PASS。
  下一步通过不传全局voxel override的临时builder入口验证默认map/aggregate闭合；通过后
  才允许1M/object正式S1。
- 首次mixed-map smoke在bucket003 validation入口停止：安全resume把临时manifest标为
  `GRID_PENDING_VALIDATION`，而public loader正确地只接受`GRID_FROZEN`，builder尚无显式
  pending读取通道；正式S1仍ABSENT。修复为loader默认继续拒绝pending，仅builder传
  `allow_pending=True`，并新增runtime拒绝/builder允许的状态机contract后换新root复跑。
- pending状态修复后ruff、7/7 unit、py_compile、diff-check和formal-root-empty均PASS；
  runtime fail-closed合同保持。下一步以全新mixed临时root复跑，不消费失败半成品。
- mixed default-map 50k smoke现3/3 PASS并正确记录`5/2.5/5mm`；max error=
  `2.875/1.386/2.485mm`，sign outside2h=0，CUDA parity全部PASS，payload总计约43MiB。
  新增canonical S1 wrapper，下一步做bash/static/source SHA终检后以1M/object正式freeze；
  该阶段仅短暂使用本地GPU做parity，不启动CEM。
- S1 formal pre-freeze矩阵PASS：wrapper bash-n、ruff、format、diff-check、collider/keep22
  SHA exact，正式S1 root ABSENT。现在允许执行canonical wrapper，默认1M/object且mixed
  resolution；任何gate失败将冻结为GRID_REJECTED并停止后续P/R/G，而非启动Full。
- 正式S1 v1在bucket003的1M验证按stop gate终止：grid本身SHA与5mm smoke exact，但新增
  5个outside-2h sign disagreement；最坏同一点exact=`-129.995mm`、grid=`+130.019mm`，
  导致epsilon假性膨胀到260mm。bucket003被不可变标记`GRID_REJECTED`，bucket004/007未跑，
  R/G/Full未启动。该现象远超插值误差，优先诊断Open3D nsamples=5罕见sign歧义。
- 下一步在冻结failure point对比nsamples 5/11/101、occupancy和trimesh contains；禁止覆盖
  v1 rejected artifact，也不原配置重跑。若证实sign backend问题，新开versioned S1 v2。
- 最坏点复判确认它位于union AABB x-min外44mm，几何上必为outside；trimesh contains=False、
  Open3D nsamples11=outside，但nsamples1/3/5/7/21/51/101多数错误inside。故v1失败是
  boolean-union ray-parity sign歧义而非grid resolution。下一步重建同1M验证集提取全部5个
  mismatch，检查AABB外比例与trimesh/nsamples11一致性，为versioned v2 sign合同提供证据。
- 首次5点批量诊断被runtime fail-closed正确阻止：`GRID_REJECTED`不能由public loader读取，
  没有访问到query结果。正式artifact未变；修订为读取SHA相同且已PASS的mixed-smoke grid
  仅作离线诊断，不放宽production loader，也不增加allow-rejected入口。
- 使用同SHA PASS-smoke grid完成1M exact复判：5个outside2h mismatch全部位于union AABB外，
  `trimesh.contains`均False；Open3D nsamples5对其中4个符号错，nsamples11也仍错3个，说明
  增加ray数不是可靠修复。v2拟用确定性规则“AABB外unsigned distance必为正”，并先在
  AABB内200k点对Open3D sign与trimesh contains做cross-check，排除隐藏内部歧义。
- AABB内200k/object cross-check发现Open3D sign vs trimesh.contains mismatch=
  bucket003/004/007 `47,671/55,922/42,843`，其中>1cm deep mismatch=
  `30,893/38,056/31,470`；故不能只修AABB外5点，v1 exact-C sign整体失效。
- 新开不可覆盖的`S1_canonical_grid_sdf_v2`：Open3D只提供unsigned nearest-triangle
  magnitude，negative-inside sign统一由watertight manifold union的trimesh.contains给出；
  新增box known-sign contract。v1 `GRID_REJECTED`永久保留，正式R/G只能引用v2 PASS SHA。
- v2经ruff、8/8 unit（含known box sign）、py_compile、diff-check与v2-root-empty全部PASS。
  下一步先在全新临时root做mixed 50k smoke，测robust sign builder wall与error；通过后才
  以1M/object写正式v2。
- v2 50k smoke在bucket003被gate拒绝：trimesh.contains sign使grid-vs-exact出现`2,281`
  个outside2h disagreement，p99/max=`74.73/186.83mm`；说明boolean-union上的trimesh ray
  containment同样不稳定，不能成为authority。v2仅存在临时GRID_REJECTED，不写正式root。
- 两次失败后已排除Open3D ray parity和trimesh ray contains；下一步检查libigl/robust winding
  或实现chunked generalized winding/solid-angle，先用known box与三object跨backend审计，
  通过才开versioned v3，禁止通过放宽sign gate掩盖问题。
- 环境无libigl/pysdf/kaolin/point-cloud-utils，但已有E181 exact solid-angle实现；更直接的
  production解是利用`C=union(convex parts)`：每个part由SciPy ConvexHull half-space解析
  判定，union sign是逐part occupancy OR，完全绕开通用mesh ray sign。
- v3改为Open3D只算boolean-union surface unsigned magnitude，sign由frozen ordered parts
  half-space给出，并用part AABB预筛；新增single-box signed distance与overlapping-box union
  occupancy contracts。该发现也使E183的Open3D-sign P计数需要在R/G前重算审计，当前
  keep22虽已按用户决策冻结，但在robust-P复核前不得进入Full。
- v3经ruff、9/9 unit、py_compile、diff-check与v3-root-empty全部PASS。下一步新临时root
  mixed50k smoke；同时记录half-space bake wall，若通过再正式1M，不覆盖v1 rejected或v2
  临时evidence。
- v3 mixed50k smoke 3/3 PASS：sign disagreement outside2h均0，p99=
  `1.428/0.602/1.131mm`，max=`2.875/1.386/2.485mm`，CUDA parity PASS；解析sign下
  bucket003/004 grid SHA相对ray版本改变，证明修复实质生效。现允许正式v3 1M/object；
  wrapper默认root已version到v3，v1 rejected不覆盖。
- 正式S1 v3 1M/object已3/3 PASS：manifest SHA bucket003/004/007=
  `f3682e48...f6914 / c7c797db...1a5da / d4f87a03...f6094`，grid SHA=
  `988dee6d...afe0f / b8676022...a026 / 3e786fd8...ae24`；p99=
  `1.223/0.394/0.971mm`、epsilon=`2.875/1.386/2.494mm`、deep sign mismatch=0。
- S1只证明D_C相对解析C可靠；因E183 P候选contact也使用过错误Open3D ray sign，下一步
  必须保持E183 points/radii/oracle labels不变，仅用convex half-space candidate sign重算27行
  confusion，审计用户冻结的22/5是否仍成立；结果闭合前不得进入R/G/Full。
- robust-P auditor、2项source/confusion contracts与canonical wrapper首版已实现：锁定E183
  query aggregate/case-table、E186 selection/collider SHA，复用27 case两族points/radii/oracle
  labels，只替换candidate sign，输出旧/新逐case confusion和selection diff；不会自动改写
  keep22/drop5。首次静态检查只报import-sort/format，formal audit root仍ABSENT。
- import/format修复后ruff、2/2 robust-P contracts、bash-n、py_compile、diff-check和formal
  root-empty均PASS。现在执行27-case CPU-only正式reaudit；若robust keep不等于原22，status
  将冻结为AUTHORITY_INVALIDATED并停止，禁止自动重写authority或继续R/G/Full。
- 正式robust-P re-audit完成并`CONFIRMED`：解析convex-union sign下keep仍exact同一22，
  drop仍同一5，分object仍`5/4/13`，selection changed=`0`；case evidence SHA=
  `1eaaf50d...9985`。因此用户冻结authority无需修改，且已消除E183 ray-sign实现风险。
- 现在允许进入S2/S3 production实现：grid runtime必须引用正式v3 manifest SHA，compound
  physics必须引用同一ordered-part SHA；下一步审计config resolver与mjwp所有box-union调用，
  以default-off backend保持E178逐值兼容。
- 已创建阶段log254并更新Tracker Phase49：明确S0/S1/robust-P通过、C4–C10仍pending，
  不夸大为Full或轨迹质量改善。下一步重建log INDEX并做全量静态/正式artifact validation；
  然后继续production backend，而不是启动Full。
- log INDEX已重建并纳入E186 Phase49。最终checkpoint验证PASS：E186 ruff/format、全部
  wrappers bash-n、compileall、authority validator、grid 9/9、robust-P 2/2、正式artifact
  状态=`AUTHORITY_FROZEN / grid_v3 PASS / robust-P CONFIRMED`，diff-check无报错。
- 当前代码/文档尚未commit；正式results按外置策略不进git。Full GPU任务启动数仍为0，
  本地/远程已有进程均未kill/暂停/抢占。下一实现步骤是default-off MJWarp grid backend与
  22行compound-convex sidecar/pair parity，完成前继续禁止canary/Full。
## 2026-08-02 E186 下一阶段启动（production R/G + compound physics）

- 已按用户批准恢复 `204_E186_22case_object_specific_prg_full_plan.md`、tracker、log 254 与数据构建 release contracts。
- 当前冻结边界不变：keep22、三个 object-specific CoACD collider、grid-SDF v3 均不重选；Full CEM 仍被 physics/RG parity 与 recorder-off canary 阻断。
- 本阶段先实现默认关闭的 grid-SDF production dispatch 和 compound-convex scene sidecar；legacy box 默认必须逐值保持，禁止启动/终止任何现有 GPU 作业。
- 代码审计确认 production reward/gates 在 `mjwp.py` 的主 tick 内共享 `geom_box_sdf_min`，但 terminal carry gate 仍直接调用 box-union helper；新 dispatch 必须同时替换两处。
- `CanonicalGridSDF` 已具备 manifest/SHA fail-closed、CUDA tensor cache、trilinear query 和 conservative query；缺的是 robot geom 采样、world→object-local 变换、per-group batching以及 runtime config 接线。
- 计划合同再次确认：reward连续查询必须使用原始 `D_C`；CEM/terminal等 hard gate 必须使用 `D_C-epsilon_grid`。因此不能只把旧共享函数整体替换，dispatch需要显式区分 nominal 与 conservative 查询。
- 现有 E175/E176 回归脚本覆盖 legacy primary/union 和 batch exact；将保留这些 helper 与默认路径不动，并为 grid backend新增独立 production 单测，避免历史配置数值漂移。
- 已新增 `spider/simulators/mjwp_object_distance.py`：按sphere center/capsule三点/mesh确定性800顶点采样，统一转到object-body local frame，并支持一次per-geom查询聚合多group。
- `Config` 已增加默认关闭的 `legacy_box|grid_sdf` backend与manifest/asset-SHA/error-bound合同；`compound` geom解析仅在显式grid backend下允许。新代码静态编译通过，尚未接入所有consumer。
- production主reward、CEM body/hand/leg gates与terminal carry gate现已统一接入backend dispatch；grid nominal cache供reward复用，hard gate只在消费端减`epsilon_grid`，legacy helper本体未修改。
- 静态编译通过。首次lint命令误用了不存在的`.venv/bin/ruff`（1次）；不是代码失败，后续将先解析仓库实际ruff入口再运行，不重复该命令。
- 已确认仓库ruff入口为`/home/ubuntu/miniconda3/bin/ruff`。
- 新增E186 production backend自包含单测，覆盖object-body平移/旋转不变性、sphere/capsule/mesh采样、multi-group cache、nominal `D_C`与conservative `D_C-epsilon`分流，以及legacy union继续拒绝non-box。
- 首次直接用系统`python`跑production测试失败（1次）：系统Python 3.13没有MuJoCo；已确认E186标准入口应使用`UV_CACHE_DIR=/tmp/e186_uv_cache uv run python`，后续不重复裸Python测试。
- 全文件ruff暴露大量`config.py/mjwp.py`历史lint债；本次新增的真实问题是import顺序、一个`zip(strict=...)`与新方法docstring。将先format并只修新增/相关告警，不扩张为全文件历史清理。
- `ruff format --diff`确认直接格式化`config.py/mjwp.py`会机械改写大量历史行；为保护审查范围，不执行全文件rewrite。只格式化新文件，并对旧文件新增区段做局部格式/定向lint。
- 一次多文件`apply_patch`因hunk分隔符拼写错误未应用（1次），随即用正确patch完成；没有产生部分写入。
- 已修复新增代码的import顺序、`zip(strict=True)`和public method/test helper docstring告警。
- `uv run` production grid backend测试PASS；新模块/测试全量ruff PASS。
- legacy exact回归PASS：E175 multi-geom与E176 group batching均通过；既有grid基础测试9/9 PASS，py_compile与`git diff --check`通过。由此确认默认`legacy_box`路径未被新dispatch破坏。
- S2输入审计完成：keep22 manifest逐行提供E178 source scene及SHA；collider lock直接冻结32个ordered OBJ parts（16/8/8）及各part SHA，grid v3的ordered-parts SHA与lock一致。
- E175 builder已有可复用的XML stripped-signature、18×K pair matrix与compiled-contract思路；E186将改为mesh asset/geom并保留E178除物体collision与robot-object pairs外的结构，不覆盖source scene。
- 三类E178 source scene的object inertial/joint/visual结构可直接保留；旧collision geoms均是active box且带统一`friction=1 0.005 0.0001, condim=3`，robot pair合同为hand `solref=0.008 1/friction=2 1/condim=4`、lower-body `solref=0.008 1/margin=gap=0/condim=1`。
- S2 builder将删除旧active box geoms、注入K个冻结OBJ mesh geoms并重建18×K robot pairs；object-floor显式pair也将对每个part重建，以避免仅part0继承E178 floor solver的歧义。mass/inertia/joints/visual/actuator等用stripped signature和compiled arrays双重校验。
- E178 full manifest提供每个keep case的base override/config_act，可据此生成仅改变`scene_name/object_collision_sdf_mode/backend/manifest/SHA/epsilon/batching`的E186 override并做Hydra parity。
- MJWarp compile gate采用已验证的`mujoco_warp.put_model(MjModel)`；这是模型转换检查，不启动CEM，也不需要终止/占用现有训练进程。
- 已实现`build_compound_physics.py`及两个正式wrapper：生成22个portable sidecar/task assets、E186 overrides、scene/asset manifests与CPU/MJWarp compile evidence；不覆盖E178 source XML。
- builder首轮静态检查仅剩import排序1项，py_compile、wrapper bash语法与`git diff --check`已通过；下一步修正排序后执行22-case正式S2构建。
- builder import排序已修复，ruff全量PASS；确认`mujoco_warp.Model`暴露`npair`合同字段。
- 单case smoke `bucket003_20231018_003_p1`通过：K=16、robot-object pairs=288、CPU/MJWarp nmesh=52一致，compile约0.71s，16个part asset SHA闭合。可以进入22-case正式构建。
- 正式S2 compound physics完成并PASS：22/22 CPU MuJoCo compile、22/22 MJWarp `put_model`；object分布5/4/13，K=16/8/8，robot pair精确为288/144/144。
- 正式aggregate：scene manifest SHA `3e7e395f679ad8efec117ad8ac5728efca6bcc2b32fd0378fa14ac645fe66512`，asset manifest SHA `53f38266c81a22227418808c6dac11c872ae88323329ce531df5d2537ce2f308`；Full仍未启动。
- 启动S3前发现可移植性缺陷：`results/E186`的真实路径经`Path.resolve()`落到外置挂载，导致生成override把grid manifest写成`/mnt/...`绝对路径。S2编译本机有效，但远程immutable snapshot不满足合同。
- 修复方向：路径序列化必须按lexical repository path而非解析symlink后的物理路径；修复builder后重生成22个override/manifest并更新S2 aggregate SHA，再开始R/G audit。
- 已修复symlink path序列化并重跑S2：22/22 CPU/MJWarp仍PASS，22个override与scene manifest的grid路径均为repository-relative，portable path audit PASS。
- 更新后的正式S2 SHA：scene manifest `c7a2a3d4a2e1f76cbc395aca73330574ace79541ae79c0545b3b839f6e207107`，asset manifest `6977b61080af0b017af8c929f28f51783b6a35af9c4a1627eddbec0c14d23952`；前一组SHA已被本次合法路径bug修复取代。
- S3合同恢复：reference/E178-final后需exact-C vs grid reward/gate审计，hard gate false-safe-accept=0、selected-valid重算100%、reward rank Spearman≥0.999；之后才是每object一条64×4 shadow/canary。
- 发现runtime对象序列化风险待修：当前grid runtime作为Config dataclass字段可能被`config_act.yaml`保存逻辑读取；在真实CEM前必须确保只保存可序列化metadata，不把Grid对象写入effective config。
- 已确认`run_mjwp.py`和fast版都会遍历所有dataclass fields写config YAML，当前skip集合未包含grid runtime；风险真实存在，需在两入口显式skip或移出dataclass后才能canary。
- 已把runtime Grid对象移出dataclass，改为process阶段私有动态属性；config YAML只保留manifest/SHA/epsilon等可序列化字段。production backend测试仍PASS。
- 真实E186 Hydra override→Config→`process_config`通过：16个compound geom、grid manifest/SHA/epsilon加载、R/G geom groups解析全部闭合。定向ruff命令仍打印旧文件74项历史E501/B/F债，但新模块测试与真实配置验证通过，未做无关全文件重排。
- reference轨迹为126×43 freejoint，E178-final outdir为126×2×42 scene-act qpos；可复用E176的freejoint→scene-act转换，final按既有`npz_qpos`口径取world0。
- exact-C authority可直接复用grid builder的Open3D unsigned union surface + convex-halfspace sign函数；S3将对production相同的sphere/capsule/mesh查询点同时计算exact与grid，而不是拿E178旧box recorder值冒充exact-C。
- E178 active object-distance reward轴确认：robot penalty、leg penalty、surface-band reward；hard gate轴为body/hand/leg，阈值分别`-5mm/-10mm/+5mm`。其他object-distance组件scale=0。
- 已把sphere/capsule/mesh生产采样抽成共享`sample_robot_geoms`，S3 exact审计和production runtime将消费同一批点，消除重复实现导致的query口径漂移；待复跑回归。
- 共享采样重构后production backend与legacy batch回归均PASS；仅补充public transform docstring，无数值退化。
- 已实现S3 reference/E178-final审计与wrapper：44条真实pose tape、active reward三组件、body/hand/leg conservative gates、paired selection/fallback、CPU query吞吐和5-step短rollout。静态检查仅剩audit脚本import排序2项；production回归仍PASS。
- audit import排序已修复、ruff PASS。首个真实tape smoke失败1次：MuJoCo CPU `geom_xmat/xmat`暴露扁平9列而脚本目标为3×3；已显式reshape修复，不重复旧调用。尚未生成正式S3结果。
- 2026-08-02继续获批进入下一阶段：已重新读取E186 tracker/plan/log与两项skill合同；冻结keep22、三object collider、grid v3和S2 portable SHA均保持不变，Full启动数仍为0。
- 当前先对首个真实reference tape拆分inside-grid/outside-grid、hard-gate与reward-active support误差；目标是区分保守AABB下界造成的无关全域raw误差和真正会改变R/G行为的production偏差。该诊断仅CPU执行，不触碰现有GPU任务。
- 代码核对确认outside规则是“query点到object AABB的非负下界”：它能保证G不产生false-safe accept，但不保证R的连续距离忠实。首case已见exact约11–21cm而grid约0.6–7cm的样本，若跨入robot/leg reward hinge会产生phantom penalty，属于真实R偏差而非仅统计口径问题。
- S3正式门应拆为：G继续检查`D_C-epsilon` false-safe=0；R按真实active support检查组件行为与解析误差；全域raw误差单独诊断。下一步先读取E178实际reward阈值并在同一production采样点输出inside/outside来源与phantom reward activation计数，再决定修outside runtime还是仅修审计聚合。
- 已核对S3计划原文：reward连续使用名义`D_C`且p99需满足epsilon传播上界；outside必须保守但未声明它可牺牲reward fidelity。因此不能把outside大误差简单豁免，必须检查是否落入active reward support。
- 首case authority闭合：bucket003 K=16、epsilon=`2.875454mm`、E186 scene/manifest均为冻结portable路径；E178对应final/config路径已定位。上一次空awk是把case_id误当第1列（实际E186第3列），未产生写入；后续按manifest header解析，不重复错误列假设。
- E178真实active阈值已核对：body/leg penalty margin均`20mm`，surface band=`[-1,3]mm`；hard gate为body `-5mm`、hand `-10mm`、leg `+5mm`。三object场景机器人geom规格一致。
- 根因合同已明确：当前grid最小padding仅约`50.0–50.5mm`，但body采样sphere/capsule最大radius=`90mm`、leg最大=`60mm`；production先查轴线/中心点再减radius。故潜在reward-active body轴线点需要至少`90+20+epsilon≈113mm`覆盖，当前outside AABB lower bound减radius会制造phantom负距离/penalty。
- 这不是CoACD C失效，而是冻结v3 `D_C`查询域与robot-geom Minkowski半径不匹配。先在首tape固化inside/outside与phantom activation证据；若计数非零，按计划走versioned grid fidelity修订（不改C/keep22），不得直接放宽R门。
- v3 manifest三物体最小实际padding约`50mm`；body最大radius=`90mm`来自torso capsule，leg最大=`60mm`，手为mesh vertex query不减radius。由此body outside-query风险是结构性而非单case偶然。
- 现有S1 validator只在grid bounds内部验证插值，因此v3的1M PASS并未覆盖“机器人采样点减radius后的reward support完整性”。将补充public in-bounds诊断与Minkowski-support coverage contract；这属于S1 validation缺口，正式修订必须新目录/新manifest SHA，不能覆盖v3。
- 已给`CanonicalGridSDF`加入与query完全同域的`in_bounds_mask`及边界单测；query本身复用该mask，避免诊断/production边界定义漂移。
- S3 auditor已扩展首case可重复diagnostic scope：逐sample拆inside/outside exact误差，记录body/hand/leg的grid-min winner是否来自outside，并统计三类reward的support、phantom-active、missed-active及support误差；同时校验诊断重算grid与production `per_geom_sdf`一致。
- 新diagnostic代码经ruff/format、py_compile、diff-check和grid contracts `10/10 PASS`；新增in-bounds mask的inclusive origin/max边界测试通过，既有outside AABB与pending fail-closed合同未退化。
- canonical S3 wrapper现透传参数，可用同一固化入口跑单case diagnostic或无参正式44-tape；下一步执行bucket003首case的reference+E178-final，不把其FAIL冒充formal结果。
- 首casev3真实diagnostic完成（reference+E178-final，均finite）：inside-grid sample误差p99仅`0.336/0.441mm`、max`1.743/1.584mm`，outside p99=`113.4/75.3mm`、max约`140mm`，证实大误差全部由outside rule主导。
- R行为偏差非零：reference leg phantom-active `17/18` support、p99 error `0.0327`；final leg phantom-active `26/39`、p99 `0.0469`；reference body phantom-active `3/3`。因此不能用“outside但reward均为0”豁免，v3正式S3必须FAIL。
- G仍安全：两tape body/hand/leg/combined false-safe均0，selected-valid exact重算`126/126`；但selected index仅`107/126=84.9%`，说明R phantom penalty已实质改变reference/final候选排序。下一步进入versioned padding修订，不改C/keep22。
- 已在active plan追加S1b v4偏差修订：冻结C/keep22/ordered parts/voxel map不变，只把grid requested padding从50mm增至120mm；理论硬门为`max query radius 90mm + active reward support 20mm = 110mm`，保留10mm工程余量。
- builder默认输出改为不可覆盖的`s1_canonical_grid_sdf_v4`，新增六方向minimum actual padding与`minkowski_reward_support_covered`硬门；canonical wrapper透传参数。v3正式artifact保留为失败诊断证据，不删除/覆盖，S2当前仍指v3直至v4 formal与首case回归通过。
- v4静态/preflight PASS：ruff/format、bash-n、py_compile、diff-check、grid contracts `11/11`；正式v4 root与smoke root运行前均为空。
- v4 120mm/50k三物体smoke `3/3 PASS`：minimum actual padding=`120.549/120.524/120.006mm`，均超过110mm hard floor；p99 error=`1.199/0.528/0.949mm`、max=`2.875/1.386/2.478mm`，deep sign mismatch=0、CUDA parity PASS。
- v4 payload为bucket003/004/007约`18.4/56.1/16.9MB`（合计约91.4MB）；bucket004因2.5mm resolution增幅最大。先用smoke grid重跑首caseR/G，确认phantom activation归零并测CPU query时间，再决定是否承担正式1M与GPU cache成本。
- S3 auditor新增diagnostic-only `--grid-root`，正式模式仍强制消费S2 manifest；override grid的epsilon从目标manifest自身读取并继续校验collider asset SHA，避免用v3 error bound误审v4。
- 首case使用v4-smoke grid回归：case级`PASS`，两tape false-safe=0、selected-valid=100%、minimum affected Spearman≈`1.0`；说明120mm padding直接消除了v3的R排序退化。aggregate仍显示FAIL仅因formal聚合硬编码要求22-case，这是预期的bounded diagnostic状态，不代表case失败。
- v3→v4首case对比：body/leg phantom-active从reference `3/17`、final `0/26`全部降为0；Spearman从`0.233/0.614`升为`1.0/≈1.0`。CPU grid query单次约从`12.0/9.0ms`到`15.3/12.0ms`（约+28–33%，仅首轮CPU诊断，GPU效率待canary）。
- 正式S1 v4 1M/object现`3/3 PASS`：p99 bucket003/004/007=`0.907/0.178/0.666mm`，epsilon=`2.875/1.386/2.537mm`，minimum padding均≥120mm、deep sign mismatch=0、CUDA parity PASS。
- 正式v4 manifest SHA：bucket003=`8043ade8...b20a6b8`、bucket004=`f9424291...16eb5e`、bucket007=`06aa4f10...de3b10`；grid SHA与50k smoke一致。下一步以formal v4复跑首case，然后重建S2 portable overrides/manifests指向v4并重做22/22 CPU/MJWarp compile。
- formal v4首case复跑仍case级PASS：两tape false-safe=0、selected-valid=100%、rho≈1；case metrics SHA与smoke grid完全相同，tape SHA只因formal epsilon/性能字段变化。由此允许晋级S2 path rebind。
- compound builder的GRID_ROOT已显式version到`s1_canonical_grid_sdf_v4`；尚未重跑S2。下一步先静态与source/authority closure，再完整重建22个override/manifests并复验CPU/MJWarp，旧v3 S2 SHA将被新合法v4路径SHA取代并记录。
- S2 v4 rebind preflight PASS：builder ruff/py_compile、wrapper bash-n、diff-check；三formal grid均`GRID_FROZEN`、margin=120mm、Minkowski coverage=true且candidate asset SHA仍匹配冻结C。
- 首次无参S2重跑被builder fail-closed阻止（1次）：22个v3 sidecar已存在，需显式`--overwrite`。未产生部分写入。该builder flag正是合法版本path rebind入口；下一步用`--overwrite`重建同名E186 sidecar/override并记录新SHA，不删除source E178或任何运行任务。
- S2 v4 path rebind完整重跑PASS：22/22 CPU MuJoCo、22/22 MJWarp compile；bucket003 K16/pairs288，bucket004/007 K8/pairs144，compound physics与冻结C本身未变。
- 新v4 S2 scene manifest SHA=`93b50eb421a69f0f9b9a15bd88c504fbf6eb7b939bfdd1579befc804cc8d33a1`；asset manifest SHA仍=`6977b610...d23952`，证明只有grid manifest/epsilon路径合同改变，portable physics assets未变。旧v3 scene SHA `c7a2a3d4...7107`作废但留在progress证据。
- 下一步做22行override/manifest portable v4 closure与真实Hydra config load，然后执行44-tape正式S3；在正式聚合PASS前仍不启动64x4 canary/Full。
- S2 v4 closure复核PASS：manifest 22行全部指v4、22个E186 override中v3引用=0/v4引用=22，scene/asset SHA与aggregate一致；production grid backend回归PASS，diff-check无报错。
- 首次给S3追加reward解析误差bound的多hunk patch因ruff格式后的上下文不匹配而整体未应用（1次，无部分写入）。下一步按当前行号拆成小hunk：body/leg hinge bound=`scale*epsilon`，surface hard-band bound包含边界jump与band内Lipschitz项，并把p99 bound纳入tape status。
- reward解析bound已按当前格式小hunk成功接入：body/leg=`scale*epsilon`；surface考虑hard-band边界jump加band内`scale*epsilon/sigma`并cap到component scale；affected-total取分量bound之和。
- 每条tape现持久化各component analytic bound与p99-within-bound，`reward_component_bounds_pass`已纳入status。下一步先ruff/unit与formal-v4首case回归，确认新增硬门不过拟合/不破坏既有case，再启动44-tape正式CPU审计。
- bound增强后formal-v4首case仍case级PASS；两tape body/leg/surface/affected p99 bound全部true、phantom-active全部0、rho≈1，ruff/py_compile/diff-check PASS。
- 当前surface hard-band因阈值不连续，其全局解析bound被cap为scale=`1.5`，本身较松；因此不会单独依赖该门，仍同时要求affected rank rho≥0.999，并把真正Full total reward rank留到64x4 shadow。现在允许启动22case×2 tape正式CPU S3，不使用GPU CEM。
- 正式44-tape S3已完成但Gate FAIL：finite=`44/44`、false-safe总计0、selected-valid exact重算=100%、short rollout=`22/22`；case status `12 PASS / 10 FAIL`，minimum affected-only Spearman=`0.696933`（门0.999）。因此canary/Full仍禁止启动。
- 逐caseFAIL均由当前affected-only稀疏rank门触发，控制台false-safe全0；下一步必须量化body/leg phantom是否全归零、surface band active/membership误差与paired selected-index变化，判断是5mm grid对1.5mm sigma过粗还是“稀疏affected-only序列不等于Full total reward rank”的审计定义错误。禁止直接放宽rho或启动Full。
- 44-tape分解：body phantom/missed=`0/0`，leg=`0/1`；surface support总368帧、phantom/missed=`14/4`。10条rho失败均由surface hard-band少数边界flip主导，inside-grid sample p99仍仅`0.024–0.656mm`；所有component解析p99 bound均PASS。
- paired reference/final index并非Full candidate选择：10个case存在少量index mismatch，但selected candidate exact-valid仍100%；最差`bucket003_...001_p2` match=`88.56%`，其surface boundary flip也最多（phantom7/missed1）。
- E182已有64×4 recorder与shadow evaluator语义可借鉴，E178 speed-probe/旧canary runner可复用CEM启动合同。下一步审计query tape能否直接提供完整raw total reward与E186 exact-C offline replacement；若可，构建三object各一条recorder-on shadow作为S3证据，正式S4效率仍另用recorder-off。
- E182 recorder确认每次CEM optimize只记录final iteration payload，至少含candidate horizon `qpos`与sample total `rewards`，并且observational-only；其既有shadow定义是`raw_rewards + geometry_delta`后比较Spearman/elite集合。
- E186正确shadow公式应为`exact_total = raw_grid_total - grid_object_components + exact_C_object_components`，不能把affected-only tape序列冒充Full total。G则用同一qpos重算exact-C与grid conservative body/hand/leg，并合并recorder已有posture gate。
- 仍需确认payload是否已包含sample posture/gate字段、qpos/reward具体shape，以及一条轨迹产生多少chunks；据此决定只审首个final-iteration chunk还是全control-step流。计划目标是bounded 64×4，不允许无界recorder拖慢canary。
- 真实E182 tape schema确认：每chunk `qpos=(64,48,42)`、`rewards=(64,)`、elite indices及body/hand/leg/posture/combined sample gate字段齐全，可直接支持E186 total-reward replacement与optimizer selection重算。
- 完整轨迹会产生`77–202` chunks，不符合“一个bounded 64×4 shadow query”。因此E186将给observational recorder增加默认0=unlimited、显式1=只保存首chunk的cap；后续optimize仍照常执行，不改变physics/reward/selection，只避免无界artifact和exact offline成本。
- 现有query-tape单测覆盖default-off、连续chunk、deterministic no-op；将新增cap=1的skip合同，并确保finalizer仍冻结1-chunk COMPLETE。E186 shadow只消费chunk0，正式S4 recorder-off效率probe仍不启用任何tape。
- 已实现`query_tape_max_chunks`（默认0=历史unlimited，E186显式1）：超过cap后返回`SKIPPED_MAX_CHUNKS`且不写文件/不改优化；负值fail-closed。Config默认行为不变。
- recorder contracts现`4/4 PASS`：default-off、连续chunk、deterministic rollout no-op、cap=1且finalizer冻结单chunk；query_tape新文件ruff、py_compile与diff-check通过。
- 下一步实现E186三代表case shadow runner/evaluator：runner固定64×4/seed0/query_tape_max_chunks=1，evaluator在chunk0用production同采样重算grid/exact object components与geometry gates，再用`raw_grid_total - grid_component + exact_component`比较total rank与elite选择。
- `run_mjwp.py`支持显式`max_sim_steps`；标准MPC每个control tick先完成一次CEM optimize，再commit `ctrl_steps`并检查上限。E178配置通常sim/ctrl步为2，因此E186 shadow可用`max_sim_steps=2`只执行一个control tick/一个chunk，而无需跑77–202个完整轨迹chunks。
- bounded shadow command将同时固定`num_samples=64,max_num_iterations=4,seed=0,max_sim_steps=2,query_tape_max_chunks=1,save_video=false,use_torch_compile=false`；它是单query功能审计，不作为recorder-off效率数值。下一步先核对三代表override的sim_dt/ctrl_dt与budget，再固化runner。
- 三代表case已按计划冻结：bucket003=`20231018_003_p1`（E182代表被drop后取keep物理行首），bucket004=`20231002_021_p1`、bucket007=`20231020_055_p1`（复用E182代表）；三者E178均`sim_dt≈1/60, ctrl_dt≈1/30`，故`max_sim_steps=2`精确是一control tick。
- 已新增resume-safe `run_shadow64x4.py`与local launch入口：逐case固定64×4/seed0/2 steps/1 chunk，运行后finalize tape并核对effective config、grid backend、artifact SHA；已完成case manifest则仅做SHA复验，半成品fail-closed不追加。
- runner尚未执行GPU。下一步补command/representative静态单测与ruff/bash/py_compile；通过后先在本机GPU0跑bucket003单query smoke，期间不kill/暂停已有进程。
- shadow runner静态contracts `2/2 PASS`：代表三object/physics状态闭合，command精确包含64×4、seed0、2 sim steps、tape cap1、recorder-on、compile-off与GPU ID；ruff/format、py_compile、bash-n、diff-check PASS。
- 现在允许本机GPU0执行bucket003单query功能smoke。该进程将与现有程序叠加，不做GPU idle wait、不kill/暂停/抢占；若OOM或runtime失败先记录并诊断，不改budget重复盲跑。
- 本机启动前GPU0为RTX5090、显存151/32607MiB、util0且无compute process；未执行任何kill/idle wait。
- bucket003首次shadow GPU run的CEM/output已返回，但finalizer报`tape chunk_manifest`不存在（1次）；这不是OOM。runner因output目录与tape父目录已存在将拒绝原ID重复，当前保留半成品fail-closed。
- 最可能是Hydra override token与load-config路径下`query_tape_enabled/max_chunks`未实际进入optimizer，需先读case log与effective config确认；修复后使用新versioned run root，不删除/续写失败root、不原样重复。
- 根因已由log/effective config确认：query_tape overrides全部正确生效，但E178 `warmup_steps=0.2s`，`max_sim_steps=2`在首次CEM前结束，输出显示`opt_steps=0`，故没有chunk；非Hydra/config bug。
- 三代表均`sim_dt≈1/60, ctrl_dt≈1/30,warmup=0.2`。control index 0–5 warmup，sim_step=12时首次真实CEM，commit后到14；因此versioned v2 shadow应固定`max_sim_steps=14`，仍只记录首次actual CEM chunk。
- 失败v1 root保留半成品证据且runner拒绝续写。下一步把正式shadow root/version改为`shadow64x4_v2`与14 steps，更新静态contract后重新跑bucket003；这与第一次配置不同，遵守不重复失败协议。
- shadow v2现固定root=`shadow64x4_v2`、max_sim_steps=14、provenance=`FIRST_POST_WARMUP_CEM`；v1失败root未覆盖。schema version同步为run_v2。
- v2 runner静态contracts仍`2/2 PASS`，ruff/py_compile/diff-check通过，正式v2 root运行前为空。现在允许本机GPU0重跑bucket003；预期前6个control ticks opt_steps0，第7个执行4-step CEM并只写chunk0。
- bucket003 shadow v2真实GPU run PASS：前6 ticks opt_steps0，第7 tick plan=`3.0567s`/opt_steps4；总控制loop约3.09s、进程wall8.40s（含MJWarp setup/compile cache）。tape COMPLETE且exact 1 chunk。
- chunk schema闭合：qpos=`64x48x42`、rewards=`64`、selected elite6，并含body/hand/leg/posture/combined gate全字段；reward范围`5.120413–5.120792`，chunk content SHA=`50e08b8c...fe0f`。
- 下一步实现/运行offline exact-C total-rank evaluator；bucket003先过shadow门后才并行跑bucket004/007，避免在evaluator合同未验证前消耗另外两条GPU query。
- offline公式复核production：body/leg是20mm hinge；surface是`[-1,3]mm` hard band + symmetric exp，并乘`surface_band_gate`与episode tail decay。shadow max_sim_steps=14会使15% tail decay只作用末端time，需要按candidate horizon各步真实time处理，不能复用reference审计按整tapeframe的简化factor。
- recorder raw rewards已经是E186 grid total，evaluator只需精确重算三active object components；其他tracking/posture等保持raw不变。仍需复建horizon的approach/contact gate（ref slice从sim_step13开始），否则surface delta错误。下一步读取run_mjwp approach-mask生成合同并复用其源mask，而非猜全1。
- 更稳健的合同已发现：rollout内部本来就有每step `robot_object_penalty/leg_object_penalty/surface_band_rew/surface_band_gate/surface_band_decay_factor`，但现recorder只保存mean total与`sample_*`，丢掉了component trace。
- 与其在offline重复run_mjwp的contact-mask resize/ramp/time逻辑，下一versioned shadow将observational-only记录上述5个`N×H` trace；evaluator用recorded grid component做raw total扣除，并用recorded gate/decay作用exact-C surface score，同时复算grid trace作一致性校验。
- 这需要新shadow v3 root重跑bucket003（v2保留为schema诊断），但不会改变reward/selection/physics；新增trace约`5×64×48` float，体积很小。下一步先改recorder payload与deterministic no-op测试，未经测试不重跑GPU。
- recorder v3已加入5个`N×H` reward traces：body penalty、leg penalty、surface reward、surface gate、surface decay；generic info聚合显式跳过所有`_query_tape_*`内部字段，避免污染普通结果。
- instrumentation回归PASS：query-tape `4/4`（含deterministic no-op与trace shape）、shadow `2/2`、ruff/py_compile/diff-check；正式`shadow64x4_v3` root为空，v1/v2均保留。
- 现在允许bucket003 v3同一64×4 seed0物理query重跑，仅artifact schema增加trace；通过后先完成offline evaluator并验证recorded grid trace可由qpos/grid逐值重建，再跑另外两object。
- bucket003 v3 GPU run PASS且5个trace shape均`64x48`；相对v2 selected indices/gate exact相同，qpos max差`1.2e-6`、reward max差`2.4e-6`，instrumentation近逐值no-op。
- 但trace暴露新口径bug：用`max_sim_steps=14`会把surface tail-decay的episode total time也缩到0.233s，首query horizon几乎全在decay尾后，surface component被压为0；该shadow不代表E178/Full reward，v3不得作为total-rank authority。
- 正确方案是保留各case原production `max_sim_steps`，新增observational `query_tape_stop_after_chunks=1`让outer MPC在首个post-warmup chunk记录后退出；这样query内部reward/time完全production一致，只缩短后续无关执行。需新v4 root，v1–v3全部保留作bug证据。
- 已实现default0的`query_tape_stop_after_chunks`与public persisted chunk-count helper；`run_mjwp`只在一个control tick完成并记录chunk后退出outer loop，query内部仍看到原production max_sim_steps。负stop值fail-closed。
- shadow v4 command移除`max_sim_steps` override，改为max_chunks=1+stop_after=1；effective config validator要求production max_sim_steps>14。v4正式root为空，schema更新run_v4。
- query-tape `4/4`、shadow `2/2`、ruff/py_compile/diff-check均PASS。现在允许bucket003 v4 GPU重跑；必须验证surface decay不再接近0、trace非平凡后才写evaluator。
- bucket003 v4 GPU run PASS：effective max_sim_steps恢复production 252，stop_after=1；首post-warmup query plan≈3.010s、总loop≈3.045s、process wall8.24s，单chunk COMPLETE。
- surface gate/decay trace现全1（已修正v3 tail-decay污染），但grid三object components在该首query均为0；raw total范围`5.120413–5.120793`。这可能是早期query尚未进入接触，而不一定说明exact也为0。
- 下一步先实现offline exact-C evaluator并检查exact component是否非零、total rank与selection；若exact/grid都全零，则该query只能证明G/运行合同，R shadow需新增预注册contact-timed record start，不能用rho=1的全零几何冒充有效R证据。
- 已实现`evaluate_shadow64x4.py`与canonical wrapper：从64x48 qpos用production同geom采样重算grid/exact-C；recorded surface gate/decay用于exact component；total correction=`raw_grid-grid_geom+exact_geom`。
- evaluator同时重算body/hand/leg horizon gate、合并recorded posture、镜像valid/fallback elite选择；硬门包括grid trace reproduction≤2e-5、geometry active>0、total rho≥0.999、grid gate/selected复现、false-safe=0、exact-selected-valid=100%。
- evaluator现经ruff/format、py_compile、bash-n、diff-check PASS，尚未运行。下一步先评bucket003；预期若geometry active=0会明确FAIL并触发contact-timed recorder，而非得到虚假rho PASS。
- bucket003 v4 offline evaluator按预期FAIL：grid trace/gate/selected复现全PASS、false-safe=0、finite；但geometry active candidates=`0/64`、grid/exact combined-valid均0，rho=1只是全零几何，明确不得放行。
- 已确认E178 Full trajectory NPZ保留每control tick×iteration的surface reward、geometry gate与opt_steps，可在不看E186 shadow结果选参的前提下，用历史E178固定证据预注册每代表case的首个“surface-active且combined-valid>0”control tick。
- 下一步从E178各tick的实际最后iteration提取该首合法tick，换算`record_start_sim_step=2*tick`；新增recorder start门只跳过早期artifact、不改变任何CEM query，v5在目标tick录1 chunk后退出。
- E178 frozen source口径选定（禁止再看E186调时点）：取actual final iteration中首个`surface_band_rew_mean≥0.3 && cem_gate_valid_frac>0` tick。bucket003/004/007分别为control tick `8/6/11`，record sim step=`16/12/22`，E178 surface=`0.342/0.525/0.302`、valid均1.0。
- 对应E178 trajectory SHA=`cfc1c1c3...69030 / 1f9c2f3f...fa01b / d72af0f3...de84`。这些时点由历史E178固定证据选择，不依赖E186 grid/exact/Full结果，避免为过门追选query。
- 下一步实现default0的record-start门并将上述object-specific steps固化进runner v5；早期CEM照常执行但不落tape，目标query录1 chunk后stop。v4早期inactive结果保留。
- 已实现default0的`query_tape_record_start_sim_step`：run_mjwp每control tick只写私有current sim step，recorder在目标前返回`SKIPPED_BEFORE_START`且不创建目录；到目标后录chunk0，stop-after结束。
- runner v5固化record steps `16/12/22`及三条E178 timing-source SHA；evaluator切到`shadow64x4_v5`。v1–v4历史root均不覆盖。
- query-tape contracts增至`5/5 PASS`（含start gate无早期artifact），shadow`2/2`，ruff/py_compile/diff-check PASS，v5 root为空。下一步本机先跑bucket003 v5；预计执行tick6–8共3次64x4 CEM，wall约14s，不启动另两case直到eval通过。
- provenance命名已更正为`E178_FROZEN_CONTACT_TIMED_CEM`，避免把tick8误写成first-post-warmup。
- bucket003 v5 GPU run PASS：record_start sim16、单chunk COMPLETE、process wall=`13.76s`，artifact/config/log SHA闭合；未启动另外两object。
- 下一步立即运行offline evaluator，重点看geometry active、grid trace reproduction、total rho、combined-valid与false-safe；若失败先按具体轴诊断，不改变冻结timing step16追选query。
- bucket003 v5 evaluator仍FAIL且与v4一致：geometry active=0、total rho=1（trivial）、grid/exact valid=0、false-safe=0；grid trace/gate/selected均精确复现，排除evaluator/runtime错误。
- 轴分解发现body/hand/leg geometry gates各`64/64 valid`，min SDF分别至少`66.5/46.6/89.1mm`；combined 0完全来自posture gate=`0/64`，不是grid/CoACD碰撞gate误拒绝。
- 这说明在E178冻结contact-timed tick8，E186 compound-physics运行已进入posture fallback且手仍离C表面>46mm，R无support。按计划应记录为optimizer/trajectory feasibility问题，不归咎grid fidelity。下一步仍跑bucket004/007冻结时点以判定是bucket003特有还是三object共性；Full继续禁止。
- bucket004 v5 GPU run PASS（record sim12，wall8.14s）；offline结果有真实R/G support：geometry active=`64/64`、grid/exact combined valid=`64/64`、false-safe/reject=0、selected0及topk exact一致、total rho=`0.999908`。
- 但bucket004 evaluator status FAIL有两点：recorded surface trace reproduction max error异常大`0.672`，以及rho略低于0.999预注册门? 实际`0.999908≥0.999`，所以唯一硬失败是trace reproduction。
- exact-vs-grid geometry delta p99仅`3.96e-5`且selection完全一致，强烈提示offline surface公式漏了bucket004特定模式（如bimanual reduce）而非grid错误。下一步先核对effective surface config与production sampling/formula，修evaluator合同后重评同一冻结chunk，不重跑GPU。
- 2026-08-02：恢复 E186 执行上下文并完整复读 `experiment-planning-zh`、`data-construction-v3-zh`。继续遵守 keep22/三个 object-specific collider/E178 冻结 record 时点不可重选，Full 仍为 0；当前只诊断 bucket004 v5 shadow 的 recorded surface trace reproduction max error=0.67178，不重跑 GPU、不改 P/R/G 门限。
- bucket004 v5 artifact 实际位于 `shadow64x4_v5/{raw_chunks,runs,eval,...}/<case>`（不是 case-first 目录）。evaluator 当前把 `cem_hand_gate_geom_names` 同时用于 surface reward，而 production 明确用独立的 `surface_band_geom_ids`；下一步核对该 case 两组 resolved geom 是否不同，并输出最大 mismatch 的 sample/frame/recorded/offline/SDF/gate/decay。
- 核对 bucket004 effective config：`surface_band_geom_ids=[49,60]` 与 `cem_hand_gate_geom_ids=[49,60]` 完全一致，故“用了错误 hand group”假设排除。body/leg trace逐值复现误差均0，只有surface最大0.6717808；recorded surface范围0–0.6717887，gate含0/1、decay全1。recorder保存的是final CEM iteration的逐步 `info_combined` 与同一步 `qpos`；下一步检查 sharp `[-1,3]mm` band 下 reward 与 qpos 的时序对齐/terminal-frame语义，并打印mismatch位置。
- terminal reward 会缩放总 reward，但其 `info.surface_band_rew` 仍来自同一次 `get_reward`，所以不是terminal component重复/缩放遗漏；rollout顺序为step→reward/info→qpos，理论上状态一致。首次用系统 `python` 做mismatch诊断失败1次（环境无`mujoco`，未写artifact）；下一步按现有eval wrapper的项目解释器重跑，不能重复同一错误命令。
- 用项目 `uv run` 完成mismatch定位：同帧support一致率91.67%、max error=0.6717808；将offline qpos trace前移一帧后，`offline[t]` 对 `recorded[t+1]` 的support一致率=100%、MAE=`9.80e-7`、max=`5.92e-5`。最大错位集中frame33，offline SDF约1.205–1.245mm且recorded为0，下一recorded frame才出现对应reward。确认是MJWarp step后qpos已积分到新状态，而派生geom_xpos仍对应step前状态（MuJoCo step语义），不是grid/CoACD/公式错误。
- v5现有qpos无法严格恢复reward frame0的geometry state，简单shift会丢一帧，不能作为exact-C全horizon authority。正确修复应保持现有`qpos`向后兼容，新增observational-only的reward/object-distance-aligned pre-step qpos trace，并在新version root重跑；禁止用截断47帧偷偷放行v5。下一步审计query-tape测试并实现该新增字段，不改变优化、碰撞体、reward或record时点。
- 现有query tape writer对payload是generic array持久化，不需改writer schema；改动点仅在sampling rollout采集与payload。E186 runner static test当前只校验64×4/timing/cap，通用recorder合同在`workspace/core4d/scripts/experiments/E182/test_query_tape.py`。计划新增`geometry_qpos`（pre-step）而保留现有`qpos`（post-step）以免破坏E182/历史消费者，并让E186 v6 evaluator强制读取新字段、拒绝旧artifact。
- 已实现observational-only dual-qpos schema：历史`qpos`继续记录post-step状态，新`geometry_qpos`在每个MJWarp step前采集并随final-iteration payload保存；E182 deterministic mock新增逐值断言`qpos=cumsum(ctrls)`、`geometry_qpos=[initial,cumsum[:-1]]`。E186 runner/evaluator切到不可覆盖的`shadow64x4_v6`，eval v2强制要求新字段；v1–v5均保留。尚未运行测试/GPU。
- 首轮验证在全文件ruff check处停止1次：`sampling.py`已有14个与本改动无关的历史lint债务，另有1个可修复import顺序；测试尚未执行。已只修正import排序，未趁机改历史API/docstring/mutable-default。下一步用项目既有的窄lint口径（E/F/I）与direct-main contracts继续，并记录真实测试结果。
- 第二轮E/F/I仍被历史E501长行阻止1次（无测试执行、无artifact）；改为本实验此前使用的format + F/I有效口径后全部通过：query-tape `5/5 PASS`、shadow `2/2 PASS`、py_compile、diff-check PASS。dual-qpos deterministic no-op与pre/post时序合同已锁定；现在允许在不kill/暂停现有任务的前提下跑bucket004 v6验证修复，尚未启动GPU。
- bucket004 v6 GPU run PASS：启动前GPU0 RTX5090仅151MiB/0%且无compute process，未kill/等待；固定64×4、seed0、record sim step12不变，wall=`7.884s`，single chunk COMPLETE，content SHA=`22d5e632...3f201`，config/result/manifest SHA闭合。下一步先评同一v6 chunk，要求surface trace复现恢复≤2e-5后才跑另外两object。
- bucket004 v6 evaluator仍按门FAIL，但错位已消除：surface reproduction max=`5.454e-5`（旧v5 0.672），rho=`0.999725`、false-safe/reject=0/0、selected0/top-k不变、grid/exact valid=64/64。剩余5e-5来自CPU MuJoCo按qpos重建transform与reward实际MJWarp derived transform的后端数值差，不能把它混入grid-vs-exact delta（p99约3.95e-5）。决定不放宽2e-5门，新增reward当时的MJWarp geom/body transforms直录，做v7精确复算；不改任何实验参数。
- 已实现v7 transform recorder：standard MPC在每步reward后复制MJWarp实际`geom_xpos/xmat`与`body_xpos/xmat`，final-iteration随query tape保存；旧post-step qpos与v6 pre-step qpos均保留。evaluator v3强制消费这四个reward-aligned transform，不再经CPU MuJoCo forward；E182 mock增加geometry-state trace shape合同。正式root切到不可覆盖的`shadow64x4_v7`。尚未测试或运行GPU。
- v7首轮全文件F/I lint被`examples/run_mjwp.py`与`mjwp.py`的7项历史import/unused债务阻止1次；没有改这些无关历史代码。实验脚本ruff全PASS，query-tape `5/5`、shadow `2/2`、六文件py_compile、diff-check全PASS。v7 transform instrumentation合同允许进入bucket004单case GPU验证，尚未启动。
- bucket004 v7 GPU run PASS：固定参数与record sim12不变，wall=`7.935s`，single chunk COMPLETE，content SHA=`33236049...e8b32`；新增transform trace未造成OOM或明显效率退化（v6 7.884s）。下一步只评该chunk，要求grid trace逐值复现门通过后才并行补003/007。
- bucket004 v7 evaluator正式PASS：surface trace reproduction max=`9.358e-6≤2e-5`，body/leg误差0；geometry active=64/64，grid/exact valid=64/64，false-safe/reject=0/0，selected0 match、top-k overlap=1，exact-selected-valid=100%，total rho=`0.99981685≥0.999`。v7成功隔离transform误差，允许按冻结时点补跑bucket003/007；Full仍未放行。
- bucket003 v7 GPU run PASS：固定record sim16，wall=`13.805s`，single chunk COMPLETE，content SHA=`e500376d...41904`。同一sequential command随后应跑bucket007，但回收输出只显示003，007是否生成需先查manifest/log，不能假报完成或盲目重复。
- 查明bucket007同一sequential command实际也PASS：固定record sim22，wall=`18.739s`，single chunk COMPLETE，content SHA=`526c6ee0...b5ea8`；只是前次工具回收漏显stdout，无需重跑。
- v7三object评估现为bucket004 PASS、bucket003/007 FAIL。003与007均finite、grid trace精确复现、gate/selection复现、false-safe/reject=0/0，但geometry active=0/64、grid/exact combined valid=0/64、rho=1仅trivial、exact-selected-valid=0；因此S4/Full仍禁止。下一步补轴分解确认007是否也由posture全拒绝，并基于冻结keep22判断可否进入“丢弃失败case/object”的既定口径，而不是换C或追选record时点。
- 复读plan硬门：S3明确规定“grid合格但combined-valid=0时记为G/optimizer问题”；S4要求三object代表各有combined-valid candidate，且S0–S4全过后才能S5。因此当前不能只带bucket004偷偷进Full，也不能沿用早期P阶段“失败case可丢”扩张为S3后自行丢object。下一步补持久化axis diagnostics，若003/007确为posture全拒，则按optimizer feasibility blocker处理。
- v7 axis diagnostics已写回三JSON并格式/ruff/py_compile/diff-check PASS。003/007的body、hand、leg geometry axis均64/64 valid且false-safe=0，combined 0完全由posture valid=0/64导致；003 min grid hand SDF=46.6mm，007=130.7mm，所有R components均inactive。004则posture=64/64且surface active=64/64并PASS。结论：C与D_C fidelity通过其可查询部分，但003/007冻结query暴露trajectory/posture feasibility blocker；不是CoACD失败，Full启动数仍0。
- posture诊断首次脚本因对bool做numpy quantile失败1次（只读、无artifact），修正dtype分支后闭合：阈值mean/terminal/drop=`0.10/0.12/0.18m`；003中位=`0.1248/0.3359/0.4411m`、violation≈10.04，007=`0.2744/0.6703/0.7268m`、violation≈25.43，均64/64全拒；004仅`0.0109/0.0208/0.0384m`且64/64通过。003/007是明显姿态坍塌，不是边界阈值抖动，不能靠微调门限解决。
- 对照同case E178 Full在冻结tick：003 final-iter posture valid=100%、z mean约0.052m、terminal约0.070m、drop约0.084m；007 valid=100%、mean约0.051m、terminal约0.018m、drop约0.079m。E186同tick恶化到0.34–0.73m，证明不是代表时点本来不可行，也不是posture门过严；compound physics已在早期执行阶段让003/007状态显著偏离。下一步定位坍塌从warmup还是首CEM开始，优先检查P动态接触/初始penetration，而非修改G阈值。
- 更正“实际状态已坍塌”的表述：outer realized qpos在冻结tick仍与E178接近（003 root z约0.793 vs 0.797；007约0.793 vs 0.782），大误差发生在48-step candidate rollout预测，不是已执行机器人倒地。E186从首个CEM tick起候选horizon posture误差逐tick累积；这仍是optimizer/rollout feasibility问题，但需区分真实执行状态与候选预测。
- E178/E186 config diff除冻结P/R/G与64×4 budget外发现两个相关差异：64×4自动令`beta_traj 0.93057→0.56234`；compound scene令hand-approach AABB在003/007大幅增大（003 `[.163,.078,.167]→[.270,.384,.231]`，007 `[.166,.059,.166]→[.268,.287,.285]`），而004基本不变且PASS。后者可能把CoACD完整外形误用于历史approach proxy语义，是首要可检验假设；需先看各iteration valid与reward，再决定是否是配置耦合bug。
- 代码确认AABB差异不是sidecar偶然值：`process_config`在grid backend下无条件把`hand_approach_obj_half_extents`覆盖为canonical grid的完整object AABB half-extents。该字段历史上驱动E025 hand-approach box-surface reward；因此启用D_C时同时悄然改变了一个非P/R/G既定reward的几何语义，违反E186“只改object collider与同源R/G backend”的隔离意图。下一步核对E025实际是否active及公式，再做最小A/B诊断；不能直接改回旧proxy而不验证。
- AABB耦合假设被配置排除：三case `hand_approach/contact_mask/hold_contact/hand_support` scale全0，active的surface-band走D_C；`object_lift`虽读取half-z，但sim/ref bottom两侧同减该常数，解析上完全抵消。因此003/007失败不能归因于approach AABB，暂不改config。剩余主假设是compound physics改变48-step rollout动力学，或64×4 sampling schedule不足；下一步用E178同tick前4 iteration逐轴对照区分。
- 同冻结tick前4 iteration对照：E178从iteration0就已有posture valid（003 71.3%、007 98.0%），到iteration4达98.7%/100%；E186四轮始终0%。004两者均接近100%。且003/007 E186从首个CEM tick6开始每轮都0并持续fallback，非“只差更多iteration才能过”的简单收敛现象。下一步比较首tick的terminal/drop分解；若mean接近但tail明显爆炸，将指向compound rollout动力学而非sampling数量。
- 首CEM tick6逐值推翻“compound从一开始破坏rollout”假设：E178/E186 iteration0的003 posture mean/terminal/drop=`.097/.251/.322` vs `.096/.246/.313`，007=`.148/.374/.465` vs `.150/.378/.470`，高度一致；004也一致。真正差异是budget：E178 003直到第15 iteration才首次出现valid，007首tick32轮仍无valid但后续control tick恢复；64×4在首tick只能fallback，之后状态滚入不可恢复区。故当前S3/S4的64×4“必须有combined-valid”对003/007是已证实的低预算false-negative，不是C失败。
- 已在active plan新增S3b消歧：不改任何冻结项，先做bounded formal-budget probe；003用1024×32到首CEM，007用1024×32到冻结sim22，先测003 wall/显存再决定007。若正式budget恢复valid，只能证明64×4 canary预算不足，不能偷换D_C/D_M fidelity门或自动放行Full；若仍0则保持阻塞。下一步实现默认关闭的transform-record开关和持久化probe runner。
- 已实现S3b runner与static test/launch wrapper：formal `1024×32 seed0`、production max_sim_steps不变、目标query后停止；被动轮询total GPU memory且不操作其他process；payload只存qpos/reward/gate/posture traces，显式禁止16MB/64-case的MJWarp transform扩张到full-budget大artifact。新增`query_tape_record_geometry_state`默认false，v7 shadow显式true以保持exact replay合同。尚未格式化/测试/GPU运行。
- 修正launch wrapper repo-root层级后首轮验证在新runner import排序处停止1次，direct-main tests尚未执行、无probe artifact。ruff format已机械整理2文件；下一步只修import顺序后重跑全部query/shadow/probe合同，不重复错误lint状态。
- 手工交换local/spider import后ruff仍判I001（第2次同类失败），说明项目ruff把E186脚本目录识别为first-party的顺序与直觉不同；仍未执行tests、无artifact。按失败协议不再手猜，下一步仅对新runner执行ruff安全`--fix --select I`，检查diff后再跑合同。
- ruff安全修复import后全部合同通过：query-tape `5/5`、shadow `2/2`、fullbudget probe `1/1`，7文件format、实验脚本ruff、bash-n、py_compile、diff-check PASS。formal-budget probe root尚为空；现在允许先在本机GPU0叠加跑bucket003（首CEM后停止），不kill/等待，wall与total-memory peak将持久化。
- bucket003 formal-budget bounded probe PASS：`1024/1024` combined-valid、`1024/1024` posture-valid、finite，wall=`30.38s`、total GPU memory peak=`2329MiB`，selected=102；surface仍inactive=0（故仅feasibility evidence，不是R fidelity）。这直接证明64×4的0/64是预算false-negative，compound physics在正式预算下可恢复。成本可接受，按S3b允许继续bucket007到sim22（约6个full-budget queries），仍不放行Full。
- bucket007 formal-budget probe已在本机GPU0启动并运行中；目标sim22、1024×32、recorder无transform，未kill/等待独占。前两次30s轮询尚无stdout（worker日志写文件、属预期），session仍活跃；不重复启动。
- bucket007同一session继续运行，追加约30s轮询仍无终端stdout/错误；未达到预估上限且可继续有意义等待。下一步读持久log tail确认已完成到哪个sim tick，再继续同一session。
- bucket007 formal-budget bounded probe PASS：6个full-budget CEM tick完成，per-query plan约19.1→21.5s，总wall=`127.02s`、total GPU memory peak=`2131MiB`；目标sim22 final query `1024/1024` combined-valid、`1024/1024` posture-valid、surface-active=`1024/1024`、finite，selected=102。003/007均在正式预算恢复，确认v7 64×4失败是预算门false-negative而非compound physics infeasible。仍需正式记录Gate修订与S4策略，Full启动数0。
- active plan新增S3c：003/007以原冻结record时点16/22做formal-budget transform shadow，离线exact按64 samples分批，三object均取得非平凡R/G evidence后才能修订Gate 4。evaluator已泛化为任意sample count/可选shadow root，并将grid/exact每64 samples分批，v7 64-case仍单batch。尚未格式化测试，也未创建S3c runner/GPU artifact。
- 已新增formal-budget fidelity runner/test/local wrapper：1024×32、原record sim16/22、transform recorder true、immutable root=`fullbudget_fidelity_v1`，被动显存poll。format/import-fix/ruff/static `1/1`/bash-n/pycompile/diff-check PASS；泛化evaluator回归bucket004 v7仍PASS且指标逐值不变。现在允许先跑003 S3c，尚未启动GPU。
- bucket003 S3c formal-budget transform shadow已在本机GPU0运行中；同一session已持续约50s、无stdout错误（worker写log），未重复启动/未操作其他process。下一步继续同session至manifest返回，再检查artifact体积与active/valid后启动分批exact eval。
- bucket003 S3c GPU run PASS：原冻结sim16、1024×32，wall=`77.40s`、peak total GPU memory=`4005MiB`、chunk=`277.1MB`；combined/posture valid=`1024/1024`且finite，但recorded grid surface-active仍=`0/1024`。因此formal budget解决feasibility，却没有在原E178时点恢复R support；按S3c预注册门暂不能成为非平凡R authority。下一步仍做分批exact确认D_M也inactive还是grid漏reward；结果前不启动007 S3c。
- bucket003 fullbudget exact分批完成并FAIL：grid/exact geometry-active均0/1024，grid/exact gate valid均1024/1024、false-safe/reject=0/0、trace reproduction exact；min hand SDF grid/exact=`45.8/48.6mm`，远超surface hard band `[-1,3]mm`。故不是D_C fidelity问题，也不是posture/physics infeasible；是R在真实C表面上的捕获域缺失——E178 handcraft proxy曾在同tick产生surface reward，真实C/D_C下手离表面约5cm且零梯度。bucket003不得直接Full。下一步补007 S3c以判断object-specific范围，再需要用户决策“重设计R approach support”或丢bucket003 cases。
- bucket007 S3c formal-budget transform shadow已启动，同一session运行中约40s且无stdout错误；目标sim22需6个query，未重复启动/未干预其他process。完成后先看surface-active/valid，再分批exact。
- bucket007 S3c GPU run PASS：原冻结sim22、1024×32，wall=`127.51s`、peak total GPU memory=`3243MiB`、chunk=`258.3MB`；combined/posture valid=`1024/1024`、surface-active=`1024/1024`、finite。与003不同，007在正式预算和真实C表面上有完整R support；下一步分批exact验证rho/false-safe/selection，未通过前仍不启动S4/Full。
- bucket007 fullbudget exact FAIL：geometry-active=1024/1024、grid/exact gate valid=1024/1024、false-safe/reject=0/0，但surface trace reproduction max=0.203（hard-band边界敏感）、exact-grid geometry delta p99=0.0810、total rho=`0.80238<0.999`、selected0 mismatch、top-k overlap=`0.5784`。这不是P/G安全误判，而是5mm D_C误差相对1.5mm sigma与4mm硬带过大，真实高密度候选下R排序不稳定。E186现有冻结方案不得启动S4/Full：003为R捕获域缺失，007为R grid/hard-band fidelity失败；Full启动数仍0。
- 已新增正式log `255_E186_production_prg_shadow_blocker_results.md`并更新tracker：C0–C4 PASS，C5 R FAIL，C6 partial，C7 blocked，C8 not started。log记录全部持久路径、formal-budget wall/显存、可视化豁免原因与两条后续路线；建议保留P/G并新版本重做R，禁止在E186内静默drop或启动Full。下一步重建log INDEX并做最终代码/文档closure检查。
- log INDEX已重建并含255；最终diff-check、query-tape `5/5`、shadow `2/2`、fullbudget feasibility `1/1`、fullbudget fidelity `1/1`全部PASS。E186按预注册门正式停在S3：当前安全动作已耗尽，下一步需要用户批准新R路线（推荐）或显式重冻drop子集；Claims未全过，按实验规则不commit、不push、不启动Full。
- 2026-08-02：用户批准新开R改进实验并要求先写详细计划；本轮只规划，不改实现、不启动GPU。已复读E178计划194–196与结果237–241、E186计划204与结果255，确认新实验编号采用E187、plan序号205。
- E187的Gate 0将先锁定E178复现合同：legacy backend默认路径逐值不变，E187新字段默认关闭，E178 scene/override/result/config不覆盖；只有代表配置diff、deterministic reward/gate测试与短replay均通过后，才允许E187 shadow/canary/Full。
- E187保持keep22与三个object-specific CoACD collider冻结：P继续compound-convex，G继续`D_C-epsilon`；只重做R为基于canonical `D_C`的远场approach→近场surface连续reward，并针对bucket007预注册细grid/analytic-C的离线选择门，禁止按Full结果调参。
- 复现合同取证：E178 Full authority SHA仍为`de9a3d...022a8`，27行冻结`1024×32 seed0`，历史结果/`config_act.yaml`/render/eval均完整；E178最终12门numeric口径为原6门加tracking `20/20/20/20/20/10`。
- 当前分支`experiment/E161-surface-release-ablation`、HEAD=`76534fe...`; worktree含大量E186未提交实现/结果文档，均视为用户当前工作，E187计划不得reset/checkout/覆盖。尝试按E186 plan中的候选通用launcher名读取时发现`run_E186_local.sh`和`run_E186_remote_a6000.sh`尚不存在（1次）；现有入口仅为S3本地probe/shadow，E187计划会把正式local/remote/pull脚本列为实现项。
- Reward实现取证：E186仍使用`surface_band_score=exp(-|D_C|/1.5mm)`并在`D_C∈[-1,3]mm`外硬置0；这解释003约50mm处完全零支持，也解释007的5mm grid误差跨硬边界后排序剧烈变化。
- v4 grid实际分辨率为bucket003/007 5mm、bucket004 2.5mm，`epsilon_grid`分别2.875/1.386/2.537mm；因此E187不能把G和R共用同一保守误差处理：G保留`D_C-epsilon`，R必须用连续名义距离，并在离线冻结阶段比较2.5/1.25mm细grid或exact convex-union backend的精度/吞吐，不能直接沿用5mm hard band。
- Tracker确认最新相邻实验E178–E186的authority链闭合；E187应登记为Phase 50“计划完成”，待实现/运行后再新增结果log并更新状态。
- E178实际R参数已核准：scale=1.5、hard band=`[-1,3]mm`、sigma=1.5mm、symmetric_abs、非bimanual、contact-mask gate、tail decay=0.15；hand-approach/contact-mask额外reward均为0。E187会保持geom group、gate、scale与tail语义，仅将surface score切到显式opt-in的连续模式。
- E187主R预注册为双尺度连续score（不做Full后sweep）：`0.25*exp(-smooth_abs(d)/50mm)+0.75*exp(-smooth_abs(d)/15mm)`，`smooth_abs(d)=sqrt(d^2+1mm^2)-1mm`；在003约50mm缺口仍有约0.12归一化支持，同时近表面接近1，深穿透仍由原G约束。legacy hard-band仅作诊断对照，不允许根据结果回选参数。
- keep22上的E178冻结十二门baseline为`8/22 PASS`（不是全27的10/27）。E187最终paired主表必须以8/22为直接基线，并保留22行逐case join、6门/12门、连续tracking/contact/penetration与效率指标。
- 已创建详细计划`plan/205_E187_canonical_distance_continuation_reward_plan.md`：包含E178 Gate0、双尺度continuation公式、per-object `5→2.5→1.25mm`最粗通过grid决策树、P/G回归、三卡production canary、keep22 Full、paired 8/22 baseline、3D/2D visual与stop rules。
- 计划明确本轮不实现/不跑GPU；后续先过E178 authority/config/legacy reward/G/recorder/replay兼容门，再冻结reward+grid。Full结果禁止用于调参，不通过case也不能在22行authority内事后丢弃。
- Tracker已新增E187 Phase 50行：状态`计划完成；未实现/未启动`，仅链接plan 205，未虚构结果log或GPU产物。
