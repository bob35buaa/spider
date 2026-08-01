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
