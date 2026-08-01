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
