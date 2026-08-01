# CORE4D 当前进度

> 完整备份：
> [E167–E179 初始恢复](progress_archive/E167_E179_20260724_full_backup.md) ·
> [E179 执行过程](progress_archive/E179_20260724_execution_full_backup.md) ·
> [E179 最终收尾](progress_archive/E179_20260725_closeout_full_backup.md) ·
> [E180–E181 完整执行](progress_archive/E180_E181_20260731_full_backup.md) ·
> [E182 Gate0–S1 canary](progress_archive/E182_gate0_s1_canary_20260801_full_backup.md) ·
> [E182 S1 continuation](progress_archive/E182_s1_continuation_20260801_full_backup.md) ·
> [E182 S2 v2–v8 完整过程](progress_archive/E182_s2_v2_v8_20260801_full_backup.md)
>
> 本文件只保留 E182 当前权威状态与下一执行入口。

## 2026-08-01：不可变执行口径

- 权威计划：`plan/200_E182_task_conditioned_coacd_full_cem_plan.md`。
- Full 与 E178 exact：27 cases、`seed0, 1024×32`；K只允许`8/16/32`，不测K4。
- heldout24 在 production SHA 冻结前 selection-forbidden，冻结后只作 evaluation；
  禁止按 heldout/Full 结果反选碰撞体。
- Full 资源只用本机单卡 + `spider-remote` RTX 6000 Ada GPU0/1；允许与现有任务
  叠加，禁止 kill、暂停、抢占或修改已有进程，不使用A100。
- bucket003 完整 P/R/G launch floor 未闭合前，grid-SDF、heldout和Full全部禁止。

## E182 已完成阶段

- Gate0 authority/preflight PASS；Gate1真实P/R/G query tape PASS，见log247。
- E181原global cavity `≤0.1%`已降为report-only；E182使用882-pose真实static P authority。
- v5 global threshold：`0/9 P PASS`。
- v6 per-segment threshold hybrid：K8/K16/K32各`0/6561 P PASS`。
- v7 segment3 Bell(4) partition：K16/K32各`0/15 P PASS`；只产生
  `TP18/phantom8/missed9`与`TP19/phantom10/missed8`，见log248。

## v8 task-aware local pre-segmentation：完整负结果

- plan200 amendment冻结family：
  `2 task-derived planes × threshold {5,10,20mm} × K {16,32}=12`；
  每row为9个nonempty segments，禁止跨presegment merge。
- dominant axis=`x`；planes=
  `-0.14311002844145554m/-0.11878629238288715m`；两split的volume delta=
  `4.77535e-11/-1.27153e-12m³`。
- pre-freeze closure：v8 contracts14、parent matrix60，共`74/74 PASS`；ruff/format/
  compileall/diff-check GREEN；freeze前v8 root为空。
- protocol SHA：`d1b19069c4be0b7f4c33884ca7aab6539ad9080ce0624a84d8eb78cf9e45fdd8`。
- 12/12 new CoACD children、6/6 composite bases、12/12 candidates BUILD_PASS；
  actual hulls=`[16,32,16,32,16,22]×2`，所有part vertices≤256。
- static P=`0/12 PASS`；aggregate SHA=
  `10666ec4d61668b528838cdab7cef9e8cb5748c2a884b2b10a827d7d694df60b`。
- 最佳near-miss：plane1/t010/K32(actual32)=
  `TP18/phantom7/missed9, precision=.720, recall=.667`。
- TP≥19的最低phantom点：plane1/t020/K32(actual22)=
  `TP19/phantom12/missed8, precision=.613, recall=.704`。
- frozen stop action：`STOP_V8_NO_PLANE_OR_FLOOR_CHANGES`；完整P/R/G未启动。
- 详细表、耗时、Claims与路径见
  `log/249_E182_bucket003_task_aware_preseg_v8_results.md`。

## 可视化实际观察

- post-result renderer固定best-balance/high-recall/TP19三代表，输出6张3D/2D图；
  visual manifest SHA=`1d1ce64ea6e97f44476dc4580c4511cf0eb6b9efe600e877c44fcf9e9558a5d3`。
- 已查看全部原分辨率图：最深phantom均为reference pose148，集中在同一顶部端部rim
  手部点；oracle clearance≈`+0.465mm`，candidate侵入`-4.01/-4.37/-4.77mm`。
- high-recall把TP提到22时phantom也升到22；best-balance降到7 phantom但漏9 TP；
  TP19仍需12 phantom。问题是局部rim convex外包trade-off，不是远处body/leg噪声。

## 当前状态与下一入口

```text
P_LAUNCH_FLOOR_NOT_CLOSED
FULL_NOT_AUTHORIZED
V8_STOPPED_COMPLETE_NEGATIVE_RESULT
```

- 禁止继续移动plane、加手工cut、降低0.70 floor或强送Full。
- v5–v8已超过三次失败协议；新的方法自由度需要用户确认并先写新plan/protocol。
- 可供决策的结构方向：
  1. 新协议同时使用两条自动plane，隔离三个transition x区间；
  2. rim-aware非平面分区/局部原mesh collision primitive；
  3. 改用MuJoCo SDF/plugin等更高保真P碰撞路线。
- 本轮没有访问heldout/grid、没有启动GPU/Full，也没有操作任何既有进程。

## 本次归档

- 原`progress.md`的1260+行完整过程已移动到
  `progress_archive/E182_s2_v2_v8_20260801_full_backup.md`；未删除过程证据。
- Tracker已链接log249，log INDEX已包含249且文档audit PASS。
- 最终closure已重跑PASS：v8=`14/14`、visual=`2/2`，4文件ruff/format/compileall、
  strict protocol/static/visual validator、log链接、Tracker/INDEX与repo diff-check全GREEN；
  三个最终SHA保持`d1b19069...45fdd8`/`10666ec...df60b`/`1d1ce64e...8a5d3`。
- v8阶段已完整结束为负结果；下一方法超出当前冻结family，必须先由用户确认结构方向，
  再新增plan amendment/protocol。当前不满足Full启动条件，持续目标尚未完成。
- 当前工作计划5/5步骤均已完成（包含0/12 stop分支）；这只代表v8阶段闭合，不代表
  plan200的C0–C9或持续目标完成。等待用户在“两plane同时使用 / rim-aware非平面 /
  MuJoCo SDF/plugin高保真P”中确认新的结构自由度。
