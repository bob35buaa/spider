# CORE4D 当前进度

> 完整备份：
> [E167–E179 初始恢复](progress_archive/E167_E179_20260724_full_backup.md) ·
> [E179 执行过程](progress_archive/E179_20260724_execution_full_backup.md) ·
> [E179 最终收尾](progress_archive/E179_20260725_closeout_full_backup.md) ·
> [E180–E181 完整执行](progress_archive/E180_E181_20260731_full_backup.md) ·
> [E182 Gate0–S1 canary](progress_archive/E182_gate0_s1_canary_20260801_full_backup.md) ·
> [E182 S1 continuation](progress_archive/E182_s1_continuation_20260801_full_backup.md)
>
> 本文件只保留 E182 当前可靠结论与下一执行入口。

## 2026-08-01：不可变口径

- 计划：`plan/200_E182_task_conditioned_coacd_full_cem_plan.md`；Tracker 为
  `✅ Gate0 PASS；🚧 S1 query tape`。
- Full authority 与 E178 exact：27 case、`1024×32 seed0`；K 只测
  `8/16/32`，不测 K4。
- heldout24 在 production SHA 冻结前 selection-forbidden，冻结后仅
  evaluation-only；不得按 heldout/Full 结果反选碰撞体。
- 资源只用本机 GPU0 + `spider-remote` RTX 6000 Ada GPU0/1；允许与已有任务
  叠加，禁止 kill、暂停、抢占或修改已有进程，不使用 A100。
- 2026-08-01 用户再次确认 Full CEM 调度口径：远程改用两张 RTX 6000
  Ada，不再使用 A100；本机与远程均可与正在运行的程序叠加，任何
  E182 launcher 都不得 kill/暂停/抢占/修改现有进程。
- 已完成资源口径一致性审计：plan、Gate1 log、Tracker、progress 和 E182
  launch/runtime contract 均指向本机 GPU0 + Ada GPU0/1；脚本无
  `kill/pkill/killall/tmux kill-session` 路径。文档中 `A100` 只以“不使用/不套用”
  的禁止性说明出现，不存在 A100 launch 命令。

## S0：Gate0 PASS

- authority=`27/3/24`，case snapshot files=`108`，E181 candidates=`54`；E178
  manifest SHA=`de9a3d165301318049da208aad52b1f5bc4d3ed671631f0097958734a0f022a8`。
- 当前 remote immutable root：
  `/home/xiayb/pHRI_workspace/e182_runs/e182_832e7fe0f61193a1/spider`；source
  `5093 files`、SHA=`832e7fe0...`，50-file runtime layer 与 CoACD dependency
  layer均 exact PASS，未修改 shared checkout/venv/已有进程。

## S1：query tape 进行中

- E178 历史 NPZ 不含原始 1024-sample tensor；dev3 用 exact E178 override/task/
  seed 做 `64×4 seed0` shadow replay。selection 唯一使用 frozen `on_a`；`on_b`
  仅诊断；production Full 强制 recorder-off。
- Gate1 硬门：deterministic mock exact、default-off no-op、same-run payload exact、
  chunk/manifest/source SHA 完整。真实 CUDA 跨进程 divergence 仅 report-only。
- bucket007：off/on_a/on_b 完成；on_a/on_b=`77/77` chunks，same-run mismatch/
  nonfinite=`0/0`，selection content SHA=`ab73e0ee...`。
- bucket004：远程 worker 已自然 PASS 并退出；on_a/on_b=`139/139` chunks。
- bucket003：远程 worker 已自然 PASS 并退出；on_a/on_b=`202/202` chunks，same-run
  integrity PASS。bucket003/004 两个 E182 tmux 都是自然结束，全程未停止、重启或
  修改已有任务。
- pull 入口 `pull_E182_remote_a6000_results.sh query-tape` 为 no-delete +
  `--ignore-existing`，远程完成后执行 dev3 same-run audit。
- no-delete pull 已完成：50-file runtime layer SHA exact，missing/mismatch=`0/0`；本地
  dev3 aggregate same-run audit=`PASS rows=3`，pull=`PASS`。下一步读取三 case manifest/
  wall/content SHA 细项并构建 bucket003/004 factored PRG tape。
- 三 case same-run 细项均 mismatch/nonfinite=`0/0`：on_a/on_b chunks 分别为
  bucket003=`202/202`、bucket004=`139/139`、bucket007=`77/77`；selection content SHA
  分别为 `561ea037.../68378948.../ab73e0ee...`。本地结果约 `804MiB`，磁盘尚余
  `734GiB`；wall time 位于 case manifest 的 `rows[0]`，下一次按正确层级聚合。
- bucket003/004 完整 factored P/R/G tape builder 已启动，限定两个 dev case且仅消费
  frozen `on_a`，最终 `PASS rows=2`。三 case逐 chunk verifier 均 PASS：bucket003
  `204` chunks/`925.20MiB`，bucket004 `141`/`636.73MiB`，bucket007 `79`/
  `352.83MiB`；总计约 `1.9GiB`，均为 `18 P geoms/7 consumers/1669 points`，压缩比
  `12.83×`，raw selection SHA exact。
- wall 汇总读取 attempt1 在 bucket007 旧 manifest 缺 `wall_seconds` 时触发 KeyError；
  003/004 已正确得到 off/on_a/on_b=`660.28/660.56/662.00s` 与
  `474.98/480.84/479.62s`。下一步按 optional field 处理 bucket007，不重复硬索引。

## Object-local P/R/G tape

- `build_prg_query_tape.py` 从 frozen config+`load_data` 恢复 reference，复用 E027b
  43→42 conversion；E178-final 来自 NPZ `(record,ctrl_step,42)` flatten。
- P 保存完整 qpos 作 full-scene replay，并从 explicit pairs + masks恢复 18 个 physics
  robot geoms；R/G=`25` unique query geoms、`1669` fixed points。
- tape 使用无损 factored geom-pose + fixed-offset 表示。bucket007 已完整构建：
  `79/79 PASS`，stored=`352.8MiB`，expanded=`4.42GiB`，缩小 `12.83×`。
- Gate1 只有在 bucket003/004 拉回、两条完整 tape 逐 chunk验证、三 case aggregate
  artifact PASS 后才可宣布闭合。

## S2：Gate1 等待期间的合成准备

- 新增 `evaluate_task_queries.py` 与 synthetic tests；exact-C 使用 manifold boolean
  union 后再做 Open3D negative-inside SDF，避免内部 overlap face 错误缩短距离。
- radius-aware paired metrics显式报告 coverage/nonfinite、sign disagreement、deep
  mismatch、false accept/reject 与 SDF p90；合成重叠 box `3/3 PASS`。
- 尚未读取真实 candidate score；Gate1 PASS 后才在 frozen dev3 tape 上评估 54 个
  K8/K16/K32 candidate。

## 当前验证

- query recorder `3/3`、pipeline `4/4`、runtime/launcher `4/4`、PRG tape `3/3`、
  preflight `5/5`、S2 synthetic `3/3` PASS。
- E182 ruff/format、四个 launcher `bash -n`、`git diff --check` PASS。
- 2026-08-01 commit preflight 已用 authoritative `uv run python` 重跑全部 E182
  direct-main groups：Gate1=`2/2`、preflight=`5/5`、PRG tape=`3/3`、query
  recorder=`3/3`、pipeline=`4/4`、runtime/launcher=`4/4`、fixture=`2/2`、task
  geometry=`5/5`，合计 `28/28 PASS`；`ruff check`、`ruff format --check`、四个
  E182 launcher `bash -n` 与 `git diff --check` 全部 GREEN。

## 已记录错误

- 远程 chunk count attempt1 使用错误目录层级返回 0；已改为 `raw_chunks/mode/case`。
- manifest摘要 attempt2 使用远程不存在的裸 `python`；未影响任务，不再重复，改用
  frozen remote Python或拉回后本地审计。
- commit preflight attempt1 使用当前 `python -m pytest`，但本地 base Python 未安装
  pytest，命令在收集前立即失败，未执行任何实验/进程操作。后续改用项目已固化的
  direct-main test 入口，不重复同一 pytest 命令。
- commit preflight attempt2 用 base Python 直接运行 direct-main，在首个需要
  `mujoco` 的 import 处立即失败；根因是绕过了项目固化的 `uv run python`/
  `.venv/bin/python`。已从 E182 launcher 确认 authoritative 本地入口为 `uv run
  python`，下一次使用该环境，不再用 base Python 运行依赖测试。

## 下一执行入口

1. 只读等待 bucket003 worker 自然 PASS；不得发信号或重启。
2. 执行 no-delete pull，校验三 case off/on_a/on_b、same-run、content SHA 与 wall time。
3. 构建 bucket003/004 完整 factored PRG tape，三 case逐 chunk验证并生成 Gate1 artifact。
4. Gate1 PASS 后更新 Tracker，再进入 S2 real candidate audit；仍禁止读取 heldout。

## 2026-08-01 checkpoint 提交准备

- 用户明确要求先提交 Git；本次定位为 Gate1 + S2 evaluator foundation
  checkpoint，不声称 C3–C9 或 Full CEM 已完成。
- 待提交集合为 23 个 E182 范围文件，cached `diff --check` PASS；包含
  Gate1 log/auditor、factored P/R/G tape、S2 frozen fixture/evaluator 基础、Ada
  0/1 no-kill launch/pull/runtime contracts 与实验台账。

## Gate1 aggregate TDD

- 已先新增 aggregate audit direct-main tests：要求真实 dev3 artifact-only audit PASS，
  且篡改 PRG selection content SHA 必须精确 FAIL。RED 已实测为预期
  `ModuleNotFoundError: audit_gate1`。现已实现 runtime/same-run/三 case PRG verifier/
  selection identity/direct-tests 聚合器，真实 artifact 与 tamper tests=`2/2 PASS`；
  正式 Gate1（包含 direct-main tests）已 `PASS cases=3`，selection set SHA=
  `3781ad96505ee597222f145fbcaa943be198a9473fb4f0faf72b56fed247c8a5`，heldout
  仍为 `NOT_ACCESSED_DEV3_ONLY`。下一步静态复核、写阶段 log、更新 Tracker 至 S2。
- Gate1 artifact 最终 SHA=`a1177e14...f6f5`，并冻结 auditor/builder/replay/test source
  SHA；runtime=`50/50`、三 case verifier mismatch=`0`、五组 direct tests 均 exit0。
  首次静态检查的 unused import/format 已 mechanical 修复；Gate1 tests `2/2`、正式
  audit、E182 ruff/format、launcher bash syntax、diff-check 全 GREEN。
- 已创建阶段日志 `log/247_E182_gate1_real_query_tape_results.md`，记录三 case chunks/
  SHA、wall time、factored storage、Gate1 evidence、claims边界与 S2 下一入口；不把
  instrumentation PASS 扩大解释为 candidate/downstream 质量改善。
- Tracker 已推进为 `✅ Gate0/Gate1 PASS；🚧 S2 task audit` 并链接 log 247；log INDEX
  已重建。全目录 ruff/format、E182 launchers syntax、diff-check继续 PASS。

## S2 规模审计

- 每个 CEM chunk 为 `(64 samples,48 horizon,42 qpos)`；bucket003/004/007 分别有
  `202/139/77` chunks。每 pose 的1669点中 R_surface/G_hand=`1600`，其余 robot/leg
  为 `21/48`；直接对54个 candidate全展开会产生约386亿 point-candidate queries。
- S2 必须在读取 candidate score 前冻结 progressive fixture：reference/E178-final可
  全点；CEM first-pass需保留所有 tick/sample/horizon和全部 safety/leg points，对手 mesh
  做固定索引 reduction，并对每 K finalist做更高密度复核。下一步先写 sampling manifest/
  representativeness tests，再做固定 candidate timing-only benchmark。
- 已先写 S2 fixture direct-main tests：冻结 screen=`mesh64` 全54候选、finalist=
  `mesh256` 每 object×K 最优、production canary=`full1669`；三个 tier保留全部 CEM
  chunk/sample/horizon，primitive points全保留且索引嵌套。另要求 Gate1非PASS时拒绝
  freeze；builder尚未实现，下一步运行预期 RED。
- S2 fixture RED 已实测为预期 `ModuleNotFoundError`。现已实现 pre-score builder：
  从 Gate1/三条 PRG manifests冻结 nested point indices、all-pose query counts和54个
  BUILD_PASS candidate identity（只读 build manifest，不读 E181 fidelity score）；下一步
  执行 GREEN、静态检查并生成正式 frozen fixture。
- Fixture tests=`2/2 PASS`；首次静态检查 ruff/diff-check通过，仅两个新文件需要
  formatter。下一步 mechanical format、复跑并生成正式 fixture manifest。
- 正式 S2 fixture 已 `FROZEN cases=3 candidates=54`，manifest SHA=`b278062d...f0bd`、
  candidate identity SHA=`d43beee1...65d1`，Gate1 SHA exact。每 pose screen/finalist/full
  points=`197/581/1669`；三 object 单 candidate screen queries=`123.72M/85.17M/
  47.24M`，所有 CEM poses与全部 safety/leg points保留，heldout未访问。
- 已复核 authoritative R/G 公式：每 pose先对 consumer geom union取 min clearance；
  G 再沿48 horizon计算 sample min/violation fraction，并应用独立 body/hand/leg
  floor与max-violation阈值；R robot/leg为hinge，surface-band为带宽/指数 score。
  evaluator需复用 frozen config值和 radius-adjusted clearance，不能只报点级SDF误差。
- 固定 timing-only benchmark（bucket007 `t005_k08_v032`，不用于selection）在605,184
  screen points上测得 Open3D oracle/candidate=`7.78/13.77 Mquery/s`，boolean union
  `0.005s`；预计54-candidate screen可实际流式完成。raw chunk顺序与 active optimizer
  tick一一对应，含原始64-sample G summaries/valid masks，可用于逐 chunk shadow重算。
- 已扩展 S2 synthetic tests，冻结 consumer min-reduction、legacy/hard-floor G semantics、
  mask flip/false accept/reject，以及 R robot/leg hinge + symmetric surface-band公式；实现
  RED 已实测为缺少新 API。现已实现 consumer clearance reduction、48-step G sample
  summary/valid mask、gate comparison和 ungated geometry R components；功能 `5/5 PASS`。
  首次静态仅 import order/format，mechanical fix后测试、ruff/format、diff-check全 GREEN。
