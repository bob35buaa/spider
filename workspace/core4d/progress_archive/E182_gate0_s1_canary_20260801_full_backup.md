# CORE4D 当前进度

> 完整备份：
> [E167–E179 初始恢复](progress_archive/E167_E179_20260724_full_backup.md) ·
> [E179 执行过程](progress_archive/E179_20260724_execution_full_backup.md) ·
> [E179 最终收尾](progress_archive/E179_20260725_closeout_full_backup.md) ·
> [E180–E181 完整执行](progress_archive/E180_E181_20260731_full_backup.md)
>
> 本文件只保留最近完成工作的可靠结论与下一会话入口。

## 2026-07-31：E181 收口（ASSET_REJECTED）

- 计划/结果：`plan/199_E181_coacd_canonical_geometry_plan.md`、
  `log/245_E181_coacd_gate_b_asset_rejected.md`、
  `log/246_E181_rejected_candidate_visual_diagnostic.md`。
- Oracle `3/3 PASS`，CoACD `54/54 BUILD_PASS`；三物体均因冻结的全局
  cavity `≤0.1%` 得到 `0/18`，因此 E181 未生成 `C*`/`D_C` 或启动 Full。
- rejected 3D/2D 可视化 `3/3 APPROVED_DIAGNOSTIC`；它说明几何误差存在，
  但不能证明真实 P/R/G 与 downstream 不可用。

## 2026-07-31：E182 task-conditioned CoACD + paired Full（计划完成）

- 新计划：`plan/200_E182_task_conditioned_coacd_full_cem_plan.md`；Tracker 已新增
  Phase 45 planning-only 行。本轮只写计划，未实现 backend 或启动 GPU。
- `broader cavity≤0.1%` 降为 report-only；dev3 deterministic query tape 对
  同一点比较 `D_M/exact-C/D_C`，分别审计 P contact、R reward 和 G mask/rank。
- production geometry 只在 object-specific K8/16/32 中按
  task-error–runtime Pareto 冻结；只有 catastrophic launch floor 阻断 Full，
  轻微 preferred gate 或 throughput 超标只进入最终 verdict。
- heldout24 在 selection manifest 冻结前不参与 E182 选型，冻结后作为
  evaluation-only 正常跑 query audit 与 Full；禁止按其结果反向调参。
- Full authority 与 E178 同序 27 rows、`9/4/14`、`seed0,1024×32`；基线六门
  `16/27`、十二门 `10/27`，质量、逐门迁移、盲审和效率均 paired 报告。
- worker 为本地单卡 + `spider-remote` RTX 6000 Ada `0/1`；三卡均允许叠加
  已有任务但保留显存/UUID/owner 双检查，不 kill/暂停/抢占。按 measured LPT
  分配，速率接近时预期 `9/9/9`。
- E181 尚未实现 grid-SDF runtime、query tape 和 sidecar；均已作为 E182 正式
  实现项列出，下一入口是用户确认后从 S0 authority/preflight 开始。
- 计划完整性审计 PASS：单 H1、10 个结构化 H2、Mermaid 含 accessibility
  metadata；Tracker 保持一句话索引；计划明确包含 report-only cavity、真实查询、
  K Pareto、三卡 Full、E178 paired 指标、效率拆分与失败协议；diff-check PASS。

## 2026-08-01：E182 计划口径修订

- 计划已按用户口径修订：删除全部 K4 rescue；K8 是最低预算，仍过慢时进入
  efficiency failure。heldout 文案改为 `selection-forbidden→evaluation-only`：
  case/E178 baseline 不保密，隔离的只是用于选型的 E182-dependent evidence。
- 修订审计 PASS：计划中 K4 仅以“明确不测试”出现；heldout 角色、冻结内容、
  失败后不反调参数和 full27 正常执行均已写明；单 H1/H2 结构、progress<100
  与 `git diff --check` 全部通过。
- Full 资源再次按用户口径修订为本机单卡 + `spider-remote` Ada GPU `0/1`；
  不再使用 A100。三张卡都允许与已有 compute process 叠加；launcher 禁止
  kill、暂停或抢占，显存不足/OOM 时只延迟或终止 E182 自身 worker。
- 资源修订一致性审计 PASS：E182 已无 A100/四卡执行入口，allocation、脚本名、
  命令、风险表和 checklist 全部改为三 worker；`git diff --check` PASS。

## 2026-08-01：E182 正式执行启动

- 用户已授权按 plan 200 持续推进完整 E182，不再是 planning-only；当前从 S0
  authority、scene snapshot 与环境 preflight 开始，按 Gate 0→Full 顺序执行。
- Full 资源冻结为本机单卡 + `spider-remote` Ada GPU `0/1`；三卡允许与已有
  compute process 叠加，现有进程不作为 idle gate，禁止 kill/暂停/抢占。
- completion 仍以 plan 的 C0–C9 和 S0–S7 全部证据闭合为准，不把阶段性通过
  当作实验完成。
- 初始只读 GPU 快照：本机 RTX 5090 free=`28367MiB`，已有 SUGAR process；
  `spider-remote` Ada0/1 free=`39774/40892MiB`，两卡均有 SUGAR process。
  三卡共存状态符合用户授权；未 kill、暂停、抢占或修改任何已有进程。
- 当前代码树没有 E182 实现/结果目录；可复用 E181 的 authority builder、环境
  probe、54 个 CoACD candidate 和 E178 manifest，S0 需要建立 E182 独立合同。
- S0 复用审计：E181 candidate tree=`7.2MiB/1149 files`，oracle=`3.9MiB`；
  E178 27 条 trajectory/contact-mask/source/effective-scene 总量约 `6.4MiB`，
  可为 E182 建立真实副本与 SHA manifest，不必只留易漂移的路径引用。
- S0 实现采用 TDD：先建立 E182 direct-main authority/environment contract test
  并确认 RED，再实现独立 authority/snapshot/probe/launcher；不修改 E181 历史结果。
- 已建立 S0 RED test，冻结了 27/3/24、`1024×32 seed0`、heldout
  `selection-forbidden→evaluation-only`、Ada 0/1 allowlist、可叠加且不 kill、
  依赖版本和 MuJoCo/MJWarp compile contract；实现模块尚未创建。
- RED 已确认：`test_preflight.py` 因缺少 `build_authority` 按预期失败。现已实现
  `e182_common.py` 与 `build_authority.py`：authority 使用三 split 目录，冻结
  CEM/heldout 合同，并对 E178 输入、E181 oracle/54 candidates 做不可变复制。
- 远程运行时审计：`spider-remote` 有 tmux/rsync/git、独立 run parent 可写，
  共享 `.venv` 的 trimesh/MuJoCo/MJWarp/Warp 版本匹配，但缺 `coacd`；共享 HEAD
  与 lock 也不同。按计划不修改共享 checkout，后续在 E182 独立 run root 部署
  本地 source/lock，并建立独立 dependency layer 后再判 source parity。
- `probe_environment.py` 已实现本地 GPU + Ada0/1 allowlist、authorized overlap、
  isolated-root lock/dependency parity 和 compile probes；direct-main S0 tests
  `3/3 PASS`。ruff 的 import/排序问题已机械修复，复查 PASS。
- 已追加 remote source snapshot TDD 合同：必须包含 lock/E182 code、生成稳定 SHA，
  且 remote root 只能位于 `/home/xiayb/pHRI_workspace/e182_runs/.../spider`；
  deployment 实现尚未创建，下一次 test 应为 RED。
- remote deployment RED 已按预期因模块缺失触发；现已实现 source file/SHA
  freeze、SHA-derived isolated root、无 `--delete` rsync、remote exact verifier
  与 root-local `.e182_deps` CoACD provision。已有完整 root 不一致时直接失败，
  不覆盖共享 checkout 或其他实验目录。
- source snapshot contract tests 已 GREEN，S0 direct-main=`4/4 PASS`；deployment
  lint 只发现 2 个集合/import 风格问题，已修复，尚未执行真实 remote deploy。
- 第一次 remote dependency provision 预检发现共享 `.venv` 无 pip，原定 remote
  pip install 不可用（attempt 1）。本地/远程均为 Python 3.12，CoACD 包由
  `coacd/ + coacd.libs/ + dist-info` 构成；改为从本地 lock 环境精确 rsync 到
  E182 `.e182_deps`，不重复 pip 方案、不修改共享 venv。
- 第二版 dependency provision 已实现：本地 CoACD distribution 逐文件 SHA
  manifest + rsync + 远端逐文件 verifier，manifest 最后写入作为 freeze marker；
  复用时只验证不覆盖。ruff/format 与 S0 tests `4/4 PASS`。
- 已增加 S0 completion audit TDD，要求 authority/deployment/environment 三份证据
  join、remote root/source SHA 一致且 overlap safety flags 全闭合；RED 已按预期
  因 `audit_preflight.py` 尚不存在触发。
- 已实现 Gate 0 audit、本地 preflight launcher 与 Ada0/1 remote launcher，并把
  Tracker/plan 切为 execution active。audit TDD 已 GREEN，S0 tests=`5/5 PASS`，
  ruff/format/bash syntax/diff-check 全部 PASS；下一步执行真实 preflight。
- S0 真实执行 PASS：authority=`27/3/24`、case snapshot files=`108`、E181
  candidates=`54`、source files=`5083`；remote root
  `e182_fbe65a69fa8a0b70` 逐文件 `0 mismatch`，独立 CoACD 10-file layer PASS。
- environment 与 completion audit PASS：本机0 + Ada0/1 allowlist/overlap、lock、
  dependency、source manifest 和 compile probes 全闭合，Gate 0=`13/13 PASS`；
  shared checkout / existing processes modified=`false/false`。下一阶段为 S1。
- S1 数据审计：E178 dev3 NPZ 有最终 qpos 与每 step reward/gate summary，但未保存
  1024 sample 的逐点 qpos/query tensor，不能从历史产物伪恢复 CEM query。
  dev3 三例 E178 均为 union SDF、同 body/hand/leg geom families。
- S1 执行路径冻结为 observational-only `64×4 seed0` shadow replay 两次；新增
  recorder 导出真实 sample 查询，同时用历史 ref/E178-final qpos 重放 pose 查询，
  并要求 recorder on/off reward、valid mask、selected index exact。
- 代码路径确认：E178 command 使用 `examples/run_mjwp.py` 与 legacy
  `sampling.py`（不是 `run_mjwp_fast.py`）；因此 S1 先对 slow rollout 增加默认
  off 的 recorder。fast path 后续仍需同合同，避免 production backend 分叉。
- recorder 设计冻结：slow rollout 在 GPU 上仅为 final CEM iteration 保留
  `N×H×nq` qpos；optimizer 选完 elite 后把 qpos、reward、G masks、selected index
  写成独立 chunk。默认 off；非 tape info 聚合排除 hidden payload，避免改变原输出。
- 已新增 recorder TDD：Config 默认 off、chunk schema/counter/SHA manifest；RED
  已按预期因 `spider.query_tape` 尚不存在触发。下一步实现 recorder 与 slow-path hook。
- recorder 首次实现 patch 因 `run_mjwp.py` 存在 SBTO/standard 两个 rollout call
  而上下文定位失败（attempt 1），apply_patch 原子拒绝、无部分落盘。改为拆分补丁，
  只修改 standard MPC slow call；SBTO 默认 off 路径保持原合同。
- 第二次已分拆落盘：新增默认-off Config 与 atomic NPZ recorder；slow rollout
  捕获 qpos，optimizer 仅保留/写出每 tick final iteration payload，并新增
  `cem_selected_index0` 供 on/off exact 对比。run_mjwp call-site 尚待定位复核。
- standard MPC call-site 已确认传入 `get_qpos`，recorder tests=`2/2 PASS`。一次
  全文件 ruff-format 对三个历史大文件产生机械格式 diff；后续只对新增文件全检，
  插桩文件使用语法/定向 lint，并单独审查 semantic diff，避免扩大格式 churn。
- 已建立 S1 replay/audit TDD：command 必须 exact E178 override/task + `64×4
  seed0`，on 模式必须写 case-scoped tape；NPZ comparison 必须逐 key exact。
  RED 已按预期因 runner/auditor 尚不存在触发。
- 会话恢复再次确认用户最新资源口径：Full 只用本机 GPU0 与
  `spider-remote` RTX 6000 Ada GPU0/1，允许同现有任务叠加；不得 kill、暂停、
  抢占或修改任何已有进程，也不使用 A100。当前仍停留在 S1 CPU 侧 lint/test，
  尚未启动 Full 或新增 GPU worker。
- S1 机械 lint 阻断已修复：仅对三个新增 replay/audit/test 文件执行 import/unused
  自动修复；未再格式化历史大文件。新增文件 ruff、插桩文件定向 ruff、py_compile、
  recorder `2/2`、pipeline `2/2` 与 S0 回归 `5/5` 全部 PASS。下一步是
  `bucket007_20231020_055_p1` 的 `off/on_a/on_b` 单 case shadow replay canary。
- canary 启动前本机只读审计：GPU0 为 RTX 5090，free=`28431 MiB`；已有 SUGAR
  PID `1139814` 占用 `3454 MiB`。该进程保持运行且不作任何修改；E182 canary
  将按用户授权叠加使用 GPU0。
- S1 `bucket007_20231020_055_p1/off` canary 已在本机 GPU0 启动并与既有任务
  共存；runner 将 stdout/stderr 写入独立 case log，前 31 秒仍在运行、前台无错误。
  未执行任何 kill/暂停/抢占操作。
- `off` canary 中途健康检查：已推进到 `88/166` sim steps，进入 CEM 后每次
  plan 约 `1.1–1.7s`；E182 进程显存约 `754 MiB`，既有 SUGAR 进程仍在。
  继续等待同一 run，不重复启动。
- `off` canary 完成 PASS：wall=`107.25s`、runner total=`101.08s`，输出 NPZ
  `457` keys、`qpos=(83,2,42)`，SHA=`7861cffcf35e...`；最终 object tracking
  error pos/quat=`0.1389/0.1175`。下一步启动同 case `on_a`，验证 recorder
  开启时轨迹及逐步汇总保持 exact。
- 同 case `on_a` recorder canary 已叠加启动；前 31 秒仍在运行且前台无错误，
  未触碰已有 GPU 进程。待完成后将先检查 chunk schema/count，再做 off/on exact
  比较，不因观察结果修改 E178 配置。
- `on_a` 中途健康检查：`84/166` sim steps 时已原子写出 `37` 个 chunk、约
  `19 MiB`；plan time 与 off 同量级（约 `1.1–1.6s`），未见 recorder 导致的
  明显 slowdown 或写盘错误。继续等待同一 run。
- `on_a` 完成 PASS：wall=`106.44s`，共写 `77` chunks、约 `40 MiB`，runtime
  与 off 基本一致。但初次 off/on exact audit FAIL：457-key schema 完全一致，
  `134` keys（含 qpos/ctrl/selected index）数值不一致；chunk manifest 终态仍为
  `RECORDING`。S1 Gate 尚未通过，下一步按冻结方案跑 `on_b`，区分底层 replay
  nondeterminism 与 recorder observer effect，并修复 manifest finalize 合同。
- `on_b` 已用相同 case/config/seed 在本机 GPU0 启动；前 31 秒运行正常，仍与
  既有任务叠加且未干预其进程。该 run 是预先设计的确定性诊断，不是重复失败实验。
- 并行代码审计确认 recorder 只在 enabled 分支分配 qpos tape、逐 step 调
  `get_qpos`，并在 final iteration 写盘；默认-off 分支未写 chunk。当前尚无证据
  能把差异归因给 recorder，因为两个独立 GPU CEM replay 自身也可能非 bitwise
  deterministic；必须等待 on_a/on_b 对照。manifest 保持 `RECORDING` 的直接原因
  是 writer 尚未实现 finalize API/call-site，后续以 TDD 修复。
- `on_b` 完成：wall=`107.06s`，chunk count=`77`。对照结论为 off/on_a 与
  on_a/on_b 均 FAIL，二者都约 `135/457` keys 不同；on_a/on_b 的 `77/77`
  chunks 也非 bitwise exact。差异规模在 recorder-on 的两次独立 replay 间同样
  存在，故当前证据不支持“recorder 导致轨迹改变”，而指向 GPU/MJWarp 跨进程
  non-bitwise determinism。S1 exact 合同需先量化首个分叉与误差尺度再修订。
- 差异量化：on_a/on_b 首个 CEM chunk 的 qpos max/mean abs diff 仅
  `7.1e-6/7.2e-8`、reward max diff `2.4e-6`、selected indices exact；随闭环
  累积到最后 chunk qpos max/mean `1.23e-2/9.49e-4`，selected 发生分叉。
  最终 on_a/on_b qpos max/mean diff=`1.11e-2/3.57e-4`；off/on_a 更大但单对样本
  不足以归因。`get_qpos` 是对 Warp qpos 的 zero-copy view，真正新增操作是复制到
  tape tensor，不修改仿真 state。下一步需以自然重放方差为基准定义等价性，而非
  不可能满足的跨进程 bitwise exact，并保留同-run payload 内部 exact 门。
- 为避免以单个 off/on 对照误判，已启动第二个独立 `off` baseline，使用同一冻结
  case/config/seed 和正式 local runner，但写入隔离的
  `s1_query_tape_diagnostic_off_repeat`。该 run 只估计 MJWarp 自然重放方差，
  不覆盖原结果、不参与 K/grid 选型，也未干预既有 GPU 任务。
- 第二个 off baseline 前 31 秒仍正常运行，未见前台错误；继续等待同一 run，
  不重复提交。
- 第二个 off baseline 已完成 PASS。下一步比较 off/off、off/on、on/on 三类差异；
  该证据将用于修订 S1 instrumentation gate：跨进程 CUDA bitwise exact 不再作为
  伪门槛，但 deterministic mock、default-off no-op、same-run payload consistency、
  chunk 完整性与冻结 SHA 仍必须 exact。
- 三类差异显示 recorder-on 会改变 CUDA 执行调度，不能宣称真实 backend 上
  bitwise observational：off/off qpos mean/max=`4.16e-4/1.44e-2`，on/on=
  `3.57e-4/1.11e-2`，而两组配对 off/on 约 `2.91–3.04e-3/7.24–7.36e-2`；
  reward/selected 也有同方向放大。因此 Gate 1 不能简单 tolerance-waive。
  但 recorder 仍生成一条真实 instrumented E178-config rollout，适合冻结为 query
  distribution；后续候选必须只在该 tape 上评估且用非 instrumented Full 作最终裁决。
- 备选“只录 controls 后离线重放”已审计：MJWarp `save_state` 使用单一 prev buffer，
  准确离线复现还需冻结 qpos/qvel/qacc/act/warmstart 等完整 state，复杂度和新的重放
  误差更高。v1 保留直接 qpos tape，但把证据改为 same-run exact integrity +
  default-off no-op + 明示 instrumented-distribution provenance，不再伪称跨进程 exact。
- 已开始按实证修订 plan 200 的 Query tape 合同：首个大补丁因 S1 原文一个词不匹配
  被 apply_patch 原子拒绝、无部分修改；随后先成功写入 frozen instrumentation 与
  CUDA non-bitwise 证据说明。S1 Gate/checklist 的结构化修订尚待继续。
- plan 200 的 S1/Gate1/checklist 已完成实证修订：硬门改为 deterministic mock
  exact、default-off no-op、same-run reward/gate/selected integrity exact、manifest
  COMPLETE 与 frozen SHA；真实 CUDA off/off、on/on、off/on divergence 作为必须报告
  的 provenance，不再错误要求跨进程 content SHA 相同。唯一 selection tape 冻结，
  诊断 replay 不参与 candidate 选择；production Full 强制 recorder-off。
- S1 修复继续采用 TDD：已先把 recorder test 扩展为 finalize/COMPLETE、ordered
  content SHA、provenance 与 idempotence 合同；实现函数尚不存在，下一步先确认
  RED，再实现并让 runner 对成功产物 finalize（包括安全修复现有 RECORDING manifest）。
- finalize test 已按预期 RED（ImportError）；现已实现 `finalize_cem_query_tape`：
  验证非空/连续 chunk、路径归属、size/SHA，生成 ordered content SHA 与 provenance，
  原子写 `COMPLETE`，重复 finalize 幂等；recorder 也拒绝向 COMPLETE tape 追加。
  尚待运行 GREEN 并接入 runner。
- finalize 单测与 ruff 已 GREEN。runner 已接入成功后 finalize，并可在发现
  “result 已完整 + manifest 仍 RECORDING”时只读验证后安全补全终态；返回行记录
  `tape_content_sha256`。尚待 lint 与对现有 on_a/on_b 产物执行幂等修复验证。
- runner/pipeline lint/tests 已 PASS；现有 on_a/on_b 经逐 chunk size/SHA 复核后
  安全补全为 `COMPLETE 77/77`，content SHA 分别 `ab73e0ee1b9b...` 与
  `7804dacc8bab...`。再次运行 on_a 得到 `SKIPPED_COMPLETE` 且 SHA 不变，证明
  finalize 幂等；未重新启动 GPU 任务。
- same-run 映射已在真实 on_a canary 验证：`opt_steps>0` 恰好对应 `77` 个 active
  MPC ticks 与 `77` chunks；每个 chunk 的 final-iteration reward max/min/median/mean、
  selected index，以及全部 1-D `sample_*` 的四种 summary 与 trajectory NPZ
  逐项 bitwise exact，failures=`0`。该检查可作为修订后 Gate1 的核心硬证据。
- 已为 same-run auditor 添加 synthetic PASS + selected-index tamper FAIL 的 TDD；
  RED 按预期因 `audit_same_run_integrity` 尚不存在触发。下一步实现公共 audit 函数，
  再把 auditor main 从“跨进程 exact 决定 PASS”改为“same-run exact 决定 PASS，
  cross-run divergence 只报告”。
- `audit_same_run_integrity` 已实现：检查 COMPLETE/content SHA、active tick↔chunk
  count、连续 index、文件 SHA、qpos/reward/sample 非有限值，并 exact join 每个 final
  iteration 的 reward、selected index 与全部 1-D sample summaries。尚待 lint/test
  GREEN；auditor main 的 verdict 逻辑仍需修订。
- same-run auditor lint 与 synthetic `3/3` tests 已 GREEN；真实 bucket007 canary
  的 on_a/on_b 均 PASS：各 `77` chunks、mismatch=`0`、nonfinite=`0`，content SHA
  分别与 finalized manifest 一致。修订后的核心 Gate1 证据成立；下一步更新 auditor
  main，使 cross-run exact FAIL 只成为定量 diagnostic，不再错误压低 gate verdict。
- auditor main 已按 plan 修订并通过 lint/tests：hard gate 仅 join on_a/on_b same-run
  integrity；off/on、on/on 与 chunk bitwise divergence 仍完整量化在 report-only 区。
  bucket007 单 case 正式 audit=`PASS`，不再把已知 CUDA non-bitwise 当成虚假失败。
- deterministic mock test 的依赖已审计：可复用 `make_rollout_fn` 与默认关闭全部
  CEM/smooth/E167/foot gates 的 `Config`，在 CPU mock state 上只切换 tape enabled，
  比较 controls/reward/terminate/可见 info exact，并断言 off 无 hidden qpos payload。
  下一步先写 RED test，再实现/验证合同；不需新增生产配置。
- deterministic CPU mock rollout test 已实现并直接 GREEN：on/off controls、reward、
  terminate 与全部可见 info bitwise exact；off 不含 hidden qpos，on 的 tape shape
  为 `(2,4,3)`。recorder suite 现为 `3/3 PASS`，满足 Gate1 deterministic mock 与
  default-off no-op 的代码级合同。
- git 提交前审计：修改范围为 recorder 三个公共插桩文件、E182 plan/tracker/progress、
  新增 `spider/query_tape.py`、E182 experiment/launch scripts；运行结果目录未出现在
  status（仍按结果规则独立保存）。`git diff --check` PASS。先前误用 ruff-format
  仍给 `run_mjwp/config/sampling` 带来非语义排版噪声（43/22、37/36、82/48），
  语义 diff 已确认集中于 query tape；提交前仍需决定是否机械清噪或保留 checkpoint。
- 为无损清理排版噪声，已在 `/tmp/e182_clean.DBVNf4` 建立三个 HEAD 基线副本，计划
  只重放 recorder 语义 hunk，再用 apply_patch 更新工作树。首次合并 temp patch 因
  sampling 最后一处上下文换行不匹配而原子拒绝，无文件被部分修改；将拆分补丁继续，
  不使用 checkout/reset，也不覆盖用户修改。
- temp clean copy 的 config/run_mjwp 全部语义 hunk及 sampling 除 final-write 外的
  recorder hunk 已用 apply_patch 重放成功；仍只位于 `/tmp`，工作树尚未替换。
  下一步补 final payload write，逐文件 AST/语义 diff 校验后再生成 apply_patch。
- 三个 temp clean copy 均 AST parse PASS；sampling 最终插入点已精确定位在 early
  stopping 之后、fake-info padding 之前。继续用 apply_patch 补齐，不改变算法顺序。
- sampling final payload write 已补齐；当前与 temp clean 三文件的 AST dump 全部
  exact，证明 clean copy 保留全部语义。首次自动生成的 workspace apply_patch 因
  标准 diff 行号 hunk header 不被 apply_patch 方言接受而原子失败、工作树未变；
  下一次会把 header 规范化为 `@@` 后重试，不改变 patch 内容。
- 规范化 patch 已成功落回工作树，三文件与 clean copy AST 仍 exact；格式噪声从
  `43/22,37/36,82/48` 收敛为真实语义 `1/0,5/0,48/1`。全套 ruff、定向 lint、
  py_compile、recorder `3/3`、pipeline `3/3`、preflight `5/5` 与 diff-check
  全部 PASS。提交前代码审计现已干净。
