# CORE4D 当前进度

> 完整备份：
> [E167–E179 初始恢复](progress_archive/E167_E179_20260724_full_backup.md) ·
> [E179 执行过程](progress_archive/E179_20260724_execution_full_backup.md) ·
> [E179 最终收尾](progress_archive/E179_20260725_closeout_full_backup.md) ·
> [E180–E181 完整执行](progress_archive/E180_E181_20260731_full_backup.md) ·
> [E182 Gate0–S1 canary](progress_archive/E182_gate0_s1_canary_20260801_full_backup.md)
>
> 本文件只保留 E182 当前可靠结论与下一执行入口。

## 2026-08-01：E182 资源与不可变口径

- 计划：`plan/200_E182_task_conditioned_coacd_full_cem_plan.md`；Tracker 为
  `✅ Gate0 PASS；🚧 S1 query tape`。
- Full authority 与 E178 exact：27 case、`1024×32 seed0`；K 只测
  `8/16/32`，不测 K4。
- heldout24 在 production SHA 冻结前 selection-forbidden，冻结后仅
  evaluation-only；不得按 heldout/Full 结果反选碰撞体。
- 资源只用本机 GPU0 + `spider-remote` RTX 6000 Ada GPU0/1；三卡允许与
  已有任务叠加。禁止 kill、暂停、抢占或修改已有进程，不使用 A100。

## 2026-08-01：S0 Gate0 PASS

- authority=`27/3/24`，case snapshot files=`108`，E181 candidates=`54`，
  source snapshot files=`5083`。
- remote frozen root：
  `/home/xiayb/pHRI_workspace/e182_runs/e182_fbe65a69fa8a0b70/spider`。
- remote source verification=`0 mismatch`；独立 CoACD dependency layer
  `10/10 SHA PASS`，未修改 shared checkout/venv/现有进程。
- completion audit=`13/13 PASS`；direct-main preflight tests=`5/5 PASS`。

## 2026-08-01：S1 query tape canary

- E178 历史 NPZ 不含 1024 samples 的原始 qpos/query tensor，不能伪恢复；
  dev3 使用 exact E178 override/task/seed 做 `64×4 seed0` shadow replay。
- 已实现默认-off recorder、chunked qpos/reward/G payload、runner、finalizer、
  same-run auditor 与 direct-main tests。公共代码只增加真实语义 diff：
  `run_mjwp +1`、`config +5`、`sampling +48/-1`；误格式化噪声已清理。
- bucket007 canary：off/on_a/on_b wall=`107.25/106.44/107.06s`；on_a/on_b
  各 `77` chunks、约 `40MiB`，manifest 均 `COMPLETE`。
- frozen content SHA：on_a=`ab73e0ee1b9b...`，on_b=`7804dacc8bab...`；
  finalize 幂等，重复 runner 为 `SKIPPED_COMPLETE`。
- same-run exact audit：on_a/on_b 均 PASS，mismatch/nonfinite=`0/0`；
  reward max/min/median/mean、selected index 与全部 1-D sample summaries exact。
- deterministic CPU mock on/off exact，default-off 无 hidden qpos；recorder tests
  `3/3`、pipeline `3/3`、preflight `5/5`、ruff/py_compile/diff-check PASS。
- 真实 MJWarp 跨进程不 bitwise：off/off 与 on/on 均自然分叉，qpos capture 还会
  改变 CUDA 调度并放大闭环差异。plan Gate1 已实证修订：跨 run divergence
  report-only；硬门为 mock/default-off/same-run/COMPLETE/frozen SHA。
- 唯一 selection tape 使用 on_a；on_b 只作诊断。recorder-on shadow 不能宣称
  downstream 改善；production Full 强制 recorder-off。

## 下一执行入口

0. progress 全文已归档到
   `progress_archive/E182_gate0_s1_canary_20260801_full_backup.md`（283 行）；活跃文件
   已压缩为 63 行。archive/链接、bash syntax、无 kill 命令与 diff-check 均 PASS。
1. git checkpoint 已创建：`exp(core4d): E182 Gate0 and query-tape canary`；未 push。
2. 对 bucket003/bucket004 跑 off/on_a/on_b，并完成 dev3 same-run audit。
3. 构建 ref、E178-final、CEM sample 的 object-local P/R/G query tape 与 provenance。
4. Gate1 全量 PASS 后进入 S2：在冻结 dev3 tape 上审计 K8/K16/K32，不读取 heldout。
5. production K/grid/config SHA 冻结后，才启动本机0 + Ada0/1 的 27-case Full；
   launcher 只叠加运行，绝不处理现有进程。

## 2026-08-01：goal continuation 恢复

- 已重新完整读取 `experiment-planning-zh`，并从 Tracker/progress 恢复 E182：当前
  仍为 Gate0 PASS、S1 进行中；资源/heldout/K/Full 不可变口径未变。最新 plan、
  E181/E178 logs 与 remote-execution 规则正在继续复核，尚未启动新 GPU run。
- 已完整复核 plan 200 与最新 E181/E178 logs：E181 的 54/54 build、0/18 旧 cavity
  reject 和 E178 27-case/10-of-27 baseline 均与 E182 authority一致。发现 plan 风险表
  仍残留旧的“on/off mismatch 就停止”措辞，与已修订 Gate1 的 same-run hard gate
  不一致；需在下一代码 patch 一并更正，避免后续 audit 双重口径。
- 已完整读取 remote-execution 规则并只采用双 Ada profile；A100 章节不适用于 E182。
  当前本机 free=`28451MiB`、既有 SUGAR=`3434MiB`；Ada0/1 free=
  `39776/40892MiB`，两卡均有既有 SUGAR。三卡符合用户授权的叠加状态，未 kill/
  暂停/抢占。现有 launcher 的 query/full 仍是 placeholder；S1 仅 bucket007 完成，
  bucket003/004 尚无 replay，因此下一步先补正式 query-tape 入口与并行执行合同。
- remote isolated root 审计发现旧 S0 snapshot 是 recorder 实现前的
  `e182_fbe65a69fa8a0b70`，缺当前 runner；且 ignored 的 E182 authority 与 E178
  manifest 本来就不在 git source snapshot。不能直接在该 root 裸跑。下一步必须
  建立新的 immutable source root，并显式冻结/同步 S1 runtime input manifest，
  不能借用或修改 remote shared checkout。
- dev3 authority 逐行复核确认 bucket003/004/007 的 scene、trajectory、E178
  scene_act 与 E174 contact mask 全部是 git-ignored runtime input；远程 source
  snapshot不会自动包含。一次只读审计错误地尝试
  `authority/authority_manifest.json`（实际 manifest 在 `s0_environment/`），记录为
  attempt1，不重复该路径。需新增 runtime-input SHA inventory + scoped rsync，并验证
  XML 引用资产闭包，之后才能安全启动 Ada S1。
- runtime closure 审计确认每个 dev case 目录约 10 个文件，scene 引用 tracked 的
  `spider/assets/robots/.../meshes` 与 ignored 的 object OBJ；config 另依赖 E174 mask。
  S1 remote input 应包含三个完整 task dir、bucket003/004/007 object asset dir、三份
  mask、E182 dev3/E178 authority，以及三份 E178 final NPZ/config（供后续 final-pose
  tape），均按原 repo-relative 路径同步。禁止依赖 config_act 中历史 A100绝对路径；
  fresh Hydra override 会从 isolated root 重建 data/model path。

## 2026-08-01：Full 资源口径再次确认

- 用户再次确认：E182 Full 不使用 A100，资源固定为本机 GPU0 + 远程 RTX 6000 Ada
  GPU0/1。三路 Full 均可与正在运行的程序叠加启动；不得 kill、暂停、抢占或修改
  任何已有进程。计划与 launcher 审计均以此为硬约束。
- 已复核 plan 的 S6、Full 命令与 local/remote launcher：执行设备均为本机 GPU0 +
  Ada GPU0/1，没有 A100 执行入口。计划顶部补充“不使用 A100”；风险表旧 Gate1
  口径已改为 same-run/default-off/完整性硬门，跨进程 divergence 仅报告。
- 一致性检查：local/remote launcher `bash -n` PASS，`git diff --check` PASS；计划中
  的 `pull_E182_remote_a6000_results.sh` 目前尚未实现，故记录为 S6 前待办，不宣称
  Full/pull 入口已闭合。本轮没有启动、停止或修改任何 GPU process。

## 2026-08-01：持续执行恢复

- 已按 `experiment-planning-zh` 恢复持续目标并建立 S1→S7 工作计划。当前先以磁盘
  与远程实际状态为准重新审计 S1，不沿用未经复核的进度声明；资源安全约束保持
  本机 GPU0 + Ada0/1 叠加、禁止干预已有进程。
- 初步磁盘审计：Tracker 仍是 `Gate0 PASS；S1 query tape`；工作树只有 plan 与
  progress 的本轮修改。E182 已有 authority/preflight/query-tape 实现和 bucket007
  artifact，但 remote runtime-input builder、正式 pull 入口及 dev3 Gate1 闭合仍需
  逐项验证；未发现额外 workspace `AGENTS.md`。
- 已重读双 Ada remote-execution 规则及 E181/E178 最近结果：E181 的失败是旧全局
  cavity 门，不是 CoACD build 失败；E182 仍应按真实 P/R/G tape 推进。E178 Full
  authority 保持 27/27、numeric 10/27、人工已审 23/27，作为 paired baseline 只读。
- S1 代码审计确认两个明确缺口：local/remote launcher 的 `query-tape` 仍是
  placeholder；`run_query_tape_replay.py` 对每个 mode 固定写同一
  `run_manifest.json`，Ada0/Ada1 并发单 case 会覆盖。下一步先以 direct-main TDD
  固化 runtime-input SHA/拒绝覆写与 case-scoped manifest，再实现 launcher/pull。
- 进一步确认 source deploy 只冻结 tracked + nonignored 文件并用 SHA 派生隔离 root，
  因而不会携带 git-ignored dev scene/object/mask/E178 result。runtime input 应作为与
  source snapshot 分离的不可变 manifest/rsync/verify 层接入同一 isolated root，不能
  修改 remote shared checkout；现有 source verifier和 dependency layer可复用。
- 遇到只读审计错误 attempt1：错误查询裸 case 目录
  `example_datasets/processed/CORE4D_Real/<case>`，且用固定列号把 trajectory 当成
  `config_act`；未产生写入。已改为按 manifest header 名解析，确认三个 task dir
  分别含 10/16/10 个文件，E178 `result_npz/config_act/override_path` 均存在明确字段。
- 已先写 `test_runtime_inputs.py`，覆盖真实 dev3 closure、逐文件 SHA verifier、冻结
  manifest identity 拒绝替换，以及 bucket003/bucket004 case-scoped replay manifest
  不碰撞。当前故意处于 RED（`runtime_inputs.py` 与 `case_manifest_path` 尚未实现），
  下一步执行测试保留失败证据后进入 GREEN。
- TDD RED 已实测：`ImportError: case_manifest_path`。现已新增 `runtime_inputs.py`：
  从 dev3/E178 header 字段构建 closure、校验 trajectory/mask authority SHA、逐文件
  freeze/verify，并在远程先做 allow-missing 冲突审计、用 `--ignore-existing` 部署、
  拒绝 frozen manifest identity 变化；runner 改为单 case 独立 manifest。
- GREEN functional：runtime-input tests `3/3`、query recorder `3/3`、pipeline
  `3/3`、preflight `5/5` PASS。静态检查 attempt1 仅失败于新文件 import 排序、一个
  unused import 和两文件 format-check；功能测试未失败。下一步用 formatter/ruff
  mechanical fix 后复跑，不重复未修复的检查。
- 静态问题已 mechanical fix；ruff/format/diff-check 与 runtime/pipeline tests 全 PASS。
  正式 runtime closure 已冻结并本地 exact verify：`50` files、约 `33MiB`、3 case、
  SHA=`25526528fe6a7fe721dfc1c4b7f91746959d2779eac7e0784a595d6fd1996123`；含
  三 task tree、三 object OBJ、三 mask、三 E178 NPZ/config 与两份 authority manifest。
- 第二轮 TDD RED 已实测两处预期失败：搬迁后的 chunk manifest 因旧绝对路径导致
  same-run audit FAIL；S1 worker/pull 脚本不存在。测试同时锁定 remote bucket003→Ada0、
  bucket004→Ada1、local bucket007、逻辑 `cuda:0`、tmux new-only、pull no-delete/
  ignore-existing 和无 kill 命令合同。
- 已实现 relocated chunk sibling+SHA resolver、独立 query worker、local bucket007
  query stage、remote source/runtime deploy + 当前 GPU 快照/16GiB floor + Ada0/1 tmux
  new-only launch，以及 query pull/audit。remote worker 同卡严格串行 off/on_a/on_b；
  launcher 遇到同名 session 只报告并保持不变，不操作任何既有 process。
- 第二轮 GREEN：relocated pipeline `4/4`、runtime/launcher `4/4` PASS，四个 shell
  `bash -n` PASS；首次静态检查只剩 import 排序/format，mechanical fix 后 ruff、
  format-check、两组测试、`git diff --check` 全 PASS。尚未启动远程 GPU run。
- 正式 local `query-tape` 入口已验证 resume-safe：bucket007 三 mode 均
  `SKIPPED_COMPLETE`，未重跑 GPU；新 case-scoped manifests 保留原 result/content
  SHA。same-run audit 再次 PASS，on_a/on_b 均 `77` chunks、mismatch/nonfinite=`0/0`。
- 远程启动前只读快照：Ada0/1 free=`39776/40892MiB`，已有 SUGAR PID 44386/44387
  保持运行；未处理任何 tmux/process。新 source root
  `e182_832e7fe0f61193a1`（5093 files）与 CoACD layer、50-file runtime layer均远程
  exact PASS。已叠加启动 Ada0 bucket003、Ada1 bucket004 两个 new-only tmux session；
  session=`e182_s1_ada{0,1}_832e7fe0f61193a1`，未使用 A100。
- 首次只读监控：两 session 和对应 worker bash 均存活，GPU used 从启动前
  `8725/7618` 增至 `9399/8270MiB`；outer log 暂无完成行、artifact 尚未生成，符合
  首个 off replay 初始化阶段。未发信号、未重启、未改变已有 SUGAR。
- 两个 off replay 已进入真实 rollout，无 path/config 错误；Ada0/1 utilization
  `98/97%`，E182 增量显存约 `674/652MiB`。共存下单 active step plan time约
  `3.1–3.5s`，明显慢于本机 bucket007 canary，S1 预计需更久但显存安全。并行期间
  已开始定位从 qpos 生成 object-local P/R/G consumer query 的现有 MJWP接口。
- 监控时 off 进度 bucket003=`74/416`、bucket004=`72/290`。代码复核确认 R/G 的
  authoritative consumer 集合由 `_object_sdf_geom_groups_for_tick(config)` 决定，查询点
  是 primitive capsule 的 `{-half,0,+half}` 轴点减 radius，以及 mesh geom 最多800个
  vertices；object union transform 来自每个 qpos 的 geom pose。可据此离线重建真实点集。
- off 继续至 bucket003=`112/416`、bucket004=`110/290`。E178 `config_act.yaml` 已
  含 resolved object/robot consumer geom IDs，可直接冻结 R 的 robot/leg/surface-band
  与 G 的 safety/hand/leg group membership；reference qpos 是 `(T,43)` 而 E178/CEM
  scene nq=`42`，必须沿用 E178 既有 person/object slicing，不能直接盲赋整行。
- 可避免重复实现 43→42 conversion：E178 result NPZ 本身保存 physics qpos
  `(record,ctrl_step,42)`，可作为 final pose source；reference 可通过同一已冻结
  run config 的 `load_data` conversion 或从每条 record 的对应 ref 时刻重建。当前 final
  NPZ不含 `qpos_ref`，因此 reference provenance 仍需显式实现/验证，不能伪称已捕获。
- 已验证可从 frozen `config_act` 通过 `Config(**filter_config_fields(...)) + load_data`
  精确恢复插值前转换的 ref：bucket007 得到 `(216,43)`、`ref_steps=2`，而
  `216-horizon48-ctrl2=166` 正好等于 E178 final qpos flatten 长度。随后复用 E027b
  43→42 body-frame slide/euler公式即可获得实际 optimizer reference pose序列。
- 第三轮 TDD RED：`test_prg_query_tape.py` 因 builder 不存在按预期
  `ModuleNotFoundError`。测试冻结 bucket007 ref=`216×42`、final=`166×42`、真实
  P/R/G consumer schema、object-local finite points，以及 ref/final/单 on_a chunk
  三源 manifest SHA；不读取 heldout24。
- 第三轮 GREEN：新增 `build_prg_query_tape.py`，复用 E027b 43→42 conversion、
  config-resolved consumer IDs 和 MJWP axis3/radius/mesh≤800 点合同；P 保留完整 qpos
  作 full-scene replay，R/G 输出 object-body-local points/masks。bucket007 ref/final/
  单 CEM chunk 实测 `3/3 PASS`，无 heldout 读取。
- PRG builder 首次静态检查仅 import/format 问题，mechanical fix 后 ruff/format/test/
  diff-check 全 PASS。远程进度：bucket004 off 已完成并进入 on_a `40/290`；bucket003
  off=`338/416`。两 worker 连续推进，无 OOM/NaN/重启，已有任务未受处置。
- 存储审计发现当前展开式 tape 为 `1669 points/pose`，预计约 `4.7GB/case`，不适合作为
  v1 canonical artifact。决定改成无损 factored geom-pose + fixed-offset 表示，P 仅保留
  full-scene qpos。首次测试 patch 因 formatter 改行导致 context mismatch，未改文件；
  下一次按当前行号精确 patch，不重复原 patch。
- 已按当前文件精确更新 PRG test：要求 artifact 不存 `points_object_local` 展开阵列，
  而存 object-local geom pos/mat + fixed offsets/radius/mask，并用
  `materialize_query_points` 证明坐标可精确恢复；manifest 必须报告展开字节数大于实际
  stored size。当前进入第二个 RED，builder 尚未提供该 API。
- factored RED 已实测为缺少 `materialize_query_points`。首次 builder patch attempt1
  因 formatter 后函数签名/换行 context mismatch 未应用，未改变实现；下一步拆成小块
  按实际代码精确修改，避免再次提交同一大 patch。
- factored implementation 已完成第一小块：P_collision 仍保留完整 geom inventory 供
  physics replay provenance，但不进入 R/G point expansion；固定点表新增 query-geom
  column 映射与 consumer mask，context 记录唯一 R/G geom IDs。下一块改每 pose
  存 geom pos/mat 并补 materializer。
- factored 第二小块完成并 GREEN `3/3`：每 pose 只存 object-local query geom
  pos/mat，固定 offsets/radius/consumer mask 每 chunk 一份；materializer 在测试中恢复
  finite exact-coordinate tensor。manifest 报告 factored contract、stored bytes 与假设
  展开 bytes，P 继续由原 qpos full-scene replay。
- 审计发现 P geom inventory 当前为 `0`：E178 scene 使用 explicit `<pair>`，robot geoms
  可同时是 contype/conaffinity=0，按 mask 过滤会漏掉 P provenance。已新增 RED 断言
  P collision geom count>0；修复应从 `pair_geom1/2` 与 object subtree 的交集解析，不能
  放松成所有 visual geom。R/G 仍为25 query geoms、1669 fixed points。
- P RED 已实测 AssertionError。实现现以 object-body subtree 枚举 object geoms，遍历
  model explicit pair 的两端收集对侧 robot geom，并与正常 collision-mask geoms取并集；
  world/object subtree仍排除。待复跑确认应恢复 E178 的18个 robot-object pair geoms。
- P 修复 GREEN：三 dev case 均恢复 `18` 个 physics robot geoms，与 E178
  expected pair count `18×5=90` 对齐；R/G=`25` unique geoms、`1669` fixed points。
  ruff/format/PRG tests/diff-check PASS。远程 on_a 进度 bucket003=`158/416`、
  bucket004=`258/290`，session均存活。
- bucket007 完整 factored PRG tape 已构建成功：reference + E178-final + 全部 on_a
  chunks，wall≈`21.8s`。下一步读取 manifest 校验 79-chunk SHA、实际 stored/expanded
  ratio 与 source content SHA；尚不把单行 PASS扩大宣称为 dev3 Gate1。
- bucket007 manifest 实测 `79` chunks、SHA bad=`0`、stored=`352.8MiB`，相对展开
  `4.42GiB` 缩小 `12.83×`；raw selection content SHA exact=`ab73e0ee...`。已新增
  verifier RED 测试，要求正常 manifest PASS 且任一 chunk append tamper 精确 FAIL，
  builder 尚未提供 `verify_case_query_tape`。
- verifier RED 已实测 ImportError；现已实现 chunk/source SHA、required array keys、
  qpos/geom transform finite、pose-axis shape、chunk count 与 stored-size 完整审计。测试会
  对临时首 chunk append tamper，确保不是“只看 manifest status”的弱验证。
- verifier attempt1 的真实 artifact `79/79 PASS`；tamper 测试因 append 同时触发
  chunk SHA 与 aggregate stored-size 两项（不是预期的恰好1项）而断言失败。已修测试为
  `mismatch_count>=1` 且必须含 `sha256`，保留更强双重检测，不削弱 verifier。
- verifier/test/static 全 GREEN；bucket007 real tape仍 `79/79 PASS`。远程：bucket004
  on_a完成并进入 on_b=`84/290`；bucket003 on_a=`272/416`。off/on_a tracking summary
  接近但不作 exact gate，符合 CUDA divergence report-only 口径。

## 2026-08-01：资源计划复核（当前轮）

- 按用户最新要求继续采用本机 GPU0 + `spider-remote` RTX 6000 Ada GPU0/1，明确
  不使用 A100。三路 Full CEM 均允许与正在运行的程序叠加；禁止 kill、暂停、抢占
  或修改任何已有进程。本轮仅复核计划与记录，尚未对运行中任务执行任何操作。
- 已完整复核 E182 plan：S6 worker 表、Full 固化命令、GPU allowlist、显存不足/OOM
  处置和 checklist 均采用上述口径；`A100` 仅出现在“禁止使用/历史数值不可套用”的
  说明中，不存在 A100 执行入口。当前 launcher 搜索也未发现 kill/pkill/tmux-kill。
- 最新 E181 visual log 与 Tracker 已复核：E182 仍为 `Gate0 PASS；S1 query tape`，
  E181 的 `0/18` 是旧全局 cavity gate reject，不改变当前按真实 P/R/G 查询推进的
  E182 路线，也不引入任何 A100 资源。

## 2026-08-01：持续目标恢复（本轮）

- 已重新完整读取 `experiment-planning-zh` 并建立 S1→S7 活跃计划。当前只读恢复
  工作树和双 Ada 现场，先闭合 dev3 Gate1；资源继续固定为本机 GPU0 + Ada0/1
  共存执行，禁止干预已有进程，Gate1 前不读取 heldout candidate evidence。
- 已完整读取 Tracker 与当前 progress；权威状态仍为 E182 Gate0 PASS、S1 未闭合，
  E178 `27-case/1024×32 seed0`、K8/16/32、heldout 隔离和双 Ada 资源合同均未改变。
- 已完整读取 plan 200 与最新 log 246。S1 Gate1 的硬门是 mock/default-off/
  same-run/chunk+SHA 完整性；跨进程 CUDA divergence 仅报告。S2 只能在 Gate1 后用
  dev3 tape 评估 54 个既有 candidate，E181 broader-cavity 只保留诊断意义。
- 远程现场只读复核：Ada0/1 的 E182 tmux 均存活，GPU util=`98/97%`，显存仍安全；
  bucket003/004 的 `off` 与 `on_a` 已各自 PASS，worker 正在自然推进 `on_b`，没有
  发信号或改动任何进程。一次 chunk 计数 attempt1 使用了错误的 case 子目录层级而
  返回 0，不能作为进度证据；下一次按 manifest 的 `chunk_paths`/实际目录统计。
- 按实际 `raw_chunks/<mode>/<case>` 复核：bucket004 on_b 已有 `140` files，疑似已
  完成；bucket003 on_b 当前 `38` files，仍在推进；bucket003/004 on_a 分别为
  `203/140` files。一次 manifest 摘要 attempt2 调用了远程不存在的裸 `python`，仅
  摘要读取失败、未影响任务；后续使用冻结 `REMOTE_PYTHON` 或本地拉回后审计。
- S2 代码审计：E181 可复用 negative-inside Open3D `D_M`、convex halfspace occupancy
  与 ordered-part SHA，但 E182 尚无 `evaluate_task_queries.py`。新 evaluator 必须先用
  合成相交/分离凸体验证 union SDF 符号和距离；不能把 `min(part SDF)` 冒充重叠凸体
  union 的 exact inside distance。Gate1 前仅做该合成合同，不读取真实 candidate score。
- 环境已确认 `trimesh 4.11.5 + manifold3d` 可用；两个重叠 box 的 boolean union 得到
  watertight/winding-consistent mesh、正确体积 `12`，中心到 union 外边界距离为 `1m`。
  S2 exact-C 将用 manifold boolean union + Open3D negative-inside SDF，并记录 engine/
  mesh provenance；这能覆盖 overlap 内部面被错误计距的问题。
- 已先新增 S2 synthetic direct-main tests：锁定重叠 parts 的真实 union boundary、
  negative-inside 距离、query radius 后的 sign/false-accept/false-reject/deep mismatch，
  以及 shape/non-finite fail-loud 合同。尚未实现 evaluator，下一步运行取得预期 RED。
- S2 synthetic RED 已实测为预期 `ModuleNotFoundError: evaluate_task_queries`。现已实现
  manifold exact boolean union、Open3D negative-inside query 与 radius-aware paired
  metrics；非 finite 点显式计入 coverage/status，不静默过滤。下一步执行 GREEN 与
  静态检查，真实 candidate 仍未读取。
- S2 geometry synthetic `3/3 PASS`。首次检查功能/ruff/diff-check 已通过，仅
  `evaluate_task_queries.py` format-check 失败；已做 mechanical format 后复跑，测试、
  ruff、format-check、diff-check 全 GREEN。该阶段只验证合成 box，没有 selection 泄漏。
- 远程复核：bucket004 worker 已自然 `PASS` 并退出，on_a/on_b=`139/139` chunks；
  bucket003 on_b=`122/202`，Ada0 session 继续运行。pull 脚本为 no-delete +
  `--ignore-existing`，拉回后会执行三 dev case same-run audit；但 Gate1 还需把 PRG tape
  verifier 与 mock/default-off 测试证据聚合，不能仅凭 remote worker PASS 宣称闭合。
- Gate1 mock evidence 由 `test_query_tape.py` 明确覆盖 config default-off、recorder schema/
  finalize 与 deterministic backend on/off exact；pipeline tests 覆盖 same-run summaries 和
  relocated pull。下一步先全量复跑现有测试，pull 后再把三 case PRG verifier 与这些
  test exits/SHA 聚合成单一 Gate1 artifact。
- 当前全量检查 GREEN：query recorder `3/3`、pipeline `4/4`、runtime/launcher `4/4`、
  PRG tape `3/3`、preflight `5/5`、S2 synthetic `3/3`，以及 E182 ruff/format、四个
  launcher `bash -n`、diff-check 全 PASS。期间 bucket003 on_b 从 `122→149/202`，
  Ada0 util=`98%`、显存安全，原任务未被干预。
