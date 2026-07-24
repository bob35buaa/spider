# CORE4D 当前进度

> 历史完整备份：
> [E167–E179（截至 2026-07-24）](progress_archive/E167_E179_20260724_full_backup.md)
>
> 本文件只保留当前活跃实验的最终状态与未决项。

## E179：box023 / E167A no-PRG / Full CEM

### 已冻结

- 正式计划：
  `plan/197_E179_box023_e167a_no_prg_full_cem_plan.md`。
- 数据 authority 为 E173 box023 的全部 16 条 CEM-eligible case；E179 与 E173
  必须逐条 paired comparison。
- treatment 为 `E167A_zOnlyBody`；保留 E167A hand/posture/surface/z-only，
  排除 E170 PRG 的 lower-body pairs、penalty、candidate gate 与 fallback。
- Full CEM 固定 seed `0`、`1024 samples × 32 opt steps`。
- 资源固定为本地 RTX 5090 GPU0（4 条）与远程 A100 GPU
  `2,3,6,7`（各 3 条）；正式启动前 GPU/显存/进程/预约检查须连续通过两次。
- 主指标为 12-gate intersection：原六个 physics gate，加 root、EEF、object
  的位置/姿态 tracking gate；missing/non-finite 一律 FAIL。
- E173 只读 baseline：legacy physics `13/16`，同一 adapter 重算 12-gate
  为 `7/16`；scoring contract 为 `core4d-e179-12gate-tracking-v1`。
- 已记录 E173 Full manifest SHA256
  `2128deb8403a0efddb7578a46b2ef466811f91ba183d955b0230776de2ae889d`
  与 E167A profile SHA256
  `666c302dcf549e517ff02b50c139551cf98635b7b951872284d6a39766548c17`。

### 2026-07-24 执行恢复

- 用户已明确授权按 E179 plan 实施；已完整重读 `experiment-planning-zh`。
- 已读取当前 Tracker、E179 全计划和 log 索引；Tracker 顶部仍为 E179
  planning-only，最近执行日志为 E178 `237–241`，E173 authority/result 见
  `log/233_E173_box024_box023_box001_screening_full_cem.md`。
- 可复用实现已定位：E173 authority/PRG manifest/queue/render/audit，E168
  rubber-hull handoff 与 E167A config audit，E170 PRG manifest/queue，以及
  E178 tracking-gate adapter；E179 目录当前不存在，需新建且不得回写 E173。
- E173 真实产物齐全：box023 16 条 override、Full NPZ 和 scene snapshot 均在
  `results/E173/`；公共场景快照入口为 `scripts/convert/snapshot_scenes.sh`。
- 实现审查确认：E173 builder 已把 rubber-hull 与 PRG 16 pair/penalty/gate
  耦合；E179 不能直接 fork 其 `base_payload`/runtime validator。E169 queue
  runner 也强制 PRG diagnostics，E179 需独立 no-PRG runner。
- E168 `audit_e167a_cem_configs.py` 可复用 profile/route 审计思路，但其 40-row
  与固定 scene 名硬编码不适用；E179 将实现 16-row、scene 可配置的严格版本。
- E167A frozen profile authority 已定位到
  `results/E168/s0_environment/e167a_profile/e167a_zonly_profile.json`，其中
  `profile_sha256` 与计划的 `666c…48c17` 一致。
- 当前 E173 manifest 文件 SHA 已现场复算为计划冻结的 `2128…889d`；box023
  恰为 16 条、`omnirt_v1/v2=15/1`，唯一 v2 为 `20231011_018_p2`。
- 评测实现已定位：E173 evaluator 为 `eval_E173_boxes.py`（首次按旧猜测路径
  `eval_E173_pipeline.py` 读取失败，未改文件）；E178 通过 E176 evaluator 的
  opt-in tracking gates 实现 12 门。
- E179 evaluator 将直接 import `eval.core.core_metrics`，复用公共 raw metrics，
  但独立固定 scoring contract、12 门、E173 baseline adapter 和 16-row join；
  不动态加载其他实验 evaluator。
- no-PRG override 路径已确认很窄：每条 E173 dcv3 override 已继承冻结 E167A；
  E179 只需继承该 override 并改 `scene_name`，不能加入任何 PRG 字段。
- 发现执行前数据缺口：当前本地 16 个 dcv3 task 目录有已跟踪的 E173 PRG
  sidecar，但 pristine `scene_act.xml`/trajectory 不在 task 目录；必须从 E173
  source scene snapshot/权威产物恢复并校验 SHA，不能从 PRG sidecar 反向删 pair。
- 缺口已有只读 authority：`results/E173/scene_snapshot/cem_sidecars/<case>/`
  对 16 条 box023 都保存了 pristine `scene_act.xml`、rubber intermediate 与
  PRG sidecar；E179 builder 可直接以 pristine snapshot 为输入生成新 sidecar。
- `patch_hand_collision.py` 会在安装目录编译验证 mesh hand；E179 可把新
  `scene_act_E179_rubberHull.xml` 安装回对应 task dir，同时把 source/sidecar
  复制到 E179 snapshot。原始 task 目录缺失 trajectory 仍需从 E173 retarget
  产物或远程 authority 恢复。
- E173 retarget 目录保存了 converted/retargeted/trimmed NPZ，但抽查 converted
  `with_obj.npz` SHA 不等于最终 `trajectory_kinematic.npz` SHA，不能直接冒充
  paired trajectory；必须用下游 config/转换器确定可复现恢复方式或取回原文件。
- 当前 `/mnt/ali-sh-1/.../spider` 不存在；本机仅有
  `/mnt/a0ccc676.../spider_workdirs/core4d` 结果备份，未发现完整 task 目录。
- E173 Full `config_act.yaml` 仍记录准确 `data_path/model_path` 与所有 resolved
  E167A+PRG 字段，可作为恢复与负审计证据；Hydra task 数据目录本身被清理。
- 已定位 E175 的 `restore_stage2b_inputs.py`，它专门从既有 Stage2b/CEM
  evidence 恢复 `scene.xml/scene_act.xml/trajectory_kinematic.npz` 并做 SHA/
  数组审计；下一步先核实能否严格恢复 E173 16 条，而非另写不可靠转换。

### 2026-07-24 Authority 实施

- 已新增 `scripts/experiments/E179/{e179_common.py,build_paired_authority.py}`，
  固化 hashes、5-worker queue、Full/canary budget、PRG 禁止字段和统一 12 门。
- authority builder 已成功运行：raw box023=`46`、paired=`16`、unique=`16`、
  v1/v2=`15/1`、worker=`4+3+3+3+3`；E173 只读 baseline 重算严格得到
  physics6=`13/16`、12-gate=`7/16`。
- 产物已写入 `results/E179/input_authority/`；当前明确记录 Stage2b task
  restored=`0/16`、restore_required=`16/16`，未把缺失输入误报为 ready。
- 已新增 `restore_e173_stage2b_inputs.py`：先 isolated authority canary，
  要求生成 trajectory SHA 精确匹配 E173 且 scene-act 语义等价，再事务式恢复
  16 条 primary artifacts；不会覆盖现有 E173 PRG sidecar。
- restore inspect 通过 schema 检查：16 条均为 primary artifacts 全缺失、0 条
  partial，因此满足安全恢复前提；尚未执行 canary/restore。
- Stage2b authority canary `045_p1` 已 PASS：trajectory SHA
  `cd9889…1111` 与 E173 精确相等、trimmed qpos exact、scene-act semantic
  exact，MuJoCo 维度 `scene=43/41/29`、`scene_act=42/41/35`。
- 16 条正式 transaction restore 已启动；当前进程仍在执行，完成前不标记
  restored 或 ready。
- transaction restore 已完成并 PASS：`restored_from_e173_trimmed=16`、
  trajectory authority SHA exact=`16/16`、audit=`16/16`、failures=`0`。
- `build_paired_authority.py --require-restored-tasks` 与 restore inspect 已再次
  成功生成输出；随后仅用于摘要显示的 `jq` 命令因本机未安装退出 127，不影响
  前两项验证或产物。后续使用 Python/标准工具解析 JSON，不重复依赖 `jq`。
- 后续 Python 复核确认 authority `restored=16/16, required=0`，restore inspect
  `complete=16, partial=0, missing=0`。
- Hydra 现场 compose E173 dcv3/E167A base：`cem_hand_gate=true`、
  `cem_posture_gate=true`、surface band `1.5`、z-only enabled；PRG
  `leg_object_*`/`cem_leg_*` 字段不在 composed config。E179 仅覆盖 scene 名即可
  保持 E167A parity 且不引入 PRG。
- 已新增 `build_e167a_no_prg_manifest.py`：fresh rubber patch、E173 pre-PRG
  semantic parity、lower-body/object pair negative audit、E167A profile compose
  audit、3-row canary 与 16-row Full manifest。
- builder `--dry-run` PASS：authority/unique=`16/16`、canary=`3`、
  worker=`4+3+3+3+3`、input/SHA failures=`0`；尚未 apply 场景与 override。
- builder `--apply --snapshot` PASS：scene audit=`16/16`、config audit=`16/16`、
  no-PRG pair=`16/16`、与 E173 pre-PRG rubber semantic exact=`16/16`；
  Full=`16`（1024×32）、canary=`3`（64×4）。
- 已现场确认 task dir 新 sidecar=`16`、E179 override=`16`。发现 audit 中
  `adapter_ref` 因 `results/` symlink resolve 成 mount 绝对路径；不影响运行
  manifest，但为远程可移植性将在统一 `rel()` 后重建产物。
- 已修复 E179 `rel()` 优先 lexical repo-relative，并重建 authority/manifest；
  source/trimmed/adapter 等现均为 `workspace/...` 可移植路径，两个 builder
  重跑均 PASS。
- 已新增并运行独立 `audit_e167a_no_prg.py --require-all`：method parity
  `16/16`、no-PRG config/scene/runtime-diagnostic negative audit `16/16`、
  canary set exact、worker queue exact，failures=`0`。
- 实现 launch 前检查了 E178 入口：remote wrapper 存在并转发 E176 通用脚本；
  猜测的 `watch_E178_full.sh` / `run_E178_local_5090.sh` 文件不存在（只读 sed
  返回 2，未修改状态），后续按实际 `rg --files` 结果选取可复用实现。
- 尚未生成 E179 manifest、scene、snapshot、脚本或输出，尚未启动 CEM。
- 首次归档补丁因纯 move hunk 不被 `apply_patch` 接受；已改为带标题变更的
  move，并保留完整备份，未丢失内容。
- 下一步：核对 Tracker/plan/最近 log 和现有 E167A/E173/E178 工具链，然后实现
  paired authority、fresh no-PRG scene、快照、launch/pull/eval/report/audit。

### 2026-07-24 E179 continuation

- 已再次完整读取 `experiment-planning-zh`，并按磁盘计划继续现有 E179，
  不新建实验、不回写 E173。
- 用户最新口径已确认：paired denominator 必须是完整 16 条；主结果必须加入
  root、EEF、object 的 position/orientation tracking gates，和 physics 6 门
  组成固定 12 门。当前计划、authority 与 E173 adapter 已按该口径冻结。
- 接下来优先验证刚新增的 `run_cem_queue.py` 和五个 worker shard；验证通过后
  再实现 launch/pull/eval/render/completion audit，未通过 canary 前不启动 Full。
- `python -m py_compile scripts/experiments/E179/*.py` 已通过。首次用系统
  Python 调 manifest builder 的 `--help` 因该解释器无 `mujoco` 失败；这是
  解释器选择错误、未执行 builder、未改变产物。后续统一使用项目 `.venv`
  Python，不重复用系统 Python 启动 MuJoCo builder。
- 已审阅新 queue runner：它显式传入 seed/samples/iterations，运行后审计
  E167A profile、scene、finite qpos 和 PRG runtime diagnostic，并可合并五个
  worker shard；仍需用正确解释器跑 builder 和所有 shard dry-run 才算验证完成。
- 使用项目 `.venv`（MuJoCo `3.7.0`）重跑 builder：
  dry-run PASS；`--apply --snapshot` PASS；scene/config/no-PRG/pre-PRG semantic
  parity 均 `16/16`；独立 `audit_e167a_no_prg.py --require-all` PASS。
- 新 manifest SHA256 为
  `ce1e4f82c43b6cda69604b06797c858f046945f5ba9722fee1c681b07a5adf12`。
  Canary 3 行和 Full 五个 shard 全部通过 queue runner dry-run。
- Full shard 复核：本地=`4`，A100 `2/3/6/7` 各=`3`，并集=`16`、
  unique=`16`、与 canonical set 精确相等；所有 Full 行预算均为
  `seed=0, samples=1024, opt_steps=32`。
- 已完整读取实验技能的 `remote-execution.md`。E179 采用计划中特化约束：
  只允许 A100 `2,3,6,7`，四张必须同时满足 `<5000MB`、无冲突 compute
  process 和显式预约/所有者允许集合，selection 与 tmux 启动前各检查一次；
  任一卡失败时整体不启动且不自动替换。
- 远程脚本将保持每 GPU 单一串行 shard、每条输出路径唯一、独立 tmux/session，
  并用 execution manifest 记录两次 GPU 快照与 allowlist；pull 只回收该
  manifest 登记的 12 条远程结果。
- 已审阅 E168/E170/E176/E178 的本地、A100、pull 和 watcher 实现。
  E176 通用 fixed-GPU 分支只按 policy 取卡、不会同时检查 fixed 卡的显存和
  compute process，不能直接作为 E179 launcher；E179 将写专用严格检查。
- E179 remote sync 必须包含 12 条 task/trajectory/scene/contact mask、完整
  override dependency、E167A immutable profile、runtime code 和固定 shard；
  远程 runner 使用独立 run root，避免依赖远端 canonical checkout 的未同步状态。
- 已新增 E179 四个 canonical 执行入口：
  `run_E179_local.sh`、`run_E179_remote_a100.sh`、
  `pull_E179_remote_a100_results.sh`、`watch_E179_full.sh`。
- 本地入口固化 3-row canary/4-row Full、16-task scene snapshot、no-PRG
  audit、canary-to-Full gate、GPU0 双重显存/compute 检查和 tmux execution
  manifest；远程入口固化 12-row/四 shard、policy+显存+compute 双检、exact-file
  sync、远程 SHA/preflight 和固定 GPU `2,3,6,7` 并行串行队列。
- pull 只接受 execution manifest 中固定四卡的 12 行路径，回收后逐行执行
  runtime/no-PRG validation 并合并五个 worker shard；watcher 对 SSH 故障保守
  判定仍在运行，并要求本地/远程 session 连续两次消失才 final pull。
- 四个 shell 入口均通过 `bash -n`，E179 Python 再次通过 `py_compile`；
  静态检索未发现换到其它 A100 或 kill/pkill 路径，PRG 名仅出现在负向审计。
- 本地 canary `PREP_ONLY` 现场 PASS：重新调用 16-task
  `snapshot_scenes.sh`、no-PRG audit、queue dry-run，并连续两次确认本地
  GPU0=`150MiB`、compute process=`0`；未启动 CEM。证据写入
  `results/E179/s0_environment/local_canary_e179_local_canary_20260724_204401/`。
- 已在正式 canary 前用 `git add -f` 精确暂存 16 条各自的
  `scene.xml`、pristine `scene_act.xml`、E179 rubber sidecar 和 override，
  共 `64/64` 个 required files；未暂存其他实验文件。
- 正式 3-row canary 已于 `2026-07-24 20:44:53+08:00` 启动在本地
  GPU0，session=`e179_local_canary_20260724_204449`。启动前再次双检
  `150MiB/0 compute`；首行状态已进入 `running`，其余两行为
  `READY_FOR_CANARY`。
- 统一 evaluator 设计已核对公共 `eval.core.core_metrics`、E173 与 E176/E178
  tracking-gate 实现。E179 将直接调用公共 `evaluate_sequence`，并沿用 E173
  已使用的 fixed-reference body-z/release-window helper，再只应用
  `e179_common.apply_12gate_scoring`；不会动态加载历史 evaluator。
- 评测产物将显式写出 16-row case metrics、`16×12=192` long-form gate
  matrix、E173/E179 paired deltas、pass migration 和独立 Markdown report；
  baseline 读取 E179 authority 中已冻结的 E173 16-row 12-gate adapter 结果，
  E173 目录保持只读。
- 已新增并通过语法/编译检查：
  `eval/runners/eval_E179_box023_12gate.py`、
  `eval/reports/gen_E179_box023_paired_report.py` 和 canonical wrapper。
  冻结 baseline 再验为 `16 unique / physics6=13 / 12gate=7`；baseline 不含
  `result_npz/video` 列，但 evaluator 对两列只作可选展示，不影响 gate/delta。
- Canary 第 1 条优化本身完成，但 runtime negative audit 判为
  `failed_validation`：outdir NPZ 出现 `cem_leg_gate_*` 与
  `leg_object_penalty_*` 聚合诊断列。当前 resolved config/scene 仍为 no-PRG；
  需诊断公共 runtime 为何在 gate disabled 时仍序列化这些列，并修成 inactive
  时不输出后 fresh 重跑 canary，不能把该失败结果冒充 PASS。
- 根因已定位到 `spider/simulators/mjwp.py::compute_reward`：它无条件把
  零初始化的 `leg_object_penalty{,_gate}` 与 `cem_leg_gate_*` 张量放进
  per-tick `info`，随后 `run_mjwp._aggregate_info_list` 无条件保存为 NPZ。
  现场值证明 penalty/gate 诊断全为零/默认一，resolved config 明确是
  `scale=0`、empty geoms、`cem_leg_gate_enabled=false`，不是 PRG 被暗中启用。
- 修复策略是让公共 runtime 只在对应 penalty/gate 实际 active 时保留这些
  PRG info keys；E170 等 active PRG 行保持原诊断，E179 inactive 行不再污染
  输出。修复后必须 fresh rerun canary 做集成证明。
- 已在 `spider/simulators/mjwp.py` 实现 active-only PRG info serialization：
  penalty 仅在 `scale>0 && geom_ids` 时保留，leg gate 仅在
  `enabled && geom_ids` 时保留；Python compile PASS，E169 active leg-gate
  plumbing regression test PASS。
- 首轮 canary 是修复前已启动的同一 Python 进程，当前前两条均按预期仍被旧
  runtime 判 `failed_validation`，第 3 条仍在执行；必须等该 session 结束后
  用新进程 fresh 重跑三条，不能把进程中途代码变更当作验证。
- 渲染链路已核对 E168 offline `render_row`、E173 Full 视频路径和 E170
  ffmpeg paired-video 实现。E173 的 16 条 box023 baseline MP4 全部已存在；
  E179 将离线渲染自己的 16 条，再按 case 生成
  `E173 PRG | E179 no-PRG` 横向对照，不依赖 CEM 时在线录制。
- 视觉 completion contract 将要求 new/baseline/paired 各 `16/16`，并在有
  MP4 后按 `video-frames` 提取证据，不能只以渲染文件存在替代实际观察。
- 已新增并通过编译/dry-run：
  `render_paired_results.py`、`run_E179_render_all.sh` 与
  `audit_completion.py`。Renderer dry-run 现场确认 E173 baseline=`16/16`、
  E179/pair 当前未就绪=`16/16`；completion audit 当前合理为
  `5/11 PASS`，未把尚未运行的 Full/metrics/video/review 误报完成。
- 首轮旧进程 canary 已结束，三条均因同一 inactive PRG diagnostic leak
  判 `failed_validation`。下一步先把这批失败产物完整归档为 attempt-1，再由
  已修复 runtime 的新 Python 进程 fresh 跑三条。
- Attempt-1 的三套 root NPZ/outdir/config 与日志已保留到
  `s6_downstream/cem/attempts/attempt1_pre_runtime_fix/` 和
  `logs/E179/cem/attempts/attempt1_pre_runtime_fix/`，未删除。
- 修复后 fresh canary 已于 `20:53:30+08:00` 启动，session
  `e179_local_canary_20260724_205326`；启动前 GPU0 双检仍为
  `150MiB/0 compute`。当前第 1 条运行中，后两条 manifest 的旧失败状态会在
  各自行启动时更新。
- 一个纯 synthetic paired-output 单测因构造数据没有可选展示字段
  `result_npz` 触发 `KeyError`；真实 evaluator 总会填该字段，但将把展示字段
  改为 `.get()` 后重测 16/192 cardinality，不影响正在运行的 canary。
- 已把 paired evaluator 的纯展示路径字段改为 optional `.get()`；compile
  PASS，synthetic frozen-baseline identity test 通过：
  paired=`16`、gate matrix=`192`、12-gate migration=`7 PASS→PASS +
  9 FAIL→FAIL`、gate-cell migration=`167 PASS→PASS + 25 FAIL→FAIL`。
- Fresh canary 第 1 条仍在运行，暂未产生修复后验证结论；后两条显示的是进入
  新 session 前保留的 attempt-1 失败状态。
- 远程 A100 只读预观察成功：远端 project Python/MuJoCo `3.7.0` 可用；
  GPU2=`4MiB/0%`、GPU3=`635MiB/0%`、GPU6/7=`0MiB/0%`。但 GPU3 当前有
  一个 `626MiB` compute process，因此正式 fixed-four preflight 会按设计阻断，
  不能提前启动或改用别卡；待 Full launch 时必须重新双检。
- Fresh canary 第 1 条仍在计算，GPU0 当前约 `892MiB/43%`，session 正常；
  未轮到后两条。
- Remote launcher 的 deliberate negative-gate test PASS：在 fresh canary
  未齐时以 `A100_POLICY_GPUS=2,3,6,7 PREP_ONLY=1` 调用，入口在任何 SSH
  sync/tmux 前因缺第 2/3 条 artifact 退出 `rc=1`，证明 Full 不会越过 canary。
- Fresh canary 第 1 条已 `run_complete_pending_eval`。现场 NPZ
  qpos shape=`(134,2,42)`、finite=true、PRG diagnostic prefix=`0`；
  config 保留 E167A hand/posture/z-only，且 PRG 为
  `scale=0/empty/leg_gate=false`。修复的 inactive serialization 已得到首条
  integration PASS；第 2 条现运行中。
