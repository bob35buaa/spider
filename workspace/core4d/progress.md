# CORE4D 当前进度

## 归档索引

- [E195 至 E194 扩展启动前完整备份](progress_archive/E195_to_E194_expansion_prelaunch_full_backup_20260810.md)
- [E192 完整执行与收尾备份](progress_archive/E192_full_backup_20260810.md)
- [E189 完整执行记录](progress_archive/E189_full_backup_20260807.md)
- 更早阶段见 `progress_archive/`。

## 当前：E194 G1 box001/box023/box021 全 case 扩展

### 2026-08-10 · plan222 执行

- 计划：[plan222](plan/222_E194_G1_box001_box023_box021_expansion_plan.md)
- 用户已批准本机 RTX 5090 GPU0 + 远程 RTX 6000 Ada GPU0/GPU1 三 worker 并行。
- 目标：新增 G1 box001 28 + box023 16 + box021 28，共 72 条 Full；不重跑 A0/G2/G3。
- 冻结配置：E167A_zOnlyBody、PRG、ref_fk、rubber_hull、gravcomp=1、
  translation/rotation gain=500/50、seed 0、Full `1024×32`、canary `64×4`。
- source authority：box001/box023 来自 E173 Full；box021 为 E170 production 24 +
  E169 audited reuse 4；三物体 retarget v1/v2 为 21/7、15/1、25/3。
- 启动前 GPU：本机 401 MiB/0%；远程两卡约 7–8 GiB、67–68%，各余约 40–41 GiB。
  按用户要求叠加运行，不 kill、不抢占、不修改既有 tmux。
- 远程工作树有独立 collab-retarget 改动；只 rsync 精确 allowlist，不 git pull/reset。
- 本地工作树当前仅 `progress.md` 有本轮未提交改动；旧 E194 实现与结果存在。
- 已复核 E194 queue 的输入 SHA、runtime config、finite qpos 与 resume-safe 逻辑；
  E195 提供可复用的 local/Ada/hybrid/pull/watcher 精确同步模式。
- 下一动作：实现独立 expansion common/builder/audit/queue、三机 launcher、pull/watcher、
  公共 metrics evaluator/report 和 render 入口，然后执行 preflight。
- 实现审计确认可复用骨架：E194 queue 已具备 input SHA、finite qpos、resolved
  gravcomp/kp/PRG 校验和 atomic resume；E195 launch/pull 已验证精确 allowlist rsync、
  local/Ada tmux、shard state merge 和 hardened watcher，可按 expansion 路径隔离复用。
- 公共 `core_metrics.py` 已正式包含 `track_obj_z_abs_err_cm_mean`；新 evaluator 将
  直接复用 E172 `evaluate_row` 的公共 metrics/gates，不修改公共 gate 阈值。
- source schema 差异已定位：E173 可直接读取 Full manifest；box021 必须从 E170
  `variants.tsv` 适配，其中 canonical G1 input 使用当前 `result_npz/scene_act/override_*`
  字段并逐行保留 `execution_source/reused_full`，不能误用 E168 baseline 字段。
- 72-row source artifact 审计发现首个输入问题：15 条 E173 row 的 manifest
  `target_scene/trajectory` 绝对旧挂载路径在本机不存在，但其 contact mask、override、
  PRG scene_act、A0 result/outdir/config 均存在。必须从 E173 持久 S3 trimmed/target
  evidence 恢复同一 SHA/variant 的 runtime task 文件，不能用其他 case 替代或静默跳过。
- 本地与远程 `.venv` 均可导入 mujoco/hydra/numpy/yaml；canonical
  `snapshot_scenes.sh` 已复核，将在恢复 72 个 runtime task 后先快照 source scenes，
  再由 expansion builder 追加 base+gravcomp sidecar 快照。
- 15 条缺失均属于 E173 box001 task package；每条 E173 S3 verify summary 保留了
  exact `source_scene` 与 persisted `trimmed` 路径，并证明 trimmed qpos 与原 SPIDER
  trajectory 完全一致。恢复策略冻结为：仅在缺失时复制 verify-summary 指向的
  source scene 与 trimmed trajectory，并强制匹配原 manifest scene/trajectory SHA；
  已存在的 57 条不改写。
- run_mjwp resolved config 确认 CEM 实际读取 task 目录的 `0/trajectory_kinematic.npz`
  和 expansion `scene_name` sidecar；恢复后会先 Hydra compose + MuJoCo load，再允许
  sidecar 构建或远程同步。
- 恢复 SHA 核对已完成第一轮并否定了直接复制假设：15/15 persisted trimmed 与
  generic `box001_person{1,2}/scene.xml` 的文件 SHA 都不等于 E173 manifest 冻结的
  task trajectory/base-scene SHA。verify summary 只证明 trimmed `qpos` 与当时 SPIDER
  trajectory 的 qpos 相同，不能证明 NPZ 容器/其余数组相同；因此未写入任何 task
  输入。下一步优先从远端工作区或 E173 scene snapshot 找 exact-SHA 原件，只有 exact
  match 才恢复；找不到则保留 preflight stop，不降级伪造输入。
- exact artifact 搜索第一轮：E173 `scene_snapshot/cem_sidecars/` 保存了这些 case 的
  `scene_act.xml`、rubberHull 和 PRG effective sidecar，但没有 base `scene.xml` 或
  input trajectory；`spider-remote` 当前工作区对首个缺失 task 同样缺这两个文件。
  因此可从 snapshot 获得运行 scene 权威，但 trajectory 仍需继续查找旧挂载/备份或
  基于数组级合同做可审计重建；尚未放宽 manifest SHA 合同。
- 本机挂载卷 `/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/spider_workdirs/` 发现后续
  E189 对首个缺失 task 的 `scene_snapshot/.../scene.xml`，说明至少 base scene 可从
  后续实验快照追回。下一步批量检查该卷对 15 条 scene 的 SHA 覆盖，并继续搜索
  trajectory；在核对前不复制。
- 进一步澄清 manifest 语义：E173 `base_scene_sha256` 实际等于
  `scene_snapshot/cem_sidecars/<case>/scene_act.xml` 的 SHA，而非 `target_scene` 字段
  指向的 task `scene.xml`；因此 15 条缺失 case 已有 exact-SHA scene authority，可直接
  由 E173 snapshot 构建 G1 sidecar，无需依赖 E189 快照。对照一个完整 case 还确认
  input trajectory 含 `qpos/qvel/ctrl/contact/contact_pos`，persisted trimmed 仅与其
  `qpos` 完全相等，另含 human_joints/fps/cost；不能把 trimmed 文件直接冒充 trajectory。
- A0 CEM artifacts 检查：15 条缺失 case 的 E173 result/outdir 均仍在，result 含完整
  优化 trace（包括 `trace_ref/qpos/qvel/ctrl`），但 runner 的 `load_data` 合同仍显式读取
  input 的 `qpos_ref/qvel_ref/ctrl_ref/contact/contact_pos`，PRG reward 会使用 contact 与
  contact_pos。故不能仅凭 result 中优化输出替换输入；下一步检查 A0 trace 是否无损
  嵌入了五组原 reference，或从旧构建脚本确定可重复的生成算法。
- `spider.io.load_data` 复核：所有五组 input 都先装入并插值；PRG 开启时 contact
  全零会直接报错，`contact_pos` 也进入后续 guidance。E173 保存的 result/outdir
  主要是优化 trace 与 sim trajectory，并未按原字段名保存 input trajectory。因此
  trajectory 恢复必须来自构建流程/备份，不能从 output NPZ 直接字段拷贝。
- 找到仓库内已验证的 exact 恢复先例：E179
  `restore_e173_stage2b_inputs.py` 会在隔离临时 task 中重放
  `create_spider_scene_from_template.py` + `spider/process_datasets/core4d.py` +
  `--generate-scene-act`，并要求生成 trajectory SHA 与 E173 authority **逐字节相等**、
  scene_act XML 语义等同、MuJoCo `(nq,nv,nu)` 合同通过后才原子补齐缺失 task。
  本轮将复用同一 deterministic Stage2b adapter 给 box001 15 条做 canary + exact audit，
  不再寻找或拼接 output NPZ。
- E194 原 queue/audit 复核完成：可保留其 atomic manifest、resume-safe、input SHA、
  finite qpos、Hydra resolved config、compiled model parity 逻辑；扩展版只需改为独立
  `g1_expansion_*` manifest/path、单臂 G1、worker/profile 分片，并将 base model authority
  指向冻结 snapshot。这样不会触碰原 E194 45-row 三臂 manifest。
- box021 source schema 已确认：canonical 28-row authority 位于
  `scripts/experiments/E170/variants.tsv`，当前 PRG fields 为 `result_npz/outdir_npz/
  config_act/scene_act/override_*`，并含 `execution_source/reused_full`；历史
  `e168_baseline_*` 仅作 provenance，不作为本轮 A0/G1 输入。扩展 loader 将逐行保留
  reuse 字段并验证 28 个唯一 case。
- Stage 0 指标来源已定位：box001/box023 的 44 条 A0 z 指标已有公共 evaluator 产物
  `E173/.../e173_object_tracking_position_error_z_by_case.tsv`；box021 需用公共
  `eval.core.core_metrics`/fixed-reference helper 对 28 条当前 E170 PRG result 重算。
  构建时会写独立 `g1_expansion_a0_metrics.tsv`，按每物体 z 最大值选 9 canary、按
  最大值+最近中位数选 6 Full sentinel，tie 取 case_id 字典序。
- 已新增独立 expansion contract `e194_g1_expansion_common.py`：冻结 72-row counts、
  G1 gains/budget、独立 manifest/result 路径、portable 旧挂载路径归一化，以及逐物体
  `local,local,ada0,ada1` 分配（严格 36/18/18、每物体三 worker 覆盖）。未修改旧
  `e194_common.py`。
- 已新增 `build_g1_expansion_manifest.py`：集成 E179 同款 deterministic Stage2b exact
  restore（15 条缺失仅补 primary artifacts、trajectory SHA/trimmed qpos/scene_act semantic/
  MuJoCo compile 四重校验），再对 72 条 source SHA、G1-only XML diff、A0 z selection、
  snapshot、36/18/18 分片做冻结，输出独立 Full/canary/sentinel/source-authority 文件。
  当前代码刚落盘，下一步先编译与 dry-run，尚未执行恢复或 sidecar 写入。
- 首轮编译通过；本机 `MUJOCO_GL=osmesa` 的 PyOpenGL loader 当前不可用，改用已可导入
  的 `MUJOCO_GL=egl`。dry-run 发现历史 cleanup 使不少已可运行 task 只保留
  `scene.xml + trajectory + E173 PRG sidecar`，缺 generic `scene_act.xml/meta`，按 E179
  “五 primary 全齐”判成 partial 过严。本轮实际 runner 必需合同是 scene + trajectory，
  G1 base scene 另由 exact E173/E170 PRG sidecar 提供；将恢复器改为逐文件只补缺失，
  对已存在 scene/trajectory 分别做 compile/SHA 审计，绝不覆盖。
- 恢复器已按上述 runtime 合同修正并重新编译；dry-run 正确识别 57 条可直接保留与
  15 条 `runtime_restore_required`，在未传 `--restore-missing` 时按预期 stop。下一步
  执行 exact adapter：每条先在临时 task 生成并过 E173 trajectory SHA 与 scene semantic
  canary，再只复制目标中缺失的文件；随后才构建 72 个 G1 sidecar/snapshot。
- 已启动 canonical 命令 `build_g1_expansion_manifest.py --restore-missing --apply
  --snapshot`（本机 `MUJOCO_GL=egl`，exec session 5217）。进程当前仍在逐 case 重放
  Stage2b；尚未收到最终 summary，继续监控，期间不启动任何 GPU CEM。
- Stage2b exact restore 已实质完成并落 audit：44/44 E173 runtime rows pass，其中
  29 条保留原 authority、15 条 `restored_missing_exact_stage2b`；随后 builder 在
  box021 A0 z 计算处停止，原因是误用了 body-z helper（其输出不含 object-z key）。
  此失败发生在 sidecar/manifest 构建前，恢复输入本身已闭环。下一步改为公共
  `core_metrics` 的 object tracking 定义后重跑，GPU 运行仍为 0。
- box021 A0 z 实现已改为直接 import `eval.core.core_metrics` 的
  `_table4_tracking_metrics` + `npz_qpos`，与既有 E173 z 报告的定义完全相同；不再
  使用 body-z helper。即将重跑 builder，预计已恢复的 15 条会走
  `preserved_runtime_authority_match`，不会再次写入或覆盖。
- builder 已成功闭合：Full 72（box001/023/021=28/16/28；worker=36/18/18）、
  canary 9（每物体×三 worker）、sentinel 6（每物体 worst+median）、A0 metrics 72；
  72 个 G1 sidecar 与 72 份 expansion snapshot 均已生成。canary cases 为
  box001 `...1_041_p2`、box023 `...11_018_p2`、box021 `...11_037_p1`。
  尚未 git add sidecar，也未启动 CEM；下一步先实现/运行正向 audit 与 queue dry-run。
- 已新增 `audit_g1_expansion.py` 与 `run_g1_expansion_queue.py`：audit 覆盖 72-row
  artifact SHA、Hydra scene/kp=500/rot=50/PRG、compiled model 中唯一 object gravcomp
  delta；queue 保留 atomic manifest、resume skip、finite qpos、resolved config/model_path
  校验，并支持按 worker 过滤。下一步编译、72-row audit 与 9-row dry-run。
- 编译、72-row positive audit 与 9-row canary dry-run 均通过：audit
  `row_failures=0/status=pass`，三物体与 36/18/18 worker 分布精确；dry-run 的每条命令
  都解析到正确 override/task、`64×4` 与 expansion G1 scene_name。Preflight GPU runs
  仍为 0。实现/authority 阶段完成，进入 scoped sync + 三 worker canary。
- 已复核 E195 经验证的 hybrid launch/pull 模式；E194 expansion 将沿用 tmux 单卡串行、
  exact allowlist rsync、remote shard state 回收/merge、SSH keepalive，但按
  `canary`/`sentinel`/`full` 使用隔离 stage/pointer/log 路径。不会对远端执行 git
  pull/reset，也不会操作任何非本实验 tmux/job。
- 已新增 local/Ada launcher：支持 `canary`(3/3/3)、`sentinel`(4/2/0)、
  `full`(36/18/18) 独立 shard；local 固定 GPU0，remote 固定 Ada GPU0/1 且单卡串行。
  远端同步集合由 manifest 精确生成，仅含本轮 runner/common、runtime source/config、
  72 sidecar 与必要 SPIDER runtime 文件；没有 git 操作或非本实验进程控制。
- 已补 hybrid/pull state merge，四个 launch 脚本 `bash -n` 全通过。启动前资源复核：
  本机 GPU0 401 MiB/0%（31.7 GiB free）；Ada GPU0/1 分别约 8.1/7.3 GiB used、
  64/61% utilization（40.4/41.2 GiB free）。远端既有 tmux 均保留不动；按用户授权
  继续 overlay。下一步精确 `git add -f` 72 个 active sidecar 后启动 9 canary。
- 双 scene safeguard 已完成第一层：Git index 精确 stage 72/72 expansion sidecar，
  `other_cached=[]`；results snapshot 早先已 72/72。已执行 `MODE=canary` hybrid launch，
  当前 launcher 仍在 exact rsync/启动阶段（exec cell 141），尚未宣告三端 session 成功。
- 9 canary 已成功启动：local session `E194_G1_canary_local_20260810_171815`（3 rows），
  remote session `E194_G1_canary_ada6000_20260810_171815`（Ada0=3、Ada1=3）；hybrid
  launcher exit 0。现进入 session/log/GPU 监控，未启动 sentinel 或 Full。
- canary 首轮运行态正常：三 session/queue 均存活，本机 GPU0 约 1.1 GiB/40%，Ada0/1
  约 8.7/8.0 GiB 且 97/96%；本机首条已完成 Hydra/sidecar/contact-mask 初始化并在
  `64×4` rollout（finite 进度持续），远端两卡也各有本轮 run_mjwp PID。暂未出现
  traceback/non-finite/config drift；继续等待 9 条终态。
- 利用 canary 运行时间开始实现后处理：已复核旧 E194 evaluator/report，可复用
  `eval_E172_box004.evaluate_row`（其内部直接使用公共 `eval.core.core_metrics`、health、
  12 门适配），但扩展版会对 72 条 A0 + 72 条 G1 全量重算，不直接拼旧两物体报告，
  并保留 object/worker 分层与 paired delta/bootstrap。
- 已新增 expansion evaluator/wrapper/report：计划重算 A0/G1 共 144 rows，直接 import
  `eval.core.core_metrics` 并复用 frozen scorer；输出 paired 72-case、by-object、
  by-device、10,000 次 seed-0 bootstrap CI、gate flips 与 C3-C8 判定。报告 C9 明确保持
  pending，直到 72/72 MP4 与 mandatory paired visual review 完成。下一步代码编译/单元
  dry check；canary 仍独立运行中。
- evaluator/report/wrapper 已 `py_compile`/`bash -n`/`git diff --check` 全通过；修正了
  new-fall 计数为严格 A0 PASS→G1 FAIL。尚未在 Full 不完整时提前执行科学评测。
- canary 三端 session 已结束；local 3/3 `run_complete_pending_eval`。首次 remote pull 后
  canonical 总计 5 complete / 4 failed，即远端仅 2/6 成功，四条产物缺失；技术
  stop-loss 已触发，**不启动 sentinel/Full**。下一步读取 remote shard failure_mode 与
  row logs，判断是否为单一环境/路径问题；不盲目重跑、不迁移 worker。
- 四个 remote failure 已精确定位为同一 scoped-sync 缺项：E173 PRG override 的 Hydra
  `defaults` 依赖 `examples/config/override/core4d_<target_task>.yaml`，本轮只同步了顶层
  `override_path`；box021 两条之所以成功，是远端恰已有其 base override。失败发生在
  Hydra compose、GPU rollout 前，无 non-finite/physics 异常。将 allowlist 明确加入 72
  个 base override 后，仅在原 Ada0/Ada1 对 failed box001/box023 做 resume rerun；已完成
  5 条按 output skip，不迁移、不改实验配置。
- remote allowlist 已修复并 `bash -n` 通过；manifest 派生的 base override 为 72/72
  unique、missing=0。现在仅重启 remote canary shard；runner 会对成功的 box021 两条
  做 validated skip，只实际运行四条原 Hydra-precompose failure。
- remote canary resume session `E194_G1_canary_ada6000_20260810_180048` 已成功启动，
  shard 仍为 Ada0/Ada1 各 3 rows 以保持 frozen identity；已完成 rows 由 queue 自检 skip，
  failed rows 原卡续跑。sentinel/Full 继续保持未启动。
- resume 运行态：remote tmux、两 queue、两张卡上的 box001 run_mjwp 均存活，Ada0/1
  utilization 98%；说明 Hydra base override 缺项已越过、当前已进入实际 rollout。
  worker log 路径暂未在普通目录可见（进程 stdout 仍持有），下一步只读检查 fd/manifest
  状态并等待终态，不因日志展示问题中断正在运行的合法 job。
- fd 检查确认 queue/run_mjwp stdout 都绑定预期 E194 canary log（目录展示异常是远端
  log 路径/挂载可见性问题，不是进程失联）；未做干预。后处理方面已确认现有 E168
  `render_row` 可按 landed `config_act + outdir_npz` 精确 self replay，扩展 renderer 将
  复用它生成 72/72 G1 MP4，并把既有 A0 video 作为 paired visual authority。
- 已新增 `render_g1_expansion.py` 与 render-all wrapper：输出隔离到
  `render/full_g1_expansion`，同时生成 72-row `paired_video_manifest.tsv` 连接每条既有
  A0 video 和新 G1 self replay。已定位统一 z 报告 generator 的 source-spec/group 入口，
  Full eval 后会追加 E194 G1 box001/023/021 三组，不与旧 box024/004 合并。
- 已将统一 z 报告 generator 的 E194 G1 expected groups 扩展为 box001=28、box023=16、
  box021=28；render 与 z report 代码已 compile、shell-check、diff-check 通过。generator
  会在 expansion case_metrics 尚不存在时自然 hard-stop，因此当前不提前重写用户原报告。
- remote resume 中间状态已核实：Ada0/1 的 box001 均转为
  `run_complete_pending_eval`，既有 box021 仍 complete，当前两卡仅 box023 为 running。
  即 allowlist 修复对原四条 failure 已成功解决一半，且无新 failure；继续等最后 2 条。
- 已新增并 shell-check `watch_E194_G1_expansion_and_finalize.sh`：按 canary/sentinel/full
  监控本轮确切 session、pull/merge、逐 row runtime output revalidation；只有 Full 72/72
  时才串行执行 eval、paired report、72 render 和统一 z 报告。视觉 review 仍单独保留，
  watcher 不会把自动渲染误当成人工/模型视觉结论。
- canary resume 已全部闭合：remote Ada0/1 三行各自 3/3 complete；pull 后 canonical
  9/9 `run_complete_pending_eval`。逐行 `output_failures` 复核为空，覆盖 result/outdir/
  config finite、scene_name、kp=500、rot=50、PRG 与 model_path；canary stop-loss 解除。
  现进入 6 条 `1024×32` Full sentinel，仍不直接启动剩余 66 条。
- 已执行 `MODE=full SENTINEL_ONLY=1` hybrid launcher；当前处于 72-row re-audit/
  scoped sync/启动阶段（exec cell 191），尚未确认 sentinel sessions，Full 主队列保持未启。
- 6 Full sentinel sessions 已成功启动：local
  `E194_G1_sentinel_local_20260810_182046` 4 rows，remote
  `E194_G1_sentinel_ada6000_20260810_182046` Ada0 2 rows / Ada1 0 rows；这是 frozen
  worst+median 分配的自然结果，不做迁移。hybrid exit 0，现持续监控 1024×32 终态。
- sentinel 初始运行态正常：local 首条在 GPU0（约 2.0 GiB/50%）；remote Ada0 正在
  首条 `1024×32`（约 9.7 GiB/97%），Ada1 保持既有非本实验 workload、不触碰。
  已启动本地 watcher `E194_G1_sentinel_watch_20260810_1821`，仅监控上述确切 sessions，
  结束后自动 pull/merge/逐行 validation；不会自动放行 Full。
- sentinel case 分布已核实：local 包含 box001 worst、box023 worst、box021 worst+median；
  Ada0 包含 box001 median、box023 median。首条 Full 实测 `opt_steps=32` 单 2-step 约
  11.3–11.7s，预计单 case 数十分钟、local 4 条约数小时；watcher 按 60s 轮询，期间
  不占用/终止其他 job。
- 18:24 监控：sentinel 启动约 4 分钟，local/Ada0 首条均保持 `running`，其余 4 条仍
  `READY_FOR_FULL`，watcher 连续三轮确认两 session active；无 premature status/error。
- 结果记录编号已预留为下一号 log273（tracker 当前 E194 仍标“扩展待执行”）；只在
  Full/eval/render/visual 真正闭合后更新为完成。当前另起只读 monitor exec cell 205，
  跟随 sentinel watcher，不改变任何 job 状态。
- 18:36 watcher 仍报告 local/remote active；monitor unified session=62459，当前等待
  下一次 60s 输出（exec cell 207）。无 failure/complete 新事件。
- 18:37–18:39 watcher 连续报告两端 active；首条仍在预计的 Full 时长窗口内，无异常。
  monitor 继续等待（exec cell 209）。
- 18:40 两端仍 active。为避免额外常驻只读 shell，已仅终止 monitor session 62459
  （exit 130）；真正的 sentinel local/remote 与 watcher tmux 均未触碰，继续正常运行。
- sentinel 长跑等待期间经过两轮 60s 无用户 steering；未执行任何状态变更。下一轮读取
  watcher/shard 终态，而非主动干预运行。
- 又经过两轮 60s 被动等待，无 steering/外部指令；sentinel job 未被操作。
- 再经过两轮 60s 被动等待；按预计首条接近完成窗口，下一动作读取 shard/watcher。
- 19:03 sentinel 状态：local 首条 box001 已 complete，第二条 box023 running；Ada0
  首条 box001 仍 running（约 170/238 sim steps）。本机 Full 每 2-step 约 11–12s；Ada0
  因与既有 workload overlay 每 2-step 约 32s，明显更慢但持续前进，无 hang/error。
  保持 frozen worker，不为提速迁移或抢占。
- 19:14 sentinel 状态：local 2/4 complete（box001/box023），第三条 box021 worst
  running；Ada0 首条仍 running、第二条 pending。watcher 两端 active，无 failure。
- 19:31 sentinel 状态：local 3/4 complete、最后 box021 median running；Ada0 1/2
  complete、最后 box023 median running。合计 4/6 complete，最后两条无 error、持续推进。
- 19:58 sentinel 状态：local session 已结束且 4/4 complete；Ada0 1/2 complete，最后
  box023 median 仍 running。合计 5/6，无 failure；watcher 仅等待 remote 终态。
- sentinel 已于 20:10 自动闭合：pull/merge 后 6/6 `run_complete_pending_eval`，逐行
  runtime validation failures={}；canonical Full 同步为 6 complete + 66 READY。local/
  remote/watcher sessions 均自然结束。下一步做 6-case public-core divergence/fall 指标
  复核，确认技术 stop-loss 后才启动 72-row resume Full。
- 已启动不带 `--require-all` 的 partial public-core evaluator（session 2679），用于 6 条
  sentinel 的 paired z/3D/fall/divergence stop-loss；它会明确写 `status=incomplete`，仅作
  pre-Full 诊断，最终 72/72 时由 require-all evaluator 覆盖。
- sentinel public-core 评测完成：A0 72 + G1 sentinel 6，errors=0。6/6 z/3D/jerk
  finite、new_falls=0、max G1 3D pos error=18.69 cm（无 E194 G2/G3 式发散）；逐 case
  Δz 全为改善（-0.34 到 -6.40 cm）。box021 median 的 Δ3D=+1.89 cm 属科学 claim
  后续关注项，但不是技术 stop-loss。sentinel 放行，准备 resume 72-row Full（已完成 6
  条不会重跑）。
- 已执行 `MODE=full` hybrid launcher；当前在 final 72-row re-audit/scoped sync/启动
  阶段（exec cell 233）。canonical Full 预状态为 6 complete + 66 READY，queue 的
  complete-status filter 将保证 sentinel 不重跑。
- final Full re-audit 再次 72/72 pass；launcher unified session=49522 仍在远端 scoped
  sync/启动（exec cell 235），尚未记录最终 tmux names。
- local Full session `E194_G1_full_local_20260810_203113` 已启动，shard 36 rows（其中
  4 sentinel complete 将 skip）；remote shard 18/18 已生成，仍在 allowlist rsync
  （exec cell 237），尚未确认 remote tmux。
- remote exact sync 仍在进行（launcher session=49522，exec cell 239），未见错误输出；
  local queue 已独立运行。继续等待 remote session 确认，不重复发起 launcher。

### 2026-08-10 20:31 — E194 G1 expansion Full remote launch confirmed

- Full remote launcher completed without relaunch: `E194_G1_full_ada6000_20260810_203113`.
- Frozen Full execution is now active across local GPU0 plus remote Ada GPU0/GPU1; next step is process/GPU identity verification and automatic finalizer watcher startup.

### 2026-08-10 20:33 — E194 G1 expansion Full workers and watcher active

- Confirmed local queue plus one `run_mjwp.py` on RTX 5090 GPU0.
- Confirmed both remote queues plus one `run_mjwp.py` on each RTX 6000 Ada physical GPU; pre-existing remote GPU processes were left untouched.
- Started finalizer tmux `E194_G1_full_finalize_20260810_203113`; it will wait for frozen shards, pull exact results, require 72/72, evaluate, report, and render.

### 2026-08-10 20:36 — E194 G1 expansion Full monitor checkpoint

- Watcher resolved the exact frozen sessions and reports `active local=1 remote=1`.
- Local first Full row remains active on GPU0 (`run_mjwp.py`, about 1.6 GiB); no shard completion is expected yet.
- Remote log lookup from the SSH default directory was inconclusive, but the already-verified remote processes continue; next check will resolve their `/proc` working directory rather than changing execution.

### 2026-08-10 20:38 — E194 G1 remote execution path resolved

- Both remote queue PIDs run from `/home/xiayb/pHRI_workspace/spider` with the intended Ada0/Ada1 shard manifests.
- Initial guessed remote log filenames were absent; this is only an observability-path issue. Queue command lines and processes remain correct and active.

### 2026-08-10 20:39 — E194 G1 first-row live progress verified

- Local box001 first row reached simulation step 82/252 with steady ~11.5–12.0 s optimization batches.
- Remote Ada0/Ada1 first rows reached step 24/260 and 24/256 respectively with steady ~31–34 s batches under the existing shared-GPU workload.
- No exception, non-finite signal, or early termination appears in any of the three first-row logs.

### 2026-08-10 20:39 — E194 G1 completion counter monitor active

- Started non-mutating monitor session 93286; it reports only changes in successful Full-row completion counts and exits when the finalizer watcher closes.
- Initial post-sentinel resume counts: local remaining shard 0 newly done, remote remaining shards 0 newly done; the six sentinels are already canonical and intentionally skipped.

### 2026-08-10 20:40 — E194 G1 Full unchanged monitor tick

- No new row completed during this interval; all three first rows remain within their expected long optimization window.

### 2026-08-10 20:41 — E194 G1 active-goal persistence check

- Goal `019fe1b4-886c-7823-ab38-fd0e97d9d9ba` remains active with no token budget cap; execution will continue through Full/eval/render/visual closure.

### 2026-08-10 20:42 — E194 G1 Full unchanged monitor tick

- Completion counters remain unchanged; no worker exited and the finalizer remains active.

### 2026-08-10 20:44 — E194 G1 Full live-step checkpoint

- First active cases advanced to local 134/252, Ada0 44/260, Ada1 42/256; completion counters still unchanged after the following monitor interval.

### 2026-08-10 20:45 — E194 G1 execution plan state synchronized

- Plan step 6 remains in progress: run/monitor/pull all 72 Full rows. Step 7 (full eval, 72 MP4, mandatory visual review, report/log/tracker closure) remains pending behind the exact 72/72 gate.

### 2026-08-10 20:49 — E194 G1 Full live-step checkpoint

- Active first rows advanced to local box001 `200/252`, Ada0 box001 `68/260`, Ada1 box001 `66/256`.
- Finalizer has continuously observed both local and remote tmux sessions active through 20:48; no new Full row has completed yet and no worker failure is visible.

### 2026-08-10 20:50 — E194 G1 Full unchanged monitor tick

- No row-completion transition in this interval; the watcher and completion counter monitor remain active.

### 2026-08-10 20:51 — E194 G1 Full unchanged monitor tick

- No successful row completion reported yet; continued execution is within the measured first-case runtime.

### 2026-08-10 20:53 — E194 G1 Full unchanged monitor tick

- No atomic `[done]` transition yet; workers remain under the finalizer watch.

### 2026-08-10 20:54 — E194 G1 first local Full row at final step

- Local box001 first row reached `246/252`; Ada0/Ada1 reached `86/260` and `82/256`.
- Next monitor transition should validate the first newly completed local Full artifact and begin the second non-sentinel row automatically.

### 2026-08-10 20:55 — E194 G1 first local row post-run interval

- No `[done]` transition in the first minute after reaching the final simulation steps; monitoring continues through output save/postprocess/validation.

### 2026-08-10 20:56 — E194 G1 first local optimization finished

- Local box001 `039_p1` completed 252/252 in 1406.94 s and saved `trajectory_mjwp_act.npz`; final raw object tracking error reported pos=0.1029, quat=0.1166.
- `run_mjwp.py` exited cleanly; queue-level postprocess/output validation has not yet emitted `[done]`, so the row is not counted complete prematurely.

### 2026-08-10 20:57 — E194 G1 local queue advanced to second row

- Local queue already launched box001 `039_p2`, proving the first row passed queue postprocess sufficiently to advance.
- The auxiliary completion counter referenced a nonexistent local worker-log filename, so its local count is observationally stale; authoritative staged-manifest status will be used and the execution queue itself is unaffected.

### 2026-08-10 20:58 — E194 G1 first new Full row validated

- Authoritative local shard manifest: 5 `run_complete_pending_eval` (4 prior sentinels + newly completed box001 `039_p1`), 1 running, 30 ready.
- Newly completed `039_p1` has empty `failure_mode`; local queue is now running `039_p2`.
- Remote authoritative shard state remains Ada0 2 prior sentinels + 1 running + 15 ready, Ada1 1 running + 17 ready.

### 2026-08-10 20:58 — E194 G1 authoritative manifest monitor corrected

- Stopped only the flawed diagnostic counter session (exit 130); no experiment process was touched.
- Started corrected read-only monitor session 50634 using local and remote shard TSV statuses directly. Baseline exactly matches the audited 5/2/0 completed plus three running rows.

### 2026-08-10 20:59 — E194 G1 second-row live checkpoint

- Local second row box001 `039_p2` is at 54/250.
- Remote first rows reached Ada0 `104/260` and Ada1 `98/256`; both execution sessions and the finalizer remain active with no status failure transition.

### 2026-08-10 21:00 — E194 G1 Full unchanged monitor tick

- No shard status transition during this interval; three active cases continue normally.

### 2026-08-10 21:01 — E194 G1 automatic eval/report chain static audit

- Eval wrapper enforces `--require-all`; evaluator requires exactly 72 G1 artifacts and returns nonzero unless A0=72, G1=72, paired=72, errors=0.
- Report generator independently requires 72 paired rows and emits by-case, by-object, by-device, bootstrap/claims artifacts.
- Mandatory visual-review language is explicitly non-final until 72/72 MP4 review, consistent with the completion contract.

### 2026-08-10 21:02 — E194 G1 pull/render closure audit finding

- Pull script merges local plus both remote shard statuses into the canonical Full manifest before the exact 72/72 watcher gate.
- Render driver records failures but currently does not fail on skipped rows or assert 72 rendered/72 paired, and treats any existing file as rendered. This is weaker than the plan's 72-MP4 completion gate and will be tightened while GPU execution continues.

### 2026-08-10 21:03 — E194 G1 render completion gate hardened

- Full-manifest render now requires exactly 72 manifest rows, 72 nonempty rendered MP4 entries, zero skipped rows, zero failures, and 72 paired evidence rows.
- Zero-byte existing videos are no longer accepted as rendered.
- Updated renderer passes Python compilation and whitespace validation.

### 2026-08-10 21:04 — E194 G1 index safeguard rechecked

- Git index still contains exactly 72 staged scene sidecars and no other staged paths, preserving the active-sidecar dual safeguard.
- Expansion automation remains unstaged/untracked by design during execution; filesystem contents are authoritative and the corrected manifest monitor reports no new transition.

### 2026-08-10 21:05 — E194 G1 plan-to-implementation audit findings

- Found three report-chain gaps against plan222 while GPU work remains healthy: evaluator currently imports another experiment runner (`eval_E172_box004`) instead of directly using public core metrics; report covers only a subset of the 12 gates/McNemar requirement; canonical report filename is shorter than the frozen deliverable name.
- These are closure-quality issues, not execution changes. They will be fixed before Full evaluation runs.

### 2026-08-10 21:07 — E194 G1 evaluator dependency analysis

- Public `eval.core.core_metrics.evaluate_sequence` now directly emits the frozen primary z MAE, 3D tracking, masked contact, and E191 lifted/xy/z-share diagnostics.
- The borrowed E172 row evaluator also adds motion health, release-window, leg-gate health, and numeric gates; these helpers must be localized or moved to a shared core module to satisfy plan222's no cross-experiment evaluator import contract.

### 2026-08-10 21:09 — E194 G1 twelve-gate contract resolved

- Confirmed the canonical 12 gates are six physics gates (`fall/body_z/contact/release/hand_penetration/lower_body`) plus six tracking gates (`root_pos/root_ori/hand_pos/hand_ori/object_pos/object_ori`).
- Existing G1 expansion evaluator carries only the six physics gates, so C7 and mandatory PASS→FAIL selection are incomplete until the six frozen tracking thresholds and per-gate migrations are added.

### 2026-08-10 21:14 — E194 G1 direct public-core 12-gate scorer implemented

- Removed the cross-experiment `eval_E172_box004.evaluate_row` dependency.
- Expansion evaluator now calls `eval.core.core_metrics.evaluate_sequence` directly, adds motion health, fixed-reference body-z diagnostics, release applicability, PRG leg-gate health, and the frozen six physics + six tracking gates.
- Paired output now carries every per-gate A0/G1 value plus the 12-gate overall pass/failure modes and artifact SHA fields.

### 2026-08-10 21:15 — E194 G1 direct scorer parity test, first half

- Updated evaluator compiles, passes diff whitespace checks, and contains no cross-experiment evaluator or dynamic-import dependency.
- On A0 box001 `041_p2`, the first ten continuous metrics—including primary z MAE, 3D, lifted diagnostics, body-z, contact, penetration, and leg penetration—match the previously landed scorer bit-for-bit (delta 0.0).
- Test harness stopped only when attempting to cast boolean `fall_flag` as float; evaluator itself completed successfully. Harness will be corrected and G1 parity finished.

### 2026-08-10 21:16 — E194 G1 direct scorer parity proven on paired sentinel

- For box001 `041_p2`, both A0 and G1 match every previously computed key metric and all six legacy physics gates exactly.
- New 12-gate layer correctly adds tracking migrations: A0 fails `hand_ori` (and legacy lower-body), while G1 passes all 12 gates for this case.
- This proves the direct scorer preserves old metric numerics while exposing the tracking gates plan222 requires.

### 2026-08-10 21:19 — E194 G1 report chain upgraded to 12-gate contract

- Report now uses all 12 gates, emits a 72×12 per-case migration matrix, records all four paired transition classes, and computes exact McNemar p per gate/object.
- Added the frozen canonical report filename `E194_G1_box001_box023_box021_expansion_report.md` while retaining the short alias.
- Summary now records SHA256 for case metrics, paired deltas, by-case, by-object, by-device, gate matrix, and canonical Markdown; decision remains explicitly incomplete until visual closure.
- Evaluator/report compile cleanly; exact McNemar helper passed representative sanity values.

### 2026-08-10 21:20 — E194 G1 report cardinality gates verified

- Added exact per-object paired counts (28/16/28) and exact 864-row gate-migration assertion.
- Report compiles and intentionally fails closed on the current partial six-pair eval (`paired rows=6 expected=72`), proving it cannot publish a partial Full report.

### 2026-08-10 21:21 — E194 G1 second new local Full row complete

- Local authoritative shard: 6 complete (4 sentinel + 2 new), 1 running, 29 ready; no failures.
- Local queue advanced to box001 `041_p1` at 92/294. Remote first rows are near their final quarter: Ada0 206/260, Ada1 196/256; both remote failure lists remain empty.

### 2026-08-10 21:22 — E194 G1 frozen A0 authority check scoped

- Builder landed `g1_expansion_a0_metrics.tsv` with exactly 72 unique cases and object counts 28/16/28.
- Full evaluator currently recomputes all 72 A0 rows but does not yet compare primary z MAE back to this frozen authority at the plan's `1e-4 cm` tolerance; the fail-closed comparison is being added.

### 2026-08-10 21:24 — E194 G1 A0 authority and metric invariant gates implemented

- Full evaluator now writes a 72-row A0 authority audit and requires every recomputed primary z MAE to match the frozen builder table within `1e-4 cm`.
- Every A0/G1 row also requires finite z and 3D errors with `z MAE ≤ 3D L2 MAE`; violations enter eval errors and prevent pass.
- Compilation passes. Single-case audit reproduced box001 `041_p2` exactly (`abs_diff_cm=0.0`) and correctly refuses overall pass until all 72 authority rows are present.

### 2026-08-10 21:26 — E194 G1 required metric coverage expanded

- Paired evaluator/report now cover primary z, 3D, lifted z/xy/z-share, body-z, raw/3mm contact, release, penetration, leg interference, pelvis/root/EEF/object tracking, acceleration, qpos/body/ankle jerk, and fall.
- By-object output will bootstrap every continuous paired delta at seed 0 with 10,000 samples; the 12 boolean gates remain reported separately through migration counts and exact McNemar tests.

### 2026-08-10 21:40 — E194 G1 first remote Ada0 Full row complete

- Expanded 22-metric paired contract compiles; representative G1 sentinel has zero non-finite required continuous metrics.
- Ada0 authoritative shard advanced to 3 complete (2 sentinel + 1 new), 1 running, 14 ready. Local remains 6 complete + 1 running; Ada1 first row is still running. No failure state appeared.

### 2026-08-10 21:41 — E194 G1 execution and authority-report checkpoint

- Ada0 completed box001 `040_p1` with empty failure mode and advanced to `042_p1` (18/218).
- Local `041_p1` reached 254/294; Ada1 `040_p2` reached 252/256, both near row completion.
- Canonical report evidence and summary SHA map now explicitly include the 72-row A0 authority parity audit.

### 2026-08-10 21:43 — E194 G1 C8 device/profile claim implemented

- Frozen Full manifest confirms every object spans all three profiles with exact per-object splits 14/7/7 (box001), 8/4/4 (box023), 14/7/7 (box021).
- By-device rows now record both worker and execution profile. Each object gets explicit C8 profile-coverage and z-delta direction-consistency verdicts; a worker-direction conflict will block a unified claim.

### 2026-08-10 21:46 — E194 G1 three additional Full transitions

- Local now has 7 complete (4 sentinel + 3 new), 1 running, 28 ready; it advanced to box001 `043_p2` at 58/238.
- Ada1 completed its first new Full row and advanced to box001 `042_p2` at 42/244; Ada0 `042_p1` is at 54/218.
- All three authoritative shard failure lists remain empty.

### 2026-08-11 — E194 G1 progress inspection resumed

- Re-read the full `experiment-planning-zh` contract before inspecting current execution state; status will be derived from the three shard manifests plus live session/process evidence.

### 2026-08-11 02:58 — E194 G1 overnight Full checkpoint

- Authoritative three-shard total: 44/72 Full complete (61.1%), 3 running, 25 ready; local=21/36, Ada0=12/18, Ada1=11/18.
- All three workers and the finalizer watcher remain active; every shard has zero failed rows/failure modes.
- Canonical manifest remains at the pre-pull six sentinels by design; remote results are merged only after both execution sessions finish. Local currently holds 23 NPZ (21 local + 2 previously pulled remote sentinels), and eval/render remain pending at 6 paired / 0 MP4.

### 2026-08-11 04:03 — E194 G1 Full checkpoint

- Authoritative total is 50/72 (69.4%): local 24/36, Ada0 14/18, Ada1 12/18; three rows running and 19 ready.
- Current rows: local box021 `034_p1` 186/286, Ada0 box021 `032_p1` 142/164, Ada1 box021 `038_p2` 106/234.
- All shard failure lists are empty; local/remote sessions and finalizer remain active.

### 2026-08-11 — E194 G1 recovery/eval/three-version comparison requested

- User requested current progress inspection, result recovery, Full evaluation, noPRG/PRG/G1 comparison workbook, and experiment log closure.
- Activated `experiment-planning-zh`, `xlsx`, `experiment-report-writer`, and `markdown-mermaid-writing`; reading their complete instructions before state-changing work.

### 2026-08-11 — E194 comparison deliverable contracts loaded

- XLSX must be professionally formatted, use formulas for derived comparisons, be recalculated with LibreOffice, and contain zero formula errors.
- Experiment log will use an ablation-report structure, separate observations from interpretation, document exact commands/config/hardware, and include metrics, limitations, decisions, and next steps.

### 2026-08-11 — E194 report/XLSX style contracts completed

- Read the Markdown and Mermaid style guides plus flowchart conventions; the log will use one H1, evidence-first tables, accessible Mermaid, and clear observed-vs-inferred wording.
- The experiment-report skill's advertised template path is absent in its installed directory; continuing with its required section structure as the documented fallback.

### 2026-08-11 08:41 — E194 G1 automatic Full closure reached

- All authoritative shards are complete: local 36/36, Ada0 18/18, Ada1 18/18; failed=0.
- Finalizer pulled and merged results, completed Full eval, rendered exactly 72/72 MP4 with zero skipped/failed, updated the unified E173 z report, and printed `automatic full closure complete`.
- New user-requested work now begins: add noPRG as a third comparable arm, generate a recalculated/error-free XLSX, perform mandatory video review, and write log273/tracker closure.

### 2026-08-11 — E194 three-arm closure plan synchronized

- Read `video-frames`; mandatory visual review will use ffmpeg timestamp extraction rather than relying only on metric tables.
- Plan now tracks six closure stages: Full verification complete; noPRG authority in progress; three-arm eval, XLSX verification, visual review, and log/tracker closure pending.

### 2026-08-11 — E194 Full eval evidence and noPRG source discovery

- Existing Full eval passes: A0(PRG)=72, G1=72, paired=72, errors=0, A0 authority parity=true; outputs include 3 by-object rows, 9 by-device rows, and 864 gate migrations.
- E189 contains a canonical noPRG Full manifest/eval and user-approved evidence for box001, box023, and box021, making it the preferred single-source noPRG authority instead of mixing E168/E179/E189.

### 2026-08-11 — E194 noPRG authority correction

- Initial comparison accidentally used old E194's 15-case A0 manifest; corrected target authority is the 72-row `g1_expansion_source_authority.tsv`.
- E189's own Full manifest covers all 28 box001 rows but not box023/box021. Its user-approved evidence manifests contain only subsets (13 box001, 7 box023, 9 overlapping box021) and cannot serve as 72-case authority.
- The complete noPRG candidate route is therefore E189 Full for box001, E179 Full for box023, and E168 production for box021; exact case/variant/artifact parity is being audited next.

### 2026-08-11 — E194 closure resumed from recovered 72/72 state

- Rechecked the active goal and worktree after user continuation; the execution result remains complete at local 36/36 + Ada0 18/18 + Ada1 18/18 with zero failures.
- Existing public-core PRG↔G1 Full outputs are present, including 72 paired cases, 3 by-object rows, 9 device rows, 864 gate migrations, and the canonical report. The remaining work is the same-case noPRG arm, workbook validation, visual evidence, and experiment-log/tracker closure.

### 2026-08-11 — E194 noPRG 72-case authority audit passed

- Joined the E194 source authority against E189 Full box001 (28), E179 Full box023 (16), and E168 production box021 (28): 72 rows, 72 unique case IDs, exact 28/16/28 object counts, no retarget-variant mismatch, and no missing target case.
- All 72 historical rows have nonempty result NPZ, outdir NPZ, resolved config, scene, trajectory, contact mask, and video. E189's three `READY_FOR_FULL` labels are stale manifest state only; their complete artifacts are present and will be scored directly.
- Added a resumable uniform public-core three-arm evaluator. It reuses the already validated 72 PRG + 72 G1 scores, computes all 72 noPRG rows under the identical metric/gate contract, and fail-closes unless it produces 216 arm-case rows, 144 paired rows, 1,728 comparison-gate migrations, and zero errors.

### 2026-08-11 — E194 three-arm evaluator environment correction

- The new evaluator passes compilation and whitespace validation. Its first invocation used system Python and stopped before scoring because that interpreter lacks MuJoCo.
- Confirmed the original Full wrapper's authority is `.venv/bin/python` with `MUJOCO_GL=egl`; continuing under that unchanged environment. No result or manifest was modified by the failed import.

### 2026-08-11 — E194 noPRG public-core scoring started

- Started the 72-row noPRG scorer under the same `.venv` + EGL + four-thread environment used by the landed Full evaluator.
- The process is live; it writes an atomic per-case TSV cache after every successful row, then will combine noPRG with the validated PRG/G1 arm metrics and enforce exact three-arm cardinalities.

### 2026-08-11 — E194 three-arm public-core evaluation passed

- Uniform scoring finished successfully: noPRG=72, PRG=72, G1=72; combined arm-case rows=216, same-case paired rows=144, two-comparison 12-gate migrations=1,728, evaluation errors=0.
- noPRG was freshly scored case-by-case; PRG/G1 were reused only from the already authority-checked Full public-core output with identical `metric_standard_id`. Exact case-set intersection is 72/72.
- Added a professional formula-driven XLSX builder with arm-level evidence, paired comparisons, by-object/case-weighted aggregate formulas, gate detail/summary, and visual-review sheets. Formula syntax is being validated through LibreOffice before delivery.

### 2026-08-11 — E194 comparison workbook built; recalc invocation corrected

- Built `E194_noPRG_PRG_G1_comparison.xlsx` successfully after Python compilation and whitespace checks.
- The first recalculation attempt executed the Python helper as a shell script and exited before touching the workbook. Correcting the invocation to `python .../recalc.py`; the workbook still requires LibreOffice recalculation and zero-error verification before it is considered final.

### 2026-08-11 — E194 XLSX recalculation passed and visual set frozen

- LibreOffice recalculated the comparison workbook successfully: 8,456 formulas, zero `#REF!/#DIV0!/#VALUE!/#NAME?` or other formula errors. Cached values and sheet cardinalities were reopened successfully.
- Mandatory PRG→G1 review uses any individual 12-gate PASS→FAIL, not merely overall PASS→FAIL. The union with `Δz>+1 cm`, `Δ3D>+2 cm`, and PRG baseline worst/median cases is 36 unique cases: box001=15, box023=10, box021=11.
- The 36-case set already covers local-gpu0=16, ada-gpu0=11, and ada-gpu1=9. There are no new fall, non-finite, or diverged cases.

### 2026-08-11 — E194 mandatory frame extractor implemented

- Added a deterministic extractor that reconstructs exact noPRG/PRG/G1 video authorities for each selected case, validates all source videos, and uses ffmpeg to sample four normalized motion stages (`grasp/lift/carry/place`).
- It will emit 432 stage frames, 36 three-arm case sheets, nine four-case review atlases, and a provenance TSV carrying selection reasons, profile, metric deltas, video paths, and durations. Concrete observations remain pending visual inspection and will not be left blank in the final review artifact.

### 2026-08-11 — E194 mandatory visual evidence extracted and inspected

- Extractor completed: 432 nonempty stage frames, 36 case sheets, nine atlases, and a 36-row selection/provenance TSV. All noPRG/PRG/G1 source videos resolved successfully.
- Inspected all nine atlases. Every selected G1 case remains finite and upright through the sampled stages; no new fall or gross simulation divergence is visible. Most individual gate PASS→FAIL rows are subtle posture/contact/orientation threshold changes rather than catastrophic failures.
- Two metric exceptions remain important despite stable playback: box001 `20231020_014_p2` has `Δz=+1.866 cm, Δ3D=+1.878 cm` with a visibly different box attitude in carry/place; box021 `20231018_028_p2` has `Δ3D=+3.043 cm` with a shifted carry pose. Box021 `20231018_028_p1` has `Δz=+1.089 cm` without a gross visual failure, so it remains a numeric case-level exception.

### 2026-08-11 — E194 concrete visual-review writer added

- Added a 36-row review writer grounded in the inspected four-stage atlases. Every row records nonempty noPRG, PRG, and G1 observations, the exact selection reason/profile/deltas, a case-level verdict, and its contact-sheet path.
- Gate-specific wording distinguishes visible posture/orientation/contact differences from 3 mm penetration thresholds that cannot be resolved visually; numeric gate failures are retained rather than being overwritten by the video impression.

### 2026-08-11 — E194 visual closure landed in workbook and canonical report generator

- Wrote all 36 concrete visual-review rows, rebuilt the workbook with the populated Visual Review sheet, and re-ran LibreOffice: 8,456 formulas, zero errors.
- Updated the canonical G1 report generator to close only when exactly 36 mandatory rows exist. Final decision becomes `PROMOTE_G1_WITH_CASE_LEVEL_EXCEPTIONS`; it records the three numeric exceptions and hashes the visual-review TSV.
- Observed PRG→G1 claims C3–C8 all pass per object. Profile-stratified z deltas are negative for every object × worker cell (9/9), excluding a device-sign confound.

### 2026-08-11 — E194 expansion experiment log drafted

- Added log273 with separate observed-results and interpretation sections, exact authority/hardware/config contracts, three-arm by-object z/3D tables, claims C0–C9, visual exceptions, scoped promotion decision, reproduction commands, limitations, and artifact paths.
- The three-arm result is explicit: PRG is near-neutral on z and worsens 3D for all three objects versus noPRG; G1 improves z and 3D versus PRG and finishes below noPRG on both metrics for every object.

### 2026-08-11 — E194 tracker/index/canonical report closed

- Tracker E194 now records 72/72 completion and links log273, the original E194 result, and both plans. Regenerated `log/INDEX.md`; log273 is listed under Phase 57.
- Re-ran the canonical report generator after visual closure. Decision is `PROMOTE_G1_WITH_CASE_LEVEL_EXCEPTIONS`, paired rows=72, and the summary now hashes the 36-row visual-review TSV.
- The canonical report identifies the three case-level numeric exceptions and scopes promotion to E167A+PRG box001/box023/box021 rather than claiming universal object or RL-policy success.

### 2026-08-11 — E194 G1 72-case expansion final verification complete

- Final cardinalities passed: arm-case 216, paired 144, gate migrations 1,728, visual review 36, G1 MP4 72, stage frames 432, case sheets 36, atlases 9.
- Final workbook recalc passed again with 8,456 formulas and zero errors; workbook SHA256 is `aa7158495c4e0c11c183a0aab7b9890176e2eedd99bc12cea31f96972439d297`.
- Canonical report and log contain no incomplete/TODO placeholders; all new/modified Python files compile, global `git diff --check` passes, and summary decisions/cardinalities reopen successfully.
- E194 G1 expansion is complete. Final scoped decision: `PROMOTE_G1_WITH_CASE_LEVEL_EXCEPTIONS`.

### 2026-08-11 — E194 explicit 12-gate comparison requested

- User requested a concrete noPRG/PRG/G1 12-gate metric comparison in both the existing XLSX and log273.
- Re-loaded the `xlsx`, `experiment-report-writer`, and `experiment-planning-zh` contracts plus tracker/plan/log/progress context. The update will add per-object/per-gate pass counts, rates, percentage-point deltas, paired flip counts, and exact McNemar p-values; the workbook will be rebuilt and LibreOffice-recalculated to zero errors.

### 2026-08-11 — E194 12-gate concrete metrics audited

- Recomputed arm-level gate counts directly from the 216-row case-metric authority. Across all 864 gate decisions per arm: noPRG=`709/864` (82.1%), PRG=`722/864` (83.6%), G1=`720/864` (83.3%); strict all-12 case pass is `15/72` (20.8%), `19/72` (26.4%), `19/72` (26.4%).
- The strongest favorable gate is `lower_body`: 48.6%→69.4%→87.5%; PRG→G1 paired flips are P→F=2, F→P=15, exact McNemar p=0.00235.
- The clearest G1 regression is `object_ori`: 98.6%→98.6%→84.7%; PRG→G1 P→F=10, F→P=0, exact p=0.001953. Other all-object G1 changes are small and mixed; detailed object-level counts have been verified for all 12 gates.

### 2026-08-11 — E194 12-gate evaluator artifacts landed

- Extended the existing three-arm evaluator with exact McNemar and explicit gate tables, without rerunning CEM or recomputing cached noPRG trajectories.
- Evaluation passed and now emits `e194_three_arm_12gate_by_object.tsv` (48 rows = 4 scopes × 12 gates) and `e194_three_arm_12gate_overall.tsv` (4 scopes), including arm pass counts/rates, three percentage-point deltas, both paired flip directions, and exact p-values.

### 2026-08-11 — E194 12-gate XLSX/log closure resumed

- Re-read the XLSX, experiment-planning, and experiment-report contracts plus the E194 tracker/plan/log/progress context before continuing the existing deliverables.
- Frozen evidence remains noPRG/PRG/G1=`709/722/720` passes out of 864 gate decisions and strict-12=`15/19/19` passes out of 72 cases; workbook recalculation and report/log synchronization are now in progress.

### 2026-08-11 — E194 12-gate workbook first recalculation exposed formula incompatibility

- Workbook builder compiled and generated both new 12-gate sheets, but LibreOffice found 96 `#NAME?` cells in exact McNemar p-value columns O/R.
- Root cause is the newly used Excel `BINOM.DIST` syntax not being accepted by the installed LibreOffice; no metric-source rows are affected. The p-value formulas will be replaced by a LibreOffice-compatible exact-binomial expression and recalculated again.

### 2026-08-11 — E194 12-gate formula source audit

- The 48 detail rows and four aggregate rows are formula-driven from `Arm Case Metrics` and `Gate Migrations`; the frozen evaluator TSV independently matches the intended values.
- The incompatibility is isolated to the modern dotted function name. The builder will use the legacy cross-compatible `BINOMDIST` alias while retaining formula-derived exact two-sided McNemar p-values.

### 2026-08-11 — E194 12-gate workbook recalculation passed

- Replaced `BINOM.DIST` with the Excel/LibreOffice-compatible `BINOMDIST` formula spelling and rebuilt the existing comparison workbook.
- LibreOffice recalculated all 9,304 formulas with zero errors; the workbook now contains formula-driven `12-Gate Comparison` and `12-Gate Overall` sheets.

### 2026-08-11 — E194 strict-12 workbook audit caught boolean typing defect

- Data-only verification confirmed 48 detailed gate rows and all headline lower-body/object-orientation values, including exact p-values.
- The aggregate sheet's strict-12 counts were zero because `numeric_release_pass_12gate` was written as text rather than an Excel boolean; this is a workbook-builder typing defect, not an evaluator/evidence defect. The source TSV remains `15/19/19` strict passes and the builder is being corrected before delivery.

### 2026-08-11 — E194 strict-12 workbook formulas corrected

- `numeric_release_pass_12gate` is now written as a real Excel boolean; rebuilt aggregate formulas reproduce noPRG/PRG/G1 strict counts `15/19/19` overall.
- Object-level strict counts also match the evaluator: box001=`6/5/8`, box023=`4/7/3`, box021=`5/7/8`. LibreOffice recalculated 9,520 formulas with zero errors.

### 2026-08-11 — E194 canonical report generator extended to concrete 12-gate evidence

- The generator now fail-closes unless the three-arm 12-gate evidence has exactly 48 detail rows and four aggregate rows.
- It will publish the full 72-case gate table, per-object aggregate/strict-12 table, paired flip counts and exact McNemar p-values, explicitly preserving the `object_ori` and box023 strict-12 regressions alongside the `lower_body` improvement.

### 2026-08-11 — E194 log273 updated with concrete 12-gate metrics

- Added all 12 all-case pass counts/rates, PRG→G1 pp deltas, paired P→F/F→P counts, and exact McNemar p-values.
- Added per-object pooled gate rates and strict-12 counts; interpretation now states that `lower_body` improvement is offset by significant `object_ori` regression and that box023 strict-12 drops `7/16→3/16`.
- Updated the workbook validation evidence to 9,520 formulas with zero LibreOffice errors and narrowed C7 wording to an aggregate-floor pass with a structural warning.

### 2026-08-11 — E194 canonical report and tracker synchronized

- Canonical/alias report generation passed with the full 12-gate table, aggregate/strict table, exact p-values, and hashed three-arm 12-gate TSV evidence.
- Tracker E194 now summarizes the result as z/3D improvement with mixed gate composition (`lower_body↑`, `object_ori↓`) and retains the gate warning in the promotion status.

### 2026-08-11 — E194 12-gate XLSX/log update final verification complete

- Rebuilt `log/INDEX.md`; log273 remains the Phase 57 evidence entry.
- Final LibreOffice audit passed with 9,520 formulas and zero errors. All 48 detail rows and four aggregate rows in the XLSX match the evaluator TSV cell-by-cell, including exact p-values and strict-12 `15/19/19`.
- Evaluator/workbook/report generators compile, both reports contain no incomplete placeholders, and global `git diff --check` passes.

### 2026-08-11 — E194 workbook metric-direction and tracking expansion

- Expanded `Paired Comparison` and `By Object` from seven to 14 metrics, adding object orientation, body-z/pelvis/root tracking, and hand position/orientation errors.
- Delta headers now mark `↑ better` or `↓ better`; conditional formatting is zero-centered and semantic across all delta sheets: green always means improvement, red regression, yellow no change.
- Preserved the single missing false-release observation as blank instead of false zero and changed by-object metric summaries to average valid observations only.
- LibreOffice recalculation passes with 10,696 formulas and zero errors; all 144×14 paired before/after/delta values match the evaluator TSV.

### 2026-08-11 — E194 semantic delta gradients restored

- Restored continuous red–yellow–green gradients while retaining per-metric directionality and delta=0 as the yellow midpoint.
- Higher-is-better metrics shade negative→red and positive→green; lower-is-better errors shade negative→green and positive→red.

### 2026-08-11 — E194 PRG/G1 numeric failure modes requested

- Workbook builder now carries raw numeric failure-mode strings in arm and paired evidence and adds a dedicated 72-row `PRG-G1 Failure Modes` sheet with formula-derived mode counts and count deltas.
- While integrating this view, two stale fixed Arm Case column references from the earlier tracking expansion were identified; 12-gate and strict-pass lookups are now derived from the metric layout instead of hardcoded letters.

### 2026-08-11 — E194 PRG/G1 failure-mode workbook verification complete

- Added raw failure modes to `Arm Case Metrics` and both before/after failure-mode columns to `Paired Comparison`; added the dedicated 72-row `PRG-G1 Failure Modes` sheet with formula-derived PRG/G1 counts, count delta, and overall migration.
- LibreOffice recalculated 11,128 formulas with zero errors. All 72 PRG/G1 strings and counts match the paired evaluator evidence.
- Re-audited all 48 detailed and four aggregate 12-gate rows after dynamic column repair; every value matches the evaluator TSV and strict-12 remains `15/19/19`.

### 2026-08-11 — E194 G1 viser review integration started

- Added an explicit E194 viewer source override pointing to `full_g1_expansion/e194_g1_expansion_case_metrics.tsv` and filtering strictly to `arm=G1`.
- Default viewer scope now includes E194; UI labels distinguish it as `E194/G1`, annotation storage remains isolated under the expansion eval directory, and thresholds fall back to E173's identical metric standard.

### 2026-08-11 — E194 G1 viser review integration verified

- `review_player.sh E194 --check` passes with indexed/evaluated/playable=`72/72/72`, strict numeric pass=19, and exact object counts box001/box023/box021=`28/16/28`.
- MuJoCo loading passed for one sentinel per object: each compiled at `nq=42`, loaded the full rollout plus reference trajectory, and produced nonempty 50 fps frame indices.
- Actual viser smoke indexed all 72 rows, opened HTTP/WebSocket on localhost, loaded the first scene, and remained live for the 15-second smoke window. The default all-experiment audit also passes at 317/317 playable.
- One initial smoke harness using temporary-file cleanup was rejected by command safety before execution; it was replaced with a direct bounded server smoke and did not affect artifacts or results.

### 2026-08-11 — viser case-name search added

- Added a `Case 名称（支持子串）` text filter to the review player; matching is case-insensitive and multiple whitespace-separated tokens must all occur in the case ID.
- The case query composes with the existing experiment/object/numeric/failure-mode/variant filters rather than replacing them.

### 2026-08-11 — viser case-name search verified

- Pure search checks pass for empty query (72 rows), case-insensitive full-case lookup, multi-token partial lookup, and no-match behavior.
- Initial test incorrectly assumed `BOX001 039_P1` was unique; it correctly matched two sessions, so the unique-case assertion was changed to include `20231003_1`. No implementation change was needed.
- E194 headless audit still passes at indexed/evaluated/playable=`72/72/72`; bounded viewer smoke started successfully with the new search widget present.

### 2026-08-11 — E173 PRG manual review join added to comparison workbook

- Audited the requested E173 filled review TSV: 37 unique reviewed rows (`USE=22`, `DO_NOT_USE=15`), of which 28 box001 cases intersect the E194 72-case authority (`USE=19`, `DO_NOT_USE=9`).
- `Paired Comparison` now carries the complete E173 review payload for exact case matches and an explicit source marker; box023/box021 rows are marked `NOT_IN_E173_REVIEW` rather than imputed from another experiment.

### 2026-08-11 — E173 PRG manual review workbook join verified

- `Paired Comparison` now has 61 columns and retains 144 rows. All nine added source/review columns match the requested TSV field-for-field for the 56 duplicated paired rows representing 28 unique cases.
- Joined row decisions are `USE=38`, `DO_NOT_USE=18`; all 88 unmatched paired rows are explicitly marked and have empty review payloads.
- LibreOffice recalculation remains at 11,128 formulas with zero errors; headline gate/strict aggregates remain `709/722/720` and `15/19/19`.

### 2026-08-11 — box001 PRG→G1 exclusion analysis started

- Recomputed the exact paired subset after excluding `box001_20231023_110_p1`: 27 box001 cases remain.
- Strong improvements persist in object z (`−1.225 cm`, bootstrap 95% CI `[−1.547,−0.862]`), object 3D (`−1.275 cm`, `[−1.885,−0.701]`), hand penetration (`−0.080`), leg penetration (`−0.048`), and lower-body/hand-penetration gates (`+22.2 pp` each).
- The evidence is not an across-the-board improvement: object orientation worsens `+2.516°` (CI `[+0.389,+5.142]`), raw contact drops `−0.074` (CI `[−0.129,−0.024]`), and contact/release/hand-orientation/object-orientation gates regress. Final judgment awaits manual/visual and worst-case audit.

### 2026-08-11 — box001 PRG→G1 exclusion judgment complete

- On 27 cases, pooled gate rate moves only `284/324→286/324` (`+0.6 pp`) and strict-12 `5/27→8/27` (`p=0.508`); mean failure-mode count changes only `1.481→1.407` with case counts improve/worse/tie=`11/9/7`.
- Favorable gate changes are hand penetration and lower body (`+22.2 pp`, exact p=`0.03125` each); adverse changes include contact/release/hand position/hand orientation and object orientation, with object-orientation pass `27/27→22/27`.
- Existing visual evidence has 14 reviewed box001 selections: 11 non-catastrophic gate-flip warnings, two stable improvements, and one explicit regression exception (`014_p2`).
- Same-case human-review overlap is currently 14/27: PRG→G1 transitions are USE→USE=6, DNU→DNU=4, USE→DNU=4, DNU→USE=0. This subset is review-selected and incomplete, but it directly contradicts an across-the-board improvement claim.
- Final judgment: G1 is a strong scoped improvement for object z/3D and penetration/lower-body behavior, but not a comprehensive improvement over PRG for box001.

### 2026-08-11 — E194 PRG box001 authority correction

- User identified `E173/s6_downstream/eval/full/user_manual_review_filled.tsv` as non-authoritative for the PRG side. The final authority is `E173/s6_downstream/rl_export/box001_user_approved/box001_user_approved_source_rows.tsv`: its 13 unique box001 cases are USE, and the other 15 cases in the frozen E194 box001 28-case universe are DO_NOT_USE.
- Rebuilt `Paired Comparison` with authority membership semantics. Its 56 box001 rows are USE=26 / DO_NOT_USE=30 because each case appears in two comparisons; all 88 box023/box021 rows are explicitly `NOT_APPLICABLE_BOX001_AUTHORITY` and receive no imputed decision.
- The old PRG counts `19/9` unique (`38/18` paired) and old excluded-set human migration `USE→USE=6, DNU→DNU=4, USE→DNU=4, DNU→USE=0` are invalid. With the corrected PRG authority and the current E194 G1 review, excluding `box001_20231023_110_p1`, the 14 reviewed transitions are USE→USE=3, USE→DNU=3, DNU→USE=3, DNU→DNU=5.
- LibreOffice recalculation passes with 11,184 formulas and zero errors. Gate totals remain `709/722/720`, strict-12 remains `15/19/19`, and all 14 semantic red-yellow-green delta gradients are preserved.
- Added immutable correction log `log/274_E194_PRG_box001_authority_correction.md` and linked it ahead of log273 in the E194 tracker row.
- The first final gradient-audit assertion expected an ARGB alpha prefix of `00`, while openpyxl correctly reopened the workbook colors with `FF`; this was a test-harness assumption, not a workbook defect. The assertion was corrected to compare RGB, after which all 14 direction-aware gradients passed.

### 2026-08-12 — E194 G1 box001 full manual review re-evaluation started

- User completed all E194 G1 box001 labels. The current G1 review TSV has 28/28 unique reviewed box001 cases with exact authority-set parity: USE=15, DO_NOT_USE=13; quality labels are CLEAN=10, MINOR_ACCEPTABLE=5, UNUSABLE=13.
- PRG remains the corrected E173 RL-export membership authority: 13 USE and 15 complement DNU. Under the previously requested primary exclusion of `box001_20231023_110_p1`, the 27-case manual transitions are USE→USE=8, USE→DNU=5, DNU→USE=7, DNU→DNU=7; PRG/G1 USE counts are 13/15, net +7.4 pp, agreement 15/27, and exact paired McNemar p=0.774414.
- The all-28 sensitivity has the same 5/7 discordant transitions and USE counts 13/15; the excluded case is DNU→DNU, so it does not change the direction or significance of the manual comparison.
- Numeric evidence on the primary 27 remains strongly favorable for z/3D and penetration/lower-body behavior, but mixed elsewhere: z −1.225 cm, 3D −1.275 cm, hand penetration −0.080, leg penetration −0.048; object orientation +2.516°, raw contact −0.074, object-orientation gate 27/27→22/27, and hand-orientation gate 18/27→15/27.
- The five manual USE→DNU regressions still improve z on average (−1.185 cm) but worsen object orientation by +8.485° and raw contact by −0.230, showing that G1 shifts failure modes rather than monotonically improving case quality. The seven DNU→USE recoveries improve z/3D by −1.076/−1.366 cm with near-neutral orientation/contact.
- The first exploratory script imported SciPy for an optional exact test; SciPy was unavailable and the script stopped after printing deterministic metric/gate aggregates. It was replaced with a standard-library exact-binomial implementation, which completed successfully; no artifact or input was modified by the failed read-only run.

### 2026-08-12 — E194 G1 box001 completed manual review re-evaluation closed

- Added reproducible evaluator `scripts/eval/reports/analyze_E194_box001_manual_review.py`; it fail-closes on the 13-case E173 PRG USE authority, exact 28-case G1 review coverage, and exact box001 paired set. It emits a 28-row case TSV plus a summary JSON with primary-27/all-28 manual, metric, gate, bootstrap, and source-hash evidence.
- Primary decision is `NOT_COMPREHENSIVE_IMPROVEMENT`, recommended policy `CASE_LEVEL_PRG_G1_SELECTION`. The manual union is 20/27 (74.1%) but is explicitly treated as a post-hoc upper bound, not an automatic selector result.
- Updated the existing XLSX: `Paired Comparison` remains 144 rows and expands to 73 columns with full G1 review provenance/decision/quality/migration/inclusion fields; new `Box001 Human Review` sheet contains the 28 case rows and formula-derived primary summary.
- LibreOffice recalculation passes with 11,512 formulas and zero errors. Workbook formulas reproduce PRG/G1 USE 13/15, transitions 8/5/7/7, agreement/churn 15/12, exact p=0.774414; all 14 paired delta gradients plus seven human-review key-metric gradients preserve green=improvement semantics.
- Re-audited unchanged global evidence after the workbook extension: gate totals remain noPRG/PRG/G1=`709/722/720`, strict-12 remains `15/19/19`, and all G1 review decisions/quality/source SHA values match the completed TSV row-for-row.
- Added log275 and linked it first in the E194 tracker row. Final policy is to preserve the five PRG-only USE cases, adopt the seven G1-only recoveries, and avoid a global G1-for-PRG replacement claim.
- Final report audit initially asserted both taxonomy and review notes were empty. Direct source inspection showed taxonomy empty 28/28 but notes nonempty 2/28 (`011_p1`, `014_p1`); log275 was corrected to state that notes are sparse rather than absent. This reporting assertion did not affect metrics, workbook formulas, or decisions.
- Final closure audit passed: analyzer JSON decision/policy, 28-row TSV, workbook cached summary, 11,512 formulas, 14+7 semantic gradients, tracker/log index links, artifact hashes, Python compilation, LibreOffice zero-error result, and `git diff --check` all agree.

### 2026-08-12 — E194 G1 object orientation 离群诊断启动

- 用户提出 box001 的 orientation 均值回退可能主要由 `20231003_2_041_p1`、`20231020_014_p2`、`20231020_014_p1` 三条长尾驱动；本轮将同时审计 box023/box021 是否存在同类模式。
- 已新增 `plan/223_E194_G1_object_orientation_outlier_diagnosis_plan.md`，冻结为纯离线诊断：不修改 scene/physics/result，仅重算 case-level robust statistics、逐帧首次分叉、配置/XML diff，并用异常/正常视频对照检验接触滑移机制。
- 预注册关键边界：MuJoCo body gravcomp 在质心补偿重力净力，不能未经证据写成“直接施加旋转 torque”；最终报告将明确区分 measured observation、indirect mechanism evidence 与 hypothesis。
- 已定位 72-row three-arm metrics、paired deltas、G1/A0 result/config/trajectory 路径与 72 对 PRG/G1 MP4；三个指定 box001 case 均有现成配对视频，可按 `video-frames` 做同时间关键帧审计。
- 公共 evaluator 的 object orientation 定义已核对：逐帧使用 `abs(dot(q_run,q_ref))` 的 quaternion geodesic angle（wxyz，单位 deg）后取 mean，因此 quaternion `q/-q` 符号翻转不会制造错误；reference 在 43-qpos 输入时直接取 robot prefix 后的 7-DoF object pose。
- G1 scene 构建与正向 audit 均冻结为 base PRG scene 的唯一 compiled-model delta：`body_gravcomp[object] 0→1`；kp_pos/kp_rot 仍为 500/50、PRG 保持开启，其余受审模型数组逐项相等。下一步先完成全 72 case robust 排名，再决定正常 control 与逐帧窗口。
- 72-case 初步稳健统计：overall mean delta `+1.570°`，但 median `−0.065°`、正/负 case=`33/39`，证明不是普遍同方向 shift。box001 all-28 mean/median=`+2.624/−0.077°`；主 27 为 `+2.516/−0.123°`。
- box001 三个指定 case 的 delta 为 `+21.876/+19.098/+18.273°`，PRG→G1 ratio=`3.86/4.03/3.59×`，全部 object_ori PASS→FAIL；它们贡献主 27 总正增量的 `87.2%`。冻结 TSV 上主 27 去三条后的普通 mean 是 `+0.361°`（与用户约 `+0.29°` 的定性判断一致；后续报告会解释精确口径），median `−0.207°`、10% trimmed mean `+0.107°`。
- box023 复现相同长尾结构：mean/median=`+2.855/+0.223°`，`20231020_042_p1/p2` 为 `+15.064/+13.357°` 且 `4.25/3.51×`；同 session 的 `040_p1/p2` 也为 `+8.957/+8.304°`。前四条几乎解释全部 object mean 回退；box021 则 mean/median=`−0.218/−0.065°`，无 orientation gate flip、无 `>3×` 或 `>18°` case。
- 逐帧输入合同已核实：PRG/G1 landed rollout 均为 `(H,2,42)`，公共 `npz_qpos` 使用第 0 轨迹；fixed reference 为同长度 `(H,43)`，object reference pose 位于 36-qpos robot prefix 后。三个指定 case 的 PRG/G1/reference 长度严格相等，不存在 padding 或错帧迹象。
- 现有配对视频为 50 fps、同 case PRG/G1 时长一致；既有 visual pipeline 的阶段锚点为视频时长的 `22%/42%/65%/88%`（grasp/lift/carry/place）。三个指定 case 已有 phase sheets，但本轮还会按数值首次分叉帧补抽，而不只依赖四个固定锚点。
- 已实际检查三个指定 case 的既有三臂 phase sheets：`041_p1` 的 G1 从 lift 起出现明显倾斜并延续到 carry/place；`014_p2` 同样在 lift 后持续大角度前倾；`014_p1` 四锚点上的差异更隐蔽，需要依赖逐帧曲线定位而不能靠缩略图猜测。
- landed NPZ 保留逐帧 `time`、robot qpos/qvel/ctrl 与完整 CEM reward/gate traces；公共接触定义来自 MuJoCo `data.contact`，3 mm clean contact 要求有 hand-object contact 且最深 `dist>=−0.003m`。逐帧诊断将读取真实 `time`，而不是假设视频 50 fps 等于 qpos 采样率。
- 三个异常 case 的实际 rollout 采样均为约 30 Hz；PRG/G1 qpos 帧数与 reference 一致。CEM trace 的 reward arrays 为 `(H,32)`，可用于辅助核对优化信号，但不能把 population mean trace 冒充最终 selected rollout 的直接物理测量。
- 关键 reward/config 结构已确认：`init_rot_actuator_gain/bias=50/50` 在 PRG/G1 间不变；object orientation tracking 仅是 bounded soft term（`task_obj_rot_rew_scale=0.3`, `sigma=0.5`）。更强的 carry rotation gate 未启用（`carry_corridor_rew_scale=0`, `terminal_carry_gate_enabled=false`），hand-support reward 也为 0；因此现有优化没有硬约束阻止少数 weightless-object rollout 旋转到 10° gate 之外。
- 编译模型 audit 比较了 mass/inertia/body pose/geom/contact/joint/actuator 等数组，72/72 仅 `body_gravcomp[object]` 差 `+1`。这支持“G1 没有直接改 rotation actuator/reward”，但仍需逐帧证明异常经由接触/轨迹分叉发生。
- 已新增并运行可复现诊断 runner `scripts/eval/reports/analyze_E194_G1_object_orientation_outliers.py`；Python compile 与 72-row authority 校验通过，16 个诊断 case 的 public-core orientation 均逐帧精确复现，写出 72-row case、robust/session/variant、2,089-row frame curve、frame summary、mechanism audit 与 JSON headline。
- 精确稳健结果：box023 去掉同日 `040_p1/p2 + 042_p1/p2` 后 mean delta=`−0.00048°`，说明其 `+2.855°` 回退几乎 100% 由这两个双人 session 驱动；全 72 去掉 box001 top3 与 box023 pair4 后 mean 仅 `+0.125°`，而原为 `+1.570°`。box021 原本即为 `−0.218°` 改善。
- box001 主 27 去用户三条后的冻结值为 `+0.361°`；若看中位数为 `−0.207°`、10% trimmed mean=`+0.107°`。因此用户的“约 +0.29°、可接受”与本轮结论一致，但报告会保留精确口径而不把近似数改写成复现值。
- 逐帧首次持续 `delta>5°`：`041_p1` 在 lift 约 2.317s；`014_p1` 在 grasp 末约 1.150s、`014_p2` 在 lift 约 1.283s。三者 carry 阶段 mean delta 分别 `+52.27/+30.62/+36.61°`，不是少数末帧尖峰。
- 三个 focal 的接触/手轨迹联动：`041_p1` clean-3mm contact `0.669→0.535`、右手相对物体轨迹 mean shift `23.4cm`；`014_p1` 总 contact `0.600→0.322`、PRG-contact/G1-missing 帧占 `30.4%`、双手 shift 约 `35/35cm`；`014_p2` 为 `0.538→0.333`、missing `30.8%`、右手 shift `17.4cm`。误差主轴三者均为 reference-object local `y` 轴，与视频中的倾斜一致。
- 机制不是简单“所有异常都少接触”：box023 `040_p2` 的 contact fraction 反而 `0.382→0.473`，但手相对物体轨迹仍偏移约 `8–8cm`、orientation 回退 `+8.30°`。更准确的证据表述是 gravcomp 改变了优化出的接触拓扑/作用位置与承重预载；少数几何敏感 session 在弱 rotation soft guidance 下进入倾斜解。
- session/variant 证据反对单一 OmniRetarget variant 根因：`014_p1/p2` 分别为 v1/v2 却都回退约 `18.7°`；box001 v1/v2 的 median delta 分别约 `−0.005/−0.263°`，两组总体均非整体恶化。异常更按 source session 聚集。
- 报告阶段已接入 `markdown-mermaid-writing`：最终 log276 将保持单一 H1、观测/解释分区、表格优先，并用带 accessibility metadata 的 Mermaid flowchart 表达“COM 重力补偿→承重/接触解变化→弱旋转约束下少数倾斜解”的证据链；数值时间曲线仍以 TSV/关键帧为主，不用示意图替代真实数据。
- 首次分叉帧精查：`014_p1` 在视频约 `1.372s` 时 PRG 有接触而 G1 无接触，双手相对物体已偏移 `21.7/23.3cm`；`014_p2` 在约 `1.533s` 同样 PRG 有接触/G1 无接触，右手偏移 `16.4cm`。`041_p1` 在约 `2.778s` 双侧仍有单手接触，但 G1 右手相对物体偏移 `26.9cm`，说明其机制更像接触作用点/拓扑改变而非完全脱手。
- 三条在 carry 峰值的逐帧 delta 分别达到 `60.23/38.55/42.01°`；对应 PRG error 仅 `6.44/1.82/0.70°`。这是持续、版本分叉型姿态回退，不是 reference 本身高动态或末帧异常。
- 跨 72 case，orientation delta 与 raw/clean3 contact delta 的 Pearson 相关约 `−0.302/−0.254`；box001 为 `−0.428/−0.427`。cluster7 的 raw/clean3 contact delta 均值为 `−0.128/−0.093`，其余 65 条为 `−0.013/+0.016`。相关性不是因果证明，但与逐帧接触分叉方向一致。
- cluster7 的平均 z/3D delta 为 `−0.056/−0.427cm`，其余 65 条为 `−1.212/−1.296cm`；orientation 长尾组几乎没有享受到 G1 的典型 position 收益。三个 focal 内部仍有正反例（`014_p1` position 明显改善、`014_p2` 回退），所以不能把 orientation 归因简化为 z/3D tracking 失败。
- `video-frames` 已按数值 onset/peak 为三个 focal 提取 12 张同时间 PRG/G1 PNG 到 `full_g1_expansion/object_orientation_outlier_visual/`；文件均非空。ImageMagick montage 探测未产生 sheet，因此不依赖该可选工具，后续直接审查独立帧并保留现有 phase sheet 作为四阶段总览。
- 已实际审查三个 focal 的 peak PRG/G1 帧：G1 sim 侧均出现固定 reference 下的明显箱体倾斜；`014_p1/p2` 最清楚，`041_p1` 的倾斜同时包含多个轴，与局部 rotation-vector 的 y 主轴和较大 z 分量一致。
- 复核渲染实现发现 landed `(T,2,nq)` 的第二维实际是每个 control tick 的两个 simulation substeps；离线 render 会 flatten 两个 substeps，而公共 evaluator 的 `npz_qpos` 冻结合同只取第 0 substep。PRG/G1 采用同一合同且本轮已精确复现，但为了排除采样 alias，下一步增加 full-substep + converted fixed-reference 敏感性复核。
- 首次 full-substep 尝试复用了 renderer 的 `converted_reference_qpos`，在 `041_p1` 与 `014_p2` 上得到与直接 world-quaternion evaluator 相反的 PRG/G1 排序，而 `014_p1` 一致。该检查不能作为敏感性结论：它暴露的是某些 scene 的 quaternion→Euler visual-reference conversion/parity 风险。已记录并停止沿用该转换；下一步直接在 world quaternion 上对 raw fixed reference 做 2× SLERP，再评估两个 substeps，避免 Euler convention 混杂。
- direct-world-quaternion full-substep 敏感性已完成 144 trajectories：所有 `72/72` case 的 delta 正负与公共 evaluator 完全一致。full-substep all-72 mean/median=`+1.573/−0.071°`，几乎等于 public `+1.570/−0.065°`；box001/023/021 mean=`+2.630/+2.866/−0.222°`。
- 三个 focal 的 full-substep delta=`+21.928/+18.339/+19.161°`，与 public `+21.876/+18.273/+19.098°` 差仅 `0.05–0.07°`。因此长尾结论不是只取第 0 substep 的采样 alias；后续会把这项敏感性固化到诊断 runner/artifact。
- runner 已加入 raw world-quaternion shortest-path SLERP 的 full-substep 逻辑并重跑成功；新增 `e194_g1_object_orientation_full_substep_sensitivity.tsv`，记录 72/72 sign agreement、每臂 public/full/substep0/substep1 数值及 interpolation contract。没有复用有 parity 风险的 renderer Euler conversion。
- 2026-08-12 resumed diagnosis：已恢复 plan223、tracker、现有 9 份 orientation 诊断 artifact 与 12 张 focal 可视化帧；下一步追查 `run_mjwp` 的 raw quaternion→内部 Euler reference 链路，并在 log276 中严格区分 measured / inferred / unresolved。`experiment-report-writer` 安装包声明的模板文件不存在，因此采用其 `SKILL.md` 规定的章节顺序手工落盘。
- 已完整读取 Markdown/Mermaid 文档规范；最终 log276 将使用单一 H1、带 `accTitle`/`accDescr` 的小型 flowchart，并用表格呈现准确数值与证据等级。
- reference 链路已定位：`examples/run_mjwp.py` 在 contact-guidance 模式内把 raw freejoint world pose 转成 scene-act 的 3 slide + 3 hinge reference；E170/E172/E173 builder 另有同型 `convert_reference_to_scene()`，renderer 复用 E168 replay。下一步逐行核对三处对 `body_pos/body_quat/euler_convention/joint order` 的定义是否严格一致。
- 代码层初审：runtime、manifest preflight 与 renderer 三处都使用 `R_body.inv() * R_world` 后 `as_euler(euler_convention)`，而 optimizer 的 object rotation reward 在 `nq_obj=6` 时直接惩罚三个 hinge qpos 的欧氏差；因此若 hinge 轴序/内外旋约定不匹配，可能改变内部 reward 的 target，但 PRG/G1 共用同一 target，最多是 session-sensitive 放大因素，不能单独解释版本差异。
- 发现 metadata availability 不是静态目录属性：当前 72 个本地 task 中 13 个缺 `scene_act_meta.json`，但 E194 G1 运行日志还显示部分当前有 meta 的 case 当时仍以 fallback `XYZ` 转换（例如 box023 `040_p1`、`042_p2`），说明三 worker 的非 git task metadata 同步不一致。三个 focal 的 G1 实际 convention 已由日志确认：`041_p1=XYZ`、`014_p2=XYZ`、`014_p1=XZY`。后续必须以每次 run log 为 authority，而不是当前文件存在性。
- 重新按实际 Full log 核验后修正上一条 focal 记录：三个 focal 的 E194 G1 Full 都明确打印 `quat→XYZ euler`；当前 `014_p1` 本地目录虽有 `XZY` meta，但运行 worker 当时未读到。E194 metric rows 提供每 case 的 exact scene/trajectory/result SHA，可用于把 run-log convention 与 raw quaternion authority 做 72-case world-pose parity 审计。E173 PRG CEM stdout 未随结果目录保存，需优先从 NPZ 内部 reference trace或历史 snapshot 反推，而不能假定与当前目录相同。
- E173 manifest 的 `log` 字段仍指向 `logs/E173/cem/full/...`，但该目录本机已不存在；E173 scene snapshot 也只保存 XML、未保存 meta。landed PRG/G1 NPZ 不含直接 `qpos_ref`，只含 `trace_ref` 等派生 trace。接下来先判断 `trace_ref` 是否可恢复内部 reference parity；若不够，再只读查询既有远端日志备份。
- 三个 focal 的 PRG/G1 `trace_ref` 数组逐元素相同，但该字段只保存 trace site 的 world position；object trace site 很可能位于 COM，对姿态 convention 不敏感，因此不能据此证明内部 Euler target 相同。首次只读远端查询没有返回 E173 Full log，仍需查挂载备份或用运行产物反推；在拿到权威前不宣称 PRG 使用哪种 convention。
- 已确认 `trace_object` site 的 local pos 正是 `[0,0,0]`，所以 `trace_ref` 相等只验证 object COM 位置 reference，不验证姿态。当前远端 `/home/xiayb` 与本机恢复盘未找到 focal 的 E173 Full stdout；PRG runtime convention 的直接日志证据缺失。后续报告会把“E194 G1 确实使用 XYZ fallback”列为 measured，把“E173 PRG 当时使用正确 convention”保留为未证实，除非能从历史 pipeline 证据闭合。
- 72-case G1 Full log + MuJoCo replay 给出强闭合证据：30 条 runtime `XYZ`、20 条 `XZY`、22 条 `ZYX`；其中 29/72 的 runtime convention 与 XML hinge axis sequence 不符。mismatch 29 条的 orientation delta mean/median=`+4.399/+1.726°`，match 43 条为 `−0.338/−0.480°`，且所有 `delta>5°` 的 8 条都在 mismatch 组。raw-target conversion error 与 PRG→G1 delta Pearson=`0.937`。
- 七条长尾正好是 conversion error 最大的 top-7：三个 box001 focal 的 wrong-target world error mean=`30.30/26.83/27.32°`，box023 `042_p1/p2,040_p1/p2`=`19.66/19.31/14.85/14.46°`。G1 相对错误内部 target 的 mean error只约 `4.66–8.28°`，却相对 raw authority 为 `13.68–29.52°`，说明 rollout 实际在跟随错误 Euler target。正确 axis sequence 回放的 world error约 `1e-7°`。
- 已定位 E194 expansion 的两处输入完整性漏洞：`restore_e173_missing()` 虽声明 meta 为 primary artifact，却用“仅 scene.xml+trajectory 存在”判 `runtime_complete`，使 13 个缺 meta task 被直接 preserve；Ada remote launcher 的精确 rsync allowlist 只同步 scene/trajectory/contact/override/sidecar，明确遗漏同目录 `scene_act_meta.json`。远端残留 meta 决定部分 case 是否正确，导致同一 manifest 在不同 worker 上非确定性 fallback 到 `XYZ`。
- 因此当前长尾不能主要归因于 gravcomp 物理作用：G1 expansion 同时意外改变了内部 orientation reference。正确 convention 的 43 case 没有任何 `delta>5°`，均值反而改善 `−0.338°`。接触/手轨迹分叉仍是实际 rollout 行为，但更合理的上游触发是错误 Euler target；gravcomp 只能作为在 target 一致后仍需受控复跑检验的次级因素。
- mismatch 与 worker/sync 强相关：Ada GPU0/1 各 `11/18` mismatch，orientation delta mean 分别 `+4.007/+2.853°`；local 为 `7/36` mismatch、worker 全体 mean `−0.290°`。现有 preflight 只核 SHA/scene/gravcomp/model arrays，完全未检查 meta 存在性、hinge axis sequence 或 raw-quaternion world-pose parity，所以 72/72 preflight pass 没有覆盖这类 reference bug。
- plan223 已扩展：新增 reference conversion helper、focal 回归测试、72-row audit/summary artifacts 与 reference-parity 成功标准。mismatch 29 条去掉 cluster7 后仍为 22 条 mean/median `+1.030/+0.348°`；cluster7 占 mismatch 净回退 `82.25%`，说明错误 convention 是广泛风险，但严重度由 session 的姿态轨迹决定。
- 已先添加 focal regression test；首次 RED 执行被环境阻断：`.venv` 未安装 `pytest`（`No module named pytest`），不是测试通过或实现失败。为避免扩大环境依赖，测试文件将增加可直接执行的 `main()`，同一断言既兼容 pytest，也可由 `.venv/bin/python test_*.py` 运行。
- 已新增 `e194_orientation_reference_conversion.py`：从 Full stdout 解析 runtime convention、从 compiled model 导出 object hinge axis sequence，并同时重放 runtime target、axis-correct target、G1 rollout 到 MuJoCo world quaternion；输出每 case raw/internal target error、public metric reproduction、worker/meta provenance。测试文件已增加零依赖 direct-entry 执行路径。
- focal direct regression 已通过：`runtime=XYZ`、`XML=XZY`、wrong-target mean error `>25°`、axis-correct max error `<1e-4°`、G1 对内部错误 target error `<10°`，且 public raw metric 复现误差 `<1e-6°`。
- 主诊断 runner 已接入 reference audit，新增 72-row case artifact、分组 summary 和 JSON headline；机制字段已改为“reference pipeline 为已诊断主触发、gravcomp 残余效应需同 convention 重跑”，不再把接触变化写成 gravcomp 根因。
- 全量 runner 重跑完成（exit 0）：frame diagnostics `16 case / 2,089 rows`、reference conversion `72/72`、full-substep `144/144`。新增 artifacts 已落盘，原有 robust headline 保持不变，证明新增根因审计没有改写公共指标或 case 排名。
- artifact 自检：reference audit `72 data rows`、summary `14 data rows`；G1 public orientation 重算最大偏差 `2.10e-12°`，axis-correct conversion 最大 world error `2.96e-6°`，position conversion max `1.45e-13cm`，且没有任何 convention-match case 出现 `delta>5°`。
- PRG/G1 `config_act.yaml::euler_convention` 都显示默认 `XYZ`，但 `run_mjwp.py` 的实际转换发生在 `setup_env()` 之前、直接读邻接 meta；config 字段并不记录真实 runtime convention，不能替代缺失的 E173 stdout。当前代码版本 `dce760e`、分支 `experiment/E161-surface-release-ablation`，worktree 保持既有 dirty 状态。
- 2026-08-12 closure resumed：已按实验规划、报告与 Markdown/Mermaid 规范恢复 plan223、log273–275 和全部诊断证据；准备生成不可变 log276。最终结论冻结为：七条 orientation 长尾由 G1 expansion 的 Euler meta/reference mismatch 主导，gravcomp 的独立姿态效应在当前受污染对比中无法识别，需修复 reference parity 后受控复跑。
- 已完整复读本轮适用技能及其 Markdown、Mermaid flowchart 与 experiment-log 模板约束；报告将以内部 artifact/代码为引用，不伪造外部来源，流程图不使用 inline style，并保留单一 H1 与可复现命令。
- closure evidence review：重新从 JSON/TSV 核对 all-72、object/session cluster、七条 case、逐帧 onset/peak、reference parity 与机制审计；并逐行定位 `run_mjwp.py` 的 meta-missing→`XYZ` fallback、builder 的 scene+trajectory-only completeness 判定和 Ada launcher 未同步 meta 的 allowlist。上述代码链与 72-case replay 数值一致，将作为 log276 的单一 H1 根因。
- 补充 cross-object 审计：box001 match/mismatch=`7/21`、对应 mean delta=`−1.138/+3.878°`；box023=`8/8`、`−0.058/+5.768°`；box021=`28/0`、总体 `−0.218°`。第八条 `delta>5°` 是已排除的 `box001_20231023_110_p1`（`+5.555°`），同样为 `XYZ`→XML `XZY` mismatch；因此 8/8 大回退均属于同一 reference family。
- artifact hash 枚举时 brace list 误包含不存在的 `...outlier_summary.tsv`，`sha256sum` 单项报 no such file；真实 summary 是 JSON，随后已取得 SHA `67a18e...`，其余 artifact 未受影响。
- 已新增不可变诊断日志 `log/276_E194_G1_object_orientation_outlier_diagnosis.md`：完整记录 distribution、cross-object/session、8 条 `delta>5°`、runtime/XML reference parity、逐帧/可视化、代码漏洞、gravcomp COM 力矩边界、measured/inferred/unresolved 与修复复跑方案。日志未改写 log273–275，也未把 PRG runtime convention 写成已知事实。
- 已更新 E194 tracker：headline 改为 Euler reference mismatch 主导 orientation 长尾，状态明确要求修复 meta 同步后重跑 29 条 mismatch，并把 log276 链接放在 log275/274/273 前；原有历史日志和人工评审结论均保留。
- log index 已重建：275 个实际日志文件，Phase 57 已包含 log276。Python compile、focal reference regression 与 `git diff --check` 均 PASS。
- 首次 Markdown 禁用模式审计把 literal `{` 放进错误的 ripgrep regex，产生 parser error；该复合 shell 因 `|| true` 未传播失败，不能视为有效检查。下一步改用 `rg -F` 分别检查 literal `§`/`%%{init}`，并单独检查 inline style，避免 false pass。
- 第二次 Markdown 审计发现 `rg -c` 零匹配时变量为空，数值 `test` 报“需要整数表达式”；以 `${var:-0}` 规范化后第三次检查有效 PASS：H1=`1`、H2=`10`、placeholder/init/inline-style=`0/0/0`。这是检查脚本问题，不是日志内容问题。
- 72-case 主诊断 runner 最终复跑 exit 0：frame `16 case / 2,089 rows`、reference `72/72`、full-substep `144/144`；headline 精确复现 all72 `+1.570/−0.065°`、box001 primary27 去 top3 `+0.361°`、box023 去 pair4 `−0.00048°`、box021 `−0.218°`。runner 的统一执行会先由 Codex async cell 返回 PTY session，再经 `write_stdin` 回收；这不是实验错误。
- 最终 cardinality/JSON assertion/hash/diff audit PASS。runner 会刷新 summary JSON 的 `created_at`，因此本次复跑后 JSON SHA 从此前 `67a18e...` 变为当前 `649486...`；日志已更新为当前 snapshot hash，所有 TSV hash 保持不变。
