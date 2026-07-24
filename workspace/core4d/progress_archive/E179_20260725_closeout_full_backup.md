# CORE4D E179 最终收尾完整备份

> 完整备份：
> [E167–E179 初始恢复](progress_archive/E167_E179_20260724_full_backup.md) ·
> [E179 当前执行明细](progress_archive/E179_20260724_execution_full_backup.md)
>
> 本文件只保留活跃 E179 的冻结事实、关键故障与未决项。

## E179：box023 / E167A no-PRG / Full CEM

### 冻结口径

- 计划：`plan/197_E179_box023_e167a_no_prg_full_cem_plan.md`。
- Population 为 E173 box023 全部 16 条 CEM-eligible case；raw 漏斗仍记录
  `46`，但 E179/E173 paired denominator 固定为 `16`，不可比较完成子集。
- Treatment=`E167A_zOnlyBody`；保留 hand/posture/surface/z-only，禁止 E170
  PRG lower-body pairs、penalty、candidate gate、fallback 和 runtime diagnostics。
- Full budget=`seed 0, 1024 samples × 32 opt steps`；canary=`64×4`。
- 资源固定：本地 RTX 5090 GPU0=`4` 条；远程 A100 GPU
  `2,3,6,7` 各=`3` 条。用户最新明确授权四卡叠加运行；两次卡号/显存
  `<5000MiB`/policy 检查保留，compute 只保存快照而不阻断，不换卡、不 kill。
- 主指标为 physics 6 门 + root/EEF/object pos/ori 共 12 门；
  missing/non-finite=FAIL；contract=`core4d-e179-12gate-tracking-v1`。
- E173 只读 baseline：physics6=`13/16`，同 adapter 12-gate=`7/16`。
- Authority SHA：E173 manifest=`2128deb…889d`；
  E167A profile=`666c302…48c17`。

### Authority / handoff / queue 已完成

- E173 paired authority=`16 unique`、`omnirt_v1/v2=15/1`；Stage2b inputs 已从
  E173 authority 事务恢复，trajectory SHA exact=`16/16`、scene semantic
  exact=`16/16`。
- Fresh E179 rubber scene/config audit：
  scene=`16/16`、E167A profile=`16/16`、no-PRG=`16/16`、
  E173 pre-PRG rubber semantic exact=`16/16`。
- Full manifest SHA=`ce1e4f82c43b6cda69604b06797c858f046945f5ba9722fee1c681b07a5adf12`。
  分片=`4+3+3+3+3`，并集/unique=`16/16`，所有 queue dry-run PASS。
- 已用 `git add -f` 精确跟踪 16 条的 `scene.xml`、pristine `scene_act.xml`、
  E179 sidecar 和 override，共 `64/64`；正式 canary 前已调用 scene snapshot。

### 已实现并验证

- 实验脚本：
  `e179_common.py`、paired authority/restore/manifest/audit、独立
  `run_cem_queue.py`；Python compile、builder apply/snapshot 与 negative audit PASS。
- 执行入口：
  `run_E179_local.sh`、`run_E179_remote_a100.sh`、
  `pull_E179_remote_a100_results.sh`、`watch_E179_full.sh`；shell syntax PASS。
- 本地 PREP_ONLY 实测：snapshot/audit/dry-run 与 GPU0 两次
  `150MiB/0 compute` PASS，未启动作业。
- Remote negative-gate test：canary 未齐时在 SSH sync/tmux 前正确退出。
- 评测：
  `eval_E179_box023_12gate.py`、paired report generator/wrapper；synthetic
  identity test=`16 paired / 192 gate cells`。
- 可视化/验收：
  `render_paired_results.py`、`run_E179_render_all.sh`、`audit_completion.py`；
  E173 baseline video=`16/16`，当前 completion audit=`5/11`（Full 尚未运行）。

### Canary attempt-1 与修复

- Attempt-1 三条优化完成但被正确判 `failed_validation`：公共 runtime 在 PRG
  inactive 时仍无条件保存零初始化的 `cem_leg_gate_*` /
  `leg_object_penalty_*` 聚合诊断。
- 失败证据已归档到
  `results/E179/s6_downstream/cem/attempts/attempt1_pre_runtime_fix/` 和
  `logs/E179/cem/attempts/attempt1_pre_runtime_fix/`，未删除。
- 已修 `spider/simulators/mjwp.py`：PRG info keys 只在相应 penalty/gate active
  时保留；compile PASS，E169 active leg-gate regression PASS。

### 当前运行状态（2026-07-24）

- Fresh canary session=`e179_local_canary_20260724_205326`。
- 第 1 条 `045_p1` 已 `run_complete_pending_eval`：
  qpos `(134,2,42)` finite，PRG diagnostic prefix=`0`，E167A/no-PRG config PASS。
- 第 2 条 `018_p2` 也已 `run_complete_pending_eval`：qpos
  `(116,2,42)` finite、PRG diagnostic prefix=`0`。第 3 条 `019_p2`
  正在 GPU0 运行。
- Fresh canary 三条日志均冻结命令参数
  `seed=0 num_samples=64 max_num_iterations=4`；前两条 effective config
  同样记录 `seed=0/64/4` 与 E179 scene。第 3 条已于 `20:58:05` 启动，
  当前仍正常运行。
- Fresh canary 已结束，三条 `validate_runtime_outputs` failures 均为空；
  16-row E167A/no-PRG audit 也再次 PASS。因此 canary 的配置、artifact、
  finite qpos 和 runtime negative contract 为 `3/3 PASS`。
- 首次 canary evaluator 用 `MUJOCO_GL=osmesa` 启动时，本机 PyOpenGL/OSMesa
  初始化报 `AttributeError: glGetError`，未生成 metrics；这是 GL backend
  选择错误，不是数值/数据失败。改用本机 CEM 已验证的 `egl` backend 重跑，
  不重复 osmesa。
- EGL canary evaluator PASS：`3 rows / 36 gate cells / errors=0`。低预算
  E179 physics6=`0/3`、12-gate=`0/3`（E173 subset=`2/3`、`1/3`）；
  这是 canary 质量信号，不违反“质量不阻断 Full”的冻结规则。
- 本地 4-row Full 已于 `21:01:24+08:00` 启动，
  session=`e179_local_full_20260724_210120`。启动前 GPU0 再次连续两次
  `150MiB/0 compute`；第 1 行已进入 `running`。
- 本地第 1 条 Full 已进入稳定 `32` opt-step 区段，约
  `9.7–10.2s / 2 sim steps`，当前 `40/268`；按现场速率单 case 约
  20–25 分钟，4 条在同卡严格串行。
- 本地第 1 条 `045_p1` 于 `21:22` 完成，runtime/no-PRG validation failures
  为空，final object error=`0.0866m / 0.0662rad`；GPU0 已切到第 2 条
  `046_p1`。
- 远程只读预观察：GPU2/6/7 空闲，GPU3 有 `626MiB` compute context。用户随后
  明确要求直接在 `2,3,6,7` 叠加运行、不等待；resource waiter 已停止。
- Remote PREP_ONLY resource-gate test 在 canary PASS 后仍正确阻断：
  `fixed A100 preflight failed: gpu3:compute_process_present`，未 rsync、未启动
  tmux。远端 `ps` 找不到该 PID 的用户进程，`nvidia-smi` 仍持有 compute
  context；按规则继续视为占用，不自行清理。
- 本地 Full 第 1 行正常运行，GPU0 当前约 `1790MiB/55%`。
- 用户随后明确授权远程固定 GPU `2,3,6,7` 叠加运行、不等待。计划与 launcher
  已更新：compute 仍保存两次快照但不阻断；卡号、显存 `<5000MiB` 和 policy
  仍双检；resource waiter 已停止并删除。
- 第一次 overlap remote launch 的 check1 PASS
  (`4/635/0/0MiB`)；bulk rsync 经网络 timeout 重试后进入远端验证，但
  launcher 漏同步 `sync_inventory.json` 本身，SHA preflight 因 missing inventory
  终止，未启动 tmux/CEM。修复为 bulk sync 后单独复制 inventory，并复用同一
  remote run root 增量续传。
- 增量重启已成功：远端 SHA verified=`2036 files`，check2 仍为
  GPU2/3/6/7=`4/635/0/0MiB`，`compute_overlap_authorized=true`。
  Remote Full session=`e179_a100_full_20260724_210702` 已启动。
- 四个远程 shard 均已首行 `running`、每卡余下两行 `READY_FOR_FULL`：
  GPU2=`040_p1→045_p2→042_p2`，GPU3=`018_p1→039_p2→041_p2`，
  GPU6=`021_p1→041_p1→042_p1`，GPU7=`046_p2→021_p2→040_p2`。
  `e179_full_watcher` 已启动，每 120s 保守检查本地/远程结束后自动 pull。

### 下一步

1. 等 fresh canary 三条结束；逐条 runtime/no-PRG audit，并跑 canary 12-gate。
2. 本地 4-row Full 已启动；按 overlap 授权对 A100 `2,3,6,7` 做两次
   卡号/显存检查后同步并启动 12-row remote Full。
3. Pull/merge 16 行，跑 12-gate paired eval/report。
4. 渲染 E179/E173/new paired 各 16 个视频；使用 `video-frames` 提取并记录
   16 条实际观察。
5. completion audit 全过后写 E179 log、更新 Tracker、commit/push。

### 2026-07-24 21:24 监控恢复

- 本地 Full：`045_p1` 已完成，`046_p1` 仍在 GPU0 运行。
- watcher 持续报告 `local=1 remote=1 absent=0/2`。
- 直接在 `spider-remote` 查询同名远程 tmux 时未找到 session；先核对 launcher
  实际 SSH host/tmux target，不据此重启、不重复提交 12 条远程作业。
- 已确认这不是远程作业丢失：E179 A100 launcher/watcher 的实际 endpoint 是
  `batchcom@61.172.170.106:30409`；`spider-remote` 是另一台常用两卡机。
- 正确 endpoint 上 tmux `e179_a100_full_20260724_210702` 存活
  (`bash`, `dead=0`)；GPU `2/3/6/7` 当前显存约
  `1551/2183/1551/1551 MiB`、利用率 `40/40/37/41%`，四卡均在工作，
  无需等待或重启。
- 本机未安装 `jq`（只读检查返回 `jq: command not found`）；已改用
  `sed`/Python 读取 JSON，不重复依赖 `jq`。远程四个首 case 均持续推进，
  暂无 case 完成或失败。
- 已完整复核 E179 plan、远程执行规范与最新 E178 log；当前执行未偏离冻结
  contract：paired denominator=`16`、primary=`12 gates`、A100 固定
  `2,3,6,7` 且 overlap 已授权。
- 后处理入口复核：watcher/pull/12-gate eval wrapper 内容与 shell syntax 正常；
  交接中写的 `scripts/experiments/E179/run_E179_render_all.sh` 路径不存在，
  正在定位 canonical render wrapper，CEM 不受影响。
- canonical render wrapper 已定位为
  `scripts/launch/active/run_E179_render_all.sh`；其默认 backend 已从本机失败过
  的 `osmesa` 改为已通过 canary 的 `egl`，显式 `MUJOCO_GL` 仍可覆盖。
- render/watcher/pull/eval shell syntax 与 render/eval/report/audit Python
  compile 均 PASS。`21:29` 本地/远程 Full session 仍存活，watcher
  `absent=0/2`。

### 2026-07-24 21:50 进度与 ETA 检查

- Full 当前完成 `3/16`：本地 `2/4`（`045_p1`,`046_p1`），远程
  `1/12`（GPU7 `046_p2`）；本地与 A100 四个 worker session 均存活。
- 正在运行：本地 `018_p2=208/232`；远程 GPU2
  `040_p1=188/214`、GPU3 `018_p1=188/260`、GPU6
  `021_p1=182/260`、GPU7 `021_p2=42/258`。
- A100 GPU `2/3/6/7` 显存约 `1551/2183/1551/1551 MiB`、利用率
  `39/40/40/37%`；日志无 Traceback/Error/Killed，watcher 连续
  `local=1 remote=1 absent=0/2`。
- ETA 计算脚本首次读取 manifest 时误用不存在的 `worker` 字段而 KeyError；
  canonical 字段为 `assigned_worker`，修正后再按真实序列长度估算关键路径。
- 已按 16 条 trajectory 的真实 frame 数修正 ETA：本地剩余约 `17–22min`
  （预计 `22:08–22:13` 完成）；远程关键路径在 GPU6，按现场稳定的
  `24–25.5s / 2 sim steps` 估计还需约 `1h40–1h50`，Full CEM 预计
  `23:30–23:45` 全部结束。
- watcher 双重确认、远程 pull、16 条 runtime/no-PRG 审计与 12-gate eval
  预计再需 `20–35min`，数值对比预计 `23:55–00:20`；16 条离线渲染和视觉
  复核再预留 `40–70min`，完整报告闭合预计 `00:40–01:30`。区间已包含
  A100 overlap 导致的当前较慢吞吐，不按空载 A100 乐观外推。

### 2026-07-25 00:30 Full 完成与收尾恢复

- 用户要求按 `experiment-planning-zh` 完成回收、EGL 渲染、12-gate 评测、
  `video-frames` 视觉复核、completion audit、E179 log、Tracker 与 git 收尾。
- Full CEM 两端 session 均已结束：本地 `4/4`、A100 GPU
  `2/3/6/7` 各 `3/3`，远端 root NPZ=`12`，所有 worker event 均为
  `run_complete_pending_eval`；合计计算完成 `16/16`，无 terminal failure。
- watcher 于 `23:35` 达到 `local=0 remote=0 absent=2/2` 后退出；本地只生成
  `pull_files.txt`、未生成 `pull_summary.json`，Full 目录仍只有本地
  `4` 个 root NPZ，故自动 pull 未闭合。下一步运行同一 canonical pull
  幂等回收，不重跑 CEM。
- 手动 canonical pull 已启动；前两次单文件 rsync 均在 30 秒触发
  `Receiver io timeout`，脚本仍按内置最多 6 次策略保留 partial 并重试。
  这是网络回收错误，不是远端 CEM/artifact failure，不启动重复计算。
- canonical pull 最终在同一 `045_p2` root NPZ 上连续 `6/6` 次超时退出；
  远端文件存在且为 `8,573,214 bytes`，SHA256=`680f4c2d…adff`。根因是
  wrapper 单文件 `--timeout=30` 对当前链路过短，而非 artifact 缺失。
- pull wrapper 已增加
  `E179_PULL_IO_TIMEOUT_SECONDS`（正整数校验，默认 `180s`），继续幂等回收；
  这是三次失败协议下的替代传输策略，不进行第 7 次原样重试。
- 180s 策略首轮不再触发 rsync I/O timeout，但约 150s 后 SSH
  `ServerAlive` 判定 A100 endpoint 不响应并断开（rsync code 12）；同一脚本
  正在重试。若仍失败，将改用不同传输机制而非继续调大 rsync 参数。
- 第二轮 180s 长连接也被 SSH `ServerAlive` 断开；已精确终止本地 pull
  shell PID `3540346` 及其 rsync child `3545246`，远端文件/session 未修改。
  下一步用独立 `scp` 试传同一 8.6MB 文件并核对 SHA。
- `scp -O` probe 在 `1.7s` 内成功回收同一文件，大小
  `8,573,214 bytes`、SHA256=`680f4c2d…adff` 与远端完全一致。已给
  canonical pull 增加 `E179_PULL_FILE_TRANSPORT=rsync|scp`（默认仍
  `rsync`），本轮改用 `scp` 回收 manifest 登记文件。
- `E179_PULL_FILE_TRANSPORT=scp` canonical pull 于 `00:46` 完成：
  remote rows=`12`、status=`12 run_complete_pending_eval`、
  runtime pass=`12/12`、pull summary=`pass`；worker shards 合并后
  canonical Full manifest=`16` rows、`16/16 run_complete_pending_eval`、
  merge status=`pass`。
- Full `audit_e167a_no_prg.py --require-all` 于 `00:47` PASS：
  rows/unique=`16/16`、method parity=`16/16`、no-PRG=`16/16`、
  audit failures=`0`，PRG runtime diagnostics expected=`false`。
- 当前 completion audit 已过 `8/11`：Full execution/runtime、paired authority、
  `4+3+3+3+3` worker、64 个 tracked scene/override、scene snapshot 与固定五卡
  execution manifests 全 PASS；只剩 12-gate metrics、E179/paired videos、
  visual review 三项。
- EGL Full evaluator 于 `00:48` PASS：evaluated=`16/16`、errors=`0`、
  gate cells=`192`。E173→E179 physics6=`13→9/16`，12-gate=`7→4/16`；
  迁移=`PASS_TO_PASS 3 / FAIL_TO_PASS 1 / PASS_TO_FAIL 4 / FAIL_TO_FAIL 8`，
  预注册 numeric verdict=`PRG_BETTER`。
- 单门主要变化：lower-body=`14→10`、hand-ori=`8→7`；root-pos=`10→11`、
  EEF-pos=`11→13`、object-pos=`15→16` 有改善。连续 delta 显示
  leg penetration mean `+0.0736`（恶化），root/EEF tracking mean error
  `+2.63/+2.71cm`（整体恶化，虽单门通过数改善）；视觉 verdict 仍保持
  `PENDING_16_PAIRED_VIDEO_REVIEW`，不提前代判。
- 逐 case 关键迁移：唯一救回=`021_p1`；四条新退化=`041_p1`
  (hand_ori)、`021_p2` (lower_body)、`040_p2`
  (lower_body+hand_ori)、`041_p2` (lower_body)。三条保持 PASS 为
  `045_p1/046_p1/042_p2`。
- EGL render preflight：本地 GPU0=`150MiB/0%`、ffmpeg
  `4.4.2` 可用；开始渲染前 Full 目录尚无 E179/paired 视频。
- EGL render 于 `00:50` PASS：baseline ready=`16`、E179 treatment
  rendered=`16`、paired rendered=`16`、failures=`0`。
- 首次 ffprobe 检查误把 treatment 视频 glob 写成
  `s6_downstream/cem/full/*.mp4`，该路径无视频而产生 `1` 个 shell-level
  failure；16 个 paired MP4 实际均可解码。下一步从 manifest 的 canonical
  `video` 列读取 16 个 treatment 路径后重检，不重复渲染。
- 按 manifest canonical `video` 列重检后，E179 treatment=`16/16`、
  paired=`16/16`，ffprobe failures=`0`；E179 分辨率 `1440×480`，
  paired=`1920×540`，帧数与各 trajectory 完全匹配。
- 已使用 `video-frames` 对每个 paired MP4 抽取
  `10/30/50/70/90%` 五帧：keyframes=`80`、contact sheets=`16`，并逐条
  实际查看。
- 视觉总观察：三条 lower-body 新退化 `021_p2/040_p2/041_p2` 在深蹲/搬运
  阶段均可见 no-PRG 下肢更贴近或跨入箱体；`042_p1` 的既有 FAIL 也显示
  no-PRG 深蹲更深。`041_p1` 仅 hand-orientation 退化，稀疏帧中不构成
  明显整体动作崩坏；唯一救回 `021_p1` 未见新增姿态异常；三条
  PASS→PASS 整体轨迹一致。
- `visual_review.tsv` 已写入 `16` 条 reviewed authority，逐条绑定 paired MP4、
  contact sheet、10/30/50/70/90% 采样、numeric migration/failure mode 与
  具体视觉观察；不代填用户人工标签。
- `audit_completion.py --require-all` 于 `00:54` PASS：`11/11` checks
  全部通过、failures=`[]`；metrics=`16`、paired=`16`、gate cells=`192`，
  videos=`E173 16 / E179 16 / paired 16`，visual review=`16/16`。
- 写日志前发现 C4 统计证据缺口：原 evaluator 未输出计划要求的 IQR、
  seed-0 paired bootstrap 95% CI、12 门逐门迁移和 exact McNemar；
  completion audit 只检查 cardinality，未覆盖这些字段。已补 evaluator 与
  report generator，下一步 compile/unit check 后重跑同一 EGL 只读评测。
- `experiment-report-writer` 安装目录缺少其声明的
  `templates/experiment-report.md`（仅有 `SKILL.md`）；最终日志按技能
  Required Report Structure 与项目既有 log 风格写，不因此阻塞。

### 2026-07-25 统计证据补齐

- 更新后的 `e179_eval_summary.json` 状态为 `pass`，正式视觉判定为
  `SUPPORTS_PRG_BETTER`；报告已纳入 IQR、固定 seed `0` 的 `10000` 次
  paired bootstrap mean 95% CI、12 门逐门迁移和 exact two-sided McNemar。
- 12-gate 迁移 `PASS_TO_FAIL=4 / FAIL_TO_PASS=1`，McNemar exact
  `p=0.375`；lower-body `PASS_TO_FAIL=4 / FAIL_TO_PASS=0`，
  exact `p=0.125`。小样本离散检验未达显著，但 leg penetration 的 paired
  mean delta=`+0.07363`，bootstrap 95% CI=`[+0.03704,+0.11427]`，
  与视频中的下肢退化方向一致。
- 下一步仅剩最终 completion audit、E179 结果日志、Tracker/INDEX 和
  E179 窄范围 git commit/push；不回写 E173 baseline 结果。
- 最终 `audit_completion.py --require-all` 于 `01:00` 再次 PASS：
  `11/11` checks、failures=`[]`；执行/审计=`16/16`、paired metrics=`16/16`、
  gate cells=`192`、E173/E179/paired videos=`16/16/16`、视觉复核=`16/16`。
- 最终日志采用 `experiment-report-writer` 的 single-experiment/ablation
  证据结构，并按 `markdown-mermaid-writing` 统一一个 H1、H2 emoji、表格和
  文本化关系图；观测、解释、局限和决策分节记录。
- Markdown 日志格式已冻结：一个 H1、标题后元信息、H2 单 emoji、结果用表格、
  technical values 用 code；项目内部 evidence 采用相对链接，不引入外部来源。
- 日志关系图选用简洁 Mermaid flowchart：同一 16 条 authority 分流为
  E173 PRG 与 E179 no-PRG，再汇聚到 12-gate、paired video 和保留 PRG
  的决策；图中加入 `accTitle/accDescr`，不使用 inline style。
- 已复核 Tracker 与最新 E178 日志风格；E179 使用下一个日志号 `242`。
  Tracker 将把原 `E179 plan` 行替换为完成态，并保留 plan 与新 log 双链接；
  描述只写“16条paired no-PRG退化，保留PRG”，细节全部留在 log。
- 复现信息确认：launch code HEAD=`64f9a33f…5448`，本地 RTX 5090 GPU0
  分片=`4`，A100 GPU `2/3/6/7` 各=`3`，remote overlap 明确授权；
  method/no-PRG audit=`16/16`。`scp_probe` 的 8.6MB 临时 NPZ 与 canonical
  NPZ 大小及 SHA256 (`680f4c2d…adff`) 完全一致，可安全精确清理临时副本。
- 已精确删除 `s0_environment/scp_probe.GBPDeX/` 临时重复副本，canonical NPZ
  仍存在。E179 最终日志已写入
  `log/242_E179_box023_e167a_no_prg_vs_e173_results.md`，包含 setup、十二门、
  bootstrap/McNemar、16条视觉复核、C0–C6、故障恢复和复现命令。
- 日志结构与全部相对链接检查通过；复现命令审查发现远程 launcher 的真实
  overlap 变量为 `E179_ALLOW_COMPUTE_OVERLAP`，而不是草稿中的
  `A100_COMPUTE_OVERLAP_AUTHORIZED`。下一步修正该日志命令并移除 evaluator
  wrapper 已内置的冗余 `--require-all` 参数。
- 日志复现命令已修正并与真实 wrapper 对齐；Tracker 的 `E179 plan` 已更新为
  完成态 `E179`，摘要为“16条paired no-PRG退化；保留PRG”，状态记录
  `16/16`、十二门 `7→4` 与 `PRG_BETTER`，并链接 log 242 和 plan 197。
- `build_log_index.py` 已重建 `log/INDEX.md`，log 242 已进入 Phase 42 索引；
  Tracker 描述长度=`43`（≤80）。提交前 `git diff --check`、E179 Python
  compile 与全部 shell `bash -n` 均 PASS。
