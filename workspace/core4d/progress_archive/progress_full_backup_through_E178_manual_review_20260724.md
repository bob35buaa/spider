# CORE4D 当前进度

> 历史完整备份：
> [progress_full_backup_through_E177_proxy_20260724.md](progress_archive/progress_full_backup_through_E177_proxy_20260724.md)
>
> 本文件只保留当前活跃实验的最终状态与未决项。

## E177：三种 bucket low-geom proxy

### 最终 scope

- desk 暂不考虑；弃掉低数量 bucket009/bucket010。
- E177 authority：
  `bucket003=9`、`bucket004=4`、`bucket007=14`，共 27 条。
- 当前只完成本地 proxy、scene contract 与视觉 review；未运行
  ref-contact gate、canary 或 Full CEM。

### 用户批准的几何

- bucket003：沿 object-local Y 的 5 个实心 body boxes，无独立 lid。
- bucket004：单个 mesh AABB box。
- bucket007：沿 object-local Y 的 5 个实心 body boxes，无独立 lid。
- geom counts=`5/1/5`；robot–object pairs=`90/18/90`。
- bucket003/007 相邻 Y 分段保留 4mm overlap；XZ 截面分别使用
  `0.94/0.82` inward scale，避免 proxy 明显大于 mesh。
- 每个 geom 同时进入 robot–object physics pairs 与 PRG box-union SDF；
  group batching 保持开启。

### 已通过证据

- 27/27 sidecar MuJoCo compile、pair matrix、authority parity PASS。
- union mesh→proxy / proxy→mesh p90：
  - bucket003：`2.43/2.14cm`
  - bucket004：`2.21/3.03cm`
  - bucket007：`3.36/3.65cm`
- 四视角 overlay 与 XY/XZ/YZ、`-Y base`、`+Y end` 截面完成。
- 用户于 2026-07-24 明确批准使用该版本；三对象 review 已落为
  `clean_reviewed/approve_clean`，builder `review_pending=[]`。
- E175 PRG multi-box union、renderer 六对象兼容与 `git diff --check` PASS。

### 持久记录

- Plan：
  `workspace/core4d/plan/193_E177_bucket_semantic5_proxy_plan.md`
- Log：
  `workspace/core4d/log/236_E177_five_step_no_lid_bucket_proxy_results.md`
- Results：
  `workspace/core4d/results/E177/`
- 通用准则已写入：
  `.codex/skills/data-construction-v3-zh/SKILL.md` v1.1.0。
- Log INDEX、Tracker、E177 summary contract 与最终 `git diff --check`
  已复核 PASS；`progress.md` 已从 285+ 行归档压缩为当前活跃状态。

### 下一步

1. 跑 27 条 ref-FK contact fidelity，三个 object-grouped p90 必须≤8cm。
2. 通过后运行三对象 `64×4` canary，验证 runtime 与 median plan time≤3s。
3. 仅在 canary 通过后，才允许 GPUs `2,3,6,7` 启动 27-case
   `1024×32` Full CEM。

### 2026-07-24 接续执行

- 用户已授权按上述 gate 继续推进。
- 已恢复 E177 plan/log/tracker/progress，并开始复核 E176 contact fidelity
  evaluator；尚未启动 contact、canary 或 Full。
- 远程 Full 固定使用 A100 `2,3,6,7`；但仍须先满足 contact 与 canary gate。
- 已完整复核 `experiment-planning-zh` 与 A100 remote-execution 规范。用户此前
  已显式授权固定使用 `2,3,6,7`、不因其他程序改卡；启动时仍保存 GPU/process
  snapshot，但不做动态换卡。
- 已完整复核 `data-construction-v3-zh`，并确认 E176 的可复用 contact
  evaluator 为 `scripts/eval/runners/eval_E176_contact_fidelity.py`；E176
  既有结果 contract 位于 `results/E176/s2_proxy/contact_fidelity/`。
- E176 evaluator 的核心 contract 与 E177 manifest 兼容：从 E174 source
  config 恢复 `ref_fk` 目标、读取 3cm active mask、检查 hand pair 覆盖全部
  object boxes，并输出 case/object/contact-row TSV + JSON/MD。需在 E177
  runner 中把固定 authority 从 `39 rows/6 objects` 改为 `27 rows/3 objects`，
  默认路径和实验标识改为 E177；几何上限仍为 `≤9`，适配当前 `5/1/5`。
- 已固化 E177 runner/wrapper：
  `scripts/eval/runners/eval_E177_contact_fidelity.py` 与
  `scripts/eval/wrappers/eval_E177_contact_fidelity.sh`；E176 engine 增加
  向后兼容的 expected cases/objects 参数化，默认行为仍为 39/6。两份 runner
  `py_compile`、wrapper `--help`、scoped `git diff --check` 均 PASS。
- 27-case ref-FK contact fidelity 已完成，27/27 可计算、0 errors，但 Gate B
  **FAIL**，因此未启动 canary/Full：
  - bucket003：p90=`0.08059m`，比 8cm 门高 `0.59mm`（borderline fail）；
  - bucket004：p90=`0.07480m`（PASS）；
  - bucket007：p90=`0.11258m`（FAIL，超门 `3.26cm`）。
- 结果已持久化到 `results/E177/s2_proxy/contact_fidelity/`。下一步先定位
  bucket007 的 contact under-coverage（与 0.82 inward scale/五段截面关系），
  不越过 gate。

### 遇到的错误

| 错误 | 尝试次数 | 处理 |
|---|---:|---|
| contact-row ad-hoc 聚合把 `hand` 错当成 `0/1`，空数组 quantile 报错 | 1 | 先读取实际 hand label 后按真实值重跑，只读分析；不影响正式 gate 结果 |
| 直接按 manifest 相对 scene 路径 `rg` 未命中 | 1 | evaluator 已通过 fallback resolver 找到 scene；后续从 contact row 的 resolved `scene_act` 或 E177 snapshot 查找 |
| E178 A100 prep-only exact rsync 遇到 SSH server timeout / broken pipe | 3 | inventory 仅约254MB；第三次长连接也超时，但 bounded retry 随后完成 base sync 并已生成 3-row queues，继续 remote preflight，tmux/CEM 未启动 |
| prep-only 后本地直接读取 `remote_sync_verification.json` 不存在 | 1 | 该文件只写在 remote run root，prep-only 不执行 pull；以控制台 1790/1790 PASS 与 execution manifest 为当前证据，正式 pull 时回收 |
| remote progress probe 错用 run-root `.venv/bin/python` | 1 | run root 只同步 exact files，不含 venv；后续使用 execution manifest 中的 canonical remote Python，不影响 canary |

### Contact gate 初步归因

- bucket007 的 1677 个 active targets 中，1665 个最近 geom 都是第 5 段；
  contact local-Y p05/p95=`0.1965/0.2879m`，说明 ref 接触几乎全集中在 `+Y`
  端段，不是下方四段或接缝 miss。
- bucket007 有 `581/1677=34.65%` rows 相对 visual mesh under-cover >3cm；
  左/右手均存在，不能归因于单手 case。第 5 段当前 X/Z half-size 仅
  `0.2206/0.2321m`，0.82 inward scale 是主因。
- 不改 geom 数、只扩大第 5 段 XZ 的只读敏感性审计：
  bucket003 scale `0.94→0.95` 时 contact p90 `8.059→7.828cm`；
  bucket007 scale `0.82→0.94` 时 `11.258→7.993cm`。因此最小 grid
  过线组合是 top segment `0.95/0.94`，而不是加 lid 或增加 geoms。
- 该组合尚未改正式 proxy；需先复核双向 mesh/proxy p90 和 overlay，随后
  才能决定是否形成新的 reviewed geometry 迭代。
- 已复核 E177 builder：当前实验标识、sidecar、override、manifest 与 review
  路径均按 E177 冻结，不适合直接覆盖。后续若落地 contact-aligned candidate，
  将用新 E178 plan/result/sidecar 保留 E177 历史，先跑本地几何与视觉 gate。
- 已创建 E178 plan
  `plan/194_E178_bucket_contact_aligned_top_segment_plan.md`，并固化 candidate
  实现/测试/builder/contact evaluator/wrapper。E177 builder 仅做向后兼容的
  experiment-id 参数化，E177 默认 sidecar/路径/几何不变；E178 使用独立
  `scene_act_E178_contactAlignedTop`、override 与 `results/E178/`。
- E177 回归 PASS，默认几何指标未漂移。E178 首版测试中 bucket003/bucket004
  双向 union gate PASS，但 bucket007 top scale 0.94 的
  `union_proxy→mesh p90` 超过 4cm，Gate A 被阻断；尚未运行 builder/contact
  正式结果，更未启动 CEM。下一步量化具体超额并寻找“不增 geom、contact≤8cm、
  proxy→mesh≤4cm”的局部截面方案。
- bucket007 等比 0.94 的精确超额很小但真实存在：
  union=`3.313/4.031cm`。分轴只读搜索找到可行域，例如 top X/Z scale
  `0.96/0.90` 时 contact p90=`7.946cm`、union=`3.306/3.982cm`；
  `0.98/0.86` 时 contact=`7.899cm`、union=`3.333/3.933cm`。下一步在
  可行域内选更均衡、留有 margin 的截面后再改 candidate，并重新跑正式 gate。
- 细网格复核后选 bucket007 top X/Z=`0.97/0.885` 作为均衡候选：
  half-size=`0.2609/0.2504m`，contact p90=`7.913cm`，
  union=`3.315/3.952cm`；相比 `0.98/0.86` 更接近方形主截面，且两门均
  保留约 0.48mm deterministic margin。尚未写入正式 E178 结果。
- E178 plan 与实现已切到 top X/Z 分轴 scale：
  bucket003=`0.95/0.95`、bucket007=`0.97/0.885`。E177 regression PASS；
  E178 object-level geometry tests PASS，union p90：
  bucket003=`2.56/2.25cm`、bucket004=`2.21/3.03cm`、
  bucket007=`3.31/3.95cm`。scoped `git diff --check` PASS。
- E178 27-case builder 已完成：27 full + 3 canary rows，object distribution
  `9/4/14`，27/27 scene audit PASS，geom=`5/1/5`、pairs=`90/18/90`，
  max geoms=5。独立 sidecar/override/results 已生成；review 正确保留
  `bucket003/004/007` 三项 pending。
- E178 正式 ref-FK contact gate **PASS**：27/27 evaluated、0 errors、
  4137 active rows；object p90 为 bucket003=`7.828cm`、
  bucket004=`7.480cm`、bucket007=`7.913cm`，三者均≤8cm。结果位于
  `results/E178/s2_proxy/contact_fidelity/`。下一 gate 是 overlay/截面人工
  review；在重新批准前仍不启动 canary。
- E178 overlay renderer 3/3 PASS，已生成
  `e178_3_object_proxy_montage.png` 与
  `e178_bucket_cross_section_montage.png`；review TSV 仍为 pending，下一步
  进行本地图片审查并向用户展示，不自动批准。
- Codex 已检查两张原图：bucket003 只轻微扩大第 5 段，未见明显突变；
  bucket004 与 E177 相同；bucket007 顶段 XZ rectangle 约
  `±0.261×±0.250m`，覆盖圆形端部主轮廓，四角存在预期 box phantom，但
  没有形成整圈明显外扩、贯通 bridge 或段间漏缝，量化 exposed p90=3.95cm。
  该判断只完成 Codex 侧审查，不能替代用户最终 visual approval。
- 新日志 `log/237_E178_bucket_contact_aligned_proxy_gates.md`、Tracker 与
  log INDEX 已更新；E178 review TSV 已从 `PENDING_CODEX_REVIEW` 推进为
  `PENDING_USER_REVIEW`，未自动批准。全量相关 `py_compile` 与 scoped
  `git diff --check` PASS；工作树既有未跟踪 E176/E177 文件保持原状，未
  commit/push。
- 用户于 2026-07-24 明确确认 E178 视觉“没问题，继续”。本轮已重新复核
  `data-construction-v3-zh` 与 `experiment-planning-zh`；下一步把三对象
  review 落为 `approve_clean`，再按 gate 启动 3-case `64×4` canary。
- 已完整复核 A100 remote-execution 规范并开始审计 E176 可复用 launch/pull、
  queue runner、runtime/throughput validators。E178 canary 必须改为 3-row
  authority 与 E178 独立路径，不能直接沿用 E176 的 6-row validator 默认值。
- 三对象 review 已依据用户批准落为 `clean_reviewed/approve_clean`；log 237
  与 Tracker 已更新为 visual PASS。下一步先用 builder `--require-review`
  复核 release contract，再固化 E178 canary launch/pull/validator。
- E178 builder `--require-review` PASS：`review_pending=[]`、27/27 scene
  audit、27 full rows、3 canary rows，release contract 已正式放行 canary。
- E176 A100 launcher/pull/watcher 已完整审计。复用前必须参数化实验 ID、
  manifest basename、expected rows、result/log prefixes 与 gate/evaluator
  路径；E176 默认行为保持不变，E178 通过 canonical active wrappers 注入。
- 已完成向后兼容参数化并新增 E178 active launch/pull/watch wrappers。
  `bash -n`、validator `py_compile/--help`、scoped diff check PASS；E178
  `SNAPSHOT_ONLY` 生成 1790-file exact sync inventory，未 SSH、未启动 CEM。
- A100 canary prep-only 最终 PASS：remote exact inventory `1790/1790`，
  selected GPUs=`2,3,6,7`、fixed mode=true、3-row queue loads=`208/145/83/0`
  frames，execution status=`preflight_passed_no_launch`；无 tmux/CEM。
- E178 3-case `64×4` canary 已正式启动：
  session=`e178_a100_canary_20260724_113702`，fixed GPUs=`2,3,6,7`，
  exact sync `1790/1790` PASS；三条 case 分配到 GPU2/3/6，GPU7 为空队列，
  未复制 case。下一步监控 tmux，完成后 scoped pull + runtime/throughput gate。
- 首次监控：tmux 存活，GPU2/3/6 均有 canary 负载；GPU7 worker 正确输出
  `[no-rows]` 并未运行重复 case。三个活跃 worker 尚未完成/flush 日志。
- 三条 case 已进入 opt_steps=4；早期 steady plan time：
  bucket003 约 `2.93–2.99s`，bucket004/bucket007 多数约 `3.01–3.05s`。
  吞吐 gate 可能临界或失败，必须等待完整日志按逐 case median 正式判定，
  当前不启动 Full。
- 30s watcher 已启动并确认 session 持续运行；GPU2/3/6 worker 均仍活跃，
  尚未触发 pull 或 gate。
- 中途只读统计：bucket007 已到 `166/166`，median=`2.9799s`（暂时 PASS）；
  bucket003 `168/416`, median=`2.9410s`；bucket004 `166/290`,
  median=`3.0207s`（暂时超门）。只以完整日志最终 validator 为准。
- E178 A100 canary `prep-only` 正在执行；固定 pool 已解析为 `2 3 6 7`，
  exact sync inventory=1790 files。当前仅同步/远程 preflight，tmux/CEM 尚未
  启动；remote sync/verification 仍在进行，连续检查暂未报告错误。
- 2026-07-24 11:52 续接 watcher：远程 session
  `e178_a100_canary_20260724_113702` 仍存活；GPU2/3/6 当前分别约
  `675/1285/0 MiB` 且有计算负载，GPU7 为空队列，尚未触发 pull/gate。
  继续等待完整 canary；在 3/3 runtime 与逐 case median `<=3s` 以前不启动
  Full。
- 2026-07-24 11:54：已完整续读两项 skill 与 A100 远程规范，并复核 E178
  plan/log 的 Gate C/D。watcher 仍显示 tmux 存活，GPU2/3 上的 canary
  worker 继续运行；未触发 pull，Full 保持禁止状态。
- 2026-07-24 11:55：一次额外只读 SSH 明细检查遇到临时 DNS 解析失败
  (`Could not resolve hostname tianyiyun-a100`)；未改变远程状态、未重启任务。
  hardened watcher 此前仍连续成功采样且 session 存活，按规范继续保守等待，
  不将单次 SSH 故障误判为完成。
- 2026-07-24 11:56：watcher 后续两次轮询恢复正常，远程 session 仍存活；
  GPU2 worker 继续有约 34% 利用率，GPU3/6 对应 case 已基本结束或处于收尾。
  仍等待 tmux 正常退出后的自动 pull 与完整 validator。
- 2026-07-24 11:57 E178 canary 完成并自动回收：3 rows、19 registered
  files；runtime gate `3/3 PASS`，throughput gate `2/3 PASS`、总体 `FAIL`
  (`threshold_s=3.0`)。按预注册 Gate C 立即禁止 Full，不启动 27-case
  `1024x32`；下一步读取逐 case 完整 timing 与 artifact contract，定位唯一
  超门项并设计不同配置的针对性修复。
- 2026-07-24 用户明确批准 E178 Gate C 吞吐豁免并要求启动 Full：
  bucket004 median=`3.0215s`，比 `3.0s` 阈值高 `0.0215s`（约 `0.72%`）；
  其余 bucket003=`2.93525s`、bucket007=`2.9799s`，runtime `3/3 PASS`。
  该授权仅豁免轻微 throughput 超门，不改变 27-case `1024x32`、固定 GPU
  `2,3,6,7`、execution snapshot 与 artifact validator 要求。
- Full launcher 审计确认 generic E176 入口会硬拒绝任何 throughput
  `status!=pass`，当前没有可审计的 waiver 参数。为执行本次用户授权且不篡改
  gate JSON，将新增默认关闭的显式 throughput-waiver 开关与必填 reason；
  runtime gate 仍不可豁免，并把 waiver 状态/原因写入 execution manifest。
- 已实现并验证显式 waiver：默认行为仍以 exit 1 拒绝当前 throughput FAIL；
  `waiver=1` 但缺 reason 以 exit 2 拒绝；`bash -n` 与 scoped
  `git diff --check` PASS。execution manifest 将记录
  `throughput_gate_waiver=true` 及用户授权原因，runtime gate 始终保持强制。
- 日志编号审计：当前最新为 237，E178 tracker 仍写“canary待执行”。将新增
  log 238 记录正式 canary 指标、用户 throughput waiver 与 Full launch；
  tracker 状态同步为 Full 运行中，237 保持为不可篡改的本地 gate 历史。
- E178 Full launcher 已接受显式用户 waiver 并进入启动流程；固定初始 pool
  解析为 `2 3 6 7`，Full exact sync inventory=`1934` files。当前正在
  rsync/remote preflight，尚未报告 tmux session 已启动；继续守到 launcher
  明确输出 session 与 execution manifest。
- Full 启动同步连续约 60s 无错误输出，launcher 进程仍存活；按 exact-file
  rsync/preflight 正常等待，不重复启动第二个 session。
- Full 首次 rsync 尝试遇到 `server not responding` / broken pipe
  (`rsync code 10`)；launcher 的 bounded retry 仍在运行。未创建重复
  tmux、未改变 GPU pool，继续观察同一启动进程的重试结果。
- 同一 Full launcher 在重试后继续存活，约 60s 无新增错误或成功输出；
  仍处于同步阶段，不误报已启动。
- E178 Full 已正式启动成功：session=`e178_a100_full_20260724_120137`，
  GPUs=`2 3 6 7`，27 rows 的 frame loads=`780/854/838/826`；remote exact
  inventory `1934/1934 PASS`。execution manifest：
  `workspace/core4d/results/E178/s0_environment/a100_full_e178_a100_full_20260724_120137/execution_manifest.json`。
  rsync 中途两次 timeout 均由 bounded retry 恢复，最终 preflight/tmux
  创建成功。
- 已复核 execution manifest：`fixed_gpus=true`、
  `throughput_gate_waiver=true`、reason 与启动 GPU/process snapshot 均完整；
  12:07:56 watcher 首检确认 Full tmux 存活。30s watcher 本地 tool session
  ID=`1995`，完成后将双确认 absent、自动 scoped pull/validate。
- 已新增 log 238 记录 canary 精确指标、用户 waiver、Full session/loads/sync；
  Tracker 更新为 Full 运行中，log INDEX 已重建并收录 238。launcher/watcher
  `bash -n`、scoped `git diff --check` 与索引引用检查均 PASS。
- Full 初始健康检查（12:08–12:09）：tmux 连续存活，GPU2/3/6/7 均有
  `38–48%` 左右计算负载，四个 worker 显存约 `1.55–2.43GiB`；未见启动即
  退出、OOM 或错误。watcher 继续在本地 session `1995` 运行。
- 2026-07-24 用户反馈远端 Full 偏慢，授权使用本地 GPU，并要求先测速再
  均衡分配。已完整复核 experiment-planning、data-construction-v3 与
  A100 remote-execution 规范；本轮先用独立 benchmark 路径测本地，不覆盖
  E178 Full result/log/manifest，再依据本地与远端实测吞吐迁移尚未开始的
  case，已运行/已完成 case 不重复。
- 资源快照：本地 GPU0=`RTX 5090 32GB`，仅 `150MiB`、util=0%，无
  `run_mjwp/run_cem_queue`；可用于单 worker。远端 tmux 仍存活、四卡
  util 约 `40–50%`，正式 Full result NPZ 仍为 0，说明四个首 case 尚未
  完成。runner 会在每条 case 结束后原子更新 shard，并对已有完整产物执行
  `[skip-complete]`；这为后续在 case 边界做无重复 requeue 提供了 contract。
- 已新增 plan 195，预注册 isolated `64x4` 可比 probe、`1024x2`
  production-density probe、LPT makespan gate 与远端 `not_run` race check。
  基准选 bucket007 canary（83 frames；A100 median=`2.9799s`）；正式 hybrid
  只有预测收益>=5% 才迁移队尾 rows。
- probe 可复用 E176 参数化 validators：throughput 支持
  `--expected-rows 1` 和自定义输出，runtime 支持单 row canary/full。
  系统 `python3` 缺 mujoco，正式入口必须固定使用项目 `.venv/bin/python`；
  这是环境选择而非代码失败。
- 已实现 canonical local probe 入口
  `scripts/launch/active/run_E178_local_speed_probe.sh`：复制 bucket007 row
  到独立 manifest/result/outdir/log，支持 `canary=64x4` 与
  `density=1024x2`，保存 GPU/git/wall-time/timing/runtime evidence。
  `bash -n`、scoped diff check、runner/validator py_compile PASS；RTX 5090
  启动前仍为 150MiB/util 0%。
- 本地 canary probe 第 1 次在 0s 内环境失败，未进入 CEM、未写正式产物：
  `MUJOCO_GL=osmesa` 导入 PyOpenGL 时 `GL.glGetError` 为 None。probe
  manifest 保持 `not_run`，runtime/timing 均按 contract FAIL，证据保存在
  `local_speed_probe_canary_20260724_123931`。下一步只切换 GL backend
  做 import/preflight，不能重复同一 osmesa 配置。
- GL preflight：`egl/glfw/disable` 均可 import MuJoCo 3.7.0；`egl` 进一步
  成功加载 E178 bucket007 sidecar（nq=42,nv=41,ngeom=67,npair=114）。
  因正式远端也是 EGL，本地 probe 改用 `egl`，与 A100 口径更一致。
- probe 入口已切换默认 EGL，`bash -n` 与 scoped diff PASS；第 2 次
  canary probe 已启动（tool session=`37105`），当前进程持续运行且未在
  import/preflight 阶段立即退出。
- 本地 canary probe 中途健康：RTX 5090 约 912MiB、util 44%，bucket007
  已到 sim_steps `132/166`；optimized plan time 多数约 `1.09–1.11s`，
  远快于同 case A100 canary median `2.9799s`（初步约 2.7x）。远端仍为
  4 running + 23 not_run、0 result NPZ；需等本地完整 validator 后再做
  production-density probe/分配。
- 本地 `64x4` probe 最终 runtime/timing `1/1 PASS`：wall=`92s`，
  median=`1.0995s`，相对 A100 同 case `2.9799s` 为 `2.7102x`。
  已按 plan 启动独立 `1024x2` density probe（tool session=`79085`），
  用于校准 Full samples 密度，仍不写正式 Full 路径。
- density probe 已完成 166/166 sim steps，CEM core total=`53.7805s`；
  opt_steps=2 的 plan time 主体约 `0.68–0.71s`，未见 OOM/numeric error，
  产物已保存。按 iterations 线性粗外推，RTX 5090 `1024x32` 主体约为
  `10.9s/optimized record`；等待外层 runtime/timing validators 收口后再
  计算 row/makespan 分配。
- density probe 最终 runtime/timing `1/1 PASS`：wall=`59s`、
  median=`0.6857s`（`1024x2`），线性 full estimate=`10.9712s/record`。
  同时远端四个 running full rows 的实测 median 为 GPU2 `29.8858s`、
  GPU3 `29.0493s`、GPU6 `30.1744s`、GPU7 bucket004 `25.5759s`；
  5090 在 production-density 外推约快 `2.3–2.75x`，正式 hybrid 的 5%
  收益 gate 明显满足。远端状态仍为 4 running + 23 not_run。
- 初步 suffix 优化：remote-only 当前预计 makespan 约 `6.07h`；按本地
  20% slowdown safety factor，候选为 GPU2/3/6/7 队尾各
  `2/4/3/3` rows（本地12、远端15），预计 `3.90h`、缩短约 `35.8%`。
  已实现 `build_hybrid_rebalance.py`，会从 launch 时 live shard/log snapshot
  重算 rates、suffix、deadline margin 和 27-row authority，不硬编码初算。
- 已实现 `run_E178_local_hybrid.sh`：live rsync shard/log snapshot→自动
  allocation→逐 row 本地 `1024x32`→单 row runtime gate→再次检查远端仍
  `not_run`→remote staging/promote→远端 `[skip-complete]` 验证。
  当前 running rows 不停止；任一 race gate 变化即停止 promotion。
- hybrid `PREP_ONLY` PASS（无生产 CEM）：基于 12:50 live snapshot，
  remote-only makespan=`5.996h`；自动选择队尾 counts=`2/3/3/3`，
  local=11 rows、remote=16 rows，conservative hybrid=`3.986h`，
  预计缩短 `33.52%`。builder py_compile、launcher bash-n、scoped diff
  均 PASS。
- PREP allocation 的 11 条 local rows 全部来自各 shard 位置 5–7 的队尾；
  最小保守 deadline margin=`2237s`（约37分钟），其余为
  `4764–8793s`，race 提前量充分。复核 staging/promote 参数时修复了
  sync-list remote 相对路径传参；尚未启动本地 production。
- 已新增 log 239 与 Tracker hybrid 索引，log INDEX 重建；最终 bash-n、
  builder py_compile、scoped diff check PASS。正式 local hybrid launcher
  已启动（tool session=`76204`），当前正在重新抓 live snapshot/重算
  allocation，尚未宣称首条 production row 已进入 CEM。
- 正式 live allocation PASS：remote-only=`5.954h`，hybrid=`3.944h`，
  local/remote=`11/16`，预计缩短 `33.75%`，tail counts 仍为
  `2/3/3/3`。首条 local production
  `bucket007_20231023_075_p2` 已在远端 gpu7 position5 仍 `not_run` 时启动；
  RTX5090 util=61%、显存=2036MiB，实际 `1024x32` plan time
  `11.14–11.43s`，与预估 `10.97s` 基本一致、远低于 20% safety。
- log 239/Tracker 已更新为 hybrid 正式运行；scoped diff check PASS，
  INDEX 已含 239。进程复核显示 local hybrid shell、single-row runner 与
  bucket007 run_mjwp 三层均存活约2分钟；GPU0 util=60%、显存=2036MiB，
  无重复本地 worker。
- 2026-07-24 本轮按用户要求开始只读进度检查：已重新完整读取
  `experiment-planning-zh`、`data-construction-v3-zh` 与 A100
  `remote-execution.md`，确认继续监控正式 Hybrid，不改配置、不重启任务；
  下一步轮询本地 tool session `76204` 与远端 watcher `1995`，核对
  promotion/runtime/race gate、远端 shard 状态与 live ETA。
- 2026-07-24 14:33 live 检查：本地 Hybrid 已连续完成并 promotion
  `6/11` rows（gpu7 队尾 3 条 + gpu6 队尾 3 条），每条单 row runtime
  gate 均 PASS；第 7 条 `bucket007_20231023_075_p1` 已在远端 gpu2
  position5=`not_run` 时启动。远端 Full watcher 到 14:33 仍持续报告
  session 存活，GPU2/3/6/7 均有负载；暂未见 OOM/numeric/validation/race
  错误。下一步读取 live shard/runner logs，确认远端完成/运行/skip 数及 ETA。
- 本地持久结果复核确认正式 Full 顶层 NPZ 已有 `6` 条；`hybrid_events.log`
  对这 6 条均出现远端 runner `[skip-complete] ... validation=run_complete_pending_eval`，
  说明 promotion 后远端确实识别并跳过，没有重复计算或覆盖。正式 allocation
  仍为 local/remote=`11/16`、预测 hybrid makespan=`3.944h`；现在需要用
  远端 live queues 而非 12:54 snapshot 重算当前剩余时间。
- 14:34 远端 live authority：正式顶层 NPZ=`11/27`（其中远端自产 5、
  本地 promotion 6），即 artifact 完成率 `40.7%`；另有远端 4 条和本地
  1 条正在计算。远端各 shard 为 gpu2=`1 complete+1 running+4 not_run`、
  gpu3=`1+1+5`、gpu6=`1+1+5`、gpu7=`2+1+4`；四卡显存/利用率均正常。
  当前 CEM 进度约 gpu2 `186/270`、gpu3 `204/280`、gpu6 `242/286`、
  gpu7 `80/262`。本地与远端错误扫描均未见 OOM、Traceback、numeric/race
  failure；下一步按每条 trajectory 长度和 live rate 重算分 shard ETA。
- live ETA 重算：本地第 7 条约 `70/210` sim steps，RTX5090
  `~57%/2.2GiB`，按约 `11.1s/optimized record` 预计本地 11 条约在
  `15:45–15:50` 完成。远端按当前四条剩余 records、后续 retained rows
  和 live rate 重算，gpu2/3/6/7 预计约在 `16:36/16:50/16:30/16:18`
  清空真实计算队列；当前 critical path 是 GPU3，Full CEM 预计还需约
  `2h15m`（约 `16:50`，另留数分钟 skip/pull/validate 开销）。
- 用户要求先把已完成结果在本地渲染为视频。已按本轮 skill 规范恢复 E178
  plan/log/progress，并读取 `video-frames`；本轮只消费已经完整落盘且通过
  runtime gate 的 Full NPZ，使用正式 `results/E178/s6_downstream/render/full/`
  路径，计划单 worker 串行离线渲染，避免明显干扰仍在运行的本地 CEM，
  随后抽帧做快速视觉 QC。
- 已审计 E173/E174 的正式离线 renderer：`render_cem_results.py` 会按
  manifest completed status 读取 `outdir_npz + config_act`，调用 E168
  `render_row` 输出 H.264 MP4，并具备已有视频 skip/resume contract；E178
  应复用这一实现但必须以 E178 manifest/结果根为 authority，不能错误读取
  E174 constants。下一步检查 E178 是否已有等价入口并核对当前可渲染 rows。
- E178 当前没有独立 renderer；production manifest 的 27 rows 仍保持
  launch authority 状态 `READY_FOR_FULL`，不能按 status 过滤。按四项产物
  existence contract（outdir NPZ/config/scene/trajectory）现有 `8` 条本地
  ready、0 条已有视频。E168 generic renderer 本身正好按这些文件检查，
  可用 `--pool all --manifest <E178 manifest> --output-dir <E178 render/full>`
  安全地只渲染 ready rows、跳过仍在计算的 rows；将固化为 E178 active
  单 worker wrapper 后先 dry-run。
- 已新增 canonical wrapper
  `scripts/launch/active/run_E178_render_completed_local.sh`：启动时冻结
  `primary NPZ + outdir NPZ + config + scene + source trajectory` 五项均完整
  的 case snapshot，使用 E168 generic renderer 单进程输出 E178 正式
  `render/full`，带 `flock` 防重复、selection 与独立 render log；未完成/
  正在运行 rows 不会误入。本步尚未启动实际渲染，下一步 bash-n + dry-run。
- wrapper `bash -n`、8-row dry-run 与 scoped `git diff --check` PASS；
  snapshot 的 8 条均为 bucket007、依赖 5/5 完整、现有 MP4=0。15:20 已启动
  正式单 worker 本地渲染（tool session `73796`），selection：
  `results/E178/s6_downstream/render/full/render_selection_20260724_152018.txt`；
  当前正在渲染首条 `bucket007_20231003_2_023_p1`。
- 正式渲染已在约 1 秒内完成：`8/8` rendered、0 failed，逐条帧数
  `118–212`、统一 50fps，均输出到 E178 `s6_downstream/render/full/`。
  首次调用 `video-frames/scripts/frame.sh` 抽帧因该脚本无 executable bit
  (`permission denied`) 失败 1 次，不影响 MP4；下一次改为显式
  `bash frame.sh`，不重复同一调用方式。
- 改用 `bash video-frames/scripts/frame.sh` 后 8/8 中间帧抽取成功；视频
  duration=`2.36–4.24s`，单视频大小约 `0.77–1.44MB`，均非空且 ffprobe
  可读。已用 ffmpeg 生成 `1304×1960` 的
  `keyframes_completed_20260724_152018/e178_completed8_midpoint_montage.jpg`，
  下一步打开 montage 做实际视觉观察并记录 QC。
- 实际检查 8-case midpoint montage：8/8 均正确显示同一 E178 bucket007
  蓝色桶 mesh 与 G1，画面为 ref/physics 左右对照；无黑帧、空 scene、模型
  拓扑串 case、明显镜头裁切或编码损坏。中间帧可见不同程度的机器人姿态/
  足部偏差，`20231020_059_p1` 的桶在中段呈倾斜状态；这些属于待用户观看
  完整时间序列的物理结果，不是 renderer 故障，也不能仅凭 midpoint 宣称
  CEM 质量通过。下一步把路径和观察追加到 E178 active log。
- 已把中间视频结果与实际观察追加到 active `log/239`。最终 ffprobe：
  8/8 均为 H.264、`1440×480`、50fps、`118–212` frames；MP4=8、
  midpoint JPG=8。renderer `bash -n` 与 log/script/progress scoped
  `git diff --check` PASS。用户现在可直接打开
  `results/E178/s6_downstream/render/full/` 查看这批视频；远端未回收 rows
  不在本次 launch-time snapshot 中。
- 用户再次要求检查进度。本轮已重新完整读取 experiment/data-construction
  skills、A100 remote-execution、E178 tracker/plan/log/progress；只做状态
  诊断，不修改运行配置。下一步轮询本地 Hybrid session `76204` 和远端
  watcher `1995`，核对是否已经完成、自动 pull/validate/eval 是否触发。
- 本地 Hybrid 已于 `15:48:56` 正常完成 `11/11` rows；11 条单 row runtime
  gate 全部 PASS，均完成 race-safe promotion 与远端 `[skip-complete]`
  验证，session `76204` 已退出且无错误。远端 watcher `1995` 到 16:03
  仍在运行，Full tmux 尚未退出，GPU2/3/6/7 仍有 CEM 负载；尚未触发最终
  auto pull/全量 validate/eval。下一步查询 live queues/当前 case 进度与
  合并 authority NPZ 计数，重算最终 ETA。
- 16:04 live 合并 authority：远端正式 primary NPZ=`23/27`（本地 promoted
  11 + A100 自产 12），即 `85.2%` 已完成；余下 4 条全部正在运行，0 条真实
  待启动。当前 gpu2/3/6/7 分别为 `82/222`、`44/240`、`118/228`、
  `186/246` sim steps；11 个 shard `not_run` rows 正是已由本地完成的
  tail，主 worker 到达后只会 skip。按 live rate 重算，GPU3 是 critical
  path，CEM 预计约 `16:54` 结束，随后双确认、pull/validate/eval 还需数分钟。
- 本地 11 个 runtime gate JSON 复核为 `11/11 PASS`，hybrid events 的
  11 条均有 promotion+skip validation；远端全量 CEM/worker 日志扫描
  `Traceback/OOM/numeric/[failed]` 命中 0。四张远端卡仍为约
  `46–49%` util，运行健康。
- 2026-07-24 用户要求先增量回收 E178 已有结果、补齐本地视频渲染并注册
  review player。当前正式播放器实际位于
  `scripts/eval/review/viser_review_player.py`；用户所指
  `scripts/eval/wrappers/review_player.py` 尚不存在。下一步先确认远端
  watcher 是否已触发最终 pull，避免竞争同步；再审计现有播放器注册格式和
  generic pull contract，仅同步远端自产且已完整结束的 rows，绝不覆盖本地
  Hybrid promotion 产物。
- 16:xx 状态复核发现 watcher 已进入最终权威回收：本地进程
  `pull_E176_remote_a100_results.sh full` 正在从
  `e178_a100_full_20260724_120137` 逐条 rsync E178 Full 产物。为避免并发
  pull 竞争，本轮不再启动 partial pull；等待现有同步完成后，以本地
  primary/outdir/config/scene/trajectory 五项完整性为 gate 补渲染，并核对
  是否达到 27/27。
- 权威 pull 随后正常退出，本地 E178 Full 已达到 primary NPZ=`27/27`、
  outdir trajectory NPZ=`27/27`、config=`27/27`，当前无残留 CEM/pull/eval
  进程；已有 MP4 仍为 `8/27`。正式 review player 的注册 authority 是
  `scripts/eval/review/review_index.py::DEFAULT_EXPS`，播放器从各实验
  `s6_downstream/eval/full/*_case_metrics.tsv` 建索引，因此需先确认/生成
  E178 正式 eval，再把 E178 加入默认实验集合并验证 `--check`。
- 16:56 已重跑 canonical completed-row renderer，启动快照冻结完整
  `27/27` rows；已有 8 条自动 skip，当前单 worker 正在补剩余 19 条，正式
  输出仍为 `results/E178/s6_downstream/render/full/`。最终 pull 已生成
  `full_runtime_gate.json`，但 E178 pull wrapper 没有配置 downstream
  `EVALUATOR`，所以 watcher 没有生成 review player 所需 case metrics；
  下一步复用 E176 multi-geom evaluator contract 建 E178 正式 eval 入口，
  而不是伪造只含视频的索引。
- E178 全量离线渲染已完成：selection=`27`、新渲染=`19`、复用已有=`8`、
  failed/not-ready=`0/0`，现在正式 MP4=`27/27`。统一使用 E168 renderer
  contract，输出 H.264 ref/physics 对照视频。已完整审计 E176 evaluator：
  E178 manifest 的 71 列覆盖其所有必需输入，适合复用，但当前 evaluator
  把 experiment id、expected rows、输出文件前缀和 paired delta 列名硬编码
  为 E176；下一步先安全参数化并保持 E176 默认行为不变，再提供 E178
  canonical wrapper。
- 已参数化 E176 low-geom evaluator 的 experiment id、expected rows、输出
  prefix 与 offline video directory，E176 默认仍保持 `E176/6/39/e176`；
  新增 E178 canonical runner/wrapper，显式固定 `E178/3/27/e178`，Full
  metrics 的 `video` 列优先指向正式 render 目录。`py_compile`、`bash -n`、
  `--help`、scoped `git diff --check` 全部 PASS，且按 manifest 逐 row 核对
  render MP4=`27/27`、missing=0。下一步正式运行 E178 Full eval。
- 已启动 `eval_E178_lowgeom.sh full --require-all`，逐 case 生成统一 metrics；
  同时把 `E178` 加入 `review_index.DEFAULT_EXPS`，并将 viser player 与
  canonical `wrappers/review_player.sh` 的范围说明更新为 E170–E178。该注册
  不新建重复的 `wrappers/review_player.py`；真实实现仍为
  `eval/review/viser_review_player.py`，wrapper 继续作为唯一启动入口。
- Full eval 当前已推进到至少 `9/27`、无逐 row error。按 `video-frames`
  流程已对全部 27 条 MP4 按各自 duration 中点抽帧，并生成 bucket003/004/007
  三张 object-group montage；frames=`27/27`、montages=`3/3`，持久路径为
  `results/E178/s6_downstream/render/full/keyframes_all_20260724_165645/`。
  下一步打开三张 montage 做实际视觉 QC，不以仅生成图片代替人工观察。
- 已实际打开 bucket003（9 rows）与 bucket004（4 rows）中点 montage：
  所有 tile 均有清晰的 ref/physics 双画面、G1 与对应蓝色 bucket mesh，
  未见黑帧、空 scene、错物体拓扑、编码破损或明显镜头裁切。bucket003 中点
  姿态/桶位差异较分散；bucket004 至少有一条中点物理侧桶与参考侧空间偏差
  明显。这里只判定 renderer/replay 可用，不据单帧判定 CEM pass。
- 已实际打开 bucket007（14 rows）中点 montage：14/14 双画面可读、物体与
  G1 拓扑正确，无 renderer/编码故障；若干 case 中点可见较大人体/物体偏差，
  至少一条桶明显倾斜，需结合完整视频与 metrics 审阅。E178 Full eval 随后
  完成：evaluated=`27/27`、errors/not-ready/missing-baseline=`0/0/0`、
  paired=`27/27`、numeric pass=`16/27`。另 `gate_health_pass=0/27` 是内部
  PRG leg-gate health 指标，不能误写成最终 numeric 0/27。
- review player headless `--check` PASS：E178 indexed/evaluated/numeric-pass/
  playable=`27/27/16/27`，全库合计 indexed/playable=`165/165`。E178 metrics
  中 outdir/scene/trajectory/video 均为 `27/27` exists，object 分布保持
  bucket003/004/007=`9/4/14`；27 个视频均为 H.264 `1440×480`、50fps。
  另做首条 E178 player 数据加载 smoke，MuJoCo nq=`42`，sim/ref shape=
  `(416,42)/(466,42)`、416 playback frames、50fps，说明不仅 index 注册，
  实际 3D loader 也能读取 E178。
- E178 分对象 numeric pass：bucket003=`5/9`、bucket004=`3/4`、
  bucket007=`8/14`；失败计数总体 hand penetration/release/fall/body-z/
  lower-body/contact=`6/3/1/1/3/3`。本轮只记录回收后即时摘要，不展开替代
  用户视觉审阅的因果结论。
- 更新 active log/Tracker 时首次 `apply_patch` 因日志末行 context 换行与
  预期不一致而未修改任何文件；已改用精确末行 context 成功追加 Full 回收、
  eval、27-video QC 与 review 注册证据，并将 E178 Tracker 从“运行中”更新为
  “Full 27/27、numeric 16/27、待人工审阅”。
- 已补齐 E178 scoped pull 的 downstream evaluator 配置：
  `pull_E178_remote_a100_results.sh` 现在指向
  `eval/runners/eval_E178_lowgeom.py`。因此未来幂等重拉 canary/full 后会自动
  运行相应 3/27-row strict eval，不再出现 runtime gate 完成但 review
  case metrics 缺失的断点。
- 最终验收全部 PASS：Python compile、shell `bash -n`、scoped
  `git diff --check`、review headless index audit 均无错误；E178 summary
  status=`pass`，primary/video/metrics/playable/video-path 均为 `27/27`，
  numeric pass=`16/27`。无残留 E178 CEM/pull/render/eval worker；本轮任务
  已完成，下一步由用户通过 canonical review wrapper 做完整时间序列审阅。
- 2026-07-24 用户要求生成与此前相同的 E178 XLSX 表。已按 `xlsx` 与
  `experiment-planning-zh` 恢复上下文，定位 E173/E174 canonical PRG
  workbook generators；目标复用 Overview + Case Metrics + Group Summary
  + review/error evidence 的既有样式与公式 contract。首次用项目 `.venv`
  读取旧 workbook 失败：`ModuleNotFoundError: openpyxl`；该环境不重复尝试，
  下一步检查系统 Python/skill 自带环境或安装依赖，再生成并用 LibreOffice
  recalc 验证零公式错误。
- 系统 `python3` 已有 `openpyxl 3.1.5`、`xlsxwriter 3.2.9`，且
  `/usr/bin/libreoffice` 可用，无需安装依赖。已审计 E173 latest workbook：
  canonical 样式为 Arial、深蓝 header、freeze A2、auto-filter、Overview
  公式汇总，以及 Case Metrics / Group Summary / Worst Cases /
  Manual Review / Codex Verification / Not Ready / Eval Errors。E178 现有
  TSV 为 Case Metrics 27×257、paired-vs-E174 27×68、Group Summary 8×69；
  新表将沿用旧结构，并加入有实际来源的 Paired Deltas sheet、合成 27-row
  Manual Review（绑定正式 MP4），不虚构缺失的人工裁决。
- 已新增 canonical report generator
  `scripts/eval/reports/gen_E178_bucket_prg_xlsx.py`，`py_compile` 与 scoped
  `git diff --check` PASS；首次生成成功：
  `results/E178/s6_downstream/eval/full/E178_buckets_prg_full_validation.xlsx`。
  workbook 含 9 sheets：Overview、Case Metrics、Paired Deltas、Group
  Summary、Worst Cases、Manual Review、Codex Verification、Not Ready、
  Eval Errors；数据量 evaluated=`27`、numeric pass=`16`、manual rows=`27`、
  worst-case rows=`105`。下一步必须执行 LibreOffice recalc 并审计公式值、
  零错误、样式和所有视频/关键帧路径。
- 已按 `xlsx` skill 使用 LibreOffice 强制重算 workbook：status=`success`、
  formula errors=`0`、total formulas=`554`。复核 data-only Overview 得到
  evaluated/numeric=`27/16`、pass rate=`59.259%`、bucket003/004/007=
  `5/9,3/4,8/14`、E174 fail→E178 pass=`16`、E174 pass→E178 fail=`1`、
  gate health=`0`、manual reviewed/USE/DNU=`0/0/0`，公式缓存均正确。
  9 sheets 的 Arial/深蓝 header/freeze/filter 样式一致；Case Metrics、
  Manual Review、Codex Verification 的 video 路径与 27 个 keyframe 路径
  全部 `27/27 exists`；XLSX zip integrity PASS，sha256=
  `ab2b3509e04b7fb8f5d20d9aee0bb89b05cc2c15ecc6c3b42d51cf86c9d3eff5`。
- 已把 workbook 路径、生成命令、9-sheet 结构与 recalc 证据写入 active
  `log/239 §13`。最终验收再次 PASS：generator `py_compile`、scoped
  `git diff --check`、Excel 2007+ file type、13 项 Overview cached-value
  assertions均通过，mismatches=`{}`；交付文件大小 `119,695 bytes`，路径为
  `results/E178/s6_downstream/eval/full/E178_buckets_prg_full_validation.xlsx`。
- 2026-07-24 用户要求为 E178 numeric eval 新增六个独立 tracking gates：
  root pos/ori=`20cm/20°`、hand(EFF) pos/ori=`20cm/20°`、object
  pos/ori=`20cm/10°`。实现将保持 E176 历史默认 gate set 不变，由 E178
  adapter 显式启用六门；failure mode 分别记录 root_pos/root_ori/hand_pos/
  hand_ori/object_pos/object_ori，并同步 review player 与 XLSX。
- 基于旧 E178 metrics 的只读预检：六门各自通过数为 root pos/ori=
  `17/27,22/27`，hand pos/ori=`17/27,14/27`，object pos/ori=
  `27/27,25/27`；六门交集=`14/27`，与旧 numeric pass 交集预计=`10/27`。
  该 `10/27` 仅为实施前 sanity estimate，最终 authority 必须来自重跑后的
  evaluator/summary。下一步先创建 plan 196，再修改代码和重跑。
- 已创建 `plan/196_E178_tracking_error_numeric_gates_plan.md`，预注册六个
  独立 failure modes、缺失/非有限即 FAIL、E176 backward compatibility、
  27-row strict eval、review/XLSX 一致性与零公式错误 claims。
- 已实现第一版：shared low-geom evaluator 新增 opt-in tracking gates 和
  六项阈值参数，E178 adapter 强制传入 `20/20,20/20,20/10`；summary 增加
  enabled flag、六项 threshold 与 pass counts；review index/player 增加
  六个 gate columns/中文标签并把红线同步为 20/20/20/20/20/10；XLSX
  Overview 增加六项 failure count。下一步先做 compile、CLI/E176-default
  regression 和单元级 gate truth-table，未验证前不覆盖正式 metrics。
- review player 兼容性补丁：只统计非空 gate columns，避免 E170–E174 因
  缺少新六列而显示成 `6/12`；legacy rows 继续显示原 6/6，E178 显示 12 门。
- 实施验证全部 PASS：Python compile、CLI help、scoped diff check；
  E176 legacy truth-test 只生成原 6 gate columns 且 PASS；E178 boundary
  等于阈值时六门均 PASS；hand ori=`20.0001°` 与缺失 object pos 精确产生
  `hand_ori,object_pos` FAIL；adapter argv 审计确认强制启用并逐项传入
  `20,20,20,20,20,10`。下一步可安全重跑 E178 27-row Full eval。
- 已启动 E178 Full `--require-all` tracking-gated 重评；当前推进到至少
  `6/27`、未见逐 row error。该步骤仅读取既有 result/outdir/scene/contact
  artifacts 并覆盖 eval/full 的派生 TSV/JSON，不重跑 CEM、不修改 27 个
  Full NPZ 或视频。
- tracking-gated 重评已推进到 `24/27`，仍无 row error/not-ready 提示；
  剩余 3 条 bucket007 完成后将审计 summary enabled flag、六阈值、六项
  pass/failure counts、12 个 gate columns 与按 object 的最终 numeric pass。
- tracking-gated E178 Full 重评完成：evaluated/paired=`27/27`、
  errors/not-ready/missing-baseline=`0/0/0`、summary status=`pass`，
  numeric pass 从旧 `16/27` 收紧为正式 `10/27`；bucket003/004/007=
  `3/9,2/4,5/14`。新增 gate pass/fail 为 root pos=`17/10`、root ori=
  `22/5`、hand pos=`17/10`、hand ori=`14/13`、object pos=`27/0`、
  object ori=`25/2`。
- 逐 row 12-gate audit：27×12 gate cells 全部为明确 true/false，
  `numeric_release_pass == all(12 gates)` mismatch=`0`；summary 的
  tracking enabled flag 与六阈值精确为 `20,20,20,20,20,10`。Review
  headless check 已同步显示 E178 numeric=`10/27`、playable=`27/27`；
  27 个 Full NPZ mtime 未变，仅 eval 派生 TSV/JSON 于 18:15 更新。
- 已用新 metrics 覆盖重建同一路径 E178 workbook 并按 `xlsx` skill 执行
  LibreOffice recalc：numeric=`10/27`、manual rows=`27`、worst rows=`105`，
  total formulas=`716`、formula errors=`0`。Overview cached values 为
  pass rate=`37.037%`、bucket003/004/007=`3/9,2/4,5/14`，六个新 failure
  counts=`10,5,10,13,0,2`；machine/final review 公式引用新的 User Reviewed
  row B23，缓存均为 PENDING_USER_REVIEW。
- XLSX/consumer audit PASS：Case Metrics 中六个 gate columns 均存在，
  review index 的 27 rows 全部识别 `12` 个 known gates、numeric=`10`；
  video/manual/codex/keyframe 路径各 `27/27 exists`，zip integrity PASS，
  新 workbook sha256=
  `6a7f08db92b6b4895424217e28f6061496c8dd560f904557dffa66d4d940c481`。
- 新 tracking gates 额外淘汰旧 6-gate PASS 共 6 条：
  bucket003 `003_p1/005_p1`、bucket004 `012_p1`、bucket007
  `021_p1/055_p1/059_p1`；hand ori 是最强单项瓶颈（13 FAIL），object pos
  为 27/27 PASS。已创建 `log/240_E178_tracking_error_numeric_gates.md`
  完整记录阈值、结果、六条 delta cases、兼容性和 claims，并把 Tracker
  E178 更新为“12门 numeric 10/27，待人工审阅”。
- 已重建 `log/INDEX.md`，新 log 240 已入索引。最终综合验收中 evaluator、
  summary、review compatibility、docs、diff 和 XLSX zip 均 PASS；唯一失败
  是自写 cached-value assertion 对 pass rate 使用浮点严格相等：
  LibreOffice=`0.37037037037037`、Python `10/27=0.37037037037037035`
  （差约 5.6e-17）。这是验证脚本精度问题而非 workbook 公式错误；下一步改用
  `math.isclose`，不重复严格比较。
- 已将最终 cached-value 验收改为容差比较并通过：E178 strict eval
  `27/27` 完整、12 门 numeric=`10/27`，分物体
  bucket003/004/007=`3/9,2/4,5/14`；XLSX LibreOffice 重算
  `716` 个公式、`0` 错误，27/27 视频与关键帧路径有效。本轮六项 tracking
  gate 接入、review player 同步、结果日志/Tracker/XLSX 更新均已完成。
- 用户要求同步 `eval/wrappers/review_player.sh` 的 tracking 红线。审计发现
  当前六项显示阈值只硬编码在 `viser_review_player.py`，数值已经是
  `20/20,20/20,20/10`，wrapper 本身没有阈值入口；下一步把六项默认值显式
  固化在 wrapper，并让 player 从 wrapper 环境读取，避免两处不可控地漂移。
- 已在 `review_player.sh` 显式设置 root/hand/object pos/ori 六项红线为
  `20cm/20°、20cm/20°、20cm/10°`，player 改为读取对应环境变量；直接启动
  Python 时仍保留同值默认行为。下一步执行 shell/Python 语法、环境传递和
  headless index 回归检查。
- review player 阈值同步验收 PASS：`bash -n`、Python compile、
  wrapper `--check`、scoped `git diff --check` 全部通过；实际导入得到六项
  tracking 红线 `[20,20,20,20,20,10]`。Headless index 仍为
  `165/165` playable，E178 numeric 保持 `10/27`，未改变任何评测产物。
- 2026-07-24 用户提交 E178 人工终审：bucket003 全 9 条、bucket007 全
  14 条，共 `23/27`；bucket004 4 条未给裁决，必须保持 pending。末尾粘连
  文本按两个合法 case（`bucket007_20231023_075_p1` 与
  `bucket007_20231003_1_021_p2`）解析。首次 glob 读取 E178
  `user_manual_review_*.tsv` 因当前尚无该文件而命中文字面量并报
  `sed: No such file`；这是只读探测错误，未修改产物。下一步审计 template/
  XLSX generator 的正式字段和既有标签映射后创建 annotation authority。
- 已确认 canonical authority 为
  `results/E178/s6_downstream/eval/full/user_manual_review_filled.tsv`，
  review player 的原子 upsert schema 为 9 列，XLSX generator 会读取该文件
  并把缺失 case 合成为 pending。沿用 E173 已有语义映射：
  `不能用→DO_NOT_USE/UNUSABLE`、`勉强能用→USE/MINOR_ACCEPTABLE`、
  `可以用→USE/CLEAN`；历史 reviewer 使用 `xiayibo`，本次用户裁决按同一
  reviewer 归档。下一步先用 27-row metrics 对 23 个 case ID 做严格集合
  对账并计算人工/12门的交叉结果，再写入 authority。
- 23 个提交 case 与 E178 27-row metrics 严格对账：missing=`0`、
  duplicate=`0`；未标注精确为 bucket004 四条。人工分布为
  `DO_NOT_USE=11`、`MARGINAL_USE=4`、`CLEAN_USE=8`，即人工 USE=`12`、
  DNU=`11`。12门 numeric 在已审 23 条上为 `8 PASS/15 FAIL`：8 个 numeric
  PASS 全被人工判 USE（precision=`100%`），但人工 USE 中有 4 条 numeric
  FAIL（recall=`8/12=66.7%`），分别由 hand penetration、release、
  hand orientation 等保守门淘汰。日志编号确认下一个为 `241`，统一记录时间
  使用 `2026-07-24T18:39:21+08:00`。
- 已创建 E178 canonical `user_manual_review_filled.tsv`，23 行全部
  reviewed，`USE=12`、`DO_NOT_USE=11`，quality 为
  `CLEAN=8/MINOR_ACCEPTABLE=4/UNUSABLE=11`；ID、状态、reviewer、note
  schema 校验全部 PASS。review player headless 已同步显示 E178
  `reviewed=23/27`、playable=`27/27`；唯一 pending 为 bucket004 四条。
  下一步重建 XLSX，使 Manual Review 和 Overview 缓存值同步该 authority。
- 已重建并 LibreOffice 重算 E178 XLSX：9 sheets、716 formulas、0 errors，
  ZIP integrity PASS。Overview 缓存值精确为 evaluated=`27`、
  numeric=`10`、reviewed=`23`、USE=`12`、DO_NOT_USE=`11`；Manual Review
  为 reviewed/pending=`23/4`，bucket004 四条保持 PENDING。因未完成 27/27
  人工审查，Machine recommendation 与 Final promotion decision 均正确保持
  `PENDING_USER_REVIEW`。workbook sha256=
  `a379891febdca1def181fcba791baedf6729371c39888e0c75513300fb18d6c9`。
- 已创建完整结果日志
  `log/241_E178_bucket_user_manual_review_results.md`，包含 23 条逐 case
  标签、分物体汇总、人工/numeric 交叉表、4 条人工 rescue case 的实际指标与
  阈值、UNUSABLE failure-mode 分布、可视化证据边界、产物/命令/Claims 和
  bucket004 待办。核心结论：numeric PASS precision=`100%`、人工 USE
  recall=`66.7%`、二分类一致率=`82.6%`；12 门适合作为高置信自动通过门，
  不能替代人工终审。
- 已更新 EXPERIMENT_TRACKER 的 E178 单行索引：状态为
  `12门 10/27；人工23/27(USE12/DNU11)，bucket004待审`，并追加 log 241
  链接；未把 partial review 错写成实验最终完成。下一步重建 log INDEX，
  然后执行 annotation/log/tracker/XLSX 的最终一致性和 diff 验收。
- 已重建 `log/INDEX.md`，log 241 正确入索引。首次综合校验误用了已知没有
  `openpyxl` 的项目 `.venv`，再次触发 `ModuleNotFoundError`；这是校验环境
  选择错误，annotation/XLSX 未被修改。该错误不再重复，下一次统一用已有
  `openpyxl 3.1.5` 的系统 `python3` 完成 cross-consumer assertions。
  同时发现 `progress.md=706` 行，超过 skill 的 200 行归档门；最终验收后
  需按归档规则保存备份并收敛 active progress。
- 改用系统 `python3` 后最终 cross-consumer assertions PASS：
  annotation confusion=`8 numeric PASS+人工USE / 11 numeric FAIL+人工DNU /
  4 numeric FAIL+人工USE`，XLSX/日志/INDEX/Tracker 全部一致，scoped
  `git diff --check` PASS。当前 progress 已达 712 行；按 skill 归档规则，
  下一步保存 through-E178 完整备份，并把 active progress 收敛为 E178 最终
  状态与唯一未决项 bucket004 review。
