# CORE4D 当前进度

## 归档索引

- [E189 完整执行记录（跨机协调/环境坑/xlsx迭代）](progress_archive/E189_full_backup_20260807.md)
- [E187 Full至人工比较完整备份](progress_archive/E187_full_manual_comparison_20260803_full_backup.md)
- [E186 production P/R/G完整备份](progress_archive/E186_production_prg_20260802_full_backup.md)
- 更早阶段见 `progress_archive/`。

## 最近完成：E189 box004/box024/box001 E167A no-PRG vs PRG 配对消融

### 2026-08-06 ~ 2026-08-07

- 43 条 case（box004×6/box024×9/box001×28，box023 引用 E179 不重跑）Full CEM 全部完成（跨机 jzsy-11 + ditg-12，中途修复了 `run_cem_queue.py` 的 NFS copy 崩溃问题）。
- 十二门 eval 全部跑通（`eval_E189_boxes_e167a_vs_prg.py`，43/43、516/516 gate cells）：**box004 `NO_PRG_NONINFERIOR`(2/6→2/6)、box024 `PRG_BETTER`(2/9→0/9)、box001 `NO_PRG_NONINFERIOR`(5/28→6/28)**。三物体 lower_body gate 一致退化（-2/-6/-4），但"PRG 随体积单调退化"假设在 no-PRG 侧不成立（box001 体积比 box024 更大却是 non-inferior）。
- 43/43 self MP4 + 43/43 paired 对照视频渲染完成；6/11 条 pass-migration case 做了关键帧视觉复核，box024 有清晰的腿部穿箱可见证据，与数值结论强吻合。
- xlsx 报表按用户反馈重做为精简版（每物体一个 sheet：case_id + 是否通过 + 失败模式 + 12门(PRG/E189/Delta)，Delta=PRG−noPRG，表头橙色区分）。
- 已注册进 viser review player（43/43 playable，`--check` 无 mismatch）。
- 完整结论、Claims verification(C0-C6全PASS)、局限性见 [log/265](log/265_E189_box004_box024_box001_e167a_no_prg_vs_prg_results.md)；执行细节（跨机协调、GL环境坑、xlsx迭代过程）见归档备份。E189 结论：box004/box001 维持 PRG default，box024 明确保留 PRG，三者均不进入 RL export。

## 最近完成：E178 final manual-USE partner RL export

### 2026-08-05

- 用户要求严格依据 E178 `user_manual_review_filled.tsv` 导出 RL 资产，并要求 partner 信息完整。
- 当前指定 authority SHA=`d430a8ef...c6c9f`，已独立确认27/27 reviewed：USE=13、DNU=14、PENDING=0；USE split为bucket003=5、bucket004=1、bucket007=7。
- 该 authority 晚于旧 log241 的23 reviewed/USE12/4 pending。本阶段以用户明确指定的当前文件为唯一选择 authority，不改写旧日志历史。
- 复用 E187 已验证的 E174 Stage2b opposite-person partner adapter与Holosoma fixed exporter；source-only fallback禁止。
- 详细计划已写入 `plan/214_E178_manual_use_partner_rl_export_plan.md`，目标为13条source/partner/paired与26条Holosoma `(cem,trajectory)` partner-enabled motions，不启动RL训练。
- 只读发现命令曾按错误的E187根路径查找partner/alignment/summary，实际产物位于canonical子目录；属于路径探查错误，未修改任何artifact，后续按真实manifest位置读取。
- E178专用导出器与canonical launcher已实现并通过`py_compile`、`bash -n`和diff-check；导出器冻结最终人工authority SHA及`USE=13/DNU=14/PENDING=0`。
- Core4D source/partner/paired/alignment均`13/13 RL_EXPORT_READY`。所有partner均为同object/date/seq的另一person，并唯一解析到E174 passing `omnirt_v1/ref_fk`；共同raw窗口范围为76–201帧。
- E178实际source合同保持`spider_method_id=E178_semantic_bucket_contact_aligned_top_segment_union_r1`与`hand_collision_variant_id=rubber_hull`；没有误用E187的COACD配置。人工USE中numeric release为8 PASS/5 FAIL，风险metadata完整保留。
- Holosoma按bucket003/004/007分别生成10/2/14条motion，共13 case × `(cem,trajectory)`=`26`；每条都带真实partner hand pose与`object_contact=(T,2)`。
- 独立NPZ审计26/26通过：唯一`(case,target_source)`=26、manifest/output frames一致、required partner/contact字段完整、numeric NaN=0。
- 三份manifest已注册为owner scope `E178_manual_use_partner_rl`；registry rebuild后`upstream=136`、`motions=81`，post-export三份PASS、pre-train 26/26 PASS。
- 最终状态为`DOWNSTREAM_RL_INPUT_VALIDATION_PASS`，只表示输入/loader readiness；未启动RL训练，也不宣称policy成功。正式记录见[plan214](plan/214_E178_manual_use_partner_rl_export_plan.md)与[log264](log/264_E178_manual_use_partner_rl_export_results.md)。

## 最近完成：E188 object tracking 跨物体分层审计

### 2026-08-05

- 用户观察到 E188 object tracking 变差，且 E187 bucket 物体位置相对参考序列已有可见偏差；要求与 E178、E188 和历史 box 类物体分层比较。
- 审计口径冻结为 `track_obj_pos_err_cm_mean` 与 `track_obj_ori_err_deg_mean`，分别统计 all cases、numeric PASS、人工 USE，并补充 per-object 与 common-case paired 结果。
- 历史 box canonical authority 冻结为 E170 box021、E171 box026、E172 box004、E173 box001/023/024；E179 等消融不重复纳入。box022 为 `DATA_NEGATIVE`，没有正式 Full metrics，不进入均值。
- E171 box026 无 filled manual review；E188 无独立人工审核。两者不会被伪装为人工通过，E188 另报 E187 manual-USE matched subset。
- 详细计划已写入 `plan/213_E188_object_tracking_cross_object_audit_plan.md`；本阶段只做离线分析，不重跑 CEM、不进入 RL。
- 首次聚合按计划预期 E171 同时含 box022/026 时 fail-closed；实际 metrics 只观察到 box026。已依据 E171 `DATA_NEGATIVE` 历史修正 authority，没有生成不完整正式结果。
- E173 authority 复核发现旧 eval `user_manual_review_filled.tsv` 只有22条 USE，早于后续 box023 人工审查；最终三个 RL-export manual snapshot 为 box001=13、box023=7、box024=3，共23条 USE。审计已改用最终 snapshot，旧表不再作为 E173 manual-only authority。
- 一次只读 `rg` 核验命令因 pattern 中反引号引发 shell parse error，未执行、未修改文件；已改用单引号安全 pattern 后完成核验。
- 初版聚合已生成163条 case-level authority：bucket E178/E187/E188=`27/22/15`，canonical box=`99`。全量均值显示 E178=`10.56cm/5.63°`、E187=`10.54cm/6.38°`、E188=`11.43cm/7.61°`、box=`13.61cm/6.26°`。
- 三版本共同15 case 的 paired 结果显示 E178→E187 基本持平，而 E187→E188 position `+1.606cm`、orientation `+1.805°`，两项 bootstrap 95% CI 均不跨0；已决定补充 per-object 与 same-device/cross-device 分解后再收口。
- 分层补充完成：E187→E188 的15条 position/orientation 非退化数均为0/15；same-device local-4=`+1.300cm/+2.103°`，cross-device11=`+1.717cm/+1.697°`；bucket003=`+2.158cm/+3.991°`，bucket007=`+1.521cm/+1.469°`。
- E187偏差量级得到量化：全量`10.54cm/6.38°`，11/22位置>10cm、14/22朝向>5°，但0条越过20cm/10°正式object gate。其position与E178基本持平，并优于canonical box全量`13.61cm`；E188 orientation `7.61°`则差于box `6.26°`。
- 正式分析写入 `log/263_E188_object_tracking_cross_object_audit_results.md` 与 `results/E188/s6_downstream/eval/object_tracking_audit/`；Tracker已完成。`experiment-report-writer`默认模板文件缺失，按skill要求的同等章节结构直接生成log263，不影响证据计算。
- 最终验证PASS：脚本py_compile/正式执行、163-row authority、E188 manual-self空集、15-case paired 0/15非退化、summary counts、Tracker/INDEX、四个冻结SHA与全仓`git diff --check`均通过。一次jq核验因未给含数字的对象键加方括号引号而编译失败，修正为 `.counts["E178"]` 后通过；未修改结果。

## 最近完成：E188 bucket 统一 5kg 对照实验已收口

### 2026-08-04

- 用户要求新增 bucket 系列统一 `5.0 kg` 的受控版本，其余 Full CEM 条件与 E187 保持一致。
- 计算资源冻结为本机 1 卡 + 远程 A100 GPU 4/5 并行；A100 启动前仍须通过显存、compute process 与预约许可二次检查，不抢占不合格 GPU。
- 本轮仅编写详细实验计划；尚未修改 scene XML、尚未生成 E188 artifact、尚未启动本地或远程 GPU。
- E187 Full authority 是 22 条：bucket003=`5`、bucket004=`4`、bucket007=`13`；预算为 `1024 samples × 32 iterations × seed 0`。
- 实际 E187 scene inertial 审计：15 条为 `2.0 kg`（bucket003 2条、bucket007 13条），7 条已为 `5.0 kg`（bucket003 3条、bucket004 4条）。E188 因此包含15条质量处理组与7条零变化内部对照。
- 质量处理定义：不覆盖 E187 scene；为22条派生独立 E188 scene。旧质量非5kg时设置 `mass=5.0`，并按 `5/old_mass` 等比例缩放 object `diaginertia`；其余 XML、轨迹、contact mask、collision/grid、reward、预算与seed保持冻结。
- 设备混杂已识别：E187远程设备是RTX 6000 Ada，本轮远程改为A100。计划将7条原本已5kg的case作为A/A漂移控制；控制漂移门失败时不得把E188-vs-E187差异强归因于质量。
- 为尽量减少调度混杂，E188拟保持E187三worker的case集合与顺序：E187 `local-0`继续本机GPU0，E187 `remote-0/1`分别映射到A100 GPU4/5；三条原canary仍作为本轮可晋级canary。
- 本机已只读确认GPU0为RTX 5090；远程GPU4/5只作为请求集合，正式启动前必须将预约许可、两次`nvidia-smi`与compute-process快照冻结到execution manifest。
- 详细计划已落盘为 `plan/212_E188_bucket_5kg_controlled_full_cem_plan.md`，Tracker登记为Phase 51 `PLAN_COMPLETE / NOT_STARTED`。
- 计划冻结E187三worker的case集合与顺序：本机GPU0=`7`条，A100 GPU4=`8`条，A100 GPU5=`7`条；每个worker均同时包含处理组与A/A控制组，支持分worker漂移校准。
- 当前边界保持：等待用户确认计划后再实现；未创建E188 scene/override/script/result，未启动GPU。
- 计划自检通过：506行；Tracker描述56字符（≤80）；targeted `git diff --check`通过；仓库中不存在E188 result目录或E188 override，符合plan-only边界。
- 用户修订：7条原本已经`5→5 kg`的case不再重跑；E188执行范围缩为15条真实`2→5 kg`处理case（bucket003=2、bucket007=13，bucket004=0）。
- 队列随之改为本机GPU0=`4`、A100 GPU4=`6`、A100 GPU5=`5`；三条canary更新为每个worker剩余队列首条，Full为canary 3条+remaining 12条。
- A/A控制和worker-stratified DiD从计划移除；本机4条保持RTX5090同设备配对，远程11条包含E187 Ada→E188 A100设备混杂，必须分层报告且不得把远程裸差异强归因于质量。
- plan212主体已同步改为15-case：authority/scene/override/Full/eval/xlsx/video均只覆盖15条，三卡4/6/5，canary后remaining=12；Tracker描述已改为15条口径。
- 修订后机器审计通过：计划队列15条/15唯一，与E187全部2kg case集合精确相等；旧22-run/7-control/DiD残留为0，`git diff --check`通过，仍无E188运行产物。
- 2026-08-04 用户已授权按plan212推进E188；已恢复计划、E187 authority与远程A100执行规范。当前worktree仅有plan/Tracker/progress变更，E188代码、scene、override和results均尚不存在，下一步进入Phase 1实现与静态门。
- E188实现前authority审计通过：plan212登记的6个E187 SHA均与当前文件逐一匹配；E187 queue为22 unique、三worker顺序可复用，15-case集合将以old scene mass=2kg与冻结evaluation manifest交集为唯一来源。runner沿用`1024×32 seed0`、video/info on、query tape off合同，但E188新建独立canary/promotion/row manifest，不继承E187 Gate0 waiver。
- 实现审计遇到两项非实验失败：首次XML审计误取evaluation manifest第15列trajectory而非第14列scene，已定位为只读命令列号错误、将用表头字段读取修正；A100 alias首次SSH报临时DNS解析失败，未产生任何远程修改，后续将按remote-execution规范用alias重试并在必要时使用计划登记的直连host。
- E188 Phase 1首个静态门通过：新增共享authority helper与mass5 scene builder；preflight从E187实际XML精确识别15条2kg/7条5kg，选中bucket003=2、bucket004=0、bucket007=13。只允许object mass与diaginertia两属性变化，scale=2.5，15/15临时scene均MuJoCo load成功；正式scene尚未freeze。
- E188 Phase 1/2静态artifact已首次freeze：15/15正式`scene_act_E188_mass5kg.xml`生成并MuJoCo load，authority manifest=15 unique、override=15、固定queue=`local-0/a100-4/a100-5 = 4/6/5`。7条5→5未进入scene/override/queue；下一步强制track活跃scene、创建scene snapshot并执行独立合同测试。
- 15个E188 scene已`git add -f`并完成`results/E188/scene_snapshot`（含HEAD与逐文件SHA）。新增原子runner：三条正式预算canary、3/3 gate、promotion、剩余12条Full、resume/row SHA/closure；合同测试3/3通过，三worker dry-run解析为canary `1/1/1`、remaining `3/5/4`，合计15，`git diff --check`通过。GPU仍未启动。
- A100直连只读检查成功：GPU4/5均为A100-SXM4-80GB、memory.used=1MiB且UUID无compute process；其它GPU有外部任务，未触碰。新增隔离snapshot/deployer、两轮GPU4/5门、local/remote launch与staging pull脚本；Python compile、shell语法与`git diff --check`通过。首轮DNS alias失败由计划登记的直连host/port/key安全替代。
- 真实preflight进展：本机RTX5090 GPU0设备/compute门通过，local canary精确解析1条正式预算命令；A100首轮门得到allowed=`[4,5]`。远程canary snapshot已冻结为4029 files / 453,448,408 bytes / SHA `55ee1d7a801f9426…`，隔离rsync/远端逐文件校验仍在进行，尚未启动CEM。
- 远程部署attempt1诊断：逐小文件rsync在约7分钟仅传9,779,518 bytes/125 files，属于高时延传输策略问题，不是GPU/CEM失败。已中断仅属于E188的rsync进程，并将不完整隔离目录可恢复地改名为`incomplete_e188_canary_55ee1d7a801f9426_aborted_smallfile_rsync`；共享checkout/外部进程零修改。部署器改为同一manifest选文件→单tar rsync→解包→4029文件逐SHA校验，科学payload不变。
- 远程部署attempt2诊断：单个未压缩tar消除了小文件开销，但链路实测约25.4MB/2.5min，仍不合理；已中断自有传输并保留partial为`incomplete_incoming_e188_canary_1ab40eeb94bf2ab5_aborted_uncompressed`。确认remote shared已有大部分代码/Unitree G1 mesh，但缺E186/E187 grid；attempt3改为remote shared只读SHA复用 + 仅缺失/不一致文件zstd overlay + 全量最终SHA，未启动GPU。
- 远程部署attempt3到达共享SHA复用阶段（远端root约193MB），随后因本机zstd不支持`--output`长选项而在overlay压缩前失败；无网络overlay、无GPU启动。已改用通用`-o`，将root/staging可恢复归档为`incomplete_e188_canary_8f92cfcca254cf2c_zstd_cli`，compile/diff通过；下一次不复用半成品。
- 远程部署attempt4验证shared复用可将网络payload降到125MB，但单rsync流传到16,252,928 bytes后连续数分钟I/O不增长；已中断自有连接并归档隔离root/partial。部署器已改为8MiB固定分块、最多8路并行rsync、远端重组SHA后解包；因用户中断尚未重跑。当前E188进程=0、canary/full row manifest=0，A100 GPU4/5各1MiB且无E188 compute，实验未误启动。
- 用户确认可用tar传输。部署清单已从全仓式4029 files/453,453,081 bytes收紧为精确runtime 1875 files/228,161,460 bytes：移除506个其他机器人asset与scene历史副本；保留grid 151,473,760、G1 54,507,553、15 task dirs 14,390,696、mask 538,182及必要代码/config/E188 authority。传输合同为remote shared SHA复用 + 差集tar.zst + 8MiB并行分块 + 重组SHA + 全量1875文件SHA。
- E188精确runtime的tar.zst分块部署已在隔离远端root完成，archive与1875文件逐SHA校验PASS；GPU4/5 runner preflight均PASS。启动合同补丁也通过`py_compile`、两份launch脚本`bash -n`和全仓`git diff --check`；当前尚无E188 CEM进程，下一步按正式预算并行启动本机GPU0、A100 GPU4和GPU5各1条canary。
- E188三条canary于2026-08-04 21:49发起：本机`bucket003_20231018_003_p1`已在GPU0运行（约2378MiB）；远端GPU4/5对应tmux启动后立即退出且仍各1MiB，无CEM进程、未消耗正式预算。按失败协议先诊断远端row日志，不重复原命令。
- 远端首次canary启动失败根因已定位：两条row均在导入阶段报`ModuleNotFoundError: spider.query_tape`，旧精确runtime snapshot漏收该模块；GPU4/5始终1MiB，未进入plan-time循环。已在E188合同测试新增远端runtime关键入口断言，后续将归档旧deployment证据、以新snapshot SHA隔离部署后再启动。
- 上述“snapshot漏收模块”诊断经逐SHA复核后更正：`spider/query_tape.py`已同时存在于本机冻结清单和远端隔离root，SHA=`c8c5cb7e...`。真实根因是共享venv执行时隔离repo root未进入`PYTHONPATH`，导致导入共享checkout的`spider`包；不需要重传228MB，改为修复远端import路径并在重启前执行真实解释器import门。
- 远端导入机制已用相同共享venv复现：`python examples/run_mjwp.py`以`examples/`为首路径，旧launcher未声明isolated root；修复为显式`REMOTE_ROOT:E188_scripts`双绝对PYTHONPATH，并新增`spider.query_tape.__file__ == REMOTE_ROOT/spider/query_tape.py`启动门。失败row将改名归档后再执行第二种启动配置。
- 修复后shell语法、E188合同测试4/4和`git diff --check`均PASS；远端attempt1的两条row与两份launcher日志已可恢复地移入`s4_canary/failed_attempts/remote_import_path_attempt1/`，正式rows路径重新为空。未删除artifact，下一次启动不重复旧PYTHONPATH配置。
- 远端attempt2于21:53启动成功：GPU4/5两个tmux持续存在，日志均识别进程内`cuda:0 = NVIDIA A100-SXM4-80GB`，已越过先前import错误；首次快照时仍处于Warp初始化（GPU各4MiB、尚无plan-time），将继续核对CUDA compute UUID和循环进度。本机GPU0已推进至`sim_steps=30/252`，plan time约25–29s。
- 远端attempt2物理映射确认PASS：GPU4/5 UUID分别有独立E188 compute进程，均识别A100。两条在首次Warp kernel编译后出现新Traceback并退出，GPU重新降至1MiB，仍未进入plan-time/未生成正式row manifest；下一步读取完整错误栈，禁止原样第三次重试。
- 远端attempt2新根因确定为A100主机缺NVIDIA EGL vendor（`/usr/share/glvnd/egl_vendor.d`仅`50_mesa.json`，无`libEGL_nvidia`），`save_video=true`创建MuJoCo Renderer时失败；与历史E168已记录的同主机结论一致，osmesa/glfw也不可用。CEM计算与数据加载本身已通过。下一步依据plan212的视频合同判断是否采用“远端compute-only + 本机离线渲染”既有方案，不能第三次原样EGL重试。
- plan212明确冻结CEM runtime `save_video=true`并要求canary逐row视频，因此改用A100 compute-only属于需用户确认的合同调整；建议保持`1024×32 seed0`和全部科学输入不变，远端11条仅`save_video=false`，回收NPZ/config/log后在本机离线生成E188与15条paired视频并登记SHA。等待确认期间本机canary继续正常运行，当前`42/252`、plan time约24–26s、GPU0约2885MiB。
- 用户明确确认继续采用“A100 11条compute-only + 本机离线渲染”。plan212与runner合同已同步：local-0=`INLINE_CEM`，a100-4/5=`DEFERRED_LOCAL_RENDER`；科学payload、`1024×32 seed0`、info与query-tape设置不变，最终15条E188/paired视频仍为硬门。
- compute-only合同实现首轮验证PASS：Python compile、local/remote/pull shell语法、4项E188合同测试和diff-check均通过；dry-run精确得到local`save_video=true/inline`、remote`save_video=false/deferred-local`。由于本机canary在用户确认前已启动，validator只对该唯一不可变local manifest提供显式legacy-inline归一化，远端仍强制新render_mode字段。
- attempt2的两条EGL失败row/config/log已可恢复归档到旧远端root的`s4_canary/failed_attempts/remote_egl_attempt2/`；旧`acbbc...`部署四份顶层metadata也移入本机`failed_attempts/a100_egl_attempt2_acbbc8109fbb5b10/deployment_metadata/`。无文件删除；正式远端rows和active deployment顶层重新为空，允许以新runner SHA构建隔离snapshot。
- compute-only canary snapshot已冻结：1875 files / 228,165,611 bytes / SHA=`499c6473c3d30d77...`，首轮GPU4/5门PASS。启动wrapper输出尚未出现最终`E188_REMOTE_PREFLIGHT=PASS`，下一步检查deployment manifest/远端root，确认是输出截断还是部署阶段失败后再继续。
- 新snapshot部署实际PASS：remote root=`e188_canary_499c6473c3d30d77/spider`，1875/1875逐SHA无mismatch；共享复用1022文件、差集853文件/128,965,474 bytes/16分块。用真实共享venv在新root手动复核query_tape路径与两worker preflight均PASS，命令精确为`save_video=false`、`render_mode=deferred-local`；先前仅为wrapper输出截断。
- compute-only A100 canary已于22:04正式启动并越过全部已知故障点：GPU4/5各有独立E188 compute进程（约1670MiB），两个tmux持续存在，均进入plan-time循环至`12/280`和`12/166`；物理GPU4/5→进程内cuda:0映射正确。本机inline canary同步运行至`82/252`，三条均未见Traceback。
- 正式32-iteration首步吞吐已观察：local约29s（`88/252`），A100 GPU4约67.8s（`14/280`）、GPU5约54.9s（`14/166`）。三session均健康且无Traceback；按plan212约定plan time仅记录、不作为停止门，不降预算也不换卡。A100实际明显慢于原估计，canary需较长时间。
- 新增plan212已规划的fail-closed watcher：仅在canary本机1/1+远端2/2 manifest存在后pull、冻结gate、promotion；随后重新执行A100双GPU门并启动remaining remote9/local3，最终pull与15/15 closure。任何session无manifest退出或worker completion缺失立即停止，不自动重试/换卡/降预算。
- watcher已通过`bash -n`、`git diff --check`和真实远端count/session probe；当前remote manifest=0/2且两个499c session均存在，符合运行中状态。脚本已设为可执行，下一步在独立tmux启动并将状态写入`logs/E188/watcher.log`。
- `e188_watch_and_pull`已于22:08在独立tmux启动，首轮状态`local=0/1 remote=0/2`且三条compute session均在运行。watcher将自动执行canary pull→3/3 gate→promotion→remaining12 launch→full pull→15/15 closure；评测/xlsx/paired视频仍在CEM closure后执行。
- 2026-08-04 22:58远程进度检查：A100 GPU4 canary=`114/280`（40.7%，last plan time 73.22s），GPU5 canary=`110/166`（66.3%，73.94s）；两tmux与对应compute PID均存活，GPU利用率40%/38%，显存约1.84/1.88GiB，无Traceback、manifest尚未生成。本机canary已`252/252`并生成manifest；watcher健康，持续等待remote 2/2后自动pull与晋级。
- 2026-08-05 01:53检查：canary已3/3完成、远程2条成功pull、`canary_gate=PASS`且3条promotion已冻结。watcher于00:55立即启动remaining12时，A100首轮门暂时得到`allowed=[]`（canary进程刚写完manifest但GPU尚未释放），按fail-closed退出；Full目前仅3条promoted canary，remaining12均未启动。当前GPU4/5已恢复1MiB/0%且无compute process，属于可安全重新触发Full的瞬时竞态，不是CEM/artifact失败。
- 用户明确授权立即启动remaining12，并要求按本地/远程实测速度重新分片；同时撤销GPU空闲/显存/compute-process安全门，允许GPU0/4/5与其他程序叠加运行。canary实测median plan time：local RTX5090=`27.9845s`，A100 GPU4=`73.0582s`、GPU5=`69.1504s`，本机约快2.47–2.61倍；下一步用12条E187 plan_time_count作为工作量做三机最小makespan分配并冻结修订队列。
- 速度感知分片计算完成：以E187 plan_time_count（已由3条E188 canary验证完全一致）×本轮设备median step time最小化makespan，并约束原E187 local-0三条仍留在本机以保留local-4同设备因果证据。最优remaining分配=`local/A100-4/A100-5 = 6/3/3`，预测约`5.084/5.114/5.129h`，比无约束最优仅差约53秒且不牺牲local-4证据；最终含canary为`7/4/4`。
- 已实现additive v2队列builder（不覆盖canary引用的v1），runner自动优先v2，远端snapshot纳入v2，合同测试改为`7/4/4`。部署器新增用户授权的overlap-policy manifest：GPU4/5固定但关闭memory/compute-process门、允许外部负载叠加，不抢占且不fallback。
- local/remote launcher已移除显存与compute-process阻断（仍保留telemetry）；pull与watcher改为从v2动态读取local6/remote6，并新增`full-only`监控入口。plan212已写入速度重排表、local-4保留约束和用户GPU叠加授权；旧v1队列及canary/promotion证据保持不变。
- v2队列已正式freeze，manifest SHA=`fb48ee9a1dff61fd...`；Python compile、4份shell语法、builder preflight/freeze、E188合同测试4/4、diff-check全部PASS。Full dry-run精确解析remaining=`6/3/3`且case集合12 unique；local为inline video，A100为deferred-local，预算仍`1024×32 seed0`。
- Full remote启动已写入`USER_AUTHORIZED_OVERLAP`执行policy（GPU4/5、不做memory/compute门），并冻结snapshot：1898 files / 253,418,731 bytes / SHA=`ae8b1295aeaa34b9...`。wrapper输出尚未显示最终deployment/launch行，下一步核对manifest、远端root和tmux后再启动本机，避免snapshot竞态。
- Full snapshot部署确认PASS（1898/1898 SHA，remote root=`e188_full_ae8b...`）；启动阶段GPU4/5各创建了首个v2 row并写出config/run.log，但当前无GPU compute，GPU5 tmux已退出、GPU4 tmux空挂，launcher log为空。尚未启动本机，避免掩盖远端新故障；下一步读取两条row完整日志与进程树定位。
- 上述远端“无compute”是采样空窗而非故障：两worker runner与run_mjwp子进程均存活，首rows已到warmup`12/210`与`12/196`并进入首个正式规划步。远端snapshot闭合后，本机remaining6已通过合同测试/Full dry-run并启动tmux `e188_full_local0`；三worker现已按v2并行运行。
- 02:16三worker健康：local首row`12/270`，A100-4=`14/210`（57.6s），A100-5=`14/196`（60.6s），GPU0/4/5均有负载。首次`full-only` watcher因TSV canary列过滤误读为`0/0`，随后被pull的“remote必须6条”硬门阻断；未回收/覆盖任何结果，计算进程不受影响。下一步修正动态case解析并重启watcher。
- 02:18已修正v2 TSV解析：`canary=$1`、`case_id=$2`，本地/远端remaining计数独立验证为`6/6`（A100-4=`3`、A100-5=`3`，12 unique）；两份脚本`bash -n`与targeted diff-check通过。`e188_watch_full`已恢复并报告`local=0/6 remote=0/6`，后续将自动pull并执行15/15 closure。
- 修复后实时证据：本机首row推进至`28/270`（约20.4–22.3s/plan step），A100 GPU4推进至`18/210`（约56.1–58.0s），GPU5推进至`18/196`（约59.9–60.6s）；三路worker tmux均为UP，均无正式remaining manifest，符合首row仍在运行。GPU重叠执行遵循用户授权，不做显存/compute-process安全门，也不抢占、不kill、不fallback。
- 02:34只读健康检查：本机首row=`96/270`（最近28.5s/plan step）、A100-4=`48/210`（最近65.7s）、A100-5=`46/196`（最近66.8s）；折算remaining队列plan-step进度分别为`7.34%/9.52%/8.61%`。三worker与watcher tmux均UP，run/launcher日志未见Traceback、OOM或RuntimeError，watcher每约32秒刷新且正确保持`local=0/6 remote=0/6`；当前0/12 manifest是三路首row尚未完整结束的预期状态。远端因用户授权的重叠负载较启动时变慢，但仍持续推进，无需干预。
- 07:02 E188 Full CEM已闭合：remaining本机`6/6`、A100-4=`3/3`、A100-5=`3/3`全部回收到本地，加canary共`15/15` manifest且逐row `status=PASS`；closure报告`checked_rows=15 / expected_rows=15 / failures=[] / status=PASS`。本地/远端worker与watcher均已正常退出，不是掉线。
- 完成后复核：7条inline视频已存在（canary目录1条+s5_full目录6条），其余8条A100 compute-only视频仍须本机离线渲染；E188-vs-E187 evaluator、xlsx与15条paired视频尚未生成。三个本地run.log中的Traceback均发生在NPZ/视频保存完成后的MuJoCo EGL析构阶段，内容为`Exception ignored ... EGL_NOT_INITIALIZED`，对应manifest仍PASS，不影响CEM artifact；本次只读汇总曾有一个局部变量名笔误，修正后未对实验数据造成修改。
- 用户已授权进入E188下一阶段。按plan212 Phase 6/7推进：先复用公共`eval.core.core_metrics`完成15条E188-vs-E187配对评测与xlsx，再本机离线渲染8条A100 compute-only视频、生成15条E187-left/E188-right视频、注册`review_player.sh`并用video-frames抽帧完成实际观察；不进入RL export/training。
- Phase 6/7实现路径已恢复：E188 evaluator将直接调用`eval.core.core_metrics.evaluate_sequence`并沿用冻结12门；15-row eval manifest从E188 row manifest与s0 authority构建。A100的8条deferred视频可复用E168已验证的`spider.viewers.render_image/setup_renderer`离线回放路径，产出与inline CEM一致的1440×480 reference/sim布局；paired renderer沿用E187的ffmpeg左右合成与ffprobe/SHA审计，改为E187-left/E188-right和15-row authority。
- E188 Phase 6首版实现已通过静态门：新增15-row eval manifest builder、直接调用公共core metrics的E188 runner、canonical wrapper和合同测试。builder精确解析最终E188 worker=`7/4/4`、inline/deferred=`7/8`，并冻结`same_device_local4=4`；py_compile、shell语法、合同测试、preflight与targeted diff-check均PASS，下一步运行15条CPU/MuJoCo评测。
- E188 Phase 6正式评测已`15/15`完成，errors=`0`、paired=`15`、numeric pass=`3/15`。相对E187的numeric转移为`FAIL→FAIL 11`、`PASS→PASS 2`、`PASS→FAIL 1`、`FAIL→PASS 1`；lower-body pass `4→8`，但paired mean leg improvement仅`+0.00627`且非退化`8/15`，未达到C5的`+0.05/10条`；接触与手穿透均值分别改善`+0.0130/+0.0256`，C6通过；C7 tracking非劣通过；numeric pass未升至5，C8失败。技术评测status=PASS，效果结论为mixed/未达升级门。
- Phase 7视频实现与注册静态门PASS：新增8条deferred本地回放renderer、15条E187-left/E188-right paired renderer、ffprobe/SHA审计与合同测试；`review_player.sh E188 --check`已可索引E188 `15/15`、numeric pass `3`、playable `15`。offline preflight确认deferred=`8`、existing inline=`7`，py_compile、两份wrapper语法、paired布局测试与diff-check通过；下一步正式补渲染8条并合成15条paired视频。
- Phase 7首次正式render：8条deferred视频均已成功生成（每条118–280帧、50fps），随后审计因E168 helper把results软链接解析为`/mnt/.../spider_workdirs/...`而在逻辑路径`relative_to(REPO)`处退出；这是render后的provenance路径归一化bug，不是视频/物理失败。已改为按`spider_workdirs/`marker恢复`workspace/`逻辑路径；下次运行将复用8条existing视频，不重复渲染。
- 修复后final-video审计已`15/15 PASS`（8 deferred复用existing + 7 inline）。15条E187-left/E188-right也全部成功合成，paired审计随后在E187左视频上遇到同类`/mnt/.../spider_workdirs`逻辑路径遗漏；15个MP4均已存在。已把同一marker归一化补到paired manifest生成器，下一步仅执行audit，禁止重复ffmpeg渲染。
- paired逻辑路径修复后，独立audit与canonical wrapper audit均`15/15 PASS`：layout=`LEFT_E187_2KG__RIGHT_E188_5KG`、failures=`[]`，final单侧视频也保持`15/15 PASS`。两次错误均限于render完成后的provenance逻辑路径，不改变任何视频内容；所有补渲染和paired视频现已具备ffprobe与SHA manifest。
- E188正式xlsx已生成：`s6_downstream/eval/full/E188_vs_E187_paired_evaluation.xlsx`，包含plan212要求的11个sheet和15条paired authority。工作簿含1538个公式，LibreOffice强制重算后公式错误`0`；data-only独立复核得到paired=15、numeric pass `3→3`、lower-body `4→8`、Claims `C5/C6/C7/C8=False/True/True/False`，缓存值与summary一致。
- video-frames实际观察已完成：对数值恢复`bucket007_20231003_1_021_p2`、同设备退化`bucket007_20231018_019_p2`、同设备近中性`bucket003_20231018_003_p1`、跨设备明显改善`bucket007_20231020_055_p1`各抽25/50/75%帧。`055_p1`最直观：E187末段机器人明显倒地，而E188保持站立并继续持桶，对应contact `+0.203`、leg `+0.108`；但hand penetration `-0.072`且仍numeric FAIL，属于明显视觉收益伴随手穿透trade-off。`019_p2`的E188末段更深俯身贴桶、手臂/下肢姿态更拥挤，对应同设备contact `-0.123`、leg `-0.091`，支持真实退化。`003_p1`左右整体姿态接近，符合leg近0变化，但接触/手穿透均回退。`021_p2`E188末段保持更直立稳定，numeric由FAIL→PASS、leg `+0.173`，但跨设备只能作为版本改善证据。
- E188已按plan212收口为T2 mixed：C0–C4/C6/C7/C9 PASS，C5/C8 FAIL；同设备local-4 leg improvement=`-0.05535`且95% CI不跨0，cross-device11为正但CI跨0，因此不升级5kg默认质量、不自动进入RL。正式结果写入`log/262_E188_bucket_5kg_controlled_full_cem_results.md`，Tracker更新为完成状态。

## 最近完成：E187 RL输入验证完成

### 2026-08-03

- 最终人工authority SHA=`42dd0ef...431b`；`USE=14`、`DNU=8`、`PENDING=0`。
- 用户批准全部14条人工USE进入下游；numeric仅`5/14 PASS`，其余9条风险作为metadata保留。
- Source/partner/paired/alignment均`14/14 RL_EXPORT_READY`；partner为同sequence另一person的E174 passing OmniRetarget motion。
- Holosoma正式生成14个case × `(CEM, trajectory)`=`28`条paired motion，全部含contact mask和真实partner手轨迹。
- 三份manifest均通过post-export；registry中28个E187 motion ID全部通过pre-train gate。
- C9仍为technical `FAIL`、progression authority=`USER_WAIVED`；当前状态是`DOWNSTREAM_RL_INPUT_VALIDATION_PASS`，不是RL policy成功。
- 正式记录：[plan211](plan/211_E187_manual_use_partner_rl_export_plan.md)、[log261](log/261_E187_manual_use_partner_rl_export_results.md)。

### 当前未决

- 若要继续完整RL实验，需另行冻结run ID、训练arm、seed/GPU、E178 baseline、成功指标和9条numeric FAIL人工USE的安全停止条件。
- 本阶段不启动完整RL训练。

### 验证

- exporter py_compile、targeted diff-check、14-row independent audit全部PASS；
- Holosoma pre-export `14/14`、NPZ结构/NaN audit `28/28`、post-export三manifest、pre-train `28/28`全部PASS；
- Tracker描述45字符，log INDEX含261；未修改E178/shared S6语义或E187 S1–S5/Full artifact。
# E187 latest manual authority / RL-export handoff (2026-08-03)

- Latest filled review: `results/E187/s6_downstream/eval/full/user_manual_review_filled.tsv`
- SHA256: `42dd0ef48cb47dde6cd35d88db0a4e46932e89bbe123227eae2b98f5252f431b`
- Final manual counts: `USE=14`, `DO_NOT_USE=8`, `PENDING=0`; quality: `CLEAN=7`, `MINOR_ACCEPTABLE=7`, `MAJOR_DEFECT=2`, `UNUSABLE=6`.
- User authority: export all 14 manually approved `USE` source cases for downstream RL validation; manual authority overrides numeric filtering for selection, while numeric status/failure modes remain required metadata.
- Partner information is a hard readiness gate: no source-only silent export; missing, ambiguous, or misaligned partner data must fail closed with an explicit blocked status.
- Governance: E187 C9 remains technical `FAIL`, progression authority remains `USER_WAIVED`; RL export/readiness is S6 downstream evidence and must not be reported as RL success.
- Plan created: `plan/211_E187_manual_use_partner_rl_export_plan.md`; next action is read-only discovery of the canonical RL consumer/export partner contract before implementation.
- Contract discovery: canonical S6 pairing is implemented by `finalize_reused_partner_rl.py`. It keeps `rl_export_input.tsv` as the source/CEM authority, infers the opposite CORE4D person in the same object/date/sequence, reuses a uniquely passing `ref_fk` Stage2b motion (variant preference `omnirt_v1`, then `omnirt_v2`), and emits a one-row-per-source paired manifest with hash-pinned partner artifacts. E187 should reuse this adapter rather than invent a new pair schema.
- E187 input facts confirmed: evaluation manifest has 22 unique source rows with `person`, `retarget_variant_id`, `scene_act`, `trajectory`, `contact_mask`, CEM result and frozen method/collision IDs; final manual file has 22 rows and is the selection authority.
- Manual `USE` set resolved to 14 cases across `bucket003`, `bucket004`, and `bucket007`. All partner case IDs are defined as the opposite person in the same sequence. E174 exposes canonical passing Stage2b manifests for `omnirt_v1/ref_fk` and `omnirt_v2/ref_fk`.
- Discovery caveat: Stage2b manifests retain historical remote absolute artifact paths under `/mnt/.../workspace/core4d/...`; the shared adapter intentionally remaps these to the local repository by the `/workspace/core4d/` marker. Ad-hoc direct path loading fails without this canonical remapping, so validation/export must use the adapter's resolver.
- Downstream contract confirmed from Holosoma v3 planning/registry docs: a new SPIDER handoff must first be ingested into the Holosoma upstream registry, then use the fixed export entry `rl_export_input.tsv + partner_omnirt_manifest.tsv -> manifest.tsv`; consumers must not scan CEM directories or use a one-off source-only path. The paired TSV remains useful audit evidence, while the fixed exporter consumes the separate source and partner manifests.
- E187 numeric risk source is `e187_case_metrics.tsv`; it contains `numeric_release_pass`, `numeric_failure_modes` and per-gate fields. These will be copied as metadata while human `USE` controls export selection.
- Holosoma fixed exporter contract inspected. It accepts all ready cases, requires a unique passing partner with existing `trimmed_npz`, aligns target/partner by common raw window (fallback min-frame crop), converts CEM `qpos[:,0,:]` plus the real opposite-person partner into `_mj_w_obj_w_partner.npz`, and supports `--dry-run`. The formal downstream sequence is `registry_gate.py pre-export` → exporter → register manifest → rebuild registry → post-export/pre-train gates.
- For this handoff, target-source validation should include both CEM and trajectory, and the final motion export should include the contact mask. No full RL training has been authorized by this input-packaging step.
- Implemented E187-specific S6 exporter at `scripts/experiments/E187/export_manual_use_partner_rl.py`. It freezes the final annotation SHA, selects exactly 14 `USE` rows across three objects, retains manual/numeric/C9 metadata, resolves same-sequence opposite-person partners through the shared E174 Stage2b adapter, hashes source/partner artifacts, and audits source/CEM/contact/partner frame counts plus common-raw-window alignment before publishing.
- Implementation intentionally leaves shared S6/E178 code unchanged; compatibility is by reuse/import of the canonical partner adapter rather than modifying legacy semantics.
- E187 exporter execution passed at `2026-08-03T18:54:06+08:00`: source `14/14`, partner `14/14`, pair complete `14/14`, all alignment audits use a valid `common_raw_window`, and all partners resolve to passing E174 `omnirt_v1/ref_fk` artifacts. Object split is bucket003=3, bucket004=2, bucket007=9.
- Exported source risk distribution is preserved: numeric release `5 true / 9 false`; manual quality `7 CLEAN / 7 MINOR_ACCEPTABLE`. Published status is `RL_EXPORT_READY`, not RL success; C9 remains `FAIL / USER_WAIVED`.
- Independent manifest audit passed: source/partner/paired/alignment tables are each 14 rows, case sets are identical, recorded artifact hashes match, and common raw overlap ranges from 76 to 201 source frames.
- Holosoma canonical `registry_gate.py pre-export` passed `14/14` for `target_source=both` with a unique existing partner per source. The subsequent fixed-exporter dry run did not start because the shell's default Python 3.13 lacks `mujoco`; this is an environment-selection issue after the input gate, not a data/partner failure. Next action is to use the repository's declared Holosoma runtime.
- Re-ran the fixed Holosoma exporter dry-run with its declared `hsretargeting` Python. It passed and selected exactly 14 source-partner pairs, planning both `cem` and `trajectory` targets (28 output motions total) with contact-mask inclusion. No files were written by the dry-run.
- Full conversion preflight found a downstream diagnostic-contract gap: Holosoma's fixed exporter has object AABB half-extents for bucket004 but not bucket003/bucket007. E177 mesh evidence gives full extents of approximately `0.541×0.763×0.466 m` and `0.547×0.574×0.570 m` respectively. These extents affect only exported partner-hand surface-distance diagnostics, not pair resolution or motion alignment; exact mesh bounds will be computed before actual conversion.
- Exact original-mesh half-extents computed: bucket003=`0.270393755,0.381407245,0.23287703`; bucket004=`0.161566625,0.231058755,0.15230013`; bucket007=`0.273275865,0.2869532,0.284911265` meters.
- First bucket003 conversion attempt exposed a downstream path bug: a relative `--out-dir` is written relative to the Holosoma root but passed unchanged to a converter whose CWD is deeper, so the converter cannot find its generated input. No manifest was published. Retry will use an explicit absolute output directory; source/partner data remain valid.
- Bucket003 formal Holosoma conversion has been restarted with an absolute output directory and is currently running; this exercises the actual converter rather than dry-run selection.
- Bucket003 formal conversion passed: 3 cases produced 6 paired motions (`cem` + `trajectory`) with output frames `334/209/204` per case and embedded contact masks. Output: `/home/ubuntu/Workspace/holosoma/workspace/v3/data/E187_manual_use_partner_rl/bucket003/`.
- Bucket004 formal conversion is running for 2 cases / 4 motions using the exact original-mesh half-extents.
- Bucket004 formal conversion passed: 2 cases produced 4 paired motions with output frames `237/207` per case and embedded contact masks. Output: `/home/ubuntu/Workspace/holosoma/workspace/v3/data/E187_manual_use_partner_rl/bucket004/`.
- Bucket007 formal conversion is blocked inside Holosoma's generic converter before manifest publication: its object model references `../bucket007/bucket007.obj`, which is absent from the converter's asset tree. The E187 scene and source/partner alignment loaded correctly up to conversion (`76` aligned input frames for the first case). This is a downstream object-asset packaging gap, not an E187 collision/partner gate failure.
- Root cause confirmed as case-sensitive legacy asset naming: Holosoma contains `models/Bucket007/Bucket007.obj` and `g1_29dof_w_Bucket007.xml`, while some E174 rows spell `object_name=bucket007` and select a non-existent lowercase directory. The E187 wrapper now normalizes all bucket007 downstream `object_name` values to canonical `Bucket007`; no mesh or motion data are changed.
- E187 source/partner manifests were regenerated after object-name normalization and again passed 14/14 source, partner, pairing and alignment checks. Bucket007 formal conversion has restarted against the canonical existing `Bucket007` Holosoma asset.
- Bucket007 converter is still running after successfully passing the previous missing-asset point; no new error has been emitted.
- Bucket007 18-motion conversion remains active; converter output is buffered, so completion will be audited from its final manifest and NPZ contents rather than inferred from silence.
- Bucket007 conversion remains active on the 9-case queue; no failure output is present.
- Bucket007 formal conversion passed: 9 cases produced 18 paired motions, with per-case output frames `125,125,189,145,155,154,222,174,174`; every case has both CEM and trajectory outputs with embedded contact masks. Output: `/home/ubuntu/Workspace/holosoma/workspace/v3/data/E187_manual_use_partner_rl/bucket007/`.
- Formal Holosoma conversion total is now 14 cases / 28 paired motion artifacts across bucket003/004/007. Next actions: independent NPZ/manifest audit, register the three manifests, rebuild registry, and run post-export/pre-train gates.
- Independent downstream artifact audit passed: all 28 NPZs have finite numeric arrays, `partner_hand_pos_w=(T,2,3)`, `partner_hand_quat_w=(T,2,4)`, embedded `object_contact=(T,2)`, manifest/output frame agreement, and unique `(case_id,target_source)` rows. Manifest SHA256: bucket003=`8c36db32...a212ce`, bucket004=`ea4d119a...60d5fc`, bucket007=`50688804...9f1b5`.
- Holosoma registry index was checked before mutation and is clean for the target registry/data paths; the three new manifests are not yet registered.
- Registered all three E187 manifests in Holosoma `workspace/v3/registry/motion_manifests.tsv` under owner scope `E187_manual_use_partner_rl`, then rebuilt `workspace/v3/registry_draft` successfully (`upstream=115`, `motions=73`).
- Holosoma `post-export` gate passed for all three manifests (`6+4+18=28` rows). Registry discovery found exactly 28 E187 motion IDs, and `pre-train` gate passed `28/28` for both target types (`spider` CEM and `omnirt` trajectory) with real OmniRetarget partners.
- Added machine-readable downstream evidence at `results/E187/s6_downstream/rl_export/holosoma_downstream_validation.json`; final status is `DOWNSTREAM_RL_INPUT_VALIDATION_PASS`, explicitly bounded below RL policy/training success.
- Formal results recorded in `log/261_E187_manual_use_partner_rl_export_results.md`, including authority, partner/alignment audit, 28-motion Holosoma conversion, registry gates, artifact SHAs, resolved downstream contract issues, governance boundary and the variables that must be frozen before any full RL training comparison.
- Tracker and log INDEX updated to log261/plan211. Final checks pass: exporter compiles, targeted `git diff --check` is clean, source/partner/paired/alignment tables remain 14/14, downstream evidence remains 28/28, bucket007 canonical asset name is frozen, and no shared data-construction/E178 file was modified by this stage.
