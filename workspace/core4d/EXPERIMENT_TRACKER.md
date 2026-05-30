# CORE4D 动力学重定向实验跟踪器

## 实验总览

| Run | 日期 | Phase | 描述 | 状态 |
|-----|------|-------|------|------|
| E100 | 2026-05-30 | Phase 21 | **contact target 重做 + 干净 A/B (exp_diagnostic_v2 Stage 2)**：用 E099 fingertip vote + quat audit (`disable_world_up=True` 17/17) 重做 contact target NPZ。**build_fingertip_aware_target.py**: 每帧把 spider FK palm 投到 E099 vote face 上 (vote 轴贴 ±half, in-plane 保留 palm 当前位置并 clip 到 ±(half-1cm))，IK 过拟合 case (vote_face='') target 退化为 palm_local; 单测 3/3 PASS (vote face 上 target 坐标精确等于 ±half[axis], 容差 0.5cm)。**全 16 case (3 skip raw 缺失) 生成 `spider_contact_target_object_local.npz`**；NPZ shape (T, 2, 3) in obj local；smoke test 验证 spider 仓库可正确加载 (log: `E085 external contact target: source=.../E100/fingertip_targets/box023_person2/... len 136→322`)。**守门反向 PASS**: face_changed_L 全 16 case = False, face_changed=False case (box025/box026/18029_p2/11035_p2/20019_p1) 全部 swap Δ=0.0cm; **DIFFER hand swap Δ 1.5-5cm**: 030_p1 R/20020_p2 R/box021_person1 R +z→+x (2.7cm), box004_083_p2 R -z→-x (4.5cm), box023_person2 R +x→+z (1.5cm) 等 7 个 case；与 E099 audit Tier 1 完全一致 (8/9 DIFFER 在 R)。**4 个 yaml override + run_E100_remote.sh** 就绪；CEM 实际触发推迟 E101 Phase 1 联合 (plan mitigation, torch.compile/triton 环境问题待 E101 Phase 0 排查)；9 个 Tier 1 case 对比 PNG 视觉直观 (绿方块=fingertip-target 移到正确 face) | ✅/⚠️ 详见 log 124 + plan 107 |
| E099 | 2026-05-30 | Phase 21 | **接触语义信息流补全（exp_diagnostic_v2 Stage 1）**：raw 5 指尖信息流补回 pipeline（不动 OmniRetarget IK，直接读 raw CORE4D 绕过 STAGE A 重跑）。**fingertip_face_vote.py**：SMPL-X 10 指尖 (L 27/30/33/36/39, R 42/45/48/51/54) Y-up→Z-up→obj local，每帧多数面投票；17/20 case 跑通，单测 8/8 PASS。**palm_face_vote_full.py**：复用 E098 anchor_refit 扩到全 17 case。**结果：9/33 hand palm vote ≠ fingertip vote 主面（27% mismatch），8/9 在 R 手**，2 个 case palm 说 contact 但 fingertip 说 no_contact → IK 过拟合直接证据；B6 假设强力验证。**quat_identity_audit.py**：扫全 17 case obj quat，**全部 disable_world_up=True（mean 84°-178°）**超 v2 §3 预测——不只 box021 D003，整个 CORE4D box family 都不能走 world-up 投影。**render_raw_contact_3d.py**：17 case × {4-view PNG + turntable mp4} = 32 文件；subagent 5/5 视觉签收 (5/5 几何贴合 + 5/5 vote 一致 + 4/5 PALM≠FINGER：palm × 飘出 box ≥10cm)。**audit 报告对 E100 列出 Tier 1/2/3 重做优先级 (Tier 1 共 9 case)** | ✅ 详见 log 123 + plan 106 |
| E098 | 2026-05-30 | Phase 21 | **诊断基础设施（exp_diagnostic_v2 Stage 0）**：落地 5 个 bug + 1 个新 gate。**B1**：5 文件的 `face_label` xy-only argmax → 全 3D（含 ±z），统一到新 `workspace/core4d/scripts/E098/face_utils.py`，9/9 单测 PASS；6 个典型 box021 D003 case anchor refit 实测 5/12 hand 主面翻到 ±z，4/6 case 至少 1 hand 翻面，cross-check v2 §3 B1 完全一致。**B2/B3**：E028b `_project_to_face` 通用化、E029 silent `+x` fallback 改 `raise`。**B4**：`spider/process_datasets/core4d.py:128` 加 deprecation comment 标注 `contact_pos` 是 IK FK palm 不是 raw mocap；v1 诊断报告加 ERRATA 块。**B5**：`anchor_face_gate.py` 实现 `anchor_face_review=true` 未 refit → raise，3 个自检 PASS。**新 CEM gate**（`replay_gate.py`）：pelvis_end_z<0.55 / pelvis_tilt_end>75° / lie_on_box_frac>0.30 三个信号；back-test 12 case (4 WORK + 8 FAIL) 召回率 **12/12 = 100%**，E094 C2 "趴箱" 与 E088B reward-hack 均被新 gate 拦下。可视化：6 case × {4-view PNG + turntable mp4}，subagent 视觉签收 12/12 hand 绿面贴合点云、5/5 DIFFER hand 旧面明显错位 | ✅ 详见 log 122 + plan 105 |
| E097 | 2026-05-29 | Phase 20 | **feature-based data_construction_v2 candidate refresh + visual correction**: 把 E095/E096/E096b 的经验写回候选挖掘逻辑：score 只用 raw contact + geometry/reach proxy + known outcome exclusion，不再用 object key 或 source scene/template readiness 加分。新增 `mine_feature_based_candidates.py` 和 `make_e097_visuals.py`。重新可视化原 6 条 Box021 enabled rows 后发现它们均已有 legacy D003/D004 outcome：`028_p1` OmniRetarget infeasible，`028_p2` D004 reject，`020_p2/035_p1/030_p2/019_p2` 为 D004 pass/review/check-shortcut；因此已从“新候选”队列移除。修正后 `38` 行 audit bank 中 `22` 条 verified excluded，`0` 条 unverified candidate，`0` 条 enabled next-batch。可视化 root: `workspace/core4d/results/E097/visual_review/`；下一步若继续找新数据，应先补 Box022 raw-contact preflight 或扩展 inventory，Box026 保持 large-reach holdout | ✅/🔬 详见 log 120/121 + plan 104 |
| E096b | 2026-05-29 | Phase 20 | **box004 mask-on full CEM rerun**: 对 E096 P1/P2 做最小 rerun，修正 E096 未接入 CORE4D raw 3cm contact mask 的问题。本轮保持 E096 的 `ref_fk + wrist5cm` target、E083-style leg+upper object pairs、5kg object、full CEM 32 iter，只把 `contact_hdmi_mask_source/path` 改为 per-case `core4d_3cm`。日志确认 P1/P2 均加载 mask：active L/R `61.8/61.0%`、`55.2/56.0%`。Full CEM 两条仍为 WORK：P1 contact `55.9%`、obj mean/max `0.007/0.018m`、pelvis `0.639m`；P2 contact `54.1%`、obj mean/max `0.011/0.034m`、pelvis `0.642m`；head/upper/hand-floor 全 `0%`。视觉帧复核支持 WORK。结论：E096 box004 positive 对 raw 3cm mask 接入稳健；后续 Holosoma RL 输入优先引用 E096b P1/P2 | ✅ 详见 log 119 + plan 103 |
| E096 | 2026-05-29 | Phase 20 | **box004 first-batch contact semantics + full CEM**: 承接 E095 三条 box004 候选，先按 E093/E094 口径做 contact semantics，再只对 preprocess-ready case 跑 full CEM。P1 `e091_box004_20231003_2_083_p1` 和 P2 `e091_box004_20231003_2_082_p1` 均为 WORK：contact `56.9/54.1%`，obj mean/max `0.007/0.022m`、`0.011/0.034m`，pelvis `0.639/0.643m`，head/upper/hand-floor 全 `0%`；high visual review 认定两者视觉支持 WORK。P3 `082_p2` no-fingertip 与 fingertip retry 均 OmniRetarget CVXPY infeasible，无 SPIDER trajectory，保持 preprocess blocked。下一步：P1/P2 加上 known-WORK `083_p2` 组成 box004 positive set，进入 Holosoma RL 输入准备；P3 不进入 CEM/RL | ✅/🔬 详见 log 118 + plan 102 |
| E095 | 2026-05-29 | Phase 20 | **Worklike data mining after Box026 failures**: 回到 Holosoma `data_construction_v2`/旧 D001-D002 inventory，按 E092/E094 经验重建 medium-box candidate bank：共 `32` rows，分层为 known work `1`、box004 priority `3`、box021 review `15`、Box022 raw-contact/reach review `6`、Box026 deprioritized `7`。第一批只跑 box004 priority 且 `REPLACE_WRIST_WITH_FINGERTIP=0`：`083_p1` 与 `082_p1` Stage2b/SPIDER verify 通过并 D005b PASS (`inside=0/0%`, support either `61.8%/47.7%`, pelvis `0.664/0.651m`)，OmniRetarget visual `2/2` OK；`082_p2` OmniRetarget CVXPY infeasible，作为 preprocess reject。新增 `box004_person1` source scene 与 E095 scene snapshot。下一步：对 `083_p1`、`082_p1` 跑 SPIDER full CEM，只有 WORK 序列进入 Holosoma RL；Box026 继续保持降权，不盲跑 | ✅/🔬 详见 log 117 + plan 101 |
| E094 | 2026-05-29 | Phase 20 | **G1-handbox-aware adaptive-support target projection + full CEM**: 基于 E093 几何审计实现 `adaptive_support` external target：guards box023/box004 reward delta p90 `0`、inside `0%`；Box026 support/inside 被修到 `95.9-100%`/`0%`，但 reward delta p90 仍 `0.36-0.63m`，预先标为 high-risk。三条 E092 case full CEM 已完成并回收：C1 `E094P1_box004_083_p2_hbproj` 为 WORK (`contact=61.0%`, obj mean/max `0.007/0.018m`, pelvis `0.658m`, head/upper/floor `0%`)；C2 FAIL (`contact=80.5%`, obj mean `0.002m`, pelvis `0.440m`)；C3 FAIL (`contact=41.5%`, obj mean `0.010m`, pelvis `0.171m`, RH floor `22.0%`)。用户指出 CEM 视频只看见上半身后，定位为 missing `front` camera fallback 到 pelvis `track` camera；已修为 `video_camera=auto` full-body free camera，并补生成三条 `*_autocam.mp4`，high review 确认 C1 视觉支持 WORK，C2/C3 是真实低髋/趴箱/倒地/箱体翻转，不应进 RL。下一步：C1 可作为 Holosoma RL 候选；Box026 应先做 E095 posture/valid-contact CEM gate | ⚠️/✅ 详见 log 116 + plan 100 |
| E093 | 2026-05-29 | Phase 20 | **Contact target geometry audit before CEM/RL**: 按用户要求在 box023/box025/box004/box021/box026 既有 case 上深究 `wrist_yaw_link + 5cm`、raw contact、G1 sphere、历史 HDMI 3-box、Holosoma handbox 的几何关系。7/7 case ready，raw target 从 raw mesh/person vertices 重新生成；summary 14 行、per-frame 1466 行；object-local/timeline/dashboard PNG `16/16` nonblank，MuJoCo keyframe/video `7/7` 非空且 mp4 decode PASS；用户指出旧 `track2` 视频只见上半身后，MuJoCo 可视化已重渲为 `auto` full-body free camera (`960x720`, 48 frames/case)，腿/脚/箱子/marker 均可见。关键结论：`wrist+5cm -> raw` 全部 >20cm，Box026/Box025 达 `49-64cm`；sphere p90 gap 全 >10cm；handbox 14/14 行相对最接近 raw 但仍非根治。box004/box023 work 是偏差仍可达/可容忍，D003 box021/Box026 fail 来自错面、inside、低 support、大 offset 组合；下一步应做 G1-handbox-aware target projection，而不是直接调 sphere/reward 或盲跑 RL | ✅/🔬 详见 log 115 + plan 99 |
| E092 | 2026-05-29 | Phase 20 | **Three-case SPIDER dynamic + direct OmniRetarget comparison corrected by full CEM**: 原 log 114 的 smoke-only 停止结论已作废。Stage A `spider_dyn full` 已补跑三卡并回收：C1 `E092D1_box004_083_p2_dyn` 为 WORK，pelvis `0.663m`、contact `64.8%`、obj mean/max `0.006/0.019m`、head/upper/hand-floor 全 `0%`；C2 `box026_039` FAIL，pelvis `0.083m`；C3 `box026_135` FAIL，pelvis `0.177m`、RH floor `17.1%`。Stage C direct Omni smoke 仍全 FAIL。机制分析：Box026 质量/摩擦已调到 5kg 且与 box004 一致，失败不是材质；contact target 是 dynamic `ref_fk + wrist+5cm`，不是固定点；C2 低 support/错面，C3 right-inside 风险。C1 应作为后续 Stage B/RL-from-SPIDER 唯一候选，但 E093 先暂停盲跑，转 contact geometry 审计 | ⚠️/🔬 log 114 需修订；full 结果在 `workspace/core4d/results/E092/spider_dyn/full/` |
| E091 | 2026-05-29 | Phase 20 | **Holosoma data_construction_v2 medium-box discovery results**: v2 链路已跑通到 top-bank + minimal smoke。Phase0/1 生成 `80` 行 medium manifest、`16/16` raw-contact PNG 非空，并补 `box026_person2`/`box004_person2` source templates。Box026 no-fingertip top3 初筛：`039_p2` Stage2b pass 但 D005b support `19.5%<30%` reject；`040_p2` CVXPY infeasible；`135_p2` Stage2b pass 但 R-inside `12.2%>10%` reject，因此不扩大同配置。box004 `e091_box004_20231003_2_083_p2` Stage2b pass (`105` frames) + D005b PASS (inside `0/0%`, support `42.9%`, pelvis `0.679m`) + high visual PASS，写入 top bank rank1。Minimal smoke 跑通：head/upper `0%`、hand-floor `0/1%`、object mean `0.006m`，但 pelvis min `0.079m`，high review 判定 REVIEW / dynamics pelvis collapse；不回滚 D005b/top bank，不在 E091 内继续优化算法 | ✅/🔬 详见 log 113 + plan 97 |
| E090 | 2026-05-28 | Phase 19 | **H2-first fingertip replacement ablation + true pre-IK topface repair + SPIDER smoke/full**: 按用户补充的 Box025 reach 动机，先验证 `--replace_wrist_with_fingertip` 而非直接归因 original OmniRetarget。Phase0 修 `g1_feasibility_gate.py` 为 world-up face + legacy local-z support 双口径，canonical D003 Box021 仍全 reject，Box023/Box025 calibration pass。Phase1/2: no-fingertip 两条可跑 case inside `0/0%` 但 support 仅 `17/15%`、`27/22%`，第三条 CVXPY infeasible；topface-preIK Box021 2/3 gate pass，剩余 1 条只因 `T=78<80`。Guard: Box023 topface pass，Box025 topface reject (`inside=48/52%`)，证明不能全局 topface 或全局删除 fingertip replacement。Phase3 smoke 派生 m10+upperobj task：S1 pass (`contact=95.6%`, head/upper/floor `0%`, pelvis `0.538m`)，S2 fail (`LH_floor=56.4%`)。Phase4 S1 full: safety 仍 `0%`、obj `0.009m`，但 pelvis collapse `0.134m`，full fail。结论：topface-preIK 修复几何安全，但单独不足以解决 CEM 低姿态局部解；不扩展 D003 13 case，下一步做 S1-only pelvis/upright elite gate 或等价姿态约束 | ❌/🔬 详见 log 112 + plan 96 |
| E089 | 2026-05-28 | Phase 19 | **G1-Feasibility gate 验证 (A 路: box021_person1 SPIDER full CEM; B 路: D003 box021 13 case post-IK top-face 修复; B4: B-path top-2 SPIDER smoke)**. 诊断驱动 — `workspace/exp_diagnostic/diagnostic_report.md` 定位 E028→E082-E088 box021 全失败的主因是 OmniRetarget G1 IK 双手目标几何不可行，设计 5 阈值 G1-Feasibility gate 区分 12 已知 case 100% 正确。**A 路** (现成 `box021_person1` + ref_fk + E088 hard-gate stack, ~24min full CEM): pelvis_min=`0.687m`，contact=`60.2%`，obj_err=`1.3cm`，head/upper/hand-floor 三项 `0.0%/0.0%/0.0%` —— **box021 首次同时满足"不摔+不穿物体+手不撑地"**，对比 E087A 89%/89%/81%、E088A 28%/55%/11%。**B 路** (subagent post-IK damped-LS, 13 D003 case): wrist world-up-face-frac 4-14%→99-100%，R-inside-box 33%→0%；按 world-up 语义 gate **9/13 PASS** (严格 gate 因 `top_face_frac` hardcoded local +z 在 box021 90°-X 旋转下误报 0/13)。**B4** (top-2 case `031_p2_btop`/`020_p1_btop` SPIDER smoke 各 4 iter): head_pen `0.0%/0.0%`，upper_pen `1.9%/0.0%`，hand-floor `7.5%/0.0%`，远低于 E082-E088 baseline 70-89%。**4/4 claims 通过**。下一步: P1 修 gate world-up，P2 B-path top-2 跑 full CEM，P3 集成到 holosoma D005b | ✅ 详见 log 111 + plan 95 |
| E088 | 2026-05-28 | Phase 18 | **Hard safety gate + absolute object clearance**: 完成 `d003_box021_20231018_029_p2` 的 `10kg` 派生 scene 三组 full CEM。A hard gate only: contact `82.17%`、obj mean/max `0.695/1.148m`、head/upper `27.91/55.04%`、fallback `90.71%`；B low contact/object: pelvis 稳到 `0.643m`，但 head/upper `71.32/75.19%` 且 valid frac 为 `0`；C + absolute clearance: obj mean 降到 `0.612m`、bottom clearance mean `5.5cm`，但通过翻箱/侧倒实现，LH floor `40.31%`、upper pen `80.62%`。三组均未通过 gate，`accepted_variants=[]`。结论：hard gate 暴露当前可行样本极稀薄；绝对 clearance 修正了旧 `object_lift_rew/object_floor_penalty` 口径，但需要 anti-tip / hand-floor / upperbody hard gate 后才可能作为 seed，不建议直接接 RL | ❌ 详见 log 110；plan 94 |
| E087 | 2026-05-28 | Phase 18 | **Box021 mass + reward breakdown audit**: 按用户建议暂停 COLA/support-body，检查物体质量与 reward 分项。mass audit 发现 D003 Box021 系列 object mass 为 `29.632kg`，而 `box023_p2`/`box025_p2` 为 `5kg`；创建 `5kg/10kg` 派生 scene 并跑 main gate。三组均失败：5kg raw contact `82.9%` 但 obj mean `0.782m`、head/upper penetration `89.1/89.1%`；10kg raw obj mean `0.735m`、head/upper `69.8/76.0%`，略好但仍不可用；5kg safety tuned contact `65.9%`、head/upper `89.1/89.1%`。reward breakdown 显示 E085 main `contact=2.747`、`qpos=1.965`、`task_obj=0.383`，但 upperbody penalty 仅 `-0.030`；E087C penalty 提到 `-0.634` 仍挡不住压箱。结论：Box021 质量异常是因素但非唯一主因；当前 soft reward 调权不足，下一步应做 hard safety gate / elite filtering，并修正 object lift/floor reward 口径 | ❌ 详见 log 109；plan 93 |
| E086 | 2026-05-28 | Phase 18 | **raw-target CEM failure iteration**: 沿 E085 发现继续验证两组局部修复。E086A 加 `hand_object_deep_penalty`、upperbody penalty `2->8`、contact gain `5->3`，但 main 仍失败且更差：contact `76.7%`、obj mean/max `0.698/1.134m`、head penetration `58.9%`、upperbody penetration `72.9%`、RH floor `10.9%`。E086B 将 left target 抬到 min vfrac `0.2`，仍失败：contact `75.2%`、obj `0.710/1.170m`、head/upper penetration `62.8/69.0%`。结论：低位 target 是因素但不是唯一主因；penalty/vfrac 小修不能把 CEM 从头胸压箱局部解拉回真实手部承重，应切换到 COLA-style support body / 6-DoF connector seed | ❌ 详见 log 108；plan 92 |
| E085 | 2026-05-28 | Phase 18 | **raw contact target repair + gate**: 修复 E084 暴露的 contact target 语义问题：动态 target 从 G1 `wrist_yaw_link+[0.05,0,0]` 改为 raw CORE4D hand/object surface 生成的 object-local external target；同时修正 raw visual target 投影到 collision box 时 inside point 误选 face 的 bug。核查显示 mask/person/time 没明显错误，old G1 pseudo target 与 raw surface 平均差 `27cm`；修复后 main left target 仍是低侧面，vfrac mean `0.057`，right 高侧面 `0.895`。CEM main contact `82.9%` 但 head/upper penetration `18.6/53.5%`、hand penetration `73.6/54.3%`；guard contact `88.0%`、obj mean `0.226m`，但 hand penetration 仍高且 head pen `2.0%`，gate `0/2`。结论：预处理 bug 已修，但直接用 raw hand target 做物体支撑仍诱导压箱/穿透局部解 | ❌ 详见 log 108；plan 91 |
| E084-audit | 2026-05-28 | Phase 18 | **E084 contact target 语义核查**: 针对用户指出的“左手是否在箱体下沿/底面附近”做 raw/contact-mask/G1-target 三层诊断。修正前一表述：MuJoCo object frame 中底面是 `-y`，不是 local `-z`；E084 G1 ref reward point 实际偏上侧，left/right surface vertical fraction `0.836/0.870`，不是底面。mask 自动选择 `eval_contact_mask_3cm`，`125→200`，person2 active `68.5/74.0%`，raw frames `75-129`，未发现错人/错帧。raw CORE4D person2 接触很强，left/right broad min dist mean `1.8/2.2mm`，但 raw left 是低侧面、right 是高侧面。核心问题是 G1 `wrist_yaw_link + [0.05,0,0]` 动态目标与 raw surface centroid 平均相差约 `27cm`，主要沿 object `x` 在对侧；说明失败主因是 retargeted wrist pseudo contact target 失真，而不是 mask 二值门控本身。E085 应先改 raw-contact/IK/support seed，不继续基于 G1 wrist target 小调参 | 🔬 详见 log 107 |
| E084 | 2026-05-28 | Phase 18 | **Box021 constraint groups main gate**: 按 plan 89 只在 `d003_box021_20231018_029_p2` main 上先跑 A/B/C 三组。A=safety penalty，pelvis min `0.596m` 且 hand-floor `0/0%`，但 upperbody penetration `80.6%`、object-floor `93.8%`、bottom gap `-11.9cm`；B=upright/ctrl trust，obj mean 降到 `0.417m`、bottom gap `-4.3cm`，但 contact 仅 `9.3%`、LH floor `39.5%` 且视觉翻箱；C=semantic/lift 与 A 类似，upperbody penetration `82.2%`、object-floor `96.9%`。三组 main gate 全失败，`accepted_groups=[]`、`guard_splits_to_run=[]`，未跑 `box023_p2` guard。结论：继续 CEM reward 小调参只会在“稳定但不抬”和“激进但翻箱”之间切换；下一步 E085 转 seed/可行性审计、kinematic/support seed、hard-gate staged CEM | ❌ 详见 log 106；plan 90 |
| E083 | 2026-05-28 | Phase 18 | **upper-body-object collision pairs 验证**: 沿 log 104 的诊断，为 3 个 D003 Box021 main 和 `box023_p2` guard 新建 `*_upperobj_e083` 派生 task；每个派生 `scene_act.xml` 保留 16 个腿/脚-`object_collision` pair，并新增 7 个 head/torso/pelvis/shoulder/elbow-`object_collision` pair，`npair=49`。本地+远程三卡 full CEM 完成并回收，Box021 main `0/3` 通过：obj mean `0.608/0.903/0.866m`，pelvis min `0.478/0.563/0.193m`，upperbody penetration `82.2/92.5/83.1%`，object-floor `94.6/100/97.3%`；视觉为趴箱/浅穿/手撑地/腿部干涉。`box023_p2` guard 通过：obj mean `0.160m`，pelvis min `0.675m`，upperbody penetration `0%`。结论：upper-body pair 对 guard 安全，能减少 E082 的深穿箱，但不能解决 Box021 的错误接触语义；下一步 E084 规划三组 reward/constraint 实验：safety penalty、upright/ctrl trust、semantic hand contact + lift | ❌ 详见 log 105；plan 89 |
| E082 | 2026-05-27 | Phase 18 | **D003 Box021 三 case 回到 E081 leg/foot-object collision 路线 + 上半身穿模诊断**: 在 `workspace/core4d` 完成；为 `d003_box021_20231018_029_p2`、`d003_box021_20231011_035_p2`、`d003_box021_20231020_019_p1` 新建 `*_legobj_e082` 派生 task，原始 source task 不改；每个派生 `scene_act.xml` 新增 16 个腿/脚-`object_collision` pair。三卡 full CEM 完成并回收，`0/3` 通过：case-window obj mean/max 分别 `0.680/1.138m`、`0.457/1.162m`、`0.849/1.480m`，sim contact `14.7/37.4/69.6%`，pelvis z min `0.355/0.183/0.188m`，视觉均为倒伏/推箱/压箱/物体漂移。追加诊断确认：scene 有 `head_collision/torso_collision` geom，但没有 head/torso/pelvis/shoulder/elbow-object pair；手-地面 pair 存在，所以 CEM 可利用“头/躯干穿箱 + 手撑地”局部解。三例 head/torso 穿入率 `76.0/85.3%`、`17.2/51.1%`、`32.4/58.8%`，box023 guard 为 `0/0%`。下一步应做 E083A upper-body-object collision pairs，必要时再加 upperbody/hand-floor/stability/ctrl penalty；不建议把 E082 输出接后续 RL | ❌ 详见 log 103/104 |
| E081 | 2026-05-16 | Phase 18 | **leg/foot-object collision 派生 scene 验证**: 不改原始 `scene_act.xml`，新建 `box025_person2_legobj` 与 `box023_person2_legobj` 派生任务，在派生 `scene_act.xml` 中追加 16 个腿/脚-`object_collision` pair。box025_p2 本地、box023_p2 guard 远程 GPU1 均完成。结论：box025 p2 腿/箱 case-window interference `28.9%→7.5%`、obj mean/max `0.146/0.289→0.143/0.271`，但 object bottom/floor-contact 没改善，仍是 partial positive；box023 guard 基本不破坏，interference `0→2.7%`、obj mean `0.162→0.164`。新增碰撞物理上必要，但主要瓶颈转向 lift/floor-contact 与性能成本 | ⚠️ 详见 log 102 |
| E080 | 2026-05-15 | Phase 18 | **box025 大物体边界复查**: 按 E079 no-hold + 3cm mask 口径跑 `box025_person1/person2`。两者 CEM 均完成；case-window 三阈值把 p1/p2 都判 True (`2/2=100%`)，fixed post2 均 False (`0/2`)。二次复核修正结论：p1 是 false positive，腿/箱几何干涉重；p2 视觉上确实接近搬/扶箱，应标为 partial positive / near-usable，但仍有右腿/脚局部干涉和箱体高度低于 ref 的问题。scene 无腿/脚-箱 contact pair，腿不会物理支撑箱子。下一步必须加入 leg-box interference、object lift/floor-contact、object max/orientation/contact continuity/semantic visual label | ⚠️ 详见 log 101 |
| E079 | 2026-05-15 | Phase 18 | **CORE4D 10+ 高接触质量 case 泛化验证**: E077 pipeline 推广到 6 个 B+C 序列 p1/p2，`11/12` 可运行（`desk021_p2` Holosoma retarget infeasible）；主验证不使用 hand-crafted hold window。按用户纠正后接入 case-specific contact/intent window，并修正 role：`box023_p2` 是 E078 positive guard，`box023_p1` 是 main/已知失败反例。main `6/10=60%` 数值成功（fixed box023-post2 旧口径 `2/10`），低于 C3 `>=7/10`；用户复查后视觉口径更保守：`box021_p1` 视觉好但 ref 接触位置异常，`bucket007_p2`/`bucket005_s2_p2` 相对可信，`desk021_p1` 前段没抬起，`bucket005_s2_p1` 物体持续受力旋转，`box023_p1` false positive，`bucket007_p1` 是 trim/ref data issue。结论: 数据 pipeline work，算法有跨 case 正信号但泛化未过关，下一步需更强语义判据 + trim/ref feasibility/stability audit | ⚠️ 详见 log 100 |
| E075 | 2026-05-14 | Phase 18 | **限时 hold_contact 组合验证**: E075B=E074A + hold_contact scale1.0/window1.8-2.5s 达到 best-so-far: frame100-145 contact 76.1%, post2 contact 67.9%, obj_err max 28.7cm, robot ctrl Linf max 0.690, f180 站稳且箱子分离; 但 first obj_err 仍 f100, f145-f166 释放不干净. E075A scale0.5 虽 hand SDF 更近但 f168 pelvis<45cm、f166-f180 摔倒/腿箱干涉. 结论: 采用 E075B 为下一步 base, 转 release/clearance 诊断 | ⚠️ 详见 log 96 |
| E074 | 2026-05-14 | Phase 18 | **post-2s hold/contact 首轮远程并行**: E074A ctrl guard 将 robot ctrl 大偏离 f101→f122, contact 45.7→54.3%, obj_err max 29.3→28.9cm, 视觉最接近成功但 f145 仍接近落地; E074C hold-contact 将 post2 contact 49.4→64.2%, SDF mean 6.6→4.3cm, 但 first zero contact 提前 f101、obj_err max 32.4cm、后段腿/箱干涉明显. 结论: ctrl guard 是安全组件; hold-contact surrogate 生效但目标错位, 不应原样组合 | ⚠️ 详见 log 95 |
| E074 preflight | 2026-05-14 | Phase 18 | **E074 base/palm normal 前置分析**: 明确 E074 base=`E073 -> E071W02 -> E062 -> E041c`; E060-E067 中仅 E065 有共享 reward 代码但默认 inactive, E066/E067 YAML inactive; E062 palm normal 是 contact_hdmi orientation reward 的 wrist-local 朝向先验, box023 双手 `[+1,0,0]` 已包含在 E071/E073 结果中, E074 主线应保留并只作为后续 ablation 验证 | 📋 详见 log 94 |
| E073 | 2026-05-14 | Phase 18 | **contact target eef_offset 口径修正**: dynamic target 从 ref wrist origin 改为 ref `wrist+eef_offset`; 训练日志确认 `uses_eef_offset=True`. early drift 未回归(yaw 0.574/1.075°, B1=0.080m); first zero contact frame100→108, post2 contact 44.4→49.4%, obj_err max 30.8→29.3cm, pelvis 不再低于45cm. 但 first obj_err>25cm 仍 frame100, 视觉 f130 后脱手/f145 箱落地, 结论=部分有效但未解决 hold | ⚠️ 详见 log 93 |
| E072 | 2026-05-14 | Phase 18 | **box023 post-2s hold/place failure 诊断**: replay E071 qpos + scene snapshot; frame100/eval2.00s obj_err=30.8cm 且 sim hand-object contact=0(ref=1), pelvis 到 frame166/eval3.32s 才低于45cm; post2 contact frames sim 44.4% vs ref 80.2%; 结论=hold/contact 先失效, 摔倒是二阶后果, object ctrl mapping 非新问题 | ✅ 详见 log 92 |
| E070 | 2026-05-14 | Phase 18 | **MJWarp ref-control parity 诊断定位根因**: CPU MuJoCo 与 MJWarp 完全一致; `qpos_ctrl` 口径精确复现 E069 yaw 12.403/22.156° 且 vs E069 qpos≈0; `orig_ctrl` 口径降到 0.574/1.075°. 根因不是 physics/gains/CEM, 而是 `run_mjwp.py` 用 `qpos_ref[:, :nu]` 把 floating base 混入 robot ctrl | ✅ 详见 log 90 |
| E070 plan | 2026-05-14 | Phase 18 | **MJWarp ref-control commit parity 诊断计划**: 固定 `qpos_ref[0]/qvel_ref[0]/ctrl_ref[0:12]`, 对比 MuJoCo `mj_step` 与 MJWarp `step_env` 的 yaw/foot/qvel/contact/actuator force, 用于定位 E069 中 ref ctrl 仍漂的动力学 mismatch | 📋 待确认 |
| E069 | 2026-05-14 | Phase 18 | **First-tick ref-control warmup 验证失败**: W02/W05 warmup ctrl diff=0, 证明 ref ctrl 确实提交; 但 t=0.017/0.033s yaw drift 仍 12.40/22.16°，B1=0.222/0.428m. 结论: first CEM override 不是主因, 真问题转向 MJWarp `step_env(ctrl_ref)` vs MuJoCo `mj_step(ctrl_ref)` 动力学不一致 | ❌ 详见 log 89 |
| E068 | 2026-05-14 | Phase 18 | **MJWP init drift 诊断修正**: `mj_forward` init 完全对齐, init `mj_step` 只偏 0.22°; 真实 E062/E063 t=0.017/0.033s yaw drift=12/22° 来自 first committed CEM ctrl, robot ctrl 首帧偏 ref 1.56rad(object 仅0.01) | 诊断完成 |
| E001 | 2026-04-30 | Phase 0 | 数据管线: holosoma → SPIDER 格式, Box025 场景 XML | 通过 |
| E002 | 2026-04-30 | Phase 1 | SPIDER MJWP 无引导 (Box025 p1): pelvis=0.10m, obj=0.83m (物体落地) | 完成 |
| E003 | 2026-04-30 | Phase 1 | SPIDER MJWP 有引导 (Box025 p1): pelvis=0.07m, obj=0.83m (物体仍落地) | 完成 |
| E004 | 2026-04-30 | Phase 1 | 强增益 kp=100/1000, decay=1.0: 物体仍落地 → 确认根因: 最终迭代归零设计 | 完成 |
| E005 | 2026-04-30 | Phase 4 | 混合轨迹导出: SPIDER机器人(物理)+运动学物体 → Holosoma格式 (52body,50fps) | 通过 |
| E006 | 2026-05-01 | Phase 1 | 前臂接触重定向: 3-box碰撞+contact_rew+高权重 → **视频证实箱子未离地** | 虚假突破(更正) |
| E007 | 2026-05-01 | Phase 1 | 路径Y物理对齐: physics_dt=0.005+Holosoma PD → PD过弱, obj_err 退化到 0.80+ | 失败 |
| E008 | 2026-05-01 | Phase 1 | 视频驱动诊断: obj z实测 max=0.307m(从未离地), 衰减PD引导期OK撤除即落 | 失败 |
| E009 | 2026-05-01 | Phase 1 | Person2力支撑5方案: 箱底面均≤8mm — **几何错位:双人对夹±x端** | 失败(几何洞察) |
| — | 2026-05-02 | Phase 2 | **规划**: Phase 2 路线图 (E010-E012), 死磕重定向 | 规划完成 |
| E010 | 2026-05-02 | Phase 2 | Connect/Kinobj重定向: **G1运动学可行性确认** (pelvis_min=0.733, 身体稳定) | **通过** |
| E011 | 2026-05-02 | Phase 2 | Mocap Partner协作: obj_z=0.460(有提升) 但pelvis崩溃; 架构限制 | 部分成功 |
| E012 | 2026-05-02 | Phase 2 | 导出Holosoma格式+partner数据: 格式完全匹配, pelvis_err=0.083m | **通过** |
| — | 2026-05-03 | Phase 3 | **规划**: Phase 3 路线图 (E013-E015), 修复E011架构限制+死磕重定向 | 规划完成 |
| E013 | 2026-05-03 | Phase 3 | Intra-rollout Mocap + Reward Sweep: 技术修复成功, 最佳r7偶尔C1+C2通过(obj=0.477,pelvis=0.712) **但高方差** | 部分成功 |
| E014 | 2026-05-03 | Phase 3 | 增大Partner碰撞体: 原始capsule从未碰到箱子(gap=0.175m), 增大后力太混乱 | 失败 |
| — | 2026-05-03 | Phase 4 | **规划**: Phase 4 路线图 (E015-E016), 小物体验证+双机器人 | 规划完成 |
| E015 | 2026-05-03 | Phase 4 | Bucket005小物体: body-only pelvis_err=0.129m, 物体未搬起; **修复scene_name bug** | 完成 |
| E016 | 2026-05-04 | Phase 4 | **双机器人Gibbs CEM**: nq=79 nu=58, obj_z=0.488(92%ref); **视频复查: connect约束强行拉物体, 机器人头歪/手穿模, 非真实搬运** | ❌ connect假象 |
| E017 | 2026-05-05 | Phase 4 | **双机器人 Soft 2-Connect**: obj_z_max=0.597(112%ref); **视频复查: 脚悬空+手张开, connect约束悬浮物体, 视觉不可用** | ❌ connect假象 |
| E018 | 2026-05-06 | Phase 4 | **Task-Space奖励(DynaRetarget)+Interaction(Harmanoid)**: obj_err↓49%; **视频复查: 头穿进箱子+身体扭曲, connect假指标; E031泛化4case全失败** | ❌ connect假象 |
| E020 | 2026-05-06 | Phase 5 | **多Case诊断(5物体×2模式)**: 全部stable=100%, 仅bucket005 pelvis_err<0.20m; chair022碰撞推飞 | 诊断完成 |
| E021 | 2026-05-06 | Phase 5 | **IK可达性+Anchor**: 行走是主因(87-96%), Pelvis XY Anchor使err↓47-72%, box025降至0.186m | **突破** |
| E022 | 2026-05-06 | Phase 5 | **Anchored+ObjRew**: desk005 lift=37%+视频确认手接触; box025臂展限制不变; bucket010 strong崩溃 | 部分成功 |
| E023 | 2026-05-06 | Phase 5 | **Full Anchor(XY+Yaw)**: bucket010 pelvis_err↓48%(0.15m), 但obj初始距离0.77m超臂展; 单人不可解 | 结构性结论 |
| E024 | 2026-05-07 | Phase 5 | **Partner Force(50-90%grav)**: 手从未主动碰物体, 90%下obj因失重飘起=偶发碰撞; CEM不产生接触 | 失败(结构性) |
| E025 | 2026-05-07 | Phase 6 | **Hand Approach Reward**: 手确实接近物体(dist 0.34→0.00), 但本质是推/碰静止物体, 非沿ref搬运; partner横向动态缺失 | 部分成功(技术有效, 目标未达) |
| E026 | 2026-05-07 | Phase 6 | **Sustained Contact (4096samp/24iter)**: bucket直立+持续接触, 但仍是推静止物体; 长horizon(2.4s)反而保守 | 同上 |
| E027 | 2026-05-07 | Phase 6 | **Partner Force Sweep + desk005**: 接触率不依赖pf强度(均64%); desk005也产生碰触但非搬运 | 同上 |
| E028 | 2026-05-07 | Phase 7 | **阻尼弹簧 4Case全覆盖**: desk005 z=84%/hand=90%/stable=100% 最佳指标; 但**所有case物体翻转**(orientation不受控); orientation spring/damping均不稳定 | C6 FAIL (视频不像搬运) |
| E029 | 2026-05-07 | Phase 7 | **Quasi-Kinematic(kp=100)**: pos跟踪改善(box025=0.07m), 但**仍翻转**; chair022 pelvis崩溃(27%); 结论:xfrc_applied无法控制freejoint orientation | FAIL (根本性限制) |
| E029-act | 2026-05-07 | Phase 7 | **PD Actuator+Kin Override**: contact_guidance CEM干扰obj ctrl; kin override不在rollout生效; **核心发现:scene.xml模式CEM不采样物体,问题是xfrc torque不稳定** | 方向明确(待debug torque) |
| E030 | 2026-05-07 | Phase 7 | **Orientation Debug+Hybrid Export**: CPU torque OK; Warp正反馈/weld被碰撞压倒; anchored body-only 4/4 stable但**hybrid export质量不足以支撑RL**——机器人姿态与搬运无因果关系 | 质量不足 |
| E031 | 2026-05-07 | Phase 8 | **双机器人Connect泛化4case**: 全部失败——行走位移+connect拽倒机器人; Gibbs在connect下有害; 58维CEM采样不够; E018的box025成功不可泛化 | FAIL (结构性) |
| E027b | 2026-05-07 | Phase 8 | **Object PD Override(scene_act+grav_comp+relative_euler)**: desk005 pos=0.10/rot=8.8°★★★, box025 rot=7.6°★★; 4/4 stable=100%; 修复3个bug(euler约定/slide偏移/body_quat相对旋转) | **desk005成功** |
| E027c | 2026-05-07 | Phase 8 | **Contact Guidance(OMOMO方案)on CORE4D**: 机器人走路(pelvis=1.6m)但物体不跟随; **关键发现:论文loco-manipulation用HDMI simulator不是MJWP**; MJWP contact_guidance的gain更新可能不生效 | FAIL (simulator限制) |
| E027d | 2026-05-08 | Phase 8 | **HDMI Physics+Debug**: 验证CUDA graph gains有效; 根因=object ctrl未重置为ref+noise_scale=0; 修复后仍需debug rollout内部state reset | 调试中 |
| E027d2 | 2026-05-08 | Phase 8 | **Body-Frame Fix+Commit Gain Restore**: 3个bug修复; box025=100%stable+1.46m; desk005=87%+1.54m; **body tracking有效(机器人站着走), 但物体通过PD actuator驱动而非真实接触搬运** | body tracking有效 |
| E032a | 2026-05-08 | Phase 9 | **Hand Approach+Reward Sweep**: HA=3提升contact(desk 9→83%, bucket 36→63%); task_body有害; base=10平衡stability/contact; PD sweep: 降低无效 | 完成 |
| E033 | 2026-05-08 | Phase 9 | **desk005 σ sweep+CEM budget**: σ=1.0达95%stable+91%<15cm(目标达成); 增加iter/horizon/samples反而恶化stability; tradeoff是根本性的 | **stable目标达成** |
| E034 | 2026-05-08 | Phase 10 | **HDMI-Style Reward: Stability Penalty**: bounded qpos失败(CEM无法区分站/倒); stability_penalty有效(不稳定时长↓86%, min_z 0.22→0.55m); **但desk005仍有t≈1s严重前倾(视频证实)**; contact mask对desk005无效(始终在范围内) | 改善但未解决 |
| E035 | 2026-05-08 | Phase 10 | **Local-Frame Body Tracking**: 移植HDMI yaw-local reward; desk005: pelvis_min=0.657m+Contact<10cm=94.4%(历史最佳); 3/3 cases零摔倒; **body tracking突破, 但视频显示机器人丢下桌子自己走了, 不是搬运** | body tracking突破 |
| E036 | 2026-05-08 | Phase 10 | **关闭hand_approach**: MPKPE 48cm→1.4cm; Contact<10cm暴跌(desk 94→7%); **确认: body tracking和contact是tradeoff, CEM无法同时优化** | body tracking突破 |
| E037-E039 | 2026-05-08~09 | Phase 11 | **Contact Reward系列**: box-SDF/HDMI-aligned设计; **发现config bug: contact reward从未执行** (E036关闭approach→body_ids=[]→reward条件false) | Bug发现 |
| E039b | 2026-05-09 | Phase 11 | **Config Bug Fix+Rotated SDF**: 修复后contact首次生效; box025=85%/bucket010=76%/desk005=81%; **但发现"手粘连物体"问题(固定target)** | 突破+新问题 |
| E040 | 2026-05-09 | Phase 11 | **Dynamic Per-Frame Target**: 动态target未解决手背接触问题; position-only reward根本缺陷=无方向约束; CEM用手背满足距离→不自然; Contact<10cm=64/66/4%; 需添加orientation reward | ❌ 不自然行为未消除 |
| E041 | 2026-05-09 | Phase 11 | **Orientation Reward(乘法门控)**: palm normal方向约束; 手掌朝向有所改善; 但乘法gating过严导致Contact/Stability退化(62/56%); body前倾问题仍存在; CEM难同时优化position+orientation | ⚠️ 方向正确但约束过严 |
| E041c | 2026-05-09 | Phase 11 | **Additive Ori(w=0.3)最佳变体**: Contact=66/57%+Stable=100%+MPKPE=1.4cm; **CEM reward sweep最佳, 但视频显示: box025趴在箱上, desk005丢下桌子走, bucket010手碰桶侧(推非搬)** | ★ CEM最佳(非搬运) |
| E042 | 2026-05-10 | Phase 11 | **Wrist Freeze(零化手腕噪声)**: 对齐HDMI做法; box025持平(64%), bucket010暴跌(48%); freeze阻止CEM补偿body tracking误差; HDMI成功因ref精确在把手; **确认contact上限≈64-66%(box025)** | ❌ 有害 |
| E043 | 2026-05-10 | Phase 11 | **原始OmniRetarget Ref对比**: Phase3(无松弛)反而更差; box025 64→52%, bucket010 66→57%; Phase4松弛版contact更优; desk005例外(stability改善) | Phase4更优 |
| E044a | 2026-05-10 | Phase 12 | **Wrist Weight=2.0**: box025 contact持平(67%)但stability崩溃(73%); pelvis_min=0.113m; 增强上半身权重→下半身stability退化 | ❌ stability退化 |
| E045 | 2026-05-10 | Phase 12 | **Sigma Sweep(0.3/0.15)**: 收紧sigma→contact全面下降(box025 66→38/46%, bucket010 57→20/23%); desk005局部改善(4→22%); **证实contact瓶颈不在tracking精度** | ❌ 无效 |
| E047a | 2026-05-10 | Phase 12 | **SBTO对齐DynaRetarget**: 修复5个偏差(Sigma EWMA/收敛准则/mean EWMA/elite fraction); 机器人摔倒(MPKPE=155cm); α_μ=0.95太保守+σ_min=0.01太紧; **SBTO+exp-kernel reward不兼容** | ❌❌ 失败 |
| E044b | 2026-05-10 | Phase 12 | **Wrist Weight=3.0**: box025 contact 59%+stable 100%(比w=2.0更稳但contact↓); bucket010 22%/92%; desk005 MPKPE=1.1cm最佳; **weight越大CEM越保守** | ❌ contact退化 |
| E047b | 2026-05-10 | Phase 12 | **SBTO放松参数(α_μ=0.5,σ_min=0.03)**: stability恢复98-100%(E047a=31%), 但MPKPE=69-87cm仍极差; **SBTO开环优化无法替代MPC闭环反馈** | ❌ tracking差 |
| — | 2026-05-11 | Bug Fix | **碰撞盒模板Bug修复**: 21/21 case碰撞盒全部修正为mesh AABB×1.05; box023从1.8x过大修正; bucket010 Y/Z互换修正; box025增大24% | 修复完成 |
| E048 | 2026-05-11 | Phase 13 | **碰撞盒修复后Baseline+HDMI对比**: 碰撞盒修复21 case; HDMI评估发现严重bug(内部ref漂移); **视频复查: 所有case均无搬运 — box023摔倒(ObjPos=14cm是假象), desk005丢下桌子走, box025趴着; 数值指标系统性误导** | ⚠️ 指标不可信 |
| E049 | 2026-05-11 | Phase 14 | **HDMI优化移植失败+eval修正**: 三优化(PD+damping+noise)直接移植使Stability崩(box025 98→57%); HDMI eval指标全部虚假(内部ref漂移); **之前"E041c object tracking优于HDMI 2-8x"结论不成立 — 都是假指标** | ❌ 移植失败 |
| E050 | 2026-05-11 | Phase 14 | **Euler Convention Fix尝试**: "xyz"→"XYZ"引发gimbal lock(box025 Y=89.4°); ObjPos 24→108cm; 已回退 | ❌ gimbal lock |
| E051 | 2026-05-11 | Phase 15 | **HDMI Scene物理配置全面诊断**: (1)Euler mismatch影响所有case(box023=178°,box025=140°); (2)修复euler反而恶化5×(内部自洽被打破); (3)**根因=scene物理配置:hand=1sphere(应3boxes),armature=1.0(应0.01),foot=4spheres(应7capsules)**; 需从suitcase模板重建scene | 方向明确 |
| E052a | 2026-05-11 | Phase 15 | **Suitcase模板+旧euler**: 3-box hand+低armature(0.01)+euler=xyz; ObjPos 37cm(比baseline 24cm更差); Joint 13.9°(比7.3°退化); **低armature损害body tracking, 错euler使好hand无效** | ❌ 单修scene不够 |
| E052c | 2026-05-12 | Phase 15 | **Suitcase模板+正确euler(XZY)**: ObjPos=98cm, Stab=32%(摔倒!); **2×2矩阵最差组合**; 结论: E048a baseline(24cm/7.3°/100%)是HDMI在CORE4D上的极限, euler/"错误"config实为CEM已适应的状态, 修正只会破坏 | ❌❌ 全矩阵失败 |
| E053 | 2026-05-12 | Phase 16 | **碰撞盒Margin Sweep(0.90/0.95/1.00×3case)**: box025上0.90最佳(pelvis_min 0.660 vs 1.05x的0.575); bucket010上1.05x反而最好(0.90/1.00 stability降至88-90%); desk005中间值(0.95-1.00)最差(Stab 66-78%); **不同物体形状需要不同margin, 无全局最优; 碰撞盒不是搬运失败的根因** | ⚠️ per-case策略 |
| — | 2026-05-12 | Phase 17 | **路线图**: E001-E053总结+下一步规划; 4路径(A收口/B warmstart/C force-closure/D差分物理); 暂不进RL | 规划完成 |
| E054 | 2026-05-12 | Phase 17 | **Case Tier + Mocap质量分析(21 case, v3 detector + 视频核实主导手)**: B+C=6(box021/023, bucket001/005_s2/007, desk021), C-only=0(数据天然空), dual-robot=2, drop=13; **重要修正:box023是Tier1非Tier3, 之前与box025同处理浪费5+实验**; **6个B+C case的hand-obj距离19-37cm验证mocap retarget后从未真接触**; **v3 detector关键: band ∩ (slow_rel OR lifted_amp_scaled), 物理必要性>运动学统计**; **dom_hand 视频核实: 仅bucket001是single-hand(L_mean=23cm vs R_mean=62cm, sym=0.37), 其他5个全是both-hand(sym≥0.94); 之前用"L最近帧占比"判错3/6, 改用"两手平均距离比"后6/6匹配视频** — 4/5 Claims通过(C4数据集原因) | **✅ 收口完成** |
| E055 | 2026-05-12 | Phase 17 | **box023 Hand-Snap Warmstart (Path B 首验证, 无CEM)**: DLS单臂IK + frame-warm-start + 5cm offset 把双手投影到box表面; intent窗口58帧×2手=116次snap, 全部关节限位满足, IK final到target≤1cm 100%; 视觉对比5帧目检 — snap中段双手对称握box, ref中仅单手贴边 ✅; **重要发现 1**: G1 palm site在wrist+8cm是手中部不是接触面, hand_collision是5cm半径球, 5mm offset会10cm穿模; 改用5cm offset后palm距box表面5cm, hand球穿模降至8cm; **诊断工具产出**: 6面命名约定 (+yz/-yz/+xz/-xz/+xy/-xy) + face_distance_timeseries.py + visualize_frames.py + ref_diagnosis 抽帧; **box023 ref 状态 (E056 视频核实后修正)**: L 在 -xy 底面(托底), R 在 -yz 远侧(扣远) — **垂直握, valid**, 之前错误地同意用户"L 在 +xz" 是 sign 漂移误判 — 3/5✅ + 2/5⚠️ | **✅ Path B 几何验证 + 诊断工具** |
| E056 | 2026-05-12 | Phase 17 | **多 case Hand-Face 诊断 (6 B+C case)**: 复用 E055 6面命名 + signed_dist 时序 + main_face 判定 (中位数<7cm + 60%帧贴近); 5类 grasp 分类 (对侧/垂直/错位/同面/单手); **关键结果**: 5/6 case valid — **bucket005_s2 是唯一真正"对侧"握 (-yz/+yz, 88帧最长 ⭐ E057首选)**; box023/bucket007/desk021 = 垂直 valid; bucket001 = 单手 valid; **box021 = 同面异常** (双手都在 +xy 顶面, 视频核实是"按压"非"搬运", 应从 Path B 移出); 视频核实 3 case × 3 帧, 算法分类 3/3 一致; **教训**: 接触面必须 signed_dist≥0(palm 在外侧), 多 case 比单 case 调参高效, half-sizes 必须从 model 读不能 hardcode | **✅ E057 决策完成 (路线 A: bucket005_s2)** |
| E057 | 2026-05-13 | Phase 17 | **bucket005_s2 Hand-Snap (Path B 第 2 case, 对侧握姿)**: 复用 E055 hand_snap_ik (无改动) + 新增 verify_snap_face.py (C6 face 验证); intent 88f×2 hands=176 snap, palm-to-surface init mean=5.04cm → final mean=4.93cm (target=表面外 5cm), IK residual mean=0.11cm/max=1.12cm, **关节限位 100% (176/176)**; **C6 PASS**: snap 后 L=-yz (med 2.90cm, 77% close)、R=+yz (med 1.87cm, 100% close), 与 E056 诊断完全一致; **5/5 keyframe 视觉合格**: intent mid (t=2.10s) 双手对称在 bucket ±yz 两侧, 教科书搬运姿势 ⭐, pre/post blend 平滑; vs E055 box023: intent +52% (88 vs 58f), grasp_type 升级 (垂直 → 对侧), snap 流水线 case-agnostic 验证 | **✅ 6/6 通过 (Path B 几何验证 + face guard)** |
| E058 | 2026-05-13 | Phase 17 | **Path B-CEM 首跑 (bucket005_s2 baseline vs warm)**: spider/config.py +warmstart_qpos_path, run_mjwp.py +30行 hook (intent 内 ref+ctrl 替换); 并行 GPU0/1 train, baseline 36min, **warm 3h54min** (intent 内 plan 14s→99s, contact 约束爆炸); **结果: 双方都摔 (pelvis_min=0.11m)**, contact% baseline 36/37/12 vs warm 35/20/9 (-3pp/-17pp R 暴跌), main_face 都 None; **唯一正信号: stable_intent 12.5%→33% (+20pp)**, 视频 t=2.10s warm 单手仍按桶 (baseline 完全脱手); **教训: warmstart 修几何不修物理稳定性, E041c reward 在 bucket005_s2 上 baseline 就站不住 → 应先验稳定 baseline 再加 warmstart**; **环境副产品**: uv re-resolve 升 torch 2.11/坏 nccl 已回滚 2.8.0; spider/interp nearest+align_corners bug 已绕过; 缺 python3.12-dev 用 use_torch_compile=false 代偿 | ❌ 2/6 (流水线 OK 内容失败) |
| E059 | 2026-05-13 | Phase 17 | **Path B-CEM 第 2 case (box023, 区分 E058 失败原因)**: 复用 spider 改动, 加 PYTHONUNBUFFERED 实时进度; 并行 32min/33min (warm 没爆炸); **真正发现: 与 E058 同一情景 = baseline 摔 + warm 摔** (baseline pelvis_min=0.14m, warm 0.08m, 都 < 0.20m 摔倒), 与 E048 视频复查 "box023摔倒" 一致; warm L contact 9%→44% 是因 warmstart 把 L 锁在 box 底面 (机器人摔倒中手仍贴), 不是搬运成功; both contact < 6% 全程, 视频 5/5 keyframe 都摔 (warm 摔得更彻底); **L main face = -xy 命中 expected 是边际信号** (warmstart hook 设计正确, 没 bug), 不构成搬运突破; **(Pre-E060 audit 修正)** 原 log "stability_penalty 是根因" 判断窄, 真正根因是**多层叠加**: 数据 (hand=sphere + box023 margin 1.05x) + reward (palm_normal hardcoded 是 box025 fingerprint, 数值证 box023 L 上是噪声) + 基础设施 (scene XML 没入 git) | ❌ 1.5/6 (与 E058 同情景, baseline 是真正 bug, 详见 audit log 70) |
| Audit | 2026-05-13 | Phase 17 | **Pre-E060 全面审查 (E041c 数据层 + reward task-specific)**: 触发 = 用户质疑 E059 单一根因. 4 个独立发现: (1) 代码层 e567fe7→HEAD 无 silent override, E049 port 字段默认关闭; (2) 数据层 04e7ecc 把 box023 collision 1.8x→1.05x, 但 E041c 从未在修复后的 box023 上验证, bucket005_s2 在 E058 之前从未跑过 E041c → **calibration set overfitting**; (3) hand_collision 至今是 5cm sphere, HDMI 3-box (E051 定位的根因) 从未推广到 G1 robot.xml → **E041 ori reward 整个动机是物理上无效的视觉补丁**; (4) box023 collision margin 仍 1.05x, E053 在 box025 上明确结论 0.85-0.90 最优, **从未推广到 box023**; (5) 数值验证 palm_normal hardcoded `[0,∓1,0]`: box025 完美 (L/R both 78-80% 校准命中), box023 L 是噪声 (mean dot 0.04), bucket005_s2 L 居然正确 (+0.79 同 box025) — 部分纠正 "bucket005_s2 朝向错" 假设; **基础设施修复**: 9 case scene XML force-add 入 git (29 文件), snapshot_scenes.sh 脚本, dual-safeguard 规则进 .claude/rules + skill | 📋 详见 log 70, **指导 E060 必须先修数据层再做 reward 消融** |
| E060.0 | 2026-05-13 | Phase 17 | **数据层修复后 baseline (E041c, no warmstart) on box023 + bucket005_s2**: Phase 1 完成 (commit fa2e181) — 3-box hand port 到 robot.xml + 9 case scene.xml (patch_hand_3box.py), box023 collision margin 1.05→0.90 (set_collision_margin.py), snapshot 入 git; **结果**: 仍摔, 数据层修复**单独不足**. box023 pelvis_min_intent 0.176m (vs E059 0.140m, +3.6cm), bucket005_s2 0.117m (vs E058 0.110m, **基本无变化 +0.7cm**), main_face 全 None; **唯一正信号**: box023 stable_intent +14.8pp 是因大部分时间 ≥0.5m, 但仍有几帧蹲到 0.18m; contact% 升降是摔倒过程中手蹭物的伪信号; **物理验证**: 12 pair 没让 plan time 爆炸 (14s/iter 同 E059); **⭐ 第二次 over-optimism (系统性教训, 见 log §5)**: 我把 box023 t=1.65s "前扑中右臂伸到箱旁" 误判为"单手抱箱站立 53 实验首次", 用户纠正后修订. 与 E059 完全同样错误模式. 强制流程: 任何 ⭐/"首次" 视觉描述前必须验证 pelvis_z 全 trace + intent end + ref 同帧对照 + 3-frame 一致性; **下一步**: 进 Phase 3 E060.1 (`contact_hdmi_ori_weight=0.0`) 验证 hardcoded palm_normal 是否真元凶 | ❌ 1/6 (流水线 OK, 数据层修复无效) |
| E060.1 | 2026-05-13 | Phase 17 | **`contact_hdmi_ori_weight=0.0` ablation on box023 + bucket005_s2** (CLI override on E041c, 数据层同 E060.0): **反直觉 case-divergent 结果**, 部分**纠正 audit §2.2 预测**; box023: stable_intent **-21pp** (74→53), pelvis_mean_intent **-11cm** — **关掉 ori 反而退化**, 即使 audit 预测 L 是噪声; bucket005_s2: stable_intent **+39pp** (15→54), pelvis_mean_intent **+17cm**, intent mid 视觉从"接近平躺"→"深蹲弓步" — 显著改善, 但 audit 预测 L 是对的 (+0.79); **新假设**: hardcoded palm normal 起作用的不是"指向物体" 而是"给 CEM 一个 wrist 朝向 prior 约束探索空间", box023 上即使次优 prior 也比无 prior 好, bucket005_s2 上 prior 跟实际接触几何 (侧地+90°) 冲突, 移除让 CEM 自由探索更好; **PASS 严格 0/4**: Δpelvis_min_intent box023 -3cm, bucket005_s2 +9.3cm 差严格阈值 7mm; **教训 1**: 静态点积分析有局限, 没考虑 reward 项作为"约束 prior"的隐式作用 → ablation 必须 ≥2 case 才发现 case-divergence; **教训 2**: 强制流程拦下第三次 over-optimism — 看 4 帧前先校验 pelvis trace, bucket005_s2 t=2.10s 的"深蹲弓步"被识别为 0.45m 蹲位而不是站立; **(post-log 74 修正)** 这个 case-divergent 结论**可能不可靠** — 见 log 74, 整个 E060 phase 建立在 3-box port 引入的物理 bug 上 | ❌ 0/4 (mixed signal, **结论需 sphere baseline 验证后重新评估**) |
| E060.2 | 2026-05-13 | Phase 17 | **case-correct palm_normal ablation** (yaml override): box023 L/R=+x (audit §2.2 真最优), bucket005_s2 L=-y unchanged R=+x; **CATASTROPHIC FAIL**: box023 stable_intent **74→5%** (-69pp), pelvis_mean_intent **0.58→0.26m** (-32cm); bucket005_s2 stable_intent **15→2%** (-12pp), pelvis_mean_intent **0.31→0.25m**; 两个 case 全程趴地; **修订自 E060.1 的"implicit prior + geometric alignment"假设被反转**, +x prior 比 hardcoded -y/+y 和 ori=0 都差; **(同时被 box025 regression 解释)**: 这个 catastrophic 程度可能不全是 reward 问题, 是 3-box geometry 与 reward eef_offset 错位在不同 palm normal 下放大不同程度; **教训累计 #4**: "E060 是诊断 reward task-specific 的好框架"假设也被反转 — 整个 phase 建立在物理 bug 上; **下一步**: 暂停 E060 reward ablation, 必做 D 方案验证 sphere baseline (log 74 §4) | ❌❌ 0/3 (catastrophic, **E060 reward ablation 全部暂停**) |
| Audit | 2026-05-13 | Phase 17 | **🚨 Box025 3-box hand regression 发现 + E060 phase invalidation**: 用户在远程跑 `run_box025_3box_regression.sh` 验证 3-box port (commit fa2e181) 对历史 baseline 的影响; 三个 reward stack (E041c, E041, E039) **全部 regression** — pelvis_min 从 sphere 时代 0.575m 跌到 0.16-0.25m, stable% 从 100% 跌到 60-69%; **E039 (无 ori reward) 也 regression** → 排除"reward 是元凶"; 视频显示物体被推开/掉地, sim 行为完全异常 (绕到 box 后方/趴在 box 上); **根因**: `contact_hdmi_eef_offset = [0.05,0,0]` 是 sphere 时代调的, 3-box 时代 box3 末端在 wrist+17.5cm, **reward 看的位置 (wrist+5cm) 跟实际碰撞末端错位 12.5cm** → CEM 优化数学高 reward 但物理上撞翻物体; **影响**: E060.0/.1/.2 全部建立在物理 bug 上, log 71/72/73 reward 结论都需重新验证; **教训 §5.1**: 物理 port 必须配套历史 case 回归测试, 没回归测试不允许做 reward 调参; **教训 §5.2**: reward eef_offset / palm_normal 等"task-specific" 实际是"geometry-specific", 改 geometry 时全部要重审; **强制下一步**: D 方案 sphere 验证 box025 + E041c 能否复现 0.575m, 通过后 (B) 修 3-box geometry 让其末端 ≈ wrist+8cm 或 (A) revert sphere; 通不过则 git bisect 找其他 regression commit | 🚨 详见 log 74, **E060 全部暂停, 必做 sphere 验证** |
| E061 | 2026-05-13 | Phase 18 | **Sphere baseline verification (D 方案)**: 临时 git checkout `fa2e181~1` 把 robot.xml + box025 scene 退回 sphere, 跑 box025 + E041c (其他完全不动); **结果决定性 PASS**: pelvis_min **0.672m** (历史 0.575m, 新 3-box 0.253m, +0.42m), Stable% **100.0%** (新 3-box 61.7%, +38.3pp), 全程 124/124 帧 ≥ 0.81m; 5/5 keyframe 视觉确认 sim 站立扶箱跟 ref 同侧, 无 T-pose/掉箱/绕后行为, 复现 E048 sphere 时代描述; **决定性结论**: **3-box port (fa2e181) 是 box025 regression 的 SOLE 根因**, E041c reward stack 在 sphere 几何下完全 work; E060.0/.1/.2 全部建立在物理 bug 上, log 71/72/73 全部需在物理修复后重做; **干净 single-variable 验证**也间接证明非 robot.xml 代码改动 (E058 warmstart hook 等) 不影响 sphere baseline; **Phase 4 已 restore**: 6 hand geoms + npair=34 (3-box) 恢复; **下一步**: 等待用户决定 B (修 3-box geometry / eef_offset) vs A (revert sphere); 详见 log 75 §7 | ✅ 5/5 通过 (诊断决定性确认) |
| Strategic | 2026-05-13 | Phase 18 | **🚨 E041c reward stack 完全是 box025 sphere 的过拟合 — A/B 决策框架失效, 转向 X1+X2 reward 泛化方向**: 用户提示已存在数据 `E041c_box023_collision_fixed.npz` (= E048_box023, sphere+E041c+box023): pelvis_min **0.193m**, stable 50% — 也摔; 加 E061 (sphere+E041c+box025 ✅ 0.672m) + E060.0 (3-box+E041c+box023 ❌ 0.176m) + 远程 (3-box+E041c+box025 ❌ 0.253m) → **4-cell 矩阵中只有 box025+sphere 这一个组合 work**; 用户 insight: "box025 work 是 task-specific 设计或偶然, 完全一样的方法换 box023 就不行了"; 重审 reward 参数: `palm_normal=[0,∓1,0]` 是 box025 motion fingerprint (audit §2.2 box023 L mean dot 0.04 噪声), `eef_offset=[0.05,0,0]` 是 sphere geometry calibration, **两者都是 hand-tuned 常数, 任一变量改变 → reward prior 失效 → 摔**; **A (revert sphere) / B (修 eef_offset) 都只是"让 box025 重新 work", 不解决泛化, 推 E054 标的另 4 个 B+C case 同样会摔**; 转向: **X2 = per-hand auto-derived eef_offset (从 robot.xml 算 hand collision centroid, geometry-agnostic), X1 = per-case auto-derived palm_normal (从 ref motion 算每个 case 的 wrist→object 主轴, case-agnostic)**; **教训 #5 (累计)**: 任何 reward 设计成功声明必须基于 ≥2 case × ≥2 hand 横向验证, 不能基于单 case 调通 | 🚨 详见 log 76, **A/B 框架失效, 转 X1+X2** |
| Correction | 2026-05-13 | Phase 18 | **🔄 E041c box025 真问题不是反关节而是臂展物理硬限制 — 修正"3-box 解决反关节"的错误论证, X1+X2 改为基于 sphere**: 用户基于 9 帧视觉重审 `E041c_box025.mp4` 发现 — sim 没有戏剧性"手背接触 / 反关节" (E041c additive mode 在 sphere 上已基本解决 E040 的反关节问题), 真正问题是 **(P1) body tracking 不准 (sim 站到 box 左前角而非正前) + (P2) G1 臂展 ~63cm < 人臂展 ~75cm 的物理硬限制 + (P3) 下半身深蹲不稳 + (P4) 右肘外翻 (P1+P2 二阶: 站偏+短臂 → 必须横身体伸右手抓左前角)**; t=2.0s 关键帧明显; "Stab 100% 是假象" — sim 靠 box 撑着 (黄色 contact marker 在 box 角); **修正错误论证链**: 我之前 (log 76 §5) 说 "sphere 各向同性 → ori reward 视觉补丁 → 必须 3-box 解决反关节", 实际 E041c additive mode 软性 wrist 朝向 prior 在 CEM 探索阶段已引导正确方向, **3-box 不是必需**, 它的唯一独有价值只是跟 HDMI/Holosoma 跨 framework 一致性; **用户决策**: P1-P4 是物理硬限制不易改善暂不修, 继续 case 泛化 (P5) 但**改为基于 sphere** (放弃 3-box, 接受失去跨 framework 一致性); X1 (auto palm_normal) 优先, X2 (auto eef_offset) 在 sphere 上是 nice-to-have; **教训 #6 (累计)**: 视觉描述必须区分实验版本, 不能用 EM 的描述评判 EM' (E040 反关节描述不能外推到 E041c) | 🔄 详见 log 77, **X1+X2 改为基于 sphere, X1 优先** |
| E062 | 2026-05-13 | Phase 18 | **⚠️ X1 Auto Palm Normal on Sphere — Mixed result (box025 PASS self-consistency, box023 PARTIAL: novel carry-fall-recover behavior)**: Phase 1 commit 61abf4c revert 9 case scene + robot.xml 的 hand 到 sphere (保留 box023 margin 0.90); Phase 2 写 `compute_palm_normal.py` (audit §2.2 工具化, ref-motion proximity-windowed dot product); 自动算: box025 L=`[0,-1,0]` R=`[0,+1,0]` (== E041c default, **完美 self-consistency**), box023 L=R=`[+1,0,0]` (audit §2.2 +x best); **box025 PASS**: pelvis_min 0.688m vs E061 0.672m (+1.6cm), stable 100% — 算法 + override 机制正确, 不破坏 baseline; **box023 PARTIAL**: pelvis_min **0.058m** (E048 0.193m, **-13.5cm 退化**) BUT pelvis_mean 0.602m (+8.7cm) + stable 77.9% (+27.9pp) + max 0.828m 全面改善; **行为模式 novel** "carry attempt + fall + push-up recovery + stand": 0-0.7s 站立接近 → 0.7-1.7s 弯腰 carry 姿态 → 2.0-2.7s 摔倒 (DEEP FALL 至 0.058m) → 2.7-3.0s push-up 紧急站起 → 3.0-4.5s 完全直立 (但 box 没搬走). 跟 E048 "fall and stay" 完全不同; **诊断**: X1 让 sim 真的"尝试搬运" (1.5-2.0s 弯腰前推), 但物理不稳/重心前移导致 t=2.0s 后栽倒, CEM 找到 active recovery 解; **教训 #7**: 自动算法 self-consistency 通过 ≠ 对新 case 有效, X1 假设可能本身不充分; **下一步**: X5 (推 X1 到 4 cases 看 outlier 模式) / X1+X2 (加 auto eef_offset) / X3 (诊断摔倒物理原因) — 等用户决策 | ⚠️ 2.5/5 (box025 ✓, box023 mixed) |
| R4-diag | 2026-05-14 | Phase 18 | **🚨 HDMI vs MJWP diagnosis — INIT POSE BUG smoking gun (sim t=0 pelvis 偏 ref 22° yaw)**: 用户授权 R4-direct, 用现成 trajectory_hdmi.npz (E052c_box023_euler_fix) 跟 MJWP 直接对比, 不训练. **HDMI box023 完美 work**: B1=0.066m ✓ (vs MJWP-E062 0.48m, **7× 高**), pelvis 9s 稳在 0.62-0.75m, 物体 z 0.49→0.61m **真搬起来 + 维持**. **HDMI ctrl = ctrl_ref + 13% residual (PPO 学到平衡 prior)**. MJWP 已经 init `ctrls=ctrl_ref`, noise scale 实际 0.025-0.05 比 HDMI residual 0.13 **还小** → 噪声不是元凶 (推翻 log 86 candidate C). **真元凶**: t=0 pelvis quat ref=(0.657,-0.029,0.006,0.753) yaw=97.8°, HDMI sim=(0.657,-0.034,0.009,0.753) yaw=97.7° (**仅差 0.1°**), MJWP-E062 sim=(0.504,-0.005,0.027,0.863) yaw=119.5° (**差 22°**), 接下来 2 帧再漂 30°到 yaw=148°. HDMI 同期保持 ±0.5°. 已 verify init code path 数学正确 (`d_act.qpos[:] = ref qpos[0,:42]` mj_forward 后 pelvis xquat 完全对齐), warmstart='', ref qvel=0 — 都不是元凶. 候选剩余: (a) mjwarp `put_data` GPU 转换 lossy, (b) 保存的 sim qpos[0,0] 是否 post-1-step physics, (c) robot scene_act kp=500 + obj kp=20 在 init 阶段 target≠state 蹬反作用力, (d) 物体 actuator init 不准. **教训 #17**: 对比 init pose 必须 dump quaternion 全维度, 不只 pelvis_z. **R5 候选** (按侵入性): A 减弱 init kp (yaml + 改 XML), B 加 init_kp_warmup config (spider 代码), C snap 第 1 frame 到 ref qpos (powe-test), D fallback HDMI, E minimal test 隔离 (a)-(d). **PAUSE 等用户决定 R5 路径** | 🚨 详见 log 87 |
| E067 | 2026-05-14 | Phase 18 | **❌❌ Port HDMI body partition (lower 12→6, upper 17→6) — CATASTROPHIC FAIL, sim 做手倒立**: 双变体 box023 only 双 GPU (33min). N (only narrow): B1=**1.30m** (vs E063 0.48m, **2.7× WORSE**), C2=0.26m (vs 0.54), C3=**6.9%** (vs 67%) — sim t=0.8s 右腿伸直朝上 **handstand**, C4=1.75cm 仍 PASS 因强 actuator 拽 box; NS (full HDMI clone narrow+soft+exp): B1=1.25m, C4=160cm. **Hypothesis log 85 §4 完全反了**: 不是 mean dilution 稀释 outlier, 而是 more body tracking 提供 DoF 约束防 CEM 找极端 pose. HDMI 用 6 bodies 是因 PPO 已学 prior, MJWP CEM 没 prior 需要更多约束. **3 round 全 FAIL, 触发 ✋#3 (handstand 是新 failure mode 第 1 次出现) + ✋#5 (4 round 累计协议)**. **诊断指向更深层**: (A) PPO learned prior vs CEM no-prior, (B) knot_dt 0.10 vs HDMI 0.20 (2× 密 control), (C) noise schedule reverse (MJWP 0.5→1.0 vs anneal-down), (D) initial pose quat 差异. **决策**: 暂停, 等用户决定 R4 方向 — 候选 R4-direct (run_hdmi.py on box023 诊断 trace), R4-hot-start (CEM warm start ref ctrl), R4-knot (knot_dt 0.20 + horizon 加长), R4-noise (anneal down), R4-data-fix (用户曾说 ref 左手发不上力). **教训 #15**: more body tracking constraints HELP CEM 即使有 mean dilution. **教训 #16**: 3 round ablation 全失败强烈暗示元凶在更深层 (solver / time res / noise), 不在 reward / dynamics 单点 | ❌❌ 详见 log 86, **PAUSE 等用户** |
| E066 | 2026-05-14 | Phase 18 | **❌ Port HDMI object actuator gains (kp 500→20, kp_rot 50→0.3, decay 1.0→0.85) — FAIL, soft actuator 让 sim 不搬箱子, 揭示第二个 mismatch: body partition 稀释**: 双变体 box023 only 双 GPU 并行 (33min). **A** (drop task_obj + soft act): B1=0.53 (vs E065-A 0.30, 反退化), C1=0.34 (大改善 vs 0.12), C2=0.60 ✓ C3=93% ✓, **C4 final_obj_err 暴涨 3.10→112.01cm** — sim "前倾 dive → 撑地起身 → 弃箱直立" 不搬箱; **D** (exp form + soft act): B1=0.75 (反退化), C4=138.18cm 更糟. **关键**: 即使 task_obj=0 + soft actuator 双重保险, sim 仍 lunge B1=0.53 → 元凶不在 actuator 也不在 task_obj. **新诊断**: MJWP `local_frame_lower_ids = list(range(2,14))` (12 bodies, hip_yaw/roll/pitch + knee + ankle_pitch/roll), HDMI 用 6 bodies (`hip_pitch + knee + ankle_roll` × 2). MJWP `error.mean()` 把脚约束**稀释 2×** — 一只脚抬高 0.30m 对 12 bodies mean 只贡献 0.025m, reward 几乎不变. 类似 dilution 在 upper (17 vs 6) 解释了 E044 wrist_weight 反方向. **教训 #13**: reward `error.mean(dim=1)` 形式下增加 tracked bodies 反向稀释 outlier penalty, 不是 "tracking 越多越好"; 搜寻 reward diff 不能只对比公式, 还要对比 body_id 列表 / mask / weights. **教训 #14**: soft actuator 是 trade-off, 必须配套 task_obj 加大补偿失去的 actuator-driven tracking. **决策 R3**: E067 = port HDMI body partition (lower=[2,5,7,8,11,13], upper=[17,20,23,24,27,30]) + 双变体 N (narrow only, keep 强 actuator + L2 task_obj) vs NS (narrow + soft act + exp task_obj 完整 HDMI clone) | ❌ 详见 log 85, 进 R3 E067 |
| E065 | 2026-05-14 | Phase 18 | **❌ task_obj_rew form ablation on box023 — FAIL, 但揭示真元凶不是 reward form 而是 actuator stiffness**: A (drop task_obj) + D (HDMI exp form) box023 only 双 GPU 并行 (33min). **A**: B1 0.48→**0.30m** (改善 38% 但仍 ✗ 阈值 0.10), C1 0.19→**0.12m** (反退化), pelvis 永不 recover; **D**: B1 0.48→**0.69m** (反恶化 44%, 极端 lunge), C1 0.11m, 但 pelvis 在 t=4s recover 到 0.74m; 两者 C4 final_obj_err = 3.10/1.75cm 都 PASS — **task_obj reward 在物体追踪中是 redundant signal, 物体跟随主要靠 ref ctrl + body tracking 拉手**. 关键诊断: E065-A task_obj=0 时 sim 仅靠 qpos_rew (= local_frame_rew, 字节匹配 HDMI sigma 0.5/1.0/0.5/0.25 + W_TRACK=0.5) 就 lunge 了. log 82 hypothesis 部分推翻 — task_obj 是 contributor (38% improvement) 但不是 sole cause. **剩余唯一未对齐项**: actuator stiffness (MJWP kp=500 vs HDMI kp=20). 物理含义: kp=500 物体几乎刚体, 手 push 必须用 25× 大力 → 反作用力把 COG 推前 → 单脚后伸 lunge 当反平衡. **决策**: ❌❌ → R2 = E066 port HDMI actuator gain (object actuator kp 500→20 + decay 1.0→0.85), 双 GPU = E066 (D + actuator) + E066b (A + actuator). **教训 #12**: 调 reward 之前先确认 actuator/PD/joint limit 等物理参数 baseline 跟 reference workflow 是否对齐; 不要把 dynamics 问题当 reward 问题调 | ❌ 详见 log 84, 进 R2 E066 |
| E065 plan | 2026-05-14 | Phase 18 | **📋 task_obj_rew form ablation 设计 (A=drop / D=HDMI exp form) — 待跑**: 综合 log 80/81/82 + 用户 HDMI 对照 + 用户提议 "对齐 hdmi 的 rew_obj_pos 指数形式". 4 yamls 创建 (E065-A/D × box023/box025), 代码改动完成 — `spider/config.py` 加 3 字段 (`task_obj_use_exp:bool=False, task_obj_pos_sigma=0.5, task_obj_rot_sigma=0.5`), `spider/simulators/mjwp.py` task_obj_rew 块加 `if use_exp: scale*exp(-err.norm/sigma) else: -scale*err²` 分支, 默认 backward-compat (use_exp=False 等价旧 L2). 4 期望矩阵: A✅+D✅→形式是元凶, 永久 port HDMI exp; A✅+D❌→ task_obj 信号本身有问题, 永久关; A❌+D❌→ 不止 task_obj, 进 E066 port actuator gain 500→20; 双 PASS 后进 contact 数据修正 (用户判断 ref 左手发不上力). 训练: 2 round 并行 GPU 0/1 ~66min total. **新增 PASS criteria** B1=t=0-2.0s max(Lf,Rf) ≤ 0.10m, B2=单脚撑帧数=0 (E062-E064 max(Rf)≥0.20m, ~50 帧单脚撑), 沿用 C1-C5 + R1-R3. 详见 log 83. **教训 #11 完整版**: 3 次失败 mode 不同 = reward 有 unbounded/总是 active 项扰动 CEM, 优先检查 saturating vs unbounded 而非调 weight | 📋 详见 log 83, 待跑 |
| Update | 2026-05-14 | Phase 18 | **🎯 HDMI workflow 对照修正 E065 方向**: 用户提供关键证据 — `run_hdmi.py` 在 box023 pre-contact body tracking **完美** (E052c_box023_euler_fix/visualization_hdmi.mp4: t=0.6/0.8s sim 双脚平地弯腰 carry-attempt vs MJWP 同时段右脚悬空 25-52cm), HDMI 失败模式只是"接触阶段没抬起 box (ref 左手位置发不上力)"; 关键诊断: **同 ref motion 同 G1 robot 同 body-tracking 公式 (逐字 port 自 HDMI)**, 但 MJWP 失败 HDMI 成功 → 元凶必在 **MJWP 的额外 reward + config**. 逐项 diff: (1) **MJWP 加了 `task_obj_pos/rot_rew_scale=1.0` 是 UNBOUNDED L2 penalty (-1.0*err²) 总是 active**, 而 HDMI 用 `rew_obj_pos = exp(-err/0.5)` saturating 且只有 phase-gated contact; (2) MJWP `init_pos_actuator_gain=500` vs HDMI 20 (25× 强), `guidance_decay_ratio=1.0` vs 0.85; (3) MJWP `knot_dt=0.10` vs HDMI 0.20 (2× 密 control DoF); 其他 (sigma/body partition/W_TRACK) 完全相同. **修正后 E065 plan**: A = `task_obj_pos/rot_rew_scale=0.0` (yaml 单行, 移除 unbounded L2); B 备选 = port actuator gain 20/0.3 + decay 0.85; C 验证 = 直接跑 run_hdmi.py 复现 ground-truth A/B. 教训 #11 修正: "3 次失败 mode 不同" 的 sharper 版本是 "reward 中有 unbounded / 总是 active 项扰动 CEM" (task_obj L2 就是这种), 不只是 "缺约束维度" | 🎯 详见 log 82 §10, **建议优先 E065-A** |
| Diagnosis | 2026-05-14 | Phase 18 | **🚨 Pre-contact body tracking failure 诊断 — box023 在 t=0.7s sim 单脚悬空 60cm, box025 work 是因 ref 不弯腰 (用户假设 100% verified)**: 用户视觉观察"box023 弯腰时左腿抬起就摔"挑战 E062-E064 的 reward 调参方向. 提取 trace_ref body indices (pelvis/L_foot/R_foot) 在 t=0-2s 全帧对比: **box023**: ref 全程双脚 z=0.00-0.05m + 深蹲 pz=0.51m, sim 全程 t=0.2-2.0s 都是 **一脚 z=0 + 另一脚 z=0.15-0.62m** 单脚撑模式 (t=0.7s ref Rf=0.01 vs sim Rf=**0.62m**, 差 61cm), t=2.0s "摔倒"是 1.5s 单脚撑的物理延续 — 不是接触失败; **box025**: ref 全程 pz=0.66-0.73 (几乎不蹲), sim 跟得上 — work 是 task 简单, 不证明 reward 完整. **元凶**: E041c `task_body_rew_scale=0.0` (E036 关闭), 只剩 local_frame_pos_sigma=0.5 (pelvis 系内 body) + joint_sigma=0.25, **世界系 foot 位置完全不被 reward 约束**. 三次失败模式 (fall/superman/prone) 都是 sim 在 pelvis-高度 + box-接触维度间换姿态, foot 始终自由 → 教训 #11: 3 次调参都失败但 failure mode 不同 = reward 缺一个维度, 找未约束的物理量加进去而非调 weight. **下一步建议 E065** = 重启 `task_body_rew_scale=5.0` 只给 left/right_ankle_roll_link 紧 sigma, **取代之前 X5 / Tier 3 提议** | 🚨 详见 log 82, 等用户决策 |
| E064 | 2026-05-14 | Phase 18 | **❌ Tier 2 + threshold raise on box023 (root_σ=0.3 + contact_gain=3.0 + thresh=0.65) — FAIL, 第三个失败模式 "lie-down-and-stay"**: 按 log 80 §8 推荐 extends E063 加 3 个 override; 并行 box023+box025 (33min); **box023**: pelvis_min_intent 0.192→0.178 (基本无改善, C1 仍差 32cm), pelvis_mean_intent +4cm, stable% intent 67→79% 接近 C3, **但 stable% full 79→49% 暴跌 -30pp** — 视频证实 sim t=2.66s 完全平躺, t=3.33s 仍 push-up 起始姿, t=4.0s **深 kneel 单膝跪从未起身**; final box err 1.96cm 任务"完成"但姿态完全错; **box025 regression**: pelvis_min 0.685→**0.643m** (R1 ≥ 0.65 边际 FAIL by 7mm) — reward 加严已开始破坏 box025; **3 实验 3 失败模式**: E062 fall (pz=0.058) / E063 superman (pz=0.192) / E064 prone (pz=0.178 + post-intent 永不起身); **教训 #10**: 同 reward 框架内 weight 调参 3 次都让 CEM 找到 NEW local optimum — 当前 reward 缺 (a) torso upright (b) foot support (c) end-state match 三个维度, **3-strike rule 触发, stop 单 case 调参**; **决策**: 进 E065 = X5 推 X1+T1 (E063 配置, 不带 T2) 到 4 cases (bucket005_s2 + box021 + bucket007 + desk021) 验证 reward 泛化; Tier 3 (walking phase + COM-in-support, 需改 reward 代码) 排队 P1 | ❌ 2/5 box023 + 2/3 box025 (FAIL, 进 E065 X5) |
| E063 | 2026-05-14 | Phase 18 | **⚠️ Tier 1 reward fix on box023 (re-enable stability_penalty=1.0 + reduce task_obj=0.5) — MIXED**: 按 log 79 §5 推荐, 单 yaml 继承 E062 加 3 个 override; 并行 box023+box025 (33min); **box023 数值方向性改善** pelvis_min **0.058→0.192m** (+13.4cm), pelvis_mean_intent 0.486→0.540m, stable% 65.5→67.2%, final_obj 1.36→0.81cm; 但严格 C1≥0.50 仍 FAIL (0.192 << 0.50); **新失败模式 (教训 #9)**: stability_penalty(threshold=0.55m)只惩罚 pelvis 高度不约束 torso 朝向, sim 没有 deep fall 反而找到"水平 superman lunge"局部最优 — t=2.0-2.7s sim 身体几乎平躺但 pelvis 在 0.19-0.46m, 像扑向 box 而非搬运; 终态 box 偏差 0.8cm (任务"完成"但姿态完全不对); **box025 regression PASS** (R1/R2/R3 全过, pelvis 0.685m / stable 100% / obj 10.7cm, reward 改动不破坏 baseline); **决策**: 按 log 79 decision tree C1 FAIL → 进 Tier 2 (E064: `local_frame_root_sigma 0.5→0.3` + `contact_hdmi_gain 5.0→3.0` + **新加 `stability_penalty_threshold 0.55→0.65`** 阻止 superman lunge) | ⚠️ 2/5 box023 + 3/3 box025 (FAIL 整体, 进 Tier 2) |
| Diagnosis | 2026-05-14 | Phase 18 | **🔬 E062 box023 深度诊断 + 优化候选 — 真相: sim 完成了 task (final object pos 仅 9.8cm 偏离 ref), 但 carry→place 转换 (t=2.0-2.7s) 摔倒**: 用户决定不看 box025 (E062 数值不退化, 视觉差异是 CEM 噪声), 专注 box023; 提取 ref vs sim 物体 xyz + pelvis_z 全帧对比发现 — ref motion 在 t=1.67-2.50s 边走边放箱 (+0.59m forward in 0.83s), sim 急追 task_obj_rew 把上身/手扯向前, 重心超出脚底支撑 → t=2.0s 右脚踩在 box 边角 → 0.3s 内栽倒 pelvis 0.42→0.13→0.06m → 2.7-4.5s push-up 起身; **sim 视频日志: "Final object tracking error: pos=0.0981, quat=0.5035"** — sim 完成了 1.6m 对角线搬运 task, 终态 box 跟 ref 仅 9.8cm 偏差; **元凶**: (a) `stability_penalty_scale=0.0` (E034 引入, E036 关闭, E041c 一直没重启), 摔到 0.06m 没任何额外 penalty; (b) `task_obj_pos/rot_rew_scale=1.0` 强力前牵; (c) `local_frame_root_sigma=0.5` 偏松; **优化 Tier 1 (推荐 E063)**: re-enable stability_penalty (1.0) + 减小 task_obj 权重 (0.5/0.5), 单 yaml 继承 E062, ~30min 跑; Tier 2 调 root_sigma + contact_gain; **教训 #8**: 诊断 reward 必须看 ref vs sim 物体轨迹 + pelvis_z 全帧对比, 不能只看 pelvis_min — pelvis_min 0.058 看起来 catastrophic 但实际是 mid-fall+recovery, sim 在做 task | 📋 详见 log 79, 推荐 E063 = T1-A + T1-B |

## 全局结论 (E001-E054, 51+ 实验)

### 视觉复查后的诚实评估

**50+ 实验, 跨 5 种方法, 没有任何实验产生视觉可接受的 loco-manipulation 结果:**

| 方法 | Body Tracking | 物体搬运 | 视觉质量 | 代表实验 |
|------|-------------|---------|---------|---------|
| E041c 单机器人 MJWP | 站着(desk)或摔倒(box) | ❌ 没搬 | ❌ | E048 |
| HDMI 单机器人 | ✅ 站着走 | ❌ 物体推歪 | 勉强(body OK) | E048a |
| 双机器人 CEM | ❌ 穿模变形 | ❌ connect假搬运 | ❌ | E016-E018 |
| 双机器人泛化 | ❌ 全部崩溃 | ❌ | ❌ | E031 |
| 阻尼弹簧/xfrc | — | ❌ 物体翻转 | ❌ | E028-E029 |

### 数值指标 vs 视觉真实的系统性偏差

| 指标 | 数值看起来 | 视觉实际 | 原因 |
|------|----------|---------|------|
| ObjPos=14cm (E041c box023) | "物体追踪好" | 机器人摔倒, 物体没动 | 物体静止=低误差 |
| Contact=93% (HDMI box023) | "手触碰物体" | 物体方向错 178° | euler 错配下的假接触 |
| obj_z=0.597m (E017d dual) | "箱子被抬起 112%" | connect 约束悬浮 | 非物理接触力 |
| Stability=100% (E041c desk) | "机器人稳定搬运" | 丢下桌子自己走了 | 只看 pelvis_z |

### 唯一有效的能力

**Body tracking (关节角度追踪)**: HDMI 7.3°, E035/E041c 15-20°
- 机器人能复现人体的站立、弯腰、行走等全身动作
- 但无法通过物理接触搬运物体

### 根本瓶颈

1. **CEM sampling-based MPC 无法产生 sustained contact**: 1024 samples × 32 iterations 的搜索空间不够
2. **Contact guidance decay**: 最后 iteration PD→0 后, CEM 没有维持接触的策略
3. **Connect 约束是 hack**: 绕过接触发现问题但引入穿模/变形
4. **CORE4D 数据是双人协作**: 单机器人物理上无法完成原始任务的搬运部分

## 关键指标演进

```
pelvis_err: E002(0.100) → E003(0.068) → E004(0.061) → E009a(0.115) → E012(0.083) → E015-d(0.129, bucket005)
E020 多Case pelvis_err: bucket005(0.157) < bucket010(0.660) ≈ desk005(0.647) ≈ box025(0.660) < chair022(0.787)
E020 多Case lift%: bucket005(60%) > bucket010(36%) > desk005(37%+obj) > chair022(130%碰撞推飞) > box025(1.5%)
E025 hand approach: bucket010 contact=55%(6/11), lift_max=0.095m; desk005 contact=40%(4/10), lift_max=0.057m
E026 sustained: bucket010 contact=64%(7/11), lift_max=0.093m, 4 consecutive >5cm
E027 generalization: bucket010 pf=50% still 64% contact; desk005 contact=80%, lift_max=0.117m (best overall)
E021 Anchored pelvis_err: box025(0.186, ↓72%) < desk005(0.279, ↓58%) < bucket010(0.293, ↓58%) < chair022(0.417, ↓47%)
joint_err:  E002(0.073) → E003(0.064) → E012(0.108 rad) → E015-d(0.202 rad, bucket005)
MPKPE (Phase 10+): E036=1.4cm → E039b=1.1-2.1cm → E040=1.2-1.9cm (全部<3cm, 优秀)
Contact<10cm演进:
  box025:  E036(56%) → E039b(85%,手粘连) → E040(64%,手背接触) | 均有不自然行为
  bucket010: E036(2%) → E039b(76%,手粘连) → E040(66%,手背接触) | 均有不自然行为
  desk005: E036(7%) → E039b(81%,stable=20%) → E040(4%,stable=81%)
  根因: position-only reward无方向约束 → CEM用手背满足距离 → 需orientation reward
Phase 12 探索(均未超越E041c baseline):
  E045a(σ=0.3): box025 38%↓ / bucket010 20%↓ / desk005 6%≈  ← sigma收紧有害
  E045b(σ=.15): box025 46%↓ / bucket010 23%↓ / desk005 22%↑  ← desk005局部改善
  E044a(w=2.0): box025 67%≈ / stability 73%↓ ← 上半身权重与stability tradeoff
  E047a(SBTO):  box025 0% / stability 31% ← SBTO+exp-reward不兼容,需调参
obj z 实测 (npz qpos):
  Phase 1: 所有实验 sim max ≤ 0.460m (E011), 实际未持续离地
  Phase 2 (kinobj): obj follows ref (PD驱动), pelvis stable
  Phase 3: E013-r7 obj_z=0.477(翻转非抬起), E013-ctrl obj_z=0.463, 高方差
  ref: 0.533m (峰值)
```

## Phase 2 结论

**G1 单人无法物理搬起 box025, 但运动学完全可行:**
- E010 证明: 当物体被驱动时, G1 身体保持稳定 (pelvis≥0.73m)
- E011 证明: Mocap partner 提供额外物理支撑 (obj_z=0.46)
- E012 证明: 可以导出高质量 hybrid 轨迹到 Holosoma RL

**最终产出**: `workspace/core4d/results/box025_person1_spider_holosoma_w_partner.npz`
- 机器人: SPIDER 物理合规 (pelvis_err=0.083m, 无脚滑)
- 物体: 运动学参考 (正确搬运曲线)
- Partner: 双手世界坐标 (供 RL interaction reward)

## Phase 1 结论 (E001-E009)

**G1 单人无法物理搬起 box025。根因不是力（gravcomp 也只离地 8mm），而是接触几何错位。**
- CORE4D box025 是双人协作任务，两人从 ±x 端对夹
- G1 从 -y 侧接近，接触方向完全不匹配
- 臂展 0.5m < box 长 0.61m，单人对夹不可解

## Phase 2 路线图

```
E010: Connect 约束 → "假设抓住了, G1 能搬吗?" (运动学验证)
  ├─ 成功 → E011: Mocap Partner → "有伙伴物理力, 能协作搬吗?"
  │         └─ 成功 → E012: 导出 + 泛化 (bucket/chair)
  └─ 失败 → 切换到 bucket005 (更小物体, 单人可行)
```

## Phase 3 路线图

```
E013: Intra-rollout Mocap Partner → "修复E011架构限制, rollout内更新partner"
  ├─ E013a: mjwp.py step_env 内 wp.copy mocap (最小改动)
  ├─ E013b: 切换到 mjwp_eq (利用已有per-step mocap更新)
  ├─ E013c: 奖励权重扫描
  │
  ├─ 成功 → E015: 导出+泛化
  └─ 失败 → E014: mjwp_eq Weld退火 + Mocap Partner
            ├─ 成功 → E015: 导出+泛化
            └─ 失败 → E015: 双机器人交替优化
```

## 关键教训
- **必须用 qpos 实测 + 视频验证** — reward 字段不可信 (E006/E008), 数值指标系统性误导 (E048-E052 全面复查)
- **几何分析先于参数调优** — 接触方向比力大小重要 (E009)
- **理解数据集语义** — 协作数据需要协作建模 (E009)
- **重定向 ≠ RL, PD 不需对齐** — 两阶段天然分离 (E007)
- **"修复" 可能破坏已适应的系统** — euler convention 修复使结果 5× 恶化 (E051b), CEM 已适应 "错误" 配置 (E052c)
- **碰撞盒大小无全局最优** — 不同物体形状需不同 margin, box025 需小(0.90), bucket010 需大(1.05) (E053)
- **connect 约束制造假指标** — 所有双机器人 "成功" 均为穿模变形 (E016-E018 视频复查)

## Logs

- E001: `workspace/core4d/log/01_E001_data_pipeline_results.md`
- E002-E004: `workspace/core4d/log/02_E002_E003_results.md`, `03_E004_strong_guidance_results.md`
- E005: `workspace/core4d/log/04_E005_holosoma_export_results.md`
- E006: `workspace/core4d/log/05_E006_forearm_contact_results.md` (更正:箱子未搬起)
- E007: `workspace/core4d/log/06_E007_path_Y_results.md`
- E008: `workspace/core4d/log/07_E008_real_lift_diagnosis_results.md`
- E009: `workspace/core4d/log/08_E009_person2_support_results.md`
- Phase 2 计划: `workspace/core4d/plan/09_phase2_retarget_roadmap.md`
- E010: `workspace/core4d/log/09_E010_kinematic_object_results.md`
- E011: `workspace/core4d/log/10_E011_mocap_partner_results.md`
- E012: `workspace/core4d/log/11_E012_export_results.md`
- Phase 3 计划: `workspace/core4d/plan/11_phase3_retarget_roadmap.md`
- E013: `workspace/core4d/log/12_E013_intra_mocap_results.md`
- E014: `workspace/core4d/log/13_E014_larger_partner_results.md`
- Phase 4 计划: `workspace/core4d/plan/14_phase4_small_object_dual_robot_plan.md`
- E015: `workspace/core4d/log/14_E015_bucket005_results.md`
- E016: `workspace/core4d/log/15_E016_dual_robot_results.md`
- 碰撞盒修复: `workspace/core4d/log/54_collision_box_bug_fix.md`
- E048-E049 评估修正: `workspace/core4d/log/57_E048_E049_eval_correction.md`
- E050 euler分析: `workspace/core4d/log/58_E050_hdmi_euler_analysis.md`
- E051 HDMI scene诊断: `workspace/core4d/log/59_E051_hdmi_scene_diagnosis.md`
- E052 suitcase模板: `workspace/core4d/log/60_E052_suitcase_template_results.md`
- E048-E052 视觉复查: `workspace/core4d/log/61_E048_E052_visual_reevaluation.md`
- E053 碰撞盒margin: `workspace/core4d/log/62_E053_collision_margin_sweep.md`
- E001-E053 阶段总结: `workspace/core4d/log/63_E001_E053_stage_summary.md`
- Phase 17 路线图: `workspace/core4d/plan/63_phase17_post_E053_roadmap_plan.md`
- E054 case分级 计划: `workspace/core4d/plan/64_E054_case_tier_analysis_plan.md`
- E054 case分级 结果: `workspace/core4d/log/64_E054_case_tier_analysis_results.md`
- E055 hand-snap 计划: `workspace/core4d/plan/65_E055_box023_hand_snap_warmstart_plan.md`
- E055 hand-snap 结果: `workspace/core4d/log/65_E055_box023_hand_snap_results.md`
- E056 多 case 诊断 计划: `workspace/core4d/plan/66_E056_multi_case_hand_face_diagnosis_plan.md`
- E056 多 case 诊断 结果: `workspace/core4d/log/66_E056_multi_case_diagnosis_results.md`
- E057 bucket005_s2 snap 计划: `workspace/core4d/plan/67_E057_bucket005_s2_hand_snap_plan.md`
- E057 bucket005_s2 snap 结果: `workspace/core4d/log/67_E057_bucket005_s2_hand_snap_results.md`
- E058 Path B-CEM 计划: `workspace/core4d/plan/68_E058_bucket005_s2_warmstart_cem_plan.md`
- E058 Path B-CEM 结果: `workspace/core4d/log/68_E058_bucket005_s2_warmstart_cem_results.md`
- E059 Path B-CEM box023 计划: `workspace/core4d/plan/69_E059_box023_warmstart_cem_plan.md`
- E059 Path B-CEM box023 结果: `workspace/core4d/log/69_E059_box023_warmstart_cem_results.md`
- Pre-E060 audit (E041c 数据 + reward task-specific): `workspace/core4d/log/70_pre_E060_audit_E041c_data_and_reward.md`
- E060 整体 plan (data fix → baseline → reward ablation): `workspace/core4d/plan/70_E060_data_layer_fix_plan.md`
- E060.0 baseline (数据层修复后) 结果: `workspace/core4d/log/71_E060_0_baseline_results.md`
- E060.1 ori_weight=0 ablation 结果: `workspace/core4d/log/72_E060_1_no_ori_results.md`
- E060.2 case-correct palm_normal 结果: `workspace/core4d/log/73_E060_2_case_correct_palm_normal_results.md`
- 🚨 Box025 3-box regression 发现 + E060 phase invalidation: `workspace/core4d/log/74_box025_3box_regression_and_E060_invalidation.md`
- ✅ E061 Sphere baseline verification (D 方案 PASS, 3-box port 决定性确认为 SOLE regression source): `workspace/core4d/log/75_E061_sphere_baseline_verify.md`
- 🚨 Strategic: E041c 完全是 box025 sphere 过拟合, A/B 框架失效, 转 X1+X2 reward 泛化方向: `workspace/core4d/log/76_E041c_box025_sphere_overfit_strategic.md`
- 🔄 Correction: E041c box025 真问题是臂展物理硬限制不是反关节, 修正"3-box 解决反关节"错误论证, X1+X2 改为基于 sphere: `workspace/core4d/log/77_E041c_box025_real_issue_arm_reach.md`
- ⚠️ E062 X1 auto palm_normal on sphere — mixed result (box025 PASS self-consistency, box023 novel carry-fall-recover behavior, pelvis_min 0.058m strict FAIL but mean/stable全面改善): `workspace/core4d/log/78_E062_auto_palm_normal_sphere_results.md`
- 🔬 E062 box023 深度诊断 + 优化候选 (sim 完成 task 但 carry→place 摔倒, 元凶 task_obj_rew 强牵 + stability_penalty=0, 推荐 E063 T1 = re-enable stability + 减 task_obj): `workspace/core4d/log/79_E062_box023_diagnosis_optimization_candidates.md`
- ⚠️ E063 Tier 1 验证 (stability_penalty=1.0 + task_obj=0.5, box023 pelvis_min +13cm 但 strict C1 FAIL, 新失败模式 = superman lunge, box025 regression PASS, 进 Tier 2): `workspace/core4d/log/80_E063_tier1_stability_taskobj.md`
- ❌ E064 Tier 2 + threshold raise (root_σ=0.3 + gain=3.0 + thresh=0.65) — 第三个失败模式 prone, post-intent 永不起身, box025 R1 边际 FAIL, **3-strike rule 触发**, 教训 #10 = 同 reward 框架内 weight 调参 dead-end: `workspace/core4d/log/81_E064_tier2_threshold_raise.md`
- 🚨 Pre-contact body tracking diagnosis (用户假设 verified) + HDMI vs MJWP reward stack diff (用户 HDMI 对照修正方向, 元凶定位为 task_obj_rew = -L2 unbounded penalty): `workspace/core4d/log/82_pre_contact_body_tracking_diagnosis.md`
- 📋 E065 plan: task_obj_rew form ablation (A=drop / D=HDMI exp form), 代码改动 backward-compat, 4 yamls + 期望矩阵 + decision tree, 待跑 ~66min: `workspace/core4d/log/83_E065_plan_task_obj_ablation.md`
- ❌ E065 results: task_obj form ablation FAIL but reveals new culprit — task_obj 关全 (A) sim 仍 lunge B1=0.30m 改善 38% 但未达阈值; HDMI exp (D) 反恶化 B1=0.69m; **唯一未对齐项是 actuator stiffness** (MJWP kp=500 vs HDMI kp=20), 教训 #12 = 调 reward 之前先对齐 dynamics, → R2 E066 port HDMI actuator: `workspace/core4d/log/84_E065_results_task_obj_ablation.md`
- ❌ E066 results: port HDMI actuator gains FAIL — soft actuator 不搬箱 (C4 jumped 3.10→112.01cm), B1 仍 lunge (0.53/0.75), 揭示第二 mismatch: body partition 稀释 (MJWP 12 lower bodies vs HDMI 6, mean() 把脚 outlier 稀释 2×). 教训 #13 (reward.mean 下 body 数加多反向稀释) + #14 (soft actuator 是 trade-off). → R3 E067 = port HDMI body partition + 双变体 N (narrow only) vs NS (narrow+soft+exp 完整 HDMI clone): `workspace/core4d/log/85_E066_results_actuator_port.md`
- ❌❌ E067 results: port HDMI body partition (12→6) CATASTROPHIC FAIL — sim 做 handstand B1=1.30m, C3=6.9%, hypothesis 完全反了 (more bodies HELP, narrow 让 CEM 找更糟极端 pose). 3 round 全 FAIL, 触发暂停 ✋#3+#5. 诊断指向更深层 (PPO prior vs CEM, knot_dt, noise schedule). 教训 #15 + #16. **PAUSE 等用户决定 R4 方向**: `workspace/core4d/log/86_E067_results_synthesis_pause.md`
- 🚨 R4-direct HDMI vs MJWP diagnosis: HDMI box023 完美 work (B1=0.066m, pelvis 稳, 物体真搬), HDMI ctrl = ctrl_ref + 13% residual; MJWP noise 实际 0.04 比 HDMI residual 0.13 还小, **真元凶是 init pose bug** — sim t=0 pelvis 偏 ref 22° yaw, HDMI 仅 0.6°. 候选: actuator kp 500 反作用 / mjwarp put_data 转换 / 保存 timing. 教训 #17. **PAUSE 等 R5 决策**: `workspace/core4d/log/87_R4_HDMI_diagnosis_init_pose_bug.md`
- 🔬 E068 MJWP init drift diagnosis: 修正 log 87 根因判断 — CPU `mj_forward` 与 ref 完全对齐, init `mj_step` 只偏 0.22°, `mjwarp.put_data` 不放大; 真实 E062/E063/E067 在 first committed step 才快速漂移 (t=0.017/0.033s yaw err=12/22°), 且首帧 robot ctrl 偏 ref 1.56rad, object ctrl 仅0.01. 结论: 元凶是 first MPC tick CEM 立即覆盖 ref ctrl, 下一步 E069 验证 `warmup_steps` / ref-control warmup: `workspace/core4d/log/88_E068_mjwp_init_drift_results.md`
- ❌ E069 first-tick ref-control warmup: W02/W05 均完整运行并保存视频/npz; 修复 `run_mjwp.py` 保存聚合 bug (`improvement` shape 不一致时跳过). 评估修正后 warmup ctrl diff=0, 但 yaw err 仍 12.40/22.16°, B1=0.222/0.428m, 视频 t=0.2s 已转身/单脚. 结论: first CEM override 被推翻, 下一步 E070 应做 MJWarp `step_env(ctrl_ref)` vs MuJoCo `mj_step(ctrl_ref)` parity trace: `workspace/core4d/log/89_E069_first_tick_warmup_results.md`
- 📋 E070 plan: MJWarp ref-control commit parity 诊断, 固定同一 ref 初态和 `ctrl_ref[0:12]`, 逐 substep 对比 MuJoCo CPU 与 MJWarp 的 qpos/qvel/contact/actuator force, 不再继续 warmup/trust-region: `workspace/core4d/plan/75_E070_mjwarp_ref_control_parity_plan.md`
- ✅ E070 results: parity 诊断反转根因 — CPU 和 MJWarp 在同一 ctrl 下完全一致; 当前 run_mjwp 的 `qpos_ctrl` (`qpos_ref[:, :nu]`) 精确复现 E069 early yaw drift 12.403/22.156°，而正确 `orig_ctrl` (原始 29-dim robot ctrl + scene_act object ctrl) 只有 0.574/1.075°. 结论: 根因是 scene_act ctrl_ref preprocessing 错误, 不是 MJWarp physics / object gains / CEM. E071 应修 `run_mjwp.py` ctrl mapping: `workspace/core4d/log/90_E070_mjwarp_ref_control_parity_results.md`
- ⚠️ E071 results: 修复 `run_mjwp.py` scene_act ctrl mapping 后 box023 early drift 消失; yaw err t=0.017/0.033 从 E069 的 12.40/22.16° 降到 0.574/1.075°，与 E070 `orig_ctrl` parity 相差 <0.001°; B1 pre-contact max foot z 从 0.222m 降到 0.069m。但用户复查指出 2s 后没拿住箱子并摔倒，补评估确认 post-2s obj_err max/mean=0.308/0.133m，first obj_err>25cm at 2.00s，first pelvis_z<45cm at 3.32s。结论: qpos-as-ctrl 是 early drift 主因，但 E071 整体 FAIL；下一步 E072 聚焦 post-2s hold/place failure 诊断: `workspace/core4d/log/91_E071_scene_act_ctrl_mapping_fix_results.md`
- ✅ E072 results: replay E071 qpos + scene snapshot 定位 post-2s failure order; frame100/eval2.00s obj_err=30.8cm 且 sim hand-object contact=0(ref=1), ref 直到 frame165/eval3.30s 才正常离手; pelvis 到 frame166/eval3.32s 才低于45cm. post2 contact frames sim 44.4% vs ref 80.2%, object ctrl diff max仅0.01. 结论: hold/contact 先失效, 摔倒是二阶后果; E073 应优先做 hold/contact consistency + robot ctrl trust-region guard: `workspace/core4d/log/92_E072_post2_hold_place_diagnosis_results.md`
- ⚠️ E073 results: dynamic contact target 改为 ref `wrist+eef_offset` 后，口径与 reward 端一致；early drift 保持修复(yaw 0.574/1.075°, B1=0.080m)，first zero contact frame100→108，post2 contact 44.4→49.4%，post2 obj_err max 30.8→29.3cm，pelvis 不再低于45cm。但 first obj_err>25cm 仍 frame100，subagent 视觉复核显示 f130 后脱手/f145 箱落地，后段是“不倒但没拿住”。结论: target 口径修正部分有效，应作为 E074 base；下一步加 robot ctrl trust-region guard: `workspace/core4d/log/93_E073_contact_target_offset_consistency_results.md`
- 🔬 E076 contact source audit: 修正 E075 后续讨论中过强的“右手不应继续强接触”推断；CORE4D raw 没有人工 hand-contact 真值，官方 contact 也是 2cm/3cm 几何 proxy。`box023_person1` 源于 raw `20231008/045`，SPIDER ref = raw[42:178]；E075 f115-f130 是 50Hz eval 的 2.30-2.60s，对应 raw 111-120。raw 几何显示 person1 右手为边界接触(2cm:5/16, 3cm:16/16)，person2 双手强接触(2cm/3cm:16/16)。结论: 下一步应做 contact source alignment + per-hand mask 修复，而不是继续手写 hold/release window: `workspace/core4d/log/97_E076_contact_source_audit.md`
- ✅ E077 results: 生成 CORE4D 3cm geometry-proxy contact mask，输出 raw `(178,2,2)` / SPIDER `(136,2,2)` / eval `(227,2,2)`；f115-f130 对应 raw111-120 下 p1R=16/16但均距2.12cm(边界接触)，p2双手强接触。构造 `box023_person2` 单人 SPIDER case 成功(qpos136×43, scene/scene_act load OK, scene_act euler=XZY)。重要 caveat: converted 层 p1/p2 object pose 完全一致，但 retarget qpos 因 person-specific smpl_scale 不同而最大差6.2cm，不能直接合成双机器人同场景，需 common-scale alignment: `workspace/core4d/log/98_E077_3cm_contact_mask_and_person2_results.md`
- ⚠️/✅ E078 results: 3cm per-EEF contact mask 接入成功，但 E078A/person1 未解决 f119-f125 右腿相位偏差，right foot step sim/ref=0.531/0.159m、right hip pitch ctrl diff abs max=0.554rad，contact 还低于 E075B，说明 p1 更像数据/retarget 质量问题；E078B/person2 完整成功且动作质量明显更接近可用，right foot step sim/ref=0.0219/0.0255m、ctrl diff abs max=0.138rad，支持“person2 ref/contact 质量更好”的判断。下一步优先沿 person2 做可用性验证，p1 暂停 reward 调参: `workspace/core4d/log/99_E078_3cm_per_eef_mask_results.md`
- ⚠️ E079 results: E077 pipeline 泛化验证完成，`11/12` 个 CORE4D single-person case 可运行；用户指出 fixed `box023` post2 window 不能横评后，已改用 `eval_contact_mask_3cm` 自动提取 case-specific contact/intent window；用户进一步纠正 `box023_p2` 才是 E078 positive guard，`box023_p1` 是 main/失败反例。重算后 main case-window 数值成功 `6/10`，fixed-window 旧口径 `2/10`；`box023_p2` guard 复现成功；用户复查视觉质量后修正：`box021_p1` 视觉好但 ref 接触位置异常，`desk021_p1` 前段没抬起，`bucket005_s2_p1` 物体持续受力旋转，`bucket007_p1` 是前摇/trim + ref 支撑问题。结论: 数据 pipeline work，CEM 有跨 case 正信号但判据/算法仍不足，下一步做更强语义指标与 trim/ref feasibility/stability audit: `workspace/core4d/log/100_E079_core4d_generalization_10plus_results.md`
- 📋 E080 plan: `box025_person1/person2` 大物体 Tier3/drop 边界负控复查；已生成显式 trim `38/124` 的 3cm mask 与 E080 override，计划本地 p1 + 远程 GPU1 p2，不预设成功，重点校准 case-window 三阈值是否会误判大物体结构性失败: `workspace/core4d/plan/85_E080_box025_boundary_control_plan.md`
- ⚠️ E080 results: box025 p1/p2 均完整运行并复核；case-window 数值 `2/2=True`，fixed post2 `0/2`。二次复核后修正：p1 是明确 false positive，腿/箱 adjusted SDF min `-13.7cm`、穿入帧 `40.7%`；p2 视觉上接近搬/扶箱，应视为 partial positive / near-usable，但仍有右腿/脚局部干涉（min `-4.6cm`, 穿入帧 `20.2%`）和箱体高度低于 ref。scene 无腿/脚-箱 contact pair，因此腿不会物理支撑箱子。下一步需加入 leg-box interference 与 object lift/floor-contact 指标: `workspace/core4d/log/101_E080_box025_boundary_control_results.md`
- ⚠️ E081 results: 派生 leg/foot-object collision scene 跑通。box025_p2 加碰撞后腿/箱穿入显著降低（case-window `28.9%→7.5%`，min `-4.6cm→-1.2cm`），物体误差略好，但箱体 lift/floor-contact 没改善，仍是 partial positive；box023_p2 guard 基本保持，仅有少量腿/箱接触。结果说明：腿/脚-箱碰撞应纳入物理合理性，但 box025 的下一瓶颈是 lift/floor-contact，而不是继续修穿模: `workspace/core4d/log/102_E081_leg_object_collision_results.md`
- ✅ E089 results: A 路 box021_person1 SPIDER full CEM **首次** head/upper/hand-floor 三项全 0%、pelvis 0.687m，B 路 D003 13 case post-IK 修复 9/13 通过语义 gate，B4 top-2 smoke head ≤2%、upper ≤2%；G1-Feasibility gate 验证为有效数据筛选/修复信号；下一步修 gate world-up 识别 + B-path top-2 跑 full CEM + 集成到 holosoma D005b: `workspace/core4d/log/111_E089_g1_feasibility_AB_results.md`
- ❌/🔬 E090 results: H2-first ablation 显示 no-fingertip 只部分修 inside，topface-preIK 修 Box021 几何安全但 Box025 guard inside `48/52%`，必须条件化；SPIDER smoke S1 safety pass、S2 手撑地 fail；S1 full safety 仍 0% 但 pelvis collapse `0.134m`，因此不扩展 13 case，下一步做 S1-only pelvis/upright 姿态约束: `workspace/core4d/log/112_E090_h2_first_retarget_and_spider_smoke_results.md`
- ✅/🔬 E091 results: Holosoma `data_construction_v2` medium-box 链路跑通到 top-bank + minimal smoke；当前 seed `e091_box004_20231003_2_083_p2` D005b PASS + high visual PASS，smoke collision/object pass 但 pelvis collapse `0.079m`，记录为 dynamics follow-up，不回滚数据筛选: `workspace/core4d/log/113_E091_data_construction_v2_medium_box_results.md`
- ⚠️/🔬 E092 results: smoke-only stop 已作废，Stage A full CEM 补跑后 C1 `box004_083_p2` WORK，C2/C3 Box026 FAIL；direct Omni smoke 全 FAIL。Box026 质量/摩擦已对齐 5kg，失败主因转向 support/inside/reach/posture feasibility: `workspace/core4d/log/114_E092_three_case_spider_dynamic_and_omniretarget_rl_results.md`
- ✅/🔬 E093 results: contact target geometry audit 量化 `wrist_yaw_link+5cm`、raw contact、sphere、3-box、handbox 的 object-local 偏差；Box026/Box025 offset 达 `49-64cm`，下一步应先做 target semantic repair: `workspace/core4d/log/115_E093_contact_target_geometry_audit_results.md`
- ⚠️/✅ E094 results: `adaptive_support` target repair 对 box004 guard 基本无扰动并保持 C1 WORK，但 Box026 C2/C3 full CEM 仍因低髋/趴箱/倒地失败；同时修复 CEM 视频 auto camera: `workspace/core4d/log/116_E094_g1_handbox_target_projection_results.md`
- ✅/🔬 E095 results: worklike candidate mining 生成 `32` row candidate bank；第一批 box004 priority Stage2b/D005b 通过 `083_p1` 和 `082_p1`，`082_p2` OmniRetarget infeasible；下一步跑两条新 case 的 SPIDER full CEM，只有 WORK 进入 Holosoma RL: `workspace/core4d/log/117_E095_worklike_data_mining_results.md`
- ✅/🔬 E096 results: box004 first-batch P1/P2 full CEM 均 WORK，P3 `082_p2` fingertip retry 仍 OmniRetarget CVXPY infeasible；P1/P2 可进入 Holosoma RL positive set，P3 不进入 CEM/RL: `workspace/core4d/log/118_E096_box004_three_case_full_cem_results.md`
- ❌ E084 results: Box021 `20231018_029_p2` 三组 main gate 全失败。A safety 修复 hand-floor/pelvis 但仍 upperbody penetration `80.6%`、object-floor `93.8%`；B upright 降低 obj mean 到 `0.417m` 但 contact 仅 `9.3%` 且视觉翻箱；C semantic/lift 仍 object-floor `96.9%`、bottom gap `-11.9cm`。`accepted_groups=[]`，未跑 guard。结论：停止 Box021 CEM reward 小调参，转 E085 seed/可行性审计与 support/kinematic seed 路线: `workspace/core4d/log/106_E084_box021_constraint_groups_results.md`

## 脚本

- 转换: `workspace/core4d/scripts/convert/convert_holosoma_to_spider.sh`
- 重定向(基线): `workspace/core4d/scripts/retarget/retarget_core4d_baseline.sh`
- 重定向(引导): `workspace/core4d/scripts/retarget/retarget_core4d_guidance.sh`
- 重定向(前臂): `workspace/core4d/scripts/retarget/retarget_core4d_forearm.sh`
- 重定向(前臂+act): `workspace/core4d/scripts/retarget/retarget_core4d_forearm_act.sh`
- 重定向(E009): `workspace/core4d/scripts/retarget/retarget_core4d_e009a.sh`
- 导出: `workspace/core4d/scripts/export/export_to_holosoma.sh`
- 评估(E009): `workspace/core4d/scripts/eval/eval_e009_lift.py`
- E054 case分析: `workspace/core4d/scripts/analyze/case_tier_analysis.py`
- E054 视频关键帧提取: `workspace/core4d/scripts/analyze/extract_case_keyframes.sh`
- E055 hand-snap 一键脚本: `workspace/core4d/scripts/run_E055_snap.sh`
- E055 snap 主体: `workspace/core4d/scripts/E055/snap_box023.py` (调用 `spider/preprocess/hand_snap_ik.py`)
- E055 snap 可视化: `workspace/core4d/scripts/E055/visualize_snap.py`
- E055 snap 关键帧: `workspace/core4d/scripts/E055/extract_snap_keyframes.sh`
- E055 单帧坐标系图: `workspace/core4d/scripts/E055/visualize_frames.py` (world / object / pelvis + 6 面命名)
- E055 face dist 时间序列: `workspace/core4d/scripts/E055/face_distance_timeseries.py` (诊断 ref 握姿是否合理)
- E056 多 case 诊断: `workspace/core4d/scripts/E056/multi_case_face_diagnosis.py` (6 case face dist + 5 类 grasp 分类)
- E056 一键脚本: `workspace/core4d/scripts/run_E056_diagnosis.sh` (诊断 + 抽帧验证)
- E057 snap 主体: `workspace/core4d/scripts/E057/snap_bucket005_s2.py` (复用 hand_snap_ik, 改 case 路径)
- E057 snap 可视化: `workspace/core4d/scripts/E057/visualize_snap.py` (ref vs snap 2×2 渲染)
- E057 snap 关键帧: `workspace/core4d/scripts/E057/extract_snap_keyframes.sh`
- E057 face 验证 (C6): `workspace/core4d/scripts/E057/verify_snap_face.py` (snap 后 main face 与 E056 一致性)
- E057 一键脚本: `workspace/core4d/scripts/run_E057_snap.sh` (snap → viz → keyframes → face 验证)
- E058 train (并行/串行): `workspace/core4d/scripts/train/train_E058.sh`
- E058 eval (contact/face/stability): `workspace/core4d/scripts/eval/eval_E058.py`
- E058 关键帧: `workspace/core4d/scripts/eval/extract_E058_keyframes.sh`
- E058 一键脚本: `workspace/core4d/scripts/run_E058.sh` (train → eval → keyframes)
- E059 train (box023, +PYTHONUNBUFFERED 实时进度): `workspace/core4d/scripts/train/train_E059.sh`
- E059 eval: `workspace/core4d/scripts/eval/eval_E059.py`
- E059 关键帧: `workspace/core4d/scripts/eval/extract_E059_keyframes.sh`
- E059 一键脚本: `workspace/core4d/scripts/run_E059.sh`
- E063 train (并行 box023 + box025): `workspace/core4d/scripts/train/train_E063.sh`
- E063 eval (含 ref vs sim obj 全帧 + scene_act XML 加载修正): `workspace/core4d/scripts/eval/eval_E063.py`
- E063 dense 关键帧 (9 frames @ box023 fall window): `workspace/core4d/scripts/eval/extract_E063_keyframes.sh`
- E063 一键脚本: `workspace/core4d/scripts/run_E063.sh`
- E064 train (并行 box023 + box025): `workspace/core4d/scripts/train/train_E064.sh`
- E064 eval (baseline = E063, 同结构): `workspace/core4d/scripts/eval/eval_E064.py`
- E064 dense 关键帧: `workspace/core4d/scripts/eval/extract_E064_keyframes.sh`
- E064 一键脚本: `workspace/core4d/scripts/run_E064.sh`
- E070 ref-control parity 诊断: `workspace/core4d/scripts/debug/diagnose_E070_ref_control_parity.py`
- E070 train/debug 入口: `workspace/core4d/scripts/train/train_E070.sh`
- E071 train: `workspace/core4d/scripts/train/train_E071.sh`
- E071 eval: `workspace/core4d/scripts/eval/eval_E071.py`
- E072 train/analysis入口: `workspace/core4d/scripts/train/train_E072.sh`
- E072 eval replay诊断: `workspace/core4d/scripts/eval/eval_E072.py`
- E073 train: `workspace/core4d/scripts/train/train_E073.sh`
- E073 eval: `workspace/core4d/scripts/eval/eval_E073.py`
- E077 contact/person2 build: `workspace/core4d/scripts/E077/`
- E078 train: `workspace/core4d/scripts/train/train_E078.sh`
- E078 eval: `workspace/core4d/scripts/eval/eval_E078.py`
- E078 remote/pull: `workspace/core4d/scripts/run_E078_remote.sh`, `workspace/core4d/scripts/pull_E078_remote_results.sh`
- E079 preprocess: `workspace/core4d/scripts/run_E079_preprocess.sh`
- E079 train: `workspace/core4d/scripts/train/train_E079.sh`
- E079 eval/contact-quality: `workspace/core4d/scripts/eval/eval_E079.py`, `workspace/core4d/scripts/eval/eval_E079_contact_quality.py`
- E079 remote/pull: `workspace/core4d/scripts/run_E079_remote.sh`, `workspace/core4d/scripts/pull_E079_remote_results.sh`
- E080 preprocess: `workspace/core4d/scripts/run_E080_preprocess.sh`
- E080 train: `workspace/core4d/scripts/train/train_E080.sh`
- E080 eval: `workspace/core4d/scripts/eval/eval_E080.py`
- E080 remote/pull: `workspace/core4d/scripts/run_E080_remote.sh`, `workspace/core4d/scripts/pull_E080_remote_results.sh`
- E081 preprocess: `workspace/core4d/scripts/run_E081_preprocess.sh`
- E081 train: `workspace/core4d/scripts/train/train_E081.sh`
- E081 eval: `workspace/core4d/scripts/eval/eval_E081.py`
- E081 remote/pull: `workspace/core4d/scripts/run_E081_remote.sh`, `workspace/core4d/scripts/pull_E081_remote_results.sh`
- E082 结果: `workspace/core4d/log/103_E082_d003_box021_e081_legobj_results.md`
- E082 上半身穿模诊断: `workspace/core4d/log/104_E082_body_fall_upperbody_collision_diagnosis.md`
- E082 preprocess: `workspace/core4d/scripts/run_E082_preprocess.sh`
- E082 train: `workspace/core4d/scripts/train/train_E082.sh`
- E082 eval: `workspace/core4d/scripts/eval/eval_E082.py`
- E082 body-fall 诊断: `workspace/core4d/scripts/eval/diagnose_E082_body_fall.py`
- E082 remote/pull: `workspace/core4d/scripts/run_E082_remote.sh`, `workspace/core4d/scripts/pull_E082_remote_results.sh`
- E083 结果: `workspace/core4d/log/105_E083_upperbody_object_collision_results.md`
- E084 计划: `workspace/core4d/plan/89_E084_box021_constraint_groups_plan.md`
- E083 preprocess: `workspace/core4d/scripts/run_E083_preprocess.sh`
- E083 train: `workspace/core4d/scripts/train/train_E083.sh`
- E083 eval: `workspace/core4d/scripts/eval/eval_E083.py`
- E083 sheet: `workspace/core4d/scripts/eval/extract_E083_contact_sheets.sh`
- E083 remote/pull: `workspace/core4d/scripts/run_E083_remote.sh`, `workspace/core4d/scripts/pull_E083_remote_results.sh`
- E084 结果: `workspace/core4d/log/106_E084_box021_constraint_groups_results.md`
- E085 计划: `workspace/core4d/plan/90_E085_box021_exit_gate_seed_routes_plan.md`
- E084 preprocess: `workspace/core4d/scripts/run_E084_preprocess.sh`
- E084 train: `workspace/core4d/scripts/train/train_E084.sh`
- E084 eval: `workspace/core4d/scripts/eval/eval_E084.py`
- E084 sheet: `workspace/core4d/scripts/eval/extract_E084_contact_sheets.sh`
- E084 remote/pull: `workspace/core4d/scripts/run_E084_remote.sh`, `workspace/core4d/scripts/pull_E084_remote_results.sh`
