# E099 — Stage 1: 接触语义信息流补全

日期：2026-05-30
分支：`exp/core4d-collab-retarget`
上游：E098（诊断基础设施已完成，face_utils + replay_gate + historical_case_manifest.tsv 可用）
下游：E100（target 重做）、E101（full CEM）、E102（mining）均需 fingertip face vote + quat audit + 3D viz 输出

## Context

v2 报告 §4 / finding 02 §4 B6 指出：wrist 不是 raw 接触点，下游所有 face / contact 决策（E017 anchor / E028 manifest / E029 D6 / E085 target 生成）都默认 wrist 是接触代理，导致：

1. CORE4D 原始有 10 个指尖（SMPL-X joint 27/30/33/36/39 = L 5 指尖，42/45/48/51/54 = R 5 指尖），STAGE A 把指尖丢弃只剩 22 body joints；
2. OmniRetarget 不依赖指尖（约束在 wrist link）；
3. G1 URDF 上 `rubber_hand_link` 是 palm site（FK 后 contact_pos 是 palm，不是指尖）；
4. 所以 spider 仓库里所有 "contact" 都是 palm 代理，跟人手真实接触面差 ~5-10 cm（指节长度）。

E098 已修了 face 投票的几何 bug（B1-B3 全 3D argmax），但「投票输入」依然是 palm，不是真指尖。E099 的目标是**把 raw 5 指尖这条信息流补回 pipeline**，覆盖全 20 historical case，让下游 face 决策有「真实接触代理」可用。

同时做两个相关的诊断：
- **quat 普查**：v2 §3 box021 quat 90° 旋转导致 face 投票方向反，需要扫描全 20 case 的 obj quat 偏角，标出哪些 case 不能走"world-up 投影"路径；
- **raw_contact 3D 可视化**：v1 时代的 `data_construction_v2/visualizations/raw_contact/` 源脚本已不可考，重写一个 3D turntable 版本，覆盖全 20 case，作为后续 stage 视觉签收的标准工具。

## 约束（贯穿）

1. **不动 OmniRetarget 算法本身**：本 stage 只读 raw CORE4D + 处理后 trajectory_kinematic.npz，所有 helper 都是离线分析工具，不进 IK loop（IK loop 的 fingertip 注入留给 E100 / E101 评估期间再决定是否启用）。
2. **覆盖全 20 historical case**（manifest 见 `workspace/core4d/scripts/E098/historical_case_manifest.tsv`）：fingertip vote / quat audit / 3D viz / audit report 都必须对所有 case 跑一遍，不局限于 box021 D003 6 case。box022 2 case + box004 082_p2 因 raw 数据待定，跑 best-effort（缺数据 raw mocap 不可用即记 N/A）。
3. **可视化优先**：3D > 2D，video > stills。每 case 至少 1 个 turntable mp4。
4. **STAGE A 重跑 `--include_fingertip_centers`** 这件事的实际触发延后到 E100（因为只有 E100 的 target 生成器才真正需要 IK 后的指尖位置）；E099 阶段只用 raw CORE4D 指尖（绕过 OmniRetarget），输出 fingertip helper API + 历史 case face audit。这是对原计划的最小化解读，符合"不动 IK 算法"约束。

## Claims

### C1 — Fingertip helper：raw CORE4D → 10 指尖 in obj local frame

**判据**：
- 提供 `fingertip_face_vote.py` API：`vote(case_name) → {hand: {face: vote_frac}}`，输入是 raw CORE4D 数据路径，内部完成 Y-up → Z-up + obj local frame 投影 + face_utils.face_label 投票；
- 对 ≥ 17/20 case 成功输出 vote 结果（缺数据 case 记 N/A，不阻塞其它 case）；
- 单元测试至少 3/3 PASS（box021/box023/box025 已知主面与人工标注一致）。

### C2 — Quat 普查：全 20 case obj 旋转偏角

**判据**：
- 提供 `quat_identity_audit.py`：对所有 case 计算 `obj quat` 全程 (T, 4) 的均值 / max 与 identity quat 的夹角；
- 输出 `results/E099/quat_audit.tsv` 含 `case / quat_mean_deg / quat_max_deg / disable_world_up`（> 30° 标记 True）；
- box021 D003 13 case 全部命中 `disable_world_up=True`（v2 §3 已断言 quat 90° X）；box004 / box023 / box025 应在 ≤ 30° → `False`。

### C3 — Raw_contact 3D 可视化生成器：全 20 case turntable mp4

**判据**：
- 提供 `render_raw_contact_3d.py`：每 case 输出 1 段 36-frame turntable mp4（≥ 5 秒）+ 1 张 4-view PNG；
- 同框叠加：box collision wireframe + raw 10 指尖散点（按手区分色）+ palm site（IK FK，从 trajectory_kinematic.npz 读）+ per-frame face label 文字 + 投票多数面高亮；
- ≥ 17/20 case 成功（缺 raw mocap case 记 N/A）；
- subagent 视觉签收：随机抽 5 case mp4，确认指尖位置贴合 box 几何 + face label 与视觉一致。

### C4 — 历史 case face audit 报告

**判据**：
- 输出 `results/E099/historical_case_face_audit.md`：所有 case 的 fingertip vote 主面 + palm vote 主面（用 E098 anchor refit demo 同算法）+ 差异标记 + v1/v2 旧主面对比；
- 至少 1 个 case 暴露 "palm 主面 ≠ fingertip 主面" 的情形（验证 B6 假设："palm 代理与真指尖接触面会偏差"）；
- 输出对 E100 target 生成器的 actionable 推荐：哪些 case 应用 `fingertip-vote face`，哪些可保留 `palm-vote face`。

## 改动文件

### spider 仓库（本期）

| 类别 | 文件 | 改动 |
|---|---|---|
| 新增 | `workspace/core4d/plan/106_E099_contact_semantics_pipeline_plan.md` | 本计划 |
| 新增 | `workspace/core4d/scripts/E099/fingertip_face_vote.py` | raw 10 指尖 → obj local → face vote helper |
| 新增 | `workspace/core4d/scripts/E099/quat_identity_audit.py` | obj quat 偏角扫描 |
| 新增 | `workspace/core4d/scripts/E099/render_raw_contact_3d.py` | 3D turntable mp4 生成器 |
| 新增 | `workspace/core4d/scripts/E099/test_fingertip_face_vote.py` | 单元测试 |
| 新增 | `workspace/core4d/scripts/E099/run_all_E099.sh` | 一键运行 |
| 新增 | `workspace/core4d/results/E099/quat_audit.tsv` | 全 case quat 偏角 |
| 新增 | `workspace/core4d/results/E099/fingertip_face_stats.tsv` | 全 case fingertip vote 主面 |
| 新增 | `workspace/core4d/results/E099/visuals/raw_contact_3d/` | 全 case turntable mp4 + PNG |
| 新增 | `workspace/core4d/results/E099/historical_case_face_audit.md` | C4 报告 |
| 新增 | `workspace/core4d/log/123_E099_contact_semantics_pipeline_results.md` | 实验日志 |

### holosoma 仓库（本期）

E099 阶段**不动 holosoma 仓库**（STAGE A 重跑 `--include_fingertip_centers` 延后到 E100；本 stage 直接读 raw CORE4D）。

如 E100 启动时确认要让 IK loop 也用指尖，再去 holosoma 加 batch convert + 重跑全 case。

### 不动的文件

- `spider/process_datasets/core4d.py`（B4 deprecation comment 已加，本期无变化）
- `workspace/core4d_collab_retarget/scripts/E0*` 中已修 B1-B3 的脚本
- spider/simulators/mjwp.py（reward / gate 不动）

## 流程

1. **P1**：写本 plan ✅（当前步骤）
2. **P2**：摸清 raw CORE4D → 指尖路径
   - 读 `convert_core4d_to_omniretarget.py:115-127`（已确认 indices: L 27/30/33/36/39, R 42/45/48/51/54）
   - 读 spider 仓库已有的 `convert_core4d_to_hdmi.py` 看 case_name → date/seq/person 映射
3. **P3**：写 fingertip_face_vote helper + 单元测试 + 全 20 case 跑 vote
4. **P4**：写 quat_identity_audit + 跑全 case
5. **P5**：写 render_raw_contact_3d + 全 case 跑 mp4 + subagent 视觉签收 5 case
6. **P6**：写 audit 报告 + log + EXPERIMENT_TRACKER + git commit + push

## 验证命令

```bash
# C1 单元测试
.venv/bin/python workspace/core4d/scripts/E099/test_fingertip_face_vote.py

# C1+C4 全 20 case fingertip vote
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python workspace/core4d/scripts/E099/fingertip_face_vote.py \
  --manifest workspace/core4d/scripts/E098/historical_case_manifest.tsv \
  --out workspace/core4d/results/E099/fingertip_face_stats.tsv

# C2 quat 普查
.venv/bin/python workspace/core4d/scripts/E099/quat_identity_audit.py \
  --manifest workspace/core4d/scripts/E098/historical_case_manifest.tsv \
  --out workspace/core4d/results/E099/quat_audit.tsv

# C3 全 20 case 可视化
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python workspace/core4d/scripts/E099/render_raw_contact_3d.py \
  --manifest workspace/core4d/scripts/E098/historical_case_manifest.tsv \
  --out workspace/core4d/results/E099/visuals/raw_contact_3d
```

## 风险

| 风险 | 应对 |
|---|---|
| raw CORE4D 某 case 缺 person*_poses.npz（box004 082_p2 / box022 2 case 已知 manifest 标 pending）| best-effort；缺 case 在 stats.tsv 记 N/A，不阻塞其它 case；C1/C3 判据已写 ≥ 17/20 |
| SMPL-X 指尖 = "3rd phalanx joint" 不是真指尖（缺末梢 ~1cm），是否够准 | 视觉签收时与点云对比；如偏差 > 2cm 在 audit 报告里加注 |
| Y-up → Z-up 转换在指尖上对不上 | 与 convert_core4d_to_omniretarget.py 同公式 `x'=x, y'=-z, z'=y`；与 trajectory_kinematic.npz obj_pos 重合验证 |
| turntable mp4 数量大（20 case × ≥ 5s） | 36 帧 dpi 90 控制单文件 < 1MB；批量 ~20MB 可接受 |

## 下游影响（明示，避免日后翻车）

| 下游 | 依赖 E099 哪个产出 | 阻塞性 |
|---|---|---|
| E100 build_fingertip_aware_target.py | fingertip_face_vote API + quat_audit.tsv | 硬阻塞 |
| E100 干净 A/B 视频 | render_raw_contact_3d（side-by-side raw vs new target） | 硬阻塞 |
| E101 失败归因 | quat_audit.tsv（判断 case 是否走 world-up）+ 3D 可视化（叠加 G1 rollout） | 硬阻塞 |
| E102 mining score | fingertip vote face 兼容性 + quat audit 排除 | 硬阻塞 |
| E099 audit 报告里的 "fingertip ≠ palm" case list | E100 优先重做 target；E101 优先 case 排序 | 软阻塞 |

## 时间预算

- P1: 0.5 h（已用）
- P2: 0.5 h
- P3: 2 h（写 + 测 + 跑全 case）
- P4: 1 h
- P5: 2 h（含 mp4 渲染时间）
- P6: 1.5 h（写报告 + log + commit）
- **合计：~7-8 h（1 工作日）**
