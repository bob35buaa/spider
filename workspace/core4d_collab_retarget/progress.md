# E001 Progress — 2026-05-17

## 当前状态

E001 已完成并提交推送；E002 freejoint leg-object control audit full CEM 已完成。结论：当前 E081-style 单机器人 reward/control 在真 freejoint object 下失败，且 guard 也失败；下一步应做 E003 physics-feasibility sweep 或进入虚拟 grasp/contact 约束路线。新工作区实验编号从 `E001` 开始；`workspace/core4d` E081 是 baseline，不作为本工作区的 E082。

## 完成步骤

- [x] 读取 `workspace/exp_task.md`，确认任务要求：基于 E081 baseline，先做深入分析和头脑风暴，再按 `experiment-planning-zh` 展开具体实验。
- [x] 读取 `workspace/core4d/EXPERIMENT_TRACKER.md`、最新 E081 plan/log、`workspace/core4d/progress.md`，恢复历史上下文。
- [x] 已将 baseline 分支 `feat/dual-robot-retarget` fast-forward 推送到远端 `main`：`ae8b342`。
- [x] 已创建新分支：`exp/core4d-collab-retarget`。
- [x] 用户明确纠正：新分支和新工作区实验编号从 `E001` 开始；本工作区按该规则执行。
- [x] 已写入新工作区骨架与 E001 plan：
  - `workspace/core4d_collab_retarget/EXPERIMENT_TRACKER.md`
  - `workspace/core4d_collab_retarget/progress.md`
  - `workspace/core4d_collab_retarget/plan/01_E001_baseline_literature_code_audit_plan.md`
- [x] 已抽取三篇论文 PDF 文本，并写入跨论文综合笔记：`workspace/core4d_collab_retarget/paper_notes/01_E001_literature_synthesis.md`。
- [x] 已按 `video-frames` 技能从 E081 两个 baseline MP4 抽取 f100/f125/f160 对应帧到 `workspace/core4d_collab_retarget/results/E001_baseline_frames/`。
- [x] 已整合 object/freejoint explorer 结论：active E079-E081 的 CEM 基本不采样 object controls，当前 baseline 是 actuator-guided object trajectory 下的 robot-control optimization。
- [x] E001 正式结果日志已写入：`workspace/core4d_collab_retarget/log/01_E001_baseline_literature_code_audit_results.md`。
- [x] E001 已提交并推送：`a912d10 exp(core4d_collab_retarget): E001 baseline literature code audit`。
- [x] E002 计划已写入：`workspace/core4d_collab_retarget/plan/02_E002_freejoint_legobj_control_audit_plan.md`。
- [x] E002 派生 task / override / train / eval 脚本已实现。
- [x] E002 预处理已运行，生成：
  - `box025_person2_freejoint_legobj`
  - `box023_person2_freejoint_legobj`
- [x] 复查 E070 qpos-as-ctrl parity bug；已修复 freejoint/scene_act 切换下的 ctrl fallback 隐患，并新增 ref/model 维度断言。
- [x] E002 GPU smoke 已通过；eval 脚本可在 freejoint `scene.xml` / `nq_obj=7` / `nu=29` 下输出 summary。
- [x] E002 full local CEM 已完成：
  - main `E002_box025_p2_freejoint`: case-window obj mean/max `0.703/1.356m`，hand contact `89.6%`，leg intf `0.0%`，floor contact `85.5%`，strict False。
  - guard `E002_box023_p2_freejoint`: case-window obj mean/max `0.830/1.488m`，hand contact `72.0%`，leg intf `0.0%`，floor contact `88.7%`，strict False。
  - 相比 E081，真 freejoint 物体在 main 和 guard 都显著退化，确认当前 pipeline 依赖 object actuator guidance。

## 待完成

- [x] 审计代码中 object/freejoint/scene_act 控制口径，回答“物体轨迹是否是 GT/是否 free joint/全靠机器人力是否可行”。
- [x] 用 deep-reading 口径分析 SPIDER、DynaRetarget、双人交互控制论文，并沉淀到 `paper_notes/`。
- [x] 汇总 E081 baseline eval+可视化验收口径。
- [x] 形成 E002+ 可执行实验列表，选择首个实验写 plan 后再实现。
- [x] 实现 E002 freejoint 派生 task / override / train / eval 脚本。
- [x] 运行 E002 smoke + full CEM，对齐 E081 eval 和可视化。
- [x] 规划 E003：true-freejoint mass/friction/contact feasibility sweep，优先使用远程双卡并行。
- [ ] 实现并运行 E003。

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|------|---------|----------|
| 本地沙箱内 `git ls-remote` DNS 失败 | 1 | 使用批准的 `git fetch origin` 网络权限确认远端状态 |
| 本地创建 `main` 引用时 `.git/refs` 被沙箱视为只读 | 1 | 使用批准的 `git branch -f` 权限创建本地 `main` |
| 初始思路误把新方向计划称为 E082 | 1 | 按用户纠正，新工作区编号从 E001 开始 |
| `video-frames/scripts/frame.sh` 无可执行位，直接运行报“权限不够” | 1 | 改用 `bash frame.sh ...` 成功抽帧 |
| E002 sandbox smoke 中 PyTorch/Warp 看不到 CUDA | 1 | 已用提升权限运行 GPU smoke；CPU run 会在 Warp graph capture 处失败，因为 MJWarp capture 要求 CUDA device |
