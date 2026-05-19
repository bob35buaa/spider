# E001 Progress — 2026-05-17

## 当前状态

E001 已完成并提交推送；E002 freejoint leg-object control audit full CEM 已完成。结论：当前 E081-style 单机器人 reward/control 在真 freejoint object 下失败，且 guard 也失败；E003 physics-feasibility sweep 已开始实施，用于区分物理参数不可行与 reward/optimizer 不足。新工作区实验编号从 `E001` 开始；`workspace/core4d` E081 是 baseline，不作为本工作区的 E082。

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
- [x] 本轮已重新读取 `EXPERIMENT_TRACKER.md`、`progress.md`、E002 log 和 E003 plan，确认工作区仅 `.codex/config.toml` 与 `workspace/exp_task.md` 为既有未提交状态，E003 实现将避开它们。
- [x] 已实现 E003 脚本骨架：
  - `scripts/E003/variants.tsv`
  - `scripts/E003/create_physics_sweep_cases.py`
  - `scripts/E003/generate_e003_overrides.py`
  - `scripts/run_E003_preprocess.sh`
  - `scripts/train/train_E003.sh`
  - `scripts/train/train_E003_remote_tmux.sh`
  - `scripts/run_E003_remote.sh`
  - `scripts/pull_E003_remote_results.sh`
  - `scripts/eval/eval_E003.py`
  - `log/03_E003_freejoint_physics_feasibility_sweep_results.md`
- [x] E003 preprocess 已完成，生成四个 true-freejoint 派生 task 和四个 Hydra override；四个配置均为 `contact_guidance=false`、`scene_name=""`、`nq/nv/nu/nq_obj=43/41/29/7`、`ctrl_ref=29`。
- [x] E003 GPU smoke 已通过：四个 variant 均生成 NPZ，`eval_E003.py` 已生成 `comparison.csv`/summary/timeseries/leg-object 指标。smoke 仅 `T=4`，不用于实验结论。
- [x] E003 setup 已提交并推送：`a12d309 exp(core4d_collab_retarget): E003 physics sweep setup`。
- [x] E003 full 已在远端 tmux session `E003` 启动，GPU0 跑 box025 两个 variant，GPU1 跑 box023 两个 variant；远端既有 dirty state 未清理，`git pull --ff-only` 和 E003 preprocess 已成功。
  - 已完成：`E003_box025_p2_m1.npz`、`E003_box023_p2_m1.npz`、`E003_box025_p2_m1_f4.npz`、`E003_box023_p2_m1_f4.npz`。
  - 远端/本地 eval 均完成，aggregate: `num_results=4`, `num_guard_physics_feasible_proxy=0`。
- [x] E003 full 结论：`box025_m1_f4` 相比 E002 有改善但仍失败；`box023` guard 未恢复并出现腿/箱干涉和摔倒。被动 mass/friction 不是主要瓶颈，下一步转向 explicit virtual collaborator/support/contact constraint。
- [x] 实现并运行 E003。
- [x] 已写入 E004 计划：`workspace/core4d_collab_retarget/plan/04_E004_freejoint_virtual_partner_support_plan.md`，使用现有 `partner_force_*` freejoint 外力路径测试虚拟协作者支持。
- [x] 用户要求把 E004 扩成较大规模实验；已补读 `workspace/core4d/log/22_E024_partner_force_results.md` 与 E028-E030 结果、参考未跟踪的 E004 v2 draft，并重写正式 E004 plan：
  - Wave A：gravity-only control + `kp=10/20/40` translation spring + box023 guard。
  - Wave B：只启用 reward-only hold-contact，不继承会打开 `scene_act/contact_guidance` 的 E078 defaults。
  - Wave C：rotation torque 仅作为 position 成功后的隔离 probe，主线 full variants 全部 `kp_rot=0`。
  - 并行策略：本地 1 卡跑 main anchor/hold-contact，远程 2 卡分别跑 box025 sweep 与 box023 guard。
- [x] E004 setup 脚本已创建并通过静态检查：
  - `scripts/E004/variants.tsv`
  - `scripts/E004/generate_e004_overrides.py`
  - `scripts/run_E004_preprocess.sh`
  - `scripts/train/train_E004.sh`
  - `scripts/train/train_E004_remote_tmux.sh`
  - `scripts/run_E004_remote.sh`
  - `scripts/pull_E004_remote_results.sh`
  - `scripts/eval/eval_E004.py`
  - `log/04_E004_freejoint_virtual_partner_support_results.md`
- [x] E004 smoke 已通过：9 个 Wave A/B variants 均生成 `trajectory_mjwp.npz`，eval 显示全部 `contact_guidance=false`、`nu=29`、`nq_obj=7`、`kp_rot=0`、`E004_freejoint_parity_ok=True`。
- [x] 远程首次启动时发现远端缺少本地 ignored 的 E002 contact mask 结果目录；已修复 `generate_e004_overrides.py`，优先使用本工作区 E002 mask，缺失时回退到 `workspace/core4d/results/E081/contact_masks`。
- [x] E004 full 已完成：9 个 Wave A/B variants 全部保持 true-freejoint parity；main `box025` 无 useful proxy，gravity-only `0.660/1.288m`、spring/hold variants 约 `0.768-0.771/1.549-1.577m`；guard `box023_s10/s20` stable 但不 transport，`box023_s10_hc` 摔倒。远程 GPU1 的 `box023_s20` 与 `box023_s10_hc` 均卡住后已终止并本地补跑完成。
- [x] E005 计划已写入：`workspace/core4d_collab_retarget/plan/05_E005_partner_force_timing_support_site_plan.md`。关键发现：`_apply_partner_force` 使用硬编码 `1/30` ref dt；box025 task_info 本身是 30Hz，所以 E004 main 不被 timing 推翻，box023 guard 使用默认 50Hz 因而需要 E005 显式复查。E005 同时做 COM vs support-site 对照。
- [x] E005 实现与 smoke 已完成：新增 `partner_force_ref_dt`、`partner_force_point_local`、force/torque clamp；support-site 用 object-local point 的等效 wrench `F, r x F`，并保持 `kp_rot=0`。9 个 variants 的 4-step smoke 全部完成，eval aggregate: `num_results=9`, `num_freejoint_parity_ok=9`, `num_guard_stable_proxy=3`。
- [x] **头脑风暴 session (2026-05-17)**：生成 5 个候选假设（H001-H005），覆盖 Physics/Algorithm/Optimizer 类别。
  - H001 (物理参数) ❌ 已被 E003 证伪：1kg + 高摩擦仍然失败
  - H003 (Partner Proxy 3D 力) ✅ 选定：与 sim2real 终端目标对齐，复用 Mode D 代码
  - H002/H004/H005 推迟为后备方案
  - 文件：`ideas/brainstorm_2026-05-17.md`, `ideas/HYPOTHESIS_BACKLOG.md`
  - E004 plan 已按用户反馈重写为大规模 H003 sweep → 下一步实现 E004 脚本并跑 smoke/full

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|------|---------|----------|
| 本地沙箱内 `git ls-remote` DNS 失败 | 1 | 使用批准的 `git fetch origin` 网络权限确认远端状态 |
| 本地创建 `main` 引用时 `.git/refs` 被沙箱视为只读 | 1 | 使用批准的 `git branch -f` 权限创建本地 `main` |
| 初始思路误把新方向计划称为 E082 | 1 | 按用户纠正，新工作区编号从 E001 开始 |
| `video-frames/scripts/frame.sh` 无可执行位，直接运行报“权限不够” | 1 | 改用 `bash frame.sh ...` 成功抽帧 |
| E002 sandbox smoke 中 PyTorch/Warp 看不到 CUDA | 1 | 已用提升权限运行 GPU smoke；CPU run 会在 Warp graph capture 处失败，因为 MJWarp capture 要求 CUDA device |
| E004 remote GPU1 `box023_s20` / `box023_s10_hc` 临近结尾无 GPU 利用率且未写 trajectory | 2 | 终止远端卡住进程，保留 interrupted log，改为本地单 variant 补跑；最终 9 个结果全部完成 |
| E005 前复查发现 partner-force ref index 硬编码 30Hz | 1 | 将 ref dt 改为 `config.ref_dt`/`partner_force_ref_dt`；box025 显式 30Hz，box023 显式 50Hz，E004 main 结论保留但 guard timing 需复查 |
| E005 远程 `box023_p2_xneg_s10` 卡住 | 1 | 进程停在 `sim_steps=34/272`，日志 21:32 后不更新且 GPU 利用率 0%；已终止该远程进程，`xpos_s10` 未启动。本轮只分析 7 个实际 full 结果，排除 4-step smoke NPZ |

## 2026-05-17 22:20 E005 远程回收与分析

- 已读取 E005 plan/log/tracker，并检查远程 tmux `E005` 状态。
- 远程完成并拉回：
  - `E005_box025_p2_com_s40`
  - `E005_box025_p2_yneg_s40`
  - `E005_box025_p2_ypos_s20`
  - `E005_box023_p2_com_s10`
- 已修改 `scripts/pull_E005_remote_results.sh`：远程 variant 缺失时跳过并记录 warning，不再让整个回收流程失败。
- 已重新评估 7 个 full 结果：6 个 `box025` main + 1 个 `box023` COM guard。
- E005 aggregate：`num_results=7`、`num_freejoint_parity_ok=7`、`num_main_useful_proxy=0`、`num_guard_stable_proxy=1`。
- 主要结论：corrected COM 最好 `box025_com_s20 = 0.678/1.325m`，接近但未超过 E004 gravity-only；support-site variants floor contact 升到 `94-97%`，不是有效双端支撑。
- 已生成可视化拼图：`workspace/core4d_collab_retarget/results/E005/e005_full_keyframe_montage.jpg`。
- 已更新：
  - `workspace/core4d_collab_retarget/log/05_E005_partner_force_timing_support_site_results.md`
  - `workspace/core4d_collab_retarget/EXPERIMENT_TRACKER.md`

## 2026-05-17 22:50 CUDA / 远程权限预检

- 本机 `nvidia-smi` 可访问 GPU：`NVIDIA GeForce RTX 5090`，Driver `580.126.09`，CUDA `13.0`。
- 普通沙箱内 `.venv/bin/python -c "import torch"` 显示 `cuda_available=False`、`cuda_device_count=0`，不能作为本机实验入口判断依据。
- 已用提升权限运行真实 MJWarp smoke：
  - 命令入口：`env UV_CACHE_DIR=/tmp/uv-cache CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py ...`
  - override/task：`core4d_collab_E005_box025_p2_com_s20` / `box025_person2_freejoint_legobj`
  - 结果：Warp 初始化 `cuda:0 = NVIDIA GeForce RTX 5090`，4-step smoke 正常完成，输出 `/tmp/e005_cuda_probe/trajectory_mjwp.npz`。
- 远程 `ssh -o BatchMode=yes spider-remote` 正常；远端 hostname 为 `embodied-2x6000Ada`，可见 2 张 `NVIDIA RTX 6000 Ada Generation`，仓库路径 `/home/xiayb/pHRI_workspace/spider` 存在。
- 远端 `.venv/bin/python` 的 PyTorch CUDA 正常：`cuda_available=True`、`cuda_device_count=2`。

## 2026-05-17 22:58 E006 support-body proxy 审计

- 已读取 COLA paper note、E005 plan/log、`spider/config.py`、`examples/run_mjwp.py` 与 `spider/simulators/mjwp.py` 的 partner-force / mocap / weld 路径。
- 关键约束：当前 `mjwp.py` 的 object reward、qpos override、eval 口径大量假设 object 是末尾 `nq_obj=7`；如果直接把 dynamic support body 作为额外 freejoint 加到模型尾部，会破坏 `qpos_ref`/`qvel_ref` 与 model `nq/nv` 的维度断言，也会让 `qpos[:, -nq_obj:]` 不再指向 object。
- 现有可复用路线：
  - E004/E005：`_apply_partner_force` 通过 `xfrc_applied` 给 object body 写外力/力矩，稳定且不改变 `nu=29`。
  - 旧 `scene_mocap_partner.xml`：用两个 mocap partner hand geom 与 object 接触，`_update_mocap_partner` 可在 rollout 内更新 mocap 位姿。
  - `scene_weld.xml`：用 `object_target` mocap + soft weld 直接拉 object，语义太接近 object actuator，不适合作为主线，但可作为 MJWarp equality smoke 参考。
- E006 最小安全路线倾向：保持 object true-freejoint 与 `nu=29`，新增“support-body proxy controller”作为独立目标/状态记录；首版用 object-local support site 的速度/高度/yaw PD 生成 proxy wrench，并记录 partner effort / end-height 指标。后续若需要更接近 COLA，再尝试 mocap contact pad 或额外 dynamic body，但不能在首轮直接改 `nq` 尾部布局。
- 已写入 E006 中文计划：`workspace/core4d_collab_retarget/plan/06_E006_cola_support_body_proxy_plan.md`。
- 已更新 tracker：新增 E006 plan 行与关键指标演进占位。

## 2026-05-17 23:08 E006 implementation draft

- 已新增 E006 support proxy 配置字段到 `spider/config.py`，默认关闭，不影响既有实验。
- 已在 `spider/simulators/mjwp.py` 中实现：
  - setup 时从 object ref support site 预计算 proxy pose/velocity；
  - step 时用 proxy-to-support-site spring/damper + gravity share 生成 object wrench；
  - `get_support_proxy_state` 输出 force/torque/proxy/support point 诊断。
- 已在 `examples/run_mjwp.py` 中把 support proxy 诊断保存进 `trajectory_mjwp.npz`。
- 已新增 E006 脚本骨架：
  - `scripts/E006/variants.tsv`
  - `scripts/E006/generate_e006_overrides.py`
  - `scripts/run_E006_preprocess.sh`
  - `scripts/train/train_E006.sh`
  - `scripts/train/train_E006_remote_tmux.sh`
  - `scripts/run_E006_remote.sh`
  - `scripts/pull_E006_remote_results.sh`
  - `scripts/eval/eval_E006.py`
- 下一步：运行 preprocess + Python 编译检查 + 7 variant smoke。

## 2026-05-17 23:05 E006 smoke

- `py_compile` 通过：`spider/config.py`、`spider/simulators/mjwp.py`、`examples/run_mjwp.py`、E006 generator/eval。
- `bash workspace/core4d_collab_retarget/scripts/run_E006_preprocess.sh` 成功生成 7 个 E006 override。
- 本机 CUDA smoke 已完成：`bash workspace/core4d_collab_retarget/scripts/train/train_E006.sh smoke 0`，7/7 变体均产出 `trajectory_mjwp.npz`。
- NPZ 已确认包含 `support_proxy_force`、`support_proxy_torque`、`support_proxy_pos`、`support_proxy_vel`、`support_point_pos`、`support_point_vel`、`support_proxy_ref_idx`。
- E006 smoke eval 已完成：`num_results=7`、`num_freejoint_parity_ok=7`、`num_support_proxy_metrics_present=7`、`num_guard_stable_proxy=2`。该结果仅验证 wiring，4-step 指标不作为实验结论。
- smoke 中 `box025_yneg_k20_v1` support force mean/max 约 `22.9/23.8N`，connector gap mean 约 `0.044m`，说明 proxy force 记录口径正常。

## 2026-05-17 23:06 E006 full 启动

- E006 setup 已提交并推送：`817fc39 exp(core4d_collab_retarget): E006 support proxy setup`。
- 已启动远程 tmux `E006`：`bash workspace/core4d_collab_retarget/scripts/run_E006_remote.sh`。
  - remote GPU0 队列：`E006_box025_p2_yneg_k40_v1`、`E006_box025_p2_yneg_k20_v05`、`E006_box025_p2_ypos_k20_v1`。
  - remote GPU1 队列：`E006_box023_p2_xneg_k10_v1`、`E006_box023_p2_xpos_k10_v1`。
- 已启动本地 full：`bash workspace/core4d_collab_retarget/scripts/train/train_E006.sh local_wave 0`。
  - local 队列：`E006_box025_p2_yneg_k20_v1`、`E006_box025_p2_yneg_k20_v1_hc`。
- 23:24 本地 `E006_box025_p2_yneg_k20_v1` 已完成并进入 `E006_box025_p2_yneg_k20_v1_hc`。
- 23:30 远程 GPU0 `E006_box025_p2_yneg_k40_v1` 已完成；远程 GPU1 `E006_box023_p2_xneg_k10_v1` 已完成。
- 23:42 本地 local_wave 两个 full 已完成并通过 eval：
  - `E006_box025_p2_yneg_k20_v1`: obj mean/max `0.769/1.514m`，hand contact `52.6%`，floor `93.1%`，support force mean `22.6N`。
  - `E006_box025_p2_yneg_k20_v1_hc`: obj mean/max `0.767/1.503m`，hand contact `64.7%`，floor `93.6%`，support force mean `22.9N`。
  - 初步判断：本地两个 main 没有超过 E005 support-site，等远程刚度/速度/side 和 guard 完成后统一分析。
- 23:48 远程 GPU0 `E006_box025_p2_yneg_k20_v05` 已完成，已进入最后一个 `E006_box025_p2_ypos_k20_v1`。
- 23:53 远程 GPU1 `E006_box023_p2_xpos_k10_v1` 已完成；远程仅剩 GPU0 `E006_box025_p2_ypos_k20_v1`。

## 2026-05-18 00:43 E006 full 回收与分析

- 远程最后一个 `E006_box025_p2_ypos_k20_v1` 已完成；`bash workspace/core4d_collab_retarget/scripts/pull_E006_remote_results.sh` 已回收远程 5 个结果、视频、keyframes 和日志。
- 已重新运行全量 eval：`.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E006.py --all`。
- 最终 aggregate：`num_results=7`、`num_freejoint_parity_ok=7`、`num_support_proxy_metrics_present=7`、`num_main_useful_proxy=0`、`num_main_beats_E005_support_site_proxy=0`、`num_guard_stable_proxy=1`。
- main best：`E006_box025_p2_ypos_k20_v1` obj mean/max `0.704/1.410m`，hand contact `78.0%`，floor `91.3%`，force mean/max `25.6/32.9N`。
- guard：`E006_box023_p2_xneg_k10_v1` obj `0.662/1.151m`、floor `60.0%` 但 pelvis min `0.069m`，判为摔倒；`xpos` pelvis stable but object `0.847/1.441m`。
- 已生成关键帧拼图：`workspace/core4d_collab_retarget/results/E006/e006_full_keyframe_montage.jpg`。
- 已写入 E006 结果日志：`workspace/core4d_collab_retarget/log/06_E006_cola_support_body_proxy_results.md`。
- 已更新 `EXPERIMENT_TRACKER.md`：E006 标记完成，并记录“工程接入成功、算法未改善”的结论。

## 2026-05-18 10:30 E006 失败模式追加诊断：平移不足、旋转过量

- 根据 E006 视频直觉重新读 `trajectory_mjwp.npz`：`box025_person2_freejoint_legobj` 参考 object 水平净位移约 `1.57m`，起终姿态旋转只有约 `2.0deg`。
- E006 `box025` main 的实际水平净位移只有 `0.21-0.50m`，但 object 起终姿态旋转约 `20.6-47.6deg`；这支持“箱体不真正平移，主要贴地/绕支撑点翻转”的失败模式。
- 典型例子：
  - `E006_box025_p2_yneg_k40_v1`: xy 净位移 `0.495m`、object path `0.832m`、旋转 `47.6deg`。
  - `E006_box025_p2_ypos_k20_v1`: xy 净位移 `0.341m`、object path `0.628m`、旋转 `33.3deg`。
  - `E006_box025_p2_yneg_k20_v05`: xy 净位移 `0.211m`、旋转 `20.6deg`。
- proxy 自身在 `box025` main 中水平位移约 `0.79-0.84m`，仍明显小于参考 object 的 `1.57m`；support point 与 proxy gap 约 `0.15-0.20m`。因此 E006 不只是力不够，而是 proxy target、object support point、robot contact 没有闭合成可传递水平牵引的系统。
- 进一步读代码发现一个 E006 时间索引风险：`_load_support_proxy()` 接收到的 `qpos_ref` 已经被 `spider/io.py::load_data()` 插值到 `sim_dt`；但 E006 override 把 `support_proxy_ref_dt` 显式设成原始 `ref_dt=0.0333`。这样 `idx=int(t / support_proxy_ref_dt)` 在 4.13s 只索引到约第 124 帧，而插值后的参考轨迹实际有约 248 帧，导致 proxy target 只走完约半段参考平移。这很可能是 E006 “不平移只翻转”的首要实现原因。
- 后续 E007 不应只继续扫 `support_proxy_connector_kp`；需要显式处理水平运输约束/接触闭环，例如 mocap contact pad、robot-side support/contact reward、或把 proxy gap/effort 纳入 reward。

## 2026-05-18 11:05 E007 计划与脚本准备

- 用户要求后续迭代直到 work，并明确 eval 要对齐 E081 而不是 E005；已读取 E081 log / `eval_E081.py` / `comparison.csv`，确认 E081 main baseline 为 `box025_p2_legobj` obj `0.143/0.271m`、hand `89.0%`、leg intf `7.5%`、floor `59.5%`，guard 为 `box023_p2_legobj` obj `0.164/0.317m`、floor `34.7%`。
- 已写入 E007 plan：`workspace/core4d_collab_retarget/plan/07_E007_support_proxy_timebase_e081_plan.md`。
- 已修改 support proxy 默认时间基准：
  - `spider/config.py`: `support_proxy_ref_dt <= 0` 默认使用 `config.sim_dt`。
  - `spider/simulators/mjwp.py`: `_load_support_proxy()` 和 `_apply_support_proxy_force()` 默认用 `sim_dt` 索引已插值的 proxy reference。
- 已新增 E007 脚本骨架：
  - `scripts/E007/variants.tsv`
  - `scripts/E007/generate_e007_overrides.py`
  - `scripts/run_E007_preprocess.sh`
  - `scripts/train/train_E007.sh`
  - `scripts/train/train_E007_remote_tmux.sh`
  - `scripts/run_E007_remote.sh`
  - `scripts/pull_E007_remote_results.sh`
  - `scripts/eval/eval_E007.py`
- E007 eval 已从 E005 support-site 对比改为 E081 对齐：输出 proxy timebase ratio、object xy transport ratio、object rotation、`E007_reaches_E081_transport_proxy`、`E007_beats_or_matches_E081_majority`。
- 下一步：运行 py_compile + preprocess；然后按用户要求先对 `train_E007.sh`、`run_E007_remote.sh`、`pull_E007_remote_results.sh` 发起预授权。
- py_compile / bash syntax 检查已通过；`bash workspace/core4d_collab_retarget/scripts/run_E007_preprocess.sh` 已成功生成 7 个 E007 override。
- 预授权已完成：
  - `bash workspace/core4d_collab_retarget/scripts/train/train_E007.sh`
  - `bash workspace/core4d_collab_retarget/scripts/run_E007_remote.sh`
  - `bash workspace/core4d_collab_retarget/scripts/pull_E007_remote_results.sh`
- E007 4-step smoke 已完成：7/7 变体均产出 freejoint `trajectory_mjwp.npz`。
- Smoke eval 已完成：`num_results=7`、`num_freejoint_parity_ok=7`、`num_support_proxy_metrics_present=7`；4-step 的 transport ratio / E081 majority 不作为结论，只验证 E007 eval 字段写出。

## 2026-05-18 11:55 E007 partial full 早停

- E007 full 已启动：本地跑 `E007_box025_p2_yneg_k20_simdt` / `E007_box025_p2_yneg_k20_hc_simdt`，远程 GPU0 跑 3 个 box025，远程 GPU1 跑 2 个 box023 guard。
- 本地首个 full `E007_box025_p2_yneg_k20_simdt` 完成后立即单独 eval：
  - `case_window_obj_err_mean/max = 0.625/1.205m`
  - hand contact `61.3%`
  - floor contact `93.1%`
  - `E007_object_xy_disp_ratio = 0.291`
  - `E007_object_rot_deg = 41.6deg`
  - `E007_proxy_xy_disp_ratio_vs_ref_obj = 0.693`
  - `E007_reaches_E081_transport_proxy = false`
  - `E007_e081_majority_score = 1/6`
- 诊断：dt 修正确实生效，日志显示 `_load_support_proxy dt=0.0166667`；但 `support_proxy_max_xy_speed=0.8` 成为新的限速瓶颈。参考 support point 在快速段速度超过 0.8m/s，proxy 最终只走完约 `69%` 参考水平位移，object 仍只走 `29%` 并旋转替代平移。
- 已停止剩余本地/远程 E007 队列，避免继续跑同一限速配置。下一步进入 E008：高/不限速 support proxy，并优先检查 proxy ratio 是否 `>=0.95`。

## 2026-05-18 12:18 E008 计划与脚本骨架

- 已写入 E008 中文计划：`workspace/core4d_collab_retarget/plan/08_E008_support_proxy_speed_unclamped_e081_plan.md`。
- E008 的首要 gate 是 proxy 自身完整平移：`proxy_xy_disp / ref_support_xy_disp >= 0.95` 且 final gap 小；第二层才比较 E081 transport 指标。
- 已新增 `workspace/core4d_collab_retarget/scripts/E008/variants.tsv`：7 个变体，覆盖 `support_proxy_max_xy_speed=0.0`（不限速）和 `2.0m/s`，main 对齐 `box025_p2`，guard 检查 `box023_p2`。
- 已从 E007 复制 E008 脚本骨架并完成 E007→E008 命名替换；下一步增强 `eval_E008.py` 的 reference support point 诊断，并做静态检查/预授权。
- 已增强 `eval_E008.py`：新增 `E008_ref_support_xy_disp_m`、`E008_proxy_xy_disp_ratio_vs_ref_support`、`E008_proxy_final_gap_to_ref_support_m`、`E008_proxy_support_tracking_ok`，并将 `E008_reaches_E081_transport_proxy` 绑定到 proxy gate。
- 静态检查已通过：`py_compile` 覆盖 E008 generator/eval，`bash -n` 覆盖 E008 preprocess/train/remote/pull 脚本。
- `bash workspace/core4d_collab_retarget/scripts/run_E008_preprocess.sh` 已成功生成 7 个 Hydra overrides；`support_proxy_ref_dt=-1.0`、`contact_guidance=false`、`object_action_dims=0`、`partner_force_scale=0.0`。
- E008 预授权已完成：
  - `bash workspace/core4d_collab_retarget/scripts/train/train_E008.sh __codex_auth_probe__ 0`
  - `bash workspace/core4d_collab_retarget/scripts/run_E008_remote.sh __codex_auth_probe__`
  - `bash workspace/core4d_collab_retarget/scripts/pull_E008_remote_results.sh __codex_auth_probe__`
- E008 smoke 已完成：7/7 变体产出 4-step NPZ；eval aggregate 为 `num_results=7`、`num_freejoint_parity_ok=7`、`num_support_proxy_metrics_present=7`、`num_main_proxy_support_tracking_ok=5`。4-step `E008_beats_or_matches_E081_majority` 不作为结论。
- 已将 `spider/simulators/mjwp.py` 中 support proxy 日志文案从 `E006 support proxy` 改为通用 `support proxy`，不改施力公式；`py_compile` 复查通过。

## 2026-05-18 12:30 E008 full 进行中

- 已启动 E008 full：
  - 本地：`E008_box025_p2_yneg_k20_vmax0`、`E008_box025_p2_yneg_k40_vmax0`
  - 远程 GPU0：`E008_box025_p2_ypos_k20_vmax0`、`E008_box025_p2_yneg_k20_vmax2`、`E008_box025_p2_ypos_k20_vmax2`
  - 远程 GPU1：`E008_box023_p2_xneg_k10_vmax0`、`E008_box023_p2_xpos_k10_vmax0`
- 远程 `E008_box023_p2_xneg_k10_vmax0` 在 `96/272` 处超过 3 分钟无进展、GPU 利用率 0%、CPU 100%，已终止并保留日志；随后单独启动 `E008_box023_p2_xpos_k10_vmax0`。
- 首个 full `E008_box025_p2_yneg_k20_vmax0` 已完成并单独 eval：
  - proxy gate 通过：`E008_proxy_xy_disp_ratio_vs_ref_support = 1.000`，final gap `~1e-7m`
  - object xy ratio `0.711`，case-window xy ratio `0.672`
  - object rotation `12.7deg`
  - obj mean/max `0.485/0.827m`
  - hand contact `43.4%`
  - floor contact `69.9%`
  - leg-box interference `22.0%`
- 结论暂定：E007 限速是关键瓶颈之一；不限速后物体开始大幅平移且旋转下降，但仍未达到 E081，主要剩余问题变成手端闭环不足与腿/箱干涉过高。
- E008 full 最终显式评估 6 个完整结果：5 main + 1 guard；远程 `xneg` full 被排除，避免使用 smoke NPZ。
- Aggregate：`num_results=6`、`num_main_proxy_support_tracking_ok=5`、`num_main_reaches_E081_transport_proxy=0`、`num_guard_stable_proxy=1`。
- 最好结果 `E008_box025_p2_ypos_k20_vmax2`：obj mean/max `0.363/0.684m`，hand `78.0%`，floor `64.7%`，leg intf `2.9%`，xy ratio `0.724`，rot `13.6deg`，proxy ratio `0.998`。
- 可视化确认：best 变体不再只是原地旋转，已明显水平平移；但后半段 sim 仍滞后 ref，手端不像 E081 那样形成稳定托举，语义仍不是干净双端搬运。
- 已写入 E008 结果日志：`workspace/core4d_collab_retarget/log/08_E008_support_proxy_speed_unclamped_e081_results.md`；下一步 E009 应基于 `ypos_k20_vmax2` 做 robot-side hold-contact/support 闭环或 contact pad。

## 2026-05-18 13:12 E009 计划与脚本骨架

- 已提交并推送 E008 结果日志：`28d2d85 docs(core4d_collab_retarget): record E008 speed gate results`。
- 已写入 E009 中文计划：`workspace/core4d_collab_retarget/plan/09_E009_ypos_hold_contact_closure_e081_plan.md`。
- E009 目标：围绕 E008 best `ypos_k20_vmax2` 加 robot-side hold-contact 闭环，测试 hand contact 是否能从 `78%` 提升到 `>=80-85%` 并同步降低 object error。
- 已新增 E009 variants：4 个 main（HC 0.5/1/2 + `vmax0_hc1`）与 2 个稳定 `box023_xpos` guard。
- 已从 E008 复制 E009 脚本骨架并完成命名替换；`eval_E009.py` 新增 `E009_improves_E008_best` 与相对 E008 best 的 delta 字段。
- 静态检查通过：E009 generator/eval `py_compile`，E009 shell scripts `bash -n`。
- `bash workspace/core4d_collab_retarget/scripts/run_E009_preprocess.sh` 已成功生成 6 个 Hydra overrides。
- E009 预授权已完成：
  - `bash workspace/core4d_collab_retarget/scripts/train/train_E009.sh __codex_auth_probe__ 0`
  - `bash workspace/core4d_collab_retarget/scripts/run_E009_remote.sh __codex_auth_probe__`
  - `bash workspace/core4d_collab_retarget/scripts/pull_E009_remote_results.sh __codex_auth_probe__`
- E009 smoke 已完成：6/6 变体产出 4-step NPZ；eval aggregate `num_results=6`、`num_freejoint_parity_ok=6`、`num_support_proxy_metrics_present=6`、`num_main_proxy_support_tracking_ok=4`。4-step majority 不作为结论。

## 2026-05-18 13:46 E009 full 启动

- 已确认 E009 当前结果均为 4-step smoke：6 个 eval summary 的 `T=4`，NPZ mtime 13:39-13:40，不能作为正式结果。
- 本轮启动 E009 full：远程执行 `run_E009_remote.sh`，本地执行 `train_E009.sh local_wave 0`。正式评估时必须确认 full NPZ 覆盖 smoke 后再使用。
- 13:58 监控：本地 `E009_box025_p2_ypos_k20_vmax2_hc1` 到 `118/248`；远程 GPU0 `E009_box025_p2_ypos_k20_vmax2_hc05` 到 `104/248`；远程 GPU1 `E009_box023_p2_xpos_k10_vmax0_hc1` 到 `106/272`。三路 GPU/日志均活跃，暂无卡住迹象。
- 14:12 监控：本地 `E009_box025_p2_ypos_k20_vmax2_hc1` 已完成并覆盖 smoke，NPZ 形状 `(124,2,43)`；本地自动进入 `E009_box025_p2_ypos_k20_vmax0_hc1`。远程 GPU0 `hc05` 已完成并进入 `hc2`；远程 GPU1 guard `hc1` 已完成并进入 guard `hc2`。远程两个 full NPZ 已落盘，后续回收时再统一本地 eval。
- 14:26 本地 E009 队列完成。local-only eval 暂时覆盖了 `comparison.csv`/`aggregate_summary.json`，正式结果需等远程 pull 后用 6 个 full 变体显式重评。初步本地指标：`hc1` obj `0.441/0.825m`、hand `74.6%`、floor `63.0%`、xy `0.737`、rot `15.2deg`；`vmax0_hc1` obj `0.382/0.719m`、hand `55.5%`、floor `69.4%`、xy `0.789`、rot `38.6deg`。二者均未达 E081 transport gate，也未改善 E008 best。

## 2026-05-18 14:36 E006 失败补充复盘：旋转替代平移

- 按用户对可视化视频的观察，重查 E006 main 的 object/proxy 轨迹：参考 `box025_p2` object 水平净位移约 `1.571m`、起终旋转约 `2.0deg`；E006 main 实际 object 水平净位移只有 `0.211-0.495m`，但起终旋转达到 `20.6-47.6deg`。
- 代表值：`yneg_k40` object xy `0.495m`、rot `47.6deg`；`ypos_k20` object xy `0.341m`、rot `33.3deg`；`yneg_k20_v05` object xy `0.211m`、rot `20.6deg`。
- 复核 support proxy：E006 写死 `support_proxy_ref_dt=0.0333`，而运行时 qpos_ref 已按 `sim_dt` 插值；最终 `support_proxy_ref_idx=124`，proxy 自身只走约 `0.789-0.841m`（v05 为 `0.433m`），未覆盖参考全程。该时间基准错误后来在 E007 修正。
- 力学解释：support force 施加在 object-local 远端点 `[0, +/-0.38, 0.30]`，`torque = r x F`；当机器人手端接触闭环弱、物体仍高比例贴地时，系统更容易绕地面/支撑点翻转来降低局部误差，而不是把 COM 水平运输出去。
- 决策含义：E006 失败不能再归因于 kp 不够；仅增大 stiffness/hold-contact 不会解决首要问题。E007/E008 已证明先修正 proxy timebase/speed gate 后平移明显增加，后续应继续围绕 robot-side 闭环/contact pad，而不是回到 E006 的直接 wrench 扫参。

## 2026-05-18 14:39 E009 full 回收与结论

- 远程 E009 已完成并回收；`pull_E009_remote_results.sh` 自动 eval 只覆盖 4 个远程变体，因此显式重评 6 个 full NPZ。
- 6 个 full NPZ 均已确认不是 smoke：main `qpos=(124,2,43)`，guard `qpos=(136,2,43)`。
- 最终 aggregate：`num_results=6`、`num_freejoint_parity_ok=6`、`num_main_proxy_support_tracking_ok=4`、`num_main_reaches_E081_transport_proxy=0`、`num_main_improves_E008_best=0`、`num_guard_stable_proxy=1`。
- E009 best-ish `hc05`：obj `0.395/0.726m`、hand `76.9%`、floor `62.4%`、leg intf `3.5%`、xy ratio `0.710`、rot `18.2deg`，比 E008 best obj mean 差 `+0.032m`。
- `hc2` 虽然 obj mean `0.344m` 略低于 E008 best，但 hand 降到 `66.5%`、floor 升到 `70.5%`、rot `38.5deg`，属于旋转/地面替代，不是搬运。
- `vmax0_hc1` xy ratio `0.789` 达标，但 hand `55.5%`、rot `38.6deg`，同样不是有效双端搬运。
- 已提取 E009 关键帧到 `workspace/core4d_collab_retarget/results/E009/keyframes/`，并写入 E009 结果日志 `workspace/core4d_collab_retarget/log/09_E009_ypos_hold_contact_closure_e081_results.md`。
- 决策：E010 不再继续 hold-contact scale sweep，转向结构性 contact pad / soft constraint，让 partner support 通过 MuJoCo contact/约束进入 object，而不是继续 direct wrench + reward。

## 2026-05-18 14:46 E010 计划

- 已写入 E010 中文计划：`workspace/core4d_collab_retarget/plan/10_E010_mocap_contact_pad_support_e081_plan.md`。
- E010 的核心假设：E006/E009 的旋转替代平移来自 direct off-COM wrench + robot hand 闭环弱；下一轮让 virtual partner 作为 mocap/contact pad 与 object 接触传力，而不是继续直接写 `xfrc_applied`。
- 计划 variants：5 个 `box025_p2` main 覆盖 ypos/yneg、pad size、vmax、轻量 HC；2 个 `box023_p2` guard 覆盖 xpos pad size/HC。
- 成功标准继续对齐 E081 transport gate：obj `<=0.20/0.40m`、hand `>=80%`、floor `<=75%`、xy `>=0.75`、rot `<=15deg`、true-freejoint parity ok；若 pad 独自搬而 hand 不参与，不算 work。

## 2026-05-18 15:58 E010 实现与 smoke

- 已新增 E010 最小实现：
  - `spider/config.py` 新增 `support_proxy_mode` 与 `support_proxy_mocap_body_name`；
  - `spider/simulators/mjwp.py` 支持 `support_proxy_mode=mocap_pad`，复用 support proxy trajectory 更新 `support_proxy_pad` mocap body，并在该模式禁用 direct object wrench；
  - 新增 `scene_contact_pad08/10/12/16.xml` 生成脚本，contact pad 为 mocap body，不进入 qpos/ctrl；
  - 新增 E010 variants、preprocess/train/remote/pull/eval 脚本。
- 静态检查通过：`py_compile` 覆盖 config/mjwp/E010 generator/eval，`bash -n` 覆盖 E010 shell 脚本。
- `run_E010_preprocess.sh` 已生成 4 个 contact-pad scene XML 与 7 个 overrides；MuJoCo 加载检查：`nq=43,nv=41,nu=29,nmocap=1,npair=43`。
- 首次 smoke 失败原因：E010 variants 误写 `person_idx=2`，而 raw mask shape 是 `(T,2,2)`；已修正为 `person_idx=1` 并重新生成 overrides。
- E010 4-step smoke 完成：7/7 变体均产出 NPZ。Smoke eval aggregate：`num_results=7`、`num_freejoint_parity_ok=7`、`num_support_proxy_metrics_present=7`、`num_main_proxy_support_tracking_ok=5`、`num_guard_stable_proxy=2`。4-step majority 不作为效果结论。
- E010 本地训练、远程启动、远程回收脚本已预授权；下一步提交 E009/E010 setup 后启动 full。

## 2026-05-18 16:01 E010 full 启动

- 已提交并推送 E010 setup：`e0a6488 exp(core4d_collab_retarget): set up E010 contact pad support`。
- 已启动 E010 full：
  - 本地 GPU0：`E010_box025_p2_ypos_pad10_vmax2` -> `E010_box025_p2_ypos_pad10_vmax2_hc05`
  - 远程 GPU0：`E010_box025_p2_ypos_pad16_vmax2` -> `E010_box025_p2_ypos_pad10_vmax0` -> `E010_box025_p2_yneg_pad10_vmax2`
  - 远程 GPU1：`E010_box023_p2_xpos_pad08_vmax0` -> `E010_box023_p2_xpos_pad12_vmax0`
- 16:01 首轮监控：远程 GPU0/GPU1 日志均已进入首个变体，GPU util 约 `44%/39%`；本地 full 正在运行首个 `pad10_vmax2`。
- 16:04 监控：本地 `pad10_vmax2` 到 `90/248`；远程 GPU0 `pad16_vmax2` 到 `76/248`；远程 GPU1 `pad08_vmax0` 到 `78/272`。三路均稳定，暂无 OOM/卡死，但 contact-pad 模式单步计划耗时约 `9-10.6s`，比 E009 更慢。
- 16:21 本地首个 `E010_box025_p2_ypos_pad10_vmax2` 完成并单独 eval：obj `0.714/1.376m`、hand `90.8%`、floor `92.5%`、leg intf `0.6%`、xy ratio `0.172`、rot `13.3deg`、proxy ratio `0.998`、connector gap mean `0.658m`。结论：小 contact pad 未能把 partner-side 支撑传进 object，虽然 hand contact 高，但 object 基本不运输；等待 pad16/vmax0 验证是否为 pad 尺寸/速度问题。
- 16:30 监控：远程首批 `pad16_vmax2` 与 `pad08_vmax0` 已完成并进入第二批；本地 `hc05` 到 `156/248`，远程 GPU0 `pad10_vmax0` 到 `98/248`，远程 GPU1 `pad12_vmax0` 到 `78/272`。首个本地结果提示 pad10 接触面可能不足或 pad-object contact 没有效传力，待 pad16/vmax0 验证。
- 16:38 本地 E010 队列完成。local-only eval（2 个 main）：
  - `pad10_vmax2`: obj `0.714/1.376m`、hand `90.8%`、floor `92.5%`、xy `0.172`、rot `13.3deg`、gap `0.658m`
  - `pad10_vmax2_hc05`: obj `0.680/1.295m`、hand `83.8%`、floor `83.8%`、xy `0.276`、rot `17.4deg`、gap `0.620m`
- 本地结论：pad10 contact pad 即使配轻量 HC 也没有有效运输，object 仍高度贴地且 support-point gap 很大；远程 pad16/vmax0/yneg 结果决定是否继续 contact-pad 尺寸/速度方向。
- 16:47 监控：远程 GPU0 已完成 `pad16_vmax2` 与 `pad10_vmax0`，进入最后一个 `yneg_pad10_vmax2`（约 `40/248`）；远程 GPU1 `pad12_vmax0` 到 `254/272`，即将完成。剩余主要等待 GPU0 最后一个 main。

## 2026-05-18 17:05 E010 full 结论与 E006 失败链条更新

- 远程 E010 已完成并回收；显式重评 7 个 full NPZ，main `qpos=(124,2,43)`，guard `qpos=(136,2,43)`，均不是 smoke。
- 最终 aggregate：`num_results=7`、`num_freejoint_parity_ok=7`、`num_support_proxy_metrics_present=7`、`num_main_proxy_support_tracking_ok=5`、`num_main_reaches_E081_transport_proxy=0`、`num_main_improves_E008_best=0`、`num_guard_stable_proxy=1`。
- E010 main 指标整体失败：`pad10_vmax2` obj `0.714/1.376m`、hand `90.8%`、floor `92.5%`、xy `0.172`；`pad16_vmax2` obj `0.698/1.334m`、hand `87.3%`、floor `84.4%`、xy `0.326`；`pad10_vmax2_hc05` obj `0.680/1.295m`、hand `83.8%`、floor `83.8%`、xy `0.276`。
- 可视化关键帧确认：ref box 已大幅水平移动，sim box 仍靠近起点；hand contact 高但不等于托举/运输，object 高比例贴地，support-point gap `0.62-0.66m`。
- 结合用户对 E006 视频的判断，当前失败链条更新为：E006 direct off-COM wrench 会出现“旋转替代平移”；E008 修正 timebase/speed 后能恢复部分平移；E009 说明 hold-contact reward 不能闭合 robot-side 支撑；E010 说明单个 mocap contact pad 又太弱/几何不对，不能有效传力。
- 已写入 E010 中文结果日志：`workspace/core4d_collab_retarget/log/10_E010_mocap_contact_pad_support_e081_results.md`，并更新 `EXPERIMENT_TRACKER.md`。
- 下一步 E011 不继续扫 pad size / vmax / HC；计划做 soft equality/weld diagnostic，在 E008 best 上定量测出达到 E081 transport 所需的最小外部 coupling，再决定是否落到真实 partner mocap hands / 双点接触。

## 2026-05-18 17:15 E011 诊断实现前置修复

- 已准备 E011 的 soft object tether 诊断口径：复用已有 `partner_force_spring` 作为 object COM reference spring，但这需要修正两个工程口径。
- `spider/simulators/mjwp.py`：`step_env` 现在在 `partner_force_spring_kp>0` 或 `partner_force_spring_kp_rot>0` 时也会调用 `_apply_partner_force`，不再要求 `partner_force_scale>0`；这允许纯 soft tether（无额外重力补偿）作为诊断。
- `spider/simulators/mjwp.py`：object `xfrc_applied` 改为每 step 只清一次，partner force 与 support proxy wrench 可以累加，避免 E011 组合 E008 best + COM tether 时互相覆盖。
- `examples/run_mjwp.py`：新增保存 `partner_force_force` / `partner_force_torque` 诊断字段，方便 E011 eval 统计外部 coupling effort。
- 静态检查通过：`.venv/bin/python -m py_compile spider/simulators/mjwp.py examples/run_mjwp.py`。
- 已写入 E011 中文计划：`workspace/core4d_collab_retarget/plan/11_E011_soft_object_tether_diagnostic_e081_plan.md`。
- E011 的核心目的不是给最终算法开绿灯，而是量化达到 E081 transport 至少需要多强 external coupling，并区分 `physical_candidate` / `external_only_success` / `robot_side_blocked` / `rotation_shortcut`。
- 已新增 E011 变体与脚本骨架：
  - `workspace/core4d_collab_retarget/scripts/E011/variants.tsv`：9 个变体，覆盖 COM-only kp 20/50/100、gravity scale、弱 rot tether、E008 best + COM tether，以及 box023 guard；
  - `workspace/core4d_collab_retarget/scripts/E011/generate_e011_overrides.py`
  - `workspace/core4d_collab_retarget/scripts/run_E011_preprocess.sh`
  - `workspace/core4d_collab_retarget/scripts/train/train_E011.sh`
  - `workspace/core4d_collab_retarget/scripts/train/train_E011_remote_tmux.sh`
  - `workspace/core4d_collab_retarget/scripts/run_E011_remote.sh`
  - `workspace/core4d_collab_retarget/scripts/pull_E011_remote_results.sh`
  - `workspace/core4d_collab_retarget/scripts/eval/eval_E011.py`
- 静态检查通过：E011 generator/eval + `spider/simulators/mjwp.py` + `examples/run_mjwp.py` 的 `py_compile`，E011 shell 脚本 `bash -n`，variants 每行均为 34 列。

## 2026-05-18 17:31 E011 preprocess / smoke

- `bash workspace/core4d_collab_retarget/scripts/run_E011_preprocess.sh` 已成功生成 9 个 E011 overrides；COM-only 变体 `support_proxy_enabled=false`，E008+COM 变体 `support_proxy_enabled=true` 且保持 `ypos/k20/vmax2` 口径。
- 首次 sandbox 内 smoke 因无 CUDA 失败：`RuntimeError: No CUDA GPUs are available`；已使用授权后的本地 CUDA 重新运行。
- E011 smoke 已完成：9/9 变体产出 4-step NPZ；显式 eval 输出 `num_results=9`、`num_freejoint_parity_ok=9`、`num_partner_force_metrics_present=9`、`num_support_proxy_metrics_present=2`。
- 4-step smoke 中 object 指标不作为实验结论；当前只确认 wiring、partner force 诊断字段、E011 eval aggregate 均可用。
- E011 本地训练、远程启动、远程回收脚本预授权 probe 已完成。

## 2026-05-18 17:34 E011 full 启动

- 已提交并推送 E011 setup：`8ba6268 exp(core4d_collab_retarget): set up E011 soft tether diagnostic`。
- 已启动 E011 full：
  - 本地 GPU0：`E011_box025_p2_com_xyz_k20` -> `E011_box025_p2_ypos_k20_vmax2_com_k25` -> `E011_box025_p2_ypos_k20_vmax2_com_k50`
  - 远程 GPU0：`E011_box025_p2_com_xyz_k50` -> `E011_box025_p2_com_xyz_k100` -> `E011_box025_p2_com_xyz_k50_g1`
  - 远程 GPU1：`E011_box025_p2_com_xyz_k50_rot1` -> `E011_box023_p2_com_xyz_k50` -> `E011_box023_p2_com_xyz_k100`
- 17:34 监控：本地首个约到 `26/248`；远程 GPU0/GPU1 首个均约到 `24/248`；三路 GPU util `44-47%`，暂无 OOM/卡死。当前 full 单步计划耗时约 `9-11.5s`。
- 17:40 监控：本地 `E011_box025_p2_com_xyz_k20` 到约 `120/248`；远程 GPU0 `E011_box025_p2_com_xyz_k50` 到约 `106/248`；远程 GPU1 `E011_box025_p2_com_xyz_k50_rot1` 到约 `100/248`。三路日志均正常推进，未见 CUDA/SSH 权限问题；远程 tmux pane 只保留启动横幅，实际进度以后直接看 variant 日志。
- 17:45 监控与 E006 复核：本地首个到约 `168/248`，远程 GPU0/GPU1 首个到约 `148/248`、`138/248`。重读 E006 日志后确认用户视频观察的量化原因：E006 一方面 `support_proxy_ref_dt=0.0333` 在 sim_dt 插值参考上只走约半段；另一方面 off-COM connector 的 `r x F` 在手端闭环弱、floor 高时形成旋转捷径。E011 的 COM-level spring 结果将直接验证“去掉 off-COM/timebase 后是否恢复平移”。
- 17:50 本地首个 full `E011_box025_p2_com_xyz_k20` 完成，NPZ 已从 smoke 覆盖为 full（约 `1.1MB`），关键帧已生成。单 variant eval：diagnostic=`insufficient_coupling`，obj `0.533/1.035m`，hand `86.7%`，floor `58.4%`，leg `1.2%`，xy `0.969/1.571m`（ratio `0.617`），rot `4.9deg`（ref `2.0deg`），partner force mean/max `26.3/35.1N`，torque max `0`，effort reasonable，parity ok。初步解释：COM k20 已明显去掉 E006 的旋转捷径和高 floor 问题，但 coupling 强度不足，仍达不到 E081 transport。
- 17:54 已部分回收远程 GPU0 首个 `E011_box025_p2_com_xyz_k50` 并 eval：diagnostic=`insufficient_coupling`，majority score `4/6`，obj `0.451/0.851m`，hand `85.0%`，floor `62.4%`，leg `0.0%`，xy `1.307/1.571m`（ratio `0.832`），rot `2.7deg`，partner force mean/max `30.1/43.4N`，torque max `0`，effort reasonable。解释更新：COM k50 能恢复水平平移且不走旋转/地面捷径，但绝对 object tracking error 仍大，说明剩余瓶颈不是单纯“能不能推动 COM”，而是姿态/高度/robot-object 相对位形没有进入 E081 级闭环。
- 17:56 已部分回收远程 GPU1 首个 `E011_box025_p2_com_xyz_k50_rot1` 并 eval：diagnostic=`insufficient_coupling`，majority score `4/6`，obj `0.427/0.835m`，hand `83.2%`，floor `56.6%`，leg `0.0%`，xy ratio `0.763`，rot `2.8deg`，partner force mean/max `28.9/42.7N`，torque max `1.5Nm`，effort reasonable。弱 orientation tether 略降 obj/floor，但没有解决 E081 error；residual rotation 不是主要瓶颈。
- 18:01 监控：本地第二条 `E011_box025_p2_ypos_k20_vmax2_com_k25` 到约 `150/248`；远程 GPU0 `E011_box025_p2_com_xyz_k100` 到约 `104/248`；远程 GPU1 `E011_box023_p2_com_xyz_k50` 到约 `88/272`。三路继续稳定，暂无 OOM/卡死；已回收的 k20/k50/rot1 共同支持“COM 施力能恢复平移，E006 失败含明显 off-COM 旋转捷径；但 E081 级精确搬运仍没闭合”。
- 18:09 本地第二条 `E011_box025_p2_ypos_k20_vmax2_com_k25` 完成并 eval：diagnostic=`insufficient_coupling`，majority score `3/6`，obj `0.420/0.792m`，hand `87.9%`，floor `69.4%`，leg `0.0%`，xy ratio `0.686`，rot `24.8deg`；proxy tracking ok，support force mean/max `27.9/58.8N`，partner force mean/max `7.9/19.7N`，effort ok，parity ok。相对 E008 best，obj mean/max 变差 `+0.057/+0.109m`，xy ratio 变差 `-0.038`，rot 增加 `+11.2deg`。结论：E008 best 加少量 COM coupling 没有跨过门槛，反而重新诱发部分旋转。
- 18:16 已回收远程 GPU0 第二条 `E011_box025_p2_com_xyz_k100` 并 eval：diagnostic=`robot_side_blocked`，majority score `4/6`，obj `0.340/0.673m`，hand `86.1%`，floor `61.3%`，leg `0.0%`，xy ratio `0.905`，rot `3.5deg`，partner force mean/max `32.6/53.1N`，effort ok，parity ok。它相对 E008 best 改善 obj mean/max `-0.023/-0.010m`、xy ratio `+0.181`、rot `-10.1deg`，但仍未达到 E081 gate（obj 仍高于 `0.20/0.40m`）。强 COM tether 已能恢复平移但不能恢复 E081 级精确轨迹，指向 robot-side/姿态闭环瓶颈。
- 18:18 已回收远程 GPU1 guard `E011_box023_p2_com_xyz_k50` 并 eval：guard stable true，diagnostic=`insufficient_coupling`，obj `0.681/1.235m`，hand `55.3%`，floor `51.3%`，leg `0.0%`，pelvis min `0.683m`，xy ratio `0.774`，rot `10.7deg`，partner force mean/max `43.0/70.2N`，effort ok，parity ok。它说明 COM 口径没有破坏 guard 稳定性，但 guard 仍不构成协作搬运。
- 18:28 本地队列完成；`E011_box025_p2_ypos_k20_vmax2_com_k50` eval：diagnostic=`insufficient_coupling`，majority score `4/6`，obj `0.378/0.718m`，hand `83.8%`，floor `65.9%`，leg `4.0%`，xy ratio `0.811`，rot `13.85deg`，proxy tracking ok，support force mean/max `27.3/46.2N`，partner force mean/max `11.9/27.8N`，effort ok，parity ok。相对 E008 best，xy ratio 改善 `+0.087`，但 obj mean/max 变差 `+0.015/+0.035m`，rot 基本持平；E008+COM k25/k50 都不是有效修补路线。
- 18:37 已回收远程 GPU0 `E011_box025_p2_com_xyz_k50_g1` 并 eval：diagnostic=`insufficient_coupling`，majority score `3/6`，obj `0.350/0.694m`，hand `75.1%`，floor `22.5%`，leg `0.0%`，xy ratio `0.971`，rot `11.1deg`，partner force mean/max `49.1/58.7N`，z mean `48.0N`，effort ok，parity ok。gravity scale=1 能显著降低 floor 并基本走完水平路径，但 hand contact 掉到 `<80%` 且 obj error 仍高；这更像外部竖直支撑拉走物体，而不是机器人手端闭合的协作搬运。

## 2026-05-18 18:43 E011 full 完成

- 远程最后一个 guard `E011_box023_p2_com_xyz_k100` 已完成并回收；远程 tmux session 已结束。
- 已对 9 个 full NPZ 显式重评，最终 aggregate：`num_results=9`、`num_main_results=7`、`num_guard_results=2`、`num_freejoint_parity_ok=9`、`num_partner_force_metrics_present=9`、`num_support_proxy_metrics_present=2`、`num_main_reaches_E081_transport=0`、`num_main_beats_or_matches_E081_majority=4`、`num_main_improves_E008_best=1`、`num_guard_stable=1`，diagnostic classes：`insufficient_coupling=7`、`robot_side_blocked=2`。
- Guard k100：obj `0.351/0.604m`、hand `57.3%`、floor `33.3%`、leg `18.7%`、pelvis min `0.113m`、xy ratio `0.963`、rot `9.2deg`，tracking 改善但不稳定/摔倒。
- 已生成并检查 E011 视觉拼图：`workspace/core4d_collab_retarget/results/E011/keyframes/e011_visual_montage.jpg`。视觉结论：COM-only main 不再像 E006 一样原地旋转，kp 越大平移越明显；`g1` 能离地和平移但手端脱开；E008+COM k25/k50 仍有明显姿态偏差；guard k100 后期摔倒。
- 已写入中文结果日志：`workspace/core4d_collab_retarget/log/11_E011_soft_object_tether_diagnostic_e081_results.md`，并更新 `EXPERIMENT_TRACKER.md`。
- 当前 E006 失败解释收敛：E006 的“只旋转、不平移”主要来自 off-COM support wrench 的 `r x F` 力矩捷径 + `support_proxy_ref_dt` timebase 截断 + robot-side 闭环弱；E011 证明改成 COM spring 可恢复平移，但 E081 级精度还需要 robot-side/partner-side 双点闭合，而不是继续加单点/单 COM 外力。

## 2026-05-18 18:52 E012 计划

- 已写入 E012 中文计划：`workspace/core4d_collab_retarget/plan/12_E012_dual_point_partner_pose_closure_plan.md`。
- 计划核心：不再扫单 COM kp/gravity；新增 dual-point partner-side local feature spring，在 `box025` partner 侧面用两个点 `[+/-x, +0.38, 0.30]` 同时跟随 reference object，对比 E011 k100 是否能降低 obj error，同时防止 E006 单点 off-COM 旋转捷径。
- E012 计划 8 个 full variants：6 个 main 覆盖 dual-point 间距/kp/gravity/object-reward shaping，2 个 guard 覆盖 `box023` 稳定性。
- 下一步实现 `partner_force_points_local` multi-point spring、E012 overrides/scripts/eval，并先跑 smoke。

## 2026-05-18 22:26 E012 实现 / 授权 / smoke

- 已实现 E012 双点虚拟协作力：
  - `spider/config.py` 新增 `partner_force_points_local`；
  - `spider/simulators/mjwp.py` 在该字段非空时，将 spring force 分配到多个 object-local feature points，再合成 net force 与 `sum(r_i x F_i)` torque 写入 object `xfrc_applied`；
  - 单点 `partner_force_point_local` 与 COM spring 行为保持兼容，双点字段优先生效。
- 已新增 E012 通用脚本：
  - `workspace/core4d_collab_retarget/scripts/E012/variants.tsv`
  - `workspace/core4d_collab_retarget/scripts/E012/generate_e012_overrides.py`
  - `workspace/core4d_collab_retarget/scripts/run_E012_preprocess.sh`
  - `workspace/core4d_collab_retarget/scripts/train/train_E012.sh`
  - `workspace/core4d_collab_retarget/scripts/train/train_E012_remote_tmux.sh`
  - `workspace/core4d_collab_retarget/scripts/run_E012_remote.sh`
  - `workspace/core4d_collab_retarget/scripts/pull_E012_remote_results.sh`
  - `workspace/core4d_collab_retarget/scripts/eval/eval_E012.py`
- `run_E012_preprocess.sh` 已生成 8 个 overrides；示例主变体使用 `partner_force_points_local: [[0.2, 0.38, 0.3], [-0.2, 0.38, 0.3]]`。
- 权限/资源探针已完成：
  - 本机 `nvidia-smi`：RTX 5090，CUDA driver 可见；
  - 远程 `spider-remote`：2 张 RTX 6000 Ada，SSH 可用；
  - E012 本地训练、远程启动、远程回收脚本前缀均已授权。
- 静态检查通过：E012 generator/eval 与核心 Python `py_compile`；E012 shell 脚本 `bash -n`；variants 每行 38 列。
- E012 4-step smoke 已完成：8/8 变体产出 NPZ，`partner_force_force` / `partner_force_torque` 诊断字段存在。
- E012 smoke eval 仅作 wiring 验证，不能作为效果结论：aggregate 显示 `num_results=8`、`num_dual_points_config_ok=8`、`num_freejoint_parity_ok=8`、`num_partner_force_metrics_present=8`、`num_main_reaches_E081_transport=0`。下一步提交 setup 后启动 full：本地跑 2 个 local main，远程两卡跑 6 个 remote 变体。

## 2026-05-18 22:49 E012 full 运行中

- 已提交并推送 E012 setup：`b507451 exp(core4d_collab_retarget): set up E012 dual point closure`。
- 远程首轮在 `E012_box025_p2_dualy_x20_k50_g05` / `E012_box025_p2_dualy_x30_k100_g05` 约 `38-40/248` 后日志停更，两个进程仍各占约 100% CPU；判断为可能的单步求解慢/卡住，而非 SSH/CUDA 权限问题。
- 已新增并推送 watchdog 修复：`03618ba exp(core4d_collab_retarget): add E012 watchdog`。远程 tmux 默认 `RUN_STALL_TIMEOUT_SECONDS=300`，单变体 300s 无日志更新则 kill 并继续队列。
- watchdog 版远程已重启并越过上次卡点，目前两卡首批均继续推进（约 `122/248`），暂无 timeout 记录。
- 本地第一条 full `E012_box025_p2_dualy_x20_k100_g05` 完成并单独 eval：obj `0.362/0.660m`、hand `80.9%`、floor `64.7%`、leg `1.2%`、xy ratio `0.788`、rot `20.6deg`、partner force mean/max `33.9/128.8N`、torque max `30.0Nm`（打到 clamp）。
- 对比 E011 best `E011_box025_p2_com_xyz_k100`：E012 dual k100 的 obj mean 差 `+0.021m`，obj max 小幅好 `-0.012m`，但 rotation 超过 `15deg` gate，`E012_pose_closure_helped=false`，diagnostic=`insufficient_coupling`。初步判断：双点闭合没有直接解决 E011 的 E081 精度缺口，且引入额外姿态力矩负担；等待 obj3、x30/g08/k150 与 guard 结果确认是否有局部例外。

## 2026-05-18 23:12 E012 远程部分回收

- 本地 `local_wave` 两条已完成；收尾时 `train_E012.sh` 报过一次 shell 语法错误，但 `bash -n` 复查当前脚本通过，且两条 full NPZ 与局部 eval 已产出。
- 已执行 `pull_E012_remote_results.sh` 回收远程；当前只拉回远程首批两条 full：`E012_box025_p2_dualy_x20_k50_g05` 与 `E012_box025_p2_dualy_x30_k100_g05`。
- 远程 tmux `E012` 仍在运行：GPU0 正跑 `E012_box025_p2_dualy_x20_k150_g05`，GPU1 正跑 `E012_box025_p2_dualy_x20_k100_g08`，两者约在 `130/248`，之后还会各自串行跑一个 `box023` guard。
- 本地结果目录里 `k150/g08/guard` 仍有 19KB smoke 占位 NPZ；在远程 full 覆盖前不能纳入最终结论。
- 已对 4 条 full 显式局部 eval：aggregate `num_results=4`、`num_main_reaches_E081_transport=0`、diagnostic=`insufficient_coupling:2` / `rotation_shortcut:2`。
- 4 条阶段性结果：`x20_k50` obj `0.466/0.899m`、xy `0.815`、rot `9.0deg`，coupling 不足；`x20_k100` obj `0.362/0.660m`、xy `0.788`、rot `20.6deg`，object max 略好但旋转超标；`x20_k100_obj3` obj `0.360/0.643m` 但 xy `0.539`、rot `147.3deg`；`x30_k100` hand `90.2%`、floor `54.9%` 但 rot `142.4deg`。当前趋势是 dual-point 越强/力臂越大越容易走姿态力矩捷径。

## 2026-05-18 23:24 E012 main 六条完成

- 已回收远程第二批 main：`E012_box025_p2_dualy_x20_k150_g05` 与 `E012_box025_p2_dualy_x20_k100_g08`，两个 `box023` guard 已在远程开始运行。
- 已对 6 条 main full 显式 eval：aggregate `num_results=6`、`num_main_reaches_E081_transport=0`、`num_main_pose_closure_helped=0`、diagnostic=`insufficient_coupling:2` / `rotation_shortcut:4`。
- 关键新增结果：`g08` hand `90.8%`、floor `52.6%`、xy ratio `1.050`，但 obj `0.363/0.729m`、rot `138.4deg`；`k150` obj 最好 `0.306/0.606m`、xy `0.980`、floor `59.5%`，但 rot `97.8deg`、torque max `30Nm`，仍是 rotation shortcut。
- 阶段性结论：E012 的双点虚拟力把 “E011 COM spring 的平移 coupling” 换成了 “off-COM torque shortcut”。`k150/g08` 虽然能改善 xy/floor/obj 的局部指标，但旋转大到不可接受，不能判为协作搬运。

## 2026-05-18 23:48 E012 full 完成

- 远程 E012 已结束，tmux session 退出；已回收 6 条远程 full 结果与视频，合并本地 2 条 full 结果。
- 8 条显式 full eval 完成：`num_results=8`、`num_main_results=6`、`num_guard_results=2`、`num_freejoint_parity_ok=8`、`num_dual_points_config_ok=8`、`num_partner_force_metrics_present=8`、`num_main_reaches_E081_transport=0`、`num_main_pose_closure_helped=0`、`num_guard_stable=0`。诊断分布：`rotation_shortcut=4`、`insufficient_coupling=2`、`guard_unstable=2`。
- Main 最好 object error 是 `E012_box025_p2_dualy_x20_k150_g05`：obj `0.306/0.606m`，相对 E011 best 改善 `-0.034/-0.067m`，但 rot `97.8deg`、torque max `30Nm`，属于旋转捷径；不能判 work。
- Guard 两条均失败：`guard_k50` pelvis min `0.052m`、leg intf `22.0%`、rot `172.9deg`；`guard_k100` pelvis min `0.543m`、floor `75.3%`、obj `0.682/1.273m`，仍未稳定。
- 已生成并检查关键帧拼图：`workspace/core4d_collab_retarget/results/E012/keyframes/e012_visual_montage.jpg`。视觉观察与量化一致：main 的强 coupling 末帧出现箱体大角度姿态偏转；两个 guard 明显倒地/跪倒。

## 2026-05-19 00:52 E013-E016 路线恢复与 E013 前置判断

- 目标文档 `workspace/core4d_collab_retarget/docs/03_agent_execution_plan_E013_E016.md` 当前是 0 行未跟踪文件，不能直接作为可执行计划；已从 `docs/01_direction_review_2026-05-18.md` 和 `docs/02_E011_k100_vs_E081_full_metric_comparison.md` 恢复路线。
- 恢复出的下一步顺序：E013 true-freejoint object oracle 必做；E014 做 COLA 特征 B（kinematic support + 6-DoF/weld 位置约束）；E014b 条件触发 stiffness sweep；E015 加 COLA 特征 A（dynamic support + PD）；E015b/E015c 条件 sweep；E016 仅作为 COLA sweep 后的回退/正交验证。
- E012 full 已完成并符合路线预期：spring/multi-point force 范式不再继续扫参，强 coupling 主要打开 rotation shortcut。
- 初步代码判断：现有 `object_pd_override` 只覆盖 `scene_act` 的 6 维 object actuator ctrl；E013 要在 true-freejoint scene 上做 oracle，不能简单设置 `object_pd_override=true`。需要复用或规范化 freejoint kinematic override 口径（`partner_force_spring_kp < 0` 分支）或新增专用配置字段。
- 已补全用户点名的执行计划文档：`workspace/core4d_collab_retarget/docs/03_agent_execution_plan_E013_E016.md`。该文档把 E013-E016 条件链明确为 E013 oracle -> E014 COLA-B -> E014b 条件 sweep -> E015 COLA-A+B -> E015b/E015c 条件 sweep -> E016 回退/正交验证。
- 已写入 E013 正式计划：`workspace/core4d_collab_retarget/plan/13_E013_true_freejoint_object_oracle_plan.md`。下一步开始实现 E013 的显式 freejoint object kinematic oracle 配置、脚本与 eval。
- 已实现 E013 第一版代码/脚本：
  - `spider/config.py` 新增 `object_kinematic_override` / `object_kinematic_ref_dt` / `object_kinematic_set_qvel`；
  - `examples/run_mjwp.py` 在 E013 口径下预加载 ref object qpos/qvel；
  - `spider/simulators/mjwp.py` 每 step 前后写 true-freejoint object qpos/qvel；
  - 新增 E013 variants、override generator、preprocess、train、eval 脚本。
- 下一步做静态检查与 override 生成，然后跑 E013 smoke。
- 静态检查通过：核心 Python 与 E013 generator/eval `py_compile`，E013 shell 脚本 `bash -n`，variants 每行 7 列。
- `run_E013_preprocess.sh` 已生成两个 override：`core4d_collab_E013_box025_p2_obj_oracle.yaml`、`core4d_collab_E013_box023_p2_obj_oracle.yaml`。人工检查关键字段符合 true-freejoint oracle：`scene_name=scene`、`contact_guidance=false`、`object_pd_override=false`、`object_kinematic_override=true`、`object_action_dims=0`、`object_actuator_ids=[]`。
- E013 4-step smoke 已完成：2/2 变体产出 NPZ；显式 eval 通过，aggregate 为 `num_results=2`、`num_freejoint_oracle_config_ok=2`、`num_near_e081_obj_oracle=2`、`num_guard_stable=1`。该结果只验证 wiring，不作为 full 效果结论。
- 下一步启动 E013 full（本地顺序跑 main + guard）。

## 2026-05-19 00:44 E013 full 启动

- 已启动 `bash workspace/core4d_collab_retarget/scripts/train/train_E013.sh full 0`。
- 当前正在跑 main `E013_box025_p2_obj_oracle`；日志确认 object kinematic oracle 已加载：`ref_qpos shape=(298, 7)`、`ref_qvel shape=(298, 6)`、`ref_dt=0.0166667`。
- 00:45 监控：main 到 `36/248`，每 2 个 sim step 约 `9.2s`，暂无 CUDA/OOM/卡住迹象。full NPZ 尚未覆盖 smoke。
- 00:50 监控：main 到 `72/248`，每 2 个 sim step 约 `8.9-9.1s`，仍稳定运行；guard 尚未开始。
- 00:56 监控：main 到 `106/248`，仍无异常日志；运行进度接近一半。
- 01:02 监控：main 到 `150/248`，约 60% 完成，计划时间仍稳定在 `~9.0s/2 sim steps`。
- 01:08 监控：main 到 `186/248`，后半程略快（约 `8.8-8.9s/2 sim steps`），预计数分钟后进入 guard。
- 01:01（日志时间）main `E013_box025_p2_obj_oracle` 已完成，full NPZ 已覆盖 smoke：`1.1MB`。脚本已进入 guard `E013_box023_p2_obj_oracle`，当前 guard 刚启动到 `12/272`。
- 01:04（日志时间）guard 到 `46/272`，每 2 个 sim step 约 `9.1s`，暂无异常；guard NPZ 仍是 smoke 占位，等待 full 完成覆盖。
- 01:12（日志时间）guard 到 `90/272`，约三分之一完成，速度稳定，暂无异常。
- 01:18（日志时间）guard 到 `136/272`，一半完成，仍稳定。
- 01:24（日志时间）guard 到 `180/272`，约三分之二完成，后半程约 `8.7-8.9s/2 sim steps`。
- 01:30（日志时间）guard 到 `224/272`，进入最后四分之一，暂无异常。

## 2026-05-19 01:21 E013 full 完成

- E013 full 已完成并自动 eval。Full NPZ 已覆盖 smoke：main `1.1MB`，guard `1.2MB`；视频、keyframes、scene_snapshot、comparison、aggregate、`e014_soft_targets.json` 均已落盘。
- Final aggregate：`num_results=2`、`num_freejoint_oracle_config_ok=2`、`num_near_e081_obj_oracle=2`、`num_guard_stable=1`。
- Main `E013_box025_p2_obj_oracle`：obj `0.011/0.038m`、hand `78.6%`、floor `57.8%`、leg `0.0%`、xy ratio `1.000`、rot `1.99deg`、pelvis min `0.765m`。
- Guard `E013_box023_p2_obj_oracle`：obj `0.017/0.064m`、hand `72.7%`、floor `34.7%`、leg `0.0%`、xy ratio `1.000`、rot `3.05deg`、pelvis min `0.688m`。
- 已用 `video-frames` skill 的 `frame.sh` 从两个视频抽取关键帧并生成 contact sheet：`workspace/core4d_collab_retarget/results/E013/keyframes_skill/main_sheet.jpg`、`guard_sheet.jpg`。视觉观察：object 与 ref 基本重合，无 E012 式大旋转；main 手端接触持续性不足，guard 姿态稳定。
- 已写入 E013 结果日志：`workspace/core4d_collab_retarget/log/13_E013_true_freejoint_object_oracle_results.md`，并更新 `EXPERIMENT_TRACKER.md`。下一步按 `docs/03_agent_execution_plan_E013_E016.md` 创建 E014 COLA-B 位置约束计划。

## 2026-05-19 01:49 E014 计划启动

- 已写入 E014 中文计划：`workspace/core4d_collab_retarget/plan/14_E014_cola_b_kinematic_weld_plan.md`。
- 计划核心：不复用旧 `scene_weld/object_target` 的 COM zero-relpose object oracle，而是在 true-freejoint scene 中添加 `support_weld_anchor` mocap body，并用 object-local support point 作为 weld `relpose`。
- E014 support target 由 object ref pose 派生：main local point `[0.0, 0.38, 0.30]`，guard local point `[0.16, 0.0, 0.10]`；mocap quat 跟随 ref object quat，以测试 COLA-B 的 kinematic support + 6-DoF soft equality。
- 下一步实现 scene generator、`support_proxy_mocap_quat_mode=object_ref`、E014 overrides/train/eval，然后先跑 smoke。
- 已完成 E014 第一版实现：
  - `spider/config.py` 新增 `support_proxy_mocap_quat_mode`，默认 `identity`，E014 使用 `object_ref`；
  - `spider/simulators/mjwp.py` 在 `_load_support_proxy` 保存 object ref quat，并在 `_update_support_proxy_mocap_pad` 中按配置写入 mocap quat；
  - 新增 E014 variants、scene generator、override generator、preprocess、train、remote launch/pull、eval 脚本。
- E014 eval 会检查 freejoint parity、非 COM oracle scene、无 direct wrench、support gap、E013 soft target、lag-free 和 push-vs-carry gate。下一步运行静态检查和 `run_E014_preprocess.sh`。
- 静态检查通过：核心 Python 与 E014 generator/eval `py_compile`，E014 shell 脚本 `bash -n`，variants 每行 21 列。
- `run_E014_preprocess.sh` 已生成 3 个 E014 scene XML 和 6 个 overrides；MuJoCo 编译检查确认三个 scene 均保持 `nq=43`、`nv=41`、`nu=29`、`nmocap=1`。
- 人工检查主/guard scene：都含 `support_weld_anchor` 与 `e014_support_weld`；main relpose 为 `0 0.38 0.3 1 0 0 0`，guard relpose 为 `0.16 0 0.1 1 0 0 0`，未出现 `object_target`。
- E014 4-step smoke 已完成：6/6 变体产出 NPZ；显式 eval 通过，aggregate 为 `num_results=6`、`num_freejoint_parity_ok=6`、`num_anchor_not_com_oracle=6`、`num_no_direct_wrench=6`、`num_support_proxy_metrics_present=6`。smoke 只验证 wiring，不作为效果结论。
- 下一步提交 E014 setup 并启动 full：本地 `local_wave` 跑 main t02 + guard t02，远程两卡跑 t05 / t02_hc1 / g08 / guard_hc1。
- 已提交并推送 E014 setup：`ed14966 exp(core4d_collab_retarget): set up E014 soft weld`。
- 远程 `spider-remote` tmux session `E014` 已启动；远程预处理成功生成 3 个 scene XML 与 6 个 overrides。
- 本地 `local_wave` 已启动，当前运行 `E014_box025_p2_jointB_t02`，之后串行跑 `E014_box023_p2_jointB_t02`。
- 本地 `local_wave` 已完成并自动 eval。Full NPZ 已覆盖 smoke：main `1.2MB`、guard `1.4MB`。
- 本地 E014 t02 是强阳性：main obj `0.056/0.087m`、hand `86.7%`、floor `51.4%`、leg `0.0%`、xy ratio `0.999`、rot `2.2deg`、support gap mean/max `0.060/0.098m`，`E014_soft_target_pass=true`；guard obj `0.043/0.080m`、hand `74.7%`、floor `33.3%`、leg `0.0%`、pelvis min `0.689m`，`E014_guard_stable=true`。
- 远程第一批 main `t05` / `g08` 已完成并生成 1.2MB full NPZ；远程第二批 `t02_hc1` / `guard_hc1` 仍在运行。
- 远程 E014 已结束并回收；6 条 full 显式总评完成。Final aggregate：`num_results=6`、`num_freejoint_parity_ok=6`、`num_anchor_not_com_oracle=6`、`num_no_direct_wrench=6`、`num_support_proxy_metrics_present=6`、`num_main_soft_target_pass=4`、`num_main_lag_free=4`、`num_main_push_vs_carry_ok=4`、`num_guard_stable=2`，diagnostic 全部为 `soft_target_pass`。
- Main 全部通过 E013 soft target：`t02` obj `0.056/0.087m`、`t05` obj `0.082/0.142m`、`g08` obj `0.057/0.088m`、`hc1` obj `0.057/0.085m`；hand `85.5-90.8%`、floor `49.7-51.4%`、leg `0.0%`、xy ratio `0.995-0.999`、rot `1.6-2.2deg`。
- Guard 两条稳定：`t02` obj `0.043/0.080m`、`hc1` obj `0.042/0.080m`，floor `33.3%`，leg `0.0%`，pelvis min `0.689-0.691m`。
- 已用 `video-frames` skill / ffmpeg 生成并检查 E014 contact sheets：main object 与 ref 基本重合，无 E012 rotation shortcut；blue support anchor 在 object 侧面 offset，不是 COM；guard 姿态稳定，无摔倒/跪倒。
- 已写入 E014 结果日志：`workspace/core4d_collab_retarget/log/14_E014_cola_b_kinematic_weld_results.md`，并更新 `EXPERIMENT_TRACKER.md`。结论：COLA-B 成立，E014b 不触发；下一步按总路线进入 E015 dynamic support + PD 或 pipeline 对接验证。

## 2026-05-19 02:47 E015 计划启动

- 已按 E014 结论进入 E015，不触发 E014b，也不跳 E016。
- 已写入 E015 中文计划：`workspace/core4d_collab_retarget/plan/15_E015_cola_ab_dynamic_support_pd_plan.md`。
- E015 实现口径确定为 3 slide + 3 hinge 的 dynamic support body，而不是 support freejoint：模型维度预计 `nq=49/nv=47/nu=29`，support qpos/qvel 插入 robot 与 object 之间，object 仍保持最后 7 维，避免破坏现有 object-last eval/reward 口径。
- 下一步实现 E015 scene/data generator、`support_proxy_mode=dynamic_weld`、qfrc PD runtime、脚本和 eval。

## 2026-05-19 03:10 E015 实现继续

- 已恢复 E015 上下文并复查总路线、E014 结果日志和当前 runtime diff。E015 继续按计划推进 dynamic support + PD，不触发 E016。
- 当前未提交的相关改动包括 `spider/config.py` 的 dynamic support 配置字段，以及 `spider/simulators/mjwp.py` 的 `support_proxy_mode=dynamic_weld`、support 6-DoF joint 地址检查、reference qpos/qvel 加载和 `qfrc_applied` PD 写入逻辑。
- 下一步补齐 E015 variants、scene/data generator、override generator、preprocess/train/remote/eval 脚本，然后做静态检查与 smoke。

## 2026-05-19 03:25 E015 上下文恢复补记

- 已按用户要求重新读取 `EXPERIMENT_TRACKER.md`、`progress.md`、`docs/03_agent_execution_plan_E013_E016*.md`、E015 plan、E013/E014 log、`results/E013/e014_soft_targets.json`。
- 权威口径确认：E015 support 使用 3 slide + 3 hinge 标量 joints，目标模型维度 `nq=49/nv=47/nu=29`；qpos/qvel 布局必须为 `robot + support(6) + object`，object 保持最后 7 qpos；support 不新增 actuator、不进 CEM ctrl，PD 只通过 `qfrc_applied` 写 generalized force。
- 代码现状确认：`spider/config.py` 与 `spider/simulators/mjwp.py` 已有 dynamic support 字段和 `support_proxy_mode=dynamic_weld` 半成品；尚缺 E015 variants、scene/data generator、override/preprocess/train/remote/pull/eval 脚本与 smoke 验证。

## 2026-05-19 03:40 E015 脚本实现

- 已新增 E015 初始 4 variants：默认 main 本地跑，剩余两个 main 走远程 GPU0，guard 走远程 GPU1。
- 已新增 `generate_e015_assets.py`：基于原始 freejoint scene 插入 `support_dynamic_anchor` 3 slide + 3 hinge 标量 joints，并生成增广 `trajectory_kinematic.npz`，布局为 `qpos robot(36)+support(6)+object(7)`、`qvel robot(35)+support(6)+object(6)`。
- 已新增 `generate_e015_overrides.py`、`run_E015_preprocess.sh`、`train_E015.sh`、`train_E015_remote_tmux.sh`、`run_E015_remote.sh`、`pull_E015_remote_results.sh`、`eval_E015.py`。
- E015 eval 会显式检查 `nq/nv/nu=49/47/29`、object-last、support 非 mocap 6 joints、`dynamic_weld` 配置、无 object actuator / object kinematic / partner-force direct wrench，以及 support PD force/torque diagnostics。

## 2026-05-19 03:45 E015 preprocess / smoke

- 静态检查通过：`spider/config.py`、`spider/simulators/mjwp.py`、`examples/run_mjwp.py`、E015 assets/override/eval 的 `py_compile`，以及 E015 shell 脚本 `bash -n`。
- `run_E015_preprocess.sh` 已生成 4 个 dynamic scene、4 个增广 data NPZ、4 个 Hydra override。
- 独立 scene/data 检查通过：4/4 variant 均为 model `nq/nv/nu=49/47/29`，data `qpos/qvel/ctrl=(T,49)/(T,47)/(T,29)`；support 非 mocap、6 joints、q/d 地址 `36/35`；object q/d 地址 `42/41`，保持最后 7 qpos / 6 qvel。
- E015 4-step smoke 已完成两遍（第二遍验证 snapshot manifest），4/4 产出 NPZ；`eval_E015.py --all` aggregate: `num_freejoint_parity_ok=4`、`num_support_dynamic_scene_ok=4`、`num_object_last_ok=4`、`num_no_direct_wrench=4`、`num_pd_metrics_present=4`。4-step 的 target/guard 指标不作效果结论。
- NPZ diagnostics 已确认包含 `support_proxy_force`、`support_proxy_torque`、`support_proxy_pos`、`support_proxy_vel`、`support_point_pos`、`support_point_vel`、`support_proxy_ref_idx`。

## 2026-05-19 03:50 E015 setup commit / full 启动

- E015 setup 已提交并推送：`82663f2 exp(core4d_collab_retarget): set up E015 dynamic support`。
- 首次远程启动在 SSH 连接阶段超时；随后单独 SSH 重试成功，远端 hostname `embodied-2x6000Ada`，两张 `NVIDIA RTX 6000 Ada Generation` 可见。
- 本地 full 已启动：`E015_box025_p2_m2_kp500`。
- 远程 tmux `E015` 已启动：GPU0 队列 `E015_box025_p2_m1_kp500 -> E015_box025_p2_m2_kp1000`，GPU1 队列 `E015_box023_p2_m2_kp500`。
- 本地 runtime 日志确认 dynamic support 加载：`qadr=36`、`dadr=35`、ref `support_qpos=(298,6)`、`pos_kp=500`、auto `pos_kd=63.2455`、`rot_kp=80`、auto `rot_kd=2.5298`。

## 2026-05-19 04:00 E015 本地 default main 完成

- 本地 `E015_box025_p2_m2_kp500` full 已完成并自动 eval。
- 结果未过 soft target：case-window obj `0.311/0.418m`，hand `63.6%`，floor `43.4%`，leg `2.3%`，object xy ratio `1.197`，rot `50.3deg`。
- Support diagnostics：support-object weld gap 小（mean/max `0.016/0.068m`），但 support target lag 明显（mean/max `0.127/0.195m`），PD force max 打到 `250N` clamp，torque max 打到 `80Nm` clamp；`E015_effort_reasonable=false`，diagnostic=`dynamic_support_lag`。
- 初步判断：default dynamic support 没有复现 E014 的近刚性 tracking；失败主要是 dynamic support target lag + PD saturation，并伴随 object rotation shortcut。等待远程 m1/kp1000/guard 完成后统一决定是否触发 E015b effort/PD tuning。

## 2026-05-19 04:25 E015 full 完成

- 远程 E015 已完成并回收：`E015_box025_p2_m1_kp500`、`E015_box025_p2_m2_kp1000`、`E015_box023_p2_m2_kp500`。
- 全量 `eval_E015.py --all` 完成：`num_results=4`、`num_freejoint_parity_ok=4`、`num_support_dynamic_scene_ok=4`、`num_object_last_ok=4`、`num_no_direct_wrench=4`、`num_pd_metrics_present=4`、`num_main_soft_target_pass=0`、`num_main_effort_reasonable=0`、`num_numerical_instability=2`、`num_guard_stable=1`。
- Main 结果：default `m2_kp500` 是 dynamic support lag + clamp；`m1_kp500` 和 `m2_kp1000` 均出现 NaN 轨迹/黑帧，diagnostic=`numerical_instability`。
- Guard `box023_m2_kp500` 稳定，obj `0.097/0.239m`、pelvis min `0.682m`、force `50/154N`，但 hand contact `47.3% < 61.7%`，不通过 guard soft target。
- 已生成 `workspace/core4d_collab_retarget/results/E015/keyframes/E015_visual_montage.jpg` 和 `results/E015/visual_eval.md`；视觉结论与 numeric 一致：default main 大旋转，m1/kp1000 后段渲染失效，guard 稳但接触弱。
- 已写入 E015 结果日志：`workspace/core4d_collab_retarget/log/15_E015_cola_ab_dynamic_support_pd_results.md`，并更新 `EXPERIMENT_TRACKER.md`。结论：E015 差于 E014，按规则触发 E015b effort/PD tuning 分析，不跳 E016。

## 2026-05-19 13:35 E016 指标与泛化验证启动

- 已按用户要求重新回顾 E001-E015 路线：E014 B-only 是当前 work candidate，E015 dynamic support 失败在 lag/PD saturation/数值不稳。
- 已读取 SPIDER / DynaRetarget / OmniRetarget 本地论文文本与 holosoma v2 eval 代码，确定新增指标：object `E_pos/E_rot`、SPIDER/Dyna success、relative smoothness、penetration duration/depth、foot skating、contact preservation、carry progress 与 object xy/z/final error。
- 已写入 E016 计划：`workspace/core4d_collab_retarget/plan/16_E016_e014_paper_metrics_generalization_plan.md`。

## 2026-05-19 13:45 E016 实现与 preprocess

- 已新增 reusable paper-aligned metrics helper：`workspace/core4d_collab_retarget/scripts/eval/paper_metrics.py`，并接入 `eval_E014.py`。
- 已新增 E016 脚本组：variants、asset/override 生成、preprocess、train、remote/pull、`eval_E016.py`。
- 静态检查通过：E014/E016 eval、paper metrics、E016 asset/override generator 的 `py_compile`，以及 E016 shell 脚本 `bash -n`。
- `run_E016_preprocess.sh --force` 已成功生成 13 个 `{source}_freejoint_legobj_e016` 派生 task、13 个 `scene_e016_jointB_*` soft-weld scene、13 个 Hydra overrides 和 `results/E016/manifest.tsv`。
- 首次 preprocess 发现部分旧 case 没有 `task_info.json`；已把该文件改为可选复制，使用默认 ref_dt 继续。

## 2026-05-19 14:05 E016 继续执行

- 已重新读取 tracker、E016 plan、E014/E015 结果脉络和 progress，确认当前路线：E014 B-only 是 work candidate；E015 dynamic support 因 target lag / PD saturation / numerical instability 不继续作为候选。
- 下一步先修评测口径：E014 aggregate 需要纳入 `paper_*` 成功计数；OmniRetarget-style penetration proxy 需要新增深穿透阈值，避免把浅层手-物接触交叠直接等同为严重穿透。

## 2026-05-19 14:12 E016 评测/训练脚本修正

- `paper_metrics.py` 新增 `2cm` deep-penetration 阈值字段：保留原始 negative-SDF duration，同时增加 deep duration / hand-deep / leg-deep / deep-penetration-ok，用于区分浅手部接触交叠和严重穿透。
- `eval_E014.py` aggregate 已加入 paper-aligned 成功计数和均值；`eval_E016.py` 的 artifact gate 改为使用 deep-penetration duration。
- 修复 `train_E016.sh quick/remote_gpu*`：额外 Hydra 参数现在走 `RUN_EXTRA_ARGS`，不会再被误当成 variant；smoke 完成后也会自动跑 E016 eval。

## 2026-05-19 14:18 E014 paper metrics 复评

- 静态检查通过：`paper_metrics.py`、`eval_E014.py`、`eval_E016.py` 的 `py_compile`，E016 shell 脚本 `bash -n`。
- 已重新运行 `.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E014.py --all`。新增 aggregate：`num_paper_spider_success=6/6`、`num_paper_dynaretarget_success=6/6`、`num_transport_success=6/6`、`num_deep_penetration_ok=6/6`、`num_contact_preservation_ok=4/6`。
- E014 paper 均值：case-window object `Epos=0.05598m`、`Erot=2.16deg`、contact preservation 5cm mean `69.49%`、deep penetration duration mean `1.0%`。该结果支持 E014 作为 paper-aligned work candidate 进入 E016 泛化验证。

## 2026-05-19 14:25 E016 smoke 边界修复

- `bash workspace/core4d_collab_retarget/scripts/train/train_E016.sh smoke 0` 已完成 13/13 个 4-step rollout 并产出 NPZ，覆盖 box / bucket / desk 的 13 个派生 task。
- 自动 eval 在 smoke 阶段触发 foot-skating 边界条件：4-step 片段没有 stance velocity 样本时，`np.concatenate([])` 抛 `ValueError`。
- 已修复 `paper_metrics.py`：当 stance velocity 全空时返回空 `float64` 数组，foot-skating max velocity 记为 `0.0`。下一步直接重跑 `eval_E016.py --all`，不重复 smoke rollout。

## 2026-05-19 14:30 E016 smoke eval 完成

- 已重新运行 `eval_E016.py --all`，13/13 smoke 结果全部评估成功，`num_config_ok=13/13`、`num_paper_spider_success=13/13`、`num_paper_dynaretarget_success=13/13`。
- 该结果只说明 wiring 和 paper 字段完整；因为 smoke 只有 4 sim steps，transport 指标不可作为效果结论：aggregate 中 `num_transport_success=1/13`、`num_generalization_pass=0/13` 是预期的短片段偏差。
- 下一步启动 13 case quick 泛化，使用完整 case-window 但低 CEM 预算：`E016_QUICK_NUM_SAMPLES=128`、`E016_QUICK_MAX_ITERS=4`。

## 2026-05-19 14:35 E016 远程并行切换

- 用户明确允许使用 `spider-remote` 并行跑实验；已读取 `.codex/skills/experiment-planning-zh/remote-execution.md`，远程路径为 `/home/xiayb/pHRI_workspace/spider`，2 张 RTX 6000 Ada。
- 本地 quick 串行已跑完 `E016_box021_p1` full-size NPZ，并开始 `E016_box021_p2`；为避免和远程队列重复，已停止本地 `train_E016.sh quick` 及其 `run_mjwp.py` 子进程。
- 下一步只提交/推送 E016 相关代码与配置，远程运行 `run_E016_preprocess.sh` 后并行执行 `remote_gpu0` / `remote_gpu1` 队列；本地可独立跑 `local` 队列两个 case。

## 2026-05-19 14:42 E016 远程脚本加固

- 已检查远端：`spider-remote` 当前在 `exp/core4d-collab-retarget`，代码落后本地，但 E079/E080 contact masks 存在，可支持远端 preprocess。
- `train_E016_remote_tmux.sh` 改为远端启动时执行 `run_E016_preprocess.sh --force`，避免 git 跟踪的 scene XML 快照与生成目录半成品冲突。
- `train_E016.sh` 新增 `local_quick` 模式，用本地 GPU 只跑 manifest 中 `queue=local` 的 2 个 case，并复用 quick 的 `num_samples/max_num_iterations/save_video/viewer` 配置。

## 2026-05-19 14:48 E016 代码同步

- 已提交并推送 E016 paper metrics + 13 case 泛化脚本与 scene XML 快照：`d14f883 exp(core4d_collab_retarget): add E016 paper metrics generalization`。
- 又补充提交远程 launcher 日志目录加固：`5bc2642 exp(core4d_collab_retarget): harden E016 remote launch`。
- 远程可通过 `run_E016_remote.sh` 拉到最新 `5bc2642` 并启动 tmux；本地计划同时运行 `train_E016.sh local_quick 0` 覆盖 `queue=local` 两个 case。

## 2026-05-19 15:02 E016 远端数据缺失

- 远端首次 `E016` tmux 很快退出，`remote_tmux.log` 显示 preprocess 在 `box021_person1/0/trajectory_kinematic.npz` 缺失处失败。
- 判断：远端已有 E079/E080 contact masks，但不是所有 13 个 source task 的 `scene.xml + 0/trajectory_kinematic.npz` 都存在；这些数据不走 git，需要显式同步。
- 下一步用 `rsync` 将 13 个 source task 目录从本地同步到 `/home/xiayb/pHRI_workspace/spider/example_datasets/processed/core4d/unitree_g1/humanoid_object/`，再重启远端 E016。

## 2026-05-19 15:05 E016 并行运行中

- 已用 `rsync` 同步 13 个 source task 到远端；第一次 while+ssh 同步只处理了首个 task，已改为 `mapfile` 数组后重跑，13 个 task 均完成同步。
- 已重启远端 `E016` tmux；远端 preprocess 已通过并开始两卡 quick：GPU0 从 `E016_box021_p1` 开始，GPU1 从 `E016_box021_p2` 开始。
- 本地 `local_quick` 已完成 `E016_box023_p2` full-size quick（final object tracking pos `0.0264m`），当前运行 `E016_box025_p2`。

## 2026-05-19 15:08 E016 local quick 完成

- 本地 `local_quick` 两个 case 已完成：`E016_box023_p2`、`E016_box025_p2`，NPZ 分别约 `1.4MB` / `1.2MB`。
- 本地 2-case eval aggregate：`num_config_ok=2/2`、`num_paper_spider_success=2/2`、`num_paper_dynaretarget_success=2/2`、`num_transport_success=2/2`、`num_deep_penetration_ok=2/2`、`num_contact_preservation_ok=0/2`、`num_generalization_pass=0/2`。
- 本地诊断均为 `contact_preservation_gap`，说明 E014-B 结构和 object tracking 在这两例成立，但 quick CEM/robot-side 接触没有复现 contact mask。

## 2026-05-19 15:12 E016 三卡并行重分配

- E016 总计 13 个 case；已完成/本地负责 `box023_p2`、`box025_p2` 两个 local case，远端已完成 `box021_p1`、`box021_p2`。
- 根据用户提醒，本地 1 卡不应空闲。已生成 `results/E016/manifest_local_extra.tsv`，本地继续接手远端队列后半段 3 个 case：`E016_bucket005_s2_p2`、`E016_bucket007_p2`、`E016_desk021_p1`。
- 当前并行分配：本地 GPU0 跑 `E016_bucket005_s2_p2`；远端 GPU0 跑 `E016_box023_p1`；远端 GPU1 跑 `E016_bucket001_p1`。后续需在远端完成非本地接手的 8 个结果后停止远端，避免重复跑本地接手的 3 个 case。

## 2026-05-19 15:16 E016 远端所需结果完成

- 本地 extra 已完成 `E016_bucket005_s2_p2` 与 `E016_bucket007_p2`，当前本地最后跑 `E016_desk021_p1`。
- 远端已完成 8 个需远端负责的结果：`box021_p1`、`box021_p2`、`box023_p1`、`box025_p1`、`bucket001_p1`、`bucket001_p2`、`bucket005_s2_p1`、`bucket007_p1`。
- 已停止远端 `E016` tmux，防止继续重复跑本地接手的 `bucket005_s2_p2` / `bucket007_p2` / `desk021_p1`。待本地最后一个完成后回收远端结果并统一评估 13 case。

## 2026-05-19 15:19 E016 本地 extra 完成

- 本地 extra 三个 case 已完成：`bucket005_s2_p2`、`bucket007_p2`、`desk021_p1`。
- 本地 extra 3-case eval：`num_config_ok=3/3`、`num_paper_spider_success=3/3`、`num_paper_dynaretarget_success=3/3`、`num_transport_success=3/3`、`num_contact_preservation_ok=1/3`、`num_deep_penetration_ok=2/3`、`num_generalization_pass=0/3`。
- 诊断：`contact_preservation_gap=2`、`artifact_failed=1`。下一步拉回远端 8 个结果并做 13-case aggregate。

## 2026-05-19 16:40 E016 完成

- 已回收远端 8 个结果，本地 E016 目录中 13 个 NPZ 均为 full-size quick 文件（约 `740K-1.5M`），无 smoke 占位。
- 13-case `eval_E016.py --all` 完成：`num_config_ok=13/13`、`num_paper_spider_success=13/13`、`num_paper_dynaretarget_success=13/13`、`num_transport_success=13/13`、`num_contact_preservation_ok=3/13`、`num_deep_penetration_ok=9/13`、`num_generalization_pass=0/13`。
- 均值：object Epos `0.04985m`、Erot `4.08deg`、carry progress ratio `1.001`、contact preservation 5cm `39.62%`、deep penetration duration `21.77%`。
- 诊断分布：`contact_preservation_gap=10`、`push_or_leg_shortcut=2`、`artifact_failed=1`。E014-B object-side 泛化成立，但完整 retargeting 泛化未过 OmniRetarget-style robot-side artifact gate。
- 已生成 3 个代表性离线可视化：`E016_box025_p2`、`E016_box021_p1`、`E016_bucket005_s2_p2`；已写入 E016 结果日志并更新 `EXPERIMENT_TRACKER.md`。

## 2026-05-19 17:20 E016 全量可视化补齐

- 根据用户要求补齐可视化结果，新增复现脚本 `workspace/core4d_collab_retarget/scripts/eval/render_E016_visuals.py`，封装 `workspace/hdmi_reproduce/scripts/render_trajectory_video.py`，从 E016 manifest 自动解析 scene/kin/phys 路径。
- 已用离线 EGL 渲染 13/13 个 side-by-side comparison mp4，并用 ffmpeg 生成 13 张 contact sheet；索引文件为 `workspace/core4d_collab_retarget/results/E016/visual/visual_eval.md`。
- 可视化结论与量化一致：object tracking / transport 在全量 case 上成立，但 robot-side 接触保持、腿/身体 shortcut 和 deep penetration 是主要失败源；已更新 E016 结果日志中的可视化章节。

## 2026-05-19 18:05 E016 可视化口径修正

- 用户指出上一版离线可视化不对。已确认原因：`workspace/hdmi_reproduce/scripts/render_trajectory_video.py` 是 qpos-only replay，不会恢复 E014/E016 moving mocap support weld 的 `support_weld_anchor`，因此不适合作为 E016 证据。
- 已修正 `workspace/core4d_collab_retarget/scripts/eval/render_E016_visuals.py`：直接用 MuJoCo front camera replay `qpos`，ref/sim 标签对齐 `run_mjwp.py`，并把 NPZ 中的 `support_proxy_pos` 写回 mocap anchor；已重刷 13/13 个 E016 视频和 sheet。
- 已对照 E014 两个基准配置：`box025_p2` 用 `E014_box025_p2_jointB_t02`，anchor `[0,0.38,0.30]`；`box023_p2` 用 `E014_box023_p2_jointB_t02`，anchor `[0.16,0,0.10]`。E016 对应 anchor 分别是 `[0.006,0.378,0.399]` 和 `[0.030,0.157,0.150]`，即继承了 E014 weld 结构/solref/solimp，但没有复用 E014 的手工 anchor。
- 视觉对比更新：`box025_p2` corrected render 与 E014 t02 接近；`box023_p2` corrected render 明显差于 E014 t02，后段接触和姿态不稳，指向 E016 自动 anchor 泛化策略问题。

## 2026-05-19 18:20 E016 视频参数对齐 E014

- 按用户要求检查 E014 原视频：`E014_box025_p2_jointB_t02.mp4` 为 `1440x480 @ 50fps / 248` 帧，`E014_box023_p2_jointB_t02.mp4` 为 `1440x480 @ 50fps / 272` 帧。
- 已将 `render_E016_visuals.py` 默认参数改为单侧 `720x480`、输出 `1440x480`、`50fps`，并展开 MJWP 保存的 `(T,2,nq)` sim substeps；reference qpos 使用与 `spider.io.load_data` 一致的 `interp` 上采样到相同帧数。
- 已重刷全量 13 个 E016 视频；批量 `ffprobe` 验证 13/13 均为 `1440x480 @ 50fps`。其中 `E016_box025_p2` 为 248 帧、`E016_box023_p2` 为 272 帧，与 E014 对照 case 帧数一致。
