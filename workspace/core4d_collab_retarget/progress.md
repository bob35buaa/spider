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
