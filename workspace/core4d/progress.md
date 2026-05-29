# E088 Progress — 2026-05-28

## 当前状态: E088 full CEM 已完成；三组 main gate 全失败，不建议接 RL

## 完成步骤

- [x] 按 `experiment-planning-zh` 恢复 E087 结果上下文，确认当前问题不是继续小幅调 `robot_object_penalty_scale`，而是 CEM elite selection 仍允许穿箱 sample 更新分布。
- [x] 检查 CEM 入口：
  - `spider/optimizers/sampling.py` 的 `_compute_weights_impl()` 当前只基于 scalar reward 做 top-k softmax。
  - `spider/optimizers/sampling_fast.py` 有并行的 fast/select-best 路径，也需要同步处理。
  - 因此 E088 的 hard gate 应该显式传递 `sample_gate_valid_mask`，而不是只把 invalid reward 设成 `-inf`。
- [x] 写入详细计划：`workspace/core4d/plan/94_E088_hard_safety_gate_lift_floor_plan.md`。
- [x] 实现第一版 E088 机制改动：
  - `spider/config.py` 新增默认关闭的 `cem_safety_gate_*` 与 `object_clearance_*` 配置，并解析 upper-body collision geoms。
  - `spider/simulators/mjwp.py` 提取 object-box SDF helper，输出 `cem_gate_min_sdf / violation / depth`，并增加绝对 `object_clearance_rew / penalty / m`。
  - `spider/optimizers/sampling.py` 与 `sampling_fast.py` 在 rollout 后聚合 sample-level gate 指标，并在 elite selection 前应用 valid mask；valid 样本不足时使用 least-violation fallback。
  - `py_compile` 已覆盖 `spider/config.py`、`spider/simulators/mjwp.py`、`spider/optimizers/sampling.py`、`spider/optimizers/sampling_fast.py`。
- [x] 补齐 E088 实验封装：
  - `workspace/core4d/scripts/E088/variants.tsv`
  - `workspace/core4d/scripts/E088/generate_e088_overrides.py`
  - `workspace/core4d/scripts/train/train_E088.sh`
  - `workspace/core4d/scripts/E088/run_remote_inside.sh`
  - `workspace/core4d/scripts/run_E088_remote.sh`
  - `workspace/core4d/scripts/pull_E088_remote_results.sh`
  - `workspace/core4d/scripts/eval/eval_E088.py`
  - overrides: `core4d_E088A_m10_gate_main.yaml`, `core4d_E088B_m10_gate_low_main.yaml`, `core4d_E088C_m10_gate_clearance_main.yaml`
- [x] 修复 `examples/run_mjwp.py` 信息聚合：旧逻辑只按第一个 tick 的 keys 保存，导致 warmup tick 后出现的 reward/gate info 被丢弃；现在改为 union-of-keys，缺失 tick 用零补齐。
- [x] 静态检查通过：
  - `py_compile` 覆盖 `examples/run_mjwp.py`、E088 eval/generator 和 E088 修改的核心模块。
  - `bash -n` 覆盖 E088 train/remote/pull 脚本。
  - `git diff --check` 覆盖 E088 相关代码和脚本。
- [x] 本地 smoke：
  - 4-step smoke 三个 variant 均通过，Hydra 新字段和数据加载正常；`CEM safety gate: 7 geoms resolved`。
  - 24-step E088C smoke 进入一次 CEM，npz 已包含 `cem_gate_*`、`sample_gate_*`、`object_clearance_*` keys。
  - 24-step smoke 也暴露严格 gate 早期 `cem_gate_valid_frac=0`、`fallback_used=1`，说明 full run 要重点看 fallback 是否长期占主导。
- [x] 远程多卡 full CEM 已启动：
  - remote host: `spider-remote`
  - remote repo: `/home/xiayb/pHRI_workspace/spider`
  - tmux session: `E088_gate_clearance`
  - GPU0: `E088A_m10_gate_main`
  - GPU1: `E088B_m10_gate_low_main`，完成后串行 `E088C_m10_gate_clearance_main`
  - 15:44 检查：两张 RTX 6000 Ada 均有负载，A/B 已进入 CEM，单 tick `opt_steps=32`，plan time 约 `10s`。
- [x] 远程 full CEM 已完成并回收本地，合并评估、gate summary、reward breakdown、keyframes 均已生成。
- [x] E088A/B/C 三组均未通过 gate，`accepted_variants=[]`：
  - E088A hard gate only: contact `82.17%`，obj mean/max `0.695/1.148m`，pelvis min `0.181m`，head/upper penetration `27.91/55.04%`，fallback `90.71%`。
  - E088B low-contact/object: contact `70.54%`，obj mean/max `0.722/1.198m`，pelvis min `0.643m`，head/upper penetration `71.32/75.19%`，valid frac `0`。
  - E088C + absolute clearance: contact `73.64%`，obj mean/max `0.612/0.985m`，head/upper penetration `19.38/80.62%`，LH floor `40.31%`；clearance 有效但视觉为翻箱/侧倒。
- [x] 已写入正式结果日志：`workspace/core4d/log/110_E088_hard_safety_gate_lift_floor_results.md`；已更新 `EXPERIMENT_TRACKER.md`。

## 待完成

- [x] 实现默认关闭的 `cem_safety_gate_*` 与 `object_clearance_*` config。
- [x] 提取 upper-body/object SDF 计算，给 rollout 输出 sample 级 violation 指标。
- [x] 修改 normal/fast CEM elite selection，加入 valid-mask top-k 与 fallback。
- [x] 增加绝对 object bottom clearance reward/penalty。
- [x] 生成 E088 override/train/eval 脚本，先跑 main case smoke，再决定是否进入 full CEM。
- [x] 启动远程多卡 full CEM：GPU0 跑 E088A，GPU1 串行跑 E088B/E088C。
- [x] 回收远程结果，运行 E088 merged eval，生成 comparison/gate summary/keyframes。
- [x] 写 E088 结果 log 并更新 tracker。

## 关键解释

Hard gate 的作用位置是 CEM top-k 之前：head/torso/pelvis/shoulder/elbow 穿箱超阈值的 sample 不允许成为 elite。它不是继续加大 soft penalty。

绝对 clearance reward 的作用是直接约束 `object_collision` 底面离地高度，例如 `bottom_clearance >= 4cm`，避免当前“相对 ref bottom”在 main 上几乎没有贡献。

---

# E085 Progress — 2026-05-28

## 当前状态: raw-contact target 修复已完成预处理，准备接入 MJWP smoke gate

## 完成步骤

- [x] 按 `experiment-planning-zh` 恢复实验上下文，当前目标是沿 E084 语义审计继续推进 E085，先修复数据层 contact target，再决定是否进入 CEM。
- [x] 写入计划：`workspace/core4d/plan/91_E085_raw_contact_target_repair_plan.md`。
- [x] 在 `spider/config.py` / `examples/run_mjwp.py` 增加默认关闭的 external contact target 接口：
  - 默认 `contact_hdmi_target_source=ref_fk`，旧实验行为不变。
  - `contact_hdmi_target_source=external` 时从 `.npz` 读取 object-local `(T,2,3)` target，并按 `qpos_ref` 长度 resize。
- [x] 新增 raw contact target 预处理：
  - `workspace/core4d/scripts/E085/generate_raw_contact_targets.py`
  - `workspace/core4d/scripts/E085/target_cases.tsv`
  - `workspace/core4d/scripts/run_E085_preprocess.sh`
- [x] 预处理已完成并输出：
  - main: `workspace/core4d/results/E085/raw_targets/E085A_rawtarget_main/raw_contact_targets.{npz,csv}`。
  - guard: `workspace/core4d/results/E085/raw_targets/E085A_rawtarget_guard/raw_contact_targets.{npz,csv}`。
- [x] 关键诊断结论：
  - `d003_box021_20231018_029_p2` 旧 G1 wrist pseudo target 与 raw contact surface 的 active-frame mean delta：left `27.20cm`，right `26.98cm`。
  - `box023_person2` guard 的对应 delta：left `24.64cm`，right `27.82cm`。
  - 因此“左手目标像在箱体下沿/底面附近”主要不是 mask 选错，而是旧 target 由 retargeted G1 wrist 反推，空间语义和 raw 手-箱表面接触点不一致。
- [x] 修正 target 输出策略：MJWP reward 使用投影到 collision box 表面的 semantic raw target，另存 raw visual mesh target 仅用于诊断，避免 visual mesh 和 collision box 尺寸/边界不一致。
- [x] 补齐 E085 执行脚本与评估入口：
  - `workspace/core4d/scripts/E085/variants.tsv`
  - `workspace/core4d/scripts/E085/generate_e085_overrides.py`
  - `workspace/core4d/scripts/train/train_E085.sh`
  - `workspace/core4d/scripts/run_E085_remote.sh`
  - `workspace/core4d/scripts/pull_E085_remote_results.sh`
  - `workspace/core4d/scripts/eval/eval_E085.py`
  - `workspace/core4d/scripts/eval/extract_E085_contact_sheets.sh`
- [x] 生成 E085 override 并完成 Hydra compose 检查：
  - `core4d_E085A_rawtarget_main`: 继承 E084C main，`contact_hdmi_target_source=external`，raw target 文件存在。
  - `core4d_E085A_rawtarget_guard`: 继承 E084C guard，`contact_hdmi_target_source=external`，raw target 文件存在。
- [x] 静态检查通过：`py_compile` 覆盖 E085 generator/eval 与 `run_mjwp.py`/`config.py`；`bash -n` 覆盖 E085 train/remote/pull/eval/preprocess shell。
- [x] 4-step smoke 通过：
  - main external target 加载日志：`eval_contact_target_object_local len 125→200`，dynamic target shape `(200,2,3)`。
  - guard external target 加载日志：`eval_contact_target_object_local len 227→322`，dynamic target shape `(322,2,3)`。
- [x] 修复 raw visual target 到 collision box 的投影策略：
  - 旧逻辑用 `abs(local)/half` 最大轴选 face；visual 点在 collision box 内部时会误把离 `+y` 侧面更近的点投到 `-z` 面。
  - 新逻辑：内部点选最近 surface margin，外部点选最大越界轴。
  - 重新预处理后，main left face 从 `+x:1/+y:15/-z:35` 变为 `+x:1/+y:35/-z:15`，mean vfrac 从 `0.037` 提到 `0.057`。
  - 但 main left target 仍然偏低，说明除了投影 bug，raw SMPL-X/object 接触几何本身也在低侧区域；需要继续做 fingertip-vs-broad target 选择审计。
- [x] nearest-face target 版本的 4-step smoke 已重跑通过，external target 加载日志仍正常。
- [x] 新增并运行 target selection 审计：`workspace/core4d/scripts/E085/audit_target_selection.py`。
  - 输出：`workspace/core4d/results/E085/target_selection_audit/`。
  - main left 对比：`broad_projected` vfrac mean/median `0.057/0.060`，`tip_best_projected` `0.052/0.044`，`tip_mean_projected` `0.063/0.068`，`high_close_projected` `0.103/0.107`。
  - main left fingertip 平均每 active frame 有 `4.55/5` 个 fingertip 在 3cm 内，tip min dist mean `0.76cm`；因此 contact mask 不是主要错误，fingertip target 也没有回到侧面中部。
  - 结论：旧 G1 pseudo target 的 27cm 偏移是方法 target 错；nearest-face projection bug 会额外制造下沿感；修复后 raw target 仍偏低，属于 raw 几何/人手姿态与 G1 可达性的形态差异，需用 CEM 实测是否能承受，否则转 support-body/COLA seed。
- [x] 启动 E085 CEM：
  - 远程 guard：`spider-remote` tmux session `E085_rawtarget_gate`，运行 `E085A_rawtarget_guard`。
  - 本地 main：`bash workspace/core4d/scripts/train/train_E085.sh local 0`。
- [x] 本地 main 已完成并自动评估，gate 未通过：
  - `workspace/core4d/results/E085/E085A_rawtarget_main.{npz,mp4}`。
  - `workspace/core4d/results/E085/comparison.csv`。
  - `workspace/core4d/results/E085/keyframes/contact_sheets/E085A_rawtarget_main_sheet.jpg`。
  - case-window contact `82.95%`，obj err mean `0.665m`，pelvis z min `0.659m`，hand-floor `0%/0%`。
  - 失败项：head-object penetration `18.60%`，upperbody-object penetration `53.49%`，first head penetration frame `35`。
  - 视觉观察：sim 没有手撑地/摔倒，但为了贴低位 contact target，长期弯腰压箱，头/右肩/双肘和箱体干涉；这支持“低位 raw hand target 对 G1 形态不可达或 reward 会利用穿透接触”的判断。
- [x] 写入下一轮迭代计划：`workspace/core4d/plan/92_E086_rawtarget_failure_iteration_plan.md`。
  - E086A：raw target + strict upperbody / hand deep penetration penalty。
  - E086B：raw target with minimum vertical fraction，验证低位 target 是否是形态瓶颈。
  - E086C：support-body/COLA seed route。
- [x] 实现 E086A 最小改动：
  - `spider/config.py`: 新增 `hand_object_deep_penalty_*` 配置，并解析 `lh/rh` geom。
  - `spider/simulators/mjwp.py`: 新增 hand-object deep penetration penalty，只惩罚 SDF 小于 `-threshold` 的深穿透，不惩罚正常接触附近。
  - `examples/config/override/core4d_E086A_rawtarget_strict_main.yaml`: E085 raw target + contact gain `5→3` + upperbody penalty `2→8` + hand deep penalty scale `10`。
  - `workspace/core4d/scripts/train/train_E086.sh` / `workspace/core4d/scripts/eval/eval_E086.py` / `workspace/core4d/scripts/E086/variants.tsv`。
- [x] E086A 静态/Hydra/smoke 通过：hand deep penalty 解析 `2` 个 geom，external target 加载 `125→200`，4-step smoke 无异常。
- [x] E086A main full 完成，gate 未通过且比 E085 更差：
  - contact `82.95%→76.74%`，obj err mean `0.665→0.698m`。
  - head-object penetration `18.60%→58.91%`。
  - upperbody-object penetration `53.49%→72.87%`。
  - left hand penetration `73.64%→35.66%` 下降，但 right hand penetration `54.26%→65.12%` 上升，right hand floor contact `0%→10.85%`。
  - 视觉观察：strict penalty 没有让姿态站起来，反而继续用头/身体压箱并翻箱；单纯堵手部深穿透不是解法。
- [x] 实现并生成 E086B target vertical-floor 变体：
  - `workspace/core4d/scripts/E086/make_vfrac_floor_target.py`。
  - 输出：`workspace/core4d/results/E086/vfrac_floor_targets/E086B_vfrac_floor_main/raw_contact_targets.npz`。
  - main left active target vfrac mean/min/max 约 `0.198/0.182/0.208`，相比 E085 `0.057/0.005/0.105` 明显抬高。
  - `core4d_E086B_rawtarget_vfloor_main` Hydra 检查和 4-step smoke 通过。
- [x] E086B full 完成，gate 未通过：
  - contact `75.19%`，obj err mean `0.710m`。
  - head-object penetration `62.79%`，upperbody-object penetration `68.99%`。
  - hand penetration left/right `58.91%/62.79%`，hand-floor `0%/0%`。
  - 视觉观察：左手 target 抬高后仍没有恢复 upright carry，sim 继续用头/胸前倾压箱；低位 target 是因素之一，但不是单独充分解释，当前 reward/optimization 会优先找“压箱支撑”局部最优。

## 待完成

- [ ] 若 smoke/gate 通过，再进入本地+远程 CEM；若失败，先分析视频/关键帧，不重复跑同配置。
- [ ] 等远程 guard 完成，pull 回本地并合并 eval。
- [ ] E085 remote guard 完成后回收并写 E085/E086 结果日志。
- [ ] 下一步不继续手调 target/penalty，转 E086C support-body/COLA seed 方案。

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|------|---------|----------|
| 初版 raw target 直接落在 visual mesh surface，和 MJWP collision box surface 有尺寸差 | 1 | 将 semantic raw visual contact point 投影到 collision box surface，MJWP 使用 projected target；CSV/NPZ 同时保留 visual target 诊断字段 |

---

# E078 Progress — 2026-05-15

## 当前状态: E078 已完成，结果日志已写入

## 完成步骤

- [x] 将 E077 验证成功的数据处理 pipeline 固化到本地文档与批处理入口：
  - `workspace/core4d/data_preprocess/README.md`
  - `workspace/core4d/data_preprocess/pipeline.sh`
  - `workspace/core4d/data_preprocess/cases_box023.tsv`
  - `workspace/core4d/data_preprocess/verify_processed_case.py`
- [x] `pipeline.sh --dry-run` 通过，box023 示例会串起 contact mask、Holosoma convert/retarget、trim、SPIDER case 生成、scene_act 和 verify。
- [x] `verify_processed_case.py` 已用 E077 现有 `box023_person2` 输出验证通过：qpos `(136,43)`，scene/scene_act `nq=43/42`，trimmed qpos 与 SPIDER trajectory 完全匹配。
- [x] `README.md` 补充 raw、OmniRetarget/Holosoma、SPIDER、eval 与 MJWP runtime reference 的 FPS/时间轴说明；`pipeline.sh` 显式支持 `REF_FPS`/`EVAL_FPS` 覆盖。
- [x] 新增 `workspace/core4d/data_preprocess/SCENE_TEMPLATE_GUIDE.md`，记录没有 `source_scene_task` 时如何制作 SPIDER `scene.xml` 模板、需要修改的字段、依据、校验步骤和常见错误。
- [x] 修正预处理 pipeline 的 ref 来源口径：SPIDER CORE4D reference 默认来自 Holosoma pipeline 的 `trim_no_contact.py` 输出；移除通用路径下的固定窗口 trim，新增 `infer_holosoma_trim_window.py` 用于反查/核验 Holosoma trimmed window；E077 固定 trim 仅保留在 E077 目录作为历史复现脚本。
- [x] 清理 `data_preprocess` pipeline 的机器相关硬编码：Holosoma/CORE4D/SMPL-X 保持为可覆盖的外部绝对路径配置；项目内路径保持相对 `REPO`；README 同步跨机器运行约定。
- [x] 按本机使用习惯恢复 `pipeline.sh` 的本机默认外部路径；换机器仍可通过 `HOLOSOMA_DIR`、`CORE4D_REAL_ROOT`、`SMPLX_MODEL_DIR` 覆盖。

- [x] 按 `experiment-planning-zh` 恢复实验上下文，确认最新计划为 `workspace/core4d/plan/83_E078_3cm_per_eef_contact_mask_cem_plan.md`。
- [x] 在 `spider/config.py` 增加默认关闭的 3cm contact mask source 配置项，旧实验默认仍走 `rotated_sdf`。
- [x] 修改 `examples/run_mjwp.py`，支持从 E077 `raw_contact_mask_3cm.npz` 读取 `(T, person, hand)` mask，并转换为 HDMI-style per-EEF `(T,2)` gating。
- [x] 修改 `spider/simulators/mjwp.py`，使 `contact_hdmi_rew` 支持 scalar、legacy `(N,)` 和 per-EEF `(N,2)` mask；`hold_contact` 只用 per-EEF mask 的 max 作为旧式 ref gate，避免形状污染。
- [x] 新增 E078 override：
  - `examples/config/override/core4d_e078a_box023_p1_3cm.yaml`
  - `examples/config/override/core4d_e078b_box023_p2_3cm.yaml`
- [x] 新增 E078 train/eval/remote/pull 脚本：
  - `workspace/core4d/scripts/train/train_E078.sh`
  - `workspace/core4d/scripts/eval/eval_E078.py`
  - `workspace/core4d/scripts/run_E078_remote.sh`
  - `workspace/core4d/scripts/pull_E078_remote_results.sh`

## 待完成

- [x] 静态验证通过：`py_compile` 覆盖 `spider/config.py`、`spider/simulators/mjwp.py`、`examples/run_mjwp.py`、`eval_E078.py`；`bash -n` 覆盖 E078 train/remote/pull。
- [x] Hydra/_build_config 验证：
  - E078A: `task=box023_person1`, `mask_person_idx=0`, E077 mask keys `(178,2,2)/(136,2,2)/(227,2,2)`。
  - E078B: `task=box023_person2`, `mask_person_idx=1`, `data_path/model_path` 均存在。
- [x] 本地短 horizon GPU smoke test 通过：
  - E078A: RTX 5090, `max_sim_steps=4`, 3cm mask 选 `eval_contact_mask_3cm`, `person_idx=0`, `len 227→322`, active L/R=46.3%/45.0%。
  - E078B: RTX 5090, `max_sim_steps=4`, 3cm mask 选 `eval_contact_mask_3cm`, `person_idx=1`, `len 227→322`, active L/R=45.0%/47.5%。
  - 两者均生成 `/tmp/e078_smoke_{a,b}/trajectory_mjwp_act.npz`，未触发 reward mask shape error。
- [x] 强制纳入远程必需数据：E077 3cm mask 与 `box023_person2` SPIDER case；未纳入 `.codex/config.toml` / `__pycache__`。
- [x] commit + push 后启动远程 E078A/E078B 并行：
  - Commit: `ddb2e69 exp(core4d): add E078 3cm contact mask sweep`
  - Remote: `spider-remote:/home/xiayb/pHRI_workspace/spider`
  - Session: `tmux E078`
  - E078A -> GPU0, PID 1387310；E078B -> GPU1, PID 1387311。
  - 15:40 初始结果计数为 `0` 个 `.npz`；远程 `nvidia-smi` 显示两张 RTX 6000 Ada 均有计算负载。
- [x] 2026-05-15 17:08: E078B 远程完成并已 scp 回收；本地 eval 跳过缺失 E078A，仅生成 E078B summary。
  - E078B npz/video: `workspace/core4d/results/E078/E078B_box023_p2.{npz,mp4}`。
  - E078B 初步数值: yaw 0.041/0.126deg, B1=0.081m, post2 contact=65.4%, post2 obj_err max/mean=0.308/0.156m, pelvis_z_min=0.700m, first robot ctrl Linf>0.5 at f140。
  - E078B f119-f125 right foot XY step sum sim/ref = 0.0219/0.0255m，right hip pitch ctrl diff abs max=0.138rad。
- [x] 远程 E078A 因 GPU0 被其他 `python` 同时占用，后段单步出现 57s/87s/80s；已停止远程 E078A 与 `tmux E078`。
- [x] 本机 RTX 5090 重新运行 E078A，17:29 完成并落盘：
  - E078A npz/video: `workspace/core4d/results/E078/E078A_box023_p1.{npz,mp4}`。
  - 统一评估已重跑：`logs/E078/eval_E078_after_local_A.log`，`workspace/core4d/results/E078/comparison.csv` 现包含 E078A+E078B。
  - E078A 初步数值: yaw 0.574/1.075deg, B1=0.084m, post2 contact=61.7%, post2 obj_err max/mean=0.287/0.160m, pelvis_z_min=0.663m, first robot ctrl Linf>0.5 at f116。
  - E078A f119-f125 right foot XY step sum sim/ref = 0.531/0.159m，right hip pitch ctrl diff abs max=0.554rad；p1 右腿相位偏差没有被 3cm per-EEF mask 解决。
  - E078B 对照: post2 contact=65.4%, first robot ctrl Linf>0.5 at f140, f119-f125 right foot XY step sum sim/ref = 0.0219/0.0255m。
- [x] 写入正式结果日志：`workspace/core4d/log/99_E078_3cm_per_eef_mask_results.md`。
- [x] 更新 `EXPERIMENT_TRACKER.md` E078 行。

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|------|---------|----------|
| 沙箱内 `uv run` 无法访问 CUDA，报 `No CUDA GPUs are available` | 1 | 使用批准的 escalated GPU smoke run；`nvidia-smi` 与 Warp 均确认 RTX 5090 可用 |
| `git diff --cached --check` 报 E077 CSV CRLF / E076 log trailing whitespace | 1 | 转为 LF 并移除末尾空格后通过 |
| sandbox 内首次 `run_E078_remote.sh` 的 `git push` DNS 失败 | 1 | 用已批准的 escalated 远程脚本重试；远程 fast-forward 并启动 tmux |
| 远程 `git pull` 提示 `trajectory_kinematic.npz` should have been LFS pointer | 1 | 文件仅 132KB 且已成功到远程，先不阻塞 E078；后续可统一整理 LFS 策略 |

---

# E075 Progress — 2026-05-14

## 当前状态: Plan 已写入，开始 E075 限时/弱化 hold_contact 远程并行实现

## 完成步骤

- [x] 按 `experiment-planning-zh` 恢复 `EXPERIMENT_TRACKER.md`、最新 plan/log、`progress.md`。
- [x] 读取远程执行指南 `.codex/skills/experiment-planning-zh/remote-execution.md`，确认 `spider-remote` 与 `/home/xiayb/pHRI_workspace/spider`。
- [x] 复读 E074 计划/脚本/config，确认 E075 不需要改 reward 实现，只需新增两个组合 override 与 E075 脚本。
- [x] 写入 E075 计划: `workspace/core4d/plan/81_E075_limited_hold_contact_remote_plan.md`。
- [x] 新增 E075 override:
  - `examples/config/override/core4d_e075b_box023.yaml` = E074A + hold_contact scale 1.0, window 1.8-2.5s。
  - `examples/config/override/core4d_e075a_box023.yaml` = E074A + hold_contact scale 0.5, window 1.8-2.5s。
- [x] 新增 E075 train/eval/remote/pull 脚本:
  - `workspace/core4d/scripts/train/train_E075.sh`
  - `workspace/core4d/scripts/eval/eval_E075.py`
  - `workspace/core4d/scripts/run_E075_remote.sh`
  - `workspace/core4d/scripts/pull_E075_remote_results.sh`
- [x] 静态验证通过: `py_compile eval_E075.py`; `bash -n` 三个 shell 脚本。
- [x] Hydra compose 验证:
  - E075B: `ctrl_ref_guard_scale=0.5`, `hold_contact_rew_scale=1.0`, window 1.8-2.5, `contact_hdmi_target_uses_eef_offset=True`。
  - E075A: `ctrl_ref_guard_scale=0.5`, `hold_contact_rew_scale=0.5`, window 1.8-2.5, `contact_hdmi_target_uses_eef_offset=True`。
- [x] 提交并推送 E075 实验脚本/config: `90d5b34 exp(core4d): E075 limited hold-contact remote sweep`。
- [x] 远程 `spider-remote:/home/xiayb/pHRI_workspace/spider` 已 fast-forward 到 `90d5b34`。
- [x] 启动远程 tmux session `E075`:
  - E075B -> GPU0, PID 1238653。
  - E075A -> GPU1, PID 1238654。
  - 远程 scene snapshot: `workspace/core4d/results/E075/scene_snapshot/`。

## 远程运行状态

- 2026-05-14 21:57: tmux 输出确认两个 run 已启动。
- 初始结果计数: `0` 个 `.npz`，符合刚启动状态。
- 2026-05-14 22:19: 远程 E075 完成，已 scp 回收并完成本地 `eval_E075.py`。

### 初步数值结果

| 指标 | E074A | E074C | E075B scale1.0 limited | E075A scale0.5 limited |
|------|------:|------:|-----------------------:|-----------------------:|
| yaw 0.017/0.033 deg | 0.574 / 1.075 | 0.574 / 1.075 | 0.574 / 1.075 | 0.574 / 1.075 |
| B1 pre-contact foot z | 0.083m | 0.085m | 0.083m | 0.085m |
| first zero contact | f110 | f101 | f110 | f111 |
| frame100-145 contact | 54.3% | 63.0% | 76.1% | 60.9% |
| post2 contact | 54.3% | 64.2% | 67.9% | 61.7% |
| post2 obj_err max | 0.289m | 0.324m | 0.287m | 0.290m |
| post2 obj_err mean | 0.177m | 0.197m | 0.158m | 0.155m |
| post2 pelvis_z min | 0.692m | 0.701m | 0.660m | 0.147m |
| first robot ctrl Linf >0.5 | f122 | f114 | f117 | f100 |
| post2 robot ctrl Linf max | 0.740 | 0.897 | 0.690 | 0.662 |
| post2 min hand SDF mean | 0.073m | 0.043m | 0.057m | 0.054m |

初步判断:

- E075B 是当前数值最好的组合：contact 高于 E074C，object error 接近/略优 E074A，robot ctrl Linf 更低，pelvis 没有摔倒。
- E075A 接触也改善，但 first robot ctrl Linf 在 f100 即超阈值，且 pelvis_z min=0.147m，稳定性明显回归，不宜作为主线。
- 需要等待 subagent 视觉复核 E075B/E075A 关键帧后写正式 log。
- [x] subagent Erdos 完成 E075A/E075B 关键帧视觉复核。
- [x] 写入正式结果日志: `workspace/core4d/log/96_E075_limited_hold_contact_results.md`。
- [x] 更新 `EXPERIMENT_TRACKER.md` E075 行。

正式结论:

- E075B 是 best-so-far partial positive：contact 大幅提升，object error 不退化，f180 站稳且箱子分离；但 f145-f166 释放仍不干净，first obj_err>25cm 仍 f100。
- E075A 失败：f166-f180 出现身体/腿/箱强干涉并摔倒。
- 下一步应以 E075B 为 base，先做 release/leg clearance replay 诊断，再设计放置后脱离/clearance reward。

### 追加诊断: E075B f115-f130 右腿相位偏差

- [x] 用户指出 E075B f120-f125 右腿相对 ref 突然前跨；已补抽 f115-f130 连续帧并交给 subagent Erdos 视觉复核。
- [x] 视觉结论: 不是单帧视觉错觉。ref 在 f120 后进入停步/弯腰/准备放箱，右脚接近地面且趋于稳定；sim 仍在继续向前走一步，右腿从后摆连续前跨，到 f125-f130 与 ref 姿态明显分歧。
- [x] 数值结论: f119-f125 sim right_foot XY 每帧位移约 7.5-9.7cm，而 ref 约 1.9-4.0cm；right_hip_pitch ctrl diff 在 f120-f124 成为主导偏离，约 -0.50rad。该段是真实步态/任务相位偏差，不只是 release 问题。
- [x] 将用户关于 hand-crafted `hold_contact_start/end_eval_time` 泛化风险、HDMI contact label vs core4d estimated mask 差异、以及 E076 应优先 audit/fix contact mask 的讨论写入 `workspace/core4d/log/96_E075_limited_hold_contact_results.md`。

## 当前实验

- **Run ID**: E075
- **阶段**: Implement
- **目标**: 在 E074A ctrl guard 基础上，对比 `hold_contact` 的限时中等强度(scale=1.0, 1.8-2.5s) 与限时弱强度(scale=0.5, 1.8-2.5s)，验证能否提升接触而不复现 E074C 的腿/箱干涉。

---

# E074+ Strategy Progress — 2026-05-14

## 当前状态: Plan 已写入，等待用户审核 E074+ 总路线

## 完成步骤

- [x] 按 `experiment-planning-zh` 恢复 `EXPERIMENT_TRACKER.md`、最新 plan/log、`progress.md`。
- [x] 启用 subagent Halley 复盘 HDMI workflow 与 E071/E073 后剩余差异。
- [x] 启用 subagent Euler 复盘 E037-E067 contact/reward 历史改动。
- [x] 本地复读 log 82-93、E041c/E062/E065-E067 yaml、`run_hdmi.py`/`run_mjwp.py`/`mjwp.py` reward 与优化循环。
- [x] 明确 E071 后结论重置：旧 pre-contact lunge 归因大多被 ctrl mapping bug 污染；当前主失败面是 post-2s hold/contact。
- [x] 写入总计划: `workspace/core4d/plan/79_E074_plus_post_E071_hold_strategy_plan.md`。
- [x] 写入 E074 前置分析日志: `workspace/core4d/log/94_E074_preflight_base_palm_normal_analysis.md`，覆盖 E060-E067 代码影响、E074 base、E062 palm normal 含义和影响。
- [x] 更新 `EXPERIMENT_TRACKER.md`，加入 E074 preflight 索引。
- [x] 写入 E074 实施与远程调度计划: `workspace/core4d/plan/80_E074_remote_execution_plan.md`。
- [x] 修改 `spider/config.py` / `spider/simulators/mjwp.py`，新增默认关闭的 E074A ctrl guard 与 E074C hold contact reward。
- [x] 新增 E074A/E074C override、训练脚本、评估脚本、远程启动与结果回收脚本。
- [x] 验证: `py_compile` 通过；`bash -n` 通过；Hydra compose 确认 E074A/E074C override 生效；`git diff --check` 通过。
- [x] 按 `experiment-planning-zh/remote-execution.md` 修正远程默认配置: `REMOTE_HOST=spider-remote`, `REMOTE_REPO=/home/xiayb/pHRI_workspace/spider`，并补充 tmux capture-pane/结果计数提示。
- [x] 首次远程启动时 SSH 网络超时且旧脚本无 timeout，已终止挂起进程，并给远程启动/回收脚本加入 BatchMode、ConnectTimeout 和 ServerAlive 参数。

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|------|---------|----------|
| 本地沙箱内 `git push` 触发 DNS 失败 | 1 | 用已批准的 escalated `run_E074_remote.sh` 重试，push 显示 up-to-date |
| 远程 SSH 间歇超时，旧启动脚本无 `ConnectTimeout` 导致挂起 | 1 | 终止挂起进程；脚本增加 `BatchMode=yes`、`ConnectTimeout=20`、`ServerAlive*` |

## 远程运行状态

- 2026-05-14 21:02: 远程 `spider-remote:/home/xiayb/pHRI_workspace/spider` 已 fast-forward 到 `b112eac`。
- tmux session: `E074`。
- 远程启动命令: `bash workspace/core4d/scripts/train/train_E074.sh parallel 0 1`。
- GPU 分配: E074A -> GPU0, E074C -> GPU1。
- tmux 输出确认:
  - `[21:02:26] launched E074A PID=1219883, E074C PID=1219884`
  - `[21:02:26] === E074A_box023 override=core4d_e074a_box023 GPU=0 ===`
  - `[21:02:26] === E074C_box023 override=core4d_e074c_box023 GPU=1 ===`
- 初始结果计数: `0` 个 `.npz`，符合刚启动状态。

## E074 回收状态

- 2026-05-14 21:36: 已从 `spider-remote:/home/xiayb/pHRI_workspace/spider` scp 回收 E074 结果与日志。
- 本地结果:
  - `workspace/core4d/results/E074/E074A_box023.npz`
  - `workspace/core4d/results/E074/E074A_box023.mp4`
  - `workspace/core4d/results/E074/E074C_box023.npz`
  - `workspace/core4d/results/E074/E074C_box023.mp4`
  - `workspace/core4d/results/E074/comparison.csv`
  - `workspace/core4d/results/E074/keyframes/{E074A,E074C}/f100..f180.jpg`
- 本地日志:
  - `logs/E074/E074A_box023.log`
  - `logs/E074/E074C_box023.log`
  - `logs/E074/eval_E074_local_after_pull.log`

### 初步数值结果

| 指标 | E073 | E074A ctrl guard | E074C hold contact |
|------|-----:|-----------------:|-------------------:|
| yaw 0.017/0.033 deg | 0.574 / 1.075 | 0.574 / 1.075 | 0.574 / 1.075 |
| B1 pre-contact foot z | 0.080m | 0.083m | 0.085m |
| first zero contact | f108 | f110 | f101 |
| frame100-145 contact | 45.7% | 54.3% | 63.0% |
| post2 contact | 49.4% | 54.3% | 64.2% |
| post2 obj_err max | 0.293m | 0.289m | 0.324m |
| post2 pelvis_z min | 0.663m | 0.692m | 0.701m |
| post2 robot ctrl Linf max | 0.778(E073 prior) | 0.740 | 0.897 |

初步判断:

- E074A 小幅改善 contact/object/stability，符合“更保守”的预期，但 contact 仍不足。
- E074C 明显提高 contact 与 hand SDF，但 first zero contact 反而提前到 f101，post2 obj_err max 变差到 0.324m，说明接触 reward 可能让手更贴近但没有改善物体跟随。
- 两者均未造成 early drift 或摔倒回归。
- 下一步需要按用户要求用 subagent 复核 E074A/E074C 视频关键帧后写正式 log 95。

## E074 正式分析

- [x] subagent Erdos 复核 E074A/E074C 关键帧。
- [x] 写入正式结果日志: `workspace/core4d/log/95_E074_remote_hold_contact_results.md`。
- [x] 更新 `EXPERIMENT_TRACKER.md` E074 行。

正式结论:

- E074A 是 partial positive：ctrl guard 将 robot ctrl 大偏离延后到 f122，视觉更接近成功，但 contact 提升不足。
- E074C 是 metric positive / visually unsafe：post2 contact 达 64.2%，但 obj_err 变差，后段腿/箱干涉明显。
- 下一步不应直接原样组合 E074A+C；推荐 E075B = E074A + time-limited weaker hold_contact。

## 核心结论

- E073 是下一步可信 base：`contact_hdmi_target_uses_eef_offset=true` 小幅改善 contact 并消除摔倒，但没有解决 f130/f145 脱手。
- E060-E067 的 task_obj/actuator/body-partition 结论暂不复用；保留代码开关但不纳入第一批。
- E074 第一批建议只做两个单变量方向：
  - E074A: E073 + robot ctrl trust-region guard。
  - E074C: E073 + hold/contact continuity reward。
- 第一波调度: 远程 A6000 GPU0 跑 E074A，GPU1 跑 E074C；本机只做编译/smoke 和结果回收后的评估。

## 下一步

- 等用户审核 `plan/79`。
- 若批准，写 E074 具体实施 plan，再开始代码与训练脚本实现。

## 追加分析: E062 palm normal 对 E074 base 的含义

- E062 的 `contact_hdmi_palm_normal_left/right` 不是接触点位置，而是 contact_hdmi orientation reward 使用的 wrist-local 朝向向量。
- E041c 默认 box025 指纹是 L=`[0,-1,0]`, R=`[0,+1,0]`; E062 对 box023 自动计算后改成双手 `[+1,0,0]`。
- 该向量进入 `mjwp.py` 的 E041 orientation block: `palm_world = quat_apply(eef_quat, palm_local)`, 再与 `target_world-contact_point` 做 dot，作为 additive ori reward 的方向项。
- 因 E073 -> E071W02 -> E062 -> E041c，E074 默认继承 E062 palm normal。它已经是 E071/E073 结果的一部分，后续不应默认移除；若要验证影响，应作为单独 ablation。

---

# E073 Progress — 2026-05-14

## 当前状态: ✅ E073 完成，target eef_offset 修正部分有效但未解决 hold

## 完成步骤

- [x] 按 `experiment-planning-zh` 读取 `EXPERIMENT_TRACKER.md`、E072 plan/log、`progress.md`。
- [x] 审查 `examples/run_mjwp.py` 与 `spider/simulators/mjwp.py` contact_hdmi dynamic target 实现。
- [x] 发现 E040 dynamic target 口径不一致：target 使用 ref wrist body origin，reward 使用 sim `wrist + eef_offset` contact point。
- [x] 写 E073 plan: `workspace/core4d/plan/78_E073_contact_target_offset_consistency_plan.md`。
- [x] 修改 `spider/config.py`，新增 `contact_hdmi_target_uses_eef_offset`，默认 false。
- [x] 修改 `examples/run_mjwp.py`，E073 打开字段时 dynamic target 改用 ref `wrist + eef_offset`。
- [x] 新增 override: `examples/config/override/core4d_e073_box023.yaml`。
- [x] 新增 E073 eval: `workspace/core4d/scripts/eval/eval_E073.py`。
- [x] 新增 E073 train: `workspace/core4d/scripts/train/train_E073.sh`。
- [x] `py_compile` 通过。
- [x] 运行 `bash workspace/core4d/scripts/train/train_E073.sh 0`；日志确认 RTX 5090 可见，且 `E040 dynamic target ... uses_eef_offset=True` 生效。
- [x] 输出 `eval_summary.json`、`timeseries.csv`、timeline plot、frame100-180 keyframes。
- [x] 按用户要求将关键帧视觉复核交给 subagent Ampere，主线程未直接 `view_image`。
- [x] 写 E073 结果 log: `workspace/core4d/log/93_E073_contact_target_offset_consistency_results.md`。

## 当前实验

- **Run ID**: E073
- **阶段**: Complete
- **目标**: 修正 dynamic target 的 eef_offset 口径，验证 frame100-145 hand-object contact 是否改善。

## 关键结果

| 指标 | E071/E072 | E073 |
|------|----------:|-----:|
| yaw err t=0.017/0.033 | 0.574 / 1.075 deg | 0.574 / 1.075 deg |
| B1 pre-contact max foot z | 0.069m | 0.080m |
| first obj_err >25cm | frame100 / 2.00s | frame100 / 2.00s |
| first sim zero contact | frame100 / 2.00s | frame108 / 2.16s |
| post2 sim contact frames | 44.4% | 49.4% |
| post2 obj_err max | 0.308m | 0.293m |
| first pelvis_z <45cm | frame166 / 3.32s | none |
| post2 pelvis_z min | 0.207m | 0.663m |

**视觉结论**: f100/f115 手和箱还较近；f130 起拿持质量明显变差；f145 箱子已明显落地/接触地面；f166/f168 未像 E071/E072 那样摔倒，但有脚/腿与箱体异常接触。

**结论**: eef_offset target 口径修正改善了接触连续性和稳定性，但没有解决 2s 后真实 hold。E074 应继承 E073，并增加 robot ctrl trust-region guard，重点压 frame100-145 的断触和 robot ctrl 快速偏离。

- 下一步: 规划 E074。

---

# E072 Progress — 2026-05-14

## 当前状态: ✅ E072 完成，hold/contact 先失效已定位

## 完成步骤

- [x] 按 `experiment-planning-zh` 读取 `EXPERIMENT_TRACKER.md`、E071 plan、E071 log、`progress.md`。
- [x] 确认 E071 结论：0-2s init/early drift 修复；2.0s 后物体跟踪误差先升至 >25cm，约 3.32s robot pelvis/body z <45cm 后摔倒。
- [x] 检查 E071 结果结构：`E071W02_box023.npz` 含 `qpos/qvel/ctrl/time/trace_ref`，scene snapshot 含 `scene_act.xml`，可以做 replay 诊断。
- [x] 写入 E072 plan: `workspace/core4d/plan/77_E072_post2_hold_place_diagnosis_plan.md`。
- [x] 新增 E072 eval 脚本: `workspace/core4d/scripts/eval/eval_E072.py`。
- [x] 新增 E072 入口脚本: `workspace/core4d/scripts/train/train_E072.sh`。
- [x] 运行 `bash workspace/core4d/scripts/train/train_E072.sh`，输出 `timeseries.csv`、`diagnosis_summary.json`、`contact_summary.csv`、timeline plot、frame-index keyframes。
- [x] 复核关键帧 f100/f115/f130/f145/f166/f180。
- [x] 写 E072 结果 log: `workspace/core4d/log/92_E072_post2_hold_place_diagnosis_results.md`。
- [x] 更新 `EXPERIMENT_TRACKER.md`。

## 当前实验

- **Run ID**: E072
- **阶段**: Complete
- **目标**: 区分 post-2s failure 是 hand-object hold/contact 先丢，还是 putdown 阶段稳定性先崩。

## 关键结果

| 指标 | 结果 |
|------|------|
| first obj_err >25cm | frame 100 / eval 2.00s / 0.308m |
| first sim hand-object zero contact | frame 100 / eval 2.00s, ref 同帧仍 contact=1 |
| first robot ctrl Linf >0.5 | frame 109 / eval 2.18s |
| first sim min hand SDF >10cm | frame 134 / eval 2.68s |
| first pelvis_z <45cm | frame 166 / eval 3.32s |
| post2 contact frames | sim 44.4% vs ref 80.2% |

**结论**: E071 post-2s 是 hold/contact 先失效，随后 CEM 追 object/body target 导致前扑和摔倒。object ctrl diff max 仅 0.01，不是新 mapping 问题。

## 下一步

- E073 优先做 hold/contact consistency 或 contact-preserving target，不先做单纯 stability weight。
- 同时考虑 robot ctrl trust-region guard，限制 frame 109 后的 robot ctrl Linf 级偏移。

---

# E068 Progress — 2026-05-14

## 当前状态: Plan 已写入，开始 init drift 诊断

## 完成步骤

- [x] 读取 `EXPERIMENT_TRACKER.md`、最新 plan/log/progress，确认最新问题来自 log 87 的 MJWP init pose mismatch。
- [x] 审查 `spider/simulators/mjwp.py::setup_env()`、`examples/run_mjwp.py` 初始化路径，发现 qpos/qvel/ctrl 写入后立即 `mj_step()` 的可疑路径。
- [x] 写 plan → `workspace/core4d/plan/73_E068_mjwp_init_drift_plan.md`
- [x] 新建 E068 诊断脚本 → `workspace/core4d/scripts/debug/diagnose_E068_init_drift.py`
- [x] 新建 E068 入口脚本 → `workspace/core4d/scripts/train/train_E068.sh`
- [x] 读取同步后的 E062-E067 真实结果，确认 E062/E063 第二个 substep 已出现约 22 deg yaw drift。
- [x] 发现 drift 不是 init `mj_step` 单独造成：CPU `mj_step` yaw drift 仅 0.22 deg；真实轨迹第一个 committed ctrl 相对 ref 有 1.56 rad 级别 robot joint 偏差。
- [x] 新建 first-commit 分析脚本 → `workspace/core4d/scripts/debug/analyze_E068_first_commit.py`
- [x] 运行 first-commit 分析，输出 `workspace/core4d/results/E068/first_commit_trace.csv` 和 `first_ctrl_delta.csv`
- [x] 视频/关键帧复查 E062/E063/E067，确认 lunge / handstand 与数值诊断一致。
- [x] 写 E068 结果 log → `workspace/core4d/log/88_E068_mjwp_init_drift_results.md`
- [x] 更新 `EXPERIMENT_TRACKER.md` 的 E068 行和 Logs 引用。
- [x] 写 E069 plan → `workspace/core4d/plan/74_E069_first_tick_warmup_plan.md`
- [x] 新建 E069 yaml 变体: `core4d_e069w02_box023.yaml`, `core4d_e069w05_box023.yaml`
- [x] 新建 E069 train/eval 脚本。
- [x] 空结果运行 `train_E069.sh eval` 验证脚本可执行，当前结果缺失为预期。
- [x] 修正 `train_E069.sh`，避免 eval-only 模式触发 scene snapshot。

## 当前实验

- **Run ID**: E068
- **Version**: R5 / Phase 18
- **阶段**: Plan → Diagnose
- **开始时间**: 2026-05-14

## 下一步

- 等待 GPU 机器运行 `bash workspace/core4d/scripts/train/train_E069.sh parallel 0 1`。
- 运行完成后检查 `workspace/core4d/results/E069/eval_summary.csv`，再写 log 89。

## 创建/修改的文件

- `workspace/core4d/plan/73_E068_mjwp_init_drift_plan.md`
- `workspace/core4d/scripts/debug/diagnose_E068_init_drift.py`
- `workspace/core4d/scripts/debug/analyze_E068_first_commit.py`
- `workspace/core4d/scripts/train/train_E068.sh`
- `workspace/core4d/log/88_E068_mjwp_init_drift_results.md`
- `workspace/core4d/results/E068/first_commit_trace.csv`
- `workspace/core4d/results/E068/first_ctrl_delta.csv`
- `workspace/core4d/EXPERIMENT_TRACKER.md`
- `workspace/core4d/plan/74_E069_first_tick_warmup_plan.md`
- `examples/config/override/core4d_e069w02_box023.yaml`
- `examples/config/override/core4d_e069w05_box023.yaml`
- `workspace/core4d/scripts/train/train_E069.sh`
- `workspace/core4d/scripts/eval/eval_E069.py`
- `workspace/core4d/progress.md`

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| 当前沙箱无 CUDA 设备，`device=cuda:0` 触发 `RuntimeError: No CUDA GPUs are available` | 1 | 将 E068 诊断入口默认改为 `E068_DEVICE=cpu`，只做 init state 诊断 |
| CPU device 无法执行 `setup_env()` 的 CUDA graph capture，触发 `RuntimeError: Must be a CUDA device` | 1 | 诊断脚本改为先保存 CPU `mj_forward`/`mj_step` 证据，并尝试手动 `mjwarp.put_data` 不 capture graph |
| `train_E069.sh eval` 初版会先 snapshot scene | 1 | 调整脚本，仅 `parallel/single_*` 训练模式执行 snapshot |

---

# E057 Progress — 2026-05-13

## 当前状态: ✅ 完成 (bucket005_s2 hand-snap, 6/6 Claims 通过)

## 完成步骤

- [x] 写 plan → `plan/67_E057_bucket005_s2_hand_snap_plan.md`
- [x] 克隆 E055 三件套 (snap / visualize / extract_keyframes), 改 CASE 路径
- [x] 新建 `verify_snap_face.py` (C6 — snap 后 main face vs E056 诊断一致性)
- [x] 写一键脚本 `run_E057_snap.sh` (snap → viz → keyframes → face 验证)
- [x] 跑流水线 — 单次跑通 4 步, 约 2 分钟
- [x] 视频核实 5 keyframe (front+side, top=ref/bot=snap)
- [x] 写 log → `log/67_E057_bucket005_s2_hand_snap_results.md`
- [x] 更新 EXPERIMENT_TRACKER (E057 行 + log/plan/scripts 引用)

## Claims 验证

| ID | 标准 | 实际 | 通过 |
|---|---|---|---|
| C1 | npz + csv + mp4 三件齐 | 101K npz + 11K csv + 2.3M mp4 | ✅ |
| C2 | palm-to-surface final ≤ 5cm | mean 4.93cm, max 5.10cm | ✅ |
| C3 | 关节限位 100% | 176/176 | ✅ |
| C4 | ≥ 4/5 keyframe 视觉合格 | **5/5 通过**, mid frame 教科书对侧握 ⭐ | ✅ |
| C5 | 一键脚本 | run_E057_snap.sh 跑通 | ✅ |
| C6 | snap face = E056 (L=-yz, R=+yz) | L=-yz 77%, R=+yz 100% | ✅ |

**6/6 通过 ✅**

## 关键结果

```
intent: (20, 107) = 88 frames @ 30fps (与 E056 诊断一致)
snap statistics (intent 内 88f × 2 hands = 176 rows):
  init  cm: mean=5.04 max=7.57   ← mocap 原始, 已经接近 5cm offset
  final cm: mean=4.93 max=5.10   ← 收敛到 target (表面外 5cm)
  ik_residual cm: mean=0.11 max=1.12
  ik_iters: mean=8.5 max=50
  joints in_limits: 176/176 (100%)
  L final: mean=4.99 max=5.09
  R final: mean=4.87 max=5.10

C6 face verification (intent 内):
  REF  L=-yz (med 0.58cm, 100%) | R=+yz (med 3.24cm, 100%)
  SNAP L=-yz (med 2.90cm,  77%) | R=+yz (med 1.87cm, 100%)
  → 双手 main face 完全保留 E056 诊断, 没漂移到错面
```

## 重要发现

1. **case 选对了, snap 几乎自由**: bucket005_s2 mocap 原始就 init=5cm, IK 几乎不需要努力。E055 box023 init~10cm, snap 后才 5cm。**E056 case 排序的物理意义在 E057 上兑现**。

2. **C6 face 验证是必要 guard**: 没这条 claim, snap 把 L 投到错面 (e.g. +yz) 数值上仍报"成功"。**后续任何 IK-to-surface 都应该 verify 接触面**。

3. **frame warm-start 持续有效**: max iter=50 偶发 (intent boundary), mean 8.5 iter, 大多数帧前一帧 qpos 就近。

## 下一步: E058 (Path B-CEM, bucket005_s2)

- 把 `warmstart_qpos.npz` 喂入 MJWP CEM 作为初始 mean trajectory
- body tracking ref 也换成 qpos_snap (让 reward "信" 修正后的 ref)
- 对比有/无 warmstart 的 contact / stability / pelvis_z
- Claims (草案):
  - contact (palm 距 bucket < 5cm 帧占比) ≥ 50%
  - stability (pelvis_z ≥ 0.5m) ≥ 90%
  - 视频: snap 阶段 CEM 没把 ±yz 两侧握姿"破坏"

## 改动文件

| 类型 | 路径 |
|---|---|
| 新建 plan | `workspace/core4d/plan/67_E057_bucket005_s2_hand_snap_plan.md` |
| 新建脚本 (×4) | `workspace/core4d/scripts/E057/{snap_bucket005_s2,visualize_snap,verify_snap_face}.py` + `extract_snap_keyframes.sh` |
| 新建一键脚本 | `workspace/core4d/scripts/run_E057_snap.sh` |
| 输出 npz | `workspace/core4d/results/E057/bucket005_s2_person1/warmstart_qpos.npz` |
| 输出 csv (×2) | `snap_diagnostics.csv` + `face_verification.csv` |
| 输出 mp4 | `snap_visualization.mp4` (2.3M, 148 帧) |
| 输出 png | `face_dist_snap.png` (2×2 时序) |
| 输出 jpg (×5) | `keyframes/frame_0[0-4]_*.jpg` |
| 新建 log | `workspace/core4d/log/67_E057_bucket005_s2_hand_snap_results.md` |
| EXPERIMENT_TRACKER | 添加 E057 行 + log/plan/scripts 引用 |

**`spider/preprocess/hand_snap_ik.py` 没动** — E055 实现 case-agnostic 已被 E057 验证。

---

## E069 进展: first-tick warmup 运行与保存修复

- [x] 已确认本机 GPU 在非 sandbox 命令下可见: RTX 5090 / Driver 580.126.09 / CUDA 13.0。
- [x] 已运行 `bash workspace/core4d/scripts/train/train_E069.sh parallel 0 0`。
- [x] W02/W05 均跑到 `sim_steps: 272/272`，不是 CUDA 或中途优化失败。
- [x] 失败点: `examples/run_mjwp.py` 结束保存 `info_list` 时 `np.stack` 遇到跨 tick shape 不一致的诊断字段，导致 `.npz`/`.mp4` 没有落盘。
- [x] 已修复保存逻辑: `qpos/qvel/time/ctrl` 等 shape 一致字段继续保存；shape 不一致或缺失的诊断字段跳过并写 warning。

### 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| E069 W02/W05 完整跑完后 `ValueError: all input arrays must have the same shape` | 1 | 修改 `examples/run_mjwp.py` 的 info 聚合逻辑，跳过 shape 不稳定的诊断字段，保留轨迹核心字段 |

### 下一步

- [x] 重跑 E069 W02/W05。
- [x] 修正 `eval_E069.py` 的 ctrl_ref 口径并重新生成 `eval_summary.csv`。
- [x] 从视频提取关键帧并写 `log/89_E069_first_tick_warmup_results.md`。
- [x] 更新 `EXPERIMENT_TRACKER.md`。

## E069 结果摘要

| 指标 | E069-W02 | E069-W05 | 结论 |
|------|----------|----------|------|
| warmup robot ctrl diff | 0.00 rad | 0.00 rad | ref ctrl 确实提交 |
| yaw err t=0.017/0.033s | 12.40 / 22.16 deg | 12.40 / 22.16 deg | warmup 无法压住 early yaw |
| B1 max foot z [0,2s] | 0.222m | 0.428m | 仍单脚/lunge |
| pelvis_min_intent | 0.578m | 0.197m | W02 数值通过但视频仍不可用 |

**新结论**: first CEM override 不是主因。即使 warmup 内提交 `ctrl_ref`，MJWarp commit step 仍复现 12/22 deg early yaw drift。下一步应做 E070 ref-control parity: MuJoCo `mj_step(ctrl_ref)` vs MJWarp `step_env(ctrl_ref)`。

## E070 计划

- [x] 写 plan → `workspace/core4d/plan/75_E070_mjwarp_ref_control_parity_plan.md`
- [x] 用户确认继续后，实现 parity 诊断脚本 → `workspace/core4d/scripts/debug/diagnose_E070_ref_control_parity.py`
- [x] 实现入口脚本 → `workspace/core4d/scripts/train/train_E070.sh`
- [x] `py_compile` 通过。
- [x] 首轮 E070 GPU parity 诊断完成。
- [x] 根据首轮结果扩展脚本: 增加 `qpos_ctrl` vs `orig_ctrl` 对照，区分 run_mjwp 当前 qpos-as-ctrl 映射与原始 29-dim robot ctrl 映射。
- [x] 重跑 E070 GPU parity 诊断。

### E070 实现补充

计划原本只比较 CPU vs MJWarp；实际脚本增加了两条控制变量：

- `zero_gains`: 保持 object actuator gains 为 0，对应 setup/start 状态。
- `restored_gains`: 按 `run_mjwp.py` commit 阶段恢复 object actuator gains，再提交 `ctrl_ref`。

这样能区分 drift 来自 MJWarp step 本身，还是来自 commit 阶段 object actuator gain 恢复后的物体反作用。

### E070 首轮发现

- `qpos_ctrl` 口径下，CPU MuJoCo 和 MJWarp 完全一致，并且都精确复现 E069: t=0.017/0.033 yaw err = 12.403/22.156 deg，`qpos_max_abs_diff_vs_e069 ≈ 0`。
- `zero_gains` 与 `restored_gains` 几乎一致，object actuator gain 恢复不是主因。
- 新疑点: `run_mjwp.py` 的 `ctrl_ref = qpos_ref[:, :config.nu]` 可能把 floating base pos/quat 当成 robot actuator ctrl；E068 的小漂移使用的是原始 29-dim robot ctrl + scene_act object ctrl。需要 `orig_ctrl` 对照验证。

### E070 最终发现

| ctrl 口径 | CPU yaw err t=0.017/0.033 | MJWarp yaw err t=0.017/0.033 | vs E069 | 结论 |
|-----------|----------------------------|-------------------------------|---------|------|
| `qpos_ctrl` (`qpos_ref[:, :nu]`) | 12.403 / 22.156 deg | 12.403 / 22.156 deg | `qpos_max_abs_diff≈0` | 精确复现 E069 错误 |
| `orig_ctrl` (原始 29-dim robot ctrl + scene_act object ctrl) | 0.574 / 1.075 deg | 0.574 / 1.075 deg | 明显不同 | early drift 基本消失 |

**根因修正**: 不是 MJWarp physics mismatch，也不是 object actuator gain 恢复。`examples/run_mjwp.py` 在 contact guidance 下的 `ctrl_ref = qpos_ref[:, :config.nu]` 把 floating-base qpos 前 7 维混入 robot actuator ctrl，导致 ref-control 本身就是错的。E071 应修 scene_act ctrl 映射: 保留原始 29-dim robot ctrl，只把 object 6DOF ctrl 从转换后的 qpos 填入末 6 维。

## E071 结果摘要

- [x] 已写 plan: `workspace/core4d/plan/76_E071_scene_act_ctrl_mapping_fix_plan.md`
- [x] 已修 `examples/run_mjwp.py`: 删除 qpos-as-ctrl fallback，保留原始 29-dim robot ctrl。
- [x] 已新增配置: `examples/config/override/core4d_e071w02_box023.yaml`
- [x] 已新增训练脚本: `workspace/core4d/scripts/train/train_E071.sh`
- [x] 已新增评估脚本: `workspace/core4d/scripts/eval/eval_E071.py`
- [x] 已运行 `bash workspace/core4d/scripts/train/train_E071.sh 0`。
- [x] 已生成 `.npz`、`.mp4`、`eval_summary.csv` 和关键帧。

| 指标 | E069-W02 | E071-W02 | 结论 |
|------|----------|----------|------|
| yaw err t=0.017/0.033s | 12.40 / 22.16 deg | 0.574 / 1.075 deg | early drift 消失 |
| vs E070 orig parity | N/A | -0.0004 / +0.0005 deg | 与正确 ctrl 口径一致 |
| warmup ctrl diff | 0.00 / 0.00 | 0.00 / 0.00 | ref ctrl 提交正确 |
| B1 max foot z [0,2s] | 0.222m | 0.069m | pre-contact lunge 消失 |
| pelvis_min_intent | 0.578m | 0.674m | 更稳定 |
| post-2s obj_err max/mean | 未统计 | 0.308 / 0.133m | post-contact FAIL |
| first post-2s obj_err > 25cm | 未统计 | 2.00s | 没有稳定拿住箱子 |
| post-2s pelvis body z min | 未统计 | 0.200m | 摔倒 |
| first post-2s pelvis z < 45cm | 未统计 | 3.32s | 摔倒开始 |

**修正结论**: E070 根因只对 early yaw/lunge 完全确认。scene_act 下 `qpos_ref[:, :nu]` fallback 是 box023 0-2s 初始漂移的主因；保留 raw robot ctrl 并由 scene_act conversion 补 object ctrl 后，初始漂移消失。但 E071 不是整体成功：2s 后 robot 没有稳定拿住箱子，约 3.3s 开始摔倒。下一步 E072 应聚焦 post-2s hold/place failure 诊断，而不是先做泛化 regression。

---

## 2026-05-15 E076 contact source audit 进展

- [x] 使用 `experiment-planning-zh` 恢复 `EXPERIMENT_TRACKER.md`、E075 log 和 `progress.md`。
- [x] 回收 subagent 结果：
  - Fermat 确认 CORE4D raw 没有人手接触人工真值；官方 contact 是 SMPL-X/object 几何生成，`prepare_hho.py` 默认 2cm，visualization runtime contact 用 3cm。
  - Laplace 确认当前 `box023_person1` 进入 SPIDER 后是单 G1 + object，`contact=(136,2)` 全帧 `[1,1]`，不是 CORE4D/HDMI label；E039/E075 mask 仍是单 G1 ref 的 scalar SDF proxy。
- [x] 本地确认源序列：
  - `box023_person1` 对应 raw `/mnt/.../CORE4D_Real/human_object_motions/20231008/045`，object=`Box023`，action=`move2_obs0`。
  - Holosoma `trimmed` qpos 与 `retargeted[42:178]` 精确一致，因此 SPIDER 30Hz ref frame `k` 对应 raw frame `k+42`。
- [x] 修正视频帧对齐：
  - E075 视频/eval 是 50Hz，`f115-f130` 是 `2.30-2.60s`。
  - 对应 30Hz ref frame 约 `69-78`，raw frame `111-120`，不是 raw/ref 的 `115-130`。
- [x] raw 几何核验结果：
  - 在 E075 `f115-f130` 对应 raw `111-120`，person1 左手强接触：min dist mean/min/max = `0.74/0.42/1.08cm`，2cm/3cm/5cm 均 16/16 帧。
  - person1 右手是边界接触：min dist mean/min/max = `2.12/1.77/2.58cm`；2cm 阈值 5/16 帧，3cm 阈值 16/16 帧。
  - person2 双手强接触：左手 `0.17/0.03/0.53cm`，右手 `0.11/0.04/0.17cm`，2cm/3cm/5cm 均 16/16 帧。
- [x] 写入新诊断日志：`workspace/core4d/log/97_E076_contact_source_audit.md`。

### 修正后的判断

此前“ref 右手不一定应该继续强接触”的说法过强。当前证据只支持：

- robot retarget / MuJoCo ref 里右手几何接触弱；
- raw SMPL-X 里 person1 右手是 2cm 阈值边界、3cm 阈值持续接触；
- person2 双手在同一阶段强接触，所以双人支撑必须纳入解释；
- 当前 SPIDER contact/mask 不是 raw/HDMI label，下一步应先做 contact source alignment 和 per-hand mask 修复，而不是继续手写 hold/release 时间窗。

---

## E077 进展: 3cm contact mask + box023_person2 计划

- [x] 已按 `experiment-planning-zh` 写入计划：`workspace/core4d/plan/82_E077_core4d_3cm_contact_mask_and_box023_person2_plan.md`。
- [x] 初步检查本地没有现成的 `20231008-045-person2-Box023` Holosoma retarget 输出；person2 需要从 raw `20231008/045` 重新跑 `convert_core4d_to_omniretarget.py` + `robot_retarget.py`，不能直接复制 person1。
- [x] 计划将 3cm mask 分成 raw/spider/eval 三个时间轴，避免再次混淆 30Hz ref frame 和 50Hz eval frame。

### 当前 E077 决策

- 先生成 3cm raw contact proxy，作为临时 contact mask “真值”。
- 同时保存 min distance 与 contact vertex count，避免二值 mask 抹掉 person1 右手的边界接触信息。
- person2 构造必须核验 `trimmed == retargeted[42:178]`、object qpos 与 person1 同窗口，以及 `scene.xml/scene_act.xml` MuJoCo load。

### E077 实现进展

- [x] 新增脚本：
  - `workspace/core4d/scripts/E077/generate_core4d_contact_masks.py`
  - `workspace/core4d/scripts/E077/trim_box023_person2.py`
  - `workspace/core4d/scripts/E077/create_box023_person2_scene.py`
  - `workspace/core4d/scripts/E077/verify_box023_person2.py`
  - `workspace/core4d/scripts/E077/build_box023_person2.sh`
- [x] `py_compile` 与 `bash -n` 通过。
- [x] 已生成 3cm contact mask：
  - `workspace/core4d/results/E077/contact_masks/box023/raw_contact_mask_3cm.npz`
  - `workspace/core4d/results/E077/contact_masks/box023/raw_contact_mask_3cm.csv`
  - `workspace/core4d/results/E077/contact_masks/box023/audit_summary_3cm.json`

### E077 3cm mask 关键结果

| eval window | raw window | p1 L | p1 R | p2 L | p2 R |
|-------------|------------|------|------|------|------|
| 100-114 | 102-110 | 15/15, mean 0.81cm | 13/15, mean 2.42cm | 15/15, mean 0.11cm | 15/15, mean 0.11cm |
| 115-130 | 111-120 | 16/16, mean 0.74cm | 16/16, mean 2.12cm | 16/16, mean 0.17cm | 16/16, mean 0.11cm |
| 131-145 | 121-129 | 5/15, mean 13.46cm | 4/15, mean 11.14cm | 2/15, mean 25.04cm | 5/15, mean 15.26cm |

这复现并固化了 E076 结论：3cm 口径下 f115-f130 person1 右手为持续接触，但距离接近阈值边界；person2 双手强接触。

### E077 person2 构造结果

- [x] 修复 build 脚本环境：`convert_core4d_to_omniretarget.py` 需要先 source Holosoma `hsretargeting` conda 环境，否则找不到 `smplx`。
- [x] person2 retarget 完成：
  - `workspace/core4d/results/E077/holosoma_box023_person2/retargeted/20231008-045-person2-Box023_with_obj_original.npz`
  - qpos shape `(178,43)`，final cost 约 `0.574`。
- [x] person2 trim 完成：
  - `workspace/core4d/results/E077/holosoma_box023_person2/trimmed/20231008-045-person2-Box023_with_obj_original.npz`
  - qpos shape `(136,43)`，且 `trimmed == retargeted[42:178]`。
- [x] SPIDER case 完成：
  - `example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2/scene.xml`
  - `example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2/0/trajectory_kinematic.npz`
  - `example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2/scene_act.xml`
- [x] 核验完成：
  - `scene.xml`: `nq=43,nv=41,nu=29`
  - `scene_act.xml`: `nq=42,nv=41,nu=35`, euler `XZY`
  - `trajectory_kinematic.npz`: qpos `(136,43)`, qvel `(136,41)`, ctrl `(136,29)`, contact `(136,2)` 全 1。
- [x] 写入结果日志：`workspace/core4d/log/98_E077_3cm_contact_mask_and_person2_results.md`。

### E077 关键 caveat

converted 层 `person1/person2` 的 object pose 完全一致，但 retarget/SPIDER 层 object qpos 不完全一致：

- max abs diff `0.0621m`
- position diff mean `[0.00018, 0.02700, -0.00697]`
- quat diff max `0`

原因是 Holosoma preprocess 按每个人的 `smpl_scale` 缩放 object xy/z 轨迹。结论：`box023_person2` 可以作为单人 case 使用，但不能和现有 `box023_person1` retarget qpos 直接合并成双机器人同场景；双人合成前必须做 common-scale/common-world alignment。

---

## E078 计划

- [x] 用户确认下一轮方向：修改 3cm contact mask 并对齐 HDMI，在 `box023_person1` 和 `box023_person2` 两个单人 case 做 CEM 动力学重定向。
- [x] 按用户要求先写计划、不执行实现。
- [x] 已写入计划：`workspace/core4d/plan/83_E078_3cm_per_eef_contact_mask_cem_plan.md`。

### E078 计划摘要

- E078A: `box023_person1`，基于 E075B，读取 E077 3cm mask 的 `person_idx=0`。
- E078B: `box023_person2`，构造 E075B-like p2 配置，读取 E077 3cm mask 的 `person_idx=1`。
- 主改动：
  - MJWP contact_hdmi mask 从 scalar `(T,)` 改成 HDMI-style per-EEF `(T,2)`。
  - 增加 `contact_hdmi_mask_source="core4d_3cm"`，从 E077 npz 读取 mask。
  - 保持旧 `rotated_sdf` 默认行为，避免影响其他实验。
- 执行前待用户确认；当前未修改代码、未启动训练。

---

## E079 进展: 10+ 高接触质量 case 泛化验证

- [x] 按 `experiment-planning-zh` 恢复 `EXPERIMENT_TRACKER.md`、最新 plan/log、`progress.md`。
- [x] 复读 `log/99_E078_3cm_per_eef_mask_results.md`，确认本轮出发点：E078B/person2 证明算法在高质量 ref/contact 下可工作；E078A/person1 作为数据质量反例。
- [x] 复读 `log/64_E054_case_tier_analysis_results.md` 和 `workspace/core4d/data_preprocess/README.md`，确认 6 个 B+C raw 序列为首批候选：box021、box023、bucket001、bucket005_s2、bucket007、desk021。
- [x] 初步实验单位确定为 single-person case：对 6 个 B+C 序列构造 p1/p2 共 12 个候选，再用 3cm contact audit 筛选接触质量；`box023_person1` 保留为反例/guard，不计入成功样本。
- [x] 读取远程执行规则；本轮远程只使用 `spider-remote` 的 GPU1，GPU0 当前占用高，不纳入调度。
- [x] 写入 E079 总体计划：`workspace/core4d/plan/84_E079_core4d_generalization_10plus_plan.md`。
- [x] E079 决策：主验证不沿用 box023 固定 `hold_contact_start/end_eval_time=1.8-2.5s`；该配置只保留作 E078B calibration，不作为泛化主结论。
- [x] 将 E077 专用 scene 创建逻辑通用化为 `workspace/core4d/data_preprocess/create_spider_scene_from_template.py`，并让 `pipeline.sh` 调用通用脚本。
- [x] 新增 E079 数据预处理 case 列表：`cases_E079_existing_p1.tsv` 用于已有 p1 的 mask/trim audit；`cases_E079_build_p2.tsv` 用于构造 6 个 p2 单人 SPIDER case。
- [x] 新增 E079 实验 manifest 与脚本骨架：
  - `workspace/core4d/scripts/E079/variants.tsv`
  - `workspace/core4d/scripts/E079/generate_e079_overrides.py`
  - `workspace/core4d/scripts/run_E079_preprocess.sh`
  - `workspace/core4d/scripts/train/train_E079.sh`
  - `workspace/core4d/scripts/run_E079_remote.sh`
  - `workspace/core4d/scripts/pull_E079_remote_results.sh`
  - `workspace/core4d/scripts/eval/eval_E079.py`
  - `workspace/core4d/scripts/eval/eval_E079_contact_quality.py`
- [x] 更新 `workspace/core4d/data_preprocess/README.md`，明确 `create_spider_scene_from_template.py` 是通用 scene 创建入口，E079 只是新增 case TSV。
- [x] E079 p1 轻量预处理完成：使用已知 Holosoma trim window 跳过 retarget，为 6 个已有 p1 case 生成 `workspace/core4d/results/E079/contact_masks/*_person1/raw_contact_mask_3cm.npz`。
- [x] 修复 `generate_core4d_contact_masks.py` 的短序列 audit 打印 bug：固定窗口越界时跳过打印，避免 `None` format 崩溃。
- [x] E079 p2 构造进展：`box021_person2`、`box023_person2`、`bucket001_person2`、`bucket005_s2_person2`、`bucket007_person2` 已完成 SPIDER scene/trajectory/scene_act verify；`desk021_person2` 在 Holosoma retarget 中 `CVXPY solve failed: infeasible`，标记为 `preprocess_fail`，后续训练跳过。
- [x] 生成 11 个 E079 per-case override（10 main + 1 guard）；Hydra compose 验证全部指向 E079 contact masks，且主验证 `hold_contact_rew_scale=0.0`。
- [x] 发现并修正 E079 p2 构造中的一个复用风险：`box023_person2` 已由 E077/E078 验证，不应在 E079 build-p2 阶段被 auto trim 重建覆盖；已把它移到 `cases_E079_existing_p2.tsv`，`cases_E079_build_p2.tsv` 中禁用该行，并恢复 tracked SPIDER 数据到既有版本。
- [x] 将通用 `pipeline.sh` 的 trim-window 控制流修正为：TSV 显式写 `trim_start/trim_frames` 时直接作为权威窗口，只有 `auto` 才从 Holosoma untrimmed/trimmed 反查；新增 `write_trim_window.py` 写入显式窗口 provenance。
- [x] 重新生成 E079 `box023_person2` contact mask：`trim_start=42, trim_frames=136`，`eval_contact_mask_3cm` 长度恢复为 227；115-130 帧 audit 显示 person2 left/right 均为 `16/16` 接触。
- [x] 更新 `workspace/core4d/data_preprocess/README.md`：补充显式 trim-window 语义、`write_trim_window.py`、`cases_E079_existing_p2.tsv`，并修正 `pipeline.sh` 处理步骤说明。
- [x] 重新跑 E079 contact-quality 汇总：11 个可用 main/guard case 均为 `high_quality_proxy=True`，`box023_person2` 行为 `T_spider=136, trim_start=42`。
- [x] 本机 RTX 5090 短 horizon smoke 通过：`core4d_E079_box023_p2`, `max_sim_steps=4`, E079 mask `eval_contact_mask_3cm` 读取成功，修正后 active L/R=45.0%/47.5%，输出 `/tmp/e079_smoke_box023_p2/trajectory_mjwp_act.npz`。
- [x] 修正 E079 preprocess 默认入口：已有 p1 默认 `--skip-retarget --skip-spider`，只做显式窗口/contact mask；`desk021_person2` 已知 retarget infeasible，默认从 build-p2 TSV 禁用，避免一键 `all` 重复失败。
- [x] 静态检查通过：E079/data_preprocess Python `py_compile`、bash `-n`、`git diff --check`。
- [x] 启动前修复 E079 train 脚本 bug：`awk -v split=...` 会和 awk 内置 `split()` 冲突，改为 `want_split`，并增加空 split 保护。之前本机/远程首轮启动因此未真正跑 variant。
- [x] 远程重启后发现 `box021_person2` scene 引用的 CORE4D object mesh 未同步到远程；补充纳入 E079 涉及的 object assets：box021、box023、bucket001、bucket005、bucket007、desk021。
- [x] E079 正式训练完成：本地 RTX5090 串行跑完 6 个 local variant（含 `box023_p1` guard 和 `desk021_p1`），远程 `spider-remote` 仅 GPU1 跑完 5 个 p2 main variant。
- [x] 远程结果已通过 `workspace/core4d/scripts/pull_E079_remote_results.sh` 回收到本地，并完成合并评估：`workspace/core4d/results/E079/comparison.csv`、`aggregate_summary.json`。
- [x] 合并数值结果：11 个结果中 10 个 main + 1 个 guard；main numeric success = `3/10 = 30%`，成功 case 为 `desk021_p1`、`box023_p2`、`bucket007_p2`；`box023_p1` 作为 E078 数据质量反例 guard，不计入成功率。
- [x] 已生成 11 个视频 contact sheet：`workspace/core4d/results/E079/keyframes/contact_sheets/*_sheet.jpg`，并启动 subagent 分组做可视化复核。
- [x] E079 结果日志已写入：`workspace/core4d/log/100_E079_core4d_generalization_10plus_results.md`；`EXPERIMENT_TRACKER.md` 已补 E079 总览、结果索引和脚本索引。
- [x] 按用户要求补充 E079 量化指标表说明：解释 `Numeric`、各指标含义、post2 统计窗口，以及指标来源于 `eval_E079.py`/`eval_E078.py` 和 `comparison.csv`。
- [x] 用户指出 `post2=2.0s-3.6s` 是 box023 派生的接触/放置诊断窗口，不能直接作为多 case 泛化评估窗口；已修正 E079 log 和 tracker，将 `Numeric=3/10` 降级为 fixed-window diagnostic，C3 改为需要 per-case contact/intent window 后再正式验证。
- [x] 已按用户要求实现并重跑 case-specific window 评估：`workspace/core4d/scripts/eval/eval_E079.py` 现在从每个 variant 的 `eval_contact_mask_3cm` 中取对应 person 左右手 OR 的首次/末次 active frame，并 padding 10 frame 作为 `case_window_*`；已更新 `comparison.csv`、`aggregate_summary.json`、每个 `eval_summary_E079_*.json/csv`、E079 log 和 tracker。
- [x] 用户纠正 E079 role：`E079_box023_p2` 才是 E078 positive guard（已验证成功样本），`E079_box023_p1` 是 main 中的已知失败/数据质量反例。已修正 `workspace/core4d/scripts/E079/variants.tsv` 并重跑 eval：guard_results=`[E079_box023_p2]`，main case-window 仍为 `6/10`，fixed post2 旧口径变为 `2/10`；`box023_p1` 记录为 visual false positive。
- [x] 用户进一步复查 `bucket007_p1` 视频：开头疑似第一视角摄像头开始录制的摸头/按摄像头前摇，trim 未剔除干净；同期 ref 右脚也疑似未稳定接地，属于 retarget/动捕支撑脚误差。sim 为避免单脚不稳左脚后撤一步，导致整体初始位置比 ref 后退并早期未接触物体，但后续 CEM 有弥补，后半段接触基本正常。已更新 E079 log/tracker：`bucket007_p1` 从简单 visual false positive 改为 trim/ref data-quality issue。
- [x] 用户继续复查 E079 视觉质量并修正分类：`box021_p1` 视觉质量很好，但 ref 接触位置本身奇怪；`desk021_p1` 不是特别好，前段没抬起来、后段才相对正常；`bucket005_s2_p1` 不好，物体持续受机器人力而旋转。已更新 E079 log/tracker：`box021_p1` 提升为视觉正例但需 ref/contact 解释，`desk021_p1` 降为 partial，`bucket005_s2_p1` 标为 case-window overestimate。
- [x] 注意：progress 中早期关于 E079 的 `box023_p1 guard`、fixed-post2 `3/10`、以及 `desk021_p1/bucket005_s2_p1 near-pass` 等条目是中间口径，已被上述 role 修正、case-specific window 复算和用户视觉复查覆盖；最终结论以 `log/100_E079_core4d_generalization_10plus_results.md` 为准。

---

## E080 进展: box025 边界/负控 case 复查

- [x] 使用 `experiment-planning-zh` 恢复 E079 最终日志、E079 plan、tracker、progress 和远程执行规则。
- [x] 回顾 E079 结论：E077 pipeline 已可泛化；下一步问题不是继续手写 hold window，而是加入更强成功判据、trim/ref feasibility audit，并用更多边界 case 校准 false positive。
- [x] 根据用户要求把 E080 第一实验定为 `box025`。历史 E054 已把 `box025_p1/p2/box025_s2_p1` 标为 Tier 3 / drop，原因是物体 `dim_max=0.89m > 0.70m`，因此 E080 的 box025 应作为“大物体结构性负控/边界 case”，不应预设为成功正例。
- [x] 核实本地 `box025_person1` 与 `box025_person2` 都已有 SPIDER case；`trajectory_kinematic.npz` 均为 124 帧，且与 Holosoma `retarget_replace_batch_trimmed` 完全一致。
- [x] 核实 Holosoma untrimmed→trimmed 窗口：`box025_person1/person2` 都是 `trim_start=38, trim_frames=124`，可直接用 E077/E079 pipeline 生成 3cm mask，不需要重跑 Holosoma retarget。
- [x] 已写入 E080 计划：`workspace/core4d/plan/85_E080_box025_boundary_control_plan.md`。E080 将 `box025_person1/person2` 作为 Tier 3 大物体边界/负控，用 E079 no-hold + 3cm mask + case-specific window 口径复查，而不是预设成功。
- [x] 根据 subagent 只读审查修正 E080 plan：`box025_person2` 的 scene/trajectory、`box025_person1` 的 trajectory 以及 `example_datasets/processed/core4d/assets/objects/box025/box025_m.obj` 都在 `.gitignore` 下，远程启动前必须 `git add -f` 纳入活跃 case 数据，不能假设远程已有。
- [x] E080 静态检查通过：`py_compile` 覆盖 E080 override/eval 以及 E079 eval 复用模块，`bash -n` 覆盖 E080 preprocess/train/remote/pull，`git diff --check` 通过。
- [x] E080 预处理完成：生成 `workspace/core4d/results/E080/contact_masks/box025_person1` 和 `box025_person2` 的 3cm mask；两者 `trim_start=38`、`spider_contact_mask_3cm.shape=(124,2,2)`、`eval_contact_mask_3cm.shape=(207,2,2)`。目标 person 的 any-active p1=`63.7%`、p2=`62.1%`。
- [x] E080 override 已生成：`core4d_E080_box025_p1.yaml` 与 `core4d_E080_box025_p2.yaml`，均继承 E079 no-hold 口径，palm normal 自动为 left `[0,-1,0]`、right `[0,1,0]`。
- [x] E080 短 horizon smoke 通过：
  - `core4d_E080_box025_p1`, `max_sim_steps=4`, mask `eval_contact_mask_3cm` len `207->298`, active L/R=`53.0%/60.4%`。
  - `core4d_E080_box025_p2`, `max_sim_steps=4`, mask `eval_contact_mask_3cm` len `207->298`, active L/R=`62.1%/62.1%`。
- [x] 已提交并推送 E080 setup：commit `711ce08 exp(core4d): E080 box025 boundary-control setup`。
- [x] 远程 `spider-remote` GPU1 已启动 `E080_box025_p2`，tmux session=`E080`；远程 `git pull --ff-only` 成功，活跃 box025 ignored 数据已随 git 同步。Git LFS 提示 3 个文件应为 pointer 但不是，本轮按普通 git blob 使用，不影响当前远程读取。
- [x] 本地首次 `train_E080.sh local 0` 在 sandbox 中看不到 CUDA 失败；已按权限规则用提升权限重启，本机 RTX5090 正常识别，`E080_box025_p1` 已开始运行。
- [x] E080 p1/p2 正式运行完成：本地 `E080_box025_p1` 与远程 GPU1 `E080_box025_p2` 均产出 `.npz/.mp4`，远程结果已回收到本地。
- [x] E080 统一评估完成：case-window main success `2/2=100%`，fixed post2 numeric success `0/2=0%`。p1 case-window `obj mean/max=0.184/0.336m, sim contact=75.3%`；p2 `0.146/0.289m, sim contact=90.8%`。
- [x] E080 视觉复核完成：subagent 判断 p2 明显好于 p1，但 p1/p2 都不能算真实搬运成功；p1 是趴箱/贴箱/推箱 false positive，p2 更像扶/推箱体。
- [x] 已写入结果日志：`workspace/core4d/log/101_E080_box025_boundary_control_results.md`；已更新 `EXPERIMENT_TRACKER.md`。核心结论：box025 继续支持 Tier3/drop 历史判断，同时证明 case-window 三阈值会误判大物体负控。
- [x] E080 二次复核：用户指出 p2 视觉上像搬箱子，该观察成立；已修正 log/tracker 表述。p1 仍是 false positive；p2 改标为 partial positive / near-usable。几何证据：scene 只含 `left_hand_object/right_hand_object/object_floor`，无腿/脚-箱 contact pair；p1 腿/脚 adjusted SDF min `-13.7cm`、穿入帧 `40.7%`，p2 min `-4.6cm`、穿入帧 `20.2%`。因此腿不会物理支撑箱子，但 p1/p2 都有不同程度视觉/几何干涉，下一步 eval 应加入 leg-box interference 与 object lift/floor-contact。

---

## E081 进展: leg/foot-object collision 派生 scene 验证

- [x] 已按用户约束写入计划：不直接修改原始 `scene_act.xml`，新建 `box025_person2_legobj` 与 `box023_person2_legobj` 派生任务；计划文件 `workspace/core4d/plan/86_E081_leg_object_collision_eval_plan.md`。
- [x] 新增 E081 脚本：`create_legobj_cases.py`、`generate_e081_overrides.py`、`run_E081_preprocess.sh`、`train_E081.sh`、`run_E081_remote.sh`、`pull_E081_remote_results.sh`、`eval_E081.py`。
- [x] E081 预处理完成：两个派生 task 均从原始 task 复制 scene/data，并只在派生 `scene_act.xml` 加 16 个腿/脚-`object_collision` pair；原始 `box025_person2/scene_act.xml` 与 `box023_person2/scene_act.xml` 无 diff。
- [x] E081 override 生成完成：`core4d_E081_box025_p2_legobj.yaml` 使用 E080 的 `box025_person2` mask，`core4d_E081_box023_p2_legobj.yaml` 使用 E079 的 `box023_person2` mask，均保持 no-hold 口径。
- [x] E081 静态检查通过：Python `py_compile`、bash `-n`、`git diff --check` 均通过。
- [x] E081 短 horizon smoke 通过：
  - `core4d_E081_box025_p2_legobj`, `task=box025_person2_legobj`, `max_sim_steps=4`, final object pos err `0.0406m`。
  - `core4d_E081_box023_p2_legobj`, `task=box023_person2_legobj`, `max_sim_steps=4`, final object pos err `0.0100m`。
- [x] E081 setup 已提交推送：commit `f26e3be exp(core4d): E081 leg-object collision setup`；远程 `spider-remote` GPU1 已启动并完成 `E081_box023_p2_legobj`，本地 RTX5090 已完成 `E081_box025_p2_legobj`。
- [x] E081 合并评估完成：`workspace/core4d/results/E081/comparison.csv`。聚合结果：main case-window `1/1=True`，main leg/lift proxy `0/1=False`，guard=`E081_box023_p2_legobj`。
- [x] E081 关键量化结论：box025 p2 case-window 腿/箱 interference 从 E080 baseline `28.9%` 降到 `7.5%`，最小 adjusted SDF `-4.6cm -> -1.2cm`，obj mean/max `0.146/0.289 -> 0.143/0.271`；但 object bottom mean `-0.073m -> -0.075m`，floor-contact `60.1% -> 59.5%`，说明 lift/floor-contact 未改善。box023 p2 guard 基本稳定，obj mean `0.162 -> 0.164`，leg interference `0 -> 2.7%`。
- [x] E081 结果日志已写入：`workspace/core4d/log/102_E081_leg_object_collision_results.md`；tracker 已更新。核心结论：新增腿/脚-箱碰撞是必要物理修正，但 box025_p2 主要瓶颈已转向 object lift/floor-contact，不是继续修腿穿模。
- [x] 已按用户要求补充 E081 指标定义：`Leg intf` / `leg_box_interference_frames_pct`、`Leg contact` / `leg_object_contact_frames_pct`、`near_2cm`、`object_floor_contact_frames_pct`、`object_bottom_proxy_m` 的计算口径和解释均写入 log 102。
- [x] 已补充 E081 脚本路径与可复现实验命令：指标脚本 `eval_E081.py`、派生 scene/override 生成脚本、train/local/remote/pull/eval/single 命令均写入 log 102。
- [x] 已补充 E081 机制分析：E081 没有新增显式腿避障 reward/优化器改动，改善来自腿/脚-箱 contact pair 改变 MuJoCo 前向动力学，使穿箱控制序列在现有 objective 下间接受罚并被 CEM elite selection 淘汰。

---

## E082 进展: E081 路线跑 3 个 D003 Box021 case

- [x] 用户纠正本轮目标工作区为 `workspace/core4d`，不是 `workspace/v2`；E082 将在 core4d 工作区推进。
- [x] 已回顾 E077-E081：
  - E077 生成 CORE4D 3cm per-person/per-hand contact mask，并构造 `box023_person2`；
  - E078 将 MJWP contact mask 改成 HDMI-style per-EEF mask，验证 p2 数据质量明显好于 p1；
  - E079 将 E077 pipeline 扩到 10+ single-person case，并改用 case-specific contact/intent window；
  - E080 发现 box025 p2 是 partial positive，但原 scene 没有腿/脚-箱 contact pair；
  - E081 新建 `*_legobj` 派生 scene，证明新增腿/脚-物体 contact pair 可显著减少腿/箱穿入，且不污染原始 task。
- [x] 已检查 3 个目标 source task：
  - `d003_box021_20231011_035_p2`
  - `d003_box021_20231018_029_p2`
  - `d003_box021_20231020_019_p1`
  三者均有 `scene.xml`、`scene_act.xml`、`scene_act_meta.json`、`task_info.json`、`0/trajectory_kinematic.npz`，且 `scene_act.xml` 只有 `left_hand_object/right_hand_object/object_floor`，没有腿/脚-物体 pair。
- [x] 已确认 3 个 3cm mask 存在：`workspace/core4d_collab_retarget/results/E029/d6/contact_masks/<source_task>/raw_contact_mask_3cm.npz`，其 `spider_contact_mask_3cm` 长度分别为 `133/75/98`，与 source `trajectory_kinematic.npz` 对齐。
- [x] 已写入 E082 计划：`workspace/core4d/plan/87_E082_d003_box021_e081_legobj_plan.md`。计划采用 E081 非-freejoint scene_act 路线，派生 `*_legobj_e082`，三卡并行，本地 1 卡 + 远程 2 卡；远程同步使用按需 `rsync`，不要求清理另一个工作区的未提交 E030 记录。
- [x] 已新增并静态检查 E082 脚本：
  - `workspace/core4d/scripts/E082/variants.tsv`
  - `workspace/core4d/scripts/E082/create_legobj_cases.py`
  - `workspace/core4d/scripts/E082/generate_e082_overrides.py`
  - `workspace/core4d/scripts/E082/run_remote_inside.sh`
  - `workspace/core4d/scripts/run_E082_preprocess.sh`
  - `workspace/core4d/scripts/train/train_E082.sh`
  - `workspace/core4d/scripts/run_E082_remote.sh`
  - `workspace/core4d/scripts/pull_E082_remote_results.sh`
  - `workspace/core4d/scripts/eval/eval_E082.py`
- [x] E082 静态检查通过：新增 Python `py_compile`、新增 shell `bash -n`、`git diff --check`。
- [x] E082 预处理完成：3 个派生 task 均生成并各自新增 16 个腿/脚-`object_collision` pair；派生 scene_act 均可由 MuJoCo 加载，`nq/nv/nu=42/41/35`，`npair=42`。
- [x] E082 overrides 已生成，均继承 `core4d_e074a_box023`、使用 `workspace/core4d/results/E082/contact_masks/<source_task>/raw_contact_mask_3cm.npz`、`hold_contact_rew_scale=0.0`，palm normal 自动计算为 left `[0,-1,0]`、right `[0,1,0]`。
- [x] E082 短 horizon smoke 通过，3 个 variant 均可加载派生 `scene_act.xml`、E082 3cm per-EEF mask 和 override，并生成 `/tmp/e082_smoke_<variant>/trajectory_mjwp_act.npz`：
  - `E082_d003_box021_20231018_029_p2_legobj`: mask `125->200`，active L/R=`68.5%/74.0%`，final object pos/quat err=`0.0395/0.0012`。
  - `E082_d003_box021_20231011_035_p2_legobj`: mask `222->316`，active L/R=`75.3%/75.6%`，final object pos/quat err=`0.0440/0.0268`。
  - `E082_d003_box021_20231020_019_p1_legobj`: mask `164->246`，active L/R=`61.4%/62.6%`，final object pos/quat err=`0.0349/0.0101`。
- [x] E082 full CEM 已完成：本地 GPU0 跑 `20231018_029_p2`，远程 GPU0/GPU1 跑 `20231011_035_p2` 与 `20231020_019_p1`，三路结果已回收。
- [x] 2026-05-27 22:49 已启动 E082 full CEM：
  - 本地 tmux `E082_local_d003_box021`: `E082_d003_box021_20231018_029_p2_legobj` on GPU0。
  - 远程 tmux `E082_remote_d003_box021`: `remote-gpu0` 跑 `E082_d003_box021_20231011_035_p2_legobj`，`remote-gpu1` 跑 `E082_d003_box021_20231020_019_p1_legobj`。
  - 注意三张卡同时存在 R108/R109/R110 Holosoma RL 训练负载；E082 已正常加载并开始优化，但运行时长可能受资源竞争影响。若后续出现 OOM/超慢，需要把它记为运行资源问题而不是方法失败。
- [x] 2026-05-27 22:54 监控：本地 `44/150`，远程 GPU0 `52/266`，远程 GPU1 `40/196`；均正常前进，尚无 `.npz/.mp4` 落盘。
- [x] 2026-05-27 22:58 监控：本地 `68/150`，远程 GPU0 `74/266`，远程 GPU1 `68/196`；三路 tmux 均存活，尚无结果落盘。
- [x] 2026-05-27 23:04 监控：本地 `108/150`，远程 GPU0 `112/266`，远程 GPU1 `116/196`；三路均过半，尚无最终 `.npz/.mp4`。
- [x] 本地 `E082_d003_box021_20231018_029_p2_legobj` 已完成并落盘 `.npz/.mp4`，运行总时长约 `1120.6s`；run 日志 final object tracking error `pos=0.5928, quat=0.2631`。统一 eval 初步显示该 case 明确失败：case-window object mean/max `0.680/1.138m`，sim contact `14.7%`，leg interference `69.8%`，pelvis_z_min `0.355m`。
- [x] 发现并修复 `eval_E082.py` 路径 bug：train/pull 脚本传入相对 `RESULTS` 时，复用 E078 eval 的 `relative_to(REPO)` 会报错；已改为在 E082 wrapper 内把 `RESULTS` 与 `VARIANTS_FILE` 解析为 repo 绝对路径，并已 rsync 到远程。
- [x] 远程 GPU1 `E082_d003_box021_20231020_019_p1_legobj` 已完成并自动 eval 成功；run 日志 final object tracking error `pos=0.7659, quat=0.1605`，初步判断也是失败。远程 GPU0 `E082_d003_box021_20231011_035_p2_legobj` 已到 `250/266`，等待收尾。
- [x] 远程 GPU0 `E082_d003_box021_20231011_035_p2_legobj` 已完成，run 日志 final object tracking error `pos=0.4112, quat=0.2128`；远程 tmux `E082_remote_d003_box021` 正常退出。
- [x] 已通过 `workspace/core4d/scripts/pull_E082_remote_results.sh` 回收远程结果并重跑本地合并 eval。`workspace/core4d/results/E082/aggregate_summary.json`: `num_results=3`, `main_case_window_success_pct=0.0`, `main_legobj_strict_proxy_success_pct=0.0`。
- [x] 已用 `/video-frames` skill / ffmpeg 生成 E082 视频 keyframe sheets：
  - `workspace/core4d/results/E082/keyframes/contact_sheets/E082_d003_box021_20231018_029_p2_legobj_sheet.jpg`
  - `workspace/core4d/results/E082/keyframes/contact_sheets/E082_d003_box021_20231011_035_p2_legobj_sheet.jpg`
  - `workspace/core4d/results/E082/keyframes/contact_sheets/E082_d003_box021_20231020_019_p1_legobj_sheet.jpg`
  - `workspace/core4d/results/E082/keyframes/contact_sheets/E082_all_cases_sheet.jpg`
- [x] 已写入正式结果日志：`workspace/core4d/log/103_E082_d003_box021_e081_legobj_results.md`。结论：E082 数据/派生 scene/运行链路跑通，但 3 个 D003 Box021 全失败，视觉均为倒伏/推箱/压箱/物体漂移；不建议把 E082 输出接后续 RL。
- [x] 已更新 `workspace/core4d/EXPERIMENT_TRACKER.md` 的 E082 行和脚本/结果索引。
- [x] 已针对用户观察的“弯腰搬箱时趴倒、手撑地、029 头部栽进箱子”完成上半身穿模诊断：`workspace/core4d/log/104_E082_body_fall_upperbody_collision_diagnosis.md`。结论：E081/E082 派生 scene 只新增腿/脚-物体 pair，未新增 `head_collision/torso_collision/pelvis_collision/shoulder/elbow` 与 `object_collision` 的 pair；手-地面 pair 已存在，所以 CEM 可用“头/躯干穿箱 + 手撑地”满足局部 objective。三个 Box021 失败 case 的 sim head/torso 穿入率分别为 `76.0/85.3%`、`17.2/51.1%`、`32.4/58.8%`，而 ref head/torso SDF 仍为正；box023 guard 同样无上半身 pair 但没有穿模，主要因为物体更小、p2 ref/contact 更可行。下一步推荐 E083A 先加 upper-body-object collision pairs，再视结果加 upperbody SDF penalty、hand-floor penalty、stability/ctrl guard。

---

## E083 进展: upper-body-object collision pairs

- [x] 已读取 `workspace/core4d/log/104_E082_body_fall_upperbody_collision_diagnosis.md` 和 `experiment-planning-zh/remote-execution.md`，确认本轮按 E083A 推进：只在派生 scene 中增加 upper-body-object collision pairs，先隔离验证碰撞约束本身。
- [x] 已写入 E083 计划：`workspace/core4d/plan/88_E083_upperbody_object_collision_plan.md`。实验矩阵为 3 个 Box021 main + `box023_p2` guard；本地 GPU0 跑 `20231018_029_p2`，远程 GPU0 跑 `20231011_035_p2`，远程 GPU1 串行跑 `20231020_019_p1` 与 guard；不 kill 其他已有实验。
- [x] 已新增 E083 脚本集：`workspace/core4d/scripts/E083/variants.tsv`、`create_upperobj_cases.py`、`generate_e083_overrides.py`、`run_remote_inside.sh`、`run_E083_preprocess.sh`、`train_E083.sh`、`run_E083_remote.sh`、`pull_E083_remote_results.sh`、`eval_E083.py`、`extract_E083_contact_sheets.sh`。
- [x] E083 静态检查通过：`py_compile` 覆盖 E083 Python 与 E082 诊断脚本，`bash -n` 覆盖 E083 shell，`git diff --check` 通过。
- [x] E083 预处理完成：4 个 `*_upperobj_e083` 派生 task 均生成；每个派生 `scene_act.xml` 可由 MuJoCo 加载，`nq/nv/nu/npair=42/41/35/49`，且包含 16 个腿/脚-`object_collision` pair 和 7 个 upper-body-`object_collision` pair。4 个 override 已生成，contact mask 已复制到 `workspace/core4d/results/E083/contact_masks/`。
- [x] E083 短 horizon smoke 通过：4 个 variant 均用 `max_sim_steps=4` 成功加载派生 scene/override/mask 并产出 `/tmp/e083_smoke_<variant>/trajectory_mjwp_act.npz`。
- [x] 2026-05-28 00:10 已启动 E083 full CEM：本地 tmux `E083_local_upperobj` 跑 `E083_d003_box021_20231018_029_p2_upperobj` on GPU0；远程 tmux `E083_remote_upperobj` 已通过按需 `rsync` 同步脚本、overrides、derived scenes、contact masks 和 assets 后启动，remote GPU0 跑 `20231011_035_p2`，remote GPU1 串行跑 `20231020_019_p1` 与 `box023_p2` guard。启动过程未 kill 任何已有 session。
- [x] E083 eval 小修：`first_case_window_*` 首帧指标改为严格限制在 case window 内；`eval_E083.py` 已重新 `py_compile` 并 rsync 到远程，不影响正在跑的 CEM，后续本地合并 eval 会覆盖远程临时 summary。
- [x] 2026-05-28 00:15 监控：本地 `20231018_029_p2` 到 `98/150`，远程 GPU0 `20231011_035_p2` 到 `78/266`，远程 GPU1 `20231020_019_p1` 到 `52/196`；三路 tmux 均存活，尚无 `.npz` 落盘。
- [x] 本地 `E083_d003_box021_20231018_029_p2_upperobj` 已完成并自动 eval，产出 `.npz/.mp4/keyframes`。初步指标仍失败：case-window obj mean `0.608m`，pelvis z min `0.478m`，sim contact `69.0%`，head/torso penetration `38.8/27.9%`，upperbody any penetration `82.2%`，LH floor contact `10.1%`，`E083_success_upperbody_physical_proxy=False`。这说明“加 pair”没有直接消除上半身穿箱/压箱局部解，后续需结合视频和远程结果确认是否是 solver 允许较大穿入、碰撞体半径口径、或 CEM 转为上身压箱。
- [x] 2026-05-28 00:27 监控：远程 GPU0 `20231011_035_p2` 到 `194/266`，远程 GPU1 `20231020_019_p1` 到 `116/196`；远程 tmux 仍存活，guard 尚未启动，远程 `.npz` 数量仍为 0。
- [x] 本地 E083 029 视觉复核完成：subagent high 判断 E083 确实避免了 E082 那种头/身体深度栽进箱体和后段箱体大翻滚，但从 f50 起仍明显倒伏/趴箱，f75-f149 基本是上身压在箱顶/箱沿，头颈/上胸仍有浅穿或卡边。时序指标一致：head f35 首次浅穿，torso f94 后浅穿，LH f137 开始撑地；pelvis 不再低于 45cm，leg-box interference 从 E082 `69.8%` 降到 `0%`，但 object-floor contact 仍 `94.6%`、object bottom 比 ref 低 `11.8cm`。结论：E083A 只加 collision pair 把“深穿箱”改成“被箱子挡住后趴箱/压箱”，没有解决站立支撑与搬运策略。
- [x] E083 远程结果已回收：remote GPU0 `20231011_035_p2` 于 `00:36` 完成，remote GPU1 `20231020_019_p1` 于 `00:38` 完成、`box023_p2` guard 于 `01:12` 完成；`workspace/core4d/scripts/pull_E083_remote_results.sh` 已拉回 3 个远程 `.npz/.mp4/logs` 并在本地重跑合并 eval。
- [x] E083 完整评估完成：`workspace/core4d/results/E083/aggregate_summary.json` 为 `num_results=4`、main case-window success `0/3=0%`、main upperbody physical proxy success `0/3=0%`、guard=`E083_box023_p2_upperobj_guard`。`comparison.csv` 与 `upperbody_diagnostics.csv` 已生成。
- [x] E083 可视化完成：`workspace/core4d/scripts/eval/extract_E083_contact_sheets.sh` 已生成 4 个 per-case sheet 和 `E083_all_cases_sheet.jpg`；subagent high 全量视觉复核结论：3 个 Box021 均不可用，标签分别为“趴箱/穿箱型失败”、“趴箱+腿部干涉严重”、“跌倒/手撑地+头部穿箱”；`box023_p2` guard 可用且未退化。
- [x] 已写入 E083 正式结果日志：`workspace/core4d/log/105_E083_upperbody_object_collision_results.md`；已更新 `EXPERIMENT_TRACKER.md`。核心结论：upper-body-object pairs 对 guard 安全、能减少 E082 的深穿箱，但不能解决 Box021 的错误接触语义，E083A 不可作为 RL seed。
- [x] 已写入下一步 E084 计划：`workspace/core4d/plan/89_E084_box021_constraint_groups_plan.md`。规划 3 组实验：A safety penalty（upperbody/hand-floor/stability）、B upright/ctrl trust（stronger ctrl guard + task_body + 降低 contact/object 牵引）、C semantic hand contact + lift（hand-only gate + object lift/floor penalty）。每组先跑 `20231018_029_p2` main + `box023_p2` guard，三卡并行；若某组有效，再 E085 扩展到 3 个 Box021 main。

---

## E084 进展: Box021 constraint groups main gate

- [x] 已按 E084 plan 改为 main-gate 策略：只在 `d003_box021_20231018_029_p2` 主 case 上先跑 A/B/C 三组；如果没有组通过，则不跑 `box023_p2` guard。
- [x] 已新增 reward/config 支持：`hand_floor_penalty_*`、`object_lift_rew_*`、`object_floor_penalty_*`；并复用既有 `robot_object_penalty_*` 做 upperbody-object SDF penalty。
- [x] 已新增 E084 脚本集：`workspace/core4d/scripts/E084/variants.tsv`、`generate_e084_overrides.py`、`run_remote_inside.sh`、`run_E084_preprocess.sh`、`train_E084.sh`、`run_E084_remote.sh`、`pull_E084_remote_results.sh`、`eval_E084.py`、`extract_E084_contact_sheets.sh`。
- [x] E084 预处理完成：6 个 overrides 已生成，main/guard mask 均复制到 `workspace/core4d/results/E084/contact_masks/`；A/B/C 三组短 smoke 通过。
- [x] 2026-05-28 02:06 启动 full CEM：本地 tmux `E084_local_main` 跑 A；远程 tmux `E084_remote_main_gate` 跑 B/C。B 首次远程 run 触发 `task_body_rew` shape bug（8 个 task bodies vs 32 个 full-body FK）。
- [x] 已修复 B 的 shape bug：`spider/simulators/mjwp.py` 在 `body_xpos_ref` 为 full-body FK 时按 `task_body_ids` 切回目标 body；本地 optimizer smoke 通过后，远程 `E084_remote_B_retry` 重跑 B 成功。
- [x] E084 三组 main 完成并回收远程结果；本地 merged eval 完成：`num_main_results=3`、`accepted_groups=[]`、`guard_splits_to_run=[]`、`stop_before_guard=true`。
- [x] E084 关键量化：A safety pelvis min `0.596m`、hand-floor `0/0%`，但 upperbody penetration `80.6%`、object-floor `93.8%`；B upright obj mean `0.417m`、bottom gap `-4.3cm`，但 contact `9.3%`、LH floor `39.5%`、视觉翻箱；C semantic/lift upperbody penetration `82.2%`、object-floor `96.9%`、bottom gap `-11.9cm`。
- [x] E084 可视化完成：`workspace/core4d/results/E084/keyframes/contact_sheets/` 生成 A/B/C sheets；subagent high 复核结论：A/C 是“稳定但不抬/蹲抱压箱”，B 是“明显翻箱/接触失控”，三者均不可作为 RL seed。
- [x] 已写入 E084 结果日志：`workspace/core4d/log/106_E084_box021_constraint_groups_results.md`；已更新 `EXPERIMENT_TRACKER.md`。
- [x] 已写入下一步 E085 计划：`workspace/core4d/plan/90_E085_box021_exit_gate_seed_routes_plan.md`。E085 规划三组：A feasibility/seed audit，B kinematic/support seed route，C hard-gate staged CEM；第一阶段先做审计，不直接继续 full CEM 小调参。

---

## E084 追加诊断: contact target 语义核查

- [x] 已新增并运行 `workspace/core4d/scripts/eval/audit_E084_contact_target_semantics.py`，输出到 `workspace/core4d/results/E084/contact_target_audit/`。
- [x] 修正坐标解释：raw mesh local 不能直接与 MuJoCo object-local 比，必须先应用 `object_visual` 的固定 `geom_pos/geom_quat`；正确 MuJoCo object frame 中 box 底面是 `-y`，顶面是 `+y`，不是 local `-z`。
- [x] mask 核查完成：E084 auto 选择 `eval_contact_mask_3cm`，`125->200`，person2 active L/R=`68.5/74.0%`，qpos active frame `41-188` 对应 raw frame `75-129`；未发现错人/错帧。
- [x] raw 接触点核查完成：person2 left/right broad hand min dist mean `1.8/2.2mm`，fingertip min dist mean `7.6/9.6mm`；raw left 是低侧面 contact，raw right 是高侧面 contact。
- [x] G1 target 核查完成：当前 dynamic target 是 `wrist_yaw_link + [0.05,0,0]`，不是 raw fingertip/contact centroid；G1 ref target 与 raw surface centroid 平均差约 `27cm`，主要沿 object `x` 轴在对侧。结论：问题主要在 G1 wrist pseudo contact target 失真，不是 contact mask 二值门控本身。
- [x] sim 实际手部补查完成：E084A/C 手不再撑地并能以约 `7cm` 均值追随 dynamic target，但 target 本身语义偏 raw；E084B 没追上目标且出现低手/撑地失败模式。
- [x] 已写入正式诊断日志：`workspace/core4d/log/107_E084_contact_target_semantics_audit.md`；已更新 `EXPERIMENT_TRACKER.md`。

---

## E085 进展: raw contact target 修复与 gate

- [x] 已按用户追问继续做全面检查：contact mask/person/time 未发现错人错帧，`eval_contact_mask_3cm` 为 person2，`125->200`，active L/R=`68.5%/74.0%`。
- [x] 已确认旧 E084 dynamic target 是 G1 `wrist_yaw_link + [0.05,0,0]`，不是 raw fingertip/contact centroid；old G1 target 与 raw surface target 平均差约 `27cm`。
- [x] 已实现 external contact target：`examples/run_mjwp.py` 支持从 `.npz` 读取 `spider_contact_target_object_local` / `eval_contact_target_object_local`，并 resize 到 `qpos_ref` 长度。
- [x] 已生成 E085 raw target，并修正 projection bug：inside visual points 用 nearest-face projection，不再按最大归一化轴误投 face。main left face counts 修正后为 `+x:1,+y:35,-z:15`，vfrac mean `0.057`；right vfrac mean `0.895`。
- [x] target-selection audit 完成：main left 的 `broad_projected`、`tip_best_projected`、`tip_mean_projected` vfrac mean 分别为 `0.057/0.052/0.063`，说明 left 低位不是 broad centroid 单独造成；active frame 平均 `4.55/5` 个 fingertips 在 `3cm` 内。
- [x] E085 main 本地 full CEM 完成：contact `82.95%`，obj mean/max `0.665/1.068m`，pelvis min `0.659m`，但 head/upper penetration `18.60/53.49%`，hand penetration `73.64/54.26%`，gate fail。
- [x] E085 guard 远程 full CEM 已回收：contact `88.00%`，obj mean/max `0.226/0.431m`，head/upper penetration `2.00/2.00%`，hand penetration `77.33/27.33%`，gate fail。
- [x] E085 可视化 sheet 已生成到 `workspace/core4d/results/E085/keyframes/contact_sheets/`。main 视觉为低头/肩肘手压箱，guard 视觉接近但手部穿透高。

## E086 进展: raw-target failure iteration

- [x] 已写入 E086 计划：`workspace/core4d/plan/92_E086_rawtarget_failure_iteration_plan.md`，验证 strict penetration penalty 与 left target vfrac floor 两条局部修复。
- [x] 已新增 `hand_object_deep_penalty` 配置与 reward 实现；E086A override/smoke 通过。
- [x] 已生成 E086B vfrac-floor target：left min vfrac `0.2`，输出 `workspace/core4d/results/E086/vfrac_floor_targets/E086B_vfrac_floor_main/raw_contact_targets.npz`。
- [x] E086A full CEM 完成：contact `76.74%`，obj mean/max `0.698/1.134m`，pelvis min `0.624m`，head/upper penetration `58.91/72.87%`，RH floor `10.85%`，失败且较 E085 更差。
- [x] E086B full CEM 完成：contact `75.19%`，obj mean/max `0.710/1.170m`，pelvis min `0.613m`，head/upper penetration `62.79/68.99%`，失败。抬高 left target 后仍然压箱，说明 target height 不是唯一主因。
- [x] 已写入正式结果日志：`workspace/core4d/log/108_E085_E086_rawtarget_cem_iteration_results.md`；已更新 `EXPERIMENT_TRACKER.md`。下一步不继续手调 penalty/vfrac，转 COLA-style support body / 6-DoF connector seed。

---

## E087 进展: Box021 质量与 reward 分项诊断

- [x] 根据用户新判断调整方向：暂不继续 COLA-style support body，先检查 `1) object mass 是否太大` 与 `2) reward 分项贡献是否在鼓励压箱局部解`。
- [x] 初步质量审计发现强信号：`d003_box021_20231018_029_p2_upperobj_e083` object mass 为 `29.632kg`，而 `box023_p2` guard 与 `box025_p2` 相关 scene 为 `5.0kg`；所有 box021 派生 scene 继承 `29.632kg`，不是 E083/E085 派生过程引入。
- [x] 已写入 E087 计划：`workspace/core4d/plan/93_E087_box021_mass_reward_audit_plan.md`。E087 分为 mass audit/sweep、reward breakdown replay、reward tuning 三步，先只跑 box021 029 main gate。
- [x] 已新增 E087 脚本：`mass_audit.py`、`create_mass_variants.py`、`generate_e087_overrides.py`、`reward_breakdown.py`、`eval_E087.py`、`train_E087.sh`、`run_E087_preprocess.sh` 和 `variants.tsv`。
- [x] 静态检查通过：E087 Python `py_compile`、shell `bash -n`、`git diff --check` 均通过。
- [x] E087 preprocess 完成：质量审计输出到 `workspace/core4d/results/E087/mass_audit/`；创建 `d003_box021_20231018_029_p2_upperobj_e083_m5_e087` 和 `_m10_e087` 两个派生 task，并按质量比例缩放 object inertia；生成 `core4d_E087A/B/C` overrides。
- [x] 已对 E085/E086 四个已有 rollout 跑离线 reward breakdown，输出到 `workspace/core4d/results/E087/reward_breakdown/`。关键结论：E085 main case-window 平均 `contact_hdmi_rew=2.747`、`qpos_rew=1.965`、`task_obj=0.383`，但 `robot_object_penalty=-0.030`、hand deep penalty 为 `0`，说明 safety 惩罚量级远小于 contact/local tracking；E086A 虽把 `robot_object_penalty` 提到 `-0.218`，仍小于 contact/qpos 总正项且引入更差姿态。
- [x] E087 三个变体 smoke 通过：`E087A_m5_rawtarget_main`、`E087B_m10_rawtarget_main`、`E087C_m5_safe_main` 均可加载派生 scene、E085 raw target 和对应 reward 配置。
- [x] 本地 `E087A_m5_rawtarget_main` full CEM 已完成并落盘；首次自动 eval 因 E087 `variants.tsv` 列顺序不兼容 E083/E081 评估链路而失败，已修正 TSV 为 E081-compatible 格式，并同步修正 `train_E087.sh`/`eval_E087.py` 的字段解析；不需要重跑 CEM，只需重跑 eval。
- [x] E087 本地+远程 full CEM 已完成并回收：本地 `E087A_m5_rawtarget_main`，远程 GPU0 `E087B_m10_rawtarget_main`，远程 GPU1 `E087C_m5_safe_main`。合并 eval 完成，`num_results=3`、`accepted_variants=[]`。
- [x] E087 关键量化：5kg raw target contact `82.9%` 但 obj mean `0.782m`、head/upper penetration `89.1/89.1%`；10kg raw target obj mean `0.735m`、head/upper `69.8/76.0%`，略好但仍失败；5kg+safety contact `65.9%`、obj mean `0.774m`，但 head/upper 仍 `89.1/89.1%`。
- [x] 已补跑 E087A/B/C reward breakdown：E087C safety-tuned 把 `contact_hdmi_rew` 降到 `0.853`、`robot_object_penalty` 提到 `-0.634`，但仍没阻止头/上身压箱，说明只靠当前 scalar penalty 调权不足以形成 hard constraint。
- [x] E087 可视化 sheet 已生成：`workspace/core4d/results/E087/keyframes/contact_sheets/E087_all_cases_sheet.jpg`。视觉观察：三组都仍是弯腰/趴箱/上身压箱；10kg 比 5kg 稳一些但不成功，5kg+safety 手部更保守但头/上身问题没解决。
- [x] 已写入正式结果日志：`workspace/core4d/log/109_E087_box021_mass_reward_audit_results.md`；已更新 `EXPERIMENT_TRACKER.md`。下一步建议不是继续小幅 weight sweep，而是 hard safety gate / elite filtering，并修正 object lift/floor reward 口径。

---

## 2026-05-28 19:30 CST: E089 启动（A+B 并行）

- 计划：`workspace/core4d/plan/95_E089_g1_feasibility_AB_validation_plan.md`
- 诊断依据：`workspace/exp_diagnostic/diagnostic_report.md` + `data_filter_recommendation.md`
- A 路：本地 GPU0 跑 `box021_person1` SPIDER (E088 reward stack + ref_fk target)
- B 路：subagent 在 holosoma 做 OmniRetarget top-face 约束

### A 路进展
- [x] 派生 task `box021_person1_upperobj_e089`（16 leg + 7 upper-body collision pair），脚本 `workspace/core4d/scripts/E089/create_e089_cases.py`
- [x] Scene snapshot 入 `workspace/core4d/results/E089/scene_snapshot/`
- [x] Override `examples/config/override/core4d_E089A_box021_person1_upperobj.yaml`（继承 E088A，切 ref_fk target）
- [x] Train script `workspace/core4d/scripts/train/train_E089.sh`（smoke/local 两模式）
- [x] Eval script `workspace/core4d/scripts/eval/eval_E089.py`（含 E085-E088 baseline 对照表）
- [x] **A 路 smoke 完成（4 CEM iter, ~6.5 min）**：
  - `contact_frac_either=60.2%`、`obj_err_mean=1.3cm`
  - **head_pen=0.0%、upper_pen=0.0%、hand_floor=0.0% 双手**
  - 对比 E087A: head/upper/floor = 89%/89%/81%；E088A = 28%/55%/11%
  - pelvis_min=0.19m（smoke 没收敛，预期 full 收敛后稳）
- [ ] **A 路 full CEM 进行中**（background task `b1wx8ktk3`，预计 30-40 min）

### B 路进展
- [x] subagent 已启动（agentId aea193977a4a6c89d，background）
- [ ] 等 B1 audit 完成

### 关键判定
- C1 已通过：smoke 阶段 head/upper/hand-floor penetration 三项全 0%，证明 G1-Feasibility gate pass 与 SPIDER dynamic feasibility 在 box021_person1 上强相关。
- C2 待 full CEM 完成后验证。

---

## 2026-05-28 20:20 CST: E089 完成

### 完整结果
- **A 路 full CEM 完成**：pelvis_min=0.687m，head/upper/hand-floor 全 0%，obj_err 1.3cm，contact 60.2%
- **B 路 13 case 修复完成**：subagent post-IK damped-LS，wrist world-up-face-frac 4-14% → 99-100%
- **B4 SPIDER smoke 完成**：top-2 case head 0%/0%，upper 1.9%/0%，floor 7.5%/0%
- **4/4 claims 通过**

### 已写入
- log: `workspace/core4d/log/111_E089_g1_feasibility_AB_results.md`
- EXPERIMENT_TRACKER 已更新（含 E089 行 + Logs 链接）
- evals: `workspace/core4d/results/E089/eval_summary.json`

### 已识别 follow-up
- P1: 修 G1-feasibility gate 让 `top_face_frac` 识别 world-up 而非 hardcoded local +z
- P2: B-path top-2 跑 full CEM（smoke 已证 head/upper 0，pelvis 待 full 收敛）
- P3: 把 gate 集成到 holosoma D005b
- P4 / P5: 真 pre-IK B-1（需 holosoma env），应用到 Box026 等新箱型

---

## 2026-05-28 21:05 CST: E090 计划探索

- [x] 已读取用户思考 `workspace/exp_diagnostic/my_thoughts.md`、诊断报告、E089 plan/log、E028/E030/E082-E088 关键记录和 Holosoma V1 README。
- [x] 关键判断：诊断和 E089 强支持“上游 retarget 后的 G1 wrist/eef 几何不可执行”是 D003 Box021 失败主因；但 D003 production 没有显式启用 Phase 4 flags，不能直接归因到 Phase 4 改进本身。更高优先级变量是 `--replace_wrist_with_fingertip`、原始 solver/code path、以及 hand target semantic。
- [x] 已写入 E090 详细计划：`workspace/core4d/plan/96_E090_original_omniretarget_ablation_plan.md`。核心实验矩阵为 current baseline、current no-fingertip、original solver same input、original full、current Phase4、topface pre-IK；先跑 3 个 canonical Box021 failure case，几何 gate 通过后再接 SPIDER smoke/full 和 13 case 扩展。

## 2026-05-28 21:25 CST: E090 根据 H2 反馈调整

- [x] 用户补充 `--replace_wrist_with_fingertip` 的原始动机是 Box025 太大、G1 臂展不够，因此用 fingertip cluster 替代 wrist 来增加 reach。该动机支持 H2：同一 reach hack 对 Box021 低位合抱可能把 target 推到侧面/下沿/箱体内部。
- [x] 已更新 `workspace/core4d/plan/96_E090_original_omniretarget_ablation_plan.md`：E090 改为 H2-first。第一批只跑 current no-fingertip 与 topface-preIK；original solver/full 和 Phase4 flags 降为第二批条件执行。新增 Box025 guard 风险：即使 no-fingertip 改善 Box021，也不能全局删除 fingertip replacement，只能按物体尺寸/接触面条件化启用。

## 2026-05-28 21:55 CST: E090 Phase 0 gate 修复

- [x] 已修 `workspace/exp_diagnostic/scripts/g1_feasibility_gate.py`：repo root 不再写死 `/mnt/ali...`，支持 `--tasks`；`top_face_frac` 改为 object 当前姿态下 outward normal 最接近 world +Z 的 face。
- [x] 校准时发现严格 world-up face 会误杀 `box025_person2`（该 case 是高侧壁/legacy local +z 接触，E080 标 partial positive）。为保留 Box025 guard，新增 `legacy_local_z_face_frac` 与 `support_face_frac=max(top_face_frac, legacy_local_z_face_frac)`，gate 用 `support_face_frac >= 30%` 判定。严格 world-up 指标仍保留，供 E090 topface-preIK 对比。
- [x] 已同步简化 `workspace/exp_diagnostic/scripts/gate_compare_b_path.py`，删除临时 world-up 补丁逻辑，直接复用新 gate。
- [x] 验证命令通过：`python -m py_compile workspace/exp_diagnostic/scripts/g1_feasibility_gate.py workspace/exp_diagnostic/scripts/gate_compare_b_path.py`；`python workspace/exp_diagnostic/scripts/g1_feasibility_gate.py --tasks d003_box021_20231018_029_p2 d003_box021_20231011_035_p2 d003_box021_20231020_019_p1 box023_person2 box025_person2 --output-json workspace/core4d/results/E090/gate_phase0_calibration.json`。结果：3 个 D003 Box021 canonical failure 全 REJECT，`box023_person2` PASS，`box025_person2` PASS。

## 2026-05-28 22:15 CST: E090 Phase 1A 脚本落地

- [x] 已新增 `workspace/core4d/scripts/E090/canonical_cases.tsv`，包含 3 个 canonical failure case：`20231018_029_p2`、`20231011_035_p2`、`20231020_019_p1`。
- [x] 已新增 `workspace/core4d/scripts/E090/rewrite_wrist_top_face_preik.py`：对 converted NPZ 的 `global_joint_positions[:,20:22]` 做 world-up face +5cm pre-IK rewrite，并写 summary JSON。
- [x] 已新增 `workspace/core4d/scripts/E090/run_holosoma_retarget_ablation.sh`：固化 convert -> 可选 topface rewrite -> robot_retarget -> trim -> SPIDER scene/trajectory/scene_act/verify 流程；默认 variants 为 `current_no_fingertip,topface_preik`。
- [x] 已新增 `workspace/core4d/scripts/eval/eval_E090_retarget_geometry.py`，从 `workspace/core4d/results/E090/variants.tsv` 读取产物并运行新 G1 gate，输出 `workspace/core4d/results/E090/geometry/geometry_summary.{json,csv}`。
- [x] 静态检查通过：`python -m py_compile workspace/core4d/scripts/E090/rewrite_wrist_top_face_preik.py workspace/core4d/scripts/eval/eval_E090_retarget_geometry.py`，`bash -n workspace/core4d/scripts/E090/run_holosoma_retarget_ablation.sh`，`git diff --check`。
- [x] dry-run 通过：`bash workspace/core4d/scripts/E090/run_holosoma_retarget_ablation.sh --dry-run --variants current_no_fingertip --case-set canonical` 展开出的三条 no-fingertip 命令路径正确。
- [x] 首次实际运行遇到环境问题：`source /home/ubuntu/Workspace/holosoma/scripts/source_retargeting_setup.sh` 后 `python` 仍缺 `smplx`，报 `ModuleNotFoundError: No module named 'smplx'`。已确认 `/home/ubuntu/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python` 可 import `smplx,mujoco`，因此更新 E090 runner 显式使用 `HS_PYTHON`，并把 `HSRETARGETING_BIN` 放到 PATH。

## 2026-05-28 22:36 CST: E090 计划按 H2 偏好再更新

- [x] 已根据用户反馈更新 `workspace/core4d/plan/96_E090_original_omniretarget_ablation_plan.md`：标题改为 H2-first，明确 `--replace_wrist_with_fingertip` 是为 Box025 reach 设计的条件性策略候选，而不是应被全局删除的错误参数。
- [x] 已在计划中新增 `world-up top face` 术语澄清：它是物体当前姿态下 6 个 collision face 中 outward normal 最接近 world/MuJoCo `+Z` 的面，不等于 hardcoded local `+z`。
- [x] 已新增 C6 / Phase 2B Box025 reach guard：若 no-fingertip 改善 Box021 但损害 Box025，最终结论必须是按物体尺寸、接触面、inside-risk 或 reach margin 条件化启用 fingertip replacement。

## 2026-05-28 22:38 CST: E090 Phase 1A 继续启动

- [x] 已按 `experiment-planning-zh` 恢复 `EXPERIMENT_TRACKER.md`、E090 plan 和 `progress.md` 当前状态；确认当前应从 Phase 1A 实际 retarget 继续，而不是重写计划。
- [x] 已启动 sidecar explorer `019e6f05-3ac9-7a60-a208-4e20037fff89` 审计 E090 Phase 1A runner/manifest/eval 的可运行性；主线程同时推进静态检查与本地实际运行。
- [x] Phase 1A 启动前静态检查通过：`bash -n`、`python -m py_compile`、`git diff --check`，以及 `HS_PYTHON` 下 `import smplx,mujoco`。
- [ ] 已启动实际运行：`bash workspace/core4d/scripts/E090/run_holosoma_retarget_ablation.sh --case-set canonical --variants current_no_fingertip,topface_preik`，日志写入 `workspace/core4d/results/E090/logs/phase1a_retarget_20260528_2239.log`。目前 `20231018_029_p2` no-fingertip 全链路完成，topface-preIK retarget 进行中。
- [x] Sidecar explorer 返回：env/path/XML/TSV schema 无硬阻塞；主要风险是 `variants.tsv` 中途失败会保留部分 manifest，不能在长跑未完成时直接 eval。
- [x] Phase 1A 首轮实际运行部分成功：`20231018_029_p2` 和 `20231011_035_p2` 的 no-fingertip/topface-preIK 四个产物均完成 verify；`20231020_019_p1` no-fingertip 在 Holosoma `robot_retarget.py` 约 frame 79 报 `RuntimeError: CVXPY solve failed: infeasible`，没有生成 retargeted/trimmed 产物。下一步不重复该失败配置，改为只继续第三个 case 的 topface-preIK，并保留已有 manifest。
- [x] 已小修 `workspace/core4d/scripts/E090/run_holosoma_retarget_ablation.sh`：新增 `--only-base-task` 与 `--append-manifest`，用于在失败后只补跑剩余可行变体并保留已有成功行；`bash -n`、`git diff --check`、dry-run 均通过。
- [x] 已修 runner 的 dry-run 副作用：dry-run 不再创建/覆盖/追加 `variants.tsv`；已清理一次 dry-run 导致的重复 manifest 行。当前 `workspace/core4d/results/E090/variants.tsv` 为 5 个成功产物。
- [x] Phase 2 geometry eval 完成：输出 `workspace/core4d/results/E090/geometry/geometry_summary.{json,csv}`。no-fingertip 两条可跑 case inside 均为 0%，但 support 仅 `17.3/14.7%` 与 `27.1/21.8%`，仍未过 30%；第三条 no-fingertip solver infeasible。topface-preIK support 为 `100/100%`、`100/100%`、`98/100%`，后两条 gate PASS；第一条只因 `T=78<80` 被 reject。
- [x] 已新增 Phase 2B guard manifest `workspace/core4d/scripts/E090/guard_cases.tsv`，包含 `box023_person2` (`20231008/045/person2/Box023`) 和 `box025_person2` (`20231011/048/person2/Box025`)。
- [x] 已扩展 E090 runner 支持 `--case-set guard`；guard dry-run 不再污染 manifest，检查后 `variants.tsv` 仍为 5 行。
- [x] Guard retarget 已完成并追加到 manifest：`box023_person2_btop_preik_e090` trim 后 `T=138`，`box025_person2_btop_preik_e090` trim 后 `T=162`。
- [x] 已重跑 E090 geometry eval，`workspace/core4d/results/E090/geometry/geometry_summary.csv` 更新为 7 行。新增 guard 结论：`box023_person2_btop_preik_e090` PASS（inside `0/0%`，support `100/100%`）；`box025_person2_btop_preik_e090` REJECT（inside `48.1/51.9%`，signed distance 均值约 `-1/-7mm`，虽 support `98.8/96.9%`）。
- [x] 已更新 E090 计划：把 Phase 1A/2B 实证结果写入 `workspace/core4d/plan/96_E090_original_omniretarget_ablation_plan.md`；下一步收敛为先跑 Box021 topface-preIK 的两个 gate-pass case (`20231011_035_p2`、`20231020_019_p1`) 的 SPIDER smoke，Box025 topface-preIK 仅作为 negative guard，不作为全局策略推广。

## 2026-05-28 23:08 CST: E090 Phase 3 smoke 准备

- [x] 已启动 sidecar explorer `019e6f16-1822-7990-bf60-0dd24e95892f` 只读审计 E081/E083/E089 的 SPIDER 派生、override、train/eval 模式；其结论与主线程一致：E090 不能直接训练原始 retarget task，必须派生 m10 + leg/upper-body-object collision 的 smoke task。
- [x] 已新增 `workspace/core4d/scripts/E090/build_spider_tasks.py`：从 `d003_box021_20231011_035_p2_btop_preik_e090` 和 `d003_box021_20231020_019_p1_btop_preik_e090` 生成两个 `*_upperobj_m10_smoke` 派生 task；复制 E090 qpos，注入 E083 的 16 leg/foot + 7 upper-body object pairs，并把 `scene.xml`/`scene_act.xml` object mass 从 `29.632kg` 缩放到 `10kg`。
- [x] 已新增 `workspace/core4d/scripts/train/train_E090_smoke.sh` 和 `workspace/core4d/scripts/eval/eval_E090.py`；override 生成到 `examples/config/override/core4d_E090S1_box021_20231011_035_p2_btop.yaml` 与 `core4d_E090S2_box021_20231020_019_p1_btop.yaml`，继承 E089A safety stack，使用 `ref_fk` target，禁用旧 contact mask。
- [x] Build 与轻量验证通过：两个派生 task 的 `scene_act.xml` 均为 `nq/nv/nu=42/41/35`、`npair=49`、object mass `10kg`，原始 freejoint trajectory qpos 分别为 `(135,43)` 和 `(101,43)`；`variants_smoke.tsv` 已生成并可被 `train_E090_smoke.sh list` 读取。
- [x] E090 Phase 3 smoke 已完成：`bash workspace/core4d/scripts/train/train_E090_smoke.sh local 0` 顺序跑 S1/S2，并自动写入 `workspace/core4d/results/E090/smoke/smoke_eval_summary.{json,csv}` 与 `workspace/core4d/results/E090/eval_summary.json`。
- [x] Smoke 量化：S1 `E090S1_box021_20231011_035_p2_btop` PASS，`T=135`、contact `95.6%`、obj mean `0.001m`、pelvis min `0.538m`、head/upper/LH/RH-floor 全 `0%`；S2 FAIL，`T=101`、contact `40.6%`、obj mean `0.011m`、pelvis min `0.180m`、head/upper `0%` 但 `LH_floor=56.4%`。
- [x] Smoke 视觉复核：S1 没有明显头胸穿箱或手撑地，但 4-iter 阶段仍弯腰、头部贴近箱面，属于“安全改善但姿态未完全收敛”；S2 后段左手/身体落地并翻箱，不能进入 full。
- [x] 已新增 `workspace/core4d/scripts/train/train_E090_full.sh`，并扩展 `workspace/core4d/scripts/eval/eval_E090.py` 支持 `--stage full`。full 默认只跑 smoke-pass 的 S1，S2 不重复推进。
- [x] E090 Phase 4 full 已完成：`bash workspace/core4d/scripts/train/train_E090_full.sh local 0` 只跑 S1，耗时约 18min，结果写入 `workspace/core4d/results/E090/full/full_eval_summary.{json,csv}`。
- [x] Full 量化：S1 full `contact=57.8%`、`obj_mean=0.009m`、head/upper/LH/RH-floor 全 `0%`，但 `pelvis_min=0.134m`，因此 full fail。视觉确认机器人趴低/跪低、头部贴近箱面，pelvis failure 是真实姿态问题。
- [x] 轻量诊断：source ref 经 `load_data`/scene-act conversion 后 pelvis min 约 `0.657m`，而 full rollout sim pelvis min `0.134m`，说明失败来自 CEM/reward 的低姿态局部解，不是 topface-preIK retarget reference 先天低 pelvis。
- [x] 已写入正式结果日志 `workspace/core4d/log/112_E090_h2_first_retarget_and_spider_smoke_results.md`，并更新 E090 plan 的 Phase 3/4 证据与决策树。下一步不扩展 D003 13 case，先做 S1-only 姿态约束验证。
- [x] 已更新 `workspace/core4d/EXPERIMENT_TRACKER.md`：新增 E090 总览行和 log 列表摘要，状态标为 `❌/🔬`（几何/safety 有效，但 full 因 pelvis collapse 未通过）。

## 2026-05-28 23:42 CST: E091 data_construction_v2 计划更新启动

- [x] 已按 `experiment-planning-zh` 恢复 E090 结果、`workspace/exp_diagnostic/data_filter_recommendation.md` 与 Holosoma `workspace/v3/data_construction` 现状。
- [x] 计划主线已根据用户反馈从 B 路 Box021 D003 扩展调整为 C 路：在 Holosoma `workspace/v3/data_construction_v2` 新建实验链路，优先开 Box026，再看 Box004/Box022。
- [x] 关键依据：E090 支持 H2 的几何语义判断，但 Box021 topface-preIK full CEM 仍因 pelvis collapse 失败；旧 Holosoma 链路 D001/D002 已发现 Box026/box004 clean 候选，而 D003 ready 队列只含 Box021，Box026/box004/Box022 需要补模板/OmniRetarget/可视化 gate。
- [x] 已写入 E091 计划：`workspace/core4d/plan/97_E091_holosoma_data_construction_v2_medium_boxes_plan.md`。计划明确 `world-up top face` 定义、`--replace_wrist_with_fingertip` 的 Box025 reach-hack 定位、Box026 > box004 > Box022 的优先级、D005b gate、可视化输出和 high-reasoning 复核。
- [x] 已吸收 sidecar 审计结果：旧 Holosoma 脚本/JSON 有 `data_constructon` 硬编码但实际目录是 `data_construction`；Box026/box004 的 blocker 是缺 SPIDER/MuJoCo source scene template；D004/D005 当前实际 PNG 产物不完整，E091 必须重新生成 raw/retarget/G1 overlay 可视化并做存在性/非空检查。
- [x] 已更新 `workspace/core4d/EXPERIMENT_TRACKER.md`，新增 E091 计划行，状态 `📋`。

## 2026-05-29 00:06 CST: E091 Phase 0 脚本实现

- [x] 已新增 `workspace/core4d/scripts/E091/make_data_construction_v2_dirs.sh`，创建 Holosoma `workspace/v3/data_construction_v2/{inputs,scripts,results,visualizations,logs,reports}`。
- [x] 已新增 `workspace/core4d/scripts/E091/build_medium_box_manifest.py`，从旧 D001/D002 生成 Box026/box004/Box022 manifest、Stage2b backlog TSV、尺寸/ready dashboard。
- [x] 已新增 `workspace/core4d/scripts/E091/make_raw_contact_visuals.py`，从旧 D002 `raw_contact_proxy.npz` 生成 per-case raw contact timeline PNG 与 aggregate dashboard，并检查 PNG 非空。
- [x] Phase 0 已实际运行：`data_construction_v2` 生成 `80` 条 medium manifest、`15` 条 Stage2b backlog（Box026 7、box004 4、Box022 stress-test 4），当前 `source_scene_exists=0`，因此 Stage2b 全部 disabled 等模板补齐。
- [x] Raw-contact 可视化已生成并通过非空检查：`16/16` PNG nonblank，输出在 `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/visualizations/raw_contact/` 与 `visualizations/dashboard/`。
- [x] 已新增并运行 `workspace/core4d/scripts/E091/template_preflight.py` / `run_template_preflight.sh`：15 条 backlog 全部 `needs_source_scene_template`；object mesh 均存在。Box026 half extents 约 `0.3145 x 0.1972 x 0.2345m`，建议 base template `box021_person1`；box004 half extents 约 `0.1740 x 0.1318 x 0.2236m`，建议 base template `box023_person1`。
- [x] 已吸收 high subagent 可视化复核：图像信息足够且 `16/16` 非空；Box026 top3 建议 `e091_box026_20231018_039_p2`、`e091_box026_20231018_040_p2`、`e091_box026_20231020_135_p2`；box004 `20231003_2/083 p2/p1` 适合作为低风险对照；Box026 fail/hold 与 Box022 暂不进主线。
- [x] 已修 E091 Stage2b 输入风险：`build_medium_box_manifest.py` 现在额外输出严格 12 列 `inputs/cases_stage2b_medium_pipeline.tsv`；`run_stage2b_medium_boxes.sh --dry-run` 在当前模板未补齐时正确报告 `ENABLED_ROWS=0` 并退出，不会误喂带诊断列的 TSV 给 SPIDER pipeline。
- [x] 已新增 `workspace/core4d/scripts/E091/create_source_scene_templates.py` 并创建首批 source templates：`box026_person2`、`box004_person2`。对应 scene 均可被 MuJoCo load (`nq=43,nv=41,nu=29`)，object mesh 已复制到 `example_datasets/processed/core4d/assets/objects/{box026,box004}/`，质量暂设 `5kg` 避免 Box021 D003 的 `29.632kg` 异常。
- [x] 重新生成 manifest/preflight 后，Stage2b ready 从 `0` 增至 `6`：Box026 person2 4 条、box004 person2 2 条。已输出首批 case files：`cases_stage2b_box026_first_pipeline.tsv`、`cases_stage2b_box026_top3_pipeline.tsv`、`cases_stage2b_box004_control_pipeline.tsv`。
- [x] 已把 `workspace/core4d/data_preprocess/pipeline.sh` 改成向后兼容的 `REPLACE_WRIST_WITH_FINGERTIP` 开关（默认 `1` 保持旧行为）；E091 `run_stage2b_medium_boxes.sh` 默认设为 `0`，符合 medium-box 不默认使用 Box025 reach hack 的计划。dry-run 已确认 convert 命令不带 `--replace_wrist_with_fingertip`。

### E091 遇到的错误

| 错误 | 尝试次数 | 处理 |
|---|---:|---|
| `e091_box026_20231018_039_p2` 首次实际 Stage2b 在 `robot_retarget.py` 报 `ValueError: string is not a file: models/Box026/Box026.obj` | 1 | 定位为新物体模型生成到了 SPIDER repo，Holosoma retarget cwd 找不到。已在 `workspace/core4d/data_preprocess/pipeline.sh` 加 `sync_generated_object_model`，convert 后把 `src/holosoma_retargeting/.../models/<object>` 同步到 Holosoma repo；dry-run 已确认 retarget 前会执行同步。 |
| 同一 case 第二次在 `InteractionMeshRetargeter` 报 `ParseXML: Error opening file 'models/g1/g1_29dof_w_Box026.xml'` | 1 | 定位为 G1-with-object retarget XML 未生成。已扩展 `create_source_scene_templates.py`，为 Box026/box004 生成并 MuJoCo load `g1_29dof_w_<object>.xml`，同时写 OBJ/URDF 到 Holosoma retarget models。 |
| `e091_box026_20231018_040_p2` 在 no-fingertip retarget 第约 67/99 帧报 `CVXPY solve failed: infeasible` | 1 | 不重复同配置；批处理因此未跑到第三条。已生成单独 case file `cases_stage2b_box026_135_p2_pipeline.tsv`，继续跑 `e091_box026_20231020_135_p2`。 |

### E091 首条 Stage2b / D005b 结果

- [x] `e091_box026_20231018_039_p2` no-fingertip Stage2b 已完整通过：retarget `142` 帧，trim 后 `123` 帧，SPIDER `trajectory_kinematic.npz` `(123,43)`，`scene.xml nq/nv/nu=43/41/29`，`scene_act.xml nq/nv/nu=42/41/35`，verify `trimmed_qpos_matches_spider_qpos=true`。
- [x] 该 case 的 D005b gate 已运行：inside `L/R=0.8%/0.0%`、signed distance `+8.1/+9.1cm`、pelvis min `0.705m`、T `123` 均好；但 support-face fraction 约 `20%`，低于 30% 阈值，因此当前判定 `REJECT: no_hand_on_support_face_≥30%`。下一步继续跑 Box026 top3 另外两条，不因单条 near-reject 直接放弃 Box026。

## 2026-05-29 00:17 CST: E091 计划按 Box026 top3 初筛更新

- [x] `e091_box026_20231020_135_p2` no-fingertip Stage2b 已完成：retarget `116` 帧，trim 后 `82` 帧，SPIDER `trajectory_kinematic.npz` `(82,43)`，verify `trimmed_qpos_matches_spider_qpos=true`。
- [x] 已补跑 Box026 两条成功预处理的 D005b gate，输出到 `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/results/d005b_g1_feasibility/box026_top_success_gate.json`。
- [x] D005b 结果：`039_p2` 仍为 near-reject，主要问题是 support-face frac `~20% < 30%`；`135_p2` 也是 near-reject，support-face frac `~62%`，但 right wrist inside `12.2% > 10%`。
- [x] 决策更新：暂停继续扩大 Box026 no-fingertip top7；下一步优先跑 `cases_stage2b_box004_control_pipeline.tsv`，先争取一个 medium-object positive pipeline。若 box004 也无 pass，再对 Box026 `039_p2` / `135_p2` 做有限 H2 variant（support-face-preIK / exterior projection），不重复 `040_p2` no-fingertip。
- [x] 已更新计划文件 `workspace/core4d/plan/97_E091_holosoma_data_construction_v2_medium_boxes_plan.md`：新增 `0.1 2026-05-29 动态更新`，并改写 Phase 2 成功标准和下一步执行顺序。

## 2026-05-29 00:23 CST: E091 box004 positive pipeline 初筛

- [x] 已跑 `cases_stage2b_box004_control_pipeline.tsv` 中的 `e091_box004_20231003_2_083_p2`，no-fingertip Stage2b 全链路完成：retarget `121` 帧，trim 后 `105` 帧，SPIDER `trajectory_kinematic.npz` `(105,43)`，`scene.xml nq/nv/nu=43/41/29`，`scene_act.xml nq/nv/nu=42/41/35`，verify `trimmed_qpos_matches_spider_qpos=true`。
- [x] 已重跑 D005b summary：`e091_box004_20231003_2_083_p2` 为当前第一条 PASS，inside `L/R=0.0%/0.0%`、signed distance `+18.5/+14.1cm`、support-face either `42.9%`、pelvis min `0.679m`、T `105`。Box026 两条保持 reject/near-reject。
- [x] 已新增并运行 `workspace/core4d/scripts/E091/make_medium_box_visual_qc.py`，输出 D005b TSV/MD、object-local overlay、timeline、MuJoCo keyframe sheet 和 aggregate dashboard。PNG 非空检查 `11/11` 通过。
- [x] 关键输出：`/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/results/d005b_g1_feasibility/d005b_summary.tsv`、`/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/results/visual_qc/summary.md`、`/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/visualizations/d005b/`、`/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/visualizations/keyframes/`。
- [x] high-reasoning subagent `019e6f65-9798-73b0-b68d-0b396e38aae9` 已完成可视化复核：box004 判定为 visually credible PASS，可作为 seed positive；Box026 两条 reject 与 overlay/timeline 一致，不建议同配置扩量。
- [x] 已写入复核报告 `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/reports/high_subagent_d005b_review.md`。
- [x] 已新增并运行 `workspace/core4d/scripts/E091/make_top_medium_box_bank.py`：top bank 3 行，top candidate 1 行。`e091_box004_20231003_2_083_p2` rank 1、score `88.878`、decision `top_candidate`；两条 Box026 为 `review_only`。
- [x] 已新增 E091 minimal smoke 脚本：`workspace/core4d/scripts/E091/build_spider_smoke_tasks.py`、`workspace/core4d/scripts/train/train_E091_smoke.sh`、`workspace/core4d/scripts/eval/eval_E091.py`。静态检查通过。
- [x] 已生成 smoke 派生 task `e091_box004_20231003_2_083_p2_upperobj_m5_smoke` 与 override `examples/config/override/core4d_E091S1_box004_20231003_2_083_p2.yaml`：scene_act `nq/nv/nu=42/41/35`、`npair=49`、object mass `5kg`、trajectory `(105,43)`，加入 16 个 leg/foot-object 和 7 个 upper-body-object pairs。
- [x] 已完成 top1 minimal smoke：`bash workspace/core4d/scripts/train/train_E091_smoke.sh local 0`。产物在 `workspace/core4d/results/E091/smoke/`，包含 rollout NPZ、mp4、keyframes、`smoke_eval_summary.{json,csv}`。
- [x] E091 smoke eval 已按本计划补充 pelvis 检查并重跑：`smoke_collision_pass=true`（head `0%`、upper `0%`、LH floor `0%`、RH floor `1.0%`、object mean error `0.006m`），但 `pelvis_min=0.079m`，`pelvis_collapse_warning=true`，因此 `stage_pass=false`。
- [x] 按 `video-frames` skill 从 smoke mp4 额外抽取 `workspace/core4d/results/E091/smoke/keyframes_skill/E091S1_box004_20231003_2_083_p2_f80.jpg`；自动抽取 keyframes `9/9` 非空。
- [x] high-reasoning subagent `019e6f6e-9002-7fe3-ae85-ec3ae02bd69d` 已完成 smoke keyframes 复核：判定 `REVIEW`。头/上身穿箱与手撑地不是主因，物体 tracking 稳定；中后段 pelvis/hip 明显塌陷，属于 dynamics follow-up。
- [x] 已写入 `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/reports/high_subagent_smoke_review.md`。
- [x] 已写正式 log `workspace/core4d/log/113_E091_data_construction_v2_medium_box_results.md`；已更新 E091 plan 0.2 结果与 `EXPERIMENT_TRACKER.md`。结论：data_construction_v2 数据链路已跑通到 smoke，当前 seed 为 box004；smoke 未 final pass 是 pelvis collapse，不回滚 D005b/top bank，不在 E091 内继续优化算法。

## 2026-05-29 00:43 CST: E091 OmniRetarget 可视化补齐

- [x] 根据用户指出“OmniRetarget 结果需要可视化”，已新增并运行 `workspace/core4d/scripts/E091/make_omniretarget_visuals.py`。它直接读取 Holosoma v2 Stage2b 的 `retargeted/*.npz` 与 `trimmed/*.npz`，用 source scene 渲染，不再用 SPIDER/D005b keyframes 代替 OmniRetarget 可视化。
- [x] 输出到 `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/visualizations/omniretarget/`：每个成功 retarget case 有 `retargeted_keyframes.png`、`trimmed_keyframes.png`、`omniretarget_timeline.png`、`retargeted.mp4`。
- [x] 覆盖 4 个已尝试 Stage2b 目录：3 个成功可视化（box004 p2、Box026 039 p2、Box026 135 p2），1 个 `e091_box026_20231018_040_p2` 标记为 `missing_retargeted_npz`（CVXPY infeasible 后没有 retargeted NPZ）。
- [x] 非空检查通过：OmniRetarget PNG `9/9` nonblank，MP4 `3/3` exists。manifest/summary 写入 `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/results/omniretarget_visuals/`。

## 2026-05-29 01:04 CST: E092 三 case SPIDER dynamics + OmniRetarget RL 规划草案

- [x] 根据用户建议写入整体规划草案：`workspace/core4d/plan/98_E092_three_case_spider_dynamic_and_omniretarget_rl_plan.md`。
- [x] 计划默认三条 case 为 E091 已成功生成 OmniRetarget 可视化的 `e091_box004_20231003_2_083_p2`、`e091_box026_20231018_039_p2`、`e091_box026_20231020_135_p2`；`e091_box026_20231018_040_p2` 因 CVXPY infeasible 无可用 OmniRetarget NPZ，暂不纳入。
- [x] 规划拆成两条路线：每 case 先跑 SPIDER dynamic retarget，只有 `WORK` 序列进入 RL-from-SPIDER；同时三条 case 全部跑 direct OmniRetarget RL 作为对照。
- [ ] 尚未实现 E092 脚本或启动训练；等待用户确认整体规划和 RL 入口定义。

## 2026-05-29 01:18 CST: E092 执行准备

- [x] 已按用户要求启动 E092 实验落地：目标是 Stage A/B/C 训练均采用本地 1 卡 + 远程 2 卡并行，按 `remote-execution.md` 回收结果。
- [x] 恢复了 E092 plan、远程执行文档、E091/E090 eval/build/train 脚本上下文；当前仓库可见训练入口是 `examples/run_mjwp.py`/MJWP 动态优化栈，未发现独立 PPO/RL trainer。
- [ ] 当前实现策略：先把 E092 的 “RL from Omni / RL from SPIDER” 接入既有 core4d MJWP 训练/评估栈，作为当前 repo 内可执行的训练入口；若后续发现独立 RL trainer，再替换 Stage B/C launcher。

## 2026-05-29 01:31 CST: E092 scaffold 实现中

- [x] 已新增 `workspace/core4d/scripts/E092/build_three_case_tasks.py`：为三条 case 生成 `spider_dyn` 与 `rl_omni` 两套路由派生 task，并写 `workspace/core4d/scripts/E092/variants.tsv`。
- [x] 已新增 `workspace/core4d/scripts/E092/build_rl_tasks.py`：计划在 Stage A full 出现 `WORK` 后，把 scene_act 动态 rollout 反转成 freejoint `trajectory_kinematic.npz`，生成 `rl_spider` task 与 manifest。
- [x] 已新增 Stage A/B/C 训练脚本：`train_E092_spider_dyn.sh`、`train_E092_rl_from_omni.sh`、`train_E092_rl_from_spider.sh`，均支持 `local|remote-gpu0|remote-gpu1` split。
- [x] 已新增评估脚本：`eval_E092_spider_dyn.py`、`eval_E092_rl.py`，输出 JSON/CSV/MD，并记录 `WORK/PASS/FAIL`。
- [x] 已新增远程脚本：`run_E092_remote.sh` 与 `pull_E092_remote_results.sh`，按 `spider-remote` 的 GPU0/GPU1 tmux 并行启动和回收。
- [ ] 下一步：运行 py_compile/bash -n，执行 `build_three_case_tasks.py --force`，验证 scene/task/override。

## 2026-05-29 02:02 CST: E092 Stage A smoke 三卡完成

- [x] 静态检查通过：`python -m py_compile` 覆盖 E092 builder/eval；`bash -n` 覆盖 E092 train/remote/pull；`git diff --check` 干净。
- [x] `python workspace/core4d/scripts/E092/build_three_case_tasks.py --force` 完成 6 个派生 task，三条 case 的 `scene_act.xml` 均 `nq/nv/nu/npair=42/41/35/49`，qpos 分别为 `(105,43)`、`(123,43)`、`(82,43)`。
- [x] 已提交并推送 E092 scaffold commit `aee4fc1`，供远程 clean clone 同步。远程原 repo 有本地改动阻塞 `git pull`，已改用独立 clean clone `/home/xiayb/pHRI_workspace/spider_e092_run`，只软链接旧 repo `.venv`，没有 reset 旧工作树。
- [x] Stage A smoke 按本地 1 卡 + 远程 2 卡完成：本地 C1，远程 GPU0 C2，远程 GPU1 C3；远程结果已通过 `pull_E092_remote_results.sh spider-dyn-smoke` 回收到本地。
- [x] 统一本地 eval：C1 `pelvis_min=0.074m`、C2 `0.191m`、C3 `0.186m`，三条均 `FAIL`。object tracking 都很好（mean `0.006/0.008/0.006m`），head/upper 均 `0%`；主要失败是 pelvis collapse，C3 另有 RH floor `39%`。
- [ ] 按计划：Stage A 无 `WORK`，因此不跑 Stage A full，不生成 Stage B `rl_from_spider` 输入；继续执行 Stage C direct OmniRetarget 三卡对照。

## 2026-05-29 02:10 CST: E092 Stage C smoke 三卡完成

- [x] Stage C `rl_from_omni` smoke 已按本地 1 卡 + 远程 2 卡完成：本地 C1，远程 GPU0 C2，远程 GPU1 C3。远程结果已通过 `pull_E092_remote_results.sh rl-omni-smoke` 从 clean clone `/home/xiayb/pHRI_workspace/spider_e092_run` 回收到本地。
- [x] 统一本地 eval 已覆盖 `workspace/core4d/results/E092/rl_from_omni/smoke/smoke_eval_summary.{json,csv,md}`：C1 `pelvis_min=0.073m`、C2 `0.135m`、C3 `0.189m`，三条均 `FAIL`。object tracking 仍很好（mean `0.006/0.008/0.006m`），head/upper 均 `0%`；主因仍是 pelvis collapse，C3 另有 RH floor `40.2%`。
- [x] 可视化产物已齐：6 个 smoke mp4 均存在，60 张 keyframe jpg 非空；已用 ffmpeg 生成 `workspace/core4d/results/E092/visual_review/contact_sheets/*_sheet.jpg` 供 high subagent 视觉复核。
- [x] high subagent `019e6fc7-1fa9-7442-af87-b56c8f376236` 已完成 6 条 smoke 视频 contact sheet 复核：视觉确认量化失败模式真实存在，C1/C3 明显 pelvis collapse，C3 右手/右臂贴地，C2 低髋/半跪/压箱；spider_dyn 与 rl_from_omni 无有意义视觉差异。
- [x] 已按 plan 决策：跳过 Stage A full、Stage B `rl_from_spider` 和 Stage C main；正式 log 写入 `workspace/core4d/log/114_E092_three_case_spider_dynamic_and_omniretarget_rl_results.md`，plan 98、tracker 和 comparison artifacts 已更新。

## 2026-05-29 11:00 CST: E092 执行纠偏 - 补跑 Stage A full CEM

- [x] 用户指出“只跑 smoke 不能支撑 SPIDER dynamic 结论”。复核后确认此前按 smoke gate 直接停止 full 过于保守：4-iter smoke 只能作为 preflight/早期失败信号，不能替代 full CEM 收敛验证。
- [x] 已启动 Stage A `spider_dyn full` 三卡补跑：本地 GPU0 跑 C1；远程 clean clone `/home/xiayb/pHRI_workspace/spider_e092_run` 的 tmux session `E092_spider_dyn_full` 中 GPU0 跑 C2、GPU1 跑 C3。
- [ ] 待 full 完成后回收 `pull_E092_remote_results.sh spider-dyn-full`，统一 eval，生成 full keyframes/contact sheets；E092 log/tracker/plan 需修正为 full 结果为准，并明确 smoke-only 停止结论作废。

## 2026-05-29 12:05 CST: E092 Stage A full CEM 回收完成

- [x] 已回收远程 full CEM 结果：`REMOTE_REPO=/home/xiayb/pHRI_workspace/spider_e092_run bash workspace/core4d/scripts/pull_E092_remote_results.sh spider-dyn-full`。
- [x] 本地统一 eval 已覆盖 `workspace/core4d/results/E092/spider_dyn/full/full_eval_summary.{json,csv,md}`：C1 `E092D1_box004_083_p2_dyn` 达到 `WORK`（pelvis `0.663m`、contact `64.8%`、obj mean/max `0.006/0.019m`、head/upper/hand-floor 全 `0%`）；C2 FAIL（pelvis `0.083m`）；C3 FAIL（pelvis `0.177m`、RH floor `17.1%`）。
- [x] full 产物齐：3 个 `trajectory_mjwp_act.npz`、3 个 mp4、30 张 keyframes，keyframes 非空。下一步应以 C1 full `WORK` 作为 Stage B `rl_from_spider` 的唯一候选输入；此前 smoke-only 停止结论作废。

## 2026-05-29 12:25 CST: E092 C1 vs C2/C3 初步机制分析

- [x] 尺寸/质量核查：E092 三条 object mass 均为 `5kg`；Box026 不是旧 D003 Box021 的 `29.632kg` 异常。collision friction 也一致为 `1 0.005 0.0001`，因此不是材质/摩擦参数主导。
- [x] 尺寸对比：box004 `0.348 x 0.264 x 0.447m`，box023 `0.306 x 0.314 x 0.353m`，属于相近 small/medium pattern；Box026 `0.629 x 0.394 x 0.469m`，体积 `0.116m^3`，约为 box004 `2.8x`、box023 `3.4x`。
- [x] contact target 口径：E092 继承 E085/E089 的 `contact_hdmi_dynamic_target=true`，并在 override 中设 `contact_hdmi_target_source=ref_fk`、`contact_hdmi_target_uses_eef_offset=true`。日志确认三条均为 `E040 dynamic target ... source=ref_fk, uses_eef_offset=True`；不是固定 object-local 点，而是 per-frame wrist+EEF offset target。
- [x] 几何差异：C1 D005b pass，inside `0/0%`，support either `42.9%`；C2 主要 reject 是 support only `19.5%`，大部分手目标落在 `local -z` / 非支撑面；C3 support 较高但 R-inside `12.2%`，full 中 RH floor 仍 `17.1%`。初步判断 C2 的参考接触面随时间/面切换并非根本“固定 target”问题，而是可支撑面占比低 + Box026 尺寸/reach 导致 CEM 用低髋姿态追目标。

## 2026-05-29 12:55 CST: E093 contact geometry audit 启动

- [x] 已按用户要求把下一步定义为 E093：在 CEM/RL 前深究 `wrist_yaw_link + 5cm`、raw contact、sphere、历史 3-box、Holosoma handbox 的几何关系。
- [x] 已恢复 tracker/latest plan/log/progress，并确认当前 E093 尚未落盘；工作树仍有 E092 full-correction 的本地改动和未跟踪 `workspace/exp_diagnostic/my_thoughts.md`，后者不纳入提交。
- [x] 已读取 Holosoma handbox 参考日志 `workspace/v2/log/44_r084_r086_box023_handbox_stagec_first_pass.md`，关键参数为 `main_mesh_collision_handbox_m5.urdf`、`left/right_handbox_link`、handbox fixed joint 约 `wrist + [0.1074, +/-0.0116, 0.0102]`，box size 约 `[0.1418, 0.0766, 0.1165]`。
- [x] 已写入计划 `workspace/core4d/plan/99_E093_contact_target_geometry_audit_plan.md`：覆盖 `box023_person2`、`box025_person2`、`box004`、`box021_person1`、`d003_box021_20231018_029_p2`、两条 `box026`，要求输出 raw/wrist/sphere/3-box/handbox 指标、object-local 可视化、MuJoCo 可视化和 high subagent review。

## 2026-05-29 13:18 CST: E093 脚本与首轮可视化完成

- [x] 已新增 `workspace/core4d/scripts/E093/build_contact_geometry_manifest.py`，并生成 `workspace/core4d/results/E093/contact_geometry/case_manifest.tsv`；7/7 case ready。
- [x] 已新增 `workspace/core4d/scripts/E093/audit_contact_geometry.py`：复用 E085 raw surface target 逻辑，从 raw mesh/person vertices 重新生成 raw contact centroid，和 `wrist+5cm`、sphere、3-box、handbox 做统一 object-local 对比。
- [x] 已新增 `workspace/core4d/scripts/E093/render_contact_geometry_mujoco.py`：在 MuJoCo reference qpos 上叠加 raw/wrist/sphere/handbox/3-box marker，输出 keyframe sheet 和 mp4。
- [x] 静态检查通过：`python -m py_compile` 覆盖 3 个 E093 脚本，`git diff --check` 干净。
- [x] 首轮全量诊断已跑通：`geometry_summary.{csv,json,md}` 14 行（7 case × 2 hands），`per_frame_points.csv` 1466 行，object-local/timeline/dashboard PNG `16/16` nonblank。
- [x] MuJoCo 可视化已跑通：7 张 keyframe sheet 非空（std 0.12-0.17），7 个 mp4 均通过 `ffmpeg -v error -i ... -f null -` 解码检查。
- [x] 首轮关键数值：`wrist+5cm -> raw` 在 box004/box023 约 `21-28cm`，D003 box021 约 `26-32cm`，Box026 约 `49-64cm`，box025 约 `58-64cm`；这说明 `wrist+5cm` 不是小误差 proxy，尤其 Box026/Box025 是大偏移。

## 2026-05-29 13:42 CST: E093 结果记录完成

- [x] high-reasoning subagent `019e720f-83fb-7193-a7a2-3924ab7d8527` 已完成 E093 可视化复核；只读检查 summary、dashboard、object-local overlay、timeline、MuJoCo keyframes。
- [x] 已写入复核报告 `workspace/core4d/results/E093/contact_geometry/visual_review/high_subagent_review.md`。结论：handbox 14/14 行相对最接近 raw，但仍不能修复 raw target 与 retargeted hand region 错面；Box026/D003 应优先修 target face assignment。
- [x] 已写正式 log `workspace/core4d/log/115_E093_contact_target_geometry_audit_results.md`，Claims C1-C5 已逐条判定。
- [x] 已更新 `workspace/core4d/EXPERIMENT_TRACKER.md`：新增 E093 行；同时修正 E092 行，明确 smoke-only stop 作废、Stage A full C1 WORK / C2-C3 FAIL。
- [x] 已给 `workspace/core4d/log/114_E092_three_case_spider_dynamic_and_omniretarget_rl_results.md` 添加 0.0 纠偏说明，记录 Stage A full CEM 结果和旧判定作废。

## 2026-05-29 13:50 CST: E093 MuJoCo 相机纠偏

- [x] 用户指出 MuJoCo 视频只能看到机器人上半身；复核旧 `box023_p2` keyframe sheet 后确认旧默认 `--camera track2` 会裁掉腿/脚，不适合判断接触和支撑。
- [x] 已修改 `workspace/core4d/scripts/E093/render_contact_geometry_mujoco.py`：默认相机从 `track2` 改为 `auto`，每个 case 用多帧 qpos、机器人 body、object collision box、contact marker 计算固定 full-body free camera；默认输出分辨率提升到 `960x720`，manifest 记录 lookat/distance/span。
- [x] 已重渲 `workspace/core4d/results/E093/contact_geometry/visuals/mujoco/` 下 7 张 keyframe sheets 和 7 个 mp4；`ffprobe` 检查为 `7/7` 视频 `960x720`、`48` 帧。
- [x] 已用 `video-frames` skill 抽查视频中段帧：`video_qc/box023_p2_f24.png` 和 `video_qc/box026_039_p2_f24.png`，视觉确认完整机器人、脚部、箱子和 marker 均在画面内。

## 2026-05-29 13:55 CST: E094 启动 - handbox-aware target projection

- [x] 已按 `experiment-planning-zh` 恢复 E093 log/plan、E092 脚本和远程执行规范；当前工作树仅有用户未跟踪文件 `workspace/exp_diagnostic/my_thoughts.md`。
- [x] 关键代码入口确认：`examples/run_mjwp.py` 已支持 `contact_hdmi_target_source=external`，外部 NPZ 通过 `spider_contact_target_object_local` / `eval_contact_target_object_local` 提供 `(T,2,3)` object-local target；reward 仍用 `wrist + contact_hdmi_eef_offset` 追 target。因此 E094 可以先生成外部 target 与可视化，不需要先改 MJWP reward 内核。
- [x] E092 三 case 的 CEM 任务与 split 可复用：C1 本地、C2 远程 GPU0、C3 远程 GPU1；若 E094 kinematic gate 通过，再按本地+远程三卡 full CEM 叠加运行，不 kill 现有 RL 进程。

## 2026-05-29 14:05 CST: E094 projection gate 与脚本 scaffold

- [x] 已落盘计划 `workspace/core4d/plan/100_E094_g1_handbox_target_projection_plan.md`。
- [x] 已实现并运行 `workspace/core4d/scripts/E094/build_handbox_target_projection.py`。首版 `handbox_compensated` 与直接 `support_patch` 都会把 guard/reward target 拉动过大或判定 inside，已保留为 `workspace/core4d/results/E094/handbox_target_projection_{compensated_initial,support_patch_initial}/`，不进入 CEM。
- [x] 当前候选改为 `adaptive_support`：非 raw-active 帧保留旧 ref-FK target；raw-active 帧仅在旧 target inside、非 support 且离 raw >`0.30m` 时投到 support face。结果目录 `workspace/core4d/results/E094/handbox_target_projection/`，5 case、10 summary rows、1042 per-frame rows、10 张 2D PNG 非空。
- [x] adaptive gate 结果：box023/box004 guards PASS 且 reward delta p90 `0`; D003 box021 PASS；Box026 两条 inside 降到 `0%`、support 提升到 `~96-100%`，但 reward delta p90 `0.36-0.63m`，标记 REVIEW/high-risk。
- [x] 已实现并运行 `workspace/core4d/scripts/E094/render_projection_mujoco.py`：5 个 full-body keyframe sheet 和 5 个 mp4 已生成，`ffprobe` 检查均为 `960x720`、`48` 帧。
- [x] 已启动 high subagent `019e722f-e771-7f20-a401-01266edbda42` 复核 projection 可视化，报告目标路径 `workspace/core4d/results/E094/handbox_target_projection/visual_review/high_subagent_projection_review.md`。
- [x] CEM scaffold 已建好但尚未启动：`build_cem_tasks.py` 生成 3 个 E094 override/variants；`train_E094_handbox_proj_cem.sh`、`run_E094_remote.sh`、`pull_E094_remote_results.sh`、`eval_E094_cem.py` 均已通过静态检查。
### 2026-05-29 13:36 CST - E094 MuJoCo video camera audit/fix

- User observed MuJoCo videos only show the upper body. I traced the CEM video path to `spider.viewers.render_image()`: it requested a named `front` camera, but the E091/E094 scene XMLs only define `track`/`track2`; the old exception fallback used camera id 0 (`track`), a pelvis-attached camera that can frame only the torso/upper body.
- Patched `spider/viewers/__init__.py` so missing `front` (or `video_camera: auto`) uses a free camera computed from current sim/ref body positions, with config knobs added in `spider/config.py` and `examples/config/default.yaml`.
- Verification: `python -m py_compile spider/viewers/__init__.py spider/config.py` passed, and `workspace/core4d/results/E094/camera_audit/auto_video_camera_test.png` shows full-body ref/sim framing on `e091_box004_20231003_2_083_p2/scene_act.xml`.
- Patched `workspace/core4d/scripts/train/train_E094_handbox_proj_cem.sh` to pass `video_camera=auto` for future E094 smoke/full launches.
- Added `workspace/core4d/scripts/E094/rerender_cem_autocam.py` so completed E094 CEM rollouts can be re-rendered as `*_autocam.mp4` from `trajectory_mjwp_act.npz` + `config_act.yaml` without rerunning CEM. Static compile passed; pre-completion dry run correctly skipped missing NPZ.
- Running E094 full CEM was not stopped. Current active local session continues; remote P2/P3 logs continue advancing. Since those jobs started before this patch, their in-run mp4s may still use the old camera and should be re-rendered from saved trajectories after completion.

### 2026-05-29 14:08 CST - E094 P1 full CEM completed; autocam rendered

- Local `E094P1_box004_083_p2_hbproj` full CEM completed and wrote `workspace/core4d/results/E094/cem/full/E094P1_box004_083_p2_hbproj.npz` plus original mp4. The wrapper printed a shell EOF after completion because the script was modified while that bash process was still reading it; current `bash -n workspace/core4d/scripts/train/train_E094_handbox_proj_cem.sh` passes.
- P1 eval status: `WORK`; `T=105`, contact `61.0%`, object mean/max `0.007/0.018m`, pelvis min `0.658m`, head/upper/LH-floor/RH-floor all `0.0%`.
- Re-rendered corrected camera video: `workspace/core4d/results/E094/cem/full/E094P1_box004_083_p2_hbproj_full_autocam.mp4` (`1440x480`, `210` frames). Keyframes under `workspace/core4d/results/E094/cem/full/keyframes_autocam/E094P1_box004_083_p2_hbproj/`; inspected `f0092.jpg`, full robot and object are visible for both ref/sim.
- Remote P3 had completed by this checkpoint but was not yet pulled; remote P2 still running.

### 2026-05-29 14:28 CST - E094 full CEM complete and logged

- Remote P2/P3 completed in `/home/xiayb/pHRI_workspace/spider_e094_run`; updated `workspace/core4d/scripts/pull_E094_remote_results.sh` to default to that clone and pull `*_outdir_full` directories, then pulled results/logs locally.
- Unified eval over all three variants wrote `workspace/core4d/results/E094/cem/full/full_eval_summary.{json,csv,md}`: C1 `WORK`; C2 `FAIL` due pelvis `0.440m`; C3 `FAIL` due pelvis `0.171m` and RH floor `22.0%`.
- Re-rendered P2/P3 corrected autocam videos with `workspace/core4d/scripts/E094/rerender_cem_autocam.py`; all three corrected videos are `1440x480` with expected frame counts (`210/246/164`). Visual spot checks show C2 is low-hip/prone-on-box despite excellent object/contact metrics; C3 is fall/box-flip.
- High subagent `019e7267-da6b-71d1-8b3b-4011d9e7552f` completed CEM autocam review at `workspace/core4d/results/E094/cem/full/visual_review/high_subagent_cem_autocam_review.md`, agreeing C1 supports WORK and C2/C3 should not enter RL.
- Wrote official log `workspace/core4d/log/116_E094_g1_handbox_target_projection_results.md` and updated `workspace/core4d/EXPERIMENT_TRACKER.md` with E094 summary.

### 2026-05-29 14:42 CST - E094 log clarified after user feedback

- User pointed out the E094 log blurred E093's next-step recommendation with E094's post-result recommendation. Rewrote `workspace/core4d/log/116_E094_g1_handbox_target_projection_results.md` to explicitly separate: E093 next step = target semantic repair / G1-handbox-aware projection; E094 implements that; E094 after-result next step = posture/valid-contact gate for Box026 or Holosoma RL only for C1.
- Added clearer sections for `adaptive_support` definition, rejected candidates (`handbox_compensated`, direct `support_patch`), final visualization paths, CEM setup/results, autocam camera fix, decisions, and next steps.
