# plan228 · E199：OmniRetarget object augmentation 打通 + full CEM（8 物体 × 6 变体）

_Core4D · Phase 62 · 2026-08-15 · Run **R285**（Phase-1 full-CEM 批）· evaluator `core4d-e154-physics-contact-v1` · 承接 [plan226](226_E198_g1xa2_factorial_and_E192_a2_expansion_plan.md) / [plan227](227_E198_box001_g1a2_a2_supplement_plan.md)_

## 0. 一句话

打通「上游 OmniRetarget/holosoma object augmentation → SPIDER 数据管线 → 正式 full CEM 重定向」的全链路，挑 **8 个物体各 1 个 case**，每 case 跑上游原生支持的 **object 初始位置 + 朝向** 增强（`original + trans×3 + rot×2 = 6 变体`），共 **48 条 full CEM**（本机 8 卡 priority 队列），量化+可视化验证「augmentation 后重定向是否仍物理可信 / 指标是否稳定」。**object 长宽高（scale）增强上游 object_interaction 路径原生不支持，分 Phase 2 上游扩展（本计划只出设计草案，不实现）。**

## 1. Context / 背景与动机

- **目标**：扩充下游 RL 训练数据。上游 OmniRetarget 的 `--augmentation` 能对同一段人-物交互生成多个「物体初始摆放不同」的重定向变体，等价于低成本地把 1 条轨迹扩成 N 条。我们想把这个能力接进 SPIDER 的 core4d 数据管线（`workspace/core4d/scripts/data_construction_v3`），当前管线只跑 `_original`，无 augmentation 参数。
- **上游机制调研结论**（`holosoma/src/holosoma_retargeting/holosoma_retargeting/`）：
  - `examples/parallel_robot_retarget.py::generate_augmentation_configs`（`object_interaction` 分支）定义固定 6 变体：
    - `original`
    - `trans_0/1/2` = 人体坐标系下 **前 [0.2,0,0] / 左 [0,0.2,0] / 右 [0,−0.2,0]**（米）
    - `rot_0/1` = 绕世界 z **±45°** yaw，各叠一个 ±0.2m 侧移
  - `src/utils.py::augment_object_poses`：平移/旋转**全量施加于物体开始运动前的接近段**，物体开始运动后按指数衰减回原轨迹（`translation_tau=50`、`rotation_tau=25` 帧）→ **只扰动初始摆放/接近，操作终点仍锚定原轨迹**。augmentation run 以 `_original` 重定向的机器人 qpos 作为 IK warm-start（**必须先跑 original**）。
  - **单序列 `robot_retarget.py --augmentation` 只产 1 个 `_augmented.npz`（默认平移 [0.2,0,0]）**；要拿全 6 变体需走 `parallel_robot_retarget.py` 的 config loop，或在我们的 wrapper 里循环 config。
  - **object scale（长宽高）在 `object_interaction` 路径不存在**，仅 `climbing` 用 `object_scale_augmented`（z-scale [0.8/0.9/1.1/1.2] + rescale URDF/XML）。→ 归入 Phase 2。
  - **无 terrain 增强于 object_interaction**（terrain/ground 仅 climbing），故用户「不需要 terrain」天然满足。
- **SPIDER 侧集成点**：`workspace/core4d/data_preprocess/pipeline.sh`（被 `data_construction_v3/stages/s3_retarget/run_stage2b.py` 调用）第 331–354 行执行 `robot_retarget.py`（单序列、仅 `_original`）。augmentation 需在此处扩展：先 original 再循环 5 个增强 config，每个增强 NPZ 各自 trim → 生成独立 SPIDER task（scene + trajectory）→ 生成 CEM override → full CEM。
- **接触掩码可复用**：`generate_core4d_contact_masks.py` 产出的是**时间维**接触掩码（哪些帧手接触物体），对物体的刚性平移/旋转不变，故 6 变体**共用 original 的接触掩码**（Phase 1 不重算）。
- **config 生成链**（已核实）：`s5_handoff/export_cem_overrides.py` 自动生成 base task yaml `core4d_dcv3_..._{case}.yaml`（含 `task:`、`contact_hdmi_mask_path`、`defaults: core4d_E167_...` reward 基座）；对象级 PRG override（如 `core4d_E173_box024_..._PRG.yaml`）只 `defaults` 到 base task + `scene_name` + gate 字段，**不硬编码 task**。→ 每个增强变体需：新 base task yaml（swap `task:` / mask 路径）+ 复用对象 PRG override（换 `defaults` 行）。

## 2. 冻结不变量（与 E197/E198 full-CEM 对齐，保证 augmented vs original 同条件）

| 项 | 冻结值 |
|---|---|
| CEM | `seed=0`、`num_samples=1024`、`max_num_iterations=32`、`+use_torch_compile=false`（Full） |
| arm | **A0 / PRG（no-gravcomp, PRG leg-gate）**——本次不做 G1/A2 干预，专注 augmentation 变量 |
| retarget variant | **`omnirt_v2/ref_fk`**（Phase-4：constraint relaxation + foot-z + contact-preservation + slide 1.0 + penetration-tol 0.8）——object 增强把物体移出可达域，omnirt_v1（无松弛）下多数增强 IK 不可行（box024 仅 2/5）；改用项目既有 omnirt_v2 提升可行性，**6 变体（含 orig）统一 v2**，单变量成立。aug task 命名 `dcv3_omnirt_v2_ref_fk_..._aug_{v}` |
| arm 标签 | E199 复用 E173 canonical rubber_hull+16-lowerbody-pair PRG builder，但产**新 E199 标签**产物：`scene_act_E199_rubberHull_PRG` + `core4d_E199_*` override，**不覆盖历史 scene**；全 8 物体统一（含 orig 复跑） |
| 上游修复 | `holosoma parallel_robot_retarget.py` 两处 bug（此增强路径此前从未被跑过）：① 增强循环把 retargeter *config* 变量覆盖成 *instance* → k>0 全部 `self_collision` AttributeError；② 单个不可行变体会中止后续变体。均已修（config 独立命名 + 逐变体 try/except skip） |
| 上游增强 config | holosoma 原生固定 5（trans×3 + rot×2），**逐字节沿用**，不改数值 |
| 接触掩码 | 复用各 case `_original` 的 `raw_contact_mask_3cm.npz`（时间维，变换不变） |
| evaluator | 公共 `eval.core.core_metrics`，`core4d-e154-physics-contact-v1` |
| 参考系 | 各增强变体的参考 = 其自身增强后轨迹的 FK（`contact_hdmi_target_source: ref_fk`），**不是 original 参考**（增强改了物体轨迹，参考必须同步） |

**单变量纪律**：original 与 5 个增强之间**唯一差异**是上游 augmentation config（物体接近段的平移/旋转）；CEM 参数、arm、reward override、evaluator 全部逐字段相同。

## 3. 执行范围

### 3.1 8 个 case（每物体 1 个，pin 到已有 `_original` dcv3 task）

| # | object | base target_task（`_original` 已存在，含 trajectory+scene，已核实） |
|---|---|---|
| 1 | box001   | `dcv3_omnirt_v1_ref_fk_box001_20231003_1_039_p1` |
| 2 | box004   | `dcv3_omnirt_v1_ref_fk_box004_20231003_2_082_p1`（E198 代表 case） |
| 3 | box021   | `dcv3_omnirt_v1_ref_fk_box021_20231011_034_p1`（E198 代表 case） |
| 4 | box023   | `dcv3_omnirt_v1_ref_fk_box023_20231008_045_p1`（E198 代表 case） |
| 5 | box024   | `dcv3_omnirt_v1_ref_fk_box024_20231011_026_p1`（E198 代表 case） |
| 6 | bucket003| `dcv3_omnirt_v1_ref_fk_bucket003_20231018_001_p1` |
| 7 | bucket004| `dcv3_omnirt_v1_ref_fk_bucket004_20231002_021_p1` |
| 8 | bucket007| `dcv3_omnirt_v1_ref_fk_bucket007_20231003_1_021_p1` |

> box004/021/023/024 沿用 E198 代表 case，方便与历史 full-CEM 结果横向对照；box001/bucket00x 取该物体已 preprocess 的首个 p1 case。若用户有偏好 case 可替换（改本表即可）。

### 3.2 每 case 6 变体命名

增强变体走**独立 SPIDER task**，命名 `{base_target_task}__aug_{variant}`，variant ∈ {`orig`,`trans0`,`trans1`,`trans2`,`rot0`,`rot1`}。`orig` = 复跑 original（同 E199 frozen config，作为**同条件基线**，不复用历史 rollout 以避免 config drift）。

**总量**：8 × 6 = **48 条 full CEM**。

### 3.3 本机 8 卡 priority 队列（沿用 E198 调度器）

- 复用 `workspace/core4d/scripts/experiments/E198/run_local_priority_queue.py` 的模式：查空闲显存派发、**与他人 job 叠加共跑、不 kill/不抢占**、resume-safe、记录落卡 GPU id。
- 为 E199 新建 manifest builder + launch 入口（见 §5）。
- GPU：`GPUS=0-7`、`PER_GPU_MEM_MIB=5000`、每卡 1 run。tier：**P0 每物体的 `orig`（8 条，先出基线）→ P1 trans×3（24）→ P2 rot×2（16）**。

## 4. 需要修改/新增的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `workspace/core4d/data_preprocess/pipeline.sh` | retarget 段（~L324-358）增加 `RETARGET_AUGMENTATION` 开关：先跑 `_original`，再循环 5 个增强 config 调 `robot_retarget.py --augmentation`（或调 `parallel_robot_retarget.py`）产出 5 个增强 NPZ；trim 段对每个增强 NPZ 各 trim；SPIDER 段对每个变体生成独立 task（`--task {target}__aug_{variant}`），接触掩码复用 original。**改动 config 驱动、默认关，不影响现有 `_original` 行为** |
| 2 | `workspace/core4d/scripts/data_construction_v3/stages/s3_retarget/run_stage2b.py` | variant params 增加 `augmentation`（bool）；透传为 `RETARGET_AUGMENTATION` 环境变量到 pipeline.sh；manifest 记录增强变体行（每变体 1 行，含 `aug_variant` 列） |
| 3 | `workspace/core4d/scripts/data_construction_v3/stages/s5_handoff/export_cem_overrides.py` | 支持为 `__aug_{variant}` task 生成 base task yaml（swap `task:` / mask 路径指回 original mask）；对象 PRG override 复用（换 `defaults` 行） |
| 4 | `examples/config/override/core4d_dcv3_..._{case}__aug_{variant}.yaml` ×48 | 自动生成的 base task yaml（由 #3 产出，`git add -f`） |
| 5 | `examples/config/override/core4d_E199_{case}__aug_{variant}_PRG.yaml` ×48 | full-CEM override（复制对象 PRG，换 defaults + task） |
| 6 | `workspace/core4d/scripts/experiments/E199/build_aug_manifest.py` | 生成 48 行 full-CEM manifest（tier/override_id/task/scene/trajectory/mask/sha256/cem_*） |
| 7 | `workspace/core4d/scripts/experiments/E199/e199_common.py` | 复用 E198 common 的路径/序列化/GPU 工具（薄封装或直接 import） |
| 8 | `workspace/core4d/scripts/launch/active/run_E199_local_8gpu.sh` | 8 卡 priority 队列入口（复用 `run_local_priority_queue.py`，指向 E199 manifest） |
| 9 | `workspace/core4d/scripts/train/train_E199.sh` | 数据构建入口：snapshot scenes → 上游增强重定向 → SPIDER preprocess → 生成 override/manifest（rule 8/10b：首步调 snapshot_scenes.sh） |
| 10 | `workspace/core4d/scripts/eval/wrappers/eval_E199_augmentation.sh` + `runners/eval_E199_augmentation.py` | 用公共 `eval.core.core_metrics` 对 48 rollout 打分，按 (object, variant) 聚合 |
| 11 | `workspace/core4d/results/E199/scene_snapshot/` | 48 条用到的 scene XML + meta 快照 + `manifest.txt`（git HEAD + sha256） |

## 5. 关键前置（reproducibility，rule 7 / 10b）

1. **上游增强产物校验**：每 case 6 个 NPZ（1 orig + 5 aug）齐全；增强 NPZ 的物体接近段 pose 与 original 有可测差异、操作终点趋同（抽查 quat/pos diff 曲线，验证指数衰减生效）。
2. **接触掩码复用合法性审计**：抽 1-2 case，确认增强前后 in-mask 帧区间一致（时间维不变）。
3. **SHA parity**：48 变体的 trajectory / scene_act / override / contact_mask 逐 case sha256 记入 `e199_aug_authority.tsv`；`orig` 变体的 CEM config 与 trans/rot 变体逐字段 diff = 仅 task/scene/trajectory 不同，reward/gate/CEM 全同。
4. **scene 快照**：48 条 scene XML + `scene_act_meta.json` 快照到 `results/E199/scene_snapshot/`，含 `manifest.txt`。
5. **base task 激活入 git**：8 个新物体 case 的增强 task scene（`scene.xml`/`scene_act.xml`/`scene_act_meta.json`/`task_info.json`）`git add -f`。

## 6. Claims / 可验证声明

| Claim | 判据（量化，无宽松阈值） |
|---|---|
| **C0 链路打通** | 8 case 各产出 6 个 SPIDER task（scene+trajectory+override 齐全）；48/48 base task yaml + PRG override 生成；SHA parity 全过 |
| **C1 单变量** | 48 run：`orig` vs `trans*/rot*` 的 CEM/reward/gate/evaluator 配置逐字段相同，唯一差异 = 上游增强 config；审计脚本 0 违例 |
| **C2 执行闭合** | 48/48 full CEM 完成；48/48 打分成功；error / non-finite / diverged = 0 |
| **C3 上游增强正确性** | 每增强变体：物体接近段 pose 相对 original 有非零刚性偏移（trans ≥0.15m 或 yaw ≥30°），操作终点 pose 与 original 偏差 ≤ 接近段偏差的 20%（指数衰减锚定生效） |
| **C4 augmentation 物理可信度** | 增强变体 full-CEM 的 6 项核心指标（body/obj-pos/obj-ori/obj-z/in-mask-contact/物理安全）**mean+std+worst-case** 相对同 case `orig` 无显著劣化：obj-pos/ori 误差 mean 增幅 ≤ 25% 且无新增 fall/diverged。**报告全 48 分布，不 cherry-pick** |
| **C5 数据增益方向** | 增强变体覆盖的物体初始摆放范围（初始 obj pose 的分布跨度）相对 orig 显著扩大（定量：初始 obj x-y 位置 spread、yaw spread）；为下游 RL「数据变多且有效」提供证据 |
| **C6 可视化门（强制）** | 每 case 至少渲染 `orig` + 2 增强变体的 rollout 视频，`/video-frames` 抽关键帧（grasp/接触/终点），逐例填写「实际观察」：检查穿模/漂浮/抖动/不自然姿态；side-by-side orig vs aug |

## 7. 成功标准

| 指标 | 基线（同 case `orig`） | 增强变体目标 |
|---|---|---|
| 链路 | — | 48/48 task 产出 + full CEM 完成 |
| obj-pos 误差 (mean/worst) | orig 数值 | mean 增幅 ≤25%，worst 不出现灾难性发散 |
| obj-ori 误差 (mean/worst) | orig 数值 | 同上 |
| 物理安全 (fall/non-finite/diverged) | 0 | 0（无新增） |
| in-mask contact 保留 | orig 数值 | 不显著回退（阈值同 E198 C5） |
| 视觉 | orig 无穿模/漂浮 | 增强变体同样无穿模/漂浮/抖动（视频核实） |

**判定**：若 ≥80% 增强变体（40/48）满足 C4 且视觉无致命 artifact → augmentation 链路可用于下游扩数据，进 Phase 2（scale）+ 全量放量；否则定位失败模式（哪些平移/旋转档位破坏物理），收紧增强幅度或增加 CEM 迭代。

## 8. Phase 2（object 长宽高 / scale，本计划只出草案，不实现）

上游 `object_interaction` 无 scale 增强，需改 holosoma：
1. `generate_augmentation_configs` object_interaction 分支增加 `scale` 变体（如各轴 ±10%/±20%）。
2. `setup_object_data`/object mesh 采样点按 scale 缩放；object URDF/mesh 重生成缩放版（参考 climbing 的 `create_scaled_*`）。
3. object 轨迹 pose 的平移分量随 scale 调整（物体变大/变小后接触几何变化，需重解 IK 而非仅缩 mesh）。
4. 接触点/接触掩码可能失效（几何变了）→ 需重算 contact，**不能复用 original mask**。

风险：scale 改变接触几何，是否物理可信需单独验证；比 trans/rot 复杂。→ 待 Phase 1 打通并确认收益后启动，另开 plan。

## 9. 训练/执行命令

```bash
# 数据构建（上游增强重定向 + SPIDER preprocess + override/manifest 生成；首步 snapshot）
bash workspace/core4d/scripts/train/train_E199.sh

# full CEM（本机 8 卡 priority 队列，与他人 job 共跑不抢占）
GPUS=0,1,2,3,4,5,6,7 PER_GPU_MEM_MIB=5000 MAX_PER_GPU=1 \
  bash workspace/core4d/scripts/launch/active/run_E199_local_8gpu.sh

# 评估（公共 evaluator，按 object×variant 聚合）
bash workspace/core4d/scripts/eval/wrappers/eval_E199_augmentation.sh
```

## 10. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 增强 config 幅度过大 → 物体初始摆放不可达 / IK warm-start 发散 | C3/C4 逐档位报告；若某档位系统性破坏，收紧幅度（如 trans 0.2→0.1、rot 45°→30°） |
| 复用 original 接触掩码在大 yaw 下失配 | §5.2 抽查审计；若失配则该 case 重算 mask |
| 48 条 full CEM 与他人 job 抢显存 | priority 队列不抢占、查空闲派发，resume-safe |
| pipeline.sh 已标 DEPRECATED | 改动 config 驱动 + 默认关，不破坏现有 `_original`；同时在 run_stage2b.py 层记录 manifest（v3 合规） |
| `orig` 变体与历史 full-CEM 结果不一致 | 有意为之——E199 用统一 frozen config 复跑 orig 作同条件基线，避免跨实验 config drift 混淆 augmentation 变量 |

## 11. 待用户确认

1. 8 个 case 的具体 identity（§3.1）——默认按上表（box004/021/023/024 沿用 E198 代表 case）；如需换 case 请指定。
2. 是否接受「`orig` 变体在 E199 内复跑」而非复用历史 rollout（推荐复跑，保证同条件）。

**未获批准前不写脚本、不占 GPU、不改 scene。**
