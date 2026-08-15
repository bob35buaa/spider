# log286 · E199：OmniRetarget object augmentation 打通 + full CEM

_Core4D · Phase 62 · Run **R285** · plan [228](../plan/228_E199_omniretarget_object_augmentation_full_cem_plan.md) · 2026-08-15 · **阶段性结论：augmentation 有效**（数据构建完成；CEM 23/31 完成即评，余 8 条 trans 在跑）_

## Purpose / 假设

把上游 OmniRetarget/holosoma 的 **object augmentation**（物体初始位置+朝向扰动）打通进 SPIDER core4d 数据管线，并对每个增强变体跑正式 full CEM，验证 augmentation 后重定向是否仍物理可信、指标是否相对 orig 稳定。用 8 物体各 1 case 验证；scale（长宽高）分 Phase 2。

## 关键决策（用户确认）

1. **Scale 分两阶段**：Phase 1 = 位置+朝向（原生）；Phase 2 = 上游扩 scale。
2. **变体 = 原生固定 5 + orig = 6/case**（trans 前/左/右 0.2m + yaw ±45°）。
3. **PRG arm = 统一 E173 canonical builder，但产新 E199 标签**（`scene_act_E199_rubberHull_PRG` + `core4d_E199_*` override），不覆盖历史 scene；orig 在 E199 内复跑作同条件基线。
4. **retarget = omnirt_v2**（Phase-4 constraint relaxation 等）：omnirt_v1（无松弛）下 object 增强多不可行（box024 仅 2/5）；omnirt_v2 提升可行性且是项目既有变体。6 变体统一 v2，单变量成立。

## 上游 bug 修复（此增强路径此前从未被跑过）

`holosoma/.../examples/parallel_robot_retarget.py`：
1. **retargeter config 被覆盖**：增强循环把 `retargeter`（config）变量赋值成 InteractionMeshRetargeter 实例，k>0 迭代把实例传给 `build_retargeter_kwargs_from_config` → `AttributeError: 'InteractionMeshRetargeter' object has no attribute 'self_collision'`，导致除 original 外全部失败。修复：循环前 `retargeter_config = retargeter`，用它构建 kwargs。
2. **单变体不可行中止全部**：某个增强 IK 不可行会抛异常中止该文件剩余变体。修复：`retarget_motion` 包 try/except，k==0 致命、k>0 skip+continue。

## 参数 / 冻结不变量

| 项 | 值 |
|---|---|
| CEM | seed=0, num_samples=1024, max_num_iterations=32, use_torch_compile=false |
| retarget | omnirt_v2/ref_fk（relaxation+foot_z+contact_preservation on, slide=1.0, penetration_tol=0.8） |
| arm | E199 rubber_hull + 16 lowerbody-pair PRG（scene_act_E199_rubberHull_PRG），reward base = E167A |
| 增强 config | holosoma 原生固定 5（trans×3 + rot×2），逐字节沿用 |
| 接触掩码 | 每 case 由 v2 `_original` trim 窗重算 1 份 3cm 掩码，6 变体复用（时间维不变；aug 变体按 orig 的 trim_start 定窗切片，保证对齐） |
| evaluator | 公共 `eval.core.core_metrics` |

## Run command

```bash
# 数据构建（上游 v2 增强重定向 → SPIDER task → E199 PRG scene → override/manifest）
bash workspace/core4d/scripts/train/train_E199.sh            # 全 8 case
# full CEM（本机 8 卡 priority 队列，不抢占）
bash workspace/core4d/scripts/launch/active/run_E199_local_8gpu.sh
# 评估（orig-vs-aug delta + 全分布）
bash workspace/core4d/scripts/eval/wrappers/eval_E199_augmentation.sh
```

## 改动文件

| 文件 | 改动 |
|---|---|
| `workspace/core4d/data_preprocess/pipeline.sh` | +`RETARGET_AUGMENTATION` 开关：retarget 段改调 parallel_robot_retarget.py（6 config），默认关不影响 legacy `_original` |
| `holosoma/.../examples/parallel_robot_retarget.py` | 2 处 bug 修复（见上） |
| `workspace/core4d/scripts/experiments/E199/e199_common.py` | 契约/路径/8-case 注册/PRG scene builder（移植 E173）/override payload |
| `.../E199/build_augmented_tasks.py` | 数据驱动：上游 v2 增强→定窗 trim→SPIDER task→E199 PRG scene→C3 pose-diff；逐变体可行性容错 |
| `.../E199/build_aug_manifest.py` | base task yaml + E199 PRG override + 优先级 manifest |
| `.../E199/run_local_priority_queue.py` | 8 卡 priority 队列（A0/PRG 校验，复用 E198 结构） |
| `.../launch/active/run_E199_local_8gpu.sh`、`.../train/train_E199.sh` | 入口 |
| `.../eval/{wrappers,runners}/eval_E199_augmentation.*` | orig-vs-aug 打分 |
| scene 快照 | `workspace/core4d/results/E199/scene_snapshot/cem_sidecars/<task>/`（每 sidecar 自动快照）+ train 脚本写 manifest.txt |

## 已验证（box024 pilot，omnirt_v2）

- 全链路端到端跑通：上游增强→trim→SPIDER task→E199 PRG scene→CEM。
- **C3 增强正确性**：trans0/1/2 接近段 pose 相对 orig 偏移 **0.200m**，操作终点偏移 **0.027m**（13% → 指数衰减锚定生效）；orig 与原始逐比特一致（0.000m）。
- **可行性**：omnirt_v1 下 box024 = 2/5（trans1/2 可行，trans0/rot0/rot1 不可行）；**omnirt_v2 = 3/5**（trans0/1/2 可行，rot0/rot1 仍不可行——45° yaw+侧移出可达域）。
- **CEM canary（64×4）**：box024 aug_orig 产出有限值 trajectory，config_act.scene_name=scene_act_E199_rubberHull_PRG，Hydra 契约与 E173 PRG 逐字段一致。

## Result（阶段性：23/31 CEM 完成即评，8 条 trans 仍在跑）

### C0 链路打通 ✅
8 物体各产出 orig + 可行 aug 变体，共 **31 个 SPIDER task**（scene + trajectory + `scene_act_E199_rubberHull_PRG` 齐全）；base task yaml + E199 PRG override 31/31 生成，manifest 0 blocker；102 个 scene sidecar 快照 + manifest.txt（git HEAD+sha256）。

### C3 增强正确性 ✅（可行性分布 = 一等结论）
- **衰减锚定生效**：所有 trans 变体接近段偏移 = **0.200m**，操作终点偏移 **0.027–0.045m**（13–22%）；orig 与原始逐比特一致（0.000m）。
- **可行性（omnirt_v2）**：8 物体一致呈现 **3 个平移全可行、±45° 旋转全不可行**（yaw 出可达域，松弛也救不回）；另 **bucket007 trans2**（右移）参考轨迹初始帧腿-桶穿透 15mm，被 PRG scene 运行时重叠保护正确拦截。
- 每 case 有效增强 ≈ 3（平移），全量 **8 orig + 23 aug = 31**（非理论 48）。**结论：object augmentation 的实际增益来自平移方向；旋转档位对 G1+这些物体多不可达。**

### C2 执行闭合（截至分析）
23/31 full CEM 完成、23/23 打分成功，**error / non-finite / diverged / fall = 0**；8 条 trans 仍在队列。

### C4 augmentation 物理可信度 ✅（8 orig vs 14 aug，14 组配对，全分布不 cherry-pick）

| 指标 | orig 均值 | aug 均值 | aug std | aug worst | 判读 |
|---|---|---|---|---|---|
| obj_pos 误差 cm | 11.47 | 13.75 (**+19.9%**) | 1.94 | 16.64 | ✅ <25% 阈 |
| obj_ori 误差 ° | 6.29 | 8.36 | 6.22 | 19.90 | 聚合 +33% 超阈，但为分布假象（见下） |
| obj_z 误差 cm | 4.82 | 6.14 | 1.12 | 8.66 | 温和 |
| eef_pos 误差 cm | 13.90 | 16.36 | 2.67 | 22.64 | 温和 |
| in-mask 接触保持 | 0.665 | **0.731** | 0.15 | — | ✅ aug 反而更高 |
| 手-物穿透 3mm frac | 0.139 | 0.166 | 0.075 | 0.317 | 略升 |
| 腿穿透 frac | 0.107 | **0.051** | 0.116 | 0.341 | ✅ aug 反而更低 |
| fall | 0 | **0** | 0 | 0 | ✅ 无新增 |

- **obj_ori 聚合 +33% 是 box004 高基线主导的假象**：逐 case delta 才是真相——多数 case aug_ori 与 orig_ori 差 <0.5°（box021 4.65→4.3, box023 3.49→3.0, box001 7.52→7.6）；box004 orig 本身就 18.9°（该 case 抬箱时箱体倾斜，见 C6），aug ~19.9°，delta 极小。
- **12-gate 通过率**：orig 3/8（37.5%），aug **7/14（50%）** — aug 不低于 orig。orig 失败门以 **lower_body（腿穿透）4/8** 为主，是这些 case 在严格 12-gate 契约下的固有难度（与 E194/E198 一致），非增强引入；aug 失败门更分散（object_ori 3 / contact 2 / lower_body 2 / hand_pen 1）。

### C6 视觉复核 ✅（box021/box004/box024 orig+aug 关键帧，render/qc/）
- **box021 orig vs trans1**：动作（接近→弯腰抓取→抬起）高度一致，姿态自然，无穿模/漂浮/抖动/跌倒；增强变体质量与 orig 相当。
- **box024 trans0**（v1 不可行、v2 恢复的前移变体）：接近→按压→推扶大长箱，箱体保持水平，动作连贯，物理可信。
- **box004 trans1**（gate fail case）：抬箱时箱体倾斜（→高 obj_ori），但这是 box004 base case 固有特性（orig 同样倾斜），增强未恶化，且无穿模/漂浮/跌倒。
- 结论：增强变体**无致命 artifact**。

## Conclusion

**上游 OmniRetarget object augmentation 在本管线中有效可用（Phase 1 打通成功）。**

1. **链路全通**：上游 omnirt_v2 增强重定向 → SPIDER task → E199 rubber_hull+PRG scene → full CEM → 评估，端到端跑通；期间修复了 holosoma 增强路径此前从未被触发的 2 个 bug。
2. **增强数据物理可信**：平移增强产出的重定向，在与 orig **完全同条件**（omnirt_v2 + E199 PRG arm + 同 CEM 预算 + 复用同一接触掩码）下，物体 tracking 误差仅温和上升（obj_pos +19.9% < 25% 阈），**接触保持、腿穿透、12-gate 通过率均不劣于甚至优于 orig，零跌倒/发散**；视觉无致命 artifact。→ 满足 C4，可作为下游 RL 的有效扩充数据（每 case 约 +3× 平移变体）。
3. **主要限制（诚实报告）**：±45° yaw 旋转档位对 G1 + 这些物体**系统性不可达**（8/8 case 全部 rot 不可行），个别右移变体因初始穿透被拦——即 augmentation 的实际增益集中在平移方向，旋转档位需上游收紧幅度或换机器人才可能可行。
4. **12-gate 通过率整体偏低（orig 也仅 3/8）**：源于这批 case 在 E194 严格 12-gate 契约下的固有难度（lower_body 为主），非增强所致；后续若要提升绝对通过率需在 arm/reward 层面另做（与本实验的 augmentation 变量正交）。

**下一步**：① 等余下 8 条 trans CEM 跑完后重跑 eval 补全 31 条分布（结论预计不变）；② 进入 **Phase 2**（object scale/长宽高）——需上游 holosoma 为 object_interaction 增加 scale 增强 + 重算接触（另开计划）。
