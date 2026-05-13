# Pre-E060 审查: E041c 数据层 + reward task-specific 全面排查

## 状态: 📋 审查日志 (无 Run ID, 不是实验; 是 E058/E059 失败后的根因调查)

**触发**: 用户对 E059 log "stability_penalty_scale: 0.0 是根因" 的判断提出质疑 — "和 stability_penalty 没啥关系", 要求重新排查 E041c → E048 期间是否有代码/数据改动覆盖了 E041c, 以及 E041c 哪些设置是 task-specific.

**结论**: E059 原 log 把根因定位错了。真正的根因是**多层叠加**:
1. **数据层 1 (碰撞)**: hand_collision 至今仍是 5cm sphere (HDMI 3-box 改造在 E051 定位为根因之一, E052 仅在 suitcase 模板验证, **从未推广到 G1 robot.xml**)
2. **数据层 2 (margin)**: box023 collision margin 仍是 1.05x (E053 在 box025 上明确结论 0.85-0.90 最优, **从未推广到 box023**)
3. **Reward 层**: `contact_hdmi_palm_normal` hardcoded `[0,∓1,0]` 是 box025 motion 的 fingerprint, 不是 G1 解剖手掌; **数值验证证明对 box023 是噪声 (mean dot 0.04)**
4. **基础设施**: scene XML 全部在 .gitignore 内 → 历次 collision/margin 改动无任何 git 痕迹 → 任何复现都不可能

stability_penalty 不是不重要, 但是 **subordinate symptom**, 不是根因。

---

## 1. Q1: E041c → E059 期间, 代码/数据有没有覆盖 E041c?

### 1.1 代码层 (spider/ + examples/) 改动

逐个 commit 检查 e567fe7 (E041 完整 log) → HEAD:

| commit | 实验 | 改动 | 是否影响 E041c 行为 |
|--------|------|------|-------------------|
| 932b688 | E042 | wrist freeze (zero_noise_joint_keywords) | ❌ 不影响 (E041c yaml 未启用) |
| f55b3a1 | E044 | wrist weight 字段 | ❌ 不影响 (新字段默认 0) |
| 98a3dab | E045 | sigma sweep | ❌ 不影响 (只加新 yaml) |
| 1ca7557 | E047 | SBTO 算法对齐 | ❌ 不影响 (SBTO 独立分支) |
| 723de69 | E049 | mjwp.py +7 行 wrist damping + config.py +2 字段 (`apply_wrist_dof_damping`, `apply_holosoma_pd`) | ❌ 不影响 (两个开关默认 False, E041c yaml 未启用) |
| b939fdf, d0844dd, 9313122 | E050-E052 | HDMI scene 修复 | ❌ 不影响 (只动 HDMI workflow, 不动 mjwp/CEM) |

**结论**: 代码层无 silent override. E041c 的代码路径在 E058/E059 时与 E041 时代完全等价.

### 1.2 数据层 (scene XML) 改动

| commit | 改动范围 | 影响 box023 | 影响 bucket005_s2 |
|--------|---------|-------------|------------------|
| **04e7ecc (E048)** | 21 case collision = mesh AABB × 1.05 | **✅ 1.8x → 1.05x (61cm → 34cm)** | **✅ 新生成** |
| 9313122 (E052) | suitcase 模板 (HDMI 路径) | ❌ | ❌ |
| 34194be (E053) | margin sweep 0.90/0.95/1.00x | ❌ (只在 box025/bucket010/desk005) | ❌ |

**关键事实**:
- E041c 当时只在 box025/bucket010/desk005 上调过参 (TRACKER line 54). **box023 在那时碰撞盒还是 1.8x 超大, 根本不可能正确接触**.
- E048 collision fix 后 box023 才有正确大小, 但 E041c 的 reward 参数**从未在"修复后的 box023"上重新验证**.
- bucket005_s2 在 E058 之前**从未跑过 E041c** (E015/E020 跑的是 bucket005, 不同 segment, 且当时 reward stack 是 E015 时代).
- E048 视频复查已经发现 "box023 摔倒, ObjPos=14cm 是假象" (TRACKER line 63), 但当时归因到 collision fix, 没归因到 reward task-specific.

**真正的反思**: E041c 被反复在 box025/bucket010/desk005 上调参 (E042-E047, E053), **形成了对这 3 个 case 的 overfitting**. 跨 case 推广失败的本质是 reward 参数从未跨过 calibration set.

---

## 2. Q2: E041c 配置里哪些是 task-specific?

逐字段读 `core4d_e041c.yaml`, 找到 4 处与 box025 强绑定:

### 2.1 ⭐⭐⭐ Hardcoded palm normal (最大嫌疑, 数值已验证)

```yaml
contact_hdmi_palm_normal_left:  [0.0, -1.0, 0.0]
contact_hdmi_palm_normal_right: [0.0, +1.0, 0.0]
```

**起源** (E041 log line 13-20, commit aa5c670):
> 通过分析 **box025 ref** 中 G1 wrist 在多个接触帧 (t=30~90) 的旋转矩阵与 wrist→object 方向的点积:
> ```
> Left wrist:  -y 轴一致指向物体 (dot = -0.61 ~ -0.87)
> Right wrist: +y 轴一致指向物体 (dot = +0.67 ~ +0.96)
> ```
> 结论: palm_normal_left = [0,-1,0], palm_normal_right = [0,+1,0]

**这不是 G1 解剖手掌法向, 是 box025 ref 在 lateral grasp 中 wrist 哪个轴恰好平均指向物体的统计观察**. 被当成 "G1 anatomy 常量" 写进 yaml, 实际是 motion-specific.

### 2.2 数值验证 — 复现 E041 分析方法, 应用到 3 个 case

每帧把 ref qpos 加载到 G1 model, 计算 wrist 各轴 (±x, ±y, ±z) 与 wrist→object 方向单位向量的点积, 统计接触帧的均值/正比例:

#### box025 (E041 校准 case) — hardcode 完美

| Axis | L mean dot | L >0 frac | R mean dot | R >0 frac |
|------|-----------|-----------|-----------|-----------|
| **-y (hardcoded L)** | **+0.78** | **98%** | -0.78 | 2% |
| **+y (hardcoded R)** | -0.78 | 2% | **+0.80** | **100%** |
| +x | -0.02 | 71% | +0.24 | 90% |

→ E041 当时找的最优轴在 box025 上没问题.

#### box023 (E048+E059 摔) — hardcode 在 L 上是噪声

| Axis | L mean dot | L >0 frac | R mean dot | R >0 frac |
|------|-----------|-----------|-----------|-----------|
| -y (hardcoded L) | **+0.04** | **51%** | -0.51 | 14% |
| +y (hardcoded R) | -0.04 | 49% | **+0.51** | 86% |
| **+x (real best)** | **+0.78** | **100%** | **+0.61** | **100%** |
| +z | +0.31 | 90% | +0.06 | 51% |

→ **L hardcode 是纯噪声** (dot ≈ 0, 50/50 正负), CEM 收到 L 朝向项是随机扰动.
→ **真实最优是 +x (fingers 轴), 不是 ±y**. CEM 被向错误方向推 (转 wrist 让 y 指物体, 应该让 x 指物体).

#### bucket005_s2 (E058 摔) — hardcode 在 L 上居然是对的

| Axis | L mean dot | L >0 frac | R mean dot | R >0 frac |
|------|-----------|-----------|-----------|-----------|
| **-y (hardcoded L)** | **+0.79** | **100%** ✅ | -0.60 | 0% |
| +y (hardcoded R) | -0.79 | 0% | **+0.60** | **100%** |
| +x | +0.47 | 100% | **+0.64** | **100%** |

→ **L hardcode 实际正确**! (跟 box025 同分数 +0.79). 这部分**纠正了我之前的判断** "bucket005_s2 因 quat 90° palm normal 朝错".
→ R hardcode 是正的但弱于 +x (+0.60 vs +0.64).
→ R 上 hardcode 不是噪声, 但是次优.

### 2.3 三 case 综合解读

- **box025**: hardcode 完美 → CEM 受益 → 站住 (但因 stability 也关闭, 到 E048 视频也是趴着)
- **box023**: hardcode 在 L 上是噪声 (扰动 CEM), R 上是次优 (轻微误导) → CEM 找不到稳定 wrist pose → 摔
- **bucket005_s2**: hardcode L 正确 + R 次优 → 不是 reward 元凶, 还有别的原因 (mocap 质量? collision? 见 §3)

E041 hardcode "L=-y / R=+y" **不是普适解, 是 box025 ref motion 的 fingerprint**. 跨 case 的有效性参差不齐.

### 2.4 物理层进一步加重: hand_collision 是 sphere

`spider/assets/robots/unitree_g1/robot.xml`:
```xml
<default class="hand_collision">
  <geom type="sphere" pos="0.1 0.0 0.0" size="0.05" />
</default>
```

`scene.xml` (例如 box023_person1):
```xml
<geom name="lh" class="hand_collision" />  <!-- 继承球 -->
<geom name="rh" class="hand_collision" />
```

**球各向同性**, 物理上手心手背完全等价. 这意味着:
- E040 视频里看到的"手背接触" 是**纯视觉伪装** (rubber_hand mesh 朝向不对, 球碰撞实际仍贴箱)
- E041 引入 ori_weight=0.3 的整个动机 (惩罚手背接触) 在物理上是无效的 — 它只在让生成视频"看起来好"
- 对 box025 reward 数值上恰好和真实最优一致 (lateral grasp), 所以"看起来有效"
- 对 box023 reward 完全不对齐, 噪声直接误导 CEM

**E051 早就定位过这个**: "scene 物理配置: hand=1sphere (应3boxes), armature=1.0 (应0.01)" (TRACKER line 66). 但 E052 只在 HDMI suitcase 路径下试改 (E052c 因 euler convention bug 反而更差), **从未把 3-box hand 改进 G1 robot.xml**.

### 2.5 其他次要 task-specific (未数值验证, 仅推测)

- `contact_hdmi_eef_offset: [0.05, 0, 0]`: EEF 局部 +x 5cm 偏移当作 palm point. 对所有 case 一刀切, 实际 5cm 在不同 grasp 下指向不同方向.
- `init_pos_actuator_gain: 500.0`: 物体 PD 锚定强度. box025/box023 = 5kg ok, **bucket005_s2 = 2kg** (1/2.5x) 同样 500 强度可能让物体初始化时漂移.
- `local_frame_pos_sigma: 0.5m`: 跟踪 bandwidth, 对大箱搬胸 OK, 对小箱深蹲举可能太松.

---

## 3. Q3: 数据层 — box023 collision margin 还是 1.05x

E053 (TRACKER line 69) 在 box025 上明确结论:

> Pelvis min: **0.575m (1.05x)** vs **0.654m (0.85x)** — 1.05x 下机器人更容易倒
>
> 建议 box025: **0.90**

E053 只 sweep 了 box025/bucket010/desk005, **box023 当时被遗漏**. 现状:

| Case | 当前 collision half-size | E053 推荐 margin | 差距 |
|------|-------------------------|----------------|------|
| box023 | 17.86, 18.30, 20.60 cm (1.05x mesh) | 0.90x → 应该是 (15.31, 15.69, 17.66) | -14% 体积 |
| bucket005_s2 | 15.76, 16.24, 23.11 cm (1.05x) | 未 sweep, 但是侧地物体, 可能需独立判定 | 未知 |

E058/E059 baseline 摔**至少有一部分原因是 box023 用了 E053 已知次优的 1.05x margin**. 这与 reward task-specific 是**两个独立 bug**.

---

## 4. Q4: 基础设施 — scene XML 没进 git (复现性灾难)

`.gitignore:205`: `example_datasets/` 整个目录被忽略.

**直接后果**:
- 04e7ecc (E048 21 case collision fix) 改的 scene XML 没记录
- E053 在 3 case 上的 0.90/0.95/1.00 margin sweep 最终用了哪个版本不可考
- E052 的 suitcase 模板改造没记录到 G1 路径
- 现在 box023 scene.xml 是哪一版只能靠 mtime 推
- 任何人 clone 后跑 E058/E059 **不可能复现**

**已修复** (本次 commit b190785):
1. `git add -f` 9 个活跃 case (29 文件) 进主 git
2. `workspace/core4d/scripts/convert/snapshot_scenes.sh` 给每个实验做 frozen snapshot + sha256 + git HEAD manifest
3. `.claude/rules/experiment.md §7` 和 `.claude/skills/experiment-planning-zh/SKILL.md §10b` 强制 "训练前必须 snapshot scene"

---

## 5. 对 E059 原 log 的修正

E059 原 log "下一步路径分析" 第 A 节写的是:

> A. (必须先做) 修 baseline reward stack — 加回 stability_penalty
> 改 core4d_e041c.yaml 的衍生 core4d_e060_stab.yaml: stability_penalty_scale: 1.0

**这个判断窄了**. 加 stability_penalty 假设 reward stack 本身正确, 只是缺一项稳定约束. 实际是:
- 数据层 (碰撞球 + margin 1.05x) 让 box023 baseline 物理上就难以稳定
- Reward 层 (palm normal hardcoded) 在 box023 上是噪声, 主动误导 CEM

**正确的修复顺序**:
1. **先修数据层** (3-box hand + box023 margin 0.90), commit scene snapshot
2. **再跑 baseline** (E041c 不动), 看是否站住
3. 如果**站住** → reward 不是元凶, E041c 没问题, 只是数据不对
4. 如果**还摔** → reward 才是元凶, 做 ori_weight=0 消融 (E060b)
5. stability_penalty 留作最后兜底 (E060c)

stability_penalty 在数据层修对 + reward 不是元凶时, 加上去可能是无意义的 (因为本来就稳了). 在 reward 是元凶时, 加上去可能掩盖问题 (用稳定项压住错误方向).

---

## 6. 关键反思 (跨实验)

### 6.1 calibration set overfitting 是 CORE4D Phase 11-16 的隐性灾难

E041 → E053 跨 13 个实验, **全部在 box025 + bucket010 + desk005 上做 reward sweep**. E041c 被反复 cherry-pick 后认为是"最佳", 实际是这 3 个 case 的 fingerprint. 跨 case 失败 (E058/E059) 是必然.

**教训**: 任何 reward 调参实验, 必须包含至少 5 个 case (覆盖不同 grasp 类型 + 不同物体大小), 否则结论不可推广.

### 6.2 "看起来好" vs "物理上对" 的混淆

E041 引入 ori reward 是为解决 E040 视频里的"手背接触". 但 hand_collision 是球, 物理上根本不分手心手背. 整个 E041 设计建立在**视觉伪装**之上.

**教训**: reward 设计前必须检查 collision geometry. 如果 collision 不区分某种姿态, 任何针对该姿态的 reward 都是无效的视觉补丁.

### 6.3 数据生成路径不入 git = 不可复现

E048 fix 一次性修了 21 case, 但脚本和最终 XML 都没 commit, 导致 E053-E059 跑的到底是哪一版完全靠 mtime + 集体记忆. 这是真实损害复现性的 bug, 比代码 bug 更难定位.

**教训**: 数据生成脚本 + 输出的最终 XML **必须 git 跟踪**, 即使在 .gitignore 大目录里, 也用 `git add -f` 单独入. 这次的 dual-safeguard 机制 (本次 commit) 解决了这个.

---

## 7. 改动文件 (本次 audit 产出)

| 文件 | 改动 | commit |
|------|------|--------|
| `example_datasets/processed/core4d/unitree_g1/humanoid_object/{9 cases}/scene*.xml` | force-add 29 文件入 git | b190785 |
| `workspace/core4d/scripts/convert/snapshot_scenes.sh` | 新建快照脚本 (sha256 + git HEAD manifest) | b190785 |
| `.claude/rules/experiment.md` | +§7 Scene/Data XML Reproducibility (Dual Safeguards) | b190785 |
| `.claude/skills/experiment-planning-zh/SKILL.md` | +§10b 同义中文规则 | b190785 |
| `workspace/core4d/log/70_pre_E060_audit_E041c_data_and_reward.md` | 本文件 | (本次 commit) |

---

## 8. 对 E060 计划的指导 (详见 plan/70_E060_*_plan.md)

### 必须做 (数据层修复, 优先级 P0)

1. **Hand collision: sphere → 3-box**
   - 抄 HDMI suitcase 配置 (E051/E052 已实现的)
   - 改 `spider/assets/robots/unitree_g1/robot.xml` 的 `hand_collision` default
   - 所有继承此 default 的 case 自动更新
   - 物理上区分手心手背, 让 ori reward 真正有意义

2. **box023 collision margin: 1.05x → 0.90x**
   - 用 E053 的 `set_collision_margin.py` 复用
   - 改 `box023_person1/scene.xml` 和 `scene_act.xml`
   - 调用 snapshot_scenes.sh 入快照

### 必须做 (实验设计, 优先级 P0)

3. **E060.0 验证 baseline 是否单凭数据层修复就能站住**
   - 改动: 仅 §1 + §2 (无 reward 改动), E041c 原样
   - case: box023 + bucket005_s2 (并行 GPU0/1)
   - 通过判别: pelvis_min_intent ≥ 0.40m (蹲下姿态合格)

### 条件分支

4. **如果 E060.0 通过** → reward 不是元凶, E060 done.
   - 进 E061 推广其他 case (bucket007, desk021)
5. **如果 E060.0 不通过** → 进 reward 消融
   - E060.1: ori_weight = 0 (砍掉 hardcoded normal 的噪声)
   - E060.2: ori_weight = 0 + palm_normal 改成 case-correct +x (基于 §2.2 数值结论)
   - E060.3: 如果都不行, 加 stability_penalty 兜底

### 不做 (避免错误优化方向)

- ❌ 不要在数据层修复前跑 reward sweep (overfitting 风险)
- ❌ 不要在不知道 reward 影响的情况下加 stability_penalty (掩盖问题)
- ❌ 不要在 box023/bucket005_s2 上验证不通过就推广其他 case

---

## 9. 关联 commit

- `b190785` infra(core4d): scene XML reproducibility — dual safeguards + 9 active cases tracked
- `706537c` exp(core4d): E057-E059 Path B-CEM — warmstart hook + honest dual failure
- (待) plan/70_E060_*.md
- (待) E060 实验执行 commit
