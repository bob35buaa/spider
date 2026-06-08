# 物理数据增广调研：在 CORE4D 人机协作上「改另一侧的力」增广

日期：2026-06-05
范围：**不建模 partner / 不使用自建 `partner_force_spring_kp` / support-proxy** 的前提下，能否将一条已跑好、验证过的动力学重定向轨迹，通过改变物体另一侧施加的力来增广数据。
依据：对 SPIDER 论文 §2.5 + 代码（`mjwp.py` / `config.py` / `sampling.py` / `run_mjwp.py` / `hdmi.py`）的代码级核查。
结论先行：**核心想法成立（就是论文 §2.5 physics variation），但当前代码做不到，存在两个硬阻碍 + 两条必须遵守的原则。**

---

## 0. 工作流前提（决定一切）

你们走的是 **HDMI loco-manipulation** 路径：CORE4D 箱子的 freejoint 被换成 **3 slide + 3 hinge 共 6 个 PD actuator**（`spider/simulators/hdmi.py:448-516`），物体可被 `object_pd_override` 伺服到 reference。

> **共享前提（所有动力学类增广都受制于它）**：只要箱子还被 `object_pd_override`（kp=2000，`config.py:155`）强伺服到 ref，改任何动力学量（力 / 质量 / 摩擦）都会被 servo 吸收 → 增广是 **no-op**。
> 量化：kp=2000 的位置 servo 跟踪等于 ref 的目标时，外加力 F 的稳态偏移约 `F/kp`，例如 40N 仅 ~2cm 就被吸收。
> ⇒ 所有 Group B/C/D 增广都要求：**箱子不被强伺服**（`object_pd_override=false` 走会衰减到 0 的 contact-guidance 脚手架，或真 freejoint 无驱动），且**每个变体重跑 CEM**（见原则 1）。

---

## 1. 当前外力机制（代码现状）

`apply_perturbation` — `spider/simulators/mjwp.py:2318-2334`：
```python
xfrc_applied[:, right_obj_id, :3] = config.perturb_force   # 2328
xfrc_applied[:, right_obj_id, 3:] = config.perturb_torque  # 2329
xfrc_applied[:, left_obj_id,  :3] = config.perturb_force   # 2331
xfrc_applied[:, left_obj_id,  3:] = config.perturb_torque  # 2332
```

- **通道**：`data.xfrc_applied`——MuJoCo 每 body 的笛卡尔外力（世界系，作用于 body COM），在约束求解器**加性**注入，与 `ctrl`/actuator 完全独立。
- **常量 / 标量**：`perturb_force`、`perturb_torque` 是 `float` 标量（`config.py:74-75`，默认 0.0），被广播到 3 个力分量和 3 个力矩分量 → 力 = `[f,f,f]`，**只有沿 (1,1,1) 对角线的幅度，没有方向**。不是真正的 6 维 wrench。
- **不分 env**：`xfrc_applied[:, obj_id, :]` 对所有 env 写同一值 → 所有 `num_samples` 个 env 力相同。
- **不随时间变**：每个 `step_env` 原样重施（`mjwp.py:3269`），整段 rollout 不变。
- **来源**：直接读 `config`，**不在 `env_param` 里** → 是**全局**量，不进 DR group 机制。
- **作用 body**：`"right_object"` / `"left_object"`（双手灵巧手命名）。**不认名为 `"object"` 的 body。**

---

## 2. 两个硬阻碍

### 阻碍 1（硬，必须改代码）：力作用不到 CORE4D 的箱子
CORE4D 协作场景物体 body 名就叫 **`"object"`**（已在 `box021_person1/scene.xml` 核实），`object_pd_override` 也是按 `embodiment_type == "humanoid_object"` 门控（`config.py:860`）。
`apply_perturbation` 只查 `"right_object"`/`"left_object"`，对 CORE4D 都返回 `-1`，两个 `if` 全跳过 → **当前 `perturb_force` 对你的工作流是彻底 no-op。**
（真正能对 `"object"` 施力的是 `_apply_partner_force`，`mjwp.py:2434`——正是你要排除的那套。）

### 阻碍 2（物理）：强伺服抵消外力
见 §0 共享前提。kp=2000 servo 把箱子钉在 ref 上，外力被 `F/kp` 量级吸收。
**要让"另一侧的力"真起作用，箱子必须不被强伺服**：
- 方案 i：`object_pd_override=false` + contact-guidance 脚手架（`init_pos_actuator_gain≈20`，`guidance_decay_ratio≈0.85`，`residual_gain_ratio=0` → 最后一次 CEM 迭代增益清零，`run_mjwp.py:1178-1189`），箱子最终由**接触+重力+外力**驱动；
- 方案 ii：真 freejoint 不驱动物体。
- **绝不**保留 `object_pd_override=true`（kp=2000）。

---

## 3. 两条必须遵守的原则

### 原则 1：不能「固定 U 只改力」，必须重优化
箱子运动 = 净力矩（机器人接触 + 重力 + 另一侧力）的积分。改了另一侧力 → 净力矩变 → 箱子偏离原轨迹 → 机器人针对**旧轨迹**解出的抓握**打滑 / 箱子漂走 / 掉箱**。
论文 §2.5 做法：**加外力 → 重跑优化器**让机器人适应新动力学。本仓对应 = 设 `perturb_force` 后跑 sampling CEM（`sampling.py:328` `make_optimize_fn`/`optimize_once`），从已验证轨迹 **warm-start** 重解 U。
⇒ **增广产物是"新力下重优化的轨迹"，不是回放旧 U。** 回放旧 U 只能当快速的不可行性探针。

### 原则 2：一次 run 出不了 N 个变体
- CEM 的 batch 轴（`num_samples`，如 2048）是**单一目标下的采样群体**，每轮归约成 1 个 elite/mean（`sampling.py:428-478`），读不出 N 条独立可行轨迹。
- DR 轴（`env_params`，`run_mjwp.py:1193-1204` 构造，`sampling.py:363-403` 消费）是 **worst-case 鲁棒**：`min_rew = torch.minimum(...)`（`sampling.py:371,403`）。即使把不同力塞进 DR group，优化器也只为**最差的力**优化，不会每个力出一条。
- 且 `perturb_force` 是 config 直读的全局标量，不在 env_param。
⇒ **N 个变体 = 跑 N 次**（每次一个力值）。

---

## 4. 所有增广方案（按改动从小到大排序）

标注 ★ = 你的核心想法（改另一侧的力）。所有 B/C/D 受 §0 共享前提约束（servo 关 + 重优化）。

### Group A — 零代码改动（纯 config/CLI + 重跑），多样性最弱
| # | 方案 | 怎么做 | 多样性 | 备注 |
|---|---|---|---|---|
| A1 | 随机种子 | `seed=0,1,2,…` 重跑 CEM | 同动力学下采样噪声变体，差异很小 | 几乎免费，本质是 stochastic replicate，非物理增广 |
| A2 | 物体初始 xy 偏移 | 已有 `xy_offset_range`，设单值重跑 | 起始位置微扰 | ±5mm 量级，偏小 |
| A3 | 接触 margin | `pair_margin_range` | 接触松紧 | 偏鲁棒性，非行为多样性 |

A 档**不强制**共享前提（servo 开也能产生变体），但也因此改不了箱子动力学，**与"另一侧力"核心想法无关**。

### Group B — 不动核心代码，只重生成场景 XML（`generate_xml.py` 换参数 + 重跑）
| # | 方案 | 怎么做 | 多样性 | 备注 |
|---|---|---|---|---|
| B1 | 物体质量/密度 | `object_density=`（`generate_xml.py:117`）重跑 | 轻/重箱 → 机器人改发力、身姿 | 论文式变化廉价版，**需 servo 关** |
| B2 | 物体摩擦 | `object_frictionloss`/`friction_scale`（`generate_xml.py:120-121`） | 抓握/打滑动力学 | 同上需 servo 关 |
| B3 | 物体尺度 | mesh scale | 可达/抓取点变 | 较大：可能需重跑 `detect_contact.py`+`ik.py`，逼近"几何增广"，超出纯物理 |

B 档改的是**物体本身参数**，不是"另一侧的力"，属动力学增广旁支。

### Group C — 小的隔离核心补丁（★ 核心想法：改另一侧的力）
| # | 方案 | 改动 | 多样性 | 备注 |
|---|---|---|---|---|
| C1 | ★ 恒定外力作用于箱子 | **必改**：`apply_perturbation` 加对 body `"object"` 的解析（现只认 `right/left_object`，对 CORE4D no-op）。然后 `perturb_force=f` + servo 关 + 重优化 | 模拟"另一侧分担不同力"→ 机器人前倾/减速/改发力的可行变体 | **这就是你要的那条**；一行级 body 查找补丁，隔离、可逆、git 追踪 |
| C2 | 方向性 3 维力 | 在 C1 上加 `perturb_force_vec: list[float]`（现标量被广播成 `[f,f,f]` 无方向） | 区分"抬升分量"vs"水平推/拉" | 比 C1 多几行；人通常主要分担**竖直**重力，方向语义有意义 |

C1/C2 是论文 §2.5 physics variation 的正解，也是"不建模 partner、只改另一侧力"的最忠实实现。
力扫法建议按箱重比例：`m·g` 的 0% / 25% / 50% / 75% 向上 + 几个水平值。

### Group D — 较大核心改动（先不碰，列全为完整性）
| # | 方案 | 改动 | 为什么大 |
|---|---|---|---|
| D1 | 时变力曲线 | 力随 episode 阶段 ramp/变化 | 需引入力的时间调度，改 step_env |
| D2 | 一次 run 出 N 变体 | `perturb_force` 提升为 per-env 张量，且把 CEM 的 worst-case `min_rew` 归约改成"每个力组独立 elite" | 重构 CEM 归约循环，风险高 |
| D3 | 状态/相位反馈力 | 力 = f(箱子状态) | 最复杂，接近重建 partner |

---

## 5. 落地建议（顺序）

1. **先验证"力能起作用"这条物理链路** —— 所有 C/D 的地基。最小动作：打 **C1 的 body-name 补丁** + 关 servo + 选 1 个力值，对 box021 一条已验证轨迹 warm-start 重跑 CEM，看箱子是否仍可行跟踪 + 机器人不掉箱。**这一步判定"改另一侧力增广"是否成立**（当前最大未验证风险点）。
2. 通了之后，**C1 标量力扫 4~5 个值**出第一批变体；按实验规范评：物体全程跟 ref 容差内（mean+std+worst，禁 cherry-pick）、SDF 无穿透（用 SDF 不用 contact 数 + CEM safety gate `sampling.py:199-217`）、机器人不摔不滑、A/B 视频对比。
3. 要"抬升 vs 水平"语义再上 **C2**。
4. **B1（质量）**作为正交第二增广轴，几乎零核心代码，可与 C 组合扩大覆盖。
5. D 组除非 C 验证后需要规模化（一次出 N 变体）再考虑。

---

## 6. 关键文件索引（file:line）
- 外力机制：`spider/simulators/mjwp.py:2318-2334`（apply_perturbation）、`:3269`（调用）、`:2434`（_apply_partner_force，排除项）
- 物体伺服：`spider/config.py:153-156`（object_pd_override/kp）、`mjwp.py:2337-2376`（_apply_object_pd_override）、`config.py:773-777`（object 维噪声清零）、`run_mjwp.py:1178-1189`（增益衰减到 0）、`:1287-1292`（object ctrl 重置为 ref）
- 力配置：`spider/config.py:74-75`（perturb_force/torque）
- 优化器：`spider/optimizers/sampling.py:328`（optimize_fn）、`:357-403`（worst-case DR 归约）、`:199-217`（CEM safety gate）
- DR group 构造：`examples/run_mjwp.py:1137-1205`
- warm-start：`examples/run_mjwp.py:485-535`（`warmstart_qpos_path`）
- 物体 XML 参数：`spider/preprocess/generate_xml.py:117`（density）、`:120-121`（frictionloss/friction_scale）
- 6-actuator 物体：`spider/simulators/hdmi.py:448-516`

---

## 7. 未验证风险（明确标注）
- 在较大另一侧力下，warm-start CEM 能否收敛到 box021 的可行搬运——**经验问题，需打完阻碍 1 补丁后实跑**（§5 步骤 1）。
- 论文 §2.5 在代码里无对应实现（`docs/workflows/workflow-mjwp.md:267` "Augmentation Methods" 仅是占位），"perturb_force = physics variation + 重优化"是从 `xfrc_applied` 机制 + CEM 推断，与论文框架一致，但具体力扫范围/调度论文未规定，需经验确定。
