# E001 Literature Synthesis: SPIDER / DynaRetarget / Harmanoid

日期：2026-05-17

## 目的

把三篇论文转成当前项目可执行实验假设。这里不做完整论文复述，只记录和 CORE4D 协作重定向直接相关的机制、风险和实验映射。

## 1. SPIDER: contact guidance 是“扩大采样可行域”，不是普通 reward

### SCQA

**Situation**: 人类演示只提供任务结构和接触意图，机器人需要在物理仿真中找到可执行控制。  
**Complication**: 同一个物体轨迹可能有多种接触模式；普通采样/annealing 可能收敛到错误接触。  
**Question**: 如何让 sampling-based optimizer 更容易采到“人类意图的接触模式”？  
**Answer**: 用 curriculum-style virtual contact guidance 早期“粘住”预期 hand-object 相对位姿，再逐步放松。

### 对本项目的关键点

- SPIDER 明确区分 soft contact reward 与 virtual contact guidance：后者通过相对接触约束扩大 basin of attraction，而不是只在 cost 里给一个距离项。
- contact guidance 应维护 robot contact point 与 object contact point 的相对位置，并只在参考接触可靠时启用。
- 论文还强调对 imperfect reference contact 的过滤：短接触或 contact point 漂移过大的片段应禁用 guidance。
- 对下游 RL，SPIDER 观点是 retargeting 应提供 feedforward nominal control，RL 学 residual feedback；如果 retarget 轨迹本身不可行，RL 会被迫靠复杂 curriculum 补洞。

### 可执行映射

1. E002 候选：把 E081 的 3cm mask 从 binary gate 升级为“稳定接触段筛选 + 相对 hand-object target drift 过滤”，先不加新 reward。
2. E003 候选：恢复/重写 virtual contact guidance，使它约束相对 hand-object contact frame，而不是只优化 min SDF 或手腕距离。
3. 验收重点：不能只看 contact%，必须看接触模式是否仍是手/箱主导，腿/地板是否变成隐式支撑。

## 2. DynaRetarget: 长 horizon 动态修正解决短视 MPC

### SCQA

**Situation**: 当前 MJWP/SPIDER 是 receding-horizon sampling，能快速修正局部动作。  
**Complication**: CORE4D 失败常发生在“早期接触/支撑选择影响后期 lift/place”的长 horizon 耦合；短 horizon 一旦掉箱或选错接触，后续难恢复。  
**Question**: 是否需要完整轨迹级别的动态 refinement，而不是逐帧 MPC？  
**Answer**: DynaRetarget 的 SBTO 逐步增长优化 horizon，早期控制变量会在更长未来代价下反复精修。

### 对本项目的关键点

- DynaRetarget 对 SBMPC 的批评与 E075-E081 现象吻合：短 horizon myopic、早期错误不可恢复、输出抖动。
- 其 cost 里 object position 权重最高，torso/foot/task-space 权重也很高，且显式包含 object velocity 和 collision terms。
- 它的 SBTO 成本约为 SPIDER 的 3.3 倍，但成功率和平滑度明显更好。
- 论文的失败模式仍是参考质量差，特别是 hand-object contact 突变或 object orientation 突变。

### 可执行映射

1. E004 候选：不一次性移植完整 SBTO，先做 “fixed open-loop segment refinement” micro-SBTO：选 `box023_p2` 或 `box025_p2` case-window 的 1.5-3.0s，固定初始状态，从短到长优化 knot 序列。
2. E005 候选：在 MJWP cost/eval 中加入 object velocity、foot position/outlier penalty、robot-object collision count 的诊断版本，先验证指标和失败模式解释力。
3. 风险：早期 E047 的 SBTO port 失败过，原因包括 exp reward 不兼容、参数过紧、闭环/开环假设不一致；新实验必须从 1 个短 segment 和 E081 baseline metric 开始。

## 3. Harmanoid / It Takes Two: 协作不是两个单人问题相加

### SCQA

**Situation**: CORE4D 是双人协作，人和物体的相对位置/接触语义由两个人共同决定。  
**Complication**: 独立单人重定向会忽略 partner dynamics，造成接触错位、穿透、不自然距离；这和 box025 大箱单人 partial carry 的瓶颈一致。  
**Question**: 另一端真实人类或虚拟 partner 应该如何进入重定向/RL？  
**Answer**: 保留交互接触和相对 root/upper-body geometry；controller 接收 partner state / contact mask，并用 interaction reward + contact force reward + curriculum 逐步加权。

### 对本项目的关键点

- contact-aware retargeting 先从 human-human mesh collision 提取接触，再映射到 robot links。
- 相对 root pose 是可优化变量，用于减少 partner 间穿透/过远，而不是固定各自独立的单人 retarget。
- controller 不只看自身 proprioception，还看 partner state summary 与双方 reference contact mask。
- contact reward 区分 expected contact 和 unexpected contact，且按 measured force 范围奖励/惩罚。
- curriculum 先保证 tracking，再逐步提高 interaction/contact 权重，避免早期冲突。

### 可执行映射

1. E006 候选：先做不训练的 partner-force/partner-mocap 诊断，把 CORE4D person1/person2 的另一端手/身体作为外部可观测/可施力对象，检查能否解释 box025_p2 lift/floor-contact 缺口。
2. E007 候选：构造 “human-partner proxy” 而非双机器人：另一端仅提供参考接触点、接触法向、限幅外力/弹簧，不让它参与主策略控制。
3. E008 候选：如果进入 RL，policy observation 应包含 partner/object relative state 和 contact mask；训练目标仍是单人策略，因为 sim2real 另一端是真实人类。

## Cross-Source Synthesis

| 维度 | SPIDER | DynaRetarget | Harmanoid | 对本项目的判断 |
|------|--------|--------------|-----------|----------------|
| 接触 | virtual guidance 扩大采样可行域 | 依赖动态 refinement 后的真实接触 | human-human contact 映射到 robot links | 先修 contact label/guidance 质量，再谈 reward 权重 |
| horizon | receding horizon sampling | 渐进式长 horizon SBTO | RL episode + curriculum | E081 后的 lift/place 是长 horizon 问题，短窗口指标会误导 |
| object | object motion 是主要成功指标 | object pos/rot/vel 是核心 cost | 交互几何间接决定 object 可搬性 | 新 eval 必须把 object floor-contact/lift 与 hand/leg support 分开 |
| 协作 | 未显式双人 | 以单 humanoid-object 为主 | 明确双 humanoid interaction | CORE4D 大箱不应强行单人化；需要 partner proxy 或 interaction model |
| 下游 RL | nominal control + residual feedback | 动态一致轨迹降低 RL 难度 | partner-aware observation/reward | 生成给 RL 的轨迹必须解释另一端真实人类，而不是双机器人闭环假象 |

## First-Principles Conclusion

CORE4D box025 类任务的核心不是“机器人手离箱子不够近”，而是“单 G1 的接触力闭环不具备双人协作任务所需的力/力矩平衡”。E081 已把腿/箱穿模这个物理漏洞压下去，剩下的 lift/floor-contact 缺口需要在三条路线中选择：

1. 更可信的 hand-object virtual contact guidance，保证单人可行 case 不掉接触。
2. 长 horizon segment refinement，避免早期支撑/接触选择导致后期不可恢复。
3. partner proxy / interaction-aware model，承认大物体另一端由真实人类承担部分力和约束。

E002 应优先做低风险审计/诊断型实验，而不是直接大改 optimizer：先明确 object freejoint 控制口径和 E081 baseline checklist，再选最小可验证的 partner proxy 或 contact-guidance 修复。
