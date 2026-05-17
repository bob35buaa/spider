# E005 Paper Note: Du et al. 2025 COLA 的虚拟协作者与虚拟力分析

日期：2026-05-17

论文：Yushi Du 等，*Learning Human-Humanoid Coordination for Collaborative Object Carrying*，arXiv:2510.14293v1，2025-10-16。  
本地文件：`/home/ubuntu/Workspace/holosoma/paper/Du 等 - 2025 - Learning Human-Humanoid Coordination for Collaborative Object Carrying.pdf`

## 阅读目的

聚焦论文里“仿真人类协作者 / 虚拟力 / 外力交互”的实现方式，判断它是否能用于当前 `core4d_collab_retarget` 的 true-freejoint 物体重定向场景，尤其是 E004 COM virtual partner force 和 E005 support-site force 的后续改进方向。

## 1. SCQA

**Situation**：协作搬运任务里，人类和人形机器人共同搬一个物体。机器人不能只跟踪自身动作，还要对人类施加在物体上的运动意图和交互力做出顺应。

**Complication**：如果训练/优化时没有显式建模“另一端的人”，机器人会把物体当成单人操控对象，容易出现速度不同步、高度不一致、物体姿态不稳、或者人类端负担过大。

**Question**：怎样在仿真里制造一个可控、可度量、能传递动力学影响的“人类协作者”，让机器人学会/优化出协作搬运行为？

**Answer**：COLA 在 closed-loop training environment 里加入一个 `support body` 来模拟人类承托端。这个 support body 位于物体远离机器人手的一端，通过 6-DoF 连接把自身运动传给物体，并用线速度、角速度、高度命令和 PD 力/力矩控制它。机器人在这种三体闭环中学习 residual policy；论文还把 support body 与物体之间的水平交互力作为 human effort 指标/惩罚项。

## 2. 这篇论文里的“虚拟力”到底是什么

论文里有三类容易混在一起的力，需要分开看。

### 2.1 虚拟协作者：support body，而不是 COM 外力

COLA 的核心仿真对象是：

- humanoid：G1 机器人；
- object：被搬运物体；
- supporting base body / support body：模拟人类搬运者的虚拟刚体。

support body 放在物体另一端，也就是人类应当扶/托的位置。物体和 support body 通过 6-DoF joint 连接，连接里的摩擦、阻尼和关节限制让 support body 的运动影响物体。也就是说，物体不是被一个全局 COM 弹簧直接拖着走，而是通过“另一端支撑体”获得力和力矩。

训练时，论文会采样一个 goal command `G`，再采样 `v_applied` 并施加到 support body。`v_applied` 的幅值来自 `(0, G)` 范围并加噪，更新频率是 goal command 的两倍。对角速度和高度，论文使用 PD 控制在 support body 上施加力矩或竖直力。

这更像“受控虚拟人端点 + 物体连接动力学”，不是“在物体 COM 上直接加一个 ref tracking force”。

### 2.2 交互力指标：support-object horizontal force

论文的 reward/eval 里有一个 `F_support-object`，定义为 support body 与 object 之间的水平交互力。它有两层作用：

- reward 里有 force penalty，权重很小，用于抑制过大的交互力；
- eval 里用 average external force 衡量人类搬运负担。

表 III 的表头把 Avg. E.F. 标为越低越好，且 COLA-L 数值低于 baseline；正文有一句把更高 Avg. E.F. 解释成更强 compliance，这和表头方向不一致。按指标定义“人类 effort”理解，应该优先按低为好解释。

### 2.3 机器人末端外力：鲁棒性/顺应性训练与测试

论文还在 WBC 训练时给机器人 end-effector 加外力，增强 payload/force adaptation。Fig. 5 也测试了给 palm 施加逐渐增大的水平外力、给末端施加竖直外力时机器人的响应。

这部分不是 support body 虚拟协作者本身，而是让机器人学会对手/臂处的外力敏感。论文观察到：

- 手/臂受到外力时，机器人倾向于顺着人类引导运动；
- 躯干/腿受到外力时，机器人更倾向于保持稳定；
- 低于约 15N 的水平外力更像姿态稳定线索，高于阈值后才触发明显跟随。

## 3. 5W2H

| 问题 | 论文答案 | 对我们的含义 |
|------|----------|--------------|
| What | 用 support body 模拟人类搬运端，并通过物体连接传递动力学 | E004 的 COM 力不是等价实现；E005 的 support-site wrench 更接近但仍是简化 |
| Why | 协作搬运的核心是速度、高度、姿态和人类负担的闭环平衡 | 只优化 object pose tracking 会掩盖“另一端谁在承担力” |
| Who | 人类端由 support body 代理，机器人端由 humanoid policy 控制 | 我们可以把 CORE4D 的另一个人抽象成 proxy，而不必先训练双机器人 |
| When | residual teacher policy 训练阶段使用，真实部署时 student 只用 proprioception | 我们是 MJWarp/CEM 重定向，不是 PPO；更适合作为 optimizer 里的物理 proxy |
| Where | 物体远离机器人手的一端，按物体类型预设 held end | CORE4D 需要从物体几何和 person/contact side 推断 support site |
| How | support body 接收线速度/角速度/高度命令，PD 施力/力矩，物体通过连接被影响 | MuJoCo 里可用 dynamic support body + soft equality/contact/constraint 近似 |
| How much | 命令范围约为线速度 `(-0.6, 1.0)m/s`、角速度 `(-0.8, 0.8)rad/s`、高度 `(0.5, 0.85)m` | 我们可从小尺度 sweep 开始，不必一开始复制全部 command 随机化 |

## 4. 和 E004/E005 的关系

### E004: COM virtual partner force

E004 的设计是对物体 COM 施加外部力，通常可理解为：

- 重力补偿比例项；
- `ref_COM - current_COM` 的位置 PD；
- 对 COM 速度的阻尼。

它的优点是实现简单，能快速验证“freejoint 物体是否需要外部协作者支撑”。但它和 COLA 差别很大：

- 没有 support side，无法表达人类在物体远端支撑；
- 力施加在 COM，缺少由力臂产生的自然力矩；
- 它容易变成“物体轨迹 actuator”，而不是协作者动力学；
- 不能直接得到人类端 interaction force/human effort。

因此，E004 可作为必要性诊断，不应作为最终协作模型。

### E005: support-site external wrench

E005 把力施加到物体局部 support site，并把 `r x F` 转成等效力矩。这已经比 E004 更接近 COLA，因为它至少把力作用点移到了“另一端”。

但 E005 仍然不是 COLA 的 support body：

- 没有独立 support body 的质量、惯性和状态；
- 没有 support body 与 object 之间的软连接/摩擦/阻尼/限制；
- force 是直接施加到 object 上的，不经过可观测的 human proxy；
- interaction force 指标只能近似为所施加的外力，不能测真实 constraint/contact load。

所以 E005 更适合当作 lightweight approximation。如果 E005 有改善但不稳定，下一步应该升级为 support-body proxy，而不是继续只调 COM/support-site force gain。

## 5. Critical Thinking: 能否直接照搬

**可以借鉴的部分**

- 用物体另一端的 support proxy 代表真实人类，而不是让单个 G1 承担全部物体动力学。
- 把 object end-height difference 作为核心指标，比只看 bottom/floor/contact 更能反映协作水平。
- 把 partner-object horizontal force 作为 human effort 指标，用来避免虚拟协作者“替机器人把活全干了”。
- 把 velocity/height/yaw command 分开建模，符合协作搬运里人类常见的引导方式。

**不能直接照搬的部分**

- COLA 是 PPO residual policy 训练；我们当前主线是 per-demo MJWarp/CEM 物理重定向。
- 论文环境中 support held end 是按物体类型预设；CORE4D 每个 demo 的人、物、接触侧可能不同，需要自动推断或按任务配置。
- 论文可以用 privileged object history 训练 teacher，再 distill 到 proprioception-only student；我们短期目标是生成可用轨迹，不是直接训练 COLA 式 policy。
- 如果 MuJoCo/MJWarp 对复杂 6-DoF 约束、contact proxy 或 equality 的梯度/性能支持不足，完整 support-body proxy 可能需要先做 CPU/MuJoCo smoke，再进 MJWarp sweep。

**证据强度判断**

论文对 closed-loop support body 的描述和指标体系很有参考价值；但其数值结果来自 RL 训练环境和真实机器人实验，不等价于 CEM retargeting 场景。对我们最可靠的借鉴是“系统结构”和“指标设计”，不是直接复用它的参数范围或阈值。

## 6. First-Principles 拆解

协作搬运的物理基本事实是：

1. 物体 freejoint 时，6D 运动由所有接触/外力/重力共同决定。
2. 大物体通常需要两端力和力矩平衡；单端机器人即便手接触正确，也未必能提供足够支撑。
3. 如果另一端的人在真实任务中提供支撑，仿真里完全移除 partner 会改变任务本身。
4. 如果直接把物体轨迹当作 actuator，会制造“看起来跟踪好、但力学解释不真实”的轨迹。

从这些事实重建我们的设计，合理路径应该是：

- 保持 object true freejoint，不给 object 加关节 actuator；
- 引入一个只作用在 partner side 的 proxy；
- proxy 自己被参考速度/高度/yaw 控制；
- object 通过接触或软约束从 proxy 获得力；
- 评估时同时看 object tracking、robot-object contact、floor contact、partner effort。

这正是 COLA support body 对 E004/E005 的启发。

## 7. Systems Thinking: 当前系统的关键反馈环

当前 E004/E005 方向可以看成四个子系统：

- optimizer：CEM/MJWarp 选择机器人动作；
- humanoid：G1 执行动作并与物体接触；
- object：true-freejoint 被重力、接触和外力驱动；
- partner proxy：外部协作者支撑/引导物体。

关键反馈环：

- **稳定环**：partner proxy 提供支撑 -> object 掉落减少 -> robot hand contact 更容易维持 -> object tracking 改善。
- **作弊环**：partner force 过强 -> object 即使机器人不接触也能跟踪 -> optimizer 学不到真实搬运动作 -> 下游 RL/真实迁移变差。
- **接触恢复环**：support site 有自然力矩 -> object 姿态更接近参考 -> robot hand 相对位姿更合理 -> 后续接触更稳定。
- **地板依赖环**：partner 支撑不足 -> object 接触地板/腿 -> 表面上位置稳定 -> 实际搬运失败。

最强杠杆不是调一个 force gain，而是改变系统结构：从 COM actuator 改成 partner-side support-body proxy，并加入 effort/contact 指标约束。

## 8. 对后续实验的建议

### E005 继续作为轻量验证

E005 已经在 support site 施加外力/力矩，建议继续跑完，重点看：

- support-site force 相比 COM force 是否减少 floor contact；
- 是否改善 `box025` 这种大物体的 lift/floor 指标；
- 是否出现“机器人不碰物体但物体被 proxy 拉着走”的作弊；
- support-site force/torque 的大小是否已经远超人类合理范围。

### E006 候选：COLA-style support-body proxy

如果 E005 结果显示 support side 有价值，下一步建议单独开 E006：

1. 在 freejoint scene 里增加一个 dynamic support body，初始位于 object opposite side。
2. support body 自身有 freejoint 或等效 6D 状态，不直接 actuator object。
3. support body 通过 soft equality / connect / weld-like constraint / contact pad 与 object support site 传力。
4. support body 用 PD 跟踪参考线速度、高度和 yaw，而不是直接跟踪 object COM position。
5. 记录 proxy-object constraint/contact force，作为 partner effort。
6. 机器人控制维度保持不变，object 仍保持 true-freejoint passive body。

最小验证任务可以先选：

- `box023_person2`：中等难度，便于和 E004/E005 对照；
- `box025_person2`：大物体，最能暴露单人化问题；
- 每个任务 2-3 个 support stiffness/damping 档位，先不要扩大到所有 case。

### 指标需要补齐

建议在 E006 前后补三个指标：

- `end_height_diff`：robot-held end 与 support-held end 的高度差；
- `partner_effort_mean/max`：support body 与 object 之间水平交互力；
- `robot_contact_share`：物体支撑中由 robot hand/arm 提供的比例，避免 partner proxy 全权代劳。

如果暂时拿不到真实 constraint force，可以先记录施加到 support body 的 PD wrench，或 E005 的 support-site equivalent wrench，作为 effort 近似。

## 9. 当前结论

这篇论文的虚拟力思路值得用于我们的场景，但不应把它简化成 E004 的 COM 外力。更准确的迁移方式是：

> 用一个位于物体另一端的 support-body proxy 模拟真实人类承托端，让它通过软约束或接触把速度、高度和 yaw 引导传给 true-freejoint object，同时用 proxy-object interaction force 约束 human effort。

E005 是通向这个方向的低成本近似；真正接近 COLA 的版本应作为后续 E006/E007 的结构性实验。
