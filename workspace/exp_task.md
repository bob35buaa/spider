# 后续实验计划

## 背景
之前的实验workspace/core4d的E077—E081 初步验证了现有动力学重定向方法在core4d双人协作数据集上的有效性。

## 一些想法
1. 目前物体的状态不太清楚是怎么控制的，物体轨迹好像是gt吗？如果是free joint呢（我理解全靠机器人的力这样，我不清楚现在的控制方法），这套方案还能work吗，你需要先确认下

2. 双人重定向：core4d的一些任务（例如box025）物体很大，单人其实很难真正搬起来的，所以maybe需要双人重定向，另外一端用模拟的方式（虚拟力？另一个机器人重定向的力？以及其他可能的方案）
> 需要注意的是,我们后面要RL训练，训的是单人的策略，但实际sim2real的时候，另一端是真实人类在搬运（在给物体施加力）

3. 算法层面的优化
相关的paper在@paper下面,可以参考，使用 $deep-reading-analyst 这个skill阅读论文
- SPIDER（本项目）@paper/Pan 等 - 2026 - SPIDER Scalable Physics-Informed Dexterous Retargeting.pdf
- DynaRetarget: @paper/Dhedin 等 - 2026 - DynaRetarget Dynamically-Feasible Retargeting using Sampling-Based Trajectory Optimization.pdf 
> 相关的材料在@paper/DynaRetarget_zh.md，并且之前的一些实验也展开过将DynaRetarget的算法迁移到本项目的探索，但当时由于一些数据和算法的bug，不确定是否有效
- 双人重定向论文: @paper/Liu 等 - 2025 - It Takes Two Learning Interactive Whole-Body Control Between Humanoid Robots.pdf

## 实验说明
1. 因为这次的实验更偏向探索型的，所以需要先基于已有的实验结果，代码和论文，进行深入分析，思考和广泛的探索，然后进行有依据的头脑风暴，列出一些可以实践的实验，然后再对每个实验用 $experiment-planning-zh 这个skill 具体展开（先plan后执行）。
2. 合理使用subagent，且subagent的模型和主agent一样
3. 实验基线应该以E081为基线，验收和评测需要对齐E081的eval + 可视化分析（需要subagent来完成，skill参考 $video-frames），实验work == 大部分指标和可视化效果>=baseline，或者有其他方面baseline不可替代的功能
4. git管理: 我们把现在的分支[feat/dual-robot-retarget]当做baseline，需要先把现在的分支合并到main分支，然后后面的实验按照大方向 新开分支来完成，里面的每个小实验都要commit&push
5. 文件管理: 新分支需要对应的开辟新的工作区在workspace下面，文件组织参考@workspace/core4d。所有的探索思考实验都要记录到本地的工作区下面。