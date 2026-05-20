# 全面的评估

## 背景
E019基本构建了一个全面的评测：
@workspace/core4d_collab_retarget/log/20a_E019_unified_eval_framework_results.md
@workspace/core4d_collab_retarget/log/20b_OmniRetarget_13case_results.md
现在要基于这套评测体系，全面的评测我们之前的一些实验

## 需要评测的实验对象
1. 运动学结果：OmniRetarget
2. spider的动力学结果
    - E081: 作为动力学重定向的baseline（case不足需要重新跑）
    - E018b: 动力学重定向 free-joint 结果
    - E022-E025: 不同类型（方向）的优化（可以by-case的挑最好的结果）

## 评测的case
当前case最大的并集是13case
有些case是数据问题： desk021_p1， box021_p1， box021_p2
有些case可能是数据问题：bucket001_p1

所以需要给出两版结果：
1. 【P0高优】：9case结果：排除4个case（desk021_p1， box021_p1， box021_p2，bucket001_p1）
2. 【P1】：13case full

## Pre-Eval 需要做的事情
1. 确认当前spider项目的kinematic轨迹数据来源
2. 确认OmniRetarget的kin 28cm contact preservation 28cm阈值的合理性，这个28cm的阈值论文里面没有提，检查下来源，因为直觉上来讲，他是衡量wrist 到object center的距离； 或者多搞几个阈值看看

## 注意事项
1. 可以启用subagent
2. 部分实验缺结果 可以本地1卡+远程服务器2卡并行跑，远程参考@.codex/skills/experiment-planning-zh/remote-execution.md，同时本地可以进行主线任务
3. 按照 $experiment-planning-zh 这个skill来展开哈，先整体规划，然后再执行，本次当做一个完整的验证实验 叫E026吧
4. 评估完成后要对比下可视化视频，看看指标和视觉 匹不匹配
