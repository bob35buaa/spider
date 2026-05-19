# Task： 收尾

之前的实验大概能说明,我们的一些方法能够将原本的spider方法应用到人机协作动力学重定向数据集上,
且物体是free-joint. E014在box023_p2和box025_p2两个case上work, 然后E018拓展到了更多的case,证明方法有一定的泛化性. 后面是一些收尾工作。
具体而言，我的想法如下：

## 1. 全面的评测
1. 对齐OmniRetarget论文Table2的指标，以及运动学重定向指标
- 指标参考@/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/workspace/v1/scripts/eval_paper_metrics.py
- 运动学重定向相关文档: @/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/workspace/v1/README.md
- 运动学数据: @/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/workspace/v2/data/core4d_replace_batch

2. 对齐SPIDER论文的Table4的指标
Joint Err. ↓ Pos. Err. ↓ Ori. Err. ↓ Obj. Pos. Err. ↓ Obj. Ori. Err. ↓ 这些吧
- 论文@paper/Pan 等 - 2026 - SPIDER Scalable Physics-Informed Dexterous Retargeting.pdf（可以只读Table4相关的）
- 可能的参考@workspace/core4d/log/41_E035_local_frame_results.md

3. 其他的能有效验证的指标

【总体要求】：
1. 构建一套统一，通用，可复用的评测脚本
2. 输出 Markdown格式表格，xlsx表格（要有归类和对比）
3. 写一个论文标准的文档，介绍这些metrics是什么，公式怎么表达，反映了什么。写到@workspace/core4d_collab_retarget/docs/eval_metrics.md

## 2. 当前结果分析
基于全面的数值指标评测和仔细的可视化结果分析（可视化包括视频和一些运动曲线分析）
找出失败的case为什么失败，是数据质量问题还是算法问题
> 前期数据质量分析：@workspace/core4d/log/97_E076_contact_source_audit.md

## 3. 技术报告文档
总结@workspace/core4d(主要看E081有关的实验，就是E081及其继承之前的实验)和@workspace/core4d_collab_retarget,写一份详细的技术报告，要求：
- 背景是什么：我们为什么要做 人机协作的动力学重定向
- 我们怎么做的：我们采用了哪些方法做到的，突出我们做了哪些事情，有哪些技术贡献（首先这里明确下，我们是基于OmniRetarget项目做的运动学重定向，然后基于spider框架做的动力学重定向）
- 做的怎么样：结果
写到@workspace/core4d_collab_retarget/report

## 4. 导出Holosoma项目RL训训练可以使用的轨迹
参考@/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/workspace/pipeline/convert_data_format_mj_p3_for_rl.py还有之前的有关实验文档
