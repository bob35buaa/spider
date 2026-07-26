# 适配新数据集

## 背景
我们初步完成了core4d数据集 从原始动捕数据到重定向 的链路，现在要探索下新数据集InterPose

## 数据
InterPose数据集：
- 论文：@paper/Zhang 等 - 2025 - InterPose Learning to Generate Human-Object Interactions from Large-Scale Web Videos.pdf
- 本地数据地址：@/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/InterPose
- 官方代码：@/home/ubuntu/Workspace/Human_related_projects/InterPose
- 官方数据构建代码：@/home/ubuntu/Workspace/Human_related_projects/InterPose-data-collection
- smplx：@/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx

【注意】 我们只关心里面的双人协作数据， 就是 人-物体-人 三元关系，

## 输出和注意事项
1. 开启新的工作区@workspace/InterPose/ （全部实验相关的都要写到这里,参考 $experiment-planning-zh）
2. 开启新的分支，继承 experiment/E161-surface-release-ablation 【注意不要继承MMHOI】
3. 统计实际的数据信息，包括动作序列，数量，物体种类，输出一个数据统计文档到@workspace/InterPose/data_stat.md
4. 如果有我们想要的数据类型，则需要输出一个全局的适配方案，适配到我们的spider数据构建管线（参考 $data-construction-v3-zh），包括工作区如何构建，整体分几个阶段，每个阶段要做什么，要验证什么事情，写清楚,落成一个文档。如果没有这样的数据，则调研 如何基于开源代码，从互联网视频 构建 人-物-人数据，并适配spider管线，也要写一个整体的计划和可行性分析
