# 适配新数据集

## 背景
我们初步完成了core4d数据集 从原数动捕数据到重定向 的链路，现在要适配到新的数据集MMHOI


## 数据
MMHOI数据集：
- 论文：@paper/Kogashi 等 - 2025 - MMHOI Modeling Complex 3D Multi-Human Multi-Object Interactions.pdf
- 本地数据地址：@/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/MMHOI
我们只关心里面的双人协作数据，（Collaborative work）

## 输出和注意事项
1. 开启新的工作区@workspace/MMHOI/ （全部实验相关的都要写到这里,参考 $experiment-planning-zh）
2. 开启新的分支，继承 experiment/E161-surface-release-ablation
3. 统计实际的协作数据信息，包括动作序列，数量，物体种类，输出一个数据统计文档到@workspace/MMHOI/data_stat.md
4. 调研链路适配管线 （参考 $data-construction-v3-zh）
5. 输出一个全局的适配方案，包括工作区如何构建，整体分几个阶段，每个阶段要做什么，要验证什么事情，写清楚,落成一个文档
