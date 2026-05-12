# 下一步实验方向
背景@workspace/core4d/log/49_E040_dynamic_target_results.md，@workspace/core4d/log/50_E041_orientation_reward_results.md，@workspace/core4d/log/51_E042_wrist_freeze_results.md，@workspace/core4d/log/52_E043_original_ref_results.md
## 1. 运动学重定向算法 body tracking
1. 看看spider原始的body tracking，hmdi workflow中的body tracking算法的效果和我们改进的P4版本的效果对比
2. 同时可以研究下我们基于OmniRetarget改进的算法有没有提升空间
> 相关文档：算法实验日志@/home/ubuntu/Workspace/holosoma/workspace/v2/log/01_v2_m1_single_case_results.md，@/home/ubuntu/Workspace/holosoma/workspace/v2/log/02_v2_m2_batch_results.md
## 2. core4d数据集更多case的效果
- 原始数据集 @/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real
- 改进的OmniRetarget运动学重定向算法参考相关文档的算法实验日志
- 转为spider参考序列@spider/process_datasets/core4d.py

## 3. CEM算法优化
可以参考DynaRetarget文章@paper/Dhedin 等 - 2026 - DynaRetarget Dynamically-Feasible Retargeting using Sampling-Based Trajectory Optimization.pdf
可以用 /deep-reading-analyst 这个skill 细读论文，然后把论文总结到本地文件@paper/DynaRetarget.md
然后根据论文中能够帮助优化本算法的点，进行实现，然后实验验证
