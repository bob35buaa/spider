# 实验诊断v3-20260604（after E105）

在第二次实验诊断@workspace/exp_diagnostic_v2之后，我们做了一系列的实验，从E098-E143，主要就是修复了之前某些box的template错误问题，构建了数据管线，批量跑了spider重定向和全面评测，以及少量的下游RL验证。
实验相关日志@workspace/core4d/log
新的data_contrustion：@workspace/core4d/scripts/data_construction_v3

你现在需要先理解整个项目，研究方向，研究目的，实验进展， 然后重新审查和评估以及进行后续方向的探索。

@workspace/core4d/results/E143/raw_mask_ref_fk_24case_omniretarget_comparison/E143_raw_mask_ref_fk_24case_omniretarget_comparison-anno.xlsx, 这个表格的spider成功逐case这个sheet的视频我都看了, 然后在备注这一列,标注了我对case的观察和判断,最下面还有颜色标识对应的解释, RL是否成功这列是已经跑完的RL实验. 你需要深入分析一下, 当前spider的接触
落后于omniretarget的原因, 是动捕数据本身质量差（不是我们需要的case,物体或者human抖动不合规）, 还是omniretarget算法本身就不好从而导致spider
更不好（或者也有的情况是omniretarget不好但是spider救回来一点了）, 还是spider算法的问题. 分析的时候注重可视化

## 输出
这次的工作区在@workspace/exp_diagnostic_v3，
代码写到@workspace/exp_diagnostic_v3/scripts
可视化等输出到@@workspace/exp_diagnostic_v3/results

## 注意
1. 不要改当前spider的算法代码，只是审查和进行下一步规划
2. 合理使用subagents，避免污染主线上下文
3. 实验是在另外的机器上展开的，如果遇到找不到的路径，先试一下路径映射，相对路径应该是一样的, spider就是本项目，holosoma在@/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma
4. 需要用下面的python运行：.venv/bin/python, 需要mujoco的时候： MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python
