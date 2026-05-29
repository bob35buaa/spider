# 实验诊断v2（after E088-E097）

在第一次实验诊断@workspace/exp_diagnostic之后，我们做了一系列的实验，E088-E097，现在需要你来重新审查和评估以及进行后续方向的探索。
实验相关日志@workspace/core4d/log
新的data_contrustion：@/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/workspace/v3/data_construction_v2

1. 后续的E088-E097
- 是否真正的解决或者解释了诊断报告里面的点@workspace/exp_diagnostic/diagnostic_report.md，有没有冲突和反直觉的地方？有没有没有验证的地方？
- face（接触面）的选择or诊断是否正确，有没有bug， 可视化是否正确，符合直觉
2. 现有case失败的原因有很多种，除开动捕数据本身精度问题的case，有些case是OmniRetarget重定向这一步的问题，
如果我们选择不绕开硬啃这块重定向的硬骨头，该从哪些角度入手，有什么思路？

## 注意和参考
1. E088-E097是在另外的机器上展开的，如果遇到找不到的路径，先试一下路径映射，相对路径应该是一样的
spider就是本项目，holosoma在@/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma
2. 原始数据路径缺失需要向我汇报
3. 需要用下面的python运行：.venv/bin/python, 需要mujoco的时候： MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python
4. OmniRetarget的v1重定向@/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/workspace/v1/README.md，
原始方法复现pipeline@/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/workspace/pipeline

## 输出
这次的工作区在@workspace/exp_diagnostic_v2
