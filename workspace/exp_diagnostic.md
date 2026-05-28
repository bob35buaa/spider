# 实验诊断

## 背景
1. @workspace/core4d_collab_retarget/ 工作区的E028-E030几组实验，在box021的几个case上做了重定向，都失败了。日志参考@workspace/core4d_collab_retarget/log/29_E028_d003_box021_spider_dynamic_retarget_results.md，@workspace/core4d_collab_retarget/log/37_E030_d6_locked_3case_cem_results.md
2. @@workspace/core4d/ 工作区的E082-E088的几组实验，在box021的3个case上做，也都失败了。日志参考@workspace/core4d/log/103_E082_d003_box021_e081_legobj_results.md - @workspace/core4d/log/110_E088_hard_safety_gate_lift_floor_results.md
这个现象很奇怪，因为同类型的任务box023是成功的，box025更大的箱子也成功了，但是box021这个中等尺寸的箱子失败

## 任务
你要全方位的分析和诊断当前的实验的问题，包括数据，数据处理，代码，实验设置，结果分析等等
请你先理解整个项目背景，研究目标和实验进展，在此基础上展开地毯式调研，然后出一个诊断的plan，之后按照这个plan再执行

## 注意事项
需要用下面的python运行：.venv/bin/python
需要mujoco的时候： MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python
