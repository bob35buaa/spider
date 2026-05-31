## 背景
1. 实验现象：
- @workspace/core4d_collab_retarget/ 工作区的E028-E030几组实验，在box021的几个case上做了重定向，都失败了。日志参考@workspace/core4d_collab_retarget/log/29_E028_d003_box021_spider_dynamic_retarget_results.md，@workspace/core4d_collab_retarget/log/37_E030_d6_locked_3case_cem_results.md
- @@workspace/core4d/ 工作区的E082-E088的几组实验，在box021的3个case上做，也都失败了。日志参考@workspace/core4d/log/103_E082_d003_box021_e081_legobj_results.md - @workspace/core4d/log/110_E088_hard_safety_gate_lift_floor_results.md
这个现象很奇怪，因为同类型的任务box023是成功的，box025更大的箱子也成功了，但是box021这个中等尺寸的箱子失败
2. 诊断
@workspace/exp_diagnostic 根据实验现象做了全面的诊断，参考@workspace/exp_diagnostic/diagnostic_report.md

## 一些思考
1. 根据这份诊断,是否意味着上游OmniRetarget的重定向在box021这几个case上严重穿模? 因为这些case的来源是我改进的版本的OmniRetarget
重定向算法,可能是我的改进导致的穿模(参考@/home/ubuntu/Workspace/holosoma/workspace/v1/README.md) ,请你深入分析一下
2. 如果可能是我的改进导致这些case穿模严重，那需要按照原始的OmniRetarget重定向算法，进行重定向，然后在进行spider动力学重定向，对比一下就能确定上游算法的改进是否真的影响了下游spider。