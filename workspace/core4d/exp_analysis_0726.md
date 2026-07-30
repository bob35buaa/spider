# 实验诊断：

## 背景
从E168开始到E178，我们使用E167A+PRG 批量对core4d数据的双人协作的序列做了整个数据管线的处理和spider重定向，但是发现整体的可用率不高（按照我们的门过滤）
并且也都跑了box类的下游RL验证，现在需要分析下，为什么可用率不高，有哪些可能性，可以从哪些角度进行改进？

## 下游RL验证结果
1. box021：
下游取的
| 1 | `box021_20231011_034_p1` | E170 | P00 accepted |
| 2 | `box021_20231011_034_p2` | E170 | P00 accepted |
| 3 | `box021_20231011_036_p1` | E170 | P00 accepted |
| 4 | `box021_20231011_036_p2` | E170 | P00 accepted |
| 5 | `box021_20231011_037_p2` | E170 | P00 accepted |
| 6 | `box021_20231011_038_p1` | E170 | P00 accepted |
| 7 | `box021_20231020_020_p1` | E170 | P00 accepted |
| 8 | `box021_20231020_022_p1` | E170 | P00 accepted |
| 9 | `box021_20231020_023_p1` | E170 | P00 accepted |
| 10 | `box021_20231018_029_p2`（runtime alias `box021_029_p2`） | E167/E167A | `USER_APPROVED_BRIDGE_OVERRIDE` |
| 11 | `box021_20231011_035_p1`（runtime alias `box021_035_p1`） | E167/E167A | `USER_APPROVED_BRIDGE_OVERRIDE` |
这11个case，
实验结果表明， 这11case RL都能训出来

2. box001：
数据用的@workspace/core4d/results/E173/s6_downstream/rl_export/box001_user_approved，
"box001_20231020_014_p2"这个case RL训不出来，其他都可以
3. box024：
@workspace/core4d/results/E173/s6_downstream/rl_export/box024_user_approved
box024_20231011_028_p2这个case RL训不出来
RL rollout视频：@/home/ubuntu/Workspace/Loco-Manipulation/SUGAR-private-worktrees/R018/outputs/experiments/R018_core4d_box021_multimotion_student_tracker/R018-7_cross_object_multirefiner/videos/REF_CROSS4_17CASE_DDP_30K/box024_20231011_028_p2/partner_on/ckpt_28000/seed_42/policy_rollout.mp4
4. box004：
@workspace/core4d/results/E172/s6_downstream/rl_export/box004_user_approved
box004_20231003_2_082_p1和box004_20231003_2_082_p2训不出来，这两个case是前面都没问题，最后时刻摔倒
RL视频：
- @/home/ubuntu/Workspace/Loco-Manipulation/SUGAR-private-worktrees/R018/outputs/experiments/R018_core4d_box021_multimotion_student_tracker/R018-7_cross_object_multirefiner/videos/REF_CROSS4_17CASE_DDP_30K/box004_20231003_2_082_p1/partner_on/ckpt_28000/seed_42/policy_rollout.mp4
- @/home/ubuntu/Workspace/Loco-Manipulation/SUGAR-private-worktrees/R018/outputs/experiments/R018_core4d_box021_multimotion_student_tracker/R018-7_cross_object_multirefiner/videos/REF_CROSS4_17CASE_DDP_30K/box004_20231003_2_082_p2/partner_on/ckpt_28000/seed_42/policy_rollout.mp4

## 需要做的事情
需要结合spider的指标，我的人工审核意见，和下游RL的表现，分析原因
几个角度吧
- 原始数据问题
- 上游的OmniRetarget算法的问题：OmniRetarget就不好，没有还原真实的动捕的动作，或者抖动严重（人和物体都要考虑）
- spider算法问题

## 输出
输出一份诊断分析报告到@workspace/core4d/analysis/xxx.md
