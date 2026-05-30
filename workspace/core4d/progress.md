# E100 Progress — 2026-05-30 (DONE offline; CEM 推迟 E101)

## 状态：E100 C1+C3 全 PASS, C2 ready/CEM 推迟 E101，已 commit + push

## 完成
- [x] P1: plan/107_E100_fingertip_target_and_clean_ab_plan.md
- [x] P2: build_fingertip_aware_target.py + 单测 3/3 + 16/20 case NPZ
- [x] P3: target_gap_summary.tsv + 9 个 Tier 1 对比 PNG
- [x] P4: 4 个 yaml override + run_E100_remote.sh; smoke test 验证 target loading; CEM 实际触发推迟 E101
- [x] P5: log/124_E100 + EXPERIMENT_TRACKER + audit + commit + push

## Claims 验证
- C1 16/20 case + 单测 3/3 ✅
- C2 config + 远程脚本 + smoke 加载成功 ⚠️（CEM 实际跑推迟 E101 Phase 1）
- C3 守门反向 swap Δ=0cm 7/7 + DIFFER 1.5-5cm 9/9 ✅

## 关键发现
- DIFFER hand 全部 8/9 在 R，与 E099 audit 完全一致
- 守门 case face_changed_L 全 16/16 = False, swap Δ=0cm
- IK 过拟合 case (028_p2 R, box023_person1 R) target 退化为 palm_local (vote_face='')
- torch.compile/triton 环境报错与 fingertip target 无关, 是 spider 历史 commit 环境问题

## 下一步
- 启动 E101（Stage 3）: 排查 torch.compile/triton 环境 → 18029_p2 + 035_p2 typical full CEM 双卡
