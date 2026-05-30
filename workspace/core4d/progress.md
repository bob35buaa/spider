# E099 Progress — 2026-05-30 (DONE)

## 状态：E099 全部完成，Claims C1-C4 全 PASS，已 commit + push

## 完成
- [x] P1: plan/106_E099_contact_semantics_pipeline_plan.md
- [x] P2: 摸清 STAGE A path（不重跑 OmniRetarget，直接读 raw CORE4D；最小变更）
- [x] P3: fingertip_face_vote.py + 8/8 单测 + 17/20 case 跑通
- [x] P4: quat_identity_audit.py + 17/17 case → all disable_world_up=True
- [x] P5: render_raw_contact_3d.py + 17 case × {PNG+MP4} = 32 文件 + subagent 5/5 视觉签收
- [x] P6: log/123_E099 + EXPERIMENT_TRACKER + audit 报告 + commit + push

## Claims 验证
- C1 fingertip helper + 17/20 case + 8/8 单测 ✅
- C2 quat 普查 17/17 disable_world_up ✅（超 v2 预测）
- C3 17/20 case turntable mp4 + subagent 5/5 ✅
- C4 historical_case_face_audit.md + 9/33 hand DIFFER + Tier 排序 ✅

## 关键新发现
- B6 强证据：palm vote ≠ fingertip vote 在 9/33 hand (27%) 上发生，8/9 在 R 手
- 2 case (028_p2 R, box023_p1 R) palm 说 contact 但 fingertip 说 no_contact → IK 过拟合直接证据
- quat 普查超预期：整个 CORE4D box family 都 disable_world_up=True，不只 box021 D003
- 决策：E100 build_fingertip_aware_target.py 默认 use_world_up=False

## 下一步
- 启动 E100（Stage 2）：build_fingertip_aware_target.py + 干净 A/B
