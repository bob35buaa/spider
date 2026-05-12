# E056 Progress — 2026-05-12

## 当前状态: ✅ 完成 (多 case hand-face 诊断 + E057 决策)

## 完成步骤

- [x] 写 plan → `plan/66_E056_multi_case_hand_face_diagnosis_plan.md`
- [x] 验证 6 case 都有 scene + traj + 全 box collision geom
- [x] 实现 `workspace/core4d/scripts/E056/multi_case_face_diagnosis.py` (~280 行)
  - 复用 E055 v3 detector 找 intent
  - 复用 E055 6 面命名 + signed dist 时序
  - 5 类 grasp_type 分类 + valid 判定
  - 自适应 box half-sizes (从 model.geom_size 读, 不 hardcode)
- [x] 跑 6 case 诊断 → 6 png + summary csv + summary md
- [x] 视频核实 3 case × 3 帧 (box023 / bucket005_s2 / box021)
- [x] 写一键脚本 `run_E056_diagnosis.sh`
- [x] 写 log → `log/66_E056_multi_case_diagnosis_results.md`
- [x] 更新 EXPERIMENT_TRACKER (E056 行 + log/plan/script 引用)

## Claims 验证

| ID | 标准 | 实际 | 通过 |
|---|---|---|---|
| C1 | 6 case 全跑通 | 6 png + 1 csv + 1 md | ✅ |
| C2 | summary csv 标注 | 15 列 6 行完整 | ✅ |
| C3 | ≥1 case 是对侧或垂直 | **4 个 valid (1 对侧 + 3 垂直 + 1 单手)** | ✅ |
| C4 | 视频核实一致 | 3/3 case 视觉与算法一致 | ✅ |
| C5 | E057 决策 | 推荐 bucket005_s2 + 路线 A | ✅ |

**5/5 通过 ✅**

## 关键结果

```
✅ valid (5/6):
   bucket005_s2 ⭐ 对侧 (-yz / +yz, 88 帧最长) ← E057 起点
   box023        垂直 (-xy / -yz)
   bucket007     垂直 (+xy / +yz)
   desk021       垂直 (+xz / -xy)
   bucket001     单手 (L on +xz)

❌ invalid (1/6):
   box021        同面 (+xy / +xy, 双手按顶, 不是搬运握) ← 应从 Path B 移出
```

## 重要修正 (诚实记录)

E055 收尾时, 我**错误地同意用户**"L 一直在 +xz 面" 的观察 — 没仔细看 +xz signed_dist 全程是**负值** (palm 在 box 内部, 不可能从外侧接触)。E056 视频核实后**纠正**: box023 的 L 实际在 -xy (底面), 是 valid 的"垂直握"。

**用户的整体直觉"box023 ref 怪" 部分对**: box023 是 valid 的, 真正"怪"的是 **box021** (双手按顶面)。

## 下一步

- **E057 (Path B snap on bucket005_s2)**:
  - 复用 E055 hand_snap_ik.py + snap pipeline
  - 改 case 路径为 bucket005_s2_person1
  - 输出 warmstart_qpos.npz + snap_visualization.mp4 + diagnostics.csv
  - **新 Claim**: 接触面与 E056 诊断一致 (L on -yz, R on +yz)
- **E058+ (Path B-CEM)**: E057 通过后接入 CEM, 加 palm orientation IK 减穿模

## 改动文件

| 类型 | 路径 |
|---|---|
| 新建 plan | `workspace/core4d/plan/66_E056_multi_case_hand_face_diagnosis_plan.md` |
| 新建脚本 | `workspace/core4d/scripts/E056/multi_case_face_diagnosis.py` |
| 新建脚本 | `workspace/core4d/scripts/run_E056_diagnosis.sh` |
| 输出 csv | `workspace/core4d/results/E056/case_grasp_type_summary.csv` |
| 输出 md | `workspace/core4d/results/E056/E056_summary.md` |
| 输出 png | `workspace/core4d/results/E056/<case>_face_dist.png` (×6) |
| 输出 jpg | `workspace/core4d/results/E056/video_verify/*.jpg` (×9) |
| 新建 log | `workspace/core4d/log/66_E056_multi_case_diagnosis_results.md` |
| EXPERIMENT_TRACKER | 添加 E056 行 + log/plan/script 引用 |
