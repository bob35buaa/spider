# E096 High-Reasoning Visual Review

日期：2026-05-29

Reviewer：high reasoning subagent `019e730e-0fb8-7f50-a1e4-ae06cdf108f5`

## Inputs

| type | path |
|---|---|
| CEM P1 sheet | `workspace/core4d/results/E096/visual_review/frame_sheets/cem_p1_sheet.jpg` |
| CEM P2 sheet | `workspace/core4d/results/E096/visual_review/frame_sheets/cem_p2_sheet.jpg` |
| geometry P1/P2 sheets | `workspace/core4d/results/E096/visual_review/frame_sheets/geom_p1_sheet.jpg`, `workspace/core4d/results/E096/visual_review/frame_sheets/geom_p2_sheet.jpg` |
| projection P1/P2 sheets | `workspace/core4d/results/E096/visual_review/frame_sheets/proj_p1_sheet.jpg`, `workspace/core4d/results/E096/visual_review/frame_sheets/proj_p2_sheet.jpg` |

## Conclusion

P1/P2 的 `WORK` 结论成立。

## Observations

- P1、P2 的 CEM 视频均未见明显倒地、趴箱、头/上身/手接地，和量化指标一致。
- 弯腰阶段较深，但更像取箱/扶箱动作，不是上身压在箱体上。
- 未看到明显穿箱；接触主要发生在手与箱体上表面/侧面附近。
- 相机视角较远但主体完整可见，没有因裁切导致误判的迹象。
- 两者都符合 box004 positive pattern：靠近箱子、弯腰、双手接触/操作箱体，部分帧有抬起或搬运箱子的趋势。
- P2 的搬箱语义更清楚；P1 因画面较小，动作可读性略弱，但整体仍像 positive。
- geometry/projection sheets 中手部彩色 contact target 基本落在箱体上表面或侧边附近。
- 未看到 target 明显跑到脚、地面、头部或远离箱体的位置。
- 少数 marker 看起来略偏到箱体边缘外侧，更像投影视角或接触边界误差，不构成明显语义错位。

## Risk

- 风险较低；P1 的可读性稍弱，后续如抽查可优先看 P1 近景确认手-箱接触细节。
- P3 因 OmniRetarget CVXPY infeasible 暂不进入 CEM，不应与 P1/P2 的 `WORK` 结果混判。
