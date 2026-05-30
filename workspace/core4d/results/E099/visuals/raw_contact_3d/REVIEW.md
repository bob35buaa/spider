# E099 raw_contact 3D 视觉评估 (REVIEW)

## 评估范围
5 张 raw_contact 3D 4-view 可视化 PNG（box023_person2, box025_person2, d003_box021_030_p1, e091_box026_039_p2, d003_box021_028_p2）。

## 逐图结论

| Case | 几何贴合 | vote face 一致 | PALM vs FINGER |
|------|---------|---------------|----------------|
| box023_person2 (L+z/R+z) | YES (L 顶面，R 顶面边+少量+x 外溢) | YES (+z 顶面与多数指尖吻合) | **PALM≠FINGER** (palm × 飘到顶面上方 0.1–0.2 m) |
| box025_person2 (L+z/R+z) | YES (双手紧贴 +z 顶面) | YES (绿面 +z 与指尖完美吻合) | PALM≈FINGER (palm × 与 fingertip 共面) |
| d003_box021_030_p1 (L+z/R+x) | YES (L 顶面，R 在 +x 边缘) | YES (绿面覆盖 +z 与 +x 两面，与双手 vote 分别一致) | **PALM≠FINGER** (palm × 散布在 box 外远处) |
| e091_box026_039_p2 (L-z/R-z 底托) | YES (双手紧贴 -z 底面) | YES (绿面 -z 与指尖完美吻合) | **PALM≠FINGER** (palm × 飞出 box 外右下 / y>0.3) |
| d003_box021_028_p2 (L-z / R 无接触) | YES (L 在 -z 底面，R 无红点符合 no-contact) | YES (绿面 -z 与 L 指尖一致) | **PALM≠FINGER** (palm × 远离指尖在 box 外) |

## 汇总统计

- **5/5 几何贴合：YES (5/5)** — 所有 case 的 raw 指尖都贴在 box wireframe 表面 ≤2 cm 内，无内部穿透、无远离漂浮。
- **5/5 vote face 视觉一致：YES (5/5)** — 绿色高亮面与多数指尖聚集面完全对应（含双手不对称 case 030_p1 和单手 case 028_p2）。
- **PALM≠FINGER 案例计数：4/5** — 仅 box025_person2 一例 palm 与 fingertip 大致重合；其余 4 例 palm × 明显脱离 box 表面、散布在远处空中（最严重为 e091 底托 case，palm 完全飞出 box 外）。

## 决策结论

**完全支持 "fingertip vote 取代 palm vote" 的决策。**

关键证据：
1. **Fingertip 几何稳健 (5/5 贴合 + 5/5 vote 一致)**：raw 指尖落点在 box 表面分布致密，与几何法向高度一致，vote 多数面在所有 5 例都被肉眼确认正确。
2. **B6 假设强力验证 (4/5 PALM≠FINGER)**：5 个真实 case 中 4 个 palm 代理与指尖落点严重偏离（飘到 box 外 0.1 m+ 空中），证明 palm site (FK) 不能代表真实接触位置；用 palm 投票必然投到错误的面或干脆 unreachable。
3. **不对称 / 单手 / 底托 case 全部正常**：030_p1 (L+z R+x 不对称) 和 028_p2 (R 无接触) 与 039_p2 (双手 -z 底托) 都被 fingertip vote 正确识别，说明 fingertip 投票对手部姿态多样性鲁棒。

**建议**：E099 应固化为新 baseline，box021 等失败 case 重跑 CEM 时统一使用 fingertip-based vote face，并将 palm-based vote 路径降级为 fallback / 测试对照。
