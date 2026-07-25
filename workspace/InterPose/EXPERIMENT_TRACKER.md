# InterPose 实验跟踪器

_InterPose 双人协作数据审计与 SPIDER 适配 · 2026-07-25_

---

| Run | 日期 | Phase | 描述 | 状态 | Log |
| --- | --- | --- | --- | --- | --- |
| R001 | 2026-07-25 | 数据审计与适配规划 | 发布包无确认 H-O-H；转向原视频几何重建 | 完成 | [01](log/01_v0.1_dataset_audit.md) |

## 📊 关键指标演进

| Run | 本地序列 | 双人序列 | 人-物-人候选 | 物体类别 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| R001 | 73,818 | 54,919 多 track | 17,278 文本筛选 / 0 确认 | 70 | 无 object SE(3)/frame sync，不能直接适配 |

## 🔗 产物索引

- 计划：[`plan/01_v0.1_dataset_audit_plan.md`](plan/01_v0.1_dataset_audit_plan.md)
- 数据统计：[`data_stat.md`](data_stat.md)
- 适配方案：[`adaptation_plan.md`](adaptation_plan.md)
- 结果日志：[`log/01_v0.1_dataset_audit.md`](log/01_v0.1_dataset_audit.md)
