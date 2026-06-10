# ⚠️ DEPRECATED — 已废弃

> **本目录已废弃**，功能由 `workspace/core4d/scripts/data_construction_v3/` 取代。
> 新实验请使用 v3 管线。本目录仅保留供历史追溯。

## 替代方案

新管线入口：
```bash
# v3 数据构建管线
workspace/core4d/scripts/data_construction_v3/orchestration/run_release_checks.sh
workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py
```

文档：`workspace/core4d/docs/data_construction_v3/README.md`

## 本目录的历史价值

- 记录了 E077-E079 时代的单 case 预处理流程
- `pipeline.sh` 展示了 convert → OmniRetarget → trim → contact mask → scene_act → verify 的串行链路
- `SCENE_TEMPLATE_GUIDE.md` 中的 scene XML 构建思路仍有参考价值

## 为什么废弃

1. 路径硬编码，换机器后无法运行
2. 不支持批量状态管理（v3 有 `case_state_registry`）
3. 不支持多阈值 raw contact（v3 同时输出 3cm/5cm）
4. 不支持 retarget variant 分叉（v3 有双轴管理）
5. 缺少 E103 修复后的 inertial/collision audit
