# E053 Progress — 2026-05-12

## 当前状态: 实验运行中

---

## 实验配置

| 实验 | Case | Margin | GPU | 位置 |
|------|------|--------|-----|------|
| E053c_box025_m100 | box025 | 1.00 | GPU0 | **本地** |
| E053a_box025_m090 | box025 | 0.90 | GPU0 | 远程 |
| E053b_box025_m095 | box025 | 0.95 | GPU0 | 远程 |
| E053c_box025_m100 | box025 | 1.00 | GPU0 | 远程 |
| E053a_bucket010_m090 | bucket010 | 0.90 | GPU1 | 远程 |
| E053b_bucket010_m095 | bucket010 | 0.95 | GPU1 | 远程 |
| E053c_bucket010_m100 | bucket010 | 1.00 | GPU1 | 远程 |
| E053a_desk005_m090 | desk005 | 0.90 | GPU0 | 远程 (Phase 2) |
| E053b_desk005_m095 | desk005 | 0.95 | GPU0 | 远程 (Phase 2) |
| E053c_desk005_m100 | desk005 | 1.00 | GPU0 | 远程 (Phase 2) |

## 对照组 (已有结果)

| 实验 | Case | Margin | 来源 |
|------|------|--------|------|
| E041c_box025 | box025 | ~0.85 (template) | pre-fix E041c |
| E041c_bucket010 | bucket010 | Y/Z swapped | pre-fix E041c |
| E048_box025_baseline | box025 | 1.05 | post-fix E048 |
| E048_bucket010_baseline | bucket010 | 1.05 | post-fix E048 |
| E048_desk005_baseline | desk005 | 1.05 | post-fix E048 |

## 进度

- [x] 创建 set_collision_margin.py
- [x] 创建 run_E053_remote.sh
- [x] 本地 E053c_box025_m100 启动
- [x] 远程 Phase 1 (box025 + bucket010) 启动
- [ ] 远程 Phase 2 (desk005) — 等 Phase 1 完成
- [ ] 收集结果
- [ ] 视频分析
- [ ] 写实验日志
