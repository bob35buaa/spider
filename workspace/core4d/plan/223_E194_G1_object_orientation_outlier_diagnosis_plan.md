# E194 G1 object orientation 离群回退诊断计划

## Context

E194 G1 仅把 object body 的 `gravcomp` 从 PRG 基线改为 1，并已完成 box001、box023、box021 共 72 个 case。box001 主分析（排除 `box001_20231023_110_p1`）显示 object orientation error 均值回退，但用户指出其中至少三个 case 的 G1 error 比 PRG 增加 18° 以上且超过 PRG 的 3 倍；去掉三者后 box001 平均差仅约 +0.29°。

### 根因分析

均值回退可能是两种不同现象：

1. 少数接触几何敏感 case 在 G1 下发生物体旋转滑移，导致长尾主导均值；
2. 所有 case 都发生小幅 orientation shift，再叠加少数严重失败。

`gravcomp` 理论上在 object body 的质心处补偿重力净力，不直接施加旋转力矩；因此需要用逐帧姿态、接触、物体位置和配置差分判断回退是否经由抓持预载/接触分布/优化解间接产生，并排除 quaternion、对称物体和 reference 对齐等评测伪影。

### 关键 insight

用 case-level robust aggregation 与逐帧首次分叉定位可以把“整体 shift”和“少数灾难性滑移”分开；同 session 的 p1/p2 重复异常还能检验问题更接近 source motion/contact geometry，还是 retarget variant。

## Claims

| Claim | 最低证据 |
|-------|---------|
| C1：box001 均值回退是否由少数离群 case 主导 | 报告全 case PRG/G1/delta/ratio、median、trimmed mean、去 top-1/2/3 均值及 top-3 对总增量贡献 |
| C2：box023/box021 是否存在同类长尾 | 对两个物体应用相同阈值与稳健统计，列出所有 `delta>18°`、`G1/PRG>3`、orientation gate flip case |
| C3：三个指定 case 的异常不是评测伪影 | 按公共 evaluator 的 quaternion convention 逐帧重算，并核对有限值、quat sign、reference 长度/对齐与对称性风险 |
| C4：G1 的作用机制与异常的关系可被边界化 | 精确核对 PRG/G1 XML/config diff；区分 gravcomp 的直接动力学作用与接触/优化导致的间接旋转机制，并给出至少两个正常 control |

## 改动

### 1. 新增离线诊断 runner

**文件**: `workspace/core4d/scripts/eval/reports/analyze_E194_G1_object_orientation_outliers.py`

- 读取冻结的 E194 three-arm case metrics 与 paired deltas；必要时从 result NPZ 用公共 metric helper 重算逐帧 orientation error。
- 生成 case 排名、稳健统计、离群贡献、跨物体对照与逐帧首次分叉诊断。
- 不修改 scene、result NPZ、人工标签或原始 E194 日志。
- 从每条 E194 G1 Full stdout 读取实际 quaternion→Euler convention，按 XML object hinge axis sequence 重建应有 convention，并用 MuJoCo FK 量化内部 Euler target 相对 raw quaternion authority 的 world-orientation 偏差。
- 增加 focal 回归测试，固定验证 `box001_20231003_2_041_p1` 的 runtime `XYZ` 与 XML `XZY` 不一致、wrong-target world error 大于 25°、正确转换误差小于 `1e-4°`。

### 2. 生成诊断证据与报告

**结果目录**: `workspace/core4d/results/E194/s6_downstream/eval/full_g1_expansion/`

- `e194_g1_object_orientation_outlier_cases.tsv`
- `e194_g1_object_orientation_robust_summary.tsv`
- `e194_g1_object_orientation_frame_diagnostics.tsv`
- `e194_g1_object_orientation_reference_conversion_audit.tsv`
- `e194_g1_object_orientation_reference_conversion_summary.tsv`
- 关键帧/对照帧（若已有 MP4 可用）

**新日志**: `workspace/core4d/log/276_E194_G1_object_orientation_outlier_diagnosis.md`

### 3. 项目状态回写

- 更新 `workspace/core4d/progress.md`。
- 诊断闭合后在 `workspace/core4d/EXPERIMENT_TRACKER.md` 的 E194 行追加 log276 链接；不覆盖 log273–275。

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `scripts/eval/reports/analyze_E194_G1_object_orientation_outliers.py` | 新增可复现的离线统计/逐帧诊断 |
| 2 | `scripts/eval/reports/e194_orientation_reference_conversion.py` | 独立封装 run-log convention 与 MuJoCo reference parity 审计 |
| 3 | `scripts/eval/reports/test_e194_orientation_reference_conversion.py` | 固化 focal conversion 回归测试 |
| 4 | `log/276_E194_G1_object_orientation_outlier_diagnosis.md` | 保存观测、解释、限制和下一步 |
| 5 | `progress.md` | 记录执行与关键发现 |
| 6 | `EXPERIMENT_TRACKER.md` | 增加 E194 诊断日志索引 |

## Reward 权重（不适用）

本轮不训练、不修改 reward 或物理配置，仅分析既有 PRG/G1 rollout。

## 运行命令

```bash
python workspace/core4d/scripts/eval/reports/analyze_E194_G1_object_orientation_outliers.py \
  --eval-root workspace/core4d/results/E194/s6_downstream/eval/full_g1_expansion
```

若已有配对 MP4，再按 `video-frames` skill 从异常 case 与正常 control 提取同阶段关键帧。

## 成功标准

| 指标 | 成功标准 |
|------|----------|
| 数据覆盖 | box001/box023/box021 72/72；box001 同时报 all-28 与排除指定 case 后的主 27 |
| 离群归因 | 能量化 top-3 对 box001 总 orientation delta 的贡献，并复现/解释用户的约 +0.29° 观察 |
| 跨物体外推 | 明确 box023/box021 是同类长尾、整体 shift，还是混合模式 |
| 机制证据 | XML/config diff 精确；逐帧首次分叉与 contact/position/rotation 联动证据可审计 |
| Reference parity | 72/72 G1 Full 日志 convention 可解析；报告 convention match/mismatch 分组和 wrong-target error 与 PRG→G1 delta 的相关性 |
| 结论边界 | 明确 gravcomp 是否直接施加 torque，以及数据究竟支持直接因果还是间接机制 |
