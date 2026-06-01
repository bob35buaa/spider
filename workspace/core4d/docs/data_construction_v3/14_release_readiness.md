# 14 发布就绪说明

本页用于交接 Core4D 数据构建 v3 的当前可用状态。详细设计看 `README.md`、阶段规则看 `02_pipeline_stages.md`，需求逐项追踪看 `13_requirements_traceability.md`。

## 当前结论

当前 v3 主链路已经达到可交接状态：

- 脚本实现已按功能拆到 `lib/`、`orchestration/`、`stages/`、`state/`、`migration/`、`qa/`，根目录不再保留平铺脚本或 symlink；
- 可以在新机器上先跑无 raw-data release check，确认代码、文档、legacy 隔离和关键 contract 齐全；
- 可以在提供 `CORE4D_Real` 和 SMPL-X 模型路径后跑 compact smoke，验证 `full-from-raw`、`resume-from-summary` 和 run 级复现性检查；
- 可以用默认 `omnirt_v1/ref_fk` 路线执行真实 Stage2b，已有 box004 与 box026 smoke 证据；
- 可以把 S1-S5 生成的数据构建状态和 S6 下游 CEM/RL evidence 分开管理；
- 可以从 S5 handoff 导出 CEM override，避免下游隐式解析数据构建 manifest。

## 推荐入口

新机器先跑：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_release_checks.sh \
  --run-root /tmp/core4d_dcv3_release_checks \
  --no-smoke
```

提供 raw data 后跑：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_release_checks.sh \
  --run-root /tmp/core4d_dcv3_release_checks_with_smoke \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --smplx-model-dir "$SMPLX_MODEL_DIR" \
  --with-smoke
```

真实构建默认路线：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py \
  --mode full-from-raw \
  --run-id dcv3_$(date +%Y%m%d_%H%M%S) \
  --run-root "$DATA_CONSTRUCTION_RUN_ROOT" \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --smplx-model-dir "$SMPLX_MODEL_DIR" \
  --retarget-variant-id omnirt_v1 \
  --target-variant-id ref_fk
```

run 结束后检查：

```bash
workspace/core4d/scripts/data_construction_v3/qa/verify_reproducibility.py \
  --run-dir "$RUN_DIR"
```

## 最新验证

最新验证命令：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_release_checks.sh \
  --run-root /tmp/core4d_dcv3_release_checks_existing_seed \
  --no-smoke
```

结果：通过；release audit `status=pass`，62/62 checks pass。

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_release_checks.sh \
  --run-root /tmp/core4d_dcv3_release_checks_existing_seed_with_smoke \
  --core4d-raw-root /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real \
  --smplx-model-dir /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx \
  --with-smoke
```

结果：通过；release audit `62/62`；`smoke_suite/smoke_suite_report.json` 为 `status=pass`，20 个步骤全 pass，warnings 为空。

## 必须保持的边界

- 默认 target route 是 `ref_fk`；`adaptive` / `fingertip_aware` 不能复用 legacy Stage2b execute 冒充真实执行。
- `fingertip_aware` 必须先通过 E099-E101 route diagnostics。
- source scene missing 不能被当作数据 reject；必须进入 template backlog 或显式构建 source template。
- 非 box 物体不自动 release，必须人工审查 template/collision/mass/inertia。
- 旧 `workspace/v3/data_construction*` 只能显式导入或作为人工参考，不能成为隐式输入输出。
- CEM/RL 失败是 S6 downstream evidence，不能反向改写 S1-S5 的数据构建事实。
- results 和 v3 run root 不进 git；复现依赖 config、manifest、git sha、环境检查和命令记录。

## 计划内未完成项

以下不是当前 release blocker，但必须在后续扩展时单独实现和验证：

- `omnirt_original` 严格 execute 需要单独 checkout/adapter；
- `adaptive` / `fingertip_aware` 的真实 Stage2b target adapter；
- 非 box 自动 template 构建；
- 大批量 full-from-raw 覆盖更多 object/case；
- 完整 RL-ready positive 需要补充 S6 CEM/RL 证据。
