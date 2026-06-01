# 06 失败分类

失败分类要区分数据、template、retarget、target gate、visual QC 和下游 CEM/RL。不能把下游失败反向写成 raw data 失败。

## 标准分类

| 分类 | 含义 |
|---|---|
| `PASS` | 当前阶段通过 |
| `REVIEW` | 机器指标不足以决定，需要人工/visual review |
| `REJECT_RAW_CONTACT` | raw contact 3cm/5cm 不满足候选条件 |
| `REJECT_RAW_FINGERTIP_MISMATCH` | 仅 `fingertip_aware` route 使用，raw fingertip/palm/face evidence 不支持该 route |
| `REJECT_FINGERTIP_ROUTE_CONTRACT` | 选择了 `fingertip_aware`，但 E099-E101 route diagnostic 缺失或未通过 |
| `REJECT_TEMPLATE_BACKLOG` | source template 缺失，需补 template |
| `REJECT_TEMPLATE_AUDIT` | template 存在但审计失败 |
| `REJECT_OMNIRETARGET` | OmniRetarget/CVXPY/preprocess 失败 |
| `VARIANT_PENDING` | retarget variant 已有身份或历史 commit，但当前缺少可执行 adapter/checkout，不是 case 数据失败 |
| `REJECT_TARGET_GATE` | target scene/trajectory/gate 失败 |
| `REJECT_VISUAL_QC` | 可视化显示明显不合理 |
| `QUARANTINE_LEGACY_TEMPLATE_BUG` | 历史结果受 template 污染，不能当 hard label |
| `DOWNSTREAM_POSTURE_FAIL` | CEM/RL 姿态失败，例如低髋、趴箱 |
| `DOWNSTREAM_MOTION_BINDING_FAIL` | CEM/RL 不能完成有效搬运或 motion-level binding 失败 |
| `DOWNSTREAM_CEM_PASS` | CEM 下游证据通过，但不等同 RL-ready |
| `DOWNSTREAM_RL_PASS` | RL 下游证据通过 |
| `DOWNSTREAM_CEM_FAIL` | CEM 失败，但没有更细 posture/motion 归因 |
| `DOWNSTREAM_RL_FAIL` | RL 失败，但没有更细 posture/motion 归因 |
| `DOWNSTREAM_NOT_RUN` | 尚无 CEM/RL 下游证据 |

## hard gate 与 diagnostic

Hard gate 会影响 pass/fail。Diagnostic 只记录，不单独作为失败归因。

当前 replay 口径：

| 指标 | 口径 |
|---|---|
| `pelvis_end_z` | 可作为 hard gate |
| `lie_on_box_frac` | 可作为 hard gate |
| `pelvis_tilt_end` | diagnostic，不单独作为 hard fail |

这个口径吸收了 E098 的 replay gate 和 E107 的修订：tilt 仍报告，但不作为单独失败原因。

## route-specific failure

`ref_fk` 是默认 route，不要求 E099-E101 全套 fingertip contract。

`fingertip_aware` route 必须额外检查：

- fingertip vote；
- palm vote；
- `face_changed_l/r`；
- quat audit；
- target gap；
- active mask；
- E101-style route evidence。

这些缺失时，不能把 `fingertip_aware` 输出升为 route positive。

v3 中这些检查以 `route_diagnostic_status=pass` 落到 S3 manifest 和 registry。缺失不是 `ref_fk` 的失败；只对显式选择 `fingertip_aware` 的 row 生效。

## variant-specific pending

`omnirt_original` 这类历史 solver 对照要区分“来源可追溯”和“当前可执行”：

- exact commit/source 已知，但当前 checkout 或 adapter 不匹配时，S3 写 `stage2b_variant_requires_solver_checkout`；
- registry 写 `stage2b_status=variant_requires_checkout` 与 `current_decision=PENDING_VARIANT_ADAPTER`；
- S5 写 `candidate_decision=VARIANT_PENDING` 与 `HANDOFF_PENDING_TEMPLATE_OR_VARIANT`。

这不是 raw/contact/template 数据失败，也不是 OmniRetarget infeasible；它表示需要准备对应 solver checkout 或专门 adapter 后才能执行该 variant。

## downstream evidence

S6 的 CEM/RL 结果用于记录下游可用性和失败模式，不参与 S1-S5 的数据构建 hard gate。

示例：

- 一个 row 的 `target_gate_status=pass` 且 `visual_qc_status=pass`，S5 可以是 `PASS/HANDOFF_READY`。
- 后续 CEM 若出现低髋、趴箱，可在 S6 写 `DOWNSTREAM_POSTURE_FAIL`。
- 该 S6 失败不能反向改成 `REJECT_RAW_CONTACT`、`REJECT_TEMPLATE_AUDIT` 或 `REJECT_TARGET_GATE`，除非发现的是新的数据构建 bug，并另开审计修复。
