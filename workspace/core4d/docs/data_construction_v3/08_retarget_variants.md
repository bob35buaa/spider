# 08 Retarget variant 管理

v3 把 OmniRetarget 求解器版本和 input conversion 参数作为一等公民管理。S3 之后所有输出必须同时绑定 `retarget_variant_id` 和 `target_variant_id`。前者描述 OmniRetarget/input rewrite，后者描述 SPIDER target route。

## 分叉点

S0-S2 可共享：

- raw inventory；
- raw contact 3cm/5cm；
- source template；
- template audit。

S3 开始分叉：

- converted NPZ；
- retargeted NPZ；
- trimmed NPZ；
- SPIDER target task；
- target gate；
- visual QC；
- CEM/RL evidence。

不同 variant 不得覆盖彼此输出。

## 初始 variant

| id | 含义 | 说明 |
|---|---|---|
| `omnirt_original` | 原始 OmniRetarget | pin 到 Holosoma git `9c238cf80f531c0e65d818348c3c1a5cc2764f5b`（Initial public release）；用于回溯或严格对照。该提交没有当前 CORE4D converter，执行前必须单独 checkout 并实现/指定 original Stage2b adapter。 |
| `omnirt_v1` | v1 优化算法 | 默认 medium box route；wrist 不替换为 fingertip。 |
| `omnirt_v1_fingertip_replacement` | v1 + fingertip replacement | 显式 reach/大物体 ablation；不是默认 production。 |

## `replace_wrist_with_fingertip`

这是 conversion/input rewrite 参数，不是 solver 内部参数。

代码语义：

- 从 raw SMPL-X `joints` 中取左右手 5 个 fingertip joint；
- 计算每只手 fingertip 均值；
- 替换 `global_joint_positions[:,20]` 和 `[:,21]`；
- 再送入 OmniRetarget。

效果：

- wrist/eef source target 被推向 fingertip cluster；
- 对 Box025 等 reach-risk 大物体可能有用；
- 对中小箱或低位合抱可能把目标推向 box 内部或错误面；
- 必须显式记录，不能作为隐式默认。

## variant manifest

每条 Stage2b row 必须记录：

- `retarget_variant_id`
- `solver_family`
- `solver_version`
- `solver_repo_path`
- `solver_git_sha`
- `solver_dirty`
- `converter_script`
- `converter_git_sha`
- `replace_wrist_with_fingertip`
- `include_fingertip_centers`
- `trim_policy`
- `params_json`
- `branch_stage=conversion`
- `converted_npz`
- `omniretarget_output_npz`
- `trimmed_npz`
- `spider_trajectory`

当前 S3 wrapper 的行为：

- `omnirt_v1` 向 legacy wrapper 显式传 `REPLACE_WRIST_WITH_FINGERTIP=0`；
- `omnirt_v1_fingertip_replacement` 向 legacy wrapper 显式传 `REPLACE_WRIST_WITH_FINGERTIP=1`；
- `target_variant_id` 由 S3 参数显式选择，默认 `ref_fk`；选择 `fingertip_aware` 时才要求 E099-E101 route diagnostics；
- 当前 `--execute` adapter 只支持 `target_variant_id=ref_fk`；`adaptive` / `fingertip_aware` 的真实执行需要后续实现专门的 target adapter；
- `omnirt_original` 已有 exact commit，但带 `requires_solver_checkout=true`；S3 会输出 `stage2b_variant_requires_solver_checkout`，不会用当前 v1 checkout 执行 original。

这保证 fingertip replacement 和 fingertip-aware target 是两个可组合但不互相强绑的轴，不会污染默认 `ref_fk` 路线。

## 新增算法

新增 retarget 算法时，不复制 pipeline。实现新的 `RetargetAdapter`，注册新的 `retarget_variant_id`，输出同样 schema。
