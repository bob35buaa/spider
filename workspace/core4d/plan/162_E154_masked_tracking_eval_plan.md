# E154 — Masked-contact(真实 3cm)+ body-tracking 评测修复,重评 E152/E153

> 计划编号 162 · exp_name: core4d · 分支 exp/core4d-collab-retarget
> 类型:**纯评测方法学修订**(复用已有轨迹,不重训)

## Context

用户审查 E152/E153 评测时发现两个真实缺陷:
1. 所有接触/穿透指标按**全序列**统计,**无 contact mask** → 机器人在动作结尾该放下物体起身时
   仍抱着箱子(假接触),反而**抬高**接触指标(奖励了失败)。
2. **无机器人本体跟踪指标**——只有 `obj_err`,且比的是逐 run 漂移的**内嵌** `qpos[:,1,:]`,
   对"弯腰不起身"完全不敏感。

调查根因(见 progress 调查记录):
- `trajectory_kinematic.npz` 的 `contact` 字段是**全 1**(`core4d.py` `contact_detection_mode="one"`
  默认)。该字段经 `spider/io.py:load_data` 同时喂给 **reward** → 优化器从训练起就被奖励
  **全程保持接触**,放手从未被激励。退化 mask 同时污染训练与评测。
- 真实 3cm mask 一直存在但未用:`workspace/core4d/results/E143/contact_masks/<case>/raw_contact_mask_3cm.npz`
  的 `spider_contact_mask_3cm`(N,2人,2手),逐帧对齐,清晰呈现 接近→搬运→放手。

## Claims

- **C1** 真实 mask ≠ 全 1:`spider_contact_mask_3cm` 有非零放手窗口(box004 尾部 ~17 帧,
  `ref_contact_frac≈0.59`)⇒ masked 指标 ≠ 全序列;书面记录全 1 管线 bug 及其向 reward 的泄漏。
- **C2** 新指标能揭露失败:box004 弯腰 run `track_pelvis_z_terminal > 0.08` ⇒ `success_tracked=False`,
  站起 run 通过;视觉佐证。
- **C3** 重评后裁定:在 tracking 门控 success 下报告仍 3/3 的 combo;核验 `(−0.010,0.10)` 是否存活、
  `(−0.005,0.05)` 是否被降级。
- **C4(诊断)** 放手普遍失败,连 b1 参考都不放(release_false 高且非单调)⇒ 不进 success 门控,
  作为指向训练侧 all-1 mask bug 的证据,引出重训后续。

## 改动

- `workspace/core4d/scripts/eval/lib/core_metrics.py`:`EvalConfig` 加
  `track_terminal_frac=0.15`、`track_pelvis_terminal_th_m=0.08`;`evaluate_sequence` 加可选
  `kin_ref_path`/`contact_mask_path`/`person_idx`;新增 `track_*`(对固定 kin 真值的 root/pelvis/joint
  误差,末段=最后15%帧)与 masked 接触(`ref_contact_frac`、`*_in_mask_frac`、`false`/`approach`/`release`
  false-contact)。复用 `spider/postprocess/get_humanoid_tracking_err.py` 约定 + `spider.math.quat_sub`。
  新增辅助 `kin_ref_for_scene`/`person_idx_from_case`/`contact_mask_for_case`。
- `eval_E153_*`/`eval_E152_*`:逐 case 传 3 个 ref,加 `track_*`/`*_in_mask`/`false_contact` 列;
  **门控 success(仅 tracking)** `success_tracked = success_pen2mm AND track_pelvis_z_terminal ≤ 0.08`;
  combo 汇总改用 `success_tracked`。

## 成功标准
- 两评测 `missing=0`;现有依赖 `core_metrics` 的评测(E151)仍可跑(新列为空)。
- mask 时间对齐(b1 手-物 SDF 搬运窗口内最小)。
- 视觉 A/B 与裁定一致。

## 运行命令
```bash
uv run python workspace/core4d/scripts/eval/eval_E153_gate_threshold_sweep.py full
uv run python workspace/core4d/scripts/eval/eval_E152_axis1_hand_object_physics_gate.py full --skip-visual
```

## 非目标
- 不重训。根因(reward 用 all-1 mask)的修复=用真实 `spider_contact_mask_3cm` 重训,
  是单独的更大后续(新 E + 可能新分支),待用户决策。
- 5cm mask 需 `CORE4D_REAL_ROOT` 原始数据重新生成,非结论所必需。
