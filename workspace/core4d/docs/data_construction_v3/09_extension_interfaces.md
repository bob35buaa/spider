# 09 扩展接口

v3 第一版不需要复杂框架，但必须预留稳定接口，避免后续新增过滤器或算法时重复造轮子。代码锚点是：

```text
workspace/core4d/scripts/data_construction_v3/lib/interfaces.py
```

该文件定义轻量 Protocol 和标准 result dataclass；当前 pipeline 不要求动态 plugin loading，但新增过滤器/算法/gate/visualizer 时应返回这些 result row，或至少包含同名字段。

## CandidateFilter 接口

职责：对 inventory/raw-contact/template rows 追加筛选逻辑。

输入：

- case row；
- raw contact metrics；
- object geometry；
- optional registry state。

输出：

- `decision`
- `status`
- `reason`
- `metrics_json`
- `evidence_paths_json`

示例过滤器：

- medium box size filter；
- raw contact 3cm/5cm filter；
- source template backlog filter；
- lower-body risk prefilter。

## RetargetAdapter 接口

职责：包装一种 retarget 算法或 OmniRetarget variant。

输入：

- case row；
- source template；
- `retarget_variant_id`；
- resolved config。

输出：

- converted NPZ path；
- `omniretarget_output_npz`；
- trimmed NPZ path；
- status；
- params/evidence manifest。

## TemplateBuilder 接口

职责：生成或审计 source template。

输入：

- object mesh；
- object type；
- clean base scene；
- policy。

输出：

- source scene；
- `task_info.json`；
- audit summary；
- visual evidence。

## TargetGate 接口

职责：判断 target scene/trajectory 是否可进入下游。

输入：

- target scene；
- trajectory；
- contact mask；
- optional external target；
- source template audit。

输出：

- pass/review/reject；
- metrics；
- per-frame evidence；
- failure mode。

## Visualizer 接口

职责：生成 release evidence。

输入：

- stage output；
- metrics；
- scene/trajectory。

输出：

- PNG sheet；
- MP4；
- visual manifest；
- optional review template。

## Result 字段约定

所有扩展 result 至少包含：

- `interface_name`
- `case_id`
- `decision`: `pass` / `review` / `reject` / `not_run`
- `status`: `pass` / `review` / `reject` / `not_run` / `fail` / `missing` / `error`
- `reason`
- `metrics_json`
- `evidence_paths_json`
- `schema_version`
- `updated_at`

`RetargetAdapterResult` 还必须包含 `retarget_variant_id`、`target_variant_id`、`converted_npz`、`omniretarget_output_npz`、`trimmed_npz`、`params_json`。

## CEM handoff adapter

S5 到 CEM 的交接通过 `export_cem_overrides.py` 固化为 adapter，而不是让下游脚本隐式解析 handoff manifest。

输入：

- `handoff_manifest.tsv`
- route-specific target 文件；
- repo-relative path 解析规则。

输出：

- `cem_override_manifest.tsv/json`
- `cem_override_summary.json/md`
- 每个可执行 row 一个 override YAML。

adapter 规则：

- `target_variant_id=ref_fk`：写 `contact_hdmi_target_source=ref_fk`，不要求 `target_npz`；
- `target_variant_id=adaptive/fingertip_aware/future external`：写 `contact_hdmi_target_source=external`，必须校验 `target_npz` path、sha256、shape 和 finite 值；
- adapter 只做配置导出与校验，不改变 registry 中 raw/template/Stage2b/gate/visual QC 的事实状态。

## RL export adapter

CEM 到 RL motion export 的交接通过 `export_rl_inputs.py` 固化为 adapter。RL 导出脚本不得自行扫描 CEM 目录或按 case 拼 `scene_act` 路径。

输入：

- `s5_handoff/handoff_manifest.tsv`
- S6 CEM `downstream_evidence_manifest.tsv`

输出：

- `s6_downstream/rl_export/rl_export_input.tsv/json`
- `s6_downstream/rl_export/rl_export_summary.json/md`

adapter 规则：

- 只把 `handoff_decision=HANDOFF_READY/HANDOFF_REVIEW_VISUAL_QC`、`target_gate_status=pass`、`visual_qc_status=pass`、`cem_status=pass` 且必需文件存在的 rows 标成 `RL_EXPORT_READY`；
- `RL_EXPORT_READY` row 必须同时带 `scene_act`、`trajectory`、`contact_mask`、`cem_result_npz`；
- CEM fail rows 标成 `SKIP_CEM_FAIL`，CEM 未跑 rows 标成 `WAIT_CEM_NOT_RUN`，非 handoff rows 标成 `SKIP_NOT_HANDOFF_READY`；
- adapter 不回写 S5，不改变 registry 中的数据构建事实。

## 原则

- 接口先保证 schema 稳定，不做过度抽象；
- 每个接口都支持 dry-run；
- 每个接口都输出 TSV/JSON；
- 所有失败都进入标准 failure taxonomy；
- 不隐式读取 legacy 目录。
