# E145 Phase 2 高优 9 Case Full CEM 结果

## 结论

按 `workspace/core4d/plan/154_E145_full_nonbox_template_release_to_rl_ready_plan.md` 的 Phase 2 路线，已对人工标注高优队列执行 full CEM，并推进到 S6 evidence 与 RL export gate。

本轮没有启动 RL，没有生成 `.pt/.pth/.ckpt` checkpoint。

## 输入队列

本轮执行队列：

- `workspace/core4d/scripts/E145/variants_phase2_high.tsv`
- 来源：`workspace/core4d/results/E145/full_nonbox_to_rl_ready/cem_ready/e145_phase2_run_this_round_high.tsv`
- rows: 9
- split: local-gpu0 3 / remote-gpu0 3 / remote-gpu1 3

执行入口：

- local: `workspace/core4d/scripts/train/train_E145_full_nonbox_raw_mask_ref_fk.sh`
- remote: `workspace/core4d/scripts/run_E145_remote.sh`
- pull: `workspace/core4d/scripts/pull_E145_remote_results.sh`
- eval: `workspace/core4d/scripts/eval/eval_E145_full_nonbox_raw_mask_ref_fk.sh`

## E144 overlap audit

启动后发现本轮高优队列中有 1 个 case 已在 E144 full CEM 中跑过：

- `bucket003_20231018_001_p2`

该 row 在 E145 中已经完成，属于重复计算。已补充 overlap audit：

- `workspace/core4d/results/E145/full_nonbox_to_rl_ready/cem_ready/e145_phase2_high_e144_overlap_audit.tsv`
- overlap_count: 1

并给 E145 train runner 增加默认防重跑 guard：若 manifest 与 E144 full-CEM cases 重叠，默认 fail；只有显式设置 `E145_ALLOW_E144_OVERLAP=1` 才允许 intentional rerun。

## Artifact audit

9/9 variants 均具备三类核心产物：

- root npz: `workspace/core4d/results/E145/full_nonbox_to_rl_ready/cem/full/<variant>.npz`
- video: `workspace/core4d/results/E145/full_nonbox_to_rl_ready/cem/full/<variant>_full.mp4`
- outdir trajectory: `workspace/core4d/results/E145/full_nonbox_to_rl_ready/cem/full/<variant>_outdir_full/trajectory_mjwp_act.npz`

本地和远程均无 E145 CEM 进程残留。

## Eval

Eval 输出：

- metrics: `workspace/core4d/results/E145/full_nonbox_to_rl_ready/eval/full/e145_raw_mask_ref_fk_full_cem_metrics.tsv`
- summary: `workspace/core4d/results/E145/full_nonbox_to_rl_ready/eval/full/e145_summary.json`
- downstream evidence input: `workspace/core4d/results/E145/full_nonbox_to_rl_ready/eval/full/e145_downstream_evidence_input.tsv`
- handoff subset: `workspace/core4d/results/E145/full_nonbox_to_rl_ready/eval/full/e145_phase2_high_handoff_subset.tsv`

Eval summary:

- manifest_rows: 9
- evaluated_rows: 9
- missing_rows: 0
- cem_status_counts: pass 1 / fail 8
- downstream_decision_counts: `DOWNSTREAM_CEM_PASS=1`, `DOWNSTREAM_CEM_FAIL=8`
- eval-side rl_export candidates: `RL_EXPORT_READY=1`, `SKIP_CEM_FAIL=8`

Per-case CEM status:

| case | status | failure |
|---|---|---|
| `bucket003_20231018_001_p2` | fail | lowerbody_interference |
| `bucket007_20231003_2_021_p1` | pass |  |
| `bucket007_20231018_019_p2` | fail | cem_work_status_fail |
| `bucket010_20231003_2_055_p2` | fail | cem_work_status_fail |
| `desk021_20231011_010_p2` | fail | cem_work_status_fail |
| `desk023_20231020_116_p1` | fail | cem_work_status_fail |
| `desk023_20231020_117_p1` | fail | lowerbody_interference |
| `desk023_20231023_116_p1` | fail | cem_work_status_fail |
| `desk023_20231030_019_p1` | fail | lowerbody_interference |

注意：`bucket007_20231003_2_021_p1` 被当前 strict gate 判为 CEM pass，但 `hand_object_contact_physics_frac=0`。因此它只能被记录为 RL export gate 的候选输入，不能解释为 RL 成功或高质量最终行为。

## S6 Evidence And RL Export

S6 downstream evidence:

- `workspace/core4d/results/E145/full_nonbox_to_rl_ready/s6_downstream/downstream_evidence_manifest.tsv`
- rows: 9
- cem_status_counts: pass 1 / fail 8
- rl_status_counts: not_run 9

RL export input:

- `workspace/core4d/results/E145/full_nonbox_to_rl_ready/s6_downstream/rl_export/rl_export_input.tsv`
- rows: 9
- `RL_EXPORT_READY=1`
- `SKIP_CEM_FAIL=8`

本表只是 RL 导出输入索引；本轮未启动 RL。

## Static checks

已通过：

- `python -m py_compile workspace/core4d/scripts/eval/eval_E145_full_nonbox_raw_mask_ref_fk.py`
- `bash -n` for E145 train/remote/pull/eval scripts
- E145 results 下无 `.pt/.pth/.ckpt`

## Partner OmniRetarget Supplement

2026-06-05 21:24 CST 按用户要求补齐两条 CEM case 的 partner OmniRetarget 序列，仅执行/整理 OmniRetarget/trim，不进入 contact/SPIDER/CEM/RL。目录级 `partner_omnirt` manifest 为全量 9 行索引：2 行已完成，7 行保持 `ready_to_execute`，未执行其它 partner。

输出按 E108 `s6_downstream/rl_export/partner_omnirt` 格式落盘：

- root: `workspace/core4d/results/E145/full_nonbox_to_rl_ready/s6_downstream/rl_export/partner_omnirt/`
- case index: `cases_rl_partner_omnirt.tsv` (`9` rows)
- manifest: `rl_partner_omnirt_manifest.tsv` (`pass=2`, `ready_to_execute=7`)
- summary: `rl_partner_omnirt_summary.json`
- run script: `run_rl_partner_omnirt.sh`
- bucket010 partner retargeted: `results/omnirt_v1/holosoma_rl_partner_omnirt_omnirt_v1_bucket010_20231003_2_055_p1/retargeted/20231003_2-055-person1-bucket010_with_obj_original.npz`
- bucket010 partner trimmed: `results/omnirt_v1/holosoma_rl_partner_omnirt_omnirt_v1_bucket010_20231003_2_055_p1/trimmed/20231003_2-055-person1-bucket010_with_obj_original.npz`
- bucket010 trim window: `start=0`, `frames=130`
- desk023 partner retargeted: `results/omnirt_v1/holosoma_rl_partner_omnirt_omnirt_v1_desk023_20231030_019_p2/retargeted/20231030-019-person2-Desk023_with_obj_original.npz`
- desk023 partner trimmed: `results/omnirt_v1/holosoma_rl_partner_omnirt_omnirt_v1_desk023_20231030_019_p2/trimmed/20231030-019-person2-Desk023_with_obj_original.npz`
- desk023 trim window: `start=57`, `frames=139`

注意：`bucket010_20231003_2_055_p1` 在 E145 S3 已有完整 OmniRetarget/trim 输出，本次同步到 S6 partner root 后刷新 manifest。`desk023_20231030_019_p2` 标准 no-fingertip route 首次 retarget 在 CVXPY solve 处 infeasible；保持 `REPLACE_WRIST_WITH_FINGERTIP=0` 不变，单独对该 partner run 启用 Holosoma 现有 fallback 参数 `--retargeter.enable-constraint-relaxation --retargeter.max-fallback-retries 3` 后通过。
