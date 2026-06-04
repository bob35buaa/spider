# E112 Contact-Aware CEM Ablation 结果

计划：`workspace/core4d/plan/121_E112_contact_aware_cem_ablation_plan.md`

总控计划：`workspace/core4d/plan/contact_improvement_plan.md`

## 目标

E112 承接 E110/E111，验证 Spider 当前手物接触下降是否可通过 contact-aware CEM 奖励恢复，而不是接触和低穿透之间的不可避免 trade-off。

Phase A 只跑 3 个代表 case，每个 case 3 个变体：

- `baseline_ref_fk`：干净 ref-FK baseline，显式 `contact_hdmi_gain=0.0`，不加载 contact mask。
- `raw_mask_ref_fk`：使用 raw 3cm contact mask 激活 contact reward。
- `hold_band`：使用 raw mask + hold-band contact 维持项。

## 产物路径

| 产物 | 路径 |
|---|---|
| manifest builder | `workspace/core4d/scripts/E112/build_contact_aware_cem_manifest.py` |
| variants | `workspace/core4d/scripts/E112/variants.tsv` |
| overrides | `examples/config/override/core4d_E112_*.yaml` |
| train runner | `workspace/core4d/scripts/train/train_E112_contact_aware_cem.sh` |
| remote runner | `workspace/core4d/scripts/run_E112_remote.sh` |
| pull script | `workspace/core4d/scripts/pull_E112_remote_results.sh` |
| evaluator | `workspace/core4d/scripts/eval/eval_E112_contact_aware_cem.py` |
| smoke results | `workspace/core4d/results/E112/cem/smoke/` |
| full results | `workspace/core4d/results/E112/cem/full/` |
| full summary | `workspace/core4d/results/E112/cem/full/full_eval_summary.md` |
| full CSV/JSON | `workspace/core4d/results/E112/cem/full/full_eval_summary.csv`, `workspace/core4d/results/E112/cem/full/full_eval_summary.json` |
| logs | `logs/E112/` |

## 执行

预检：

- 生成 9 个 override、`variants.tsv` 和 `phaseA_preflight.tsv`。
- `train_E112_contact_aware_cem.sh` 增加 hard preflight gate：preflight 任意 `False` 都拒绝 CEM。
- subagent 只读审查发现 `baseline_ref_fk` 初版继承 `contact_hdmi_gain`，会触发 rotated-SDF fallback mask；已修正为 baseline `contact_hdmi_gain=0.0`，raw/hold `contact_hdmi_gain=5.0`。
- 远端 `spider-remote:/home/xiayb/pHRI_workspace/spider` 通过 `rsync` 同步 E112 未提交脚本、overrides、preflight、mask、derived task 和 object mesh assets；没有 kill 远端已有进程。

烟测：

```bash
SMOKE_MAX_NUM_ITERATIONS=4 bash workspace/core4d/scripts/train/train_E112_contact_aware_cem.sh local smoke 0
ssh spider-remote 'cd /home/xiayb/pHRI_workspace/spider && SMOKE_MAX_NUM_ITERATIONS=4 bash workspace/core4d/scripts/run_E112_remote.sh smoke'
bash workspace/core4d/scripts/pull_E112_remote_results.sh smoke
bash workspace/core4d/scripts/eval/eval_E112_contact_aware_cem.sh smoke
```

smoke 结果：9/9 变体成功，`smoke_eval_summary.md` 写出。box004/box021 的 contact-aware 变体接触显著提升；box026 提升很小。所有 smoke strict 仍 FAIL，按短迭代烟测不作为最终结论。

full CEM：

```bash
bash workspace/core4d/scripts/train/train_E112_contact_aware_cem.sh local full 0
ssh spider-remote 'cd /home/xiayb/pHRI_workspace/spider && bash workspace/core4d/scripts/run_E112_remote.sh full'
bash workspace/core4d/scripts/pull_E112_remote_results.sh full
bash workspace/core4d/scripts/eval/eval_E112_contact_aware_cem.sh full
```

资源分配：

- 本地 GPU0：`baseline_ref_fk` 3 个 case。
- 远端 GPU0：`raw_mask_ref_fk` 3 个 case。
- 远端 GPU1：`hold_band` 3 个 case。

full 产物完整性：9 个 root NPZ + 9 个 MP4 均存在；所有日志均记录 final object tracking error。远端日志有 MuJoCo EGL teardown 的 ignored exception，但均发生在 NPZ/MP4 保存后，不影响产物。

## Full 结果

| case | ablation | contact | delta contact | obj mean | pelvis min | leg int | status | Phase B |
|---|---|---:|---:|---:|---:|---:|---|---|
| box004_082_p1 | baseline_ref_fk | 10.1% | +0.0% | 0.011m | 0.649m | 0.0% | FAIL | NO |
| box004_082_p1 | raw_mask_ref_fk | 54.1% | +44.0% | 0.011m | 0.642m | 0.0% | WORK | YES |
| box004_082_p1 | hold_band | 55.0% | +45.0% | 0.011m | 0.635m | 0.0% | WORK | YES |
| box021_035_p1 | baseline_ref_fk | 8.5% | +0.0% | 0.008m | 0.648m | 0.0% | FAIL | NO |
| box021_035_p1 | raw_mask_ref_fk | 77.5% | +69.0% | 0.008m | 0.638m | 3.1% | WORK | YES |
| box021_035_p1 | hold_band | 77.5% | +69.0% | 0.008m | 0.644m | 0.8% | WORK | YES |
| box026_135_p1 | baseline_ref_fk | 8.5% | +0.0% | 0.009m | 0.740m | 0.0% | FAIL | NO |
| box026_135_p1 | raw_mask_ref_fk | 17.1% | +8.5% | 0.010m | 0.585m | 0.0% | FAIL | NO |
| box026_135_p1 | hold_band | 26.8% | +18.3% | 0.009m | 0.676m | 0.0% | FAIL | NO |

Aggregate:

| ablation | mean contact | mean obj err | min pelvis | max leg int | work statuses |
|---|---:|---:|---:|---:|---|
| baseline_ref_fk | 9.1% | 0.009m | 0.648m | 0.0% | REVIEW+, REVIEW+, REVIEW+ |
| raw_mask_ref_fk | 49.6% | 0.010m | 0.585m | 3.1% | WORK, WORK, FAIL |
| hold_band | 53.1% | 0.009m | 0.635m | 0.8% | WORK, WORK, REVIEW+ |

## 结论

E112 Phase A 支持以下结论：

1. 接触和低穿透不是必然二选一。box004/box021 在保持 obj mean 约 0.8-1.1cm、leg interference 很低的同时，contact 从约 8-10% 提升到 54-78%，并从 FAIL 提升到 WORK。
2. 当前 Spider 接触下降主要是方法侧缺少 contact maintenance objective，而不是指标一定错误。raw mask 与 hold band 都能恢复大部分手物接触。
3. `hold_band` 比 `raw_mask_ref_fk` 更稳健。它在 box004/box021 达到同等或略高 contact，并且 box021 leg interference 低于 raw_mask；box026 也比 raw_mask 接触更高且 pelvis min 更安全。
4. box026 不是简单加 contact reward 就能解决。raw_mask 虽提升接触，但 pelvis min 降到 0.585m；hold_band contact 只有 26.8%，仍未到 WORK。box026 需要 surface-target/approach corridor 或 posture-aware contact schedule，而不应直接进入 Phase B 放大。

## Phase B 决策

进入 E113 的优先方案：

- 保留 `hold_band` 作为默认 contact-aware CEM 变体。
- `raw_mask_ref_fk` 作为对照保留，但要增加 lower-body safety guard，尤其关注 box021 的 `3.1%` leg interference。
- Phase B 先扩到 E109 的 20/24-case work set 中的 box004/box021/已有 positive-like cases，不把 box026 直接纳入 release candidate。
- box026 单独开诊断分支：比较 raw mask active window、surface target face、approach corridor、posture penalty 与 contact gain sweep。

数据侧配合：

- E111 的 `raw_contact_artifact_npz`、per-threshold mask 和 S6 contact alignment evaluator 已满足 E112/E113 最小需求。
- 后续 E113 需要 data_construction_v3 在 S5 handoff 中继续稳定输出 `contact_mask_*_npz`、`contact_label`、`contact_person_idx`、active/run-length 和 target gap diagnostics。
- 对 box026 这类接触弱提升 case，需要 S1/S3 增加 surface/face provenance 与 raw-contact target object-local coverage，便于区分“contact mask 时窗正确但 target surface 不对”和“mask 本身来自弱/偏 contact”。

## 验证

通过：

```bash
python3 -m py_compile workspace/core4d/scripts/E112/build_contact_aware_cem_manifest.py
python3 -m py_compile workspace/core4d/scripts/eval/eval_E112_contact_aware_cem.py
bash -n workspace/core4d/scripts/train/train_E112_contact_aware_cem.sh
bash -n workspace/core4d/scripts/run_E112_remote.sh
bash -n workspace/core4d/scripts/pull_E112_remote_results.sh
bash -n workspace/core4d/scripts/eval/eval_E112_contact_aware_cem.sh
bash workspace/core4d/scripts/eval/eval_E112_contact_aware_cem.sh smoke
bash workspace/core4d/scripts/eval/eval_E112_contact_aware_cem.sh full
```

`git diff --check` 在脚本生成和 smoke 前通过；E112 full 后仅新增结果和日志。
