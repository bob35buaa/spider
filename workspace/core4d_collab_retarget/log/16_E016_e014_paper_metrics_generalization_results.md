# E016 结果：E014 paper-aligned metrics + 13-case generalization

日期：2026-05-19

## 状态

E016 已完成：论文对齐指标、E014 复评、13 个 CORE4D case 派生资产、smoke wiring、3 卡并行 quick 泛化验证、13-case 总评和全量 13-case 离线可视化。

结论：E014 B-only 结构在 13/13 case 上保持 true-freejoint/config parity，并且 13/13 同时通过 SPIDER/DynaRetarget object success 与 transport success。这说明 kinematic support + soft weld 的 object 侧泛化成立。但 OmniRetarget-style robot-side 质量没有过门：contact preservation 仅 3/13，deep penetration ok 9/13，完整 generalization gate 0/13。失败主因不是 object tracking，而是 robot contact preservation、leg/floor shortcut 和局部 penetration artifact。

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E016_preprocess.sh --force
bash workspace/core4d_collab_retarget/scripts/train/train_E016.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E014.py --all

git commit -m "exp(core4d_collab_retarget): add E016 paper metrics generalization"
git push
git commit -m "exp(core4d_collab_retarget): harden E016 remote launch"
git push

bash workspace/core4d_collab_retarget/scripts/run_E016_remote.sh
VARIANTS_FILE=workspace/core4d_collab_retarget/results/E016/manifest_local_extra.tsv \
  E016_QUICK_NUM_SAMPLES=128 E016_QUICK_MAX_ITERS=4 \
  bash workspace/core4d_collab_retarget/scripts/train/train_E016.sh quick 0
bash workspace/core4d_collab_retarget/scripts/pull_E016_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E016.py --all
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/render_E016_visuals.py --force
```

3 卡分配：远端 GPU0/GPU1 跑 8 个非本地结果；本地 GPU 先跑 `box023_p2/box025_p2`，再接手 `bucket005_s2_p2/bucket007_p2/desk021_p1`。远端在目标 8 个结果完成后已停止，避免重复跑本地接手 case。

## 结果路径

| 产物 | 路径 |
|------|------|
| Plan | `workspace/core4d_collab_retarget/plan/16_E016_e014_paper_metrics_generalization_plan.md` |
| Results | `workspace/core4d_collab_retarget/results/E016/` |
| Comparison | `workspace/core4d_collab_retarget/results/E016/comparison.csv` |
| Aggregate | `workspace/core4d_collab_retarget/results/E016/aggregate_summary.json` |
| Manifest | `workspace/core4d_collab_retarget/results/E016/manifest.tsv` |
| Visual renderer | `workspace/core4d_collab_retarget/scripts/eval/render_E016_visuals.py` |
| Visual index | `workspace/core4d_collab_retarget/results/E016/visual/visual_eval.md` |
| Visual videos | `workspace/core4d_collab_retarget/results/E016/visual/*_comparison.mp4` |
| Visual sheets | `workspace/core4d_collab_retarget/results/E016/visual/*_frames/sheet.jpg` |

## Aggregate

```json
{
  "num_results": 13,
  "num_config_ok": 13,
  "num_paper_spider_success": 13,
  "num_paper_dynaretarget_success": 13,
  "num_transport_success": 13,
  "num_contact_preservation_ok": 3,
  "num_deep_penetration_ok": 9,
  "num_artifact_ok": 0,
  "num_generalization_pass": 0,
  "mean_paper_Epos_case_m": 0.04985,
  "mean_paper_Erot_case_deg": 4.08,
  "mean_carry_progress_ratio_case": 1.001,
  "mean_contact_preservation_5cm_pct": 39.62,
  "mean_deep_penetration_duration_pct": 21.77,
  "diagnostic_classes": {
    "contact_preservation_gap": 10,
    "push_or_leg_shortcut": 2,
    "artifact_failed": 1
  }
}
```

## Per-case 摘要

| Variant | Epos m | Erot deg | progress | contact 5cm % | deep pen % | leg % | 诊断 |
|---------|--------|----------|----------|---------------|------------|-------|------|
| `E016_box021_p1` | 0.098 | 12.9 | 1.030 | 77.8 | 77.2 | 37.9 | push/leg shortcut |
| `E016_box021_p2` | 0.061 | 4.0 | 1.001 | 0.9 | 0.0 | 13.2 | contact gap |
| `E016_box023_p1` | 0.044 | 1.6 | 1.000 | 12.9 | 0.0 | 0.0 | contact gap |
| `E016_box023_p2` | 0.041 | 1.7 | 0.998 | 4.4 | 0.0 | 0.0 | contact gap |
| `E016_box025_p1` | 0.056 | 3.1 | 0.990 | 28.6 | 0.0 | 3.4 | contact gap |
| `E016_box025_p2` | 0.059 | 2.3 | 0.992 | 57.5 | 0.0 | 2.3 | contact gap |
| `E016_bucket001_p1` | 0.033 | 2.6 | 0.996 | 0.0 | 0.0 | 0.0 | contact gap |
| `E016_bucket001_p2` | 0.041 | 7.0 | 1.001 | 26.4 | 5.6 | 34.3 | contact gap |
| `E016_bucket005_s2_p1` | 0.038 | 5.1 | 1.007 | 97.3 | 89.1 | 34.6 | push/leg shortcut |
| `E016_bucket005_s2_p2` | 0.037 | 4.4 | 1.000 | 93.9 | 62.6 | 13.8 | artifact failed |
| `E016_bucket007_p1` | 0.054 | 4.1 | 0.999 | 33.2 | 20.1 | 0.0 | contact gap |
| `E016_bucket007_p2` | 0.041 | 2.8 | 0.999 | 28.7 | 15.8 | 21.7 | contact gap |
| `E016_desk021_p1` | 0.046 | 1.4 | 1.002 | 53.5 | 12.6 | 0.0 | contact gap |

## 可视化

离线 EGL 已渲染 13/13 个 case，使用 `workspace/hdmi_reproduce/scripts/render_trajectory_video.py` 生成 side-by-side comparison：左侧为 kinematic/reference qpos，右侧为 MJWarp physics output。视频、contact sheet 与指标索引见 `results/E016/visual/visual_eval.md`。

全量可视化确认：object 轨迹在多数 case 中能跟住 reference，因此 SPIDER/Dyna object success 与 transport success 不是假阳性；失败集中在 robot-side 接触形态。代表性观察如下：

- `E016_box025_p2`: object 跟随参考箱体运动较好，后段姿态没有大旋转；但手端相对参考接触位置偏离，符合 contact preservation 不过门。
- `E016_box021_p1`: 后段机器人明显下探，箱体附近存在推挤/腿部干涉风险，符合 `leg=37.9%` 与 deep penetration `77.2%`。
- `E016_bucket005_s2_p2`: 桶的全局轨迹接近参考，但机器人身体与桶贴合过深，符合 contact preservation 高但 deep penetration `62.6%` 的 artifact 诊断。

## Claims 验证

| Claim | 结论 | 证据 |
|-------|------|------|
| C1 E014 paper-aligned 指标能在已有 E014 full 上稳定产出 | ✅ 通过 | E014 复评 `SPIDER/Dyna/transport=6/6`，deep penetration ok `6/6` |
| C2 指标能识别 object 之外的 artifact | ✅ 通过 | E016 `object/transport=13/13`，但 contact/deep/leg/floor 将 13 个全部拦下 |
| C3 E014 B-only scene/override 生成泛化到 10+ case | ✅ 通过 | 13 个 case 均 `config_ok=13/13`，smoke 与 quick 均可运行 |
| C4 能区分结构泛化失败和 robot contact 失败 | ✅ 通过 | 诊断集中在 `contact_preservation_gap=10`、`push_or_leg_shortcut=2`、`artifact_failed=1`，不是 config/object tracking failure |

## 结论

E014 B-only 可作为 object-side pipeline candidate：在 13 个 case 上，object `Epos/Erot`、SPIDER/Dyna success 和 carry progress 都稳定通过。它不能被声明为完整 retargeting 泛化成功，因为 OmniRetarget-style contact preservation / penetration / leg shortcut 仍不达标。

下一步不应继续调 weld 本身；应转向 robot-side contact policy/reward 或 support anchor/person-side selection：在保持 E014 object constraint 的前提下，提高 contact preservation，并对 hand-object shallow intersection 与 severe penetration 做更细的分层约束。
