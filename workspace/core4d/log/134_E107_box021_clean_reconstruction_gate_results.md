# E107：Box021 clean reconstruction 与 CEM-entry gate 结果

日期：2026-06-01
计划：`workspace/core4d/plan/114_E107_box021_clean_reconstruction_gate_plan.md`
上游：E103 source template rebuild；D003 Box021 OmniRetarget + SPIDER preprocess summary

## 目标

E103 修复 Box021/Box022/Box026 scene template bug 后，Box021 历史 D003/E101 结果不能作为 hard label。本实验重新从 clean `box021_person1/2` source templates 构造 Box021 target tasks，并重新跑 gate，判断是否有 case 能进入 CEM。

E107 只做数据重建和 gate，不启动 CEM。

## 执行内容

新增脚本：

- `workspace/core4d/scripts/E107/build_box021_clean_gate.py`
- `workspace/core4d/scripts/E107/render_box021_clean_replays.py`

执行结果：

- 读取 D003 Box021 production summary：15 个 case-person。
- D003 `ok` 的 13 条全部重建为 `{target_task}_e107_clean`。
- D003 `failed` 的 2 条保持 `preprocess_infeasible`，不伪造 target。
- 每个 rebuilt target：
  - 从 clean `box021_person1/2/scene.xml` 复制 source scene。
  - 用旧 D003/SPIDER `trajectory_kinematic.npz` 第一帧 patch object `pos/quat`。
  - 复制 qpos 到新 target 的 `0/trajectory_kinematic.npz`。
  - 重新生成 `scene_act.xml` / `scene_act_meta.json`。
  - 补 leg/foot + upper-body object collision pairs。
  - 校验 scene/scene_act MuJoCo load、robot inertial、qpos 一致性。
  - 读取 D002 raw_contact_proxy，按当前 D002 3cm/5cm pass/review/fail 阈值重新打 gate。

## 产物

| 类型 | 路径 |
|---|---|
| gate summary TSV | `workspace/core4d/results/E107/box021_clean_gate_summary.tsv` |
| gate summary JSON | `workspace/core4d/results/E107/box021_clean_gate_summary.json` |
| gate summary MD | `workspace/core4d/results/E107/box021_clean_gate_summary.md` |
| build metadata | `workspace/core4d/results/E107/box021_clean_build_meta.json` |
| replay review | `workspace/core4d/results/E107/visuals/box021_clean_replay/REVIEW.md` |
| replay manifest | `workspace/core4d/results/E107/visuals/box021_clean_replay/render_manifest.json` |

## Gate 结果

| failure_mode | count | 说明 |
|---|---:|---|
| `cem_ready` | 13 | D003 pass + clean reconstruction pass + qpos match + MuJoCo load pass + 3cm/5cm raw-contact pass |
| `preprocess_infeasible` | 2 | D003 OmniRetarget/SPIDER preprocess infeasible；raw contact 通过也不能进 CEM |

关键汇总：

- `15/15` rows 有 D002 raw_contact_proxy。
- `15/15` rows 在 3cm raw-contact 下为 `raw_contact_pass`。
- `15/15` rows 在 5cm raw-contact 下为 `raw_contact_pass`。
- `13/13` rebuilt rows scene load 为 `nq=43,nv=41,nu=29`。
- `13/13` rebuilt rows scene_act load 为 `nq=42,nv=41,nu=35`。
- `13/13` rebuilt rows `qpos_matches_legacy_spider=True`。
- `13/13` rebuilt rows scene/scene_act 均无 `robot_polluted_mass_29_632`。
- `13/13` rebuilt rows leg/foot + upper-body object collision pairs 完整。

## CEM-ready case

| derived task | frames | raw 3cm both | raw 5cm both |
|---|---:|---:|---:|
| `d003_box021_20231011_034_p1_e107_clean` | 143 | 0.9425 | 0.9540 |
| `d003_box021_20231011_035_p1_e107_clean` | 129 | 0.9867 | 1.0000 |
| `d003_box021_20231011_035_p2_e107_clean` | 133 | 0.9600 | 0.9733 |
| `d003_box021_20231018_028_p2_e107_clean` | 92 | 1.0000 | 1.0000 |
| `d003_box021_20231018_029_p1_e107_clean` | 71 | 0.8750 | 0.9062 |
| `d003_box021_20231018_029_p2_e107_clean` | 75 | 0.8750 | 0.8750 |
| `d003_box021_20231018_030_p1_e107_clean` | 88 | 0.9167 | 0.9444 |
| `d003_box021_20231018_030_p2_e107_clean` | 75 | 0.9722 | 0.9722 |
| `d003_box021_20231018_031_p2_e107_clean` | 107 | 1.0000 | 1.0000 |
| `d003_box021_20231020_019_p1_e107_clean` | 98 | 0.9211 | 0.9211 |
| `d003_box021_20231020_019_p2_e107_clean` | 102 | 0.9474 | 0.9474 |
| `d003_box021_20231020_020_p1_e107_clean` | 87 | 1.0000 | 1.0000 |
| `d003_box021_20231020_020_p2_e107_clean` | 87 | 1.0000 | 1.0000 |

## 不进 CEM

| target task | reason |
|---|---|
| `d003_box021_20231011_034_p2` | `CVXPY solve failed: infeasible` |
| `d003_box021_20231018_028_p1` | `CVXPY solve failed: infeasible` |

这两条 raw contact 本身通过，但没有可用 D003/SPIDER qpos，不能通过 E107 target reconstruction gate。

## 可视化

输出：

- 13 张 replay sheet。
- 52 张 keyframe PNG。
- 13 个 MuJoCo kinematic replay MP4。

文件完整性检查：

- `render_manifest.json` 记录 13 条 rebuilt target。
- 13/13 sheet 存在。
- 13/13 MP4 存在且非零。
- 抽查 `d003_box021_20231018_030_p2_e107_clean_kinematic_replay.mp4`：`960x720`、`24fps`、`75 frames`、`3.125s`。

实际观察：

- replay sheet 中机器人与 Box021 物体均可见，目标物体随轨迹移动，画面非空。
- 抽查 `d003_box021_20231018_030_p2_e107_clean`：四个关键帧显示机器人围绕 Box021 执行搬运/放置动作；右侧 panel 显示 source clean、rebuild、qpos match、scene/scene_act dims、raw-contact gate 均通过。
- 这些 replay 是 kinematic/data validation 证据，不是 CEM dynamics success label。

## 验证声明复核

| claim | 状态 | 说明 |
|---|---|---|
| C1 clean source template 可用 | PASS | `box021_person1/2` source clean；13 个 target 均从 clean source 重建 |
| C2 target 重建不继承污染 scene | PASS | 13/13 scene/scene_act 无 `29.632` robot inertial |
| C3 重建轨迹保持数据一致性 | PASS | 13/13 `qpos_matches_legacy_spider=True` |
| C4 gate 与当前 D002 raw-contact 规则对齐 | PASS | 3cm/5cm 均按 D002 阈值重算，15/15 raw-contact pass |
| C5 明确判断是否进入 CEM | PASS | 13 条 `cem_ready`，2 条 `preprocess_infeasible` |

## 结论

Box021 在 E103 template bug 修复后并不是“没有 case 能进 CEM”。相反，D003 里已有 13 条可 clean reconstruction 且通过 3cm/5cm raw-contact gate 的 `cem_ready` target。

下一步应另开 E108，优先对这 13 条 `_e107_clean` target 跑 full CEM。E101 的旧 Box021 CEM 失败只能作为“Box021 dynamics 可能仍难”的风险先验，不能再作为阻止 E107 clean cases 进入 CEM 的数据门依据。
