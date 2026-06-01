# E107 计划：Box021 clean reconstruction 与 CEM-entry gate

日期：2026-06-01
上游：

- E103：`box021_person1/2` canonical source templates 已从 clean `box023_person1` base 重建；旧 Box021 runtime artifacts 已移出，不复用旧 dynamics/CEM label。
- E101：Box021 历史 CEM 0/4 WORK，但这些结果发生在 E103 template bug 修复之前，只能作为风险先验，不能作为 hard reject。
- D003：Box021 15 个 case-person 中 13 个 OmniRetarget + SPIDER preprocess pass，2 个 infeasible。

## 目标

重新构造 Box021 clean target tasks，然后重新过数据 gate，回答一个窄问题：修复 scene template bug 后，Box021 当前 D003 候选里是否有 case 可以进入 CEM。

本实验只做 data reconstruction + gate，不启动 CEM。若 gate 产出 `cem_ready=True`，下一步另开 E108 多卡 CEM。

## 候选范围

输入使用 D003 Box021 production summary：

- `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction/results/d003_omniretarget_spider_production/summary.json`
- 对象：`Box021`
- 目标：所有 `d003_box021_*` case-person
- 处理方式：
  - D003 `ok` 的 13 条：重建为 `{target_task}_e107_clean`
  - D003 `failed` 的 2 条：保留为 `preprocess_infeasible`，不伪造 target

## 验证声明

| claim | 验证方式 | 成功标准 |
|---|---|---|
| C1 clean source template 可用 | audit + MuJoCo load | `box021_person1/2` scene clean，`nq=43,nv=41,nu=29` |
| C2 target 重建不继承污染 scene | 从 clean source 复制 `scene.xml`，用旧 D003/SPIDER qpos 第一帧 patch object pose/quat，重新生成 `scene_act.xml` | 每个 rebuilt target 无 `29.632` robot inertial，scene/scene_act 均可 load |
| C3 重建轨迹保持数据一致性 | 对比旧 D003 `trajectory_kinematic.npz` 与 rebuilt qpos | shape 一致且 `np.allclose=True` |
| C4 gate 与现有 D002 raw-contact 规则对齐 | 直接读取 D002 raw_contact_proxy，按 `score_raw_contact_candidates_v2.py` 的 3cm/5cm pass/review/fail 阈值重算 | 输出 3cm/5cm 两档 gate；不使用旧 polluted CEM label |
| C5 可以明确判断是否进入 CEM | 汇总 D003 preprocess、raw contact、MuJoCo load、qpos、visual replay | `cem_ready=True/False` 和失败模式分类完整 |

## 实现

新增脚本目录：

- `workspace/core4d/scripts/E107/`

计划新增脚本：

- `build_box021_clean_gate.py`
  - 读取 D003 summary。
  - 对 D003 pass rows 生成 `{target_task}_e107_clean`。
  - 从 `box021_person1/2/scene.xml` 复制 clean source scene。
  - 用旧 D003 `0/trajectory_kinematic.npz` 的第一帧 `qpos[-7:]` patch object body `pos/quat`。
  - 复制 qpos 到新目录。
  - 运行 `generate_scene_act.generate_scene_act()` 重新生成 `scene_act.xml` 与 `scene_act_meta.json`。
  - 用 E083 `patch_scene_act()` 补 leg/foot + upper-body object collision pairs。
  - 审计 scene/scene_act inertial，验证 MuJoCo dims 与 qpos。
  - 读取 D002 raw_contact_proxy，按 3cm/5cm 输出 pass/review/fail。
  - 生成 `gate_summary.tsv/json/md`。
- `render_box021_clean_replays.py`
  - 对 rebuilt targets 生成 MuJoCo kinematic replay sheet/MP4。
  - 视频是数据重建检查证据，不作为 CEM success label。

## Gate 规则

每条 case-person 的最终 gate：

- `preprocess_infeasible`：D003 retarget failed，不能进入 CEM。
- `reconstruction_failed`：clean target 无法生成或 MuJoCo load/qpos 校验失败。
- `raw_contact_fail_3cm_5cm`：3cm 与 5cm 都不满足 D002 pass/review。
- `gate_review`：raw contact 至少一档为 review，但没有 pass；需要人工视觉复核后再决定。
- `cem_ready`：D003 pass + clean reconstruction pass + qpos一致 + MuJoCo load pass + raw contact 至少一档为 pass。

D002 raw-contact pass/review/fail 阈值严格采用当前脚本：

- pass：`target_both_active_frac >= 0.25` 且 `min(left,right) >= 0.35` 且 `partner_any_active_frac >= 0.25`
- review：`target_any_active_frac >= 0.40` 且 `min(left,right) >= 0.20` 且 `partner_any_active_frac >= 0.15`
- fail：其余

## 产物

| 类型 | 路径 |
|---|---|
| gate summary | `workspace/core4d/results/E107/box021_clean_gate_summary.tsv` |
| gate summary json | `workspace/core4d/results/E107/box021_clean_gate_summary.json` |
| markdown summary | `workspace/core4d/results/E107/box021_clean_gate_summary.md` |
| build metadata | `workspace/core4d/results/E107/box021_clean_build_meta.json` |
| replay visuals | `workspace/core4d/results/E107/visuals/box021_clean_replay/` |
| clean target dirs | `example_datasets/processed/core4d/unitree_g1/humanoid_object/*_e107_clean/` |

## 停止规则 / 下一步

- 若 `cem_ready=0`：记录失败模式分布，不启动 CEM，回到数据候选扩展或重定向修复。
- 若 `cem_ready>0`：E107 收尾后，另开 E108，按 `cem_ready` case 跑 full CEM，并先做人工/子代理 replay 审查。
