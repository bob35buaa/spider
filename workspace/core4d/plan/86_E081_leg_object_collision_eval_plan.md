# E081 Plan: leg/foot-object collision and eval audit

日期：2026-05-16

## Context

E080 二次复核说明：`box025_person2` 视觉上接近搬/扶箱，应作为 partial positive / near-usable，而不是简单失败；但当前 scene 只启用了 `left_hand_object`、`right_hand_object`、`object_floor`，没有腿/脚-箱 contact pair。离线几何复核发现 `box025_person2` 仍有轻中度腿/脚-箱穿入，`box023_person2` 是 E078/E079 已验证的高质量 guard。

用户要求：E081 试验加入腿和脚的箱体碰撞，并同步更新 eval 指标，先在 `box025_p2` 和 `box023_p2` 上做，本地 + 远程 GPU1 并行。

关键约束：**不直接修改原始 `scene_act.xml`**。E081 必须新建派生任务目录：

- `box025_person2_legobj`
- `box023_person2_legobj`

原始 `box025_person2` / `box023_person2` 目录只读作为 source。

## Claims

| Claim | 验证方式 | 成功/解释标准 |
|------|----------|---------------|
| C1 新 scene 不污染原始数据 | diff/路径检查 | 原始 `box025_person2/scene_act.xml`、`box023_person2/scene_act.xml` 不变；只新增 `*_legobj` 派生目录。 |
| C2 MuJoCo 中腿/脚-箱 contact 生效 | smoke/eval contact count | 派生 scene 中有腿/脚-`object_collision` pair；eval 能统计 `leg_object_contact_count`。 |
| C3 box025_p2 加碰撞后是否更可信 | CEM + 视频 + eval | 关注腿/脚穿入是否下降、箱体 lift/floor-contact 是否改善、是否仍保持手-箱接触和不摔倒。 |
| C4 box023_p2 guard 不被破坏 | CEM + eval | 作为已知高质量 guard，加碰撞后不应显著恶化接触/物体跟踪/稳定性。 |

## 改动

1. 新增 `workspace/core4d/scripts/E081/create_legobj_cases.py`
   - 从 source task 复制 `scene.xml`、`scene_act.xml`、`scene_act_meta.json`、`task_info.json`、`0/trajectory_kinematic.npz` 到派生 task。
   - 只在派生 task 的 `scene_act.xml` 里追加腿/脚-`object_collision` contact pairs。
   - 生成 provenance JSON。

2. 新增 `workspace/core4d/scripts/E081/generate_e081_overrides.py`
   - 生成 `core4d_E081_box025_p2_legobj.yaml`、`core4d_E081_box023_p2_legobj.yaml`。
   - 继承 E079 no-hold 口径，使用 E081 自己复制的 3cm masks。

3. 新增 E081 脚本
   - `workspace/core4d/scripts/E081/variants.tsv`
   - `workspace/core4d/scripts/run_E081_preprocess.sh`
   - `workspace/core4d/scripts/train/train_E081.sh`
   - `workspace/core4d/scripts/run_E081_remote.sh`
   - `workspace/core4d/scripts/pull_E081_remote_results.sh`
   - `workspace/core4d/scripts/eval/eval_E081.py`

4. Eval 新指标
   - `leg_box_sdf_min_m`
   - `leg_box_interference_frames_pct`
   - `leg_object_contact_frames_pct`
   - `object_floor_contact_frames_pct`
   - `object_bottom_proxy_m = object_z - object_half_z`
   - case-window 与 full-window 都记录。

## 运行安排

| Variant | Source task | Derived task | Split | GPU |
|---------|-------------|--------------|-------|-----|
| `E081_box025_p2_legobj` | `box025_person2` | `box025_person2_legobj` | local | 本机 GPU0 |
| `E081_box023_p2_legobj` | `box023_person2` | `box023_person2_legobj` | remote | `spider-remote` GPU1 |

## 命令

```bash
bash workspace/core4d/scripts/run_E081_preprocess.sh
bash workspace/core4d/scripts/train/train_E081.sh local 0
bash workspace/core4d/scripts/run_E081_remote.sh
REMOTE_HOST=spider-remote REMOTE_REPO=/home/xiayb/pHRI_workspace/spider bash workspace/core4d/scripts/pull_E081_remote_results.sh
.venv/bin/python workspace/core4d/scripts/eval/eval_E081.py
```

## 预期风险

- 加腿/脚碰撞可能让 CEM 更难优化，尤其 box025 大物体可能被腿碰撞约束卡住。
- 若 leg-object contact 很多，说明腿正在物理推/挡箱；这不是简单成功，需要结合视觉和 object lift 判断。
- `box023_p2` guard 若明显恶化，说明新增碰撞改变了原先可用解空间，需要后续更细 contact pair 或 collision margin 设计。
