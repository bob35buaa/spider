# E094 Plan: G1-Handbox-Aware Contact Target Projection

日期：2026-05-29

## Context

E093 证明当前 `ref_fk + wrist_yaw_link + 5cm` contact target 不是 raw CORE4D contact 的小误差 proxy：

- Box026/Box025 的 `wrist5->raw mean` 达 `49-64cm`。
- Box026 039 的 support face 覆盖仅 `19.5/15.4%`，CEM full 低髋失败。
- Box026 135 有 right-inside `12.2%`，CEM full 中 RH floor `17.1%`。
- handbox proxy 是 14/14 行相对最接近 raw 的几何，但单独替换 proxy 不够，必须配套 target face assignment。
- box004/box023 能 work 的共同点是 no-inside、较小物体、偏差仍落在可达外侧/上沿附近。

E094 的目的不是直接调 reward 权重，而是把 raw surface contact 映射成 G1 wrist/handbox 可达的 support patch，并生成 MJWP 可直接消费的 external contact target。

## Hypothesis

H1. 对 Box026/D003 这类失败 case，把 raw target 自适应投到 world-up support face，可以显著降低 wrong-face / inside / low-support 指标。

H2. 这个 projection 对 box004/box023 positive guards 不应破坏 no-inside 和可达性；如果 guard 也变差，说明 projection 不是可泛化修复。

H3. 若 E094 kinematic gate 通过，三条 E092 case 可以用同一 external-target CEM 脚本跑 full CEM；成功标准仍按 E092 full WORK gate，不因为 target repair 降低动态质量要求。

## Scope

### Kinematic + visualization gate

覆盖 5 条诊断 case：

| case | task | role |
|---|---|---|
| `box023_p2` | `box023_person2` | known positive guard |
| `box004_083_p2` | `e091_box004_20231003_2_083_p2` | E092 C1 WORK / positive guard |
| `box021_d003_029_p2` | `d003_box021_20231018_029_p2` | known D003 fail diagnostic |
| `box026_039_p2` | `e091_box026_20231018_039_p2` | E092 C2 FAIL low-support |
| `box026_135_p2` | `e091_box026_20231020_135_p2` | E092 C3 FAIL inside-risk |

### CEM gate

只跑 E092 三条 CEM case：

| case | variant | split |
|---|---|---|
| C1 box004 | `E094P1_box004_083_p2_hbproj` | local GPU0 |
| C2 box026 039 | `E094P2_box026_039_p2_hbproj` | remote GPU0 |
| C3 box026 135 | `E094P3_box026_135_p2_hbproj` | remote GPU1 |

当前本地/远程 GPU 上已有 RL 训练时，不 kill 进程；CEM 直接叠加跑。

## Target Projection

每帧每只手生成三类 object-local 点：

1. `support_patch_object_local`: raw surface target 投到当前物体姿态的 world-up support face，保留 raw 的两个切向坐标并 clamp 到 collision box 内。
2. `reward_target_object_local`: 给 `run_mjwp.py` 使用的 external target。最终候选为 `adaptive_support`：
   - 非 raw-active 帧保留旧 `ref_fk + wrist+5cm`，避免无接触阶段 target 跳变。
   - raw-active 帧若旧 target 不 inside 且已经是 support face，或旧 target 到 raw 距离 `<=0.30m`，保留旧 target。
   - 只有旧 target inside、非 support 且离 raw 太远时，才改用 `support_patch_object_local`。
3. `handbox_closest_object_local`: reference handbox 到 support patch 的最近点，仅用于诊断 handbox/patch gap，不直接作为当前 reward target。

首版被否决的候选为 `handbox_compensated`：

```text
reward_target_world = support_patch_world + (wrist5_world - closest_point_on_handbox_to_support_patch_world)
```

这个公式在首轮 gate 中会把 positive guard 和部分 fail case 的 reward target 拉动过大或判定 inside，因此只保留为对照产物，不进入 CEM。

## Artifacts

输出根目录：`workspace/core4d/results/E094/handbox_target_projection/`

| artifact | 内容 |
|---|---|
| `projection_summary.{csv,json,md}` | 每 case/hand 的 old vs projected gate 指标 |
| `per_frame_projection.csv` | raw、old wrist5、support patch、reward target、delta、face/inside/support |
| `targets/*.npz` | `spider_contact_target_object_local` / `eval_contact_target_object_local` external target |
| `visuals/object_local/*_projection.png` | object-local overlay |
| `visuals/timeline/*_projection_timeline.png` | inside/support/delta 时间线 |
| `visuals/mujoco/keyframes/*_projection_keyframes.png` | full-body MuJoCo reference + old/new markers |
| `visuals/mujoco/videos/*_projection.mp4` | MuJoCo projection video |
| `visual_review/high_subagent_review.md` | high subagent 视觉复核 |

若进入 CEM：

| artifact | 内容 |
|---|---|
| `workspace/core4d/scripts/E094/variants.tsv` | E094 CEM variants |
| `examples/config/override/core4d_E094*.yaml` | external-target overrides |
| `workspace/core4d/results/E094/cem/full/` | full CEM npz/mp4/keyframes/eval |
| `logs/E094/cem/full/` | full CEM logs |

## Implementation Plan

1. 新增 `workspace/core4d/scripts/E094/build_handbox_target_projection.py`
   - 读取 E093 `case_manifest.tsv` 与 raw target 生成逻辑。
   - 生成 support patch、adaptive-support reward target、per-frame CSV、NPZ target、2D plots。
   - 计算 projection gate：old/new support、inside、reward delta、handbox-to-patch gap、filled frames。

2. 新增 `workspace/core4d/scripts/E094/render_projection_mujoco.py`
   - full-body `auto` camera 复用 E093 思路。
   - 渲染 old `wrist+5cm`、raw、support patch、new reward target、handbox。

3. 新增 `workspace/core4d/scripts/E094/build_cem_tasks.py`
   - 对 C1/C2/C3 写 variants 和 overrides。
   - overrides 继承 E092 Stage A dyn 配置，仅把 `contact_hdmi_target_source` 改成 `external`。

4. 新增 `workspace/core4d/scripts/train/train_E094_handbox_proj_cem.sh`
   - 支持 `list/local/remote-gpu0/remote-gpu1/single/eval` 与 `smoke/full`。
   - 若 kinematic gate 通过，优先直接 full；本地 C1，远程 GPU0 C2，远程 GPU1 C3。

5. 新增 `workspace/core4d/scripts/run_E094_remote.sh` 与 pull 脚本
   - 远程 clean clone 若主 repo 有本地改动，沿用 E092 的 `/home/xiayb/pHRI_workspace/spider_e092_run` 风格，不 reset 用户工作树。

6. 可视化复核
   - 先本地检查 PNG/MP4 非空、mp4 decode、抽关键帧。
   - 使用 high subagent 复核 projection/CEM 可视化，报告写入结果目录。

## Success Criteria

Kinematic gate:

1. 5 条 diagnostic case 全部生成 target NPZ、overlay、timeline、MuJoCo keyframes/video。
2. C1/box023 guard 的 new target 不引入明显 inside：per-hand `reward_inside_frac <= 5%`，`reward_delta_p90 <= 0.35m`。
3. C2/C3 的 support-patch support fraction 显著提升到 `>=80%`，且 reward target 不出现大面积 inside。
4. 视频视觉上能看到完整机器人/箱子/old-new target，不允许只看上半身或 marker 不可见。

CEM gate:

1. 三条 E094 full CEM 都完成并有 eval summary。
2. `WORK` 判定沿用 E092 full 标准：T>=80、obj mean<=0.10、obj max<=0.30、pelvis>=0.55、head/upper/floor<=5%、contact>=30%。
3. 若 C2/C3 仍失败，log 必须明确失败是 target projection 无效、姿态/低髋局部解、还是 handbox/sphere physics mismatch。

## Commands

```bash
python workspace/core4d/scripts/E094/build_handbox_target_projection.py --force
python workspace/core4d/scripts/E094/render_projection_mujoco.py --video-frames 48
python workspace/core4d/scripts/E094/build_cem_tasks.py --force
bash workspace/core4d/scripts/train/train_E094_handbox_proj_cem.sh local full 0
```

远程 full CEM 只在 kinematic gate 通过后启动：

```bash
git push
ssh spider-remote 'cd /home/xiayb/pHRI_workspace/spider_e092_run && git pull'
ssh spider-remote 'cd /home/xiayb/pHRI_workspace/spider_e092_run && tmux new-session -d -s E094_hbproj_full "bash workspace/core4d/scripts/run_E094_remote.sh full"'
```

## Non-goals

- 不在 E094 初始实现里改 MJWP reward kernel。
- 不把 raw target 直接当 external target 重跑；E085 已证明这不是正确方向。
- 不启动 Holosoma RL；用户这里的 CEM 指的是 SPIDER/MJWP dynamic retargeting。RL 后续要另开计划。
