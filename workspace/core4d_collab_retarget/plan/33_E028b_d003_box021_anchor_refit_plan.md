# E028b 实验计划：D003 Box021 candidate anchor refit

日期：2026-05-27

## Context

E028 已把 Holosoma D003 Box021 的 13 条 `进入下一轮` case 接入 SPIDER dynamic retarget，但 strict clean 为 `0/13`。后续用户指定只看：

`workspace/core4d_collab_retarget/results/E028/candidates.json`

当前候选只有 5 条：

- `E028_d003_box021_20231011_034_p1_canonical_t02`
- `E028_d003_box021_20231011_035_p1_canonical_t02`
- `E028_d003_box021_20231011_035_p2_canonical_t02`
- `E028_d003_box021_20231018_029_p2_canonical_t02`
- `E028_d003_box021_20231020_019_p1_canonical_t02`

E028 candidate anchor visualization 已显示：

- 4/5 face selection 本身不稳，主要是 `ambiguous_side_face_margin` 或 `weak_side_face_support`。
- 即使唯一 face 清晰的 `20231018_029_p2`，旧 canonical anchor 与 selected-face contact cloud centroid 仍偏 `0.415m`，Z 方向偏 `0.330m`。
- 当前 E028 固定 `canonical_z=0.62 * half_z`、face 面中心 `other_axis=0` 的策略不适合这批 D003 Box021 candidate。

因此 E028b 不继续重跑旧 anchor，而是先把 support proxy anchor 改成 contact-cloud driven 的 projected centroid，再做可视化和 dynamic rollout 复核。

## Claims

| Claim | 最低证据 |
|---|---|
| C1: E028b 分母严格等于 E028 `candidates.json` 的 5 条 | `results/E028b_anchor_refit/manifest.tsv` 只有 5 行，且 `source_e028_variant` 全部来自 candidates |
| C2: 新 anchor 不再使用固定 `canonical_z=0.62` | manifest 中 `anchor_policy=e028b_contact_centroid_projected`，`support_proxy_point_local` 的 free axes 来自 selected-face contact cloud robust centroid |
| C3: 新 anchor 几何合法 | 5/5 anchor 位于 selected side face surface；另两个自由轴在 object half extent 内；scene XML 维度保持 `nq/nv/nu=43/41/29` |
| C4: 新 anchor 相比 E028 更贴近 selected-face contact cloud | 5/5 `anchor_to_selected_face_centroid_m` 小于 E028 candidate baseline，目标至少降低 `>=40%` |
| C5: 可视化必须覆盖并被分析 | 输出 5 条 anchor cloud / motion MP4 / sheet；使用 `video-frames` 抽帧；high reasoning subagent 对可视化做独立分析，结果写入 log |
| C6: dynamic pipeline 不退化为 object actuator 或 direct wrench | smoke/full/eval 显示 `contact_guidance=false`、`object_action_dims=0`、`support_proxy_mode=mocap_pad`、`no_direct_wrench=true` |

## 改动

### 1. Manifest / anchor policy

新增 `workspace/core4d_collab_retarget/scripts/E028b/build_e028b_manifest.py`：

- 读取 E028 `candidates.json` 和 E028 `manifest.tsv`。
- 对每条 candidate 重建 object-local active contact cloud，active mask 与 E028 一致：`trajectory_kinematic.contact` 与 D003 `spider_contact_mask_3cm` 取 union。
- 保留 E028 的 dominant side face 选择逻辑，但 anchor 点从固定 canonical center 改成：
  - 取 selected side face 上 contact points 的 trimmed median / robust centroid；
  - face axis 投影到 object surface：`±half_axis`；
  - 另两个自由轴按 robust centroid 取值，并 clamp 到 `0.90 * half_extent`，避免落到边缘外；
  - 若 selected-face points 过少，则 fallback 到 E028 anchor 并强制 `anchor_face_review=true`。
- manifest 额外记录旧 anchor、new anchor、selected-face centroid、old/new centroid distance、distance ratio。

### 2. Assets / overrides

复用 E028 的 asset 和 override generator，但输入改为 E028b manifest：

- `workspace/core4d_collab_retarget/scripts/E028/generate_e028_assets.py --manifest results/E028b_anchor_refit/manifest.tsv`
- `workspace/core4d_collab_retarget/scripts/E028/generate_e028_overrides.py --manifest results/E028b_anchor_refit/manifest.tsv --result-root results/E028b_anchor_refit`

新增 wrapper：

- `workspace/core4d_collab_retarget/scripts/run_E028b_preprocess.sh`
- `workspace/core4d_collab_retarget/scripts/train/train_E028b.sh`
- `workspace/core4d_collab_retarget/scripts/eval/eval_E028b.py`

### 3. 可视化

增强 `render_E028_anchor_visuals.py`，支持通过 CLI 指定 manifest/candidates/out-dir，然后用于 E028b：

- `results/E028b_anchor_refit/anchor_visual/*_anchor_cloud.png`
- `results/E028b_anchor_refit/anchor_visual/*_anchor_motion.mp4`
- `results/E028b_anchor_refit/anchor_visual/*_anchor_motion_sheet.jpg`
- `results/E028b_anchor_refit/anchor_visual/anchor_summary.csv`

此外对代表视频使用 `video-frames` 抽帧，避免只看静态图。

### 4. 可视化分析 subagent

用户明确要求交给 subagent，模型 high。执行方式：

- spawn 一个 high reasoning subagent。
- 输入 E028 old montage、E028b new montage、E028b summary、代表 motion sheet/frame。
- 要求 subagent 独立回答：新 anchor 是否比旧 anchor 更合理、哪些 case 仍然不可作为 clean、是否值得进入 full dynamic rollout。
- 将 subagent 输出原文或摘要写入 E028b log 的「可视化 → subagent 分析」。

### 5. Dynamic rollout / eval

固定只跑 5 个 E028b candidate variants：

```bash
cd /home/ubuntu/Workspace/spider
bash workspace/core4d_collab_retarget/scripts/run_E028b_preprocess.sh --force
bash workspace/core4d_collab_retarget/scripts/train/train_E028b.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/train/train_E028b.sh local 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E028b.py --all
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/index_E028b_online_videos.py --force
```

本轮不把 5 条以外的 D003 Box021 case 加回分母。

## 成功标准

| 指标 | 最低标准 |
|---|---:|
| Manifest rows | `5/5` |
| Anchor geometry pass | `5/5` |
| Anchor distance improvement | `5/5` old distance 降低，mean distance ratio `<=0.60` |
| Smoke | `5/5` |
| Full artifact count | root NPZ / outdir trajectory / online MP4 均 `5/5` |
| Config parity | `5/5` true-freejoint support proxy，无 object actuator/direct wrench |
| Clean dynamic success | 不设硬性通过数；本实验首要验证 anchor 修复是否成立 |

## Stop / next

- 如果可视化显示新 anchor 仍明显错位，不进入 full rollout，先修 anchor policy。
- 如果 anchor 改善但 full 仍失败，则后续结论应从 anchor 问题转向 robot-side stability/contact，而不是重复 anchor sweep。
- 如果某 case face selection 仍严重 ambiguous，应标成 boundary，不作为 clean denominator。
