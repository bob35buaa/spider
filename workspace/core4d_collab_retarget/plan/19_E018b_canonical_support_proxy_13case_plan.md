# E018b Plan: canonical support proxy 13-case generalization

日期：2026-05-19

## Context

E018 已在两个 E014 GT case 上通过 gate：

- `box023_p2`: canonical `[0.1531, 0, 0.109492]`，到 E014 GT `[0.16, 0, 0.10]` 距离 `1.17cm`。
- `box025_p2`: canonical `[0, 0.3778, 0.290904]`，到 E014 GT `[0, 0.38, 0.30]` 距离 `0.94cm`。
- 2/2 `config_ok`、2/2 `soft_target_pass`、2/2 SPIDER/Dyna/transport success。

E018b 扩展到 E016 的 13 个 case。用户要求：

- 直接用在线 rollout 视频，不再依赖离线 replay/离线 comparison render。
- 本地 1 卡 + 远程 2 卡，共 3 卡并行。

## Claims

| ID | Claim | 验证方式 |
|----|-------|----------|
| C1 | E018b 对 13 个 E016 case 都生成 canonical support proxy anchor | manifest 13/13；每个 anchor 在 face center，`z=0.62*half_z`，非 COM |
| C2 | GT/template case 保持 E018 验证过的语义 | `box023_p2` 使用 `+x`，`box025_p2` 使用 `+y`；与 E014 GT dist 仍 `<0.03m` |
| C3 | 无 GT case 不再使用 palm median point | face 可以来自 E017 audit weak evidence，但 point placement 统一 canonical，禁止负 z 与 tangential median offset |
| C4 | 三卡并行执行 | manifest 分配 `local` / `remote_gpu0` / `remote_gpu1`，本地和远端各自顺序跑 queue |
| C5 | 在线视频覆盖 13/13 | 每个 variant 由 `run_mjwp.py video_output_path` 直接产出 mp4；只做 ffmpeg sheet/index，不做离线 replay |
| C6 | 评测沿用 E016 paper-aligned 指标并分层汇报 | SPIDER/Dyna object gate、transport、OmniRetarget contact/penetration/foot skating、leg/floor shortcut、anchor audit 分层 |

## 改动

1. 新增 `scripts/E018b/generate_e018b_assets.py`
   - 读取 `scripts/E016/variants.tsv` 13 cases。
   - 读取 `results/E017/anchor_audit.csv` 的 `selected_face` 作为弱证据。
   - 对 E014 GT case 覆盖 face：`box023_person2=+x`，`box025_person2=+y`。
   - 生成 canonical point：`face center + 0.62*half_z`。
   - 生成 scene XML：`scene_e018b_jointB_*_canonical_t02.xml`。
2. 新增 `scripts/E018b/generate_e018b_overrides.py`
   - 生成 `examples/config/override/core4d_collab_E018b_*.yaml`。
3. 新增 `scripts/run_E018b_preprocess.sh`。
4. 新增 `scripts/train/train_E018b.sh`
   - 支持 `smoke`、`local`、`remote_gpu0`、`remote_gpu1`、`one`、`eval`。
   - 直接保存在线视频到 `results/E018b/online_video/<variant>.mp4`。
5. 新增 `scripts/run_E018b_remote.sh`
   - 远程 GPU0/GPU1 并行跑对应 queue。
6. 新增 `scripts/eval/eval_E018b.py`
   - 复用 E018/E017/E016 paper metrics 口径。
   - 添加 canonical anchor/audit fields 和分层 pass。
7. 新增 `scripts/eval/index_E018b_online_videos.py`
   - 只从在线视频抽 sheet，写 `online_video/online_video_eval.md`。

## 3-card queue

按参考帧数做 greedy 分配，目标是 13 case 在 3 张卡上尽量均衡。实际 queue 写入 manifest 后确认。

## 成功标准

| 层级 | 标准 |
|------|------|
| Setup | preprocess + py_compile + smoke 13/13 成功 |
| Runtime | 13/13 full 产出 NPZ + 在线 MP4 |
| Config | 13/13 `E018b_config_ok=true` |
| Object | SPIDER/Dyna object success 与 transport 数量不低于 E016 |
| Artifact | 汇报 contact preservation、deep penetration、foot skating、floor/leg shortcut，不要求一次性全过 |
| Decision | 明确哪些 case 是 anchor 修复有效、哪些仍是 robot-side contact/artifact 问题 |

## 命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E018b_preprocess.sh --force
bash workspace/core4d_collab_retarget/scripts/train/train_E018b.sh smoke 0

# 3-card full
bash workspace/core4d_collab_retarget/scripts/train/train_E018b.sh local 0
ssh spider-remote "cd /home/xiayb/pHRI_workspace/spider && bash workspace/core4d_collab_retarget/scripts/run_E018b_remote.sh"

# 回收远程结果后本地统一评测
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E018b.py --all
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/index_E018b_online_videos.py --force
```

## 结果路径

| 产物 | 路径 |
|------|------|
| Plan | `workspace/core4d_collab_retarget/plan/19_E018b_canonical_support_proxy_13case_plan.md` |
| Log | `workspace/core4d_collab_retarget/log/19_E018b_canonical_support_proxy_13case_results.md` |
| Results | `workspace/core4d_collab_retarget/results/E018b/` |
| Online videos | `workspace/core4d_collab_retarget/results/E018b/online_video/` |
| Logs | `logs/core4d_collab_retarget/E018b/` |
