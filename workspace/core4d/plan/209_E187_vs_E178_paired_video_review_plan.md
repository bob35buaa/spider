# E187 vs E178 Paired Video Review 计划

## Context

- E187 Evaluation已完成22/22 paired numeric comparison，用户要求补充E178（左）vs E187（右）可视化视频，并注册到`workspace/core4d/scripts/eval/wrappers/review_player.sh`。
- 两侧现有单实验视频均已包含`ref | sim`画面，因此最终paired视频采用四画面结构：`E178 ref | E178 sim || E187 ref | E187 sim`。
- 本阶段是纯离线可视化与review索引注册，不运行物理仿真，不修改E178/E187 CEM、metrics、reward/grid/P/G或C9治理状态；按规则无需scene snapshot。

## Claims

| Claim | 验证方式 | 成功标准 |
|---|---|---|
| V1 paired完整性 | E187 eval case set按`case_id`连接E178/E187视频 | 22/22一一配对；missing/duplicate/unexpected=0 |
| V2 视频可读性 | ffprobe审计生成MP4 | 22/22 H.264/yuv420p、1920×540、可读、时长>0 |
| V3 左右语义正确 | 标签与输入manifest审计+关键帧抽查 | 左=`E178 baseline`，右=`E187 continuation`；抽查improved/regressed/stable三类 |
| V4 review player注册 | canonical `review_player.sh --check` | 默认索引含E187，显示22 indexed、6 numeric pass、22 playable |
| V5 provenance | paired manifest与summary | 每行保留两侧video路径/SHA、output路径/SHA、duration/resolution/status |
| V6 治理不漂移 | log/progress/tracker审计 | C9保持`FAIL / USER_WAIVED`；paired numeric仍`6/22 vs 8/22` |

## 改动

1. 新增`workspace/core4d/scripts/experiments/E187/render_paired_evaluation_videos.py`：只读E178/E187 metrics，使用ffmpeg生成22条左右对比视频与machine-readable manifest/summary。
2. 新增`workspace/core4d/scripts/eval/wrappers/render_E187_vs_E178_paired.sh`，固化`preflight/run/audit`入口。
3. 在`eval/review/review_index.py`注册E187为默认实验，并同步更新`viser_review_player.py`与`review_player.sh`说明。
4. 更新E187 Evaluation log、Tracker和progress，记录视频路径、SHA与实际视觉观察。

## 成功标准

- 输出目录包含22条`*_E178_vs_E187.mp4`，无额外或缺失case。
- paired manifest 22行且所有input/output SHA闭合；summary status=`pass`。
- `review_player.sh --check`对全部默认实验无mismatch，并明确E187 `22/22 playable`。
- 抽取3条代表paired MP4中帧，实际观察确认标签、左右方向和画面可读。

## Canonical commands

```bash
bash workspace/core4d/scripts/eval/wrappers/render_E187_vs_E178_paired.sh preflight
bash workspace/core4d/scripts/eval/wrappers/render_E187_vs_E178_paired.sh run
bash workspace/core4d/scripts/eval/wrappers/render_E187_vs_E178_paired.sh audit
bash workspace/core4d/scripts/eval/wrappers/review_player.sh --check
bash workspace/core4d/scripts/eval/wrappers/review_player.sh --exps E187 --port 8080
```

## 禁止项

- 不覆盖E178/E187原始视频；paired输出写E187独立目录。
- 不修改E178历史metrics/workbook/log或E187 Full artifacts。
- 不把视觉spot-check当成12门数值结论或用户人工终审。
- 不改变C9 technical=`FAIL`、progression authority=`USER_WAIVED`。
