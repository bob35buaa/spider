# E187 vs E178 Paired Video Review 结果

日期：2026-08-03
实验：E187
阶段：Evaluation visual review
结论：22/22 paired MP4闭合；E178固定在左、E187固定在右；E187已注册到默认review player索引

## Context

[log 258](258_E187_vs_E178_paired_evaluation_results.md) 已冻结 E187 vs E178 的22-case
paired numeric结果：E187 `6/22`，同case E178 `8/22`。用户要求补充左右可视化视频并注册到
`workspace/core4d/scripts/eval/wrappers/review_player.sh`。本阶段执行
[plan 209](../plan/209_E187_vs_E178_paired_video_review_plan.md)，只做离线ffmpeg编码与review
索引注册，不修改两侧CEM、metrics或治理状态。

## 视频合同

每条输入视频原本均为 `ref | sim`。paired输出固定为：

```text
LEFT - E178 BASELINE                    RIGHT - E187 CONTINUATION
[ E178 ref | E178 sim ]                 [ E187 ref | E187 sim ]
              case_id | E178_{PASS/FAIL}_TO_E187_{PASS/FAIL}
```

| 项目 | 结果 |
|---|---:|
| expected / rendered | `22 / 22` |
| independent audit | `22 / 22 PASS` |
| missing / unexpected / invalid | `0 / 0 / 0` |
| codec / pixel format | `H.264 / yuv420p` |
| resolution / fps | `1920×540 / 50fps` |
| duration | `2.36–8.04s`，每对与两侧输入一致 |
| total MP4 bytes | `5,912,002` |
| transition labels | F→F 11、P→F 5、F→P 3、P→P 3 |

Canonical commands：

```bash
bash workspace/core4d/scripts/eval/wrappers/render_E187_vs_E178_paired.sh preflight
bash workspace/core4d/scripts/eval/wrappers/render_E187_vs_E178_paired.sh run
bash workspace/core4d/scripts/eval/wrappers/render_E187_vs_E178_paired.sh audit
bash workspace/core4d/scripts/eval/wrappers/review_player.sh --check
bash workspace/core4d/scripts/eval/wrappers/review_player.sh --exps E187 --port 8080
```

## Review player 注册

`review_player.sh`调用的`review_index.DEFAULT_EXPS`已加入E187，viser/player入口说明同步为
E170–E187。Headless canonical check：

| exp | indexed | evaluated | numeric pass | reviewed | playable |
|---|---:|---:|---:|---:|---:|
| E187 | 22 | 22 | 6 | 0 | 22 |
| 全部默认实验 | 187 | — | — | 66 | 187 |

所有默认实验均与各自`summary.json`一致，无mismatch；E187可通过`--exps E187`单独筛选。

## 可视化 → 实际观察

使用`video-frames`从三类paired MP4抽取2s中帧：

- improved `bucket004_20231002_021_p1`：顶部左右标签明确，底部显示
  `E178_FAIL_TO_E187_PASS`；四个ref/sim画面均完整，E187 sim物体位置与手部姿态变化可见；
- regressed `bucket003_20231018_001_p2`：底部显示`E178_PASS_TO_E187_FAIL`；两侧均为
  人体处于桶体内部/边缘的高风险构型，E187足部/下肢接触差异可见；
- stable-pass `bucket004_20231003_1_012_p2`：底部显示`E178_PASS_TO_E187_PASS`；
  两侧站姿与物体相对位置总体相近，方向和语义无反转。

中帧确认中央分隔线、标签、case ID和transition均可读；该spot-check不替代完整时序数值门
或用户人工终审。

## Claims

| Claim | 状态 | 证据 |
|---|---|---|
| V1 paired完整性 | PASS | 22 unique case IDs；missing/duplicate/unexpected=0 |
| V2 视频可读性 | PASS | 22/22 ffprobe H.264/yuv420p/1920×540/50fps |
| V3 左右语义正确 | PASS | 固定input0=E178左、input1=E187右；三类关键帧抽查 |
| V4 review player注册 | PASS | E187 22 indexed / 6 pass / 22 playable |
| V5 provenance | PASS | manifest逐行保存两侧/output路径、SHA、probe与status |
| V6 治理不漂移 | PASS | C9仍`FAIL / USER_WAIVED`；numeric仍`6/22 vs 8/22` |

## 结果路径与 SHA256

| 内容 | 路径 | SHA256 |
|---|---|---|
| paired目录 | `workspace/core4d/results/E187/s6_downstream/render/full/paired_e178_vs_e187/` | 22 MP4 + manifest/summary/keyframes |
| manifest | `workspace/core4d/results/E187/s6_downstream/render/full/paired_e178_vs_e187/paired_video_manifest.tsv` | `7edb73a047fd65fc1c666b320438d384b2790e9801605b6fc326f47b15e8cafc` |
| summary | `workspace/core4d/results/E187/s6_downstream/render/full/paired_e178_vs_e187/summary.json` | `aa16bd1030b1969311ee31a4a6a76276cac7389e6c5de45e1f43575f57728314` |
| improved示例 | `workspace/core4d/results/E187/s6_downstream/render/full/paired_e178_vs_e187/bucket004_20231002_021_p1_E178_vs_E187.mp4` | `e22be094308584ec20ffca4e89dc98f44d4383dc55fb632f2a0dc8d50e993ab1` |

## 下一步

视觉包和review player注册已完成。用户可用播放器筛选E187并做人工标注，或直接打开paired
目录逐条观看。人工裁决必须写入非破坏性的`user_manual_review_filled.tsv`；不得回写E178
历史review或把视觉观察用于修改E187冻结参数。
