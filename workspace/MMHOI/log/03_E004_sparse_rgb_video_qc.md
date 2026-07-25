# R003 / E004：MMHOI 稀疏 RGB 视频 QC

## 目标

从已解压的 `C_2/C_8` camera-0 RGB 中挑选少量代表 case，生成不插值的
稀疏帧视频，供人工判断约 1 Hz 发布帧是否仍有动作判读价值。

## 时间与编码口径

源数据：

```text
capture_fps = 30
released_annotation_stride = 30
released_rgb_rate ≈ 1 Hz
```

1 fps 版本：

- 一张 released RGB 对应一个编码帧；
- 播放时每张图持续 1 秒；
- 近似保持 released sparse timeline 的真实时间长度。

30 fps 快速预览：

- 仍是一张 released RGB 对应一个编码帧；
- 不补帧、不插值、不做光流或 blending；
- 将约 1 Hz sparse timeline 加速 30 倍播放；
- 只用于快速扫动作，不代表恢复了 30 Hz motion。

## Case 选择

| Scenario | Case | Sparse frames | 选择依据 |
|---|---|---:|---|
| `C_2` | `20240423_personB_personC_and_C2_all__start-end/20240423_C_2__30skip` | 99 | 78 active-box / 61 cooperative-box，覆盖最高 |
| `C_2` | `20240426_personB_personG_all_30skip_start-end/20240426_C_2__30skip` | 45 | 35 active-box / 32 cooperative-box，短而集中 |
| `C_2` | `20240508_personE_personD_all_30skip_start-end/20240508_C_2K2__30skip` | 100 | 第二人员组合，75 active-box / 56 cooperative-box |
| `C_8` | `20240412_personA_personB/20240412__C_8__30skip` | 91 | 用户示例，包含 frame `00271`，84 cooperative samples |
| `C_8` | `20240508_personJ_personK_30_skip_start-end/20240508_C_8__30skip` | 50 | 不同人员组合，49 cooperative samples |

五个 case 的 camera-0 覆盖为 385/385，没有缺失 released RGB；每个 case
内部 source frame-id gap 全为 30。

## 产物

主目录：

```text
workspace/MMHOI/results/E004/s0b_sparse_rgb_visual_qc/
```

1 fps：

| Scenario | Frames | Duration | Resolution | Output |
|---|---:|---:|---:|---|
| `C_2` high coverage | 99 | 99 s | 2048×1536 | `videos/20240423_personB_personC_and_C2_all_start_end__20240423_C_2_30skip_cam0_1fps.mp4` |
| `C_2` compact | 45 | 45 s | 2048×1536 | `videos/20240426_personB_personG_all_30skip_start_end__20240426_C_2_30skip_cam0_1fps.mp4` |
| `C_2` second pair | 100 | 100 s | 2048×1536 | `videos/20240508_personE_personD_all_30skip_start_end__20240508_C_2K2_30skip_cam0_1fps.mp4` |
| `C_8` user example | 91 | 91 s | 2048×1536 | `videos/20240412_personA_personB__20240412_C_8_30skip_cam0_1fps.mp4` |
| `C_8` second pair | 50 | 50 s | 2048×1536 | `videos/20240508_personJ_personK_30_skip_start_end__20240508_C_8_30skip_cam0_1fps.mp4` |

用户示例 30 fps 快速预览：

```text
fast_preview_30fps/videos/
20240412_personA_personB__20240412_C_8_30skip_cam0_30fps.mp4
```

其 ffprobe 结果：

```text
codec:          h264
resolution:     2048 x 1536
avg_frame_rate: 30/1
input/output:   91 / 91 frames
duration:       3.034 s
interpolation:  none
```

用户提供的 `00271/0_00271.jpg` 是这个 30 fps 视频的第一帧。

用户反馈 30 fps 过快后，追加相同 91 张图的无插值速度对照：

| Playback | Input / output frames | Duration | 说明 |
|---:|---:|---:|---|
| 10 fps | 91 / 91 | 9.100 s | 稀疏时间轴加速 10 倍 |
| 15 fps | 91 / 91 | 6.067 s | 稀疏时间轴加速 15 倍 |
| 20 fps | 91 / 91 | 4.550 s | 稀疏时间轴加速 20 倍 |
| 30 fps | 91 / 91 | 3.034 s | 稀疏时间轴加速 30 倍 |

这些版本仅修改输入图片序列的 playback framerate，均未使用 ffmpeg
`minterpolate`、frame blending 或其他插帧 filter。

## 全部 `C_2/C_8` 的 10 fps 扩展

用户确认 10 fps 可读后，将相同无插帧口径扩展到全部主范围 capture。

口径：

```text
scope:             all C_2/C_8 scenario captures
camera:            0
playback_fps:      10
interpolation:     none
frame_blending:    false
optical_flow:      false
resolution:        2048 x 1536
codec:             H.264
```

注意：`C_2` 包含 box；`C_8` 的物体实际是 `chair_wood/table_wood`，没有
box。由于用户明确要求全部 `C_2/C_8`，本轮仍包含 12 个 `C_8` capture。

| Scenario | Cases | Input/output frames | Duration | MP4 bytes |
|---|---:|---:|---:|---:|
| `C_2` | 12 | 636 / 636 | 63.6 s | 58,674,170 |
| `C_8` | 12 | 653 / 653 | 65.3 s | 60,969,559 |
| **合计** | **24** | **1,289 / 1,289** | **128.9 s** | **119,643,729** |

产物：

```text
all_c2_c8_10fps/
├── README.md
├── render_manifest.tsv
├── run_manifest.json
├── videos/             # 24 MP4
├── frames/             # 24 per-frame TSV
└── keyframes/
    ├── C_2_mid_sheet.jpg
    ├── C_8_mid_sheet.jpg
    └── *_mid.jpg       # 每个 MP4 解码一张中间帧
```

全量 ffprobe gate：

- 24/24 MP4 存在且可解码；
- 24/24 为 `avg_frame_rate=10/1`；
- 24/24 为 H.264、2048×1536；
- 每个 case 的 `nb_frames` 与 frame manifest 行数相等；
- 1,289/1,289 source frame id 的相邻 gap 为 30；
- run manifest 中没有 `-vf`、`minterpolate`、optical flow 或 blending；
- 不存在残留的临时 frame staging 目录。

两张 12-case 中间帧总览已人工检查：`C_2` 均为有效人物/搬箱画面，
`C_8` 均为有效双人 chair/table 画面，没有空帧或明显串 case。

证据：

```text
render_manifest.tsv
run_manifest.json
frames/*.tsv
fast_preview_30fps/render_manifest.tsv
fast_preview_30fps/run_manifest.json
fast_previews/10fps/run_manifest.json
fast_previews/15fps/run_manifest.json
fast_previews/20fps/run_manifest.json
keyframes/selected_cases_mid_sheet.jpg
all_c2_c8_10fps/render_manifest.tsv
all_c2_c8_10fps/run_manifest.json
all_c2_c8_10fps/keyframes/C_2_mid_sheet.jpg
all_c2_c8_10fps/keyframes/C_8_mid_sheet.jpg
```

## 可复跑命令

1 fps 五 case：

```bash
python3 workspace/MMHOI/scripts/visual_qc/render_sparse_rgb_videos.py \
  --data-root /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/MMHOI \
  --case-manifest workspace/MMHOI/configs/visual_qc/E004_sparse_rgb_cases.tsv \
  --output-dir workspace/MMHOI/results/E004/s0b_sparse_rgb_visual_qc \
  --camera-id 0 \
  --playback-fps 1 \
  --crf 20
```

30 fps 单 case：

```bash
python3 workspace/MMHOI/scripts/visual_qc/render_sparse_rgb_videos.py \
  --data-root /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/MMHOI \
  --case-manifest workspace/MMHOI/configs/visual_qc/E004_sparse_rgb_30fps_case.tsv \
  --output-dir workspace/MMHOI/results/E004/s0b_sparse_rgb_visual_qc/fast_preview_30fps \
  --camera-id 0 \
  --playback-fps 30 \
  --crf 20
```

全部 24 个 `C_2/C_8` 的 10 fps 版本：

```bash
python3 workspace/MMHOI/scripts/visual_qc/render_sparse_rgb_videos.py \
  --data-root /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/MMHOI \
  --case-manifest workspace/MMHOI/configs/visual_qc/E004_all_c2_c8_10fps_cases.tsv \
  --output-dir workspace/MMHOI/results/E004/s0b_sparse_rgb_visual_qc/all_c2_c8_10fps \
  --camera-id 0 \
  --playback-fps 10 \
  --crf 20
```

若重跑已有目录，显式追加 `--overwrite`。

## 验证

- 5 个 `unittest` 通过；
- `py_compile` 通过；
- 全量 batch 24 个 MP4 均为 H.264、2048×1536、10 fps；
- 每个 MP4 的 ffprobe `nb_frames` 与输入 RGB 数完全相等；
- 1 fps 五视频总计 385 帧，30 fps 预览为 91 帧；
- 全量 10 fps batch 总计 1,289 输入帧 = 1,289 输出帧；
- 全部 manifest 均记录 `interpolation=none`；
- 中间帧总览人工检查：三个 `C_2` 可见双人搬/堆 box，两个 `C_8`
  可见双人操作 table/chair。

## 执行问题

首次直接执行 `video-frames/scripts/frame.sh` 遇到脚本无 executable bit；
改为显式 `bash frame.sh` 后成功提取 QC keyframes。该问题只影响检查包装脚本，
不影响已生成 MP4。

全量中间帧第一次用 TSV pipe 驱动 wrapper 时，ffmpeg 从 stdin 消费了隔行
manifest，只提取到 12/24 张。修复为每次 wrapper 显式
`</dev/null` 后得到完整 `C_2=12/C_8=12` keyframes，并重建两张总览。
这个问题同样只影响 QC 抽帧，不影响 24 个 MP4。

## 状态

E004 的 sparse RGB visual QC 已完成。human-world、SMPL-X 和 box 6DoF
全量 representation gate 尚未执行，因此 E004 总体保持进行中。
