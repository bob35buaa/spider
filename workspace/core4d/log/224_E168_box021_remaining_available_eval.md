# E168 Box021 剩余 case 独立评测

日期：2026-07-17

状态：按用户要求不等待最后一条运行中的 case；已对当前可用的 `11/12` 条完成回收、视频补渲染、量化评测和独立 xlsx。

## 范围

范围由 production manifest 中 Box021 `28` 条减去上一批 canonical 人工核验 `16` 条得到，固定为 `12` 条：

```text
box021_20231018_030_p2
box021_20231018_031_p2
box021_20231018_032_p1
box021_20231018_032_p2
box021_20231018_034_p2
box021_20231018_035_p2
box021_20231020_019_p1
box021_20231020_019_p2
box021_20231020_022_p1
box021_20231020_022_p2
box021_20231020_023_p1
box021_20231020_023_p2
```

与上一批 16 条逐 case 集合求交结果为 `0`。本轮评测时 `022_p1` 已完成，`022_p2` 仍在运行，因此完整指标为 11 条，“未就绪”sheet 单独保留 `022_p2`。

## 回收与视频

执行 production pull 后 Box021 为 `27 complete / 1 running`。A6000 SSH 在 pull 日志同步时出现两次临时 kex 断开，retry 后成功；A100 产物与 shard status 合并成功，不计 case failure。

11 条可评测结果均来自 A100 compute-only 队列，远端不输出 MP4。本地使用既有渲染器补齐：

```bash
.venv/bin/python workspace/core4d/scripts/experiments/E168/render_a100_cem_videos.py \
  --pool a100 --cases <11 ready case ids> --require-all
```

结果为 rendered `11/11`、failed `0`，视频均写回 production manifest 指定路径。

## 评测结果

固定入口：

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E168_box021_remaining_available.sh
```

本轮继续使用用户确认的新门槛：body-z p95 `<=0.20m`、raw contact `>=0.50`、penetration `<=0.30`、lower body `<=0.10`。

| gate | pass |
|---|---:|
| all numeric | `2/11` |
| tracking | `10/11` |
| fixed-reference z p95 | `9/11` |
| raw contact | `8/11` |
| release | `11/11` |
| penetration | `10/11` |
| lower body | `3/11` |
| no fall | `10/11` |

两条 numeric pass：

- `box021_20231020_023_p1`
- `box021_20231020_023_p2`

主要失败源仍是 lower body，共 `8/11`。其余失败计数为 contact `3`、z p95 `2`、tracking `1`、penetration `1`；失败模式可叠加。

这些 11 条均未获得用户人工结论，保持 `PENDING_MANUAL_REVIEW`。本记录不把 numeric pass 直接等同于可用。

## Excel

```text
workspace/core4d/results/E168/s6_downstream/cem/eval/box021_remaining12_available/E168_box021_remaining12_available_case_metrics.xlsx
```

- 10 sheets。
- “完整指标”为 `11 rows x 192 columns`。
- “未就绪”为 `1` 条，仅 `box021_20231020_022_p2`。
- “人工核验”为范围内空快照，公式读回 `0/0/0`。
- 与上一批人工核验 16 条 case ID 交集为 `0`。
- LibreOffice 重算 `334` 个公式，扫描结果 `0` formula errors。

## 后续

`022_p2` 完成后可重新运行同一 wrapper；届时范围仍是同一 12 条，新表会自动更新为 `12/12`，仍不会混入上一批 16 条。
