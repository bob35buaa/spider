# E085 Plan: raw-contact target repair after E084 audit

日期：2026-05-28

## Context

E084 追加诊断 `workspace/core4d/log/107_E084_contact_target_semantics_audit.md` 明确了新的根因：

- `core4d_3cm` contact mask 只提供 per-hand 二值门控，不提供接触点；
- mask 对 `d003_box021_20231018_029_p2` 的 person/time 选择没有明显错误；
- raw CORE4D person2 双手几何接触很强，left/right broad-hand min dist mean 为 `1.8/2.2mm`；
- 当前 `contact_hdmi_dynamic_target` 由 G1 retargeted `wrist_yaw_link + [0.05,0,0]` 反推，和 raw contact surface centroid 平均相差约 `27cm`；
- 因此继续基于 G1 wrist pseudo target 调 CEM reward 是错方向。

E085 先做数据层目标修复：从 raw SMPL-X hand/object surface 生成 object-local per-hand contact target，并让 MJWP 可直接加载该 target。只有 target 修复后的 reference/audit 显示目标几何更合理，才进入 CEM。

## 实验范围

第一阶段只覆盖：

```text
main:  d003_box021_20231018_029_p2_upperobj_e083
guard: box023_person2_upperobj_e083
```

说明：

- main 继续用 E083 upper-body-object 派生 scene，保持物理约束完整；
- guard 用 `box023_person2_upperobj_e083`，防止 raw-target 接口破坏此前 positive guard；
- 不先扩到另外两个 Box021，除非 main/guard target audit 过关。

## Claims

| Claim | 验证方式 | 成功标准 |
|---|---|---|
| C1 raw target 生成可复现 | 预处理脚本 + npz/csv/json | main/guard 都输出 raw/spider/eval target，含 mask、surface target、face、vertical fraction |
| C2 坐标系正确 | scene `object_visual` transform audit | raw surface points 已转到 MuJoCo object body frame；target 与 collision half-extents 同坐标系 |
| C3 target 修复了 G1 pseudo target 偏移 | target audit 对比 | main 的 raw-target vs raw contact surface delta 接近 0；raw-target vs 旧 G1 target 记录为约 `27cm` 偏移 |
| C4 MJWP 可加载 external target | smoke run | `run_mjwp.py` 日志显示 external target key/axis/len，短 horizon 成功 |
| C5 CEM 只在数据目标合理后启动 | gate 文件 | 若 main target 可达性/guard smoke 不通过，不启动 full CEM |
| C6 若进入 CEM，必须本地+远程并行并回收 | scripts + results | local main + remote guard/variant 完成，pull 后本地 eval/visual sheet 完整 |

## 改动

### 1. Raw target 生成

新增：

```text
workspace/core4d/scripts/E085/generate_raw_contact_targets.py
workspace/core4d/scripts/run_E085_preprocess.sh
```

生成格式：

```text
workspace/core4d/results/E085/raw_targets/<case>/raw_contact_targets.npz
```

核心数组：

```text
raw_contact_target_object_local      (T_raw, 2, 3)
spider_contact_target_object_local   (T_spider, 2, 3)
eval_contact_target_object_local     (T_eval, 2, 3)
*_target_valid                       同时间轴, (T, 2)
*_target_face                        同时间轴, string/int face label
*_target_vertical_frac               同时间轴, (T, 2)
```

target 选择规则：

- 使用与 mask 一致的 broad hand vertex ranges；
- 对 mask active 帧，取 `<3cm` hand vertices 的 nearest object surface points centroid；
- 若 active 但没有 close vertices，fallback 到 fingertip nearest surface；
- 对 inactive 帧 forward/backward fill，保证 reward resize 后不会出现 NaN，但 mask 仍控制 reward 是否启用；
- 坐标输出为 MuJoCo object body frame，包含 `object_visual` 固定 mesh transform。

### 2. MJWP external target 支持

修改：

```text
spider/config.py
examples/run_mjwp.py
```

新增配置：

```yaml
contact_hdmi_target_source: ref_fk   # ref_fk | external
contact_hdmi_target_path: ""
contact_hdmi_target_time_axis: auto  # auto | spider | eval | raw
```

行为：

- 默认 `ref_fk` 完全保持旧逻辑；
- `external` 时从 npz 加载 object-local target，按 qpos_ref 长度 nearest-neighbor resize；
- external target 和 `contact_hdmi_mask_source=core4d_3cm` 共同使用：mask 负责启用时间，target 负责空间位置。

### 3. E085 overrides / smoke / eval

新增：

```text
workspace/core4d/scripts/E085/generate_e085_overrides.py
workspace/core4d/scripts/train/train_E085.sh
workspace/core4d/scripts/run_E085_remote.sh
workspace/core4d/scripts/pull_E085_remote_results.sh
workspace/core4d/scripts/eval/eval_E085.py
workspace/core4d/scripts/eval/extract_E085_contact_sheets.sh
```

首批 variant：

| Variant | Role | 设计 |
|---|---|---|
| `E085A_rawtarget_main` | main | E084C-like safety/semantic 设置 + external raw target |
| `E085A_rawtarget_guard` | guard | 同设置，验证 box023_p2 不退化 |
| `E085B_rawtarget_upright_main` | optional | raw target + E084B trust/upright，用于 A 仍蹲抱时迭代 |

第一轮只在 A main/guard smoke 通过后跑 full CEM。

## Gate

预处理后必须先检查：

| Gate | 标准 |
|---|---|
| G1 target delta 被解释清楚 | raw-target vs old G1 target delta 写入 JSON |
| raw target 不落到地面/箱内奇异位置 | active target surface_dist to collision box `<=3cm` 或在合法 surface face |
| main target face 语义可接受 | 不再是旧 G1 target 的对侧漂移；left/right face 与 raw contact 一致 |
| guard smoke 通过 | `max_sim_steps=4` 成功，外部 target 加载日志正常 |

若 gate 失败，本轮停止在数据/IK 层，不启动 CEM。

## CEM 运行

如果 gate 通过：

```bash
bash workspace/core4d/scripts/run_E085_preprocess.sh
bash workspace/core4d/scripts/train/train_E085.sh local 0 E085A_rawtarget_main
bash workspace/core4d/scripts/run_E085_remote.sh
bash workspace/core4d/scripts/pull_E085_remote_results.sh
python workspace/core4d/scripts/eval/eval_E085.py
bash workspace/core4d/scripts/eval/extract_E085_contact_sheets.sh
```

本地跑 main，远程跑 guard 或 optional B。若 full CEM 失败，需要先做视频/关键帧观察，再决定是：

1. target 需要 surface offset / sphere radius 修正；
2. raw target 可行但 CEM 搜索失败，进入 support-body seed；
3. raw target 本身对应单 G1 不可行，停止 Box021 正例路线。

## 成功标准

数据层成功：

- raw target 生成和 MJWP external load 完整；
- main raw-target 与 raw contact surface 对齐，不再出现旧 G1 target `~27cm` 对侧偏移；
- guard 不破坏。

CEM 成功：

- main 不再蹲抱/压箱；
- upperbody penetration `<5%`；
- hand-floor `<5%`；
- object floor-contact 明显低于 E084C；
- object bottom gap vs ref 改善至少 `5cm`；
- sim hand contact `>=50%`；
- 视频不是翻箱、趴箱或手撑地。

若 CEM 失败但 gate/target 已修复，则记录为方法搜索失败，进入 support-body/COLA seed route，而不是再回到 wrist pseudo target。
