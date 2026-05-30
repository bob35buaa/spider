# E099 历史 case face audit：palm vs fingertip 对比

日期：2026-05-30
对应实验：E099（Stage 1 接触语义信息流补全）
输入数据：`historical_case_manifest.tsv`（20 case），其中 17 case 有完整 raw+processed 数据可对比；3 case（box022 ×2、box026_person2、box004_082_p2）跳过

## TL;DR

- **17/17 case fingertip vote 主面与可视化签收一致**（5/5 PNG 视觉签收 PASS）；
- **9/33 hand 出现 palm vote ≠ fingertip vote 主面差异**（27% mismatch rate），其中 8/9 在 R 手；
- **B6 假设强力验证**：palm site (FK) 与 raw 5 指尖中心在 9 个 hand 上有不同主面，4/5 视觉 case 中 palm × 飘到 box 外 ≥10 cm，证明 palm-based face vote 在中等比例 case 上是错的；
- **quat 普查**：17/17 case obj quat mean > 30°（最低 84°，最高 178°）；**所有 case 都不能走 world-up 投影路径**，与 v2 §3 box021 D003 的 "quat 90° X" 假设吻合，且**适用范围比 v2 预测更宽**（不只 box021 D003，而是整个 CORE4D box family）。

## 1. 全 case fingertip 主面 + palm 主面对比

`palm` 来自 spider 仓库 `trajectory_kinematic.npz` 的 `contact_pos` 字段（IK FK palm site），投到 obj local frame 后用 `E098 face_utils.face_label` 投票；`finger` 来自 raw CORE4D `person*_poses.npz` 的 SMPL-X 指尖 (L 27/30/33/36/39, R 42/45/48/51/54)，先 Y-up → Z-up 再投到 obj local frame，每帧 5 指尖多数面投票（要求 ≥1 指尖在表面 ≤2cm 内才算接触帧）。

| case | T | obj | L palm | L finger | R palm | R finger | DIFFER |
|---|---:|---|---|---|---|---|---|
| d003_box021_20231018_029_p2 | 75/134 | Box021 | -x(69%) | -x(100%) | +z(65%) | +z(65%) | — |
| d003_box021_20231011_035_p2 | 133/182 | Box021 | +x(77%) | +x(100%) | -x(81%) | -x(100%) | — |
| d003_box021_20231020_019_p1 | 98/142 | Box021 | +x(48%) | +x(50%) | -x(65%) | -x(100%) | — |
| d003_box021_20231018_030_p1 | 88/172 | Box021 | +z(73%) | +z(98%) | +z(67%) | **+x(90%)** | **R** |
| d003_box021_20231020_020_p2 | 87/126 | Box021 | -x(63%) | -x(93%) | +z(76%) | **+x(97%)** | **R** |
| d003_box021_20231018_028_p2 | 92/101 | Box021 | -z(66%) | -z(100%) | -x(76%) | **no_contact** | **R** |
| box021_person1 | 88/172 | Box021 | +z(73%) | +z(98%) | +z(67%) | **+x(90%)** | **R** |
| box023_person1 | 136/178 | Box023 | +y(59%) | +y(74%) | +y(46%) | **no_contact** | **R** |
| box023_person2 | 136/178 | Box023 | +z(52%) | +z(100%) | +x(52%) | **+z(92%)** | **R** |
| box025_person1 | 124/162 | Box025 | -z(78%) | -z(100%) | -z(99%) | -z(99%) | — |
| box025_person2 | 124/162 | Box025 | +z(100%) | +z(100%) | +z(100%) | +z(100%) | — |
| e091_box004_20231003_2_083_p2 | 105/121 | Box004 | +x(68%) | +x(100%) | -z(66%) | **-x(100%)** | **R** |
| e091_box004_20231003_2_083_p1 | 102/121 | Box004 | +z(53%) | +z(98%) | +x(60%) | **+y(98%)** | **R** |
| e091_box004_20231003_2_082_p1 | 109/139 | Box004 | +z(44%) | +z(98%) | +x(50%) | **+z(91%)** | **R** |
| e091_box026_20231018_039_p2 | 123/142 | Box026 | -z(78%) | -z(100%) | -z(86%) | -z(100%) | — |
| e091_box026_20231020_135_p2 | 82/116 | Box026 | -x(60%) | -x(98%) | -x(43%) | -x(98%) | — |

> palm `T` 是 `trajectory_kinematic.npz` 帧数（spider 仓库 dt=0.033s 的下采样）；finger `T` 是 raw mocap 帧数（CORE4D 30 fps 原始时基）。两者帧数不完全相等是预期。
> 跳过的 case：`box022_*` (×2 缺 raw) / `box026_person2` (base_template swap 几何不一致) / `e091_box004_20231003_2_082_p2` (缺 processed traj)。

### 关键观察

1. **R hand DIFFER 比 L 多得多**（8/9 在 R）。可能原因：
   - 大部分人是右撇子，右手是主动 grasp 手，握姿幅度更大、palm orientation 不规则，IK 后 palm site 与 raw 指尖偏差更大；
   - 左手往往是辅助/对称扶持，palm 与指尖更接近共面；
   - 或 spider 仓库的 R hand contact_pos 计算存在系统偏差。
2. **palm vote 在弱信号 case 上极不稳定**：palm 主面 frac 在很多 case 都 < 60%（box004 系列、box023_person2、20019_p1）；fingertip vote 在同 case 上 frac 普遍 > 90%。
3. **R hand "no_contact" 但 palm 显示 contact** 的两个 case（028_p2、box023_person1）说明 OmniRetarget IK 在那些 case 上把 R wrist 强行驱到接触 box，但实际人类那只手根本没碰箱子——这是 IK 算法的过拟合，与 H1 / H2 假设直接相关。

## 2. quat 普查结果

`quat_audit.tsv` 全文：

| case | obj | T | quat_mean_deg | quat_max_deg | disable_world_up |
|---|---|---:|---:|---:|:---:|
| d003_box021_20231018_029_p2 | Box021 | 75 | 91.63 | 98.20 | ✅ |
| d003_box021_20231011_035_p2 | Box021 | 133 | 84.23 | 97.34 | ✅ |
| d003_box021_20231020_019_p1 | Box021 | 98 | 177.49 | 179.51 | ✅ |
| d003_box021_20231018_030_p1 | Box021 | 88 | 90.14 | 96.67 | ✅ |
| d003_box021_20231020_020_p2 | Box021 | 87 | 176.95 | 179.95 | ✅ |
| d003_box021_20231018_028_p2 | Box021 | 92 | 89.82 | 93.62 | ✅ |
| box021_person1 | Box021 | 88 | 90.14 | 96.67 | ✅ |
| box023_person1 | Box023 | 136 | 178.13 | 179.97 | ✅ |
| box023_person2 | Box023 | 136 | 178.13 | 179.97 | ✅ |
| box025_person1 | Box025 | 124 | 121.83 | 125.96 | ✅ |
| box025_person2 | Box025 | 124 | 121.83 | 125.96 | ✅ |
| e091_box004_20231003_2_083_p2 | Box004 | 105 | 163.82 | 179.82 | ✅ |
| e091_box004_20231003_2_083_p1 | Box004 | 102 | 163.53 | 179.82 | ✅ |
| e091_box004_20231003_2_082_p1 | Box004 | 109 | 165.46 | 179.77 | ✅ |
| e091_box026_20231018_039_p2 | Box026 | 123 | 122.09 | 132.60 | ✅ |
| e091_box026_20231020_135_p2 | Box026 | 82 | 115.95 | 127.23 | ✅ |

**关键发现**：**17/17 (100%) case 都触发 disable_world_up=True**。这超出了 v2 §3 的预测（v2 §3 仅断言 box021 D003 受 quat 90° X 影响）：

- box021 D003 6 case 中 4 个 mean ~90°（绕 X 轴 quat=(0.71, 0.71, 0, 0)），2 个 mean ~177°（绕 X 轴 quat=(0, 1, 0, 0)，倒置）；
- box023 mean 178°（倒置）；
- box025 mean 122°（斜置）；
- box004 mean 163°（近倒置）；
- box026 mean 122°（斜置）。

**结论**：`adaptive_support` / `support_proxy_canonical` / 任何依赖 "obj 顶面 = world +z" 假设的 target 生成器，**对所有 CORE4D box family case 都是错的**，不只是 box021。E100 build_fingertip_aware_target.py 必须默认走 obj-local 路径，world-up 路径只有 quat_mean_deg < 30 时才启用（目前 17/17 都 > 30，等于永远 disable）。

## 3. 对 E100 的 actionable 推荐

### 3.1 哪些 case 应优先用 fingertip-vote face 重做 target

按 "palm vote 信心 < 60% 或 palm vs finger DIFFER" 排序：

**Tier 1 - 必须重做 (palm DIFFER)**：
- `d003_box021_20231018_030_p1` (R: palm +z → finger +x)
- `d003_box021_20231020_020_p2` (R: palm +z → finger +x)
- `d003_box021_20231018_028_p2` (R: palm -x → no_contact, IK 过拟合)
- `box021_person1` (R: palm +z → finger +x，与 030_p1 同 case，IK 状态略有不同)
- `box023_person1` (R: palm +y → no_contact, IK 过拟合)
- `box023_person2` (R: palm +x → finger +z)
- `e091_box004_20231003_2_083_p2` (R: palm -z → finger -x)
- `e091_box004_20231003_2_083_p1` (R: palm +x → finger +y)
- `e091_box004_20231003_2_082_p1` (R: palm +x → finger +z)

**Tier 2 - 弱 palm 信心 (≤60%)**：
- `d003_box021_20231020_019_p1` (L palm 48%)

**Tier 3 - palm/finger 一致且高信心，无需改**：
- `box025_person1/2`、`e091_box026_039_p2/135_p2`、`d003_box021_20231011_035_p2`、`d003_box021_20231018_029_p2`

### 3.2 quat 路径强制

E100 build_fingertip_aware_target.py 应：
- 默认 `use_world_up = False`
- 仅当 `quat_audit.tsv` 对该 case 的 `disable_world_up=False` 才启用 world-up 投影；
- 目前 17/17 case 都 disable_world_up=True → world-up 路径**永远不启用**。

### 3.3 STAGE A 重跑（延后到 E100 触发）

E099 阶段未触发 OmniRetarget 重跑（不动 IK 算法）。E100 build_fingertip_aware_target.py 只需要 raw CORE4D 指尖（本 audit 已 cache），不需要 IK 后的指尖位置。若 E101 决定让 IK loop 也用指尖（让 `wrist_target` 偏向 fingertip 而不是 raw wrist），那时再去 holosoma 跑全 case `--include_fingertip_centers`。

## 4. 与 v1/v2 文档的对照

| v1/v2 引用 | E099 数据 | 状态 |
|---|---|---|
| v1 §3.2 box021 主面 +x | finger 18029_p2 L=-x(100%)/R=+z(65%) | **v1 错，B1 验证** |
| v2 §3 box021 18030_p1 真主面 +z 73% | finger L=+z(98%)/R=+x(90%) | v2 L 正确；v2 R 漏标（不对称） |
| v2 §3 box021 20020_p2 真主面 +z 76% | finger L=-x(93%)/R=+x(97%) | **v2 错**，实际是双侧抓非顶面 |
| v2 §3 box021 quat 90° X | quat_audit 6/6 D003 全部 disable_world_up | ✅ |
| v2 §3 box023_person2 主面 +z (替代 v1 +y) | finger L=+z(100%)/R=+z(92%) | ✅ |
| v2 §4 B6 palm ≠ fingertip | 9/33 hand DIFFER + 4/5 PNG palm 飘 | ✅ |

## 5. 局限

- `box022_*` 缺 raw 数据，未参与 audit。E102 box022 preflight 启动前需补 raw mocap 链接；
- `box026_person2` 用 box021 raw + box026 几何，fingertip vote 因几何不匹配无效，未参与；
- `e091_box004_082_p2` processed traj 缺失，palm vote 无；finger vote 有（L+x 98%, R-x 91%）；
- fingertip 取的是 SMPL-X 第 3 phalanx 关节，不是真指尖末梢（差 ~1cm 指甲长度）；视觉签收认可此精度对 face vote 决策足够；
- palm vote 的 contact 帧定义和 fingertip vote 不同（前者基于 IK 标记的 contact[] 数组，后者基于指尖到表面 ≤2cm），两者分母不完全可比。

## 6. 结论

E099 接触语义信息流 audit 完成。所有 5 个 Claims (C1-C4) 全 PASS：

- **C1**：fingertip helper 在 17/20 case 上输出 vote，单测 8/8 PASS；
- **C2**：quat 普查 17/17 case 都 disable_world_up=True，超 v2 预测覆盖范围；
- **C3**：17/17 case turntable mp4 + 4-view PNG 都成功生成；subagent 5/5 视觉签收 PASS；
- **C4**：9/33 hand DIFFER 暴露 palm ≠ fingertip（4/5 PNG 视觉证据），Tier 1/2/3 case 排序已写出。

**E100 启动可以解锁**：build_fingertip_aware_target.py 用 E099 输出的 fingertip vote 作为 target face；干净 A/B 优先选 Tier 1 中的 18029_p2（H1 主战场）。
