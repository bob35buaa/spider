# log299 · E210：bucket007 增强变体 × PRG+G1（object gravcomp）

_Core4D · Phase 69 · Run **R296** · 计划 [plan240](../plan/240_E210_bucket007_aug_g1only_gravcomp_plan.md) · 承接 [E207/log296](296_E207_bucket_g1only_gravcomp.md) / [E202/log291](291_E202_bucket_e178_translation_augmentation.md) / [E208/log297](297_E208_deskchair_translation_augmentation.md) / [E209/log298](298_E209_desk_chair_prg_g1_gravcomp.md) · 2026-09-05～09-06 · 分支 `feat/E207-bucket-g1only-gravcomp` · **状态：PARTIAL SUCCESS（C3 压线通过，C5a 破）**_

## 一句话进展

把 E207 的唯一干预（object `gravcomp=1`）移植到 E202 已建好的 15 个 bucket007 平移增强变体上，补齐 2×2 第四格：**gravcomp 在增强数据上的代价明显大于它在 orig 上的代价**——严格单变量的 `C→D` 让 15 条里 5 条掉出 14-gate narrow（+0/−5），而同一干预在 orig 上只掉 1/5；更重要的是，**代价以两种互不相同的形态出现，任何单一检测指标都会漏掉其中一种**。

---

## 一、执行

| 阶段 | 结果 |
|---|---|
| P0 编号/分支/git | 15 个 aug task × 6 文件 `git add -f`（90 个） |
| P1 契约 | 15/15 定位；两份 authority sha256 pin；`075_p2` 缺席**显式断言**（非静默） |
| P2 sidecar | 15/15 过 `assert_gravcomp_diff`；**篡改样本反向自测**证守卫会拒绝非单变量 diff |
| P3 override | Hydra compose 全 key diff **恰为 `{scene_name}`**，15/15 |
| P4 manifest | 15 行；轨迹/掩码 sha 与 E202 逐条相等且与磁盘一致 |
| P5 快照 | `results/E210/scene_snapshot/` 15 task / 106 文件（HEAD 1c0a099） |
| P6 smoke | 1 条 64×4，运行时 9/9 |
| P7 full | **15/15 cem_ok**（中途队列被打断，见 F1；恢复后跑完）；运行时审计 15/15 |
| P8 评测 | 4 格 × 40 rollout；evaluator 无漂移（cell B 重打分 5/5 复现 E207 判决） |
| P9 视觉 | 15 条 E210 + 3 条 E202 对照渲染（osmesa）；抽帧复核完成 |

**范围**：5 case × trans0/1/2 = 15 变体。`bucket007_20231023_075_p2` 排除——E202 对它三档记的是 `runtime_initial_overlap`（平移后参考**首帧**腿-桶几何重叠），属参考层几何事实，gravcomp 不改变；用户口径排除 rot 后已无未尝试变量。

---

## 二、主表：2×2 四格 14-gate（同 E201 漏斗，与 E207 同尺子）

| 格 | 实验 | n | hard | wide_all | **narrow_all** |
|---|---|--:|--:|--:|--:|
| A orig × PRG | E178 | 5 | 5 | 4 | **3 (60%)** |
| B orig × PRG+G1 | E207 | 5 | 5 | 4 | **2 (40%)** |
| C aug × PRG | E202 | 15 | 15 | 11 | **9 (60%)** |
| **D aug × PRG+G1** | **E210** | 15 | 15 | 7 | **4 (27%)** |

逐门 narrow 通过数（只列有差异的）：

| gate | A orig×PRG | B orig×G1 | C aug×PRG | **D aug×G1** |
|---|--:|--:|--:|--:|
| root_pos | 5/5 | 5/5 | 13/15 | 13/15 |
| root_ori | 5/5 | 5/5 | 15/15 | **13/15** |
| eef_pos | 5/5 | 5/5 | 13/15 | 12/15 |
| eef_ori | 3/5 | 3/5 | 14/15 | 13/15 |
| **contact** | 5/5 | 5/5 | 11/15 | **8/15** |
| release | 5/5 | 5/5 | 13/15 | 15/15 |
| hand_pen | 5/5 | 4/5 | 14/15 | 13/15 |
| 其余 7 门 | 全过 | 全过 | 全过 | 全过 |

### 严格单变量 `C→D`（只差 gravcomp，n=15）

`narrow **+0 / −5**`。门翻转：`+contact 3`、`+eef_ori 2`、`+root_ori 2`、`+hand_pen 1`、`+root_pos 1`、`+eef_pos 1`；`−release 2`、`−eef_ori 1`、`−root_pos 1`。

| 指标 | 方向 | mean Δ | 变差/变好 | 最坏 |
|---|---|--:|--:|--:|
| `track_eef_ori_err_deg_mean` | 越低越好 | **+2.811** | 7 / 8 | **+24.06** |
| `track_root_ori_err_deg_mean` | 越低越好 | +1.865 | 5 / 10 | **+22.40** |
| `hand_object_physics_contact_in_mask_frac` | 越高越好 | **−0.055** | 6 / 9 | **−0.489** |
| `hand_object_physics_penetration_3mm_frame_frac` | 越低越好 | −0.015 | 4 / 10 | +0.179 |
| `track_obj_pos_err_cm_mean` | 越低越好 | −0.281 | 3 / 12 | +0.528 |
| `track_obj_ori_err_deg_mean` | 越低越好 | −0.209 | 2 / 13 | +0.511 |

**读法**：均值全都很小甚至有利，**代价完全在尾部**。物体侧（obj_pos/obj_ori）一如 E207/E209 是净收益；机器人侧出现极端个例。

### 对照：同一干预在 orig 上（`A→B`，n=5）

`narrow +0 / −1`；`eef_ori +1.505`、`hand_pen +0.051`、`contact −0.001`。

### 交互项（C4，本实验才能算）

| 指标 | aug 上 (C→D) | orig 上 (A→B) | **aug 侧超出** |
|---|--:|--:|--:|
| eef_ori (°) | +2.811 | +1.505 | **+1.306** |
| contact | −0.055 | −0.001 | **−0.055** |
| hand_pen | −0.015 | +0.051 | −0.067 |
| obj_pos (cm) | −0.281 | +0.143 | −0.424 |

**gravcomp 在增强数据上的机器人侧代价约为 orig 上的 1.9 倍（eef_ori），而物体侧收益反而更大。**

---

## 三、核心发现

### F2 · 【最重要】代价有**两种互斥形态**，任何单一检测指标都会漏掉一种

5 条 narrow 损失全部来自 2 个 case，且两者机理完全不同：

| case | 有效位移 | C→D narrow | 失效门 | contact 变化 | eef_ori 变化 |
|---|--:|---|---|--:|--:|
| **021_p1** | 0.134 m | 3/3 → **0/3** | 只有 `contact` | **0.689 → 0.200（−0.49）** | 12.36 → **10.96（更好）** |
| **059_p1** | 0.200 m | 3/3 → **1/3** | `eef_ori`/`root_ori`/`eef_pos`/`root_pos` | 0.711 → **0.778（更好）** | 14.24 → **36.72（+22.5）** |
| 075_p1 | 0.200 m | 3/3 → 3/3 | — | 0.843 → 0.843 | 16.83 → 15.71 |
| 021_p2 | 0.200 m | 0/3 → 0/3 | （C 已全败） | — | — |
| 073_p1 | 0.200 m | 0/3 → 0/3 | （C 已全败） | — | — |

- **021_p1 = 接触塌陷**，姿态反而更好；
- **059_p1 = 末段姿态崩溃**，接触反而更好。

**这直接反驳了 plan240 我自己做的判据改写。** 我依据 E209（"`contact_in_mask` 不是虚扶检测器，`eef_ori` 才是"）把 C5a 从 contact 换成了 eef_ori——**换掉的那个指标恰好是唯一能抓到 021_p1 的**；而 E207 原本的 contact-only C5 又会漏掉 059_p1。E209 的结论在 desk/chair 上成立，但**它是"contact 不充分"，不是"contact 无用"**，我把前者读成了后者。**两个指标都必须留。**

### F3 · 【方法学】跨视频 A/B 无效——**即使参考文件逐字节相同**

plan240 P9 我写的是「C↔D 的画面 A/B 安全，因为两侧参考轨迹逐字节相同」。**这是错的。**

实测（`render/ab/refpane_camera_artifact_t4.95.png`）：C 与 D 的**参考栏**取自同一个 `trajectory_kinematic.npz`（sha `b30178d9…`，manifest 已验证相同）、同一帧，却渲染成**明显不同的姿态**——C 侧蹲伏、D 侧直立举臂。

根因：`_auto_video_camera` 用 **sim ∪ ref** 的并集包围盒逐帧算机位；`sim` 不同 → 机位不同 → 连同一份参考都会看成两个姿势。这比 E208 F13 的表述更强：F13 说跨视频比较**绝对位姿/屏幕位置/大小**没意义，实测连**关节姿态本身**（蹲伏 vs 直立）都不可比。

**唯一有效的视觉判据 = 同一视频内的 sim vs ref**（共享机位）。本 log 的所有视觉结论都按此口径得出。

### F4 · 损害高度集中，不是均匀退化

15 条里 5 条掉门，全部来自 2 个 case；075_p1 三档完全不受影响（contact/eef_ori 都持平）。加上 021_p2 / 073_p1 在 C 阶段就已全败（与 gravcomp 无关），**gravcomp 真正伤到的只有 2/5 个 case**。

### F5 · 059_p1 正是 E207 的过冲 case —— 与 E209 的收缩模型一致

E207 F4 记录的 3 个 |bias| 退化 case 中，`bucket007_20231020_059_p1` 是**原本最不下沉**的一个（PRG |bias| 0.243 cm → G1only 0.871）。E209 证明 gravcomp 是**收缩型**修正（对原本不沉的 case 过冲）。E210 里它就是姿态崩溃的那个 case。**"orig 上被过冲"与"aug 上会崩"指向同一批 case**——这给了一个可用的事前筛选信号（不需要跑 aug 就能预测风险）。

### F6 · 我在首版报告里把 contact 的方向搞反了（已修）

首版 evaluator 用统一的 `n_worse = count(Δ>0)`，对 `contact_in_mask`（越高越好）是**反的**，会把"9 条变差"报成"9 条变好"，直接翻转结论。同时 14-gate 的 `contact` 门读的是 `hand_object_physics_contact_in_mask_frac`，而我另外追踪的是 RL-export schema 里的 `..._3mm_...`，**是两个不同字段**。两处都已修：所有对比算术移到 `gen_E210_four_cell_workbook.py` 单一实现，带 per-metric 方向表。

### F1 · 队列对「中断」不是 resume-safe（已在 progress 登记）

`run_local_priority_queue.py:30` 的 `ELIGIBLE` 不含 `running`。它能从产物文件重认领**已完成**的行，却把**被打断**的行变成墓碑：重启后静默跳过并报 `0 pending`，看起来像跑完了。E210 首轮 8 条中招（23:48 派发、进程与队列一并死亡、状态没写回），第二个队列实例只跑了剩下 7 条。新增 `E210/reset_stale_rows.py` 复位（三道安全闸：产物齐全不动、log 有输出不动、近 N 分钟被碰过不动——共享 /mnt 上"别的机器在跑"是真实场景）。E202/E208 共用同一队列，会复现。

---

## 四、视觉复核（rule 9）

工具：`run_E210_render_all.sh`（osmesa）+ ffmpeg 抽帧。**口径见 F3：只做同一视频内的 sim vs ref**；跨视频只用于「手是否贴在桶面上」这类**帧内关系**属性，并已标注机位不可比。

**实际观察**：

1. **059_p1 末段姿态崩溃（`simref_posture_*_t4.40.png` / `_t4.95.png`）**：t=4.4s 与 4.95s，**D 的 sim 仍然趴在桶上**，而它自己的参考栏里机器人**已经站直、抬起一只手**完成放置；C 的 sim 与自己的参考接近得多。逐帧数值吻合：t=4.40 root_ori C=32.1° / D=111.7°，t=4.80 C=2.2° / D=121.8°，峰值 t=5.04s 达 129.8°。**失效形态 = 物体失重后机器人没有完成"放下并起身"，一直挂在桶上**，误差集中在最后 1 秒（前 3.5 s 两者几乎重合）。
2. **021_p1 接触塌陷（`contact_hands_*_t1.84.png`）**：t=1.84s，C 的左手**指腹贴在桶的远侧壁上**，D 的同一只手**悬在桶面外侧、与桶壁有可见间隙**。物理接触逐帧统计证实：**C 有 65/116 帧手-桶接触，D 只有 19/116**（47 帧只有 C 有）；对照组 075_p1 是 C 43 / D 46（无差异）。
3. **无跌倒、无穿地、无发散抖动**：15/15 `fall_flag=0`，画面上机器人姿态整体可信，没有 E202 bucket003 那种仰面倒地或膝盖穿桶。
4. **中段（t≈2–3.5s）两臂几乎重合**，肉眼分不出——这也是为什么最初按固定比例（30%/55%/80%）抽帧什么都没看出来；**必须先用逐帧数值定位分歧峰值再抽帧**。

---

## 五、Claims 判定

| Claim | 门 | 实测 | 判定 |
|---|---|---|---|
| **C0** 复用无漂移 | 15 条轨迹/掩码 sha == E202 且 == 磁盘；0 条新 retarget | 15/15 | ✅ |
| **C1** 单变量合同 | compose diff == `{scene_name}`；hand-gate == A0；sidecar 15/15 | 场景 15/15 + 配置 1 key + 运行时 15/15（9 项/条） | ✅ |
| **C2** 执行闭合 | 15/15 cem_ok，0 fail/diverge | 15/15，0 fail（经 F1 恢复） | ✅ |
| **C3** aug×G1 不劣于 orig×G1 | `rate(D) ≥ rate(B) − 0.15` 且 fall=0 | 0.267 vs 0.400，gap **0.133**；fall 0/15 | ✅（压线） |
| **C4** gravcomp 在 aug 上的效应 | 出结论即可，不设门 | 交互项已给：eef_ori 超出 +1.31°，物体侧收益反而更大 | ✅ |
| **C5a** 虚扶未在 aug 上放大 | eef_ori 超出 ≤ +1.0° | **+1.306** | ⛔ |
| **C5b** 手穿透未失控 | `hand_pen rate(D) ≥ rate(B) − 0.20` | 0.867 vs 0.800（**更好**） | ✅ |
| **C6** 有效位移量化分层 | 全部报出；<0.05 m 剔除 | 12 full / 3 partial；**0 条低于 0.05 m 下限**；min 0.134 | ✅ |
| **C7** 075_p2 显式登记 | 契约断言 + log 明列 | 断言 E202 命中 0 行，回填会主动报错 | ✅ |
| **C8** 视觉无新增失效 | 具体描述，不得留空 | 见第四节；无跌倒/穿地/抖动，两种失效形态各有画面证据 | ⚠️ |

**总判定：PARTIAL SUCCESS**

依 plan240：SUCCESS 需 C0–C3 全过 **且** C5a/C5b 达标。C5a 破 → PARTIAL。

**C3 通过但不该被当成好消息**：它是比率门（0.133 ≤ 0.15）压线过的，而同一批数据的严格单变量对比是 **+0/−5**。C3 之所以还能过，是因为基线 B 自己就只有 40%（E207 在这 5 例上本来就弱）。**用一个弱基线做"不劣化"判据，会让真实退化被吸收掉**——这是 C3 这条门的设计缺陷，如实记录，不回溯改门。

**显式不作为判据**（plan240 预声明）：z bias/MAE 的任何数值门（E209 已证 E207 常量模型不可外推）；`B→D` 的因果归因（含 omnirt_v1→v2 confound，只声称"不劣、可用"）；真实重力可执行性。

---

## 六、结论与建议

1. **这 15 条不建议直接当 RL 训练数据出片**，除非按 case 过滤：`021_p1`（接触塌陷）与 `059_p1`（末段姿态崩溃）共 6 条应剔除，剩 9 条中 `021_p2`/`073_p1` 在无 gravcomp 时就已不过门。**真正干净的只有 `075_p1` 的 3 条。**
2. **若要用 gravcomp 扩增强数据，先按 E207 的 orig 侧 |bias| 做事前筛选**（F5）：orig 上被 gravcomp 过冲的 case，在 aug 上有崩溃风险。这个信号不需要跑 aug 就能拿到。
3. **14-gate 需要同时保留 contact 与 eef_ori 两条**（F2）。E209 建议新增的 hand-object 相对姿态门仍然值得做，但不能替换 contact。
4. 承接 E209 的建议：**部分补偿 `gravcomp ≈ 0.5`** 在 aug 上可能比全补更合适——059_p1 这类过冲 case 正是收缩型修正的受害者。

---

## 七、结果路径

| 类型 | 路径 |
|---|---|
| CEM | `results/E210/s6_downstream/cem/full/E210_<case>_aug_<trans>_G1only*`（15） |
| 四格评测 | `results/E210/s6_downstream/eval/four_cell/four_cell_rollout.tsv`（40 行） |
| 判据/对比 | `.../four_cell/{e210_four_cell_verdicts.json, e210_four_cell_summary.json}` |
| 工作簿 | `.../four_cell/E210_four_cell_comparison.xlsx`（README/PerCell/Contrasts/PerCase/PerRollout/Verdicts） |
| 运行时审计 | `results/E210/preflight/e210_runtime_config_audit_full.json`（15×9 断言） |
| sidecar 证据 | `results/E210/preflight/e210_gravcomp_sidecars.json` |
| 渲染 | `results/E210/s6_downstream/render/full/*.mp4`（15）+ `render/ab/*.png`（抽帧） |
| manifest | `results/E210/s6_downstream/manifests/aug_g1only_full_manifest.tsv`（15） |
| 场景快照 | `results/E210/scene_snapshot/`（15 task / 106 文件 + manifest.txt，HEAD 1c0a099） |

> `results/` 是指向外部盘的符号链接，**快照本身进不了 git**（E199/E202/E206/E208 同）；rule 7 保障 2 以 manifest.txt 的 sha256 记录形式满足，保障 1 靠 P0 的 90 个 `git add -f` 文件。

## 八、改动文件

**新建**：`plan/240_*`、`log/299_*`、`scripts/experiments/E210/{e210_common,build_gravcomp_sidecars,build_overrides,build_manifest,audit_runtime_config,reset_stale_rows}.py`、`scripts/launch/active/run_E210_{local_8gpu,render_all}.sh`、`scripts/eval/runners/eval_E210_aug_g1only.py`、`scripts/eval/reports/gen_E210_four_cell_workbook.py`、`examples/config/override/core4d_E210_*_PRG_gravcomp.yaml`（15）、15 个 `scene_act_E210_bucketAlignedTop_PRG_gravcomp.xml`

**修改**：`EXPERIMENT_TRACKER.md`（R296）、`progress.md`

**`spider/` core 与 `examples/run_mjwp.py` 零改动**；E202/E207/E208/E209 历史产物零覆盖（只读 import）。

## 九、下一步

1. 若要继续这条线：先做 **partial gravcomp (0.5)** 在同样 15 个变体上的一格，与 D 直接对比（同样零 retarget，15 条 CEM）。
2. bucket 的 **rot0/rot1 从未真正跑过**（E208 F6：E202 的 trans-only 是 scipy 崩溃导致的误判）。若要给 bucket 扩产能，这是现成的一倍，且 `075_p2` 也才有救回的可能。
3. `E210/reset_stale_rows.py` 的逻辑应上提到 `E199/run_local_priority_queue.py`，让所有共用队列的实验都受益（E202/E208/E210）。
