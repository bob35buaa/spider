# log296 · E207：bucket G1-only（object gravcomp 单变量）

_Core4D · Phase 67 · Run **R293** · 承接 [plan237](../plan/237_E207_bucket_g1only_gravcomp_plan.md) · 2026-09-04～09-05 · 分支 `feat/E207-bucket-g1only-gravcomp` · **状态：总判定 PARTIAL SUCCESS（偏强）**_

> **最终判定见第七节。C4 是唯一未达门项，且只输在「改善 case 数」子句。**

## 一句话进展

在 E178 的 9 个 bucket case 上以严格单变量加入 object `gravcomp=1`：**系统性欠抬升被消除**（z bias −1.379 → +0.566 cm，|bias| 均值 −59%），14-gate 不劣化（3/9 vs PRG 4/9，压在门上），承重接触不降反升；但 gravcomp 的修正是**常量偏移**，对原本几乎不下沉的 case 会过冲，故预注册的「≥7/9 改善」只拿到 6/9。

---

## 一、执行

| 阶段 | 结果 |
|---|---|
| P0 分支/计划/git 保障 | 9 case × 6 个 scene 文件 `git add -f`（补 rules §7 保障 1 的历史缺口，此前只有 E174/E188 被追踪） |
| P1 契约 | `assert_gravcomp_diff` 9/9 PASS；反向测试确认守卫会触发 |
| P2 override | 全 key compose diff **恰为 `{scene_name}`**，9/9 |
| P3 manifest | 9 行，CEM 1024×32 seed0；轨迹/掩码 sha256 与 E178 逐 case 相同 |
| P4 快照 | `results/E207/scene_snapshot/` 9 case / 86 文件 + manifest.txt（HEAD c43b84b） |
| P5 smoke | 1 case 64×4，7 项运行时断言全过 |
| P6 full | **9/9 cem_ok，0 fail**，8×L20Y，1h10m |
| P7 评测 | 四臂 14-gate + 三臂 z 诊断 + E207ARM review set |
| P8 视觉 | 8/9 已渲染（osmesa），A/B 抽帧复核完成 |

---

## 二、单变量合同（C1）如何被机器验证

三处独立验证，任一不符即中止：

1. **场景**：`scene_act_E205_contactAlignedTop_gravcomp.xml` 经 `e200_common.assert_gravcomp_diff` 逐字节确认 == `scene_act_E178_contactAlignedTop.xml` + object `gravcomp="1"`，9/9。
2. **配置**：Hydra compose 后逐 key diff，交叉验证非空转——
   ```
   E178 → E207   1 key  (scene_name)              ← 隔离 gravcomp
   E178 → E205   3 key  (scene_name + 2 个 A2)
   E207 → E205   2 key  (仅 hand-gate)            ← 隔离 A2
   ```
3. **运行时**：9/9 回读 `config_act.yaml`，`cem_hand_gate_max_violation_pct=0.10`、`hard_floor=−0.020`（A0 默认，非 A2 的 0.05/−0.015）、`leg_object_penalty_scale=2.0`、`cem_leg_gate_enabled=true`、`init_pos_actuator_gain=500`、`init_rot_actuator_gain=50`。

### F1 · plan237 M3 的一处修正
E205 override **文件**写 4 个 key，但 `cem_hand_gate_min_sdf_m: -0.01` 与 E163 默认相同，**实际 compose 差异是 3 个 key**（A2 只真正改了 2 个值）。

### F2 · plan237 P3 退出检查笔误
原文要求 `effective_scene_sha256` 与 E178 相等——但 E207 用的就是 gravcomp scene，本就该不同。已改判为「等于该 sidecar 实际 sha 且不等于 E178」，全过。

---

## 三、主判定：四臂 14-gate（同 E201 漏斗，n=9）

| arm | hard | wide_all | **narrow_all** | L3_auto |
|---|---:|---:|---:|---:|
| PRG (E178) | 9/9 | 6/9 | **4/9 (44%)** | 4 |
| noPRG (E204) | 9/9 | 6/9 | 3/9 | 3 |
| G1A2 (E205) | 9/9 | 4/9 | 3/9 | 3 |
| **G1only (E207)** | 9/9 | 5/9 | **3/9 (33%)** | 3 |

逐门 narrow 通过数（只列有差异的）：

| gate | PRG | noPRG | G1A2 | G1only |
|---|---:|---:|---:|---:|
| eef_ori | 7 | 6 | 5 | **5** |
| hand_pen | 8 | 6 | 7 | 7 |
| release | 6 | 6 | 6 | 6 |
| 其余 11 门 | 9 | 9 | 9 | 9 |

`PRG → G1only` paired：narrow **+0 / −1**。掉出的是 `bucket007_20231023_073_p1`，输在 **eef_ori**（17.49° → 19.83°，门 20°）。

### F3 · gravcomp 的代价集中在末端姿态与释放期

| 指标 | PRG | G1only | Δ |
|---|---:|---:|---:|
| track_obj_pos_err_cm_mean | 8.162 | **7.830** | −0.333 |
| track_obj_ori_err_deg_mean | 4.661 | 4.623 | −0.038 |
| hand_object_physics_contact_in_mask_frac | 0.7838 | **0.7965** | **+0.013** |
| track_eef_ori_err_deg_mean | 17.490 | 19.829 | **+2.339** |
| hand_object_release_false_contact_3mm_frac | 0.1220 | 0.1935 | **+0.072** |
| hand_object_physics_penetration_3mm_frame_frac | 0.1834 | 0.2184 | **+0.035** |
| leg_penetration_frac | 0.0031 | 0.0060 | +0.003 |

物体跟踪改善、承重接触**上升**（C5 通过，未出现 E194 C4 担心的"虚扶"塌陷），代价是末端朝向、释放期假接触与手穿透变差。

---

## 四、z 诊断与机制（本次最有价值的发现）

| Arm | z bias (cm) | \|bias\| 均值 | bias std | z MAE (cm) | 3D pos (cm) |
|---|---:|---:|---:|---:|---:|
| PRG | −1.379 | 1.433 | 0.978 | 2.563 | 8.162 |
| **G1only** | **+0.566** | **0.590** | **0.531** | 2.329 | 7.830 |
| G1A2 | +0.622 | 0.719 | 0.632 | 2.198 | 7.844 |

**|bias| 均值降 59%，离散度（std）也降 46%** —— 不只是均值回零，逐 case 也更集中。

### F4 · gravcomp 是常量偏移，不是按需修正

逐 case 提升量 `Δ = G1only bias − PRG bias`：**+1.945 ± 0.711 cm**（范围 +0.628 ~ +3.225）。

```
Δ vs 参考抬升幅度 ref_z_range   pearson r = +0.110   ← 几乎无关
Δ vs PRG 原始 bias              pearson r = -0.848   ← 强负相关（原本越沉，抬得越多）
Δ vs 物体质量                    pearson r = +0.710
```

`Δ` 与抬升幅度**无关**、与质量相关，说明它修的是一个**与运动无关的静态偏置**——正是 E194 的 `sag = m·g/kp`。

**退化的 3 个 case 恰好是 PRG |bias| 最小的 3 个**（0.243 / 0.362 / 1.014，全 9 例中最小的三个），是确定性过冲而非噪声：

| case | PRG \|bias\| | G1only \|bias\| |
|---|---:|---:|
| bucket007_20231020_059_p1 | 0.243 | 0.871 |
| bucket003_20231020_068_p1 | 0.362 | 1.258 |
| bucket003_20231018_005_p2 | 1.014 | 1.432 |

### F5 · 提升量只有自由悬挂预测的 29~43%，差额由手分担

物体质量：2 kg × 7 例、5 kg × 2 例。

| 质量 | n | 实测 Δ | 预测 `m·g/kp` (kp=500) | 实测/预测 |
|---|---:|---:|---:|---:|
| 2 kg | 7 | +1.691 | 3.924 | 0.43 |
| 5 kg | 2 | +2.836 | 9.810 | 0.29 |

5kg/2kg 的提升量比 = 1.68，小于质量比 2.50 —— 与"手通过接触分担了部分重量、伺服并未承担全部 m·g"一致，且重物分担比例更高。这修正了把 `m·g/kp` 当作实际下沉量的朴素读法。

### F6 · A2 对 z 的贡献可忽略（C7）

`G1only` 与 `G1A2` 的 z bias 逐 case 平均只差 **0.169 cm**，obj_pos 7.830 vs 7.844。**物体跟踪的全部改善来自 gravcomp，A2 hand-gate 贡献≈0** —— 这正是 E207 要拆开的东西，E205 的 G1A2 此前无法回答。

---

## 五、视觉复核（rule 9）

工具：`run_E207_render_all.sh`（**本机仅 osmesa 可用**，见 F7）+ ffmpeg 抽帧，与 E178 同 case 同时刻 A/B。
复看：`bucket003_20231018_001_p2`（bias 改善最大 −3.148→+0.076）、`bucket003_20231020_068_p1`（过冲最严重 0.362→1.258）。

**实际观察**：

1. **抬升峰值（001_p2, 帧124, t=4.96s；E178 z_diff −11.75 cm / E207 −5.16 cm）**：E178 画面里桶**直立贴地、底边与阴影相连**，机器人俯身趴在桶上，参考要求的 0.548 m 完全没抬起来；E207 同一时刻桶被**向后倾倒拉起**、底边与阴影分离。两臂是**不同策略**而非单纯高度平移。
2. **最大过冲（068_p1, t=2.64s, +9.32 cm，z_sim 0.500 vs z_ref 0.406）**：桶**完全离地、倾斜约 45°**，阴影明显分离；机器人右手确实扣在桶沿上（橙色接触标记贴合）。→ **没有出现"物体飘着而手脱开"的虚扶**，与 contact_in_mask +0.013 一致。
3. **但承重是"假"的**：同一时刻 E178 用**膝/大腿抵住**桶体协助支撑，E207 仅靠桶沿单手扣持就把 2 kg 桶悬在空中并高出参考 9 cm。真实重力下该姿态会翻转/脱手。这是 plan237 已声明接受的建模落差的**具体可见形态**，不是新缺陷，但下游用这批轨迹时必须知道。
4. **手-桶互穿**：001_p2 峰值处前臂与桶近侧壁有可见相交，与 hand_pen +0.035 对应。
5. 未见物体穿地、未见抖动/发散、机器人姿态整体可信。

---

## 六、我自己引入的问题（全部已修，记录以免重犯）

### F7 · 本机 egl/glfw 均不能渲染
实测：`egl` → `EGLError`，`glfw` → 无 `_mjr_context`，**只有 `osmesa` 可用**。若照抄 E178 的 `MUJOCO_GL=egl` 会在 P8 直接卡住。`run_E207_render_all.sh` 已固定 osmesa。CEM 本身 headless，不受影响。

### F8 · `relative_to(REPO)` 在符号链接下抛异常
`workspace/core4d/results` 是指向外部盘的符号链接，`out.relative_to(REPO)` 在自定义 `--out-dir` 下抛 `ValueError`。**预验证阶段（用已有 E178/E204/E205 数据跑通评测脚本）抓到的**，否则会在拿到结果后才炸。已加 `rel_label()`。

### F9 · 模块同名自我遮蔽
E207 的 review TSV builder 原本也叫 `build_arm_review_tsv.py`，而脚本把 E207 目录插在 `sys.path` 最前，导致 `import build_arm_review_tsv` 导入了**自己**而非 E204_E205 的同名模块（`AttributeError: no attribute 'GATE_MAP'`）。已改名 `build_e207_arm_review_tsv.py` 并在 docstring 注明原因。

### F10 · review player 的 "available review sets" 是硬编码
用户执行 `review_player.sh E205` 得到"never wired into the review player"，据此以为 E205 没有 viser 支持。实际 **E204ARM 一直可用且包含 E205 的 G1A2 臂**——是那句提示的硬编码列表漏掉了 `E198/E194_FULL/E203F/E204ARM/E206ARM`。已改为从 `SOURCE_OVERRIDES` 注册表推导（现列 22 个），并顺带注册了 `E205` 单臂视图（27 case 全可播）。

---

## 七、最终判定

| Claim | 门 | 实测 | 判定 |
|---|---|---|---|
| **C0** 数据复用无漂移 | trajectory/mask sha == E178；0 条新 retarget | 9/9 相同 | ✅ |
| **C1** 单变量合同 | compose diff=={scene_name}；hand-gate==E163；scene 9/9 | 场景 9/9 + 配置 1 key + 运行时 9/9 | ✅ |
| **C2** 执行闭合 | 9/9 cem_ok | 9/9，0 fail | ✅ |
| **C3** 14-gate 不劣化 | narrow ≥ PRG−1 = 3 | **3/9**（压线，掉 1 例于 eef_ori） | ✅ |
| **C4** z 欠抬升被消除 | \|bias\|≤0.8 **且** ≥7/9 改善 | \|bias\|=**0.566** ✅ ／ 改善 **6/9** ⛔ | ⛔ |
| **C5** 承重接触未塌 | contact_in_mask 降幅 ≤0.05 | **+0.013**（上升） | ✅ |
| **C6** 视觉无新增失效 | 无虚扶/穿透/抖动 | 无虚扶、无穿地、无抖动；手穿透略增(+0.035)、承重非真实 | ⚠️ |
| **C7** A2 效应被隔离 | 给出 E207→E205 paired | z bias 仅差 0.169 cm → A2 对 z 贡献≈0 | ✅ |

**总判定：PARTIAL SUCCESS（偏强）**

依 plan237：SUCCESS 需 C0–C3 全过 **且** C4 达标；C4 未达 → PARTIAL。

**为什么不是 SUCCESS**：C4 的「≥7/9 改善」只有 6/9。

**为什么说"偏强"**：C4 的**幅度子句大幅超额达成**（0.566 vs 门 0.8，且 |bias| 均值降 59%、std 降 46%），未达的是计数子句；而 F4 已定量解释其成因——gravcomp 是 +1.95 cm 的**常量**修正，对原本 |bias| 只有 0.24/0.36 cm 的 case 必然过冲。**预注册该子句时假定"干预对每个 case 都应有益"，这个假定对常量偏移型干预本就不成立**。此处如实记录，不回溯改门。

**显式不作为判据**（plan237 预声明）：z MAE 降幅（实测 2.563→2.329，4/9 改善）、2×2 交互项、真实重力可执行性。

---

## 八、结果路径

- CEM：`results/E207/s6_downstream/cem/full/E207_<case>_G1only/trajectory_mjwp_act.npz`（9）
- 四臂 14-gate：`results/E207/s6_downstream/eval/four_arm/four_arm_rollout.tsv`（36 行）
- z 诊断：`.../four_arm/{E207_object_z_diff_report.md, e207_object_z_diff_by_case.tsv, e207_object_z_diff_summary.json}`
- review set：`.../four_arm/e207_arm_case_metrics.tsv`（36 行全可播，仅 G1only 有 mp4）
- 渲染：`results/E207/s6_downstream/render/full/*.mp4`
- manifest：`results/E207/s6_downstream/manifests/g1only_full_manifest.tsv`（9 行）
- 场景快照（rule 10b）：`results/E207/scene_snapshot/`（9 case / 86 文件 + manifest.txt）
- E178 侧 z 基线：`results/E178/s6_downstream/eval/full/E178_object_z_diff_report.md`

## 九、改动文件

**新建**：`plan/237_*`、`log/296_*`、`scripts/experiments/E207/{e207_common,build_overrides,build_manifest,build_e207_arm_review_tsv}.py`、`scripts/launch/active/run_E207_{local_8gpu,render_all}.sh`、`scripts/eval/runners/eval_E207_g1only.py`、`scripts/eval/reports/{gen_E207_object_z_diff,gen_E178_object_z_diff_report}.py`、`examples/config/override/core4d_E207_*_G1only.yaml`（9）

**修改**：`EXPERIMENT_TRACKER.md`（R293 行）、`scripts/eval/review/{review_index.py,viser_review_player.py}`、`scripts/eval/wrappers/review_player.sh`（E205/E207ARM 注册 + F10 修复）

**`spider/` core 与 `examples/run_mjwp.py` 零改动** —— E207 只换 scene sidecar 与 override。

## 十、下一步

1. **不建议**据此把 gravcomp 设为 bucket 默认：它对 |bias| 已小的 case 会引入 ~+1 cm 过冲，且 eef_ori/release/hand_pen 三项变差。更合理的形态是**按 case 的实测 sag 缩放补偿量**（部分补偿，如 0.5·m·g），而非全补——F4/F5 给出了定标依据。
2. 若要把结论推到 **bucket004**（本次缺席，抬升最高、E178 下欠抬升最重），需单独补跑；当前结论明确不外推。
3. `eef_ori` 是四臂共同的瓶颈（PRG 7、G1only/G1A2 5），且是 E207 唯一掉门项，值得单独立项。
4. C6 的"承重非真实"若对下游 RL 有影响，可在 gravcomp 轨迹上做一次真实重力 replay 检验（plan237 中用户已选择不做，此处仅登记）。
