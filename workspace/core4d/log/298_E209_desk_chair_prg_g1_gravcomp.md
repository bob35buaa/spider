# log298 · E209：desk/chair PRG + G1（object gravcomp）单变量

_Core4D · Phase 68 · Run **R295** · plan239 · 2026-09-05 · 分支 `feat/E207-bucket-g1only-gravcomp` · **FAIL（主门 C3 破，但取得决定性机制结论）**_

## 一句话进展

gravcomp 把 desk/chair 的物体 z 下沉从 **−2.517 cm 修到 +1.041 cm**（22/22 全跑通、17/22 改善、Wilcoxon p=7.3e-5），**并证伪了 E207「gravcomp 是 +1.945 cm 常量偏移」的机制结论**（实测 r(pre,delta)=**−0.915**、OLS slope=**0.408**，是收缩型不是加性型）；**但 14-gate 主门退化**（narrow 13→10、L1 3→11），退化几乎全部集中在**末端姿态 eef_ori**（narrow 失败 2→9），机理是**物体失重后 CEM 可以用一个姿态跑偏的机器人「托」着物体走**——即 E194 C4 担心的「虚扶」，且**接触率门看不见它**（contact 仅 −0.016）。

---

## 一、判定表

| Claim | 门 | 实测 | 判定 |
|---|---|---|---|
| **C0** 输入零漂移 | 22/22 traj+mask+baseline scene sha256 == E206 交付表 | 22/22 | ✅ |
| **C1** 单变量·配置层 | compose 对称差 == `{scene_name}` | 22/22 | ✅ |
| **C1b** 单变量·场景层 | `assert_gravcomp_diff` + 编译层 ngeom/npair/nq/nv/nu/nbody 相等 | 22/22 | ✅ |
| **C1c** 单变量·运行时 | `config_act.yaml` 全键 diff 仅 THE_VARIABLE；A0 hand-gate | 22/22 | ✅ |
| **C2** 执行闭合 | 22/22，0 fail | 22/22，`problem rows: []` | ✅ |
| **C3** **主门 · 14-gate 不劣化** | narrow ≥ 12/22 ∧ hard == 22/22 | **narrow 10/22**（基线 13），hard 22/22 | ❌ |
| **C4** z 修正 · S⁻(n=18) | ①\|宏\|≤1.0 ②≥16/18 改善 ③溢出≤2 | ①**0.603** ✅ ②**17/18** ✅ ③**3** ❌ | ❌（仅③破） |
| **C4b** z 危害上限 · S⁺(n=4) | \|宏\|≤3.99 ∧ 无单例>6.0 | +3.011，最差 3.310 | ✅ |
| **C4c** 全 22 例反挑拣 | \|宏\|<2.517 ∧ mean\|b\|<3.241 ∧ p<0.05 | **1.041** / **1.104** / **p=7.3e-5** | ✅ |
| **C4d** 传递函数判别 | 胜者 RMSE≤1.0 ∧ 比次者低≥2× | 胜者 **M_shrink** 1.103；比值 1.54× | ❌（方向明确，量化未达门） |
| **C5** 承重接触不塌 | contact≥0.85 ∧ leg_pen≤0.02 | 0.863 / 0.0089 | ✅ |
| **C6** 副作用天花板 | release≤0.18 ∧ hand_pen≤0.26 | 0.098 / 0.216 | ✅ |
| **C7** 人审不降级 | 22/22 已审 ∧ USE≥18 | **待 P8** | ⏳ |
| **C8** 吞吐无回归 | median ∈ [33.0, 55.0] min | **42.2**（基线 44.0） | ✅ |
| **C9** 复现性双保障 | git add -f + 快照含双臂 sha | 132 文件 + 22 目录 | ✅ |

**总判定 = FAIL**（plan239 定义：C3 narrow ≤ 11/22 即 FAIL）。这是**干预在主判据上真实失败**，不是流水线问题——C0–C2 与 C8 全过说明执行侧完全干净。

---

## 二、z 诊断：干预本身是有效的

| | E206 PRG（基线） | E209 G1 | Δ |
|---|---:|---:|---:|
| z bias 宏平均（全 22） | −2.517 | **+1.041** | +3.558 |
| mean\|bias\| | 3.241 | **1.104** | −2.137 |
| 下沉 case 数 | 18/22 | 2/22 | |
| obj 3D pos 误差 (cm) | 11.08 | **9.92** | **−1.16** |
| obj_ori (°) | 5.64 | **4.86** | **−0.78** |

**物体侧全线改善**：z 下沉基本消除，3D 位置与姿态跟踪都变好。C4 只输在「溢出」子句——3 例（`chair006_20231003_2_015_p1` / `desk007_20231030_028_p1` / `desk007_20231030_034_p1`）的 \|post\| 超过 1.5 cm，门允许 2 例。幅度子句（0.603）不仅过门，还过了 E207 的 0.8 stretch 线；改善计数 17/18 也远超门。

**S⁺ 子群（chair006 4 例）如预注册所料继续过冲**（+1.994 → +3.011），但落在 C4b 的危害上限内。这是 P0 就冻结的分层，不是事后划线。

---

## 三、**核心机制结论：gravcomp 是收缩型，不是加性型 —— E207 的结论被证伪**

E207 报的是「+1.945 ± 0.711 cm 常量偏移」。E209 实测：

| 判据 | 加性型预测 | 收缩型预测 | **E209 实测** |
|---|---|---|---|
| `r(pre_bias, delta)` | ≈ 0 | 显著负 | **−0.915** |
| OLS slope | ≈ 1.0 | < 1 | **0.408** |
| delta 是否恒定 | 是 | 否 | +3.558 ± **1.576**（SD 是 E207 的 2.2×） |

实测 OLS `post = 2.069 + 0.408·pre`，与 E207 拟合的 `M_shrink`（1.094 + 0.383·pre）**斜率几乎一致**（0.408 vs 0.383）。三模型 RMSE：**M_shrink 1.103 < M_mass 1.700 < M_pooled 2.230**。

**为什么 E207 会得出「常量」**：它的 9 例 pre-bias 只覆盖 [−3.148, +0.243]，且物体质量与 pre-bias 共线（mass=5 的 2 例 pre 均值 −2.08 vs mass=2 的 7 例 −1.18）。在那么窄的区间里，收缩型与加性型的预测差小于噪声。E209 的 22 例质量全为 5.000 kg（零方差）而 pre-bias 跨 −5.646…+2.668，这才把两族分开。

**C4d 形式上 FAIL 但不推翻上述结论**，要如实区分两件事：
- **模型族已判定**（r=−0.915、slope=0.408 是决定性的，与门无关）；
- **E207 拟合的具体系数不可直接外推**（M_shrink 残差偏置 −0.910 cm，说明 desk/chair 的截距比 bucket 高）。C4d 的两条门（RMSE≤1.0、比值≥2×）问的是后者，答案是「不够精确」。这两条门本就是按 E207 系数能否平移设计的，判 FAIL 是正确的信息，不是判据设计失误。

**物理读数**：收缩型意味着修正量正比于亏欠量，不动点 ≈ 2.069/(1−0.408) = **+3.50 cm**。理论满悬挂沉降 `m·g/kp = 5×9.81/500 = 9.81 cm`，实测修正 3.558 cm 只占 36%——与 E207「手分担了大部分载荷」的结论一致。

---

## 四、**为什么 14-gate 会退化：接触率门看不见「虚扶」**

逐门配对 delta（G1 − PRG，正 = 变差）：

| 门 | PRG | G1 | Δ | 说明 |
|---|---:|---:|---:|---|
| **eef_ori** (°) | 16.26 | 19.05 | **+2.78**（最差 +33.7） | **主凶** |
| root_ori (°) | 8.18 | 9.99 | +1.81（最差 +32.2） | 次凶 |
| root_pos (cm) | 13.94 | 14.76 | +0.82 | |
| eef_pos (cm) | 13.09 | 13.52 | +0.43 | |
| obj_pos (cm) | 11.08 | **9.92** | **−1.16** | 改善 |
| obj_ori (°) | 5.64 | **4.86** | **−0.78** | 改善 |
| contact | 0.879 | 0.863 | −0.016 | 几乎不动 |
| hand_pen | 0.211 | 0.216 | +0.005 | 几乎不动 |
| leg_pen | 0.0102 | 0.0089 | −0.001 | 改善 |

narrow 失败模式：`eef_ori` **2 → 9**，其余基本持平（hand_pen 5→5、release 2→3、root_pos 2→4）。

**9 例翻转**：3 例转好（chair006 ×2、desk021 ×1，都是原先卡 hand_pen 的），6 例转坏，其中 4 例是 eef_ori 越过 20° 窄门。

**最坏的一例 `desk007_20231030_034_p1` 把机理讲得最清楚**：

| 指标 | PRG | G1 |
|---|---:|---:|
| obj_pos (cm) | 11.38 | **9.84**（更好） |
| obj_ori (°) | 5.93 | 7.21 |
| root_ori (°) | 9.73 | **41.90** |
| eef_ori (°) | 17.65 | **51.31** |
| hand_pen | 0.099 | **0.331** |
| body_z_p95 (m) | 0.059 | **0.170** |
| contact | 0.781 | 0.760（几乎不动） |

**物体跟得更准，机器人自己彻底跑偏，而接触率几乎没变。** 这正是「虚扶 / ghost-carry」的定量指纹：物体失重后，CEM 不再需要用正确的手腕姿态去对抗重力，于是把优化预算让渡出去，允许躯干与末端姿态漂移，物体照样被「托」着走。

**方法学结论（对后续 arm 有用）**：
1. **`contact_in_mask` 不是这类失效的检测器**（−0.016 完全没反应），**`eef_ori` 才是**。E194 C4 当年担心的风险是对的，但盯错了指标；E207 因为 bucket 上 eef_ori 没被触发而没暴露。
2. gravcomp 在**物体侧是净收益、在机器人侧是净代价**。它优化的是 object tracking，代价由 robot tracking 支付。desk/chair 抬升幅度大（25.9 cm）、物体重（5 kg），这个 trade-off 比 bucket 上更不划算。

**「虚扶」这个词要用得准确**：接触率几乎没掉（0.781 → 0.760，desk007_034_p1 亦然），所以**不是「手没碰到物体」**。准确的说法是——**手仍然碰着，但用来托住它的姿态跑偏了**：末端姿态与躯干朝向大幅偏离参考，而物体因为失重，用这种错误姿态也照样被托住。检测它需要看姿态门，不是接触门。

---

## 四之二、视觉复核（rules §9）与一个渲染方法学陷阱

### 先说陷阱：**两臂视频之间不能直接比姿态**

复核 `desk007_20231030_034_p1` 时，我一度从视频里读出「G1 的参考在抬腿后仰、PRG 的参考在站立前倾」，并据此推断 G1 偏离更大。**这个读法是错的，已撤回。**

查证过程：
1. 两臂的 ref 面板逐帧像素不同（2.7%–5.6% 像素差），但纯平移对齐无法消除。
2. 直接取两臂实际渲染用的参考数组比对：**`ref_qpos` 两臂逐元素最大差 = 0.000e+00**（同一条轨迹，与 C0 一致），而 `sim_qpos` 最大差 1.057。
3. 读渲染代码：`spider/viewers/__init__.py:334` 每帧调 `_video_camera(config, model, mj_data, mj_data_ref)`；`_auto_video_camera`（:262-291）的 **lookat 中心与半径由「该帧 sim 与 ref 的 body 位置并集」的包围盒算出**。方位角/俯仰角来自 config（固定 135° / −22°），但**中心与距离随 sim 变化**。

→ 两臂 sim 不同 ⇒ 相机 pan/zoom 不同 ⇒ **同一个参考姿态在两个视频里呈现为不同大小与位置**。FK 是确定性的，且 P1 已证两臂编译出的 `ngeom/npair/nq/nv/nu/nbody` 完全相等，所以 ref 面板**必然**是同一姿态，差异只可能来自相机。我的误读是 pan/zoom 造成的知觉错误。

**这条对 P8 人审直接有影响**：22 例 A/B 对看时，PRG 与 G1 的视频**不是同一机位**。判断「谁偏离参考更多」必须在**单个视频内部比 sim 与 ref**（同帧共享相机，这是有效的）；**跨视频比绝对姿态、比物体在画面里的位置或大小，都是无效的**。

### 有效的视觉观察（单视频内 sim vs ref，帧 135）

| Arm | 单视频内 sim 与 ref 的偏离 | 与数值一致性 |
|---|---|---|
| **PRG** | sim 与 ref 姿态接近：躯干前倾角度、双脚站位、手臂搭在桌沿的位置都对得上 | root_ori 9.73°、eef_ori 17.65° — 一致 |
| **G1** | sim 与 ref 明显分离：躯干朝向与下肢站位对不上 | root_ori 41.90°、eef_ori 51.31° — 一致 |

**结论**：视觉复核**支持**数值判定（G1 在该例上机器人自身跟踪显著恶化），但**不能**从这批视频里额外确认「手是否脱离物体」——接触率数值（0.781 → 0.760）已经回答了这个问题：没有脱离。

### 遗留

要拿到跨臂可直接对比的视频，需要用**固定机位**重渲。scene 里只有两个 `mode="trackcom"` 的具名相机（`track` / `track2`），仍随 COM 移动；真正的固定机位需要给 `_auto_video_camera` 传固定 lookat/distance（config 已有 `video_auto_camera_*` 四个旋钮，但 center/radius 目前是数据推导的，不可外部锁定）。**本次不做**（44 条重渲约 1.5 h CPU，且不影响本实验的任何判定），登记为遗留项 U1。

---

## 五、执行侧

- **P6**：22/22 `run_complete_pending_eval`，`problem rows: []`，wall median **42.2** min（mean 44.6 / min 27.1 / max 67.6），8 卡约 2.1 h。基线 E206 PRG 同 22 例 median 44.0 —— C8 通过，且证明 gravcomp 不改变计算成本。
- **PRG 基线逐位复现**：本次重打分得到 L3/L2/L1 = 13/6/3、narrow 13、hard 22，与 P0 冻结基线完全一致 → 证明 evaluator 无漂移，两臂差异确实来自 gravcomp。
- **C1c 运行时审计**：22/22 证明跑的是 A0 hand-gate（0.10 / −0.020）而非 A2（0.05 / −0.015），`init_pos_actuator_gain=500`，object `gravcomp="1"`。

---

## 六、我自己引入 / 避开的问题

| # | 问题 | 处置 |
|---|---|---|
| F1 | `e200_common.build_gravcomp_sidecar` 输出名硬编码为 `scene_act_E199_rubberHull_PRG_gravcomp`（:161），直接套用会往 desk/chair 目录写名字撒谎的文件 | 自写 12 行 writer；`assert_gravcomp_diff` 逐字复用 |
| F2 | `e200_common.TIER_RANK = {"P1":1}`，E209 manifest 是 `tier=P0` → 派发即 KeyError | 队列用 **E199 版** |
| F3 | 给 `e206_common.ARMS` 加第三 arm 会静默污染在跑的 E208（`e208_common:53` import 它） | 新写 runner，`e206_common` 只读 |
| F4 | 想把 `review_index.py:1011` 的 arm-sweep 元组改成派生式 `SOURCE_OVERRIDES[exp]["arm_sweep"]`——**回归测试发现该 flag 还挂在 E194_FULL/E198/E199/E199P/E200G/E200N 上**，而该 elif 触发条件是 summary.json *不存在*，派生式会静默改掉这 6 个实验 | 放弃派生式，显式元组只加 `E209ARM`，注释写明 |
| F5 | 运行时契约首跑 FAIL：NaN≠NaN 自比不等 + smoke 阶段预算键 + `model_path` | 加 `same()` 处理 NaN；三分类 THE_VARIABLE / RUN_LOCAL / STAGE_BUDGET；`model_path` 改为**正向断言 basename** 而非忽略 |
| F6 | **我自己在视觉复核时误读了视频**，从两臂 ref 面板的差异推断参考姿态不同 | 查出 `_video_camera` 每帧按 sim 数据算 lookat/半径 → 跨视频只有 pan/zoom 差。已撤回该读法，并把「跨视频不可比姿态」写进 §4-2 供 P8 使用 |
| F7 | 「虚扶 / ghost-carry」这个措辞会让人以为手脱离了物体 | 接触率 0.781→0.760 证明没脱离；改述为「手仍接触，但托住它的姿态跑偏」 |

**已存在缺口（非本次引入，本次修补）**：
- **rules §7 保障 1 此前完全为空**——22 个 dcv3 task dir 在 git 里一个文件都没有。已 `git add -f` 132 文件 + 22 个新 sidecar。
- **E206 的 `scene_snapshot/` 拍的是 `<obj>_person<N>/` 源模板，不含真正跑 CEM 的 `scene_act_E206_lowgeom_PRG.xml`**。E209 照 E207 拍 dcv3 task dir，一份快照同时覆盖基线与处理组。
- **`log/296` 撞号**：E207 已落盘占用 296，而 E208 的 commit 也声明用 296 → 裁定 E208 改用 **log297**，E209 用 **log298**。

**协作观察**：本分支有并发提交者。`git add -f` 的 132 个 scene 文件被并发的 `95d0000`（E207 RL 导出提交）一并扫走，落进语义不相干的 commit。后续提交一律显式列路径。

---

## 七、结果路径

| 内容 | 路径 |
|---|---|
| 计划 | `workspace/core4d/plan/239_E209_desk_chair_prg_g1_gravcomp_plan.md` |
| 冻结基线（z / 14-gate / 人审 + sha256） | `results/E209/baseline/` |
| CEM 22 条 | `results/E209/s6_downstream/cem/full/E209_<case>_G1/` |
| 运行时契约审计 | `results/E209/s6_downstream/cem/full/e209_runtime_contract.json` |
| manifest（含全部输入 sha256） | `results/E209/s6_downstream/manifests/e209_g1_full_manifest.tsv` |
| 场景审计（双臂 sha + 编译不变量） | `results/E209/s6_downstream/manifests/e209_scene_audit.json` |
| 14-gate 逐 case / 逐门 / 逐物体 / summary | `results/E209/s6_downstream/eval/two_arm/e209_two_arm_{rollout,per_gate,per_object}.tsv` + `e209_two_arm_summary.json` |
| z 诊断 + 三模型判别 | 同目录 `e209_object_z_diff_by_case.tsv` / `e209_object_z_diff_summary.json` / `E209_object_z_diff_report.md` |
| 渲染 22 条 | `results/E209/s6_downstream/render/full/E209_<case>_G1.mp4` |
| 复审集（E209ARM，双臂带 mp4） | `results/E209/s6_downstream/eval/two_arm/e209_arm_case_metrics.tsv` |
| 场景快照（rules §7 保障 2） | `results/E209/scene_snapshot/`（22 目录 + manifest.txt） |

---

## 八、下一步

1. **P8 人审（唯一待办）**：`bash workspace/core4d/scripts/eval/wrappers/review_player.sh E209ARM`，22 例 A/B 对看。
   **必读 §4-2 的机位陷阱**：PRG 与 G1 的 mp4 不是同一机位，判「谁偏离参考更多」只能在单个视频内部比 sim 与 ref，跨视频比绝对姿态/物体在画面中的位置与大小都无效。
   重点看：末端与躯干姿态相对参考的偏离（数值指向 eef_ori +2.78°、desk007_034_p1 +33.7°）。**不要**去找「手脱离物体」——接触率已证明没脱离。降级须填 taxonomy。
2. **不建议把 G1 用于 desk/chair 出片**：主门退化 3 例、L1 从 3 涨到 11，代价由 robot tracking 支付。
3. **建议的后续实验**（如果要救）：gravcomp 是连续量，`gravcomp=1` 是全额补偿。既然传递函数是收缩型且不动点在 +3.5 cm，**部分补偿（如 gravcomp≈0.5）理论上能把 bias 停在 0 附近而少付 eef_ori 的代价**——这是一个单参数扫描，且 E209 的拟合直接给出了预测起点。
4. **回填 E207**：E207 log296 的「+1.945 cm 常量偏移」应加一条指向本 log 的修正说明（机制是收缩型；该常量只是它那 9 例窄区间上的局部近似）。**不修改 log296 正文**（历史不可篡改），在 TRACKER 与本 log 交叉引用。
5. **给 14-gate 提一个缺口**：目前没有任何一门直接度量「物体被托着但手没真正夹紧」。`eef_ori` 是偶然的代理。若后续还要做失重类干预，建议加一条 hand-object 相对姿态门。
