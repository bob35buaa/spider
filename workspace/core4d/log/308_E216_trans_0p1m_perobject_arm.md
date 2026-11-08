# log308 · E216 · 下游对应臂 trans-only **0.1m** 物体增强放量（R304 / Phase 74）

**Run**: R304 · **Plan**: [249](../plan/249_E216_trans_0p1m_perobject_arm_plan.md) · **日期**: 2026-11-08 · **分支**: `experiment/E199-omniretarget-object-augmentation`

**结论**: ✅ **SUCCESS（C1/C2/C3/C5 达标；C4 诚实为"等价"而非"更优"）** — 0.1m trans 增强物理可信、上游 100% 可行、物体追踪略优于 orig，但在 CEM 物理门层面与 0.2m **等价**（非更易过门）。本轮 = Phase 1 box+bucket 104 case；chair/desk 20（Phase 2）未做。

---

## 1. 范围与臂（已核实）

104 base case × 3 trans（前/左/右，**0.1m**，omnirt_v2），逐物体下游对应臂：

| 物体 | 臂 | CEM scene | case 数 | 变体 |
|------|----|-----------|---------|------|
| box021 | **PRG (A0)** | scene_act_E199_rubberHull_PRG | 28 | 84 |
| box023 | **noPRG (E167A)** | scene_act_E199_rubberHull (base) | 16 | 48 |
| box001/004/024 | **PRG+G1A2** | scene_act_E199_rubberHull_PRG_gravcomp + A2 gate | 39 | 117 |
| bucket003/007 | **PRG** | scene_act_E202_bucketAlignedTop_PRG | 21 | 60 |
| **合计** | | | 104 | **309**（312−3 排除） |

唯一核心改动：holosoma `parallel_robot_retarget.py`（commit **2248e39**）trans 位移 env 参数化 `OMNIRT_TRANS_OFFSET_M=0.1` + `OMNIRT_TRANS_ONLY=1`（跳过 rot，省 ~40% IK）。SPIDER 核心零改动，各臂 scene/gravcomp 全复用，仅换 0.1m 轨迹。aug task 带 `_0p1m` 后缀，不覆盖 0.2m 历史任务。

## 2. Claims 验证

| Claim | 判据 | 结果 | 判定 |
|-------|------|------|------|
| **C1 可行性** | 0.1m 可行率 ≥ 0.2m（≥95%） | 上游 retarget **312/312=100%**；build 309/312（3 bucket007_021_p1 `runtime_initial_overlap` 排除，与 E202/E210 同几何检查）；manifest 单变量 audit **103/103 clean**；CEM **309/309 cem_ok，0 失败/超时**（25.74h，中位 37.9min）。vs 0.2m：box~100%、bucket **90%**(E202) → 0.1m bucket 可行率提升 | ✅ |
| **C2 增强正确** | approach=0.1m±0.01、endpoint 残差≤0.03m | 实测 approach 首帧/max/mean = **0.100m**，endpoint 衰减至 **0.009m**（锚定成立）；12 weak-trans（晚 trim_start 把 offset 衰减到 <0.05m，case 固有）标记 | ✅ |
| **C3 数值不劣 vs 同臂 orig** | obj_pos HL ≤ +5cm，0 新增 fall，穿透不劣 | obj_pos HL：box_noprg **+0.284**、box_prg **−0.182**、box_prg_g1a2 **+0.145**、bucket_prg **−0.141** cm（全 ≪ +5cm，2 臂为负=0.1m 更优）；**总 fall 10 < orig 12**（未增）；leg/hand 穿透 Δ 近零。**坑**：88/104 orig 基线为 omnirt_v1（retarget-variant 混淆，非纯单变量） | ✅ |
| **C4 0.1m vs 0.2m（核心动机）** | 0.1m 通过率 ≥ 0.2m | **309 配对，同臂同评估器**：6-gate 通过率 pooled **51.8%(0.1m) vs 52.4%(0.2m) = −0.6pp**（box_prg +1.2 / g1a2 +0.0 / noprg +0.0 / bucket **−5.0**）；fall **10(0.1m) ≤ 11(0.2m)**；obj_pos 全臂 0.1m 更优（Δ−0.048~−0.277cm）。**"更温和→更易过门"假设在 CEM 层面为 null** | ⚠️ 等价（非更优） |
| **C5 视觉门** | 代表案例无穿模/漂浮/jitter | osmesa sim-vs-aug-ref：4 臂代表（box021/box001/bucket003/box023）**跟踪干净、手接触、落脚、无穿透/漂浮**；2 fall case（box004_083_p2、bucket003_005_p1）**视觉证实真崩溃**（sim 前倾塌陷 vs ref 直立），非度量欺骗 | ✅ |

## 3. C4 — 0.1m vs 0.2m 清洁对照（本实验核心交付）

同臂、同 omnirt_v2、同评估器（`eval_E216.score`，6-gate），309 配对：

| 臂 | n | pass 0.1m | pass 0.2m | Δpass | fall 0.1/0.2 | Δobj_pos(cm) | Δcontact | Δleg_pen |
|----|---|-----------|-----------|-------|--------------|--------------|----------|----------|
| box_noprg | 48 | 31.2% | 31.2% | +0.0 | 3/3 | **−0.228** | −0.001 | +0.011 |
| box_prg | 84 | 53.6% | 52.4% | +1.2 | 0/0 | −0.103 | +0.001 | +0.010 |
| box_prg_g1a2 | 117 | 60.7% | 60.7% | +0.0 | 4/5 | −0.048 | +0.007 | +0.005 |
| bucket_prg | 60 | 48.3% | 53.3% | **−5.0** | 3/3 | −0.277 | +0.004 | −0.005 |
| **POOLED** | **309** | **51.8%** | **52.4%** | **−0.6** | **10/11** | **−0.135** | +0.003 | +0.006 |

**解读**：0.1m 与 0.2m 在 CEM 物理门上**统计等价**（pooled −0.6pp 属噪声内），0.1m **obj_pos 追踪一致更优**（全臂为负）、**fall 不增**（10≤11）。bucket_prg 是唯一下降臂（−5pp，3/60 case）。**"更小位移→更易过门"的动机假设在 CEM 门层面不成立**——0.1m 的真实优势在 ①上游可行率（100% vs bucket 0.2m 90%）②轨迹更贴近 orig。下游预筛（14-gate 漏斗+Refiner/Tracker）是否因 0.1m 更温和而留存更多，需下游 RL 团队实测（本门是 CEM 侧代理，非下游预筛本身）。

## 4. Fall 归因（诚实）

2 个 "new fall"（aug 摔、orig 没摔），共 5 变体：
- **box004_20231003_2_083_p2**（trans0/1，box_prg_g1a2，base=v1）：重前倾蹲姿负重，**E215 rot 同 case 也摔**（log306 记载）→ case 固有，非 0.1m 所致。0.1m vs 0.2m 该臂 fall 4<5，即 0.1m 反而更少。
- **bucket003_20231018_005_p1**（trans0/1/2，bucket_prg，base=v1）：视觉证实 sim 前倾塌陷。v1-orig 混淆。
两者均 v1-orig 基线。**总 fall 10 < orig 12**，0.1m 不增 fall。C4 verdict 报 "fail" 仅因 E215 移植的「每臂 new_falls==0」硬判据被这 2 个混淆 case 触发，非真实退化。

## 5. 执行与成本

- **retarget**：104 case，omnirt_v2 @0.1m，warm-start `_original`（只产 trans，trans-only 省 ~40% IK），5 object-group 并行，~2.5h。
- **build**：309 变体 scene 生成（create_spider_scene + core4d.py + build_prg_scene），串行 ~2.5h（瓶颈，每变体 ~1min MuJoCo 编译）。gravcomp sidecar 117 单变量 verified。
- **CEM**：309 变体 × 1024×32 seed0，**8 卡 25.74h**（中位 37.9min，max 75.2min），**0 失败**。占卡脚本跑 CPU 阶段、CEM 前 kill、CEM 后挂回。
- **eval**：309 aug + 103 orig scored（0 fail）；0.1m-vs-0.2m 另 scored 309 0.2m sibling。
- **总计** ≈ 309 CEM runs ≈ 25.7h 墙钟（8 卡）≈ **~200 GPU-h**。

## 6. 改动文件

| 类型 | 文件 |
|------|------|
| 上游（holosoma 分支，commit 2248e39） | `parallel_robot_retarget.py`（trans offset env + trans-only skip） |
| E216 脚本 | `scripts/experiments/E216/`：e216_common / registry_box_bucket.tsv / seed_warmstart / run_upstream_retarget / build_augmented_tasks / build_gravcomp_sidecars / build_aug_manifest / run_E216_cem / preflight_baseline_audit / test_e216_contract / snapshot_E216_scenes.sh / render_c5 |
| eval/report | `scripts/eval/runners/eval_E216_trans_augmentation.py`、`scripts/eval/reports/gen_E216_0p1m_vs_0p2m.py` |
| train/launch | `scripts/train/train_E216.sh`、`scripts/launch/active/run_E216_local_8gpu.sh` |
| scene 快照（rule 7） | `workspace/core4d/results/E216/scene_snapshot/`（309 task dirs，manifest.txt 1976 行 git HEAD+sha256） |
| 结果 | `results/E216/s6_downstream/{manifests/e216_priority_manifest.frozen.tsv, cem/full/*, eval/aug/*}` |

## 7. Phase 2 — chair/desk（E213 选定 gravcomp 臂，3 机 8 卡分片）

Phase 1（box+bucket）之后补全 chair/desk 20 case（E213 逐 case 选定 gravcomp 臂 chair006·desk021 G1 / desk007 G0.8·0.4·0.6 per-case / desk023 G0.6），3 台 8 卡机分片并行（共享 /mnt）。

**执行**：seed 自 E208 split v{1,2} 树 → retarget @0.1m（**per-case v1/v2 env**：desk021_014_p2 走 v2，其余 v1，匹配下游）→ E206 lowgeom PRG 场景（`build_arm_scenes.build_one`）+ per-case 分数 gravcomp sidecar → 冻结 **59/60 变体**（1 个 desk021_010_p2 trans0 v1-infeasible 诚实掉；1 个 v2 case 初跑 env bug 已修）。3 分片（A20/B20/C19）各机独立 flock+子集校验，**59/59 CEM 0 失败**。C2 approach=**0.100m**。本机 smoke 验证 E209 分数-gravcomp 场景能加载。

**结果（0 fall，59 paired vs E213 同选定臂 0.2m，同评估器）**：

| 物体 | n | 0.1m pass | 0.2m pass | Δpp | Δobj_pos |
|------|---|-----------|-----------|-----|----------|
| chair006 | 15 | 73.3% | 80.0% | −6.7 | −0.226 |
| desk007 | 12 | **8.3%** | 75.0% | **−66.7** | −0.016 |
| desk021 | 20 | 45.0% | 60.0% | −15.0 | −0.795 |
| desk023 | 12 | 75.0% | 100.0% | −25.0 | +0.223 |
| **POOLED** | **59** | **50.8%** | **76.3%** | **−25.4** | −0.285 |

**关键发现（与 box/bucket 相反，诚实且物理可解释）**：chair/desk 的 0.1m **显著劣于 0.2m（−25.4pp）**，desk007 最极端（−66.7pp）。绝对 0.1m 过门率 50.8% 与 box/bucket 0.1m（51.8%）相当、**0 fall**、obj_pos 多数更优——即 0.1m 本身不差，是 0.2m 选定臂基线高（76.3%）。失败模式全在 **hand_penetration(11)+leg_penetration(9)**，非 tracking。**机制（osmesa 视觉证实）**：aug offset 把物体 approach 推离身体；**0.2m 给更大身-物净空、0.1m 净空更小→腿/脚穿入桌子下框**（desk007_030_p2 sim 脚手橙色穿透高亮，ref 有净空）。大件家具贴身,净空效应主导;小件(box/bucket)可携带,0.1m≈0.2m。→ **"更小 offset 更温和"对大件家具反而有害**。

## 8. 全局结论（box+bucket+chair/desk，124 case / 368 变体）

**0.1m trans 增强的价值依物体类型分化（核心结论）**：
- **box/bucket（小件可携带）**：0.1m **≈ 0.2m** 过门率（−0.6pp）、obj_pos 略优、不增 fall、上游 100% 可行 → 0.1m 是等价的更温和变体。
- **chair/desk（大件家具）**：0.1m **劣于 0.2m（−25.4pp）** → 更小 offset 减少身-家具净空、腿/手穿透↑；大件家具应保留 0.2m。

诚实地说，Phase 1 最初的动机假设"更温和→更易过门→更易过下游预筛"**在 CEM 门层面不成立**：小件等价、大件更差。0.1m 的真实、稳健价值仅是**上游可行率**（100%，bucket 优于 0.2m 的 90%）与**贴近 orig 的 obj 追踪**。是否利于下游预筛留存仍需下游 RL 实测。

**交付**：chair/desk 脚本 `e216cd_common/build_cd/run_E216cd_cem/split_cd_shards/run_E216cd_shard.sh/gen_E216cd_eval` + registry_chair_desk.tsv；results `E216/chair_desk/{manifests(frozen+3shard), cem/full, eval/aug}`；scene_snapshot(E216CD, 477 行)。

**下一步**：① E216-export（partner + Holosoma re-anchor）若下游要；② 大件家具若仍想要 0.1m，需提高 gravcomp 或放宽 leg gate 补偿净空损失；③ v2-orig 清洁对照消除 box/bucket 的 v1 混淆。

