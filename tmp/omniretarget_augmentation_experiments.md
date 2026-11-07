# OmniRetarget 物体数据增强(trans/rot)+ SPIDER 重定向 —— 实验汇总

> 范围：上游 OmniRetarget/holosoma `object_interaction` 对**物体 approach 段**做平移(trans)/偏航旋转(rot)扰动、指数衰减锚回原轨迹，再经 SPIDER 重定向 + full CEM 的系列实验。
> 数据来源：`workspace/core4d/EXPERIMENT_TRACKER.md`（Phase 62–73）+ memory。日期 2026-08-15 ~ 2026-09-12。

## 1. 增强机制（背景，非实验）
- 扰动对象：object approach-segment（translation + yaw），物体起动后按 `translation_tau=50 / rotation_tau=25` 帧指数衰减回原轨迹，操作终点锚定不变。
- 原生档位：`original` + `trans_0/1/2`（前/左/右 0.2m）+ `rot_0/1`（±45° yaw + 0.2m 侧移）。`object_interaction` 无 scale/terrain 档。
- 可行性前提：**omnirt_v2**（Phase-4 约束放松 + foot_z + contact_preservation 等）才能让移位后的物体留在可达域；omnirt_v1 下 aug 常 IK 不可行。
- 验证：approach 偏移 0.200m，终点残差 0.027m（衰减生效）。

## 2. 核心增强放量实验

| 实验 | 日期 | 类型 | 物体/Case 规模 | 结果要点 |
|------|------|------|----------------|----------|
| **E199**(pilot) | 08-15 | box · trans | 8 物体 · 31 变体(8 orig + 23 trans；rot 全不可达) | 打通上游→SPIDER→PRG→CEM；修 holosoma 2 处 bug；obj_pos +5.0%(<25% 阈)，0 跌倒，视觉无 artifact |
| **E199**(全量, R286) | 08-17 | box · trans | **83/87 case × 3 trans = 249 aug** | 平移 100% 可行(83/83 full 3/3)；249/249 评 0 err；obj_pos +1.6%，eef −27%/手穿透 −10% 更优；gate 51% |
| **E202** | 08-20 | bucket · trans | **27 case × 3 trans = 73/81 可行(90.1%)** | 复用 E178 碰撞体+CEM；73/73 CEM 0 err；obj_pos −0.6%；接触 0.70→0.60；C7 osmesa 视觉确认 |
| **E210**(R296) | 09-06 | bucket · trans + gravcomp | **5 case × 3 = 15 变体**(复用 E202 trans 资产) | PARTIAL：叠加 object gravcomp，严格单变量 C→D +0/−5；只 075_p1 3 条干净，不建议整批出片 |
| **E215**(R302) | 09-12 | bucket+box · **rot** | **38 case(7 bucket + 31 box) × rot0/1** | ⭐ 首个 rot 放量成功。上游 rot **72/72 可行(100%，远好于 45° yaw 可达域担忧)**；CEM 70/70 0 失败；逐变体 C4 通过 61/68=89.7%；obj_ori +35~61% 为旋转固有代价；C6 视觉门过 |
| **E212/E213**(R299) | 09-07 | selected-arm · **trans+rot** | **21 case 选定臂 object-aug** | 复用 E208 retarget/task，仅换选定 gravcomp sidecar + 重跑 CEM；source 100/100 cem_ok；obj_pos HL +0.213cm(≪+5 门)，fall 全 0 |

## 3. 复用增强资产的臂消融

| 实验 | 日期 | 内容 | 规模 |
|------|------|------|------|
| **E200**(R288) | 08-18 | 复用 E199 的 249 aug 重跑 CEM，noPRG vs PRG/G1/A2 同口径对比(E201 14-gate) | 249/249, 0 err；PRG 胜(L3 30.5%>noPRG 20.5%)，差异全在 leg 穿透 |
| **E204/E205**(R289/R290) | 08-24 | 27 bucket 三臂 14-gate 对比（omnirt_**v1** 轨迹，仅改 arm，非新增强） | 54/54 cem_ok；narrow PRG 9/27>G1A2 8/27>noPRG 7/27 |

## 4. 下游配对 + Holosoma reanchor RL 导出

| 实验 | 日期 | 内容 | 规模 |
|------|------|------|------|
| **E202-export** | 08-28 | USE13 bucket 平移增强 → partner 配对 + Holosoma reanchor RL 导出（无新 CEM） | **12 case × 3 trans = 36 变体**全 pair-complete；72 motion validation PASS；post-reanchor grasp 手-source 表面 4.5–8.3cm |
| **E213-export** | 09-08 | E213 xlsx 挑选选定臂 aug → dcv3 paired RL 导出 | **32 单元(20 case)**，RL_EXPORT_READY 32/32 |
| **E215 RL-export**(R302 §12) | 09-12 | rot 增强选 27 单元 → dcv3 partner-paired 导出 | **26 source + 26 partner**，26/26 paired READY；52 份 re-anchored motion |

## 4b. 重定向算法版本(arm) 定义

增强只存在于上游 retarget 产出的参考轨迹里，与下游 SPIDER CEM 的 arm **完全解耦**；同一批 aug 轨迹可挂不同 arm 跑 CEM。四档 arm（权威 = `e173/e198_common`）：

| Arm | scene_act(碰撞对) | object gravcomp(G1) | hand-gate(A2) | leg-object 碰撞/gate | reward base |
|-----|-------------------|---------------------|---------------|---------------------|-------------|
| **noPRG (E167A)** | base，无碰撞对 | 0 | 无 | 关(scale=0, gate off) | E167A_zOnlyBody |
| **PRG (=A0)** | rubber_hull + 16 lower-body/object 对 | 0 | 无 | 16 对 + leg gate on | E167A_zOnlyBody |
| **PRG+G1** | PRG + gravcomp sidecar | 1(或部分 0.4~0.8) | 无 | 16 对 + gate | E167A_zOnlyBody |
| **PRG+G1A2** | PRG + gravcomp + A2 gate | 1 | A2_GATE | 16 对 + gate | E167A_zOnlyBody |

> E198 是 G1×A2 的完整 2×2 析因（A0/G1/A2/G1A2）。E212 用的是 **部分 gravcomp**（per-case 选 G1/G0.8/G0.6/G0.4/PRG），是 G1 轴的细分档。

## 4c. 每个增强实验用的 arm（对照表）

| 实验 | 增强 | 下游 CEM arm | 说明 |
|------|------|-------------|------|
| **E199**(pilot + 249 全量) | box trans | **PRG (A0)** 单一 | `scene_act_E199_rubberHull_PRG`，reward=E167A，gravcomp=0、无 A2；orig 基线复用 E198 A0 arm |
| **E200** | 复用 E199 249 aug | **noPRG(E167A)**[R288] + **PRG+G1A2**[R287]，并与 E199 **PRG** 三臂横比 | noPRG 249/249 完成；PRG+G1A2 从 PRG scene 建 G1 gravcomp + 挂 A2_GATE |
| **E202** | bucket trans(73) | **PRG** 单一（E178 栈：E174 PRG + E170_PRG reward，A0） | 无 gravcomp/handgate |
| **E210**(R296) | bucket007 子集 trans(15) | **PRG** + **PRG+G1**（G1-only gravcomp，2×2 orig/aug × PRG/G1） | 仅 object gravcomp，不含 A2 |
| **E212/E213**(R299) | selected-arm trans+rot(21 case) | **per-case 选定 gravcomp 臂**：G1 / G0.8 / G0.6 / G0.4 / PRG | 每 case 取 E212 最优臂，非统一；100 条新 CEM + 5 条复用 E208 PRG |
| **E215**(R302) | rot(38 case) | **逐 case 混合（5 组臂）**：box021=PRG · box023=**noPRG** · box001/004/024=**PRG+G1A2** · bucket=**PRG+G1**(bucket_prg_gravcomp) | 蓝本 E208；box 臂统一走 E199.build_prg_scene + E200 分派 |

## 4d. 下游 RL 导出 case 数量

| 导出 | 日期 | 来源增强 | arm | source 单元 | partner | 总 motion / 其他 |
|------|------|----------|-----|-------------|---------|------------------|
| **E202-export** | 08-28 | E202 bucket trans | PRG(E178) | **36 变体**(12 case × 3 trans；075_p2 orig-only) | 36(2 建 + 10 复用 retarget) | 72 motion validation PASS |
| **E213-export** | 09-08 | E213 selected-arm trans+rot | per-case 选定 gravcomp 臂 | **32 单元(20 case)**，32/32 READY | 32(15 E213partner_aug + 17 E208source-aug) | 64 motion；对齐 32/32 |
| **E215 RL-export**(R302 §12) | 09-12 | E215 rot | 逐 case 混合 | **26 source**(27 选 1 排除 C4 fall) | 26 | 26/26 paired READY；下游 re-anchor 52 份 motion |

> 合计 RL 导出 source 单元：36 + 32 + 26 = **94**（分别对应 bucket-trans / selected-arm-trans+rot / rot 三批），各自带等量 partner 配对经 Holosoma re-anchor。

## 4e. 与下游 RL provenance 的交叉核对（R018-51~73）

下游 RL 团队的 `R018-51_onward_augmented_test_upstream_provenance.md` 给出了他们采用的增强数据逐 case 上游来源。**结论：完全对得上**，且反向印证了"增强轨迹 arm-independent、与下游 CEM arm 解耦"这条核心判断。

**① 每个上游实验的归属一致：**

| 下游标注的来源 | 下游说明 | 与我方上游记录 |
|----------------|----------|----------------|
| **E215** ×25 | box/bucket 的 **rot**，target=omnirt_v2 | ✅ = 我方 E215（rot, 38 case, omnirt_v2） |
| **E213** ×17 | chair/desk 的 **rot+trans**，selected arm，v1/v2 混合 | ✅ = 我方 E212/E213（selected-arm trans+rot） |
| **E200** ×16 | box001/024 trans=**PRG_G1A2**；box023 trans=**noPRG**；box004 heldout | ✅ = 我方 E200（在 E199 的 249 aug 上重跑 noPRG + PRG+G1A2 两臂） |
| **E199** ×4 | box021 trans=**E199_PRG_omnirt_v2**（即 PRG/A0） | ✅ = 我方 E199（box trans, PRG/A0） |
| **E202** ×4 | bucket trans=E202 PRG | ✅ = 我方 E202（bucket trans, PRG） |

> partner 侧 desk 用 **E208**/v1（source-aug）、chair/box 用 E199/E213/E215——与我方 "E208=source-aug、下游由 Holosoma re-anchor 保证两人一致" 一致。**E210 未被下游采用**（我方 E210 结论即 "不建议整批出片"）——也对得上。

**② 反向印证解耦：同一批 box trans，两种命名口径**

下游对 box trans 用了两套命名，本质是同一份数据从不同环节命名：
- **CEM 口径**（取 CEM 优化结果 npz）：按**下游 CEM arm** 分 → box021→**E199(PRG)** ×4、box001/004/023/024→**E200(noPRG / PRG+G1A2)** ×16。
- **Omni 口径**（取上游 retarget trajectory）：轨迹是 arm-independent 的，故**全部 20 条 box trans 都归 E199**（=轨迹生产者）。

这正是我方 4b 节 "增强烘焙在 retarget 轨迹里、与下游 arm 完全解耦" 的直接证据：按轨迹命名=E199，按 CEM 臂命名=E199(box021)/E200(其余)。

**③ 下游验证出的逐 case arm 分派（与 E215 的 "E199.build_prg_scene + E200 分派" 完全吻合）：**

| 物体 | 下游采用的 arm | 来源实验 |
|------|---------------|----------|
| box021 | **PRG (A0)** | E199(trans) / E215(rot) |
| box023 | **noPRG (E167A)** | E200(trans) / E215(rot) |
| box001 / box004 / box024 | **PRG+G1A2** | E200(trans) / E215(rot) |
| bucket003 / bucket007 | PRG(trans) · PRG+G1(rot) | E202(trans) / E215(rot) |
| chair / desk | per-case selected gravcomp 臂 | E213 |

**④ 数量口径差异（非矛盾）：** 我方导出口径是 "各批 source 单元"（E202-export 36 / E213-export 32 / E215 26）；下游是从候选池（**CEM 104 / Omni 116**）里**正式选 66**（29 rot + 37 trans，44 父 case，11 物体）做测试集，另有 R018-73 复用 R018-30 的 79 case（50 trans + 29 orig）。选择子集 ≠ 全量导出，分母不同属正常。下游未入选的父 case 与 2 条 SBTO 缺失(box004_082/box023_045 的 trans0)已在其文档第 5 节如实列出。

## 5. 配套基础设施 / 相邻实验（非 trans/rot 本体）
- **E201**(08-18)：三级数据筛选漏斗(14 门宽/窄双口径) + VLM 初审；自动决策 60.8%，漏斗上线。增强系列的统一评估口径。
- **E207 / E209 / E211**：object **gravcomp**(部分补偿)单变量扫描——与增强相关但属"重力补偿 sidecar"，非 trans/rot 扰动。E207 bucket G1-only、E209 desk/chair、E211 desk007 补偿扫描(0.4/0.6/0.8)。
- **E206**(R291/R292)：desk+chair dcv3 全流程 65 case/8 物体（含 partner OmniRetarget 导出），非增强档本体。

## 6. 速查结论

| 维度 | 范围 |
|------|------|
| **实验编号跨度** | E199 → E215（Phase 62–73），含 3 个 export 变体 |
| **trans 增强物体** | box（E199，83 case）+ bucket（E202，27 case）→ 子集 E210（5 case） |
| **rot 增强物体** | bucket+box（E215，38 case = 7 bucket + 31 box），首个成功放量 |
| **trans+rot 合并** | E212/E213（21 case 选定臂） |
| **单实验最大 case** | E199 box 83 case（×3 trans = 249 aug 变体） |
| **可行性关键** | trans ≈100% 可行(omnirt_v2)；rot 在 E199 pilot 全不可达 → E215 证实 **72/72 可行**（档位/蓝本改进后） |
| **两人一致性** | 不在重定向层保证，由下游 Holosoma partner re-anchor 强制 |
