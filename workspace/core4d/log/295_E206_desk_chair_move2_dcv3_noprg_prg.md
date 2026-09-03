# log295 · E206：desk+chair 走 dcv3 全流程 + 非凸碰撞代理 + noPRG/PRG 双 arm

_Core4D · Phase 66 · Run **R291**(noPRG) + **R292**(PRG) · 承接 [plan236](../plan/236_E206_desk_chair_move2_dcv3_noprg_prg_plan.md) · 2026-09-02～09-03 · 分支 `feat/E206-desk-chair-move2-dcv3-arms` · **状态：S0–S2 全部闭合（P0/P1/P2/P2.3b/P4 完成，复审已签）；P3 及 P5–P10 未开始**_

> 本 log 是**中途快照**，不是结论。写它的目的是把已确认的事实、已修的坑、和**还没解决的问题**固定下来，避免后续重复踩。最终结论待实验跑完后补。

## 一句话进展

desk+chair 的 dcv3 上游（S0/S1/S2）**已全部闭合**：**65 case / 8 物体**（desk005 在 S1 落地 0 case、chair021 因代理质量被人工弃用），8 个物体的碰撞代理全部由**人工在 3D 界面里逐个重摆 box** 完成，契约 **8/8 hard gates pass（含新增 G10 支撑面共面）、`all_pass=true`、G6 豁免需求归零**，15/15 源模板已装上并通过 MuJoCo 加载 / 签名稳定 / 与代理逐位比对，复审 **15 行全签、65/65 case 模板 `clean_reviewed`**。

手编相对自动代理是实质改进：chair022 腔体过填 **0.41→0.12**、chair006 **0.26→0.06**、chair005 **0.41→0.23**、desk021 的 mesh→proxy p90 **0.042→0.005**。

过程中挖出 **5 个真 bug**（F4/F6/F7 在 E176/dcv3 上游；F10/F13 在 E206 自身，其中 F13 会让 34/65 个 case 在 Stage2b 被静默 hold）和 **8 个我自己引入的性能/口径问题**（第四节）。CEM 尚未开跑。

---

## 一、已完成阶段

| 阶段 | 状态 | commit |
|---|---|---|
| P0 覆盖前快照 + 消费者预检 | ✅ | `b76d8fb` |
| P1 S0+S1 案例集推导 | ✅ | `b33b27f` |
| P2 lowgeom 契约（几何） | ✅ | `0b52e77` |
| P4 17 模板装代理 + 补建缺失 | ✅ | `a51f892` `2bea582` |
| P2.3 3D 复审器 + box 删除 | ✅ 工具就绪 | `b7a4177` `939b399` |
| P2.3b 自由编辑碰撞体界面（F9/F10） | ✅ | — |
| P2.3b **人工重摆 8 物体 box + G10 对齐**（F11） | ✅ 契约 8/8 all_pass | — |
| P2.3b chair021 退出（F12） | ✅ 65 case / 8 物体 | — |
| P4b 15/15 模板装手编代理 | ✅ | — |
| P2.3 **人工 approve_clean 复审**（F13） | ✅ 15 行/8 物体，65/65 case | — |
| P3 吞吐实测冻结 N_MAX | ⏳ 未开始 | — |
| P5–P10 | ⏳ 未开始 | — |

---

## 二、S1 漏斗（P1 结论）

三层过滤，逐层实测：

```
desk/chair inventory                     652 行
  ↓ action ∈ move2_{obs0,obs1,obs3} 且过 AABB 尺寸门
                                         164 行
  ↓ raw_contact 3cm
                                          74 行 / 9 物体
```

逐物体：desk007 9 / desk020 2 / desk021 17 / desk023 14 / chair005 2 / chair006 13 / chair020 1 / chair021 9 / chair022 7。

**`desk005` 落地 0 case，退出 E206**（6 个候选全被运动门/接触质量拒）。
**`chair021` 后来被人工弃用**（F12，代理质量）→ 最终规模 **65 case / 8 物体，双 arm 130 条 CEM**。

### F1 · 口径先定 5cm，实测后改回 3cm

计划期担心 3cm 样本不够，锁的是「5cm 主口径 + 3cm 桥接层」。**实测证伪**：

| 口径 | 落地 case | 覆盖物体 |
|---|---:|---:|
| 3cm | **74** | 9 |
| 5cm | 76 | 9 |

放宽阈值**只救回 2 个**（desk007 +1、desk020 +1），却要付出与 box/bucket 全线 + E174 口径断裂、外加维护独立桥接层（独立 S3、独立 task dir、C5a 特殊处理）的代价。**用户据此改回 3cm 单一口径，桥接层取消。** 收益：口径断裂风险消失、C5a 可直接与 E174 desk007 同尺对比、S3 少跑一轮、CEM 少 18 条。

### F2 · 真正的瓶颈是旋转门，不是接触阈值 ⚠️ 未解决，记为后续方向

| 拒因 | 数量 | 占 164 |
|---|---:|---:|
| **`object_rotation >= 45°` 运动硬门** | **56** | **34%** |
| `object_lift <= 0.30m`（不含同时旋转超限） | 8 | 5% |
| 接触质量不足（weak_two_hand_overlap / unbalanced） | 25 | 15% |

双人搬桌/椅本来就常要转向（绕门、调头），而 45° 门是为 box 搬运设的。**E206 不动此门**（保持与 bucket 线 E174/E178/E202/E204/E205 同一 S1 口径，结论可横向比）。放宽到 90°/180° 预计能把样本从 74 抬到 ~130，但重定向难度未知——**建议另立实验验证**。

> 我在提问时的猜测（"瓶颈是 lift 门，桌椅是推不是抬"）**被证伪**——lift 只卡掉 8 个。

### F3 · 样本量诚实声明

`chair020` **n=1**、`chair005`/`desk020` **n=2**。这三个物体**不出 per-object arm 推荐**，只并入总体统计，log 中逐物体必须标 n。

---

## 三、碰撞代理（P2/P4 核心）

硬约束：`spider/config.py:69-79` 的 `object_collision_sdf_mode='union'` **fail-closed 只接受 box geom** → 代理必须是轴对齐 box，不能用 mesh/CoACD。

现网 desk/chair 是 26-cell 体素草稿：chair021 **127 个 box**（→2286 对碰撞对，落在 E175 已被否决的 52–66h 区间）、desk005 74、desk021 73、desk020 69、chair005 64、desk007 41、desk023 37。

### F4 · E176 的 `center_inside_count` 断言会卡死全部 5 把椅子（已解决）

`E176/lowgeom_proxy.py:105-113` 在任何 box 包住 mesh AABB 中心时抛错。这是为**空心 bucket/desk** 写的启发式；椅面天然占据 AABB 中心。**实测 5/5 椅子全部触发**。

**没有简单删掉**（删了就失去 E176 C4/C5 建立的腔体保证），而是换成**直接度量** `cavity_metrics()`：在代理并集内按体积采 6000 点，统计到真实 mesh 距离 >5cm 的占比。对桶/桌/椅一视同仁，不依赖"中心该是空的"这种类别假设。

### F5 · 体素分辨率对代理质量**非单调**（已解决）

原选择规则是"取最细可行 target_cells"。**实测 chair022**：

| target_cells | boxes | mesh→proxy p90 | 过填>5cm | 过保真门 |
|---:|---:|---:|---:|---|
| 3 | 8 | 0.084 | 0.73 | ✗ |
| **4** | **8** | **0.064** | **0.43** | **✓** |
| 6（最细可行） | 14 | 0.100 | 0.56 | ✗ |

硬 box 上限下，更细的体素要用同样预算覆盖更多体素，贪心合并只能吐出**更少更大、跨腔的块**。按原规则会选中 tc=6 这个更差的代理（契约 8/9）。

改成**实测驱动**：`score_target_cells()` 把每个可行 tc 的保真度全部实测，在通过全部保真门的候选中取**最小腔体过填**，以 p90、box 数依次 tie-break → **契约 9/9**。

### F6 · 6 个物体的 mesh 资产从未物化（已解决，dcv3 上游缺陷）

`desk020 / desk023 / chair005 / chair006 / chair020 / chair021` 在 `example_datasets/processed/core4d/assets/objects/` 下**没有 `<key>_m.obj`**，其源模板连 MuJoCo 都加载不了 —— **覆盖 41/74 个 case（55%）**。

根因：`build_or_audit_templates.py` 只在**创建**模板时 `shutil.copy2(raw_mesh, asset_mesh)`，这几个物体的模板是更早期建的，资产没跟上。用 `ensure_asset_mesh()` 同源同方式补齐；已验证 asset 与 `object_models/` 的 mesh 顶点/面/extents/centroid 完全一致（desk021 仅文件格式差异，几何相同）。

### F7 · dcv3 材质替换 bug —— 静默 no-op（已修，真 bug）

新建的非 box 模板 MuJoCo 加载即失败：`material 'chair006_material' not found`。

根因：`build_or_audit_templates.py` 的替换正则找 `<material name="box_material" ...>`，而 base scene（box023_person1）实际写的是 **`box023_material`**。`re.sub` 匹配不到就**静默原样返回**，于是 `<asset>` 里留着 `box023_material`，而生成的 `object_visual` geom 引用 `<object_key>_material`。两个调用点（box/nonbox）都有；因为只有**新建**非 box 模板才会走到，一直没被触发。

修法：抽出 `substitute_object_material()`，正则同时容纳 `box_material|box023_material`，且**替换次数必须为 1，否则抛错**——不再允许静默 no-op。

### F8 · 朴素按 XZ 聚类分腿是错的（已解决）

用户 3D 复审后要求按语义部件切（4 腿各 1 + 座面 + 椅背 + 扶手）。第一版实现每条"腿"box **68–82% 是空的**：

```
chair020  leg0 dims=0.191x0.367x0.182  empty>5cm=71%
chair022  leg2 dims=0.221x0.364x0.265  empty>5cm=82%
```

根因：**座面正下方的裙板/横档把四条腿连成一片**。实测 XZ 占用率——纯腿区 2–3%（4 个独立柱子），裙板高度 10–18%（1 个连通块）。KMeans 跨过这个边界，每个象限的 AABB 就把整个座下空间收进去了。

修法：`leg_zone_top()` 按 XZ 占用率突变检测裙板边界；裙板**并入座面 box**（它本就贴着座面，不占腿摆动空间），腿只取纯柱区并向上延伸到座面。

| | 修前 p2m / 过填 | 修后 |
|---|---|---|
| chair020 | 0.149 / 0.56 | **0.044 / 0.19** |
| chair022 | 0.193 / 0.60 | **0.097 / 0.41** |
| chair006 | 0.151 / 0.38 | **0.060 / 0.26** |

### 曾经的三条代理路线（历史，已被手编统一取代）

| 物体 | 路线 | box | 说明 |
|---|---|---:|---|
| desk007 | 体素 + 人工删 4 | 12 | |
| desk023 | 体素 + 人工删 5 | 9 | |
| chair021 | 体素 + 人工删 1 | 12 | 几何差，**CEM 排最后** |
| desk021 | 语义 | 5 | 桌板 + 4 腿 |
| desk020 | 语义 | 7 | 桌板 + 6 个人工保留底座块（底座是整块连通实体，非四腿） |
| chair005 | 语义 | 2 | 椅背 + 其余整体，**G6 豁免（过填 0.41）** |
| chair020 | 语义 | 6 | 4 腿 + 座面 + 椅背 |
| chair006 | 语义 | 8 | 4 腿 + 座面 + 椅背 + 2 扶手 |
| chair022 | 语义 | 8 | 同上，**G6 豁免（过填 0.41）** |

体素 vs 语义实测对比（语义列为用户指定方案）：

| 物体 | 体素 box/p90/p2m/过填 | 语义 box/p90/p2m/过填 | 判读 |
|---|---|---|---|
| desk021 | 13 / .010 / .024 / .00 | **5** / .042 / **.007** / .00 | box −62%，p2m 好 3 倍 |
| desk020 | 11 / .017 / .045 / .00 | **7** / **.011** / .045 / .03 | box 减，过填 +.03 可忽略 |
| chair006 | 15 / .044 / .089 / .24 | **8** / .073 / **.060** / .26 | box 砍半，p2m 更好 |
| chair022 | 8 / .064 / .127 / .43 | 8 / **.051** / **.097** / **.41** | 三项全好 |
| chair020 | 11 / .044 / .054 / **.06** | 6 / .032 / .044 / **.19** | ⚠ 过填 3 倍（用户选语义） |
| chair005 | 15 / .020 / .048 / **.04** | 2 / .063 / .097 / **.41** | ⚠ 过填 10 倍（用户选语义 + 豁免） |

### F9 · 自由编辑碰撞体的 3D 界面（2026-09-03，用户要求）

删 box 的自由度不够：chair005 / chair022 过填 0.41，**不存在任何 box 子集是对的** —— 座下空间被填实这件事，靠删救不回来，必须重新摆。故新建 `edit_proxy_3d.py`（`review_proxy_3d.py` 的超集，原文件保留可回退）。

**存储从「索引」改成「几何」。** 旧的 `box_edits.json` 记的是原始 build 序索引，只对它当时看的那次自动构建有意义；代理一变索引就指向别的 box。新的 `manual_boxes.json` 存 `center/half_size/label` 绝对几何，与种子代理彻底解耦 —— 这类 bug 在设计上不再可能发生。`build_effective_proxy` 的优先级变为 **manual > semantic > voxel+edits**，自动路线降级为「种子」。

编辑能力：中心 gizmo 平移 + box 局部 min/max 两个角点 gizmo 拉伸（viser 无 scale gizmo，两角定 AABB）+ 6 个数值框精调；新建 / 复制 / 删除 / 30 步撤销；`⇲ 贴合`（把 box 收缩到其内部真实 mesh 的 AABB，先粗摆再一键收紧）、`⇔ 镜像 X/Z`（椅腿、扶手对称）。

两个诊断显示做在界面里，把「碰撞体不好」变成可见而非推断：
- **未覆盖红点云** —— mesh 采样点中在 box 并集**外部** > 阈值的染红。用的是**有符号**并集 SDF，不是 E176 的 `point_to_proxy_surface_distance`（后者取绝对值，会把深埋在大 box 内部的 mesh 点也判成"远"，整个内部会误染红）。
- **逐 box 腔体过填** —— box 列表每行直接标该 box 内部离 mesh >5cm 的比例。实测 chair020：5 个 box 是 0%，靠背那一个是 37% —— 一眼定位到该改哪个，union 级的单一数字做不到这件事。

仍为**轴对齐**。核实过 `mjwp.py:282` 与 `eval_E175_proxy_fidelity.py:132` 都用 per-geom `geom_xmat`，旋转 box 在物理与评测侧其实都安全，fail-closed 只卡 geom type=box —— 但本轮用户决定不引入 `quat`，记为后续可选项（斜靠背/斜腿的过填有真实下降空间）。

指标实时刷新 **0.37s/次**（chair020，降采样 + R-tree），另设「🎯 全精度重算」按 audit 口径核对。

### F10 · 又两个物体栽在同一类 stale-index 上（已修，真 bug）

`desk020` / `chair006` 在本次 audit 中 `build_ok=False`：

```
ValueError: desk020: box edit is stale (n_boxes_original 16 != 7)
ValueError: chair006: box edit is stale (n_boxes_original 16 != 8)
```

与 U1 里记的 chair020 同一类：voxel 时代的索引记录被拿去套语义代理。根因是 `build_effective_proxy` 的语义分支**无条件**重放 `box_edits.json`：
- `chair006` 的记录（16-box voxel 的 `removed=[15]`）对 8-box 语义代理纯属过期；
- `desk020` 更糟 —— 它的记录已经被 `_edited_voxel_boxes_below` **正当消费过一次**（`keep_edited_below` 就是靠它重建底座块），语义分支再重放一遍等于**同一条编辑应用了两次**。

修法：语义分支只重放 `proxy_kind == "semantic"` 的记录。缺 `proxy_kind`（voxel 时代）或标 `voxel` 的一律不重放，desk020 的记录仍由 `_edited_voxel_boxes_below` 正常使用。修后 **9/9 物体 build_ok**。

`chair020` 的损坏记录（`n_boxes_original=6` vs `removed=[12,13,14]`）直接删除，备份在 `box_edits.json.bak-20260903`；死文件 `box_edits.json.inflight` 一并删除（`refreeze_after_edit.sh:33` 写了但从无人读，注释描述的竞态从未真正实现）。

### F11 · 支撑面共面检查 G10（用户 2026-09-03 提出，已实装）

用户在手编完后指出：桌椅的 3/4 条腿必须**同时踩到地**，不能一高一低。这是对的，而且**现有的门一个都看不见它** —— G3/G4/G5 都是聚合距离，一条腿差 12mm 对 p90 的影响是零。

两种失效：
- **一高一低** → 物体只站在最低那条腿上，其余悬空。这个倾斜完全是代理捏造的，真实物体没有。
- **穿地**（box 底面低于 mesh 底面）→ 整个物体浮在空中。

新增 **G10 · 支撑面共面**（`lowgeom_proxy_v2.support_contact_metrics`）：
- `floor_y = mesh.bounds[0][1]`（物体局部 +Y 向上）
- 「承重 box」= 底面在 floor 上方 2cm 以内的（座面、扶手自然不算）
- 门：所有承重 box 的 `|bottom − floor_y| ≤ 5mm`
- 同时报 `support_bottom_spread_m`（高低差，就是用户问的那个数）、`floating_leg_labels`（标签叫 leg 却离地 >2cm 的）

**手编后的实测（对齐前）—— 8/8 全部不合格**：

| 物体 | 承重 box | 高低差 | 离地区间 | 说明 |
|---|---:|---:|---|---|
| chair020 | 4 | **11.7 mm** | +0.2 … +11.9 | leg3 比其余低 11.7mm，椅子只站一条腿 |
| chair022 | 4 | **16.0 mm** | +0.4 … +16.4 | m08/m09 悬空 13–16mm |
| chair006 | 4 | 5.5 mm | 0.0 … +5.6 | |
| desk021 | 4 | 5.0 mm | +0.6 … +5.6 | |
| desk023 | 4 | 0.4 mm | **−5.5 … −5.2** | 四条腿整体穿地 5mm（平但陷下去） |
| desk020 | 3 | 8.0 mm | **−29.7 … −21.7** | 三块底座穿地 2–3cm，桌子浮空 |
| chair005 | 1 | 0.0 mm | +8.8 | 整张椅子悬空 8.8mm |
| desk007 | 4 | 0.0 mm | −10.7 | 自动体素版，平但整体穿地 1cm |

`align_supports.py`：把承重 box 的**底面**落到地面，**顶面固定不动** —— 腿是连在座面上的，短腿要往下长，不能整体下移脱离座面。逐 box 打印位移量，**绝不静默改几何**。执行后 **8/8 高低差与离地全部 0.0mm，G10 全过**。

> chair022 的 `leg0/leg1` 离地 185mm 被标为「悬空腿」告警 —— 查证后是**正常的**：用户把两条斜后腿拆成上下两段 box 堆叠（leg0 叠在 m09 上、leg1 叠在 m08 上），上段本来就不该着地。告警保留为提示，不是门。

### F12 · chair021 退出 E206（用户 2026-09-03 决定）

chair021 的几何不支持一个值得跑的 ≤16-box 代理 —— 它是唯一一个到最后仍停留在「体素草稿 + 人工删 1 个」而没有重摆过 box 的物体，本来就已经被排到 CEM 队尾。用户直接弃用。

**规模变化：74 case / 9 物体 → 65 case / 8 物体。** 逐物体：desk007 9 / desk020 2 / desk021 17 / desk023 14 / chair005 2 / chair006 13 / chair020 1 / chair022 7。
双 arm CEM 条数 **148 → 130**。

`e206_common.DROPPED_OBJECT_KEYS = ("chair021",)`，保留在 `OBJECT_KEYS` 内使排除可审计（与 `SIZE_GATE_EXCLUDED_KEYS` 同一处理）。`landed_object_keys()` 与 `is_in_scope()` 同步过滤。
**遗留**：chair021 的两个源模板仍带着 E206 装的 12-box 代理（已出范围，不会被 S3/CEM 触及）；覆盖前状态在 `results/E206_pre/scene_snapshot/chair021_person{1,2}/` 可恢复。

### 手编后的最终代理（P2.3b 收敛结果）

全部 8 个物体现在都走 `proxy_kind=manual`，**契约 8/8 hard gates pass，`all_pass=true`，G6 豁免需求归零**：

| 物体 | box | mesh→proxy p90 | proxy→mesh p90 | 过填>5cm | 手编前(自动)过填 |
|---|---:|---:|---:|---:|---:|
| desk007 | 12 | 0.034 | 0.052 | 0.04 | 0.04 (未手编，仅 G10 对齐) |
| desk020 | 7 | 0.010 | 0.026 | 0.04 | 0.03 |
| desk021 | 5 | **0.005** | **0.007** | 0.00 | 0.00 |
| desk023 | 9 | 0.011 | 0.018 | 0.00 | 0.00 |
| chair005 | 2 | 0.040 | 0.048 | **0.23** | **0.41** ← 不再需要豁免 |
| chair006 | 10 | 0.042 | 0.048 | **0.06** | **0.26** |
| chair020 | 7 | 0.021 | 0.042 | 0.25 | 0.19 |
| chair022 | 10 | 0.041 | 0.047 | **0.12** | **0.41** ← 不再需要豁免 |

**人工重摆 box 的收益是实的**：chair022 过填 0.41→0.12、chair006 0.26→0.06、chair005 0.41→0.23，desk021 的 p90 从 0.042 降到 0.005。R3 里「N=9 椅子腔体过填 42–45%」这条风险，以及 G6 豁免与随之而来的「结论边界声明」，**全部消失**。

### 3D 交互复审器

`review_proxy_3d.py`（viser）：真实 mesh 半透明 + 逐 box 上色叠加、透明度滑条、线框、地面网格；**点 3D 里的 box 或勾选框直接删/恢复**；判定写 `nonbox_template_review.tsv`（E145/E174 同 schema）。

box 删除记录进 `s2_proxy/box_edits.json`，带 `proxy_kind`/`target_cells`/原始 box 数**指纹**——代理重建后旧索引会被判 stale 并拒绝应用，不会悄悄删错 box。编辑后**全部保真指标按编辑后的 box 集重算**。

---

## 四、我自己引入的问题（全部已修，记录以免重犯）

| # | 问题 | 后果 | 修法 |
|---|---|---|---|
| S1 | `footprint_profile` 用 Python `set` 对 8 万点建 40 次索引集 | 单物体语义构建**几分钟**，3D 复审器起不来 | 改 numpy 布尔数组 → **0.1–0.3s** |
| S2 | `_edited_voxel_boxes_below` 从**契约**读 `target_cells` | desk020 转语义后契约写 `-1` → `pitch = extents/(-1)` 负值 → 体素化**挂死** | 改从**编辑记录**读（那里存着索引真正对应的那次构建的 tc=9） |
| S3 | audit 每次都重扫全部 target_cells 候选 | 一轮 **35 分钟**，编辑→重算循环没法用 | 加 `--frozen-target-cells` 快路径 → **2m46s** |
| S4 | G7（过填>pitch）被套用到语义代理 | 5 个语义物体全判失败（契约假报 4/9） | G7 是**体素合并的健全性检查**（"内部不该有点离 mesh 超过一个体素"），语义 box 没有体素概念，属**类别错误**；对语义标 `n/a` |
| S5 | 契约在 audit 跑到一半时被用户编辑覆盖 | 契约描述的代理**不是实装的那个**（desk007 契约 16 box / 实装 12 box） | refreeze 脚本开跑时快照 edits 文件；根因是 S3 的慢 |
| S6 | `trimesh.voxelized` 对某些 mesh 分钟级 | 复审器每次启动都重算 | `_boxes_at` 加磁盘 memo |
| S7 | `edit_proxy_3d` 的 `flush_reviews` 按 `keys` 全量重写复审 TSV | 用 `--object chair020` 起编辑器，会把 9 行 TSV **截断成 1 行**（冒烟测试当场触发） | 保留不在本次编辑范围内的行（`foreign_rows`，按 task 键） |
| S8 | 编辑器种子用 `voxel(tc=N)` = `_boxes_at()`，那是**编辑前**的 box | desk007 打开就从 12 box 退回 16 box，静默丢弃之前记录的 4 次删除 | 默认种子改成 `effective(当前生效)`，走 `build_effective_proxy` —— 起点必须就是实际会 ship 的那套 |

---

## 五、逐项状态（已关闭 / 遗留）

### ~~U1 · 契约与实装模板不一致~~ ✅ **已关闭**

契约、手编代理、实装模板三者现已一致：

- `audit --frozen-target-cells` rc=0，**8/8 hard gates pass, `all_pass=true`**（含新的 G10）
- `install --apply` rc=0：**15/15 scene applied**，全部 `signature_stable=true` / `all_box=true` / `nq,nv,nu = 43,41,29`
- `effective_boxes.json` 已补写
- 快照：`results/E206/scene_snapshot/`（43 文件 + git HEAD + sha256 manifest），15 个 task 的 `scene.xml` / `task_info.json` 已 `git add -f`

> **注**：`workspace/core4d/results` 是指向外部存储的**符号链接**，`git ls-files` 下 0 个 results 文件 —— 本仓库从来没有、也无法把快照提交进 git（rule 7 safeguard 2 在此仓库布局下只能落到磁盘）。safeguard 1（活跃 case 的 scene XML 强制入库）已满足。
>
> `chair020_person2` 模板不存在（M5 遗留）。chair020 唯一落地的 case 是 `chair020_20231011_066_p1` = **person1**，故无影响。

### F13 · 复审 TSV 只写 person1，34/65 case 会被 Stage2b 静默 hold（已修，真 bug）

`approve_clean` 不是走过场 —— `run_stage2b.py:168` 对 `template_status not in {clean, clean_reviewed}` 直接返回 `stage2b_template_*` 并跳过该 case，而 `clean_reviewed` 只能由 `run_stage2b.py:77` 读到 `review_decision == "approve_clean"` 产生。

问题在于这个闸是**按 task（`<obj>_person{1,2}`）**判的，而 `review_proxy_3d.source_task()`（:68）只返回 person1：

```python
for person in ("person1", "person2"):
    if (PROCESSED_ROOT / f"{object_key}_{person}" / "scene.xml").exists():
        return f"{object_key}_{person}"      # <- 命中 person1 就 return
```

于是复审 TSV 从来只有 8 行 person1。后果：
- **34/65 case（全部 person2）会在 Stage2b 被静默 hold** —— 不报错，只是 taxonomy 里多一堆 `stage2b_template_*`
- 反过来 **`desk020_person1` 拿到了 approve，但它一个 case 都没有**（desk020 的 2 个 case 全是 person2）

修法：`edit_proxy_3d.person_tasks()` 枚举磁盘上真实存在的每个 person 模板，逐 task 出一行。两个 person 共用同一物体 mesh 与同一套实装 box，所以一次判定天然覆盖两行 —— 只是必须写到两行上。TSV 从 8 行变 **15 行**，65/65 case 的模板全部 `approve_clean`。

### ~~U2 · P2.3 人工 approve_clean~~ ✅ **已签（15 行 / 8 物体，reviewer=xiayibo）**

用户在 3D 界面里逐物体重摆 box（15:41–16:20）并为每个物体写了 notes —— **那就是复审本身**；原先残留的 7 行 `needs_manual_edit` 是 01:2x 提修改要求那轮的产物，早已过期。用户 2026-09-03 明确授权按其判断签署，notes 保留其原话并追加「3D 手编 box 后 + G10 支撑面对齐后确认」。

> **对齐改动已如实告知后再签**：G10 对齐动的是用户最后一次查看之后的几何 —— desk020 的 base2/4/5 各缩短约 3cm（原穿地 2–3cm）、chair022 的 m08/m09 各下伸 13–16mm、chair020 四腿各下伸 9–12mm，其余 5 个 ≤6mm。

### U3 · `closest_point_naive` → R-tree（**已按"双路径"解决**）

trimesh 的 `closest_point_naive` 是 **O(点数 × 三角形数)** 暴力版。本轮实测 chair022 / 4000 点（同一批采样点，逐点对照）：

| | 耗时 | max\|Δ\| | 过填>5cm |
|---|---|---|---|
| `closest_point_naive`（E176 继承） | 5.76 s | — | 0.418 |
| `closest_point`（R-tree） | 0.39 s | **6.7e-7** | 0.418 |

**14.8× 加速，数值等价。** 处理方式不是直接替换，而是**双路径**：`L.mesh_surface_distance(..., fast=)` + `L.fidelity_metrics_fast()`，**默认仍走 naive**。理由：契约数字必须与 log235 的 E176 数字逐位可比，而编辑器每次拖拽都要重算 —— 6 s/帧不可用，4e-7 无关紧要。写入合同的永远是精确路径，编辑器界面用快路径，界面上也明写了这一点。

### U4 · audit 重复计算

`main()` 里 `evaluate()` 已经构建过一次代理，缓存循环又 `build_effective_proxy` 了一遍——语义物体的 80k 采样 + 保真度量算了两次。应改为复用 `evaluate` 的结果。

### ~~U5 · chair021 的 "CEM 排最后"~~ ✅ **已消解** —— chair021 直接退出 E206（F12），无需排序约束。

### U6 · P3 及之后未开始 —— 见第八节的阶段表

### U7 · `eval_E176_contact_fidelity.py:172-176` 的硬编码 `>9` 未参数化

计划期就记为 R10，但当时假设代理是 9 box 量级。**现在实装的 desk007(12) / chair006(10) / chair022(10) 都 >9 → P6 的 G8 一跑就 `AssertionError`。** 必须在 P6 之前改成 `--max-object-geoms`（默认 9，向后兼容）。

---

## 六、改动文件

**新建** `workspace/core4d/scripts/experiments/E206/`：
`e206_common.py`（scope/arm/路径单一真源）、`report_s1_funnel.py`、`lowgeom_proxy_v2.py`、`audit_lowgeom_contract.py`、`install_lowgeom_templates.py`、`semantic_proxy.py`、`review_proxy_3d.py`、`refreeze_after_edit.sh`、**`manual_boxes.py`**（手编 box 的绝对几何存储）、**`edit_proxy_3d.py`**（自由编辑界面，F9）

新增 **`align_supports.py`**（G10 支撑面批量对齐 CLI，默认 dry run）。

**E206 内部修改（2026-09-03，F9–F12 两轮）**：
| 路径 | 改动 |
|---|---|
| `semantic_proxy.py` | 加 manual 分支（优先级最高）；语义分支只重放 `proxy_kind=="semantic"` 的编辑记录（F10）；`build_effective_proxy(..., measure_metrics=False)` 供编辑器取种子 |
| `lowgeom_proxy_v2.py` | `mesh_surface_distance(fast=)` + `fidelity_metrics_fast()` 双路径（U3）；`build_lowgeom_boxes(measure_metrics=)`；**`support_contact_metrics()` + `align_support_boxes()`（G10，F11）** |
| `audit_lowgeom_contract.py` | G7/`target_cells` 对 `manual` 与 `semantic` 一律记 `n/a`；`--frozen-target-cells` 跳过 `n/a` 行不再崩；`effective_boxes.json` 收录 manual 与实际 tc；**加 G10 门 + 6 个 support_* 列**；`landed_object_keys()` 过滤 `DROPPED_OBJECT_KEYS` |
| `install_lowgeom_templates.py` | `target_cells="n/a"` 不再 `ValueError` |
| `e206_common.py` | **`DROPPED_OBJECT_KEYS=("chair021",)` + `is_in_scope()` 同步过滤（F12）** |

**修改（dcv3 管线代码，隔离可逆，已 git 跟踪）**：
| 路径 | 改动 |
|---|---|
| `data_construction_v3/stages/s2_templates/build_or_audit_templates.py` | 修材质替换静默 no-op（F7）；抽出 `substitute_object_material()` 并加替换次数断言 |

**SPIDER core（`spider/`）零改动。** `union` 的 box-only 约束被当作必须绕开设计的硬约束，全程未放松。

**计划简化**：原计划要给 `build_or_audit_templates.py` 加 `--nonbox-collision-policy` 开关；实测其 `--apply-build` 本就同时建 box 与 nonbox 代理模板，故改为「先用默认草稿策略建模板，再由 `install_lowgeom_templates` 统一替换碰撞块」，**dcv3 管线只剩材质 bug 这一处必要改动**。

**数据产物**：17 个源模板 + 6 个新物化 mesh 资产已 `git add -f`（其中 15 个是本轮在范围的 task，另 2 个是已出范围的 `chair021_person{1,2}`）；训练态快照 `results/E206/scene_snapshot/`（15 task / 43 文件），覆盖前状态 `results/E206_pre/scene_snapshot/`（59 目录）。两者都带 git HEAD + sha256 manifest。

## 七、结果路径

- S1 漏斗：`results/E206/s1_raw_contact/e206_s1_funnel.{tsv,md,json}` + `e206_s1_case_terminal.tsv`
- 案例权威：`results/E206/s1_raw_contact/raw_contact/raw_contact_pass_3cm_move2only.tsv`（74 行；**下游按 `DROPPED_OBJECT_KEYS` 过滤掉 chair021 的 9 行 → 实际 65**）
- 代理契约：`results/E206/s2_proxy/lowgeom_contract.{md,json}` + `lowgeom_contract_n{16,9}.tsv` + `lowgeom_sweep_n*.json`
- **手编碰撞体（权威）**：`results/E206/s2_proxy/manual_boxes.json`
- 生效 box 缓存：`results/E206/s2_proxy/effective_boxes.json`
- 人工删 box（遗留索引式）：`results/E206/s2_proxy/box_edits.json`（+ `.bak-20260903`）
- 模板安装：`results/E206/s2_templates/lowgeom_install.tsv`（15 applied）+ `lowgeom_assets.tsv`
- 复审判定：`results/E206/s2_templates/review/nonbox_template_review.tsv`（**15 行，全部 `approve_clean` / `clean_reviewed`**）
- 训练态快照：`results/E206/scene_snapshot/manifest.txt`（15 task / 43 文件 + git HEAD + sha256）
- 覆盖前快照：`results/E206_pre/scene_snapshot/manifest.txt`
- 预检：`results/E206/preflight/task_dir_consumers.txt` + `affected_overrides.txt`

## 八、下一步

**S2 已全部闭合**（契约 8/8 all_pass、15/15 模板实装、15 行复审全签、65/65 case 模板 `clean_reviewed`）。剩下的按 plan236 的 P3→P10 走，两处口径因 P2.3b 而放宽：

| 阶段 | 内容 | 相对 plan236 的变化 |
|---|---|---|
| **P3** | 吞吐实测 + 冻结 `N_MAX` / `use_torch_compile` / 队列优先级序 → `admission_decision.json` | **压力显著下降**：队列 148→**130 条**，实装最大 box 数 **12**（不是 16，pair 288→216）。A2 的中位墙钟上限 155→**177 min/task**。A2b 优先级序仍保留（desk007 9 例优先，C5a 命脉），但 U5 的「chair021 排最后」已随 F12 消失 |
| **P5** | S3 重定向（omnirt_v1 主 + v2 rescue），`KEEP_GOING=1` | 65 case。顺带做 19 个重叠 case 与 E145 的**可复现性交叉校验** |
| **P6** | S4 目标门 + 视觉 QC + **G8 接触保真**（ref-FK 接触目标→代理表面 p90 ≤ 0.08 m） | G8 是唯一还没验的代理门；`eval_E176_contact_fidelity.py:172-176` 的硬编码 `>9` 仍需参数化（R10，实装最大 12 box 会**直接触发**） |
| **P7** | 双 arm 场景/override，单变量断言 | pair 数按实装 box 数逐 case 算（2N 手对 / 18N），不再是固定 16 |
| **P8** | CEM 130 条，8 卡 | 先 2 case × 2 arm smoke + diff `config_act.yaml` |
| **P9** | 配对 noPRG vs PRG + 强制视觉复核 | 逐物体表标 n；chair020(n=1)、chair005/desk020(n=2) 不出 per-object 推荐 |
| **P10** | 收尾：本 log 补最终结论、tracker 加 R291/R292、`build_log_index.py` | |

**立即要做的两件小事**（都在 P6 之前）：
1. `eval_E176_contact_fidelity.py` 的 `--max-object-geoms` 参数化（R10）—— 现在实装 desk007/chair006/chair022 都 >9 box，不改 G8 一跑就 `AssertionError`。
2. U4：audit 的 `evaluate()` 与缓存循环重复构建代理，语义/手编物体的采样+度量算了两遍。纯浪费，不影响正确性。

**若后续再动任何一个物体的 box**，固定流程：
```
edit_proxy_3d.py  →  align_supports.py --apply  →  refreeze_after_edit.sh
                  →  重签该物体的 person1/person2 两行复审  →  snapshot_scenes.sh + git add -f
```

**后续实验方向**（不在 E206 内）：
- **F2 的旋转门**：`object_rotation >= 45°` 卡掉 34% 的候选，是本实验最大的样本流失来源。放宽到 90°/180° 预计 74→~130 case，但重定向难度未知，需另立实验。
- **带 quat 的旋转 box**：已核实 `mjwp.py:282` 与 `eval_E175_proxy_fidelity.py:132` 都走 per-geom `geom_xmat`，物理与评测侧都支持；斜靠背/斜腿的腔体过填有真实下降空间。本轮用户决定不引入。
- **board / stick**：仍被 S1 的 AABB 尺寸门硬拒（board020 是 2.7cm 薄板被误判 `too_large`），需先修尺寸门。
