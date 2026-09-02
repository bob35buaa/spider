# log295 · E206：desk+chair 走 dcv3 全流程 + 非凸碰撞代理 + noPRG/PRG 双 arm

_Core4D · Phase 66 · Run **R291**(noPRG) + **R292**(PRG) · 承接 [plan236](../plan/236_E206_desk_chair_move2_dcv3_noprg_prg_plan.md) · 2026-09-02～09-03 · 分支 `feat/E206-desk-chair-move2-dcv3-arms` · **状态：进行中（P0/P1/P2/P4 完成，P2.3 人工复审待做，P3/P5–P10 未开始）**_

> 本 log 是**中途快照**，不是结论。写它的目的是把已确认的事实、已修的坑、和**还没解决的问题**固定下来，避免后续重复踩。最终结论待实验跑完后补。

## 一句话进展

desk+chair 的 dcv3 上游（S0/S1/S2）已打通：**74 case / 9 物体**落地，17/17 源模板装上 ≤16-box 全 box 碰撞代理并通过加载校验；碰撞代理做到了**自动体素 + 人工 3D 编辑 + 语义部件分解**三条路并存。过程中挖出 **3 个 dcv3/上游真 bug** 和 **4 个我自己引入的性能/口径问题**。CEM 尚未开跑。

---

## 一、已完成阶段

| 阶段 | 状态 | commit |
|---|---|---|
| P0 覆盖前快照 + 消费者预检 | ✅ | `b76d8fb` |
| P1 S0+S1 案例集推导 | ✅ | `b33b27f` |
| P2 lowgeom 契约（几何） | ✅ | `0b52e77` |
| P4 17 模板装代理 + 补建缺失 | ✅ | `a51f892` `2bea582` |
| P2.3 3D 复审器 + box 编辑 | ✅ 工具就绪 | `b7a4177` `939b399` |
| P2.3 **人工 approve_clean 复审** | ⏳ **未完成（0/9）** | — |
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

### 当前三条代理路线并存

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

---

## 五、⚠️ 尚未解决 / 待办

### U1 · 契约与实装模板不一致（**必须在 P5 之前修**）

当前磁盘状态：

- **源模板装的是体素版**（desk007 12 / desk020 11 / desk021 13 / desk023 9 / chair005 15 / chair006 16 / chair020 15 / chair021 13 / chair022 8）
- **语义版已构建但未安装**
- 契约 TSV 是 02:51 那版，**早于 G7 修正**，仍显示 4/9

→ 需要跑一次 `audit --frozen-target-cells` + `install --apply` 让三者对齐，并逐个 `union_geoms_are_boxes` 复校。

### U2 · P2.3 人工复审 0/9

`nonbox_template_review.tsv` 里 9 个物体全部 `review_decision` 为空。按 `docs/data_construction_v3/04_scene_template_policy.md:63-77`，**MuJoCo 能加载 + 渲染通过不足以放行**，必须显式 `approve_clean` 才能进 Stage2b。**这是 P5 的硬前置。**

其中 **chair005 / chair022 需单独判 G6 豁免**（过填均 0.41 —— 座下空间被填实，机器人腿无法从椅下摆过）。

### U3 · `closest_point_naive` 未替换（性能债）

trimesh 的 `closest_point_naive` 是 **O(点数 × 三角形数)** 暴力版。实测 chair022 / 4000 点：

| | 耗时 | 差异 |
|---|---|---|
| `closest_point_naive`（现用，继承自 E176） | 5.90s | — |
| `closest_point`（R-tree） | 0.50s | max 3.9e-7 |

**12 倍加速、结果一致**。没改是因为它继承自 E176 的度量代码，换掉会让 E206 与 E176 的保真数字失去逐位可复现性（虽然数值等价）。**建议改，但要在 log 里显式声明并对一个物体做 before/after 对照。**

### U4 · audit 重复计算

`main()` 里 `evaluate()` 已经构建过一次代理，缓存循环又 `build_effective_proxy` 了一遍——语义物体的 80k 采样 + 保真度量算了两次。应改为复用 `evaluate` 的结果。

### U5 · chair021 的 "CEM 排最后" 尚未落到代码

用户要求 chair021（几何差）在 CEM 队列里排最后，目前只在 plan/log 里写着，**P3 的 A2b 优先级序和 P8 的 driver 都还没实现这个约束**。

### U6 · 未开始

P3 吞吐实测冻结 N_MAX（A1 ≤120min/task、A2 ≤48h、A2b 优先级序、A3 回落阶梯）；P5 S3 重定向；P6 S4 门 + G8；P7 双 arm 场景/override；P8 CEM；P9 评测。

---

## 六、改动文件

**新建** `workspace/core4d/scripts/experiments/E206/`：
`e206_common.py`（scope/arm/路径单一真源）、`report_s1_funnel.py`、`lowgeom_proxy_v2.py`、`audit_lowgeom_contract.py`、`install_lowgeom_templates.py`、`semantic_proxy.py`、`review_proxy_3d.py`、`refreeze_after_edit.sh`

**修改（dcv3 管线代码，隔离可逆，已 git 跟踪）**：
| 路径 | 改动 |
|---|---|
| `data_construction_v3/stages/s2_templates/build_or_audit_templates.py` | 修材质替换静默 no-op（F7）；抽出 `substitute_object_material()` 并加替换次数断言 |

**SPIDER core（`spider/`）零改动。** `union` 的 box-only 约束被当作必须绕开设计的硬约束，全程未放松。

**计划简化**：原计划要给 `build_or_audit_templates.py` 加 `--nonbox-collision-policy` 开关；实测其 `--apply-build` 本就同时建 box 与 nonbox 代理模板，故改为「先用默认草稿策略建模板，再由 `install_lowgeom_templates` 统一替换碰撞块」，**dcv3 管线只剩材质 bug 这一处必要改动**。

**数据产物**：17 个源模板 + 6 个新物化 mesh 资产已 `git add -f`；覆盖前状态存于 `results/E206_pre/scene_snapshot/`（59 目录 + sha256 manifest，rule 7 双保障）。

## 七、结果路径

- S1 漏斗：`results/E206/s1_raw_contact/e206_s1_funnel.{tsv,md,json}` + `e206_s1_case_terminal.tsv`
- 案例权威：`results/E206/s1_raw_contact/raw_contact/raw_contact_pass_3cm_move2only.tsv`（74 行）
- 代理契约：`results/E206/s2_proxy/lowgeom_contract.{md,json}` + `lowgeom_contract_n{16,9}.tsv` + `lowgeom_sweep_n*.json`
- 人工编辑：`results/E206/s2_proxy/box_edits.json`
- 模板安装：`results/E206/s2_templates/lowgeom_install.tsv` + `lowgeom_assets.tsv`
- 复审判定：`results/E206/s2_templates/review/nonbox_template_review.tsv`（**待填**）
- 覆盖前快照：`results/E206_pre/scene_snapshot/manifest.txt`
- 预检：`results/E206/preflight/task_dir_consumers.txt` + `affected_overrides.txt`

## 八、下一步

1. **修 U1**：audit（快路径）+ install 对齐，让契约、语义代理、实装模板三者一致
2. **U2 人工 3D 复审**：`review_proxy_3d.py --reviewer <name>`，9 个物体逐个 `approve_clean`，chair005/chair022 判 G6 豁免
3. P3 吞吐实测冻结 N_MAX + 队列优先级（含 U5 的 chair021 排最后）
4. P5 起走 S3→S6
