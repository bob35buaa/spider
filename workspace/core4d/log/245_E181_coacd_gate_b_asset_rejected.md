# E181 结果日志：CoACD canonical geometry 在 Gate B 被拒绝

_Core4D Phase 44 · 2026-07-31 · plan
[199](../plan/199_E181_coacd_canonical_geometry_plan.md)_

## 0. 一句话结论

E181 完成了 E178 同源 authority、三个 bucket 的 cleaned-mesh oracle 和
`3×18=54` 个 CoACD candidate 构建，但冻结的 dev-only Gate B 中三个物体均为
`0/18 PASS`。当前 `K≤32、vertices≤64` 的 CoACD 搜索空间不能可靠保留 bucket
空腔，因此状态为 **`ASSET_REJECTED`**；没有生成 `C*`，heldout24 保持封存，
`D_C`、physics canary 以及与 E178 同 27 case 的三卡 Full CEM 均未启动。

## 1. 实验目标与冻结合同

本实验验证以下几何同源路线是否可以替换 E178 的 handcraft box proxy：

```text
cleaned original mesh M*
  ├─ D_M：只作原始几何 fidelity oracle
  └─ CoACD candidates C_i
       ├─ P：compound-convex physics
       └─ D_C：R/G 查询同一个 canonical collision set
```

最终 Full authority 已预先冻结为 E178 canonical manifest 的同序 27 条
case，CEM budget 保持 `seed=0, 1024×32`。若 Gate B 后续通过，计算分配才是：

| Worker | Device | Case 数 |
|---|---|---:|
| local | 本地 GPU 0 | 9 |
| ada0 | `spider-remote` RTX 6000 Ada GPU 0 | 9 |
| ada1 | `spider-remote` RTX 6000 Ada GPU 1 | 9 |

分片按 manifest 物理行序生成的连续 `authority_row_index=1..27` 轮转，
而不是使用有跳号的 E178 `ordinal`。Gate B 未通过时，该 9/9/9 合同只作为
下游执行条件，不构成启动授权。

## 2. S0 authority 与环境预检

Gate 0 为 **PASS**：

- E178 source manifest SHA：
  `de9a3d165301318049da208aad52b1f5bc4d3ed671631f0097958734a0f022a8`；
- full/dev/heldout=`27/3/24`，dev 与 heldout 交集为空；
- object 分布保持 bucket003/004/007=`9/4/14`；
- trajectory、contact、E174/E178 scene 与 snapshot 共 `162` 项 SHA
  检查通过；
- CoACD=`1.0.11`、trimesh=`4.11.5`，`real_metric=True` API 已实测；
- MuJoCo 3.7 与 MJWarp 的 mesh/SDF 最小编译 probe 通过；
- 本地 1 GPU、远程 2 GPU 只读环境检查通过，launcher 合同禁止
  kill、pause 或抢占已有远程任务。

## 3. S1 cleaned oracle 与 `D_M`

Gate A 为 **PASS**，三个 object 的数值与人工视觉审核均通过：

| Object | Raw topology | 清理 | `D_M`/sign | Visual |
|---|---|---|---|---|
| bucket003 | 5000 faces，1 component | 不删除组件 | PASS | APPROVED |
| bucket004 | 主体4990 faces + 10-face碎片 | 删除1个数值碎片 | PASS | APPROVED |
| bucket007 | 主体4994 faces + 4/2-face碎片 | 删除2个数值碎片 | PASS | APPROVED |

三个 `M*` 均 watertight、winding-consistent；主组件顶点未移动，Open3D
multi-ray sign 与 exact solid-angle winding 在 1mm band 外一致。
`D_M<0` 冻结为 inside。

实际视觉观察：

- bucket003 raw/cleaned 三个投影视图逐点一致，中心截面连续闭合；
- bucket004 删除的 10-face 数值碎片位于外表面附近，bucket 主体和 rim
  没有可见缺口或变形；
- bucket007 删除的 4/2-face 碎片未影响主体和 rim，三个主轴截面连续；
- Gate B cavity heatmap 中所有 candidate 均显著高于 0.1% 红线；
  task heatmap 显示增加 hull/vertex 常改善 must-cover 或 contact，但 cavity
  往往同步变差，不存在被绘图遗漏的全门通过区域。

## 4. S2 CoACD 构建

固定搜索空间：

| 参数 | 值 |
|---|---|
| `threshold_m` | 5 / 10 / 20 mm |
| `max_convex_hull` | 8 / 16 / 32 |
| `max_ch_vertex` | 32 / 64 |
| seed / scale | `1` / real meters |
| post-process | `merge=True, decimate=True` |
| child threading | OMP/TBB/OpenBLAS 均为 1 |

构建结果：

- `54/54 BUILD_PASS`，每个 object `18/18`；
- 实际 hull 数为 `8–32`，单 part 最大顶点数为 `64`；
- OBJ round-trip 最大位移为 `6.94e-18m`；
- 单 child 最大 peak RSS 约 `888MiB`；
- `44/54` 出现 hull-cap concavity warning，warning 被保留为证据，但没有在
  build 阶段提前代替 fidelity gate；
- `bucket003/t020_k08_v032` 两次隔离单线程构建的 ordered-part SHA 与
  candidate asset SHA 完全一致：
  `7be5da1ae82a5a5daa7524bef8670ce4f516ff99c88348906896be6a76be075f`。

## 5. Gate B：正式 dev-only fidelity

Compound convex parts 的切割面彼此零接触。直接拼接 part triangles 或使用
manifold boolean 都会把内部 partition faces 计入外表面。正式 evaluator
因此冻结 `2mm accessibility probe`：只有沿法线外移 2mm 为 free、内移
2mm 为 occupied 的采样点才计为可达 `C` boundary。该参数已经进入 evaluator
config SHA，未按 candidate 单独调节。

正式结果：

| Object | PASS / 18 | 最低 broader cavity false occupied | 硬门 |
|---|---:|---:|---:|
| bucket003 | 0 | 19.070% | ≤0.1% |
| bucket004 | 0 | 5.950% | ≤0.1% |
| bucket007 | 0 | 4.625% | ≤0.1% |

最低值仍分别是硬门的约 `191× / 60× / 46×`，不是阈值边缘抖动。除此之外：

- bucket003 多数候选同时失败双向 surface、must-cover、core 和 normal；
- bucket004 的较细候选可通过部分 surface/contact 门，但 broader cavity
  仍无法通过；
- bucket007 同时普遍失败 C→M、core、normal，并有较多 must-cover failure。

Canonical evidence：

- candidate metrics SHA：
  `c8f78169c7ac11315b6a9756eecf0f2d5978beef6a32a5288b7993dee1e75794`；
- evaluator config SHA：
  `86d610fdec40709b80e1234755ae54c9d48e06a2f7fac6b1f8603ac2bad57a6a`；
- no-selection record SHA：
  `bfae422beb5f4f5f9aabef8b26b1f8d4c22414a08e10708281ca439f46dafe62`。

## 6. Heldout 与 Full 执行状态

正式 Gate B evaluator 只读取 dev3。三个 fixture NPZ 仅包含：

- `M*` surface；
- broader/core cavity samples；
- dev target points；
- dev `D_M`。

结果明确记录：

```text
selected={}
heldout_status=SEALED_NO_C_STAR
downstream_status=S3_S6_NOT_AUTHORIZED_BY_GATE_B
```

因此：

- 没有生成或冒充 `C*`；
- 没有烘焙 canonical `D_C`；
- 没有修改 P/R/G production backend；
- 没有启动 S4 physics、S5 canary 或 S6 Full；
- 本地 5090 与远程 Ada0/Ada1 均未执行 E181 Full，远程已有任务未被触碰；
- E178 历史 scene、结果和日志未修改。

## 7. Claims 验证

| Claim | 结果 |
|---|---|
| C0 authority | ✅ PASS — E178 27-row authority 与 162 项输入/snapshot SHA 通过 |
| C1 oracle | ✅ PASS — 三个 `M*`、`D_M` 数值门与 3/3 visual review 通过 |
| C2 CoACD reproducibility | ✅ PASS — 54/54 build；隔离重跑 asset SHA exact |
| C3 asset fidelity | ❌ FAIL — 三个 object 均 0/18，无 `C*` |
| C4 canonical SDF | ⏭️ NOT AUTHORIZED — Gate B 后停止 |
| C5 legacy compatibility | ⏭️ NOT AUTHORIZED — 未接 production backend |
| C6 physics contract | ⏭️ NOT AUTHORIZED |
| C7 oracle contact | ⏭️ NOT AUTHORIZED |
| C8 throughput | ⏭️ NOT AUTHORIZED |
| C9 G health | ⏭️ NOT AUTHORIZED |
| C10 downstream | ⏭️ NOT AUTHORIZED — Full 0/27 |

## 8. 执行中发现的问题与处置

| 事件 | 处置 |
|---|---|
| authority 首轮错误地把 E178 base SHA 套到当前可变 scene | 改为分别验证 E174 base、E178 effective scene 与两份 snapshot，共162项 |
| environment probe 顶层导入 Torch 触发 `RC=139` | 移除 S0 不需要的 Torch import，以 package metadata 与 `nvidia-smi` 记录环境 |
| bucket003 严格5mm单候选约25min | 中止本地 E181 launcher后以 isolated child 做 bounded resume；不影响已完成 candidate |
| raw part surface 包含内部切割面 | 冻结统一的 2mm accessibility probe，并把参数纳入 config SHA |
| 首轮 evaluator 曾读取 heldout target | 废弃该轮 evidence，彻底移除 full27/heldout 读取并以新 config SHA 重跑54项 |

heldout 重跑前后 cavity 数值和 dev verdict 一致；正式记录只引用重跑后的
dev-only artifact。

## 9. 决策与下一步

本次负结果否定的是当前冻结搜索空间：

```text
CoACD threshold=5/10/20mm
K≤32
max_ch_vertex≤64
```

它不否定“`C → P` 与同源 `D_C → R/G`”的总体架构。下一步若继续，必须新开
实验号，并改变至少一个实质变量，例如更高 hull budget、任务感知 decomposition、
允许其他 convex decomposition 方法，或重新论证 cavity 采样/阈值。不得在
E181 内临时放宽 Gate B、人工编辑 hull 或解封 heldout24。

只有新实验先生成通过 Gate B 的 `C*`，才能继续 `D_C`、physics/canary；
正式 Full 仍必须复用 E178 同一 27 case，并按本地+Ada0+Ada1 的固定
`9/9/9` 三卡分配执行。

Claims 未全部通过，本实验不按“成功实验”规则 commit/push。

## 10. 结果路径

```text
workspace/core4d/results/E181/
├── s0_environment/
│   ├── authority_manifest.json
│   └── dependency_manifest.json
├── s1_oracle/
│   ├── gate_a_summary.json
│   ├── visual_review.json
│   └── bucket{003,004,007}/
├── s2_coacd/
│   ├── build_summary.json
│   ├── determinism_probe_t020_k08_v032.json
│   └── bucket{003,004,007}/
├── s2_asset_eval/
│   ├── candidate_metrics.tsv
│   ├── pareto_summary.json
│   ├── selected_canonical_sets.json
│   ├── cavity_false_positive_heatmap.png
│   └── must_cover_contact_excess_heatmap.png
└── s6_downstream/manifests/
    ├── full27.tsv
    ├── dev3.tsv
    └── heldout24.tsv
```

主要可复现入口：

```text
workspace/core4d/scripts/launch/active/run_E181_local.sh
workspace/core4d/scripts/experiments/E181/
```
