# E146 — 手部碰撞体几何可视化对比（球 vs forearm-3box vs 橡胶手 mesh）

日期：2026-06-08
分支：exp/core4d-collab-retarget
git HEAD（运行时）：b230fa9

## Purpose（假设）

下游 RL 侧已观察到：手部碰撞体用 **橡胶手 mesh** 而非 handbox/sphere 能提升 RL 成功率。但 SPIDER 上游重定向当前手部碰撞体是 **5cm 球**（`hand_collision` class `<geom type="sphere" pos="0.1 0 0" size="0.05">`，scene `lh`/`rh`），rubber_hand mesh 仅作 `class="visual"` 渲染。

假设：换更贴合真实手形的碰撞体（橡胶手 mesh）能减少手-箱穿透、提升接触几何保真度。本实验**纯可视化验证**这个几何直觉，不改算法、不重训、不重跑 CEM。

## 方法

- **轨迹**：回放 E143 已有的 SPIDER 轨迹（`trajectory_mjwp_act.npz`），运动学 `mj_forward`（无物理）。手的位置三面板完全一致，只换叠加的碰撞体几何 → 纯几何对比。
- **三种碰撞体**（同帧叠加渲染）：
  - sphere：现状 5cm 球（lh/rh）。
  - 3box：HDMI 风格 forearm 三盒（抄自 box025_person1 `scene_forearm.xml:284-286,330-332`）。
  - rubber：橡胶手 mesh（`left/right_rubber_hand.STL`，22876 顶点）作渲染壳。
- **SDF 度量**：解析 point-box（复用 `unified_replay_eval.py:signed_point_box`）。球=球心到 box SDF−半径；3box=各盒 8 角点 min；rubber=mesh 顶点子采样(400) min。object box pose/size 取自 `object_collision` geom。
- **case**：好坏各半 4 个。

## Run command

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python \
  workspace/core4d/scripts/E146/render_hand_collision_compare.py --all --stride 3
```

## 改动文件

| 文件 | 说明 |
|---|---|
| `workspace/core4d/scripts/E146/render_hand_collision_compare.py` | 新建：scene 增广 + 回放渲染 + SDF 度量 |
| `workspace/core4d/results/E146/scene_snapshot/<case>/scene_act_handviz.xml` + `manifest.txt` | 增广 scene 快照（git HEAD + 源 sha256），满足 experiment.md §7 |
| `workspace/core4d/results/E146/hand_collision_viz/<case>/{compare.mp4,keyframes.png,sdf_per_geom.tsv}` | 产物 |

源 scene（`workspace/core4d/results/E143/scene_snapshot/*/scene_act.xml`）**未改动**（git status 确认）。

## Result（全 case，stride=3，逐帧 SDF 统计）

**必须同时看「接触(near-band)」和「穿透」两个方向**——穿透少可能只是碰撞体离箱更远导致接触也差。near=SDF≤阈值的帧占比(越高越贴)，penet=SDF<0(浅穿)，deep=SDF<−2cm。

| case | quality | 碰撞体 | 接触3cm | 接触5cm | 接触10cm | 穿透any | 深穿2cm |
|---|---|---|---:|---:|---:|---:|---:|
| box021_035_p2 | good | sphere | 0.76 | 0.78 | 0.80 | **0.64** | 0.00 |
| | | 3box | 0.76 | 0.78 | 0.80 | 0.38 | 0.00 |
| | | **rubber** | **0.78** | 0.78 | 0.80 | **0.04** | 0.00 |
| box023_person2 | good | sphere | 0.46 | 0.46 | 0.50 | 0.24 | 0.00 |
| | | 3box | 0.43 | 0.46 | 0.50 | 0.15 | 0.04 |
| | | rubber | 0.46 | **0.48** | **0.52** | **0.30** | 0.00 |
| box026_133_p1 | bad(rot155°) | sphere | 0.28 | 0.38 | 0.41 | 0.12 | 0.00 |
| | | 3box | 0.31 | 0.38 | 0.44 | 0.19 | 0.09 |
| | | rubber | **0.34** | 0.38 | 0.44 | 0.12 | 0.06 |
| box026_134_p2 | bad(lift0.03m) | sphere | 0.36 | 0.39 | 0.42 | 0.11 | 0.00 |
| | | 3box | 0.39 | 0.42 | 0.42 | 0.17 | 0.06 |
| | | rubber | **0.42** | **0.42** | 0.42 | 0.17 | 0.03 |

读法：
- **box021（干净大箱）= rubber 理想场景**：接触 3cm 0.78≥球 0.76 **且** 穿透 64%→4%。不是靠离远减穿透，是真贴合。
- **box023（小箱）**：rubber 接触最高(5cm 0.48)**但穿透也最高(0.30)**——小箱+真实手形 mesh 体积大，更贴也更易蹭进。
- **box026 坏 case**：rubber/3box 接触 3cm 高于球，但出现球没有的**深穿透 2cm**(3~9%)；mesh/盒体积大，坏轨迹上深插。
- **球的"0 深穿透"是假象**：球单点+偏前(`pos=0.1`)根本没真贴，所以插不深；它 64% 浅穿才是真问题。

## 视觉观察（关键帧，必看）

- **box021_035_p2（双手抬箱，最干净）**：sphere 两红球插进箱左右侧面；3box 绿盒穿透更深（有一截进箱内）；**rubber mesh 贴箱两侧外表面、几乎不穿透**（SDF +0.8cm）。与数据一致：rubber 穿透 4.4% vs sphere 64%。
- **box023_person2**：sphere 球偏离手真实位置（飘在箱顶/侧），rubber 最贴手形。
- **box026_133_p1（大旋转坏 case）**：sphere 球飘在箱上方（没接触），3box 斜插进箱，rubber 贴箱角——坏 case 三者都能看出手-箱关系紊乱。

## Conclusion

1. **rubber 接触不输甚至略优于球（不是"穿透少但接触差"）**：全 4 case 的接触 3cm 指标 rubber ≥ sphere（0.78/0.46/0.34/0.42 vs 球 0.76/0.46/0.28/0.36）。所以减穿透不是靠离远——接触方向同时持平/改善。
2. **干净大箱（box021）是 rubber 最佳场景**：接触 3cm 略升（0.78 vs 0.76）+ 浅穿透 64%→4%。理想的"既贴又不穿"。
3. **rubber 不是无脑更好**：小箱（box023）rubber 接触最高但**浅穿透也最高（0.30 vs 球 0.24）**；坏轨迹（box026）rubber/3box 引入球没有的**深穿透 2cm（3~9%）**。原因：球是单点+偏置，mesh 是真实体积——更易贴也更易插。
4. **球的"0 深穿透"是假象优势**：球 `pos=0.1` 偏前 + 单点，根本没真正贴合所以"插不深"；它的 64% 浅穿透才是真问题（呼应 [[project_contact_metric_pitfall]]）。
5. **3box 不推荐**：接触≈球，但全 case 都有深穿透（盒角易插），是为侧抓 forearm 设计的，不适合替代手接触代理。
6. **此结果仅几何层（运动学回放，手位不变）**。静态下 mesh 在坏轨迹/小箱深插，但这正是 **换 rubber 后重跑 CEM** 要解决的——CEM 的 `deep_penalty` 会主动把手推开，届时才能看 rubber 的真实净收益。静态结论：rubber 几何潜力最好，但需物理重优化兑现，且小箱要警惕过穿。

## Next steps

- 若推进：在 SPIDER scene 把 `hand_collision` 从 sphere 换成 rubber mesh 碰撞体（注意 MJWP/Warp 凸性，`geom_rbound=0` 需凸包/凸分解才能进物理接触），重跑 box021_035_p2 一条 CEM 验证。
- 注意接触对 condim/friction/solref（scene `:401-402`）为球调，换 mesh 需重调。

## 范围约束（已遵守）
- 未改 SPIDER 算法 / reward / 源 scene XML。
- 未重跑 CEM（纯运动学回放）。
- 未做 mesh 凸分解 / 未依赖 MuJoCo mesh 接触（用解析 SDF）。

## 注记
- 首次误把产物写进已占用的 E145 目录（真实 E145 = nonbox template release），已迁移到 E146；E145 里的误写拷贝由用户手动 `rm` 清理。
