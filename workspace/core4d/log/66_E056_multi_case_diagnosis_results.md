# E056: 多 case Hand-Face 诊断 — 结果

## 状态: ✅ 完成 (2026-05-12)

## 实验配置

| 项 | 值 |
|---|---|
| 实验类型 | 数据分析 (无物理仿真, 无 CEM) |
| Cases | 6 个 B+C case (来自 E054): box021 / box023 / bucket001 / bucket005_s2 / bucket007 / desk021 |
| 输入 | `example_datasets/processed/core4d/unitree_g1/humanoid_object/<case>/0/trajectory_kinematic.npz` |
| 算法 | E055 v3 detector 找 intent window + 计算 L/R 对 6 面 signed distance time series + 分类 grasp_type |
| 运行 | `bash workspace/core4d/scripts/run_E056_diagnosis.sh` (实际只跑 multi_case_face_diagnosis.py + 抽帧验证) |
| 时长 | < 30s (诊断) + < 5s (抽帧) |

## 输出文件

| 路径 | 内容 |
|---|---|
| `workspace/core4d/results/E056/case_grasp_type_summary.csv` | 6 行, 15 列分类结果 |
| `workspace/core4d/results/E056/E056_summary.md` | 人读用分类汇总表 + E057 推荐 |
| `workspace/core4d/results/E056/<case>_face_dist.png` | 6 张 face dist 时间序列图 (粗线 = main face) |
| `workspace/core4d/results/E056/video_verify/<case>_t<time>s.jpg` | 9 张视频核实关键帧 (3 case × 3 帧) |

## 算法关键点

### 1. main face 判定 = "中位数 |dist| 最小 + ≥60% 帧 |dist|≤7cm"

```python
def find_main_face(signed_in_window):
    abs_dist = |signed_in_window|             # (T, 6)
    med = median(abs_dist, axis=0)            # (6,)
    frac_close = (abs_dist ≤ 0.07).mean(axis=0)
    for face in argsort(med):
        if frac_close[face] ≥ 0.6 and med[face] ≤ 0.07:
            return face
    return None  # 没有稳定贴近的面
```

**关键设计**:
- 用**中位数** (不是均值) 避免边界帧 (approach/release) 拉高距离
- 7cm 阈值 = G1 hand_collision 球半径 (5cm) + 接触余量 (2cm)
- 60% 稳定性阈值避免短暂飞过的面被误判

### 2. grasp_type 5 类分类

```
对侧 (opposite):  L 与 R 在同一平面两侧 (±yz 或 ±xz)         ⇒ valid (force-closure 可能)
垂直 (perpendicular): 一个在 ±xy (top/bot), 另一个在 ±yz/±xz   ⇒ valid (一个支撑重力 + 一个稳定)
错位 (adjacent):  两个相邻面 (例如 -yz + +xz)                   ⇒ INVALID (力学不闭合)
同面 (same):      L 与 R 在同一个面                              ⇒ INVALID (异常)
单手 (single):    dominant_hand=L/R, 只看 dominant 那只          ⇒ valid 取决于 dominant 是否贴面
```

### 3. 物体 collision box 半尺寸自适应

不 hardcode (E055 box023 用了 hardcode), 改从 model 读取：每个 case 的 object body 上找 `geom_type=BOX, contype>0` 的 collision geom, 用 `m.geom_size` 作为 half-sizes。

```
box021:        half = (0.160, 0.209, 0.265) m
box023:        half = (0.179, 0.183, 0.206) m
bucket001:     half = (0.110, 0.189, 0.253) m
bucket005_s2:  half = (0.158, 0.162, 0.231) m
bucket007:     half = (0.288, 0.300, 0.344) m
desk021:       half = (0.336, 0.285, 0.289) m
```

### 4. 视频核实 (C4)

抽 3 个代表 case × 3 帧 (intent 起/中/末) 作为视觉 ground truth, 与算法分类做交叉验证。

## 数值结果

### 6 case 完整分类

| Case | dominant | intent | L main | L med | L close% | R main | R med | R close% | grasp_type | valid |
|------|----------|--------|--------|-------|----------|--------|-------|----------|------------|-------|
| box021 | both | 17–84 (68f) | +xy | +3.2cm | 100% | +xy | +0.8cm | 100% | **同面** | ❌ |
| box023 | both | 21–78 (58f) | -xy | +2.6cm | 88% | -yz | +1.3cm | 100% | **垂直** | ✅ |
| bucket001 | L | 19–76 (58f) | +xz | +1.7cm | 100% | – | +5.7cm | 59% | **单手 (L on +xz)** | ✅ |
| **bucket005_s2** ⭐ | both | 20–107 (88f) | -yz | +0.6cm | 100% | +yz | +3.2cm | 100% | **对侧** | ✅ |
| bucket007 | both | 19–60 (42f) | +xy | +2.3cm | 100% | +yz | +5.2cm | 84% | **垂直** | ✅ |
| desk021 | both | 24–92 (69f) | +xz | +2.2cm | 89% | -xy | +2.5cm | 92% | **垂直** | ✅ |

### 分布

```
✅ valid (5/6):
   bucket005_s2 (对侧, ⭐ 推荐 E057 起点 — 88 帧最长 + 唯一真正对侧握)
   box023        (垂直, 托底+扣远)
   bucket007     (垂直)
   desk021       (垂直)
   bucket001     (单手 L on +xz)

❌ invalid (1/6):
   box021        (同面 +xy/+xy = 双手都在 box 顶部, 不是搬运握)
```

## Claims 验证

| ID | 描述 | 量化标准 | 实际 | 通过 |
|---|---|---|---|---|
| C1 | 6 case 全跑通 face 诊断 | 6 png + 1 csv 完整 | 6 png + 1 csv ✅ | ✅ |
| C2 | summary csv 标注每 case 类型 | csv 含 L_main_face / R_main_face / grasp_type | 15 列 6 行完整 | ✅ |
| C3 | ≥1 case 是对侧或垂直 | grasp_type ∈ {对侧, 垂直} 计数 ≥1 | **4 个** (1 对侧 + 3 垂直) | ✅ 超目标 |
| C4 | 视频核实与算法一致 | 抽 3 case × 3 帧目检 | 3/3 case 视频与算法分类一致 (见下) | ✅ |
| C5 | 输出 E057 决策 | log 末尾给出明确 case + 路径 | 推荐 bucket005_s2_person1 走路线 A | ✅ |

**5/5 通过 ✅**

### C4 视频核实详情

| Case | 帧 | 视觉观察 | 算法分类 | 一致性 |
|---|---|---|---|---|
| box023 | t=1.65s | L 在 box 左下 (从下往上托); R 在 box 右后 (从远侧扣) | 垂直 -xy / -yz | ✅ |
| bucket005_s2 | t=2.5s | L 在 bucket **左侧**, R 在 bucket **右侧**, 真正两侧夹 | 对侧 -yz / +yz | ✅ |
| box021 | t=2.0s | 双手**都在 box 顶面**, 从上往下按 (像按压) | 同面 +xy / +xy | ✅ |

**3/3 一致, 算法可信。**

## 与 E055 box023 结论的修正

E055 与用户讨论时，我**错误地同意了"L 一直在 +xz 面"**。E056 视频核实 + 时间序列重新分析后**纠正**:

- L 在 +xz 的 signed distance 全程在 -1 ~ -8cm (**负值 = palm 在 box 内部 y 范围**), 不是接触 +xz 面
- L 在 -xy 的 signed distance 全程在 0 ~ +3cm (**正值 = palm 在 box 底面外侧**), 是真正接触 -xy 底面
- 用户当时看 +xz 曲线接近 0 线得出"L 在 +xz" 的结论, 但**忽略了符号** — 接触面必须 signed_dist ≥ 0
- 视频核实 (box023 t=1.65s) 显示 L palm 在 box 的左下角, 从下方托起, **是 -xy 接触, 不是 +xz**

**E055 closest_face 算法本质上是对的**, 我之前回应"算法选错"是错的。box023 ref 的握姿 (-xy 托底 + -yz 扣远) **是 valid 的垂直握**, 不是错位。

**用户的整体直觉"box023 ref 看起来怪" 仍部分有效**, 但具体表现是 box021 (双手按顶) 这种 case, 不是 box023。

## 教训 (写入 EXPERIMENT_TRACKER 教训段)

1. **接触面判定必须看 signed distance 的符号**: |dist| 接近 0 + signed = 负值 ≠ 接触, palm 在 box 内部不能从外侧接触面。**只有 signed_dist ≥ 0 的面才可能是接触面**。

2. **不要轻易因 user 的视觉判断推翻算法结论**: 在 E055 收尾时我看到用户说"L 在 +xz" 就快速同意, 没仔细对比时间序列符号。**正确流程**: 算法 + 用户视觉**双向验证 + 视频抽帧核实**, 一致才接受, 不一致要追到底找一致点。

3. **多 case 诊断比单 case 调参更高效**: 6 case 跑完 30s, 立即看出"box023 不是最差的", 真正最差的是 box021 (双手按顶)。**任何方法验证应该先跑 6+ case 看分布, 不要只在 1 个 case 上死磕**。

4. **half-sizes 不能 hardcode**: E055 在 hand_snap_ik.py 里 hardcode 了 box023 的 (0.18, 0.18, 0.21), 不能复用到其他 case。E056 改从 `model.geom_size[gid]` 读取, 6 case 通用。**任何"按物体几何"的逻辑必须从 model 读, 不写常量**。

5. **"同面"分类发现意外异常 (box021)**: 之前 E054 把 box021 归 B+C, 没注意到双手都在顶面的异常。**E054 的几何 + intent 检测不够, E056 的 hand-face 诊断是必要补充**。box021 应该从 Path B 候选中移出。

## 改动文件

| 文件 | 改动 |
|---|---|
| `workspace/core4d/scripts/E056/multi_case_face_diagnosis.py` | 新建 (~280 行) — 6 case face dist 诊断 + 5 类 grasp 分类 + 自适应 half-sizes |
| `workspace/core4d/scripts/run_E056_diagnosis.sh` | 新建 — 一键脚本 (诊断 + 抽帧验证) |
| `workspace/core4d/results/E056/case_grasp_type_summary.csv` | 新生成 |
| `workspace/core4d/results/E056/E056_summary.md` | 新生成 |
| `workspace/core4d/results/E056/<case>_face_dist.png` | 新生成 (×6) |
| `workspace/core4d/results/E056/video_verify/*.jpg` | 新生成 (×9) |
| `workspace/core4d/EXPERIMENT_TRACKER.md` | 添加 E056 行 + scripts 引用 |

## E057 决策

**推荐路线 A**: 在 `bucket005_s2_person1` 上重做 E055 snap (更新 hand_snap_ik.py 路径, 跑诊断, 复用 E055 输出格式)

**理由**:
1. 唯一真正"对侧"握姿 (-yz / +yz, 同平面两侧)
2. intent 窗口 88 帧最长 (vs box023 的 58 帧)
3. obj_z_amp 40.6cm 也是 Tier 1 充足抬升
4. 视频核实双手对称两侧夹, 是教科书搬运姿势 — snap 在这里能产出**物理合理**的 warmstart

**E057 范围**:
- 改 `snap_box023.py` → `snap_bucket005_s2.py` (核心调用 hand_snap_ik 不变)
- 输出 `workspace/core4d/results/E057/bucket005_s2_person1/{warmstart_qpos.npz, snap_diagnostics.csv, snap_visualization.mp4}`
- Claims: 与 E055 同 (palm-to-surface, 仅手臂改动, 视觉合格, 不爆炸, 一键脚本)
- **新 Claim**: 接触面与 E056 诊断一致 (L on -yz, R on +yz 之 ±5cm 内), 不漂移到其他面

如果 E057 在 bucket005_s2 上视觉 + 数值都通过, 进 E058 (Path B-CEM)。
