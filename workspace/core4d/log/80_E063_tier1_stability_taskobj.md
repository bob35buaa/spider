# E063 Tier 1 — re-enable stability_penalty + reduce task_obj on box023 (+ box025 regression)

## 状态: ⚠️ **MIXED — pelvis_min 0.058→0.192m (+13.4cm 但严格 C1≥0.50 仍 FAIL); 失败模式由"摔倒+趴地"变为"前扑+水平 superman 姿"; box025 regression PASS**

**TL;DR**: 按 log 79 §5 推荐, 在 E062 基础上加 `stability_penalty_scale=1.0` + 减半 `task_obj_pos/rot_rew_scale=0.5`. box023 数值方向性改善 (pelvis_min +13cm, pelvis_mean_intent +5cm) 但严格 C1 仍 FAIL; **新失败模式**: stability_penalty(threshold=0.55m)只惩罚 pelvis 高度不约束 torso 朝向, CEM 没有"摔倒躺地"反而找到了"水平前扑"局部最优 — t=2.0-2.5s sim 身体几乎平躺但 pelvis 仍在 0.19-0.46m, 像 superman 姿势冲向 box. 终态 box 偏差仍只 0.81cm (E062 1.36cm), 说明任务"完成"但姿态完全不对. box025 regression guard PASS (pelvis_min 0.685m, stable 100%, final_obj 10.7cm — 全部维持). **决策**: 按 log 79 §5 decision tree, C1 FAIL → 进 Tier 2 (E064: `local_frame_root_sigma 0.5→0.3` + `contact_hdmi_gain 5.0→3.0`), 同时 raise `stability_penalty_threshold` 0.55→0.65 阻止 superman lunge.

## 1. 改动 (E063 vs E062)

| 参数 | E062 | E063 | 动机 (log 79 §3) |
|------|------|------|------------------|
| `stability_penalty_scale` | 0.0 | **1.0** | 给摔倒帧 negative reward, 阻止 deep fall (T1-A) |
| `task_obj_pos_rew_scale` | 1.0 | **0.5** | 减弱 forward 牵引, 让 sim 不为追物体牺牲平衡 (T1-B) |
| `task_obj_rot_rew_scale` | 1.0 | **0.5** | 同上 |
| `stability_penalty_threshold` | 0.55 | 0.55 (未改) | E034 默认 |
| 其他 (palm_normal, gain, sigma...) | 同 E062 | 同 E062 | 控制变量 |

## 2. 训练设置

- 配置: `examples/config/override/core4d_e063_box023.yaml` + `core4d_e063_box025.yaml` (各继承 `core4d_e062_*`)
- 脚本: `workspace/core4d/scripts/train/train_E063.sh parallel 0 1`
- 用时: box025 31min (GPU1), box023 33min (GPU0); 并行总耗 33min
- 物理稳定性: plan time 14-15s/iter, 与 E062 同 (无 contact 爆炸)
- Snapshot: `workspace/core4d/results/E063/scene_snapshot/` (manifest git HEAD `1921f1e`)
- 输出 npz/mp4: `workspace/core4d/results/E063/E063_{box023,box025}.{npz,mp4}`

## 3. 量化结果 (eval_E063.py)

### box023 (target case)

| 指标 | E062 | E063 | Δ | C 阈值 | PASS? |
|------|------|------|----|--------|-------|
| pelvis_min (full) | 0.058m | **0.192m** | +0.134m | — | — |
| pelvis_min_intent | 0.083m | 0.192m | +0.109m | C1 ≥ 0.50 | ❌ |
| pelvis_mean_intent | 0.486m | 0.540m | +0.054m | C2 ≥ 0.50 | ✅ |
| pelvis_max | 0.83m | 0.83m | 0 | — | — |
| stable% (full) | 77.94% | 79.41% | +1.5pp | — | — |
| stable% (intent) | 65.52% | 67.24% | +1.7pp | C3 ≥ 80% | ❌ |
| final_obj_pos_err | 1.36cm | **0.81cm** | -0.55cm | C4 ≤ 20 | ✅ |
| obj_err_mean (全帧) | 9.82cm | 11.68cm | +1.86cm | — | — |
| L palm contact% | 96.55% | 91.38% | -5.2pp | — | — |
| R palm contact% | 91.38% | 96.55% | +5.2pp | — | — |
| both palm contact% | 87.93% | 87.93% | 0 | — | — |

**Verdict**: C1+C3 FAIL, C2+C4 PASS, C5 (vis) FAIL → **2/5 PASS, 整体 FAIL**

### box025 (regression guard)

| 指标 | E062 | E063 | R 阈值 | PASS? |
|------|------|------|--------|-------|
| pelvis_min | 0.688m | **0.685m** | R1 ≥ 0.65 | ✅ |
| stable% (full) | 100% | **100%** | R2 ≥ 99% | ✅ |
| final_obj_pos_err | 12.77cm | **10.66cm** | R3 ≤ 25 | ✅ |
| both palm contact% | 51.85% | 43.21% | — | (-8.6pp 但仍 high) |

**Verdict**: **3/3 PASS — reward 改动不破坏 box025 baseline**.

## 4. ref vs sim 物体轨迹 + pelvis_z 全帧对比 (E063 box023)

| 时间 | ref obj xyz | sim obj xyz | obj err | sim pz | 解读 |
|------|------------|------------|---------|--------|------|
| 0.40s | (-0.83, -1.18, 0.14) | (-0.83, -1.18, 0.14) | 1.6cm | 0.83 | 站立接近 |
| 1.00s | (-0.75, -1.12, 0.40) | (-0.83, -1.16, 0.20) | 21cm | 0.62 | 滞后弯腰 |
| 1.67s | (-0.12, -0.55, 0.66) | (-0.50, -0.71, 0.23) | 50cm+ | 0.67 | sim 仍在原地, ref 已走 |
| 2.00s | (+0.25, -0.29, 0.54) | (+0.04, -0.44, 0.47) | 26cm | **0.46** | sim 开始前扑 |
| 2.27s | (+0.40, -0.20, 0.39) | (+0.26, -0.20, 0.20) | 19cm | **0.29** | 深 lunge |
| **2.37s** | (+0.43, -0.19, 0.31) | (+0.30, -0.20, 0.16) | 13cm | **0.19** ⭐ | 最低点 (E062 此时 0.06m) |
| 2.50s | (+0.45, -0.18, 0.21) | (+0.26, -0.27, 0.18) | 21cm | 0.27 | 仍 horizontal |
| 3.00s | (+0.46, -0.16, 0.15) | (+0.38, -0.20, 0.16) | 8.6cm | 0.55 | 半起身 |
| 4.00s | (+0.46, -0.16, 0.15) | (+0.46, -0.15, 0.16) | 1.1cm | 0.81 | 直立但 box 已落定 |
| 4.50s | (+0.46, -0.15, 0.15) | (+0.46, -0.15, 0.16) | **0.8cm** | 0.76 | 终态完美 |

**对比 E062 (log 79 §2)**:
- E062 在 t=2.66s 最低 0.058m (完全趴地), E063 在 t=2.37s 最低 0.192m (深前扑) — **+13cm 但仍远低于站立**
- E062 t=2.0s pz=0.42m, E063 同帧 0.46m — 临界点向后挪了一点
- E062 7-8 帧 pz<0.30m, E063 4 帧 pz<0.30m — 摔倒窗口缩短一半但仍存在

## 5. 视觉证据 (9 dense frames @ box023)

| 时间 | E063 sim 行为 | 对比 E062 同帧 |
|------|--------------|----------------|
| 0.40s | 站立微弯腰接近 box | 类似 |
| 1.00s | 弯腰 box 在身前, 手伸向 box | 类似 |
| **1.67s** | **深 lunge 姿势, 右脚伸向后, 身体前倾** | E062 此时已经"半抬箱" |
| **2.00s** | **左脚已完全抬空脚后退, 右脚踩在 box 边角附近, 身体水平** | E062 完全前倾, 还没栽 |
| **2.33s** | **几乎水平 superman 姿, 双手扑向 box** | E062 此时 pz=0.13m, 趴向 box |
| **2.66s** | **完全水平身体悬浮在 box 上方, 双手扣 box** | E062 完全趴地 (pz=0.06m) |
| 3.00s | 起身但身体仍倾斜, 手仍在 box | E062 push-up 姿 |
| 3.33s | 类似站姿但弓步 | 类似 |
| 4.00s | **直立站立但右腿仍后撑** (push-up 残影) | E062 完全直立 |

**核心区别**: E062 是"摔倒→push-up 起身→站立", E063 是"水平 superman lunge 全程→最后站起". E063 没有典型 fall, 但也没有典型 carry → 是个新的局部最优.

**为什么 stability_penalty=1.0 没阻止 superman lunge**:
- threshold = 0.55m, sim 大部分时间 pz=0.46-0.55m, 只有 4 帧 < 0.30m
- penalty 总贡献 ~ -4 (4 帧 × -1) vs task_obj 总贡献 +50+ → CEM 接受 4 帧 deep lunge 换取整体 task 完成
- threshold 把"pelvis 高度"当 stability proxy, 但没约束 **torso 朝向 / 脚地接触 / COM 在脚底**

## 6. Claims 验证表

| ID | 标准 | 实际 | 通过 |
|----|------|------|------|
| C1 | pelvis_min_intent ≥ 0.50m | 0.192m | ❌ FAIL (差 31cm) |
| C2 | pelvis_mean_intent ≥ 0.50m | 0.540m | ✅ PASS |
| C3 | stable% intent ≥ 80% | 67.2% | ❌ FAIL |
| C4 | final_obj_pos_err ≤ 20cm | 0.81cm | ✅ PASS (excellent) |
| C5 | t=2.0/2.3/2.7s 视觉无摔倒 phase | superman lunge 替代 fall | ❌ FAIL (新失败模式) |
| R1 | box025 pelvis_min ≥ 0.65m | 0.685m | ✅ PASS |
| R2 | box025 stable% ≥ 99% | 100% | ✅ PASS |
| R3 | box025 final_obj ≤ 25cm | 10.7cm | ✅ PASS |

**box023: 2/5 PASS (FAIL 整体)**, **box025: 3/3 PASS (regression guard 通过)**

## 7. 改动文件

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `examples/config/override/core4d_e063_box023.yaml` | 新建 | 3 行 override (stability_penalty + task_obj_pos/rot) |
| `examples/config/override/core4d_e063_box025.yaml` | 新建 | 同 box023 (regression guard) |
| `workspace/core4d/scripts/train/train_E063.sh` | 新建 | 复用 E062 结构, 并行 GPU 0/1, 含 snapshot |
| `workspace/core4d/scripts/eval/eval_E063.py` | 新建 | 复用 eval_E060_0 + 加 ref vs sim obj 全帧 + final pos err; **修了 scene_act vs scene 加载 bug** (复用代码 silently 用错 XML, nq=42 vs 43, obj_xpos 全错) |
| `workspace/core4d/scripts/eval/extract_E063_keyframes.sh` | 新建 | 9 dense frames @ box023 (含 fall 窗口), 4 frames @ box025 |
| `workspace/core4d/scripts/run_E063.sh` | 新建 | 一键 train→eval→keyframes |
| `workspace/core4d/results/E063/scene_snapshot/` | 新建 | manifest.txt git HEAD `1921f1e` + sha256 box023+box025 scene XML |
| `workspace/core4d/results/E063/E063_*.{npz,mp4}` | 新建 | 训练产出 (1.9M+0.9M npz, 1.1M+0.9M mp4) |
| `workspace/core4d/results/E063/eval_summary.csv` | 新建 | 4 行 (E062 + E063 × box023 + box025) |
| `workspace/core4d/results/E063/obj_trace_E063_*.csv` | 新建 | 全帧 ref vs sim obj xyz + err + pelvis_z |
| `workspace/core4d/results/E063/face_dist_*.png`, `pelvis_obj_*.png` | 新建 | 可视化 |
| `workspace/core4d/results/E063/keyframes/E063_*_kf*.jpg` | 新建 | 13 frames |

## 8. 下一步: E064 = Tier 2 + threshold raise

按 log 79 §5 decision tree 第 3 分支 ("C1 FAIL: 仍摔") → 进 Tier 2. 但**不只是抄 log 79 §4 推荐, 还要回应 E063 的新失败模式**:

```yaml
# core4d_e064_box023.yaml
defaults:
  - core4d_e063_box023   # 继承 E063 (含 T1-A + T1-B)
  - _self_

# T2-A: tighter root tracking (Tier 2 from log 79 §4)
local_frame_root_sigma: 0.3        # was 0.5

# T2-B: weaker contact pull (Tier 2 from log 79 §4)
contact_hdmi_gain: 3.0             # was 5.0

# T2-C (NEW, 应对 E063 superman 模式): raise stability threshold
stability_penalty_threshold: 0.65  # was 0.55, 阻止 deep lunge (sim 当前在 0.46-0.55m 安全区)
```

**预期效果**:
- T2-A 收紧 root_sigma → CEM 不能让 pelvis 远离 ref 的 0.7+m, 直接惩罚 superman 时的 pz=0.19m
- T2-B 减小 contact gain → sim 不那么急切伸手追 box, 减少前扑动机
- T2-C 提 threshold → 当前 E063 大部分 superman 帧 pz=0.30-0.55m **都被 penalty 覆盖**, 不只 4 个 deep 帧

**风险**:
- T2 三项叠加可能过强压制 task — 同时跑 box025 regression. 若 box025 stable% 跌 < 95% 或 obj_err > 20cm 立即回退最严的一个
- 若仍 FAIL: 回 log 79 §4 Tier 3 = **rebuild reward**, 加 walking phase / foot-contact / COM-in-support 等 hand-crafted term

**不再尝试**: T1 ablation (拆 stability vs task_obj 单独跑) — 浪费 GPU 时间, E063 已证明组合方向正确 (pelvis_min +13cm, box025 不退化), 单项不会更好.

## 9. 教训 #9 (累计 8 个之前)

之前 8 个教训 (log 79 §7 + 历史):
1-7: 见 log 78 §9 + log 79 §7
8: 诊断 reward 必须看 ref vs sim 物体轨迹 + pelvis_z 全帧对比

**新教训 #9 (本 log)**:

> **stability_penalty(threshold) 是 height proxy 不是 stability metric. CEM 在 height-only 约束下能找到"水平身体 + 高 pelvis"的 superman / lunge 局部最优**. E063 把 stability_penalty_scale 从 0.0 → 1.0 后, sim 的 pelvis_min 从 0.058m → 0.192m 但是 torso 全程水平地"飞向" box, 终态 box 完美 (0.8cm 偏差) 但姿态完全错. 单一 height 约束不构成 "stable carry" 的充分条件.

具体规则:
- 任何"防摔"reward 设计必须**同时约束** (a) pelvis 高度 (b) torso 朝向 (c) COM 在脚底支撑 — 三者缺一就有局部最优逃出
- 任何 reward 改 PASS 标准必须**包含视觉 5 帧检查**, 不能只看 pelvis_min — E063 看数值像"摔得轻了", 看视频是另一种姿态崩坏
- "数值方向性改善"≠"接近成功". E063 pelvis_min +13cm 听起来不错, 但绝对值 0.19m 仍是水平身体, 离 0.55m"扶箱站立"还差很远. 严格 strict 阈值是必要的, 不能因为"改善 13cm"就放宽

## 10. 关联

- log 78: E062 X1 auto palm_normal mixed result, box023 carry-fall-recover 模式
- log 79: E062 box023 深度诊断 + Tier 1/2/3 候选 — 本 log 是 Tier 1 验证
- log 79 §5 decision tree 第 3 分支 → 本 log 进 E064 Tier 2
- audit log 70 §1.2: stability_penalty 历史 (E034 引入, E036 关闭) — 本 log 重启确认 height-only 不够
