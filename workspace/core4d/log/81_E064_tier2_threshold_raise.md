# E064 Tier 2 — root_sigma 0.3 + contact_gain 3.0 + stability_threshold 0.65 (extends E063)

## 状态: ❌ **FAIL — 第三个失败模式 "lie-down-and-stay" (E062 fall, E063 superman, E064 prone). 教训 #10: 单 case reward 调参 3 次都让 CEM 找到新局部最优, 不是真正的 stability 解. box025 regression R1 边际 FAIL (0.643 vs 0.65, -7mm). 建议: 停止单 case 调参, 转 X5 多 case 验证 X1+T1 baseline**

**TL;DR**: 按 log 80 §8 推荐进 Tier 2 + threshold raise, 跑 box023 + box025. **box023**: pelvis_min_intent 0.192→0.178 (基本不变), pelvis_mean_intent +4cm, stable% intent 67→79% 接近 C3, 但 stable% **full** 79→49% (-30pp 暴跌, post-intent 全程 pelvis 0.40-0.49m, sim 倒地后**不恢复站立**). final box err 1.96cm 任务"完成"但姿态完全错. **box025 regression**: pelvis_min 0.685→**0.643m** (R1 ≥ 0.65 边际 FAIL by 7mm), 说明加严的 reward 已开始破坏 box025 baseline. 视频证实新失败模式: **t=2.66s sim 完全平躺手伸向 box, t=3.33s 仍 push-up 起始姿, t=4.0s 深 kneel 一腿前伸 — 全程不站起**. **E062→E063→E064 三次 reward weight 调参都让 CEM 找到 NEW local optimum, 没一次真正解决"carry+place+stand"**, 证明这是参数空间 dead-end. 决策: stop 单 case 调参, 进 X5 = E065 推 X1+T1(E063 weights, 不是 E064) 到 4 cases 验证泛化, 同时排队 Tier 3 (walking phase / COM-in-support, 需 reward 代码改).

## 1. 改动 (E064 vs E063)

| 参数 | E063 | E064 | 动机 (log 80 §8) |
|------|------|------|------------------|
| `local_frame_root_sigma` | 0.5 | **0.3** | T2-A: 紧 root tracking 阻止 pelvis 远离 ref |
| `contact_hdmi_gain` | 5.0 | **3.0** | T2-B: 弱 contact 拉力, 减少前扑动机 |
| `stability_penalty_threshold` | 0.55 | **0.65** | T2-C: 提高摔倒阈值, 覆盖 superman 帧 (E063 在 0.30-0.55m) |
| 其他 (stability_scale=1, task_obj=0.5, palm_normal...) | 同 E063 | 同 E063 | 控制变量 |

## 2. 训练设置

- 配置: `examples/config/override/core4d_e064_box023.yaml` + `core4d_e064_box025.yaml`
- 脚本: `workspace/core4d/scripts/train/train_E064.sh parallel 0 1`
- 用时: box025 30min (GPU1), box023 33min (GPU0); 并行总耗 33min
- 物理稳定性: plan time 14s/iter, 与 E063 同
- Snapshot: `workspace/core4d/results/E064/scene_snapshot/` (manifest git HEAD `1921f1e`)

## 3. 量化结果 (eval_E064.py)

### box023 (target case)

| 指标 | E063 | E064 | Δ | C 阈值 | PASS? |
|------|------|------|---|--------|-------|
| pelvis_min (full) | 0.192m | 0.178m | -1.4cm | — | — |
| pelvis_min_intent | 0.192m | 0.178m | -1.4cm | C1 ≥ 0.50 | ❌ |
| pelvis_mean_intent | 0.540m | **0.583m** | +4.3cm | C2 ≥ 0.50 | ✅ |
| stable% (full) | 79.4% | **49.3%** | **-30.1pp** | — | ⚠️ collapse |
| stable% (intent) | 67.2% | **79.3%** | +12.1pp | C3 ≥ 80% | ❌ (差 0.7pp) |
| final_obj_pos_err | 0.81cm | 1.96cm | +1.1cm | C4 ≤ 20 | ✅ |
| obj_err_mean (全帧) | 11.68cm | 10.96cm | -0.7cm | — | ✅ better |
| both palm contact% | 87.93% | 75.86% | -12.1pp | — | weaker |

**Verdict**: C1 仍 FAIL (0.178 << 0.50), C3 边际 FAIL (差 0.7pp), C2+C4 PASS, **C5 vis FAIL (新失败模式)**. **整体 FAIL (2/5)**.

### box025 (regression guard)

| 指标 | E063 | E064 | R 阈值 | PASS? |
|------|------|------|--------|-------|
| pelvis_min | 0.685m | **0.643m** | R1 ≥ 0.65 | ❌ (差 7mm) |
| stable% (full) | 100% | 100% | R2 ≥ 99% | ✅ |
| final_obj_pos_err | 10.66cm | 11.77cm | R3 ≤ 25 | ✅ |
| both palm contact% | 43.21% | 35.80% | — | -7.4pp |

**Verdict**: **R1 边际 FAIL** (差 7mm 是噪声级别但严格 fail), R2+R3 PASS. Reward 加严已开始 marginally regress box025.

## 4. 关键诊断: post-intent pelvis collapse (E064 box023)

E064 box023 frame-by-frame pelvis_z (frames where pz < 0.55):

```
INTENT (frames 21-78, t=0.7-2.6s):
  frame 60-78 (t=2.0-2.6s): pz dips 0.53→0.18→0.54  (深 V 形 lunge, 同 E062/E063 形态)

POST-INTENT (frames 79-135, t=2.6-4.5s):
  全部 56 帧 pz 0.39-0.49m, NEVER recovers to 0.55m
  E063 同时段: t=3.0s pz 0.55m, t=4.0s pz 0.81m (站起)
  E064 同时段: t=3.0s pz 0.41m, t=4.0s pz 0.42m (持续 prone)
```

**关键发现**: E064 把 stable% 拆分看:
- intent% 提升 (67→79%) 是因为深 V 触底后 quick recover **回到 intent 内**
- full% 暴跌 (79→49%) 是因为 post-intent **彻底躺平不站起**

T2-C (threshold 0.65) 让 sim 在 intent 内 "尽快回到 0.55+" 拿 reward, 但 post-intent (无 task_obj 推力) sim 找到了 "lie flat" 局部最优 — pelvis 0.42m 离 threshold 不远, deficit penalty 较小, 而站起需要 muscular effort 增加 ctrl noise → CEM 不选.

## 5. 视觉证据 (9 dense frames @ box023)

| 时间 | E064 sim 行为 | vs E063 同帧 | vs E062 同帧 |
|------|--------------|---------------|---------------|
| 0.40s | 站立微弯腰 | 同 E063 | 同 |
| 1.00s | 弯腰 box 在身前 | 同 | 同 |
| 1.67s | 深 lunge 右脚伸后 | 同 E063 | E062 已抬箱 |
| 2.00s | 左脚抬空 + 身体倾斜 | 类似 E063 | 临界前 |
| 2.33s | **几乎水平 superman** | 同 E063 形态 | E062 在前栽 |
| **2.66s** | **完全平躺**, 双脚翘起, 腹部贴地 | E063 是悬浮 superman, E064 完全趴地 | E062 也趴地 (pz=0.06m) |
| **3.00s** | push-up 起始姿, 双手在 box | E063 此时半起身 (0.55m) | E062 push-up 类似 |
| **3.33s** | 仍 push-up start 姿势, 身体水平脚后撑 | E063 弓步直立 | E062 站立 |
| **4.00s** | **深 kneel 单膝跪, 一腿前伸** (sit pose) | E063 直立微残影 | E062 完全直立 |

**核心区别 vs E063**: E063 sim 在 t=3-4s push-up 起身回到站立 (0.81m), E064 sim **从未起身**, 终态 kneeling/sitting 0.42m. 加严 reward 反而剥夺了 sim 的 active recovery 能力.

**核心区别 vs ref**: ref 在 t=4s 完全直立离 box 1m, 任务彻底完成. sim t=4s 仍坐在 box 旁边 0.4m 高度.

## 6. Claims 验证表

| ID | 标准 | 实际 | 通过 |
|----|------|------|------|
| C1 | pelvis_min_intent ≥ 0.50m | 0.178m | ❌ FAIL (差 32cm, vs E063 差 31cm — **基本无改善**) |
| C2 | pelvis_mean_intent ≥ 0.50m | 0.583m | ✅ PASS |
| C3 | stable% intent ≥ 80% | 79.3% | ❌ FAIL (差 0.7pp) |
| C4 | final_obj_pos_err ≤ 20cm | 1.96cm | ✅ PASS |
| C5 | t=2.0/2.3/2.7s 视觉无摔倒 phase | **更糟: full prone, post-intent 不起身** | ❌ FAIL (新失败模式) |
| R1 | box025 pelvis_min ≥ 0.65m | 0.643m | ❌ FAIL (边际 -7mm) |
| R2 | box025 stable% ≥ 99% | 100% | ✅ PASS |
| R3 | box025 final_obj ≤ 25cm | 11.77cm | ✅ PASS |

**box023: 2/5 PASS (FAIL 整体)**, **box025: 2/3 PASS (R1 边际 FAIL — reward 改动开始破坏 box025 baseline)**

## 7. 三次失败模式总结 (E062 → E063 → E064)

| 实验 | reward 改动 | failure mode | pelvis_min | post-intent |
|------|------------|--------------|------------|-------------|
| E062 | baseline (X1 only) | **fall + recover** (deep, then push-up to standing) | **0.058m** | recovers to 0.81m |
| E063 | + stab=1.0 + task_obj=0.5 | **superman lunge** (horizontal flying carry) | **0.192m** | recovers to 0.81m |
| E064 | + root_σ=0.3 + gain=3.0 + thresh=0.65 | **lie-down-and-stay** (prone throughout, sit at end) | **0.178m** | **stays at 0.42m, never stands** |

**三次都没一次真正解决问题**. CEM 找到的是 "least painful" 局部最优:
- E062: task_obj reward 主导 → 急追物体 → 摔; 但无 stab penalty → 摔了 OK 起身
- E063: stab penalty 阻止 deep fall → CEM 找 "horizontal but high pelvis" superman
- E064: 高 threshold 让 superman 不够好 → CEM 找 "stay near 0.4m permanently" — 全程 deficit penalty 但无站起 effort

**根本问题**: reward 数值不能区分 "carry+place+**stand**" 和 "carry+place+**lie**". stable% 只看 pelvis_z (height proxy), 不约束 (a) torso upright (b) feet support (c) end-state matches ref pose. CEM 总能在 reward landscape 找到 game.

## 8. 决策树更新 + 下一步

按 log 79 §5 + 教训 #9 (log 80) + 教训 #10 (本 log):

### 不再做 (3-strike rule, skill 规则 §12)
- **不再** 单变量调参 yaml on box023 (E063, E064 已证明这是 dead-end)
- **不再** 加严 stability_penalty (E064 边际破坏 box025)
- **不再** 假设 "再调一组 weight 就能 PASS"

### 推荐下一步 (按优先级)

**(优先 P0) E065 = X5 推 X1+T1(E063, 不是 E064) 到 4 cases**:
- log 79/80 都把 X5 列为 X1 通过后的下一步, **现在不能再推迟**
- 用 E063 配置 (T1 only, 不带 T2 — 因为 T2 边际破坏 box025) 跑 4 个 B+C cases:
  - bucket005_s2_person1 (E057 已 snap, sym contact)
  - box021_person1 (E054 Tier 1, sym contact)
  - bucket007_person1 (E054 Tier 1, sym contact)
  - desk021_person1 (E054 Tier 1, sym contact)
- 看 X1+T1 这套 reward 在新 case 上的失败模式分布: 是不是每个 case 都有自己的 superman/prone 方式? 还是某些 case 靠 X1+T1 就 PASS?
- 这才是真正的"reward 泛化"测试. 不再靠 box023 单 case overfit.

**(并排 P1, 需要 reward 代码改) Tier 3 = walking phase + COM-in-support reward**:
- log 79 §4 T3-A: 鼓励双脚交替 (foot-z alternating high-low)
- 加 COM-in-support polygon 约束 (sim foot positions 形成 polygon, COM 必须在内)
- 加 end-state pose match reward (sim 终态 pelvis_z + torso_quat 匹配 ref 最后 30 帧)
- **风险**: 改 spider 核心 reward, 不再"yaml only", 必须严格写好 + 跑历史 baseline 回归 (避免重复 fa2e181 的 box025 regression)

**(不做) E064 微调 weight 救场**:
- 比如降 stability_threshold 0.65→0.60 平衡 intent vs full stable
- 这是又一次 "再来一组 weight" 重蹈 E063/E064 覆辙, 浪费 GPU

### 用户确认事项

1. ✅ 进 E065 X5 (跑 4 case X1+T1 验证) — 这是 plan 78 §6 一开始就排队的事, 现在执行
2. ❓ 是否同时 P1 改 reward 代码 (Tier 3)? 这需要 ~2-4h 写 reward + 跑回归 + log
3. ❓ 是否 commit 当前 E063 yaml (T1) 配置? E063 box023 不是 PASS, 但 box025 regression 是 PASS, 配置本身保留有意义

## 9. 改动文件

| 文件 | 改动 | 说明 |
|------|------|------|
| `examples/config/override/core4d_e064_box023.yaml` | 新建 | 3 行 override on E063 |
| `examples/config/override/core4d_e064_box025.yaml` | 新建 | 同 (regression guard) |
| `workspace/core4d/scripts/train/train_E064.sh` | 新建 | clone E063 |
| `workspace/core4d/scripts/eval/eval_E064.py` | 新建 | clone E063 (baseline = E063 not E062) |
| `workspace/core4d/scripts/eval/extract_E064_keyframes.sh` | 新建 | 9 frames @ box023 + 4 @ box025 |
| `workspace/core4d/scripts/run_E064.sh` | 新建 | 一键 |
| `workspace/core4d/results/E064/scene_snapshot/` | 新建 | manifest git HEAD `1921f1e` (规则 10b) |
| `workspace/core4d/results/E064/E064_*.{npz,mp4}` | 新建 | 训练产出 |
| `workspace/core4d/results/E064/eval_summary.csv` | 新建 | 4 行 |
| `workspace/core4d/results/E064/obj_trace_E064_*.csv` | 新建 | 全帧 ref vs sim |
| `workspace/core4d/results/E064/face_dist_*.png`, `pelvis_obj_*.png` | 新建 | 可视化 |
| `workspace/core4d/results/E064/keyframes/` | 新建 | 13 jpg |

## 10. 教训 #10 (累计 9 个之前)

之前 9 个教训 (log 80 §9 + 历史):
1-7: log 78 §9 + log 79 §7
8: 诊断 reward 必须看 ref vs sim 物体轨迹 + pelvis_z 全帧对比 (log 79)
9: stability_penalty(threshold) 是 height proxy 不是 stability metric (log 80)

**新教训 #10 (本 log)**:

> **同一 reward 框架内的 weight 调参 3 次都让 CEM 找到 NEW local optimum, 不会真正解决问题. 当前 reward = (stability_height_proxy, task_obj, contact_dist, root_track) 这 4 维标量, 没有 (a) torso upright (b) foot support (c) end-state match — 只要这 3 个维度 unconstrained, CEM 总能找到 game**. E062→E063→E064 三次都符合这个模式: 每次封堵一个 known failure mode (fall / superman / prone), CEM 立刻找到 new failure mode.

具体规则:
- **3-strike 规则**: 同一 reward 框架内调参, 第 3 次再 FAIL 就**不能再调一次**, 必须 (a) 加新 reward 维度 / (b) 换 case / (c) 换技术路线
- **single case 不可信**: 任何 reward 改动必须有 ≥2 case regression guard. E064 看似 box023 mean 改善, 实际 box025 R1 已边际 FAIL, 证明继续加严会全面 regress
- **post-intent 不可忽略**: 之前 6 个实验都只看 intent 内指标, E064 暴露 sim 可以 "intent 完美但 post-intent 躺地不起" — eval 必须包含 full trajectory
- **failure mode 多样性是有 reward 缺陷的信号**: 如果 3 个实验产生 3 个不同的 failure mode (fall / superman / prone), 说明 reward landscape 有多个等高的 dead-end pit, 调参只是在 pit 之间换. 解法不是"找完美 weight", 是"加新约束维度"

## 11. 关联

- log 78: E062 baseline (X1 only) — failure mode 1 = fall
- log 79: E062 box023 诊断, 提出 Tier 1 (E063), Tier 2 (E064), Tier 3
- log 80: E063 Tier 1 — failure mode 2 = superman, 引入教训 #9
- 本 log (81): E064 Tier 2 — failure mode 3 = prone, 引入教训 #10, 决策 stop 单 case 调参
- audit log 70: reward task-specific 警告 — 现在被 E063/E064 进一步证实, single-case overfitting 模式仍未解决
