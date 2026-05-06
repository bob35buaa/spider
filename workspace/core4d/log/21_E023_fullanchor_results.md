# E023: Full Anchor (XY+Yaw) — 结果

## 状态: 部分成功 (pelvis_err ↓48% on bucket010, 但物体交互无改善)

## 核心发现

1. **Full anchor 对 bucket010 的 pelvis_err 改善显著**: 0.293m → 0.154m (↓48%)
2. **但物体交互反而恶化或持平**: lift 从 26%→0.5% (body-only), 20% (with-obj)
3. **根因: 物体初始就不在单人臂展范围**: bucket010 frame0 obj-pelvis=0.77m, 远超 G1 臂展 (0.5m)
4. **chair022 仍为碰撞推飞** (140%): full anchor 不改变碰撞动力学
5. **CORE4D 双人数据的结构性限制**: 物体位置由两人共同决定, 单人 anchor 后物体漂移

## 实验矩阵

| Case | Mode | pelvis_err | lift% | stable% | vs E021/E022 |
|------|------|-----------|-------|---------|------|
| bucket010 | full-anchor body | **0.154** | 0.5% | 100% | pelvis ↓48%, lift ↓ |
| bucket010 | full-anchor+obj | 0.199 | 19.7% | 100% | pelvis ↓, lift ≈ |
| chair022 | full-anchor body | 0.442 | 136% | 100% | 碰撞推飞不变 |
| chair022 | full-anchor+obj | 0.402 | 140% | 100% | 碰撞推飞不变 |
| desk005 | full-anchor body | 0.261 | 0.9% | 100% | pelvis ↓7%, lift ↓ |
| desk005 | full-anchor+obj | 0.261 | 0.9% | 100% | obj_rew 无效 |

## Claims 验证 (严格标准)

| Claim | 结果 | 通过? |
|-------|------|------|
| C1: pelvis_err ↓20% (bucket010/chair022) | bucket010 ↓48% ✅, chair022 +6% ❌ | **部分** (1/2) |
| C2: 手到 obj <0.10m 持续 15帧 | 无 case 达到 | **FAIL** ❌ |
| C3: obj_Δz>0.03m 持续 30帧 | 无 case 达到 | **FAIL** ❌ |
| C4: 稳定性 ≥95% | 全部 100% | **PASS** ✅ |

## 可视化观察

### bucket010 with-obj (t=45%)
- **ref**: G1 正面抱桶, 双手在桶侧面
- **sim**: 桶在远处 (初始就 0.77m), G1 手在身侧未伸向桶
- **分析**: 物体离机器人太远, CEM 找不到有效的接触策略

### chair022 with-obj (t=45%)
- **ref**: G1 弯腰从椅背侧搬起
- **sim**: G1 弯腰姿态相似, 手碰到椅子底部/腿 → 推飞
- **分析**: 接触确实发生, 但方向错误 (碰底部非抓椅背)

## 结构性结论

**CORE4D 双人搬运数据对单机器人 SPIDER 的限制**:

1. **物体位置由两人共同决定** — 单人 pelvis anchor 后, 物体相对位置不再对准手
2. **初始 obj-pelvis 距离 0.5-0.8m** — 超过 G1 臂展 (≈0.5m 从肩到指尖)
3. **无论怎么 anchor, 单人无法触碰由两人合作维持的物体位置**

这与 E001-E009 的 box025 结论一致: **协作数据需要协作建模**。

## 下一步方向

E017-E018 (双机器人+connect) 仍是唯一证明有效的方案。
单人 anchor 方向的价值: **作为 RL 训练的高质量 body-only 参考轨迹** (pelvis_err 0.15-0.26m), 但不能期望单人物理搬运。

## 结果路径

| 产出 | 路径 |
|------|------|
| 脚本 | `scripts/convert/anchor_pelvis_full.py` |
| bucket010 | `results/E023_fullanchor/bucket010/{bodyonly,withobj}.{npz,mp4}` |
| chair022 | `results/E023_fullanchor/chair022/{bodyonly,withobj}.{npz,mp4}` |
| desk005 | `results/E023_fullanchor/desk005/{bodyonly,withobj}.{npz,mp4}` |
| 关键帧 | `results/E023_fullanchor/keyframes/*.png` |
| 计划 | `plan/23_E023_fullanchor_plan.md` |
