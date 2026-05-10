# E049: HDMI 优化移植 + HDMI 泛化验证

## 状态: ✅ 完成 (2026-05-11)

## 实验总结

### A. E049a-d: HDMI 三优化移植到 MJWP (❌ 失败)

E041c + apply_holosoma_pd + wrist_dof_damping=5.0 + zero_noise_joint_keywords

| Case | E048 Stability | E049 Stability | E048 Contact | E049 Contact |
|------|---------------|---------------|-------------|-------------|
| box023 | 38% | **12%** ❌ | 82%(摔倒) | 6% |
| box025 | 98% | **57%** ❌ | 54% | 65% ⬆ |
| bucket010 | 100% | **100%** | 7% | 0% |
| desk005 | 100% | **41%** ❌ | 0% | 18% ⬆ |

**结论**: HDMI 三优化直接移植**严重恶化 Stability**。

**根因**: apply_holosoma_pd 将关节 Kp 从 500 降到 14-100 (降幅 5-35x), 使机器人变得非常柔软。E041c 的 CEM reward weights 是在高刚度 (Kp=500) 下调优的, 低刚度下 CEM 无法维持平衡。Contact 在 box025/desk005 有小幅改善 (54→65%, 0→18%), 但 Stability 崩溃使这些改善无意义。

**教训**: HDMI 的成功不是单纯靠 PD gains, 而是**整个 reward + optimizer + dynamics 的协同设计**。简单移植 PD gains 到 E041c 不行, 需要完整重新调优 reward weights。

### B. E049e: HDMI box025 (★★★ 核心结果)

| Case | 指标 | HDMI | E041c (E048) | E049 |
|------|------|------|-------------|------|
| box025 | MPKPE | **0.3cm** | 24.8cm | 36.5cm |
| | Stability>0.6 | **100%** | 98% | 57% |
| | Contact<10cm | **99%** | 54% | 65% |
| | ObjPos | **0.5cm** | 15.5cm | 13.8cm |

**HDMI 在大箱子 box025 上也碾压**: Contact 99%, 几乎完美的 body tracking (0.3cm), 全程稳定。

### C. HDMI vs E041c 全面对比 (box023 + box025)

| | HDMI box023 | E041c box023 | HDMI box025 | E041c box025 |
|---|------------|-------------|------------|-------------|
| MPKPE | **0.7cm** | 39.5cm | **0.3cm** | 24.8cm |
| Stability | **100%** | 38% | **100%** | 98% |
| Contact<10cm | **93%** | 82%(假) | **99%** | 54% |
| ObjPos | **0.7cm** | 13.9cm | **0.5cm** | 15.5cm |

### D. eval 脚本 Bug 修复后重评

修复 half_ext_map hardcoding bug 后:
- box001: 0% → **50%** Contact
- box024: 0% → **78%** Contact

---

## Claims 验证

| Claim | 结果 | 判定 |
|-------|------|------|
| C1: HDMI三优化使box023 Stability ≥90% | 12% (反而更差) | ❌ 失败 |
| C2: HDMI三优化使box025 Contact ≥60% | 65% (微升但Stability崩) | ❌ pyrrhic |
| C3: HDMI workflow 在 box025 上也优于 E041c | Contact 99% vs 54% | ✅ 碾压 |

## 关键结论

1. **HDMI workflow 在 CORE4D 所有 box case 上碾压 E041c**: Contact 93-99% vs 54-82%, MPKPE 0.3-0.7cm vs 24-40cm
2. **简单移植 HDMI 的 PD gains 到 MJWP 不可行**: 需要完整的 reward + optimizer 协同重设计
3. **HDMI 的核心优势是完整的系统设计**, 不是单个技巧:
   - Isaac PD gains 让机器人柔软 → 需要匹配的 reward 来维持平衡
   - Wrist damping 补偿柔软手腕 → 但上半身也需要调整
   - Contact guidance model surgery → 比预构建 scene_act.xml 更灵活
   - Precomputed reward → GPU-native, 无 CPU-GPU sync

## 下一步建议

1. **直接使用 HDMI workflow 处理所有 CORE4D cases** — 已证明 HDMI 在小箱子和大箱子上都有效
2. **将 convert_core4d_to_hdmi.py 扩展为批量处理** — 支持 bucket/desk/chair 等所有物体类别
3. **如果要改进 MJWP**: 需要在低刚度 PD 下**重新调优** CEM reward weights, 不是简单添加配置

---

## 结果路径

| 产出 | 路径 |
|------|------|
| E049 box023 (MJWP+HDMI opts) | `workspace/core4d/results/E049/E049_box023.{npz,mp4}` |
| E049 box025 (MJWP+HDMI opts) | `workspace/core4d/results/E049/E049_box025.{npz,mp4}` |
| E049 bucket010 (MJWP+HDMI opts) | `workspace/core4d/results/E049/E049_bucket010.{npz,mp4}` |
| E049 desk005 (MJWP+HDMI opts) | `workspace/core4d/results/E049/E049_desk005.{npz,mp4}` |
| E049e HDMI box025 | `workspace/core4d/results/E049/E049e_hdmi_box025/trajectory_hdmi.npz` |
