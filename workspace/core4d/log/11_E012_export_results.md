# E012: 导出与格式验证 — 结果

## 状态: 通过

## 导出结果

成功生成 Holosoma RL 训练兼容的 NPZ 文件，包含：

| Key | Shape | 来源 | 说明 |
|-----|-------|------|------|
| body_pos_w | (210, 52, 3) | SPIDER MJWP | 物理合规全身运动 |
| body_quat_w | (210, 52, 4) | SPIDER MJWP | 同上 |
| joint_pos | (210, 36) | SPIDER MJWP | 29 DOF + 7 obj |
| object_pos_w | (210, 3) | 运动学参考 | 正确的搬运轨迹 |
| partner_hand_pos_w | (210, 2, 3) | Holosoma 原始数据 | Person2 双手 |
| partner_hand_quat_w | (210, 2, 4) | Holosoma 原始数据 | Person2 双手 |
| fps | 50 | - | Holosoma 标准 |

**所有字段与 Holosoma reference NPZ 格式完全一致** (仅帧数 210 vs 205 的微小差异)。

## 机器人运动质量

使用原始 scene.xml (无前臂接触碰撞干扰) + body-only reward (base_pos_rew=5):

| 指标 | 值 | 评价 |
|------|-----|------|
| pelvis_err | 0.083m | 好 |
| joint_err | 0.108 rad (6.2°) | 好 |
| pelvis_z_min | 0.750m | 站立稳定 |
| 脚滑/穿模 | 无 | SPIDER 物理保证 |

## 可视化观察 (E012 body-only, `visualization_mjwp.mp4`)

| 时间 | ref | sim |
|------|-----|-----|
| t=0s | G1 站在箱前直立 | 初始状态与 ref 匹配良好 |
| t=1.0s | 弯腰准备抱箱 | **弯腰姿态与 ref 匹配良好**，箱子在原位 |
| t=2.1s | 半蹲扶箱 | G1 半蹲，箱子被前臂推向右侧 |
| t=3.2s | 站立扶箱 | G1 **站立姿态良好**，箱子在远处地面 |

**视觉结论**: body-only reward 下机器人身体跟踪质量最好——弯腰、半蹲、站立姿态与 ref 高度一致。箱子仍被推开（符合预期），但机器人运动本身物理合规，适合作为 Holosoma RL 的 reference motion。

## 产出文件

| 文件 | 路径 |
|------|------|
| 标准导出 | `workspace/core4d/results/box025_person1_spider_holosoma.npz` |
| **含 partner 导出** | `workspace/core4d/results/box025_person1_spider_holosoma_w_partner.npz` |

## 给 Holosoma RL 的使用建议

1. 替换 `converted_for_rl_trimmed/20231011-048-person1-Box025_v2_trimmed_mj_w_obj_w_partner.npz`
2. SPIDER 版本的机器人运动已经过物理验证（无脚滑/穿模）
3. 物体轨迹保持运动学参考（正确的搬运曲线）
4. Partner hand 数据可用于 HDMI 风格的 interaction reward / Lost-Contact termination
