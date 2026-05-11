# E048 Results — 碰撞盒修复后的 E041c Baseline

## 配置

所有 baseline 实验使用 `+override=core4d_e041c` (与 E041c 完全相同的配置),
唯一区别是碰撞盒已通过 `fix_collision_boxes.py` 修复.

## 文件对应关系

| E048 文件 | 等价于 | 说明 |
|-----------|--------|------|
| E048_box025_baseline.{mp4,npz} | E041c box025 (碰撞盒修复) | 碰撞盒增大24% |
| E048_bucket010_baseline.{mp4,npz} | E041c bucket010 (碰撞盒修复) | Y/Z互换修正 |
| E048_desk005_baseline.{mp4,npz} | E041c desk005 (碰撞盒修复) | 新case,无旧E041c对比 |
| E048_box023.{mp4,npz} | E041c box023 (碰撞盒修复) | 新case,无旧E041c对比 |
| E048_box001.{mp4,npz} | E041c box001 (碰撞盒修复) | 新case |
| E048_box024.{mp4,npz} | E041c box024 (碰撞盒修复) | 新case |
| E048a_hdmi_comparison.mp4 | HDMI workflow box023 | HDMI vs E041c 对比 |

## 在 E041 目录下的 symlinks

E041 目录下创建了 `E041c_*_collision_fixed.{mp4,npz}` symlinks 指向对应 E048 文件,
方便在 E041 上下文中查找碰撞盒修复后的结果.
