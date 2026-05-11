# E051 Progress — 2026-05-11

## 当前状态: 运行 E051 box023 (euler + physics fix)

---

## E051a+b: Data Verification (完成)

### box023_person1:
- FK consistency: PERFECT (0.000 cm position, 0.000° rotation at frame 0)
- **BUG FOUND**: Scene XML body_pos [0.155, -0.124, 0.310] vs actual frame-0 pos [-0.826, -1.184, 0.140] = **145 cm offset!**
- Object is NOT near-identity rotation: quat [-0.007, -0.001, 0.701, 0.713] = **179.3° from identity**
- Euler convention mismatch: **180°** (not < 1° as E050 suggested!)
- Best euler convention: **XZY** (max middle angle 8.6°)

### box025_person1:
- FK consistency: PERFECT
- Scene XML body_pos offset: only 0.039 cm (correct)
- Euler convention mismatch: **120-140°**
- Best euler convention: **YXZ** (max middle angle 6.3°)

### Key Insight Correction
E050 log 的 "box023 euler diff < 1°" 是 middle angle (gimbal risk), 不是 convention mismatch.
实际 convention mismatch 对 box023 = 180°, 对 box025 = 120-140°.
**Euler convention bug 影响所有 CORE4D cases** (不仅仅是 box025)!

---

## 修复已应用

| 文件 | 修复 |
|------|------|
| box023 HDMI scene XML | body_pos → [-0.826, -1.183, 0.140], mass → 2.0 |
| box025 HDMI scene XML | mass → 2.0 |
| `spider/simulators/hdmi.py` L445 | `_make_contact_guidance_model(euler_convention)` 参数化 joint 顺序 |
| `spider/simulators/hdmi.py` L659 | `as_euler(euler_conv)` 使用 config convention |
| `spider/simulators/hdmi.py` L1303 | `as_euler(euler_conv)` 使用 config convention |
| `spider/config.py` | 添加 `euler_convention: str = "XYZ"` |
| `examples/config/hdmi.yaml` | 添加 `euler_convention: "XYZ"` |

---

## 运行中

### E051a (body_pos fix + euler fix + mass=2.0) — ❌ 失败!
- ObjPos: 131.8cm (worse than 24.4cm baseline!)
- ObjRot: 92.1° (worse than 66.4° baseline!)
- **根因**: body_pos 修正后 slide offsets ≈ 0 → PD 无恢复力 → 物体漂移
- HDMI 设计依赖 large slide offsets 提供隐式追踪力 (kp × offset)
- 教训: body_pos 偏移不是 bug, 是 feature!

### E051b (ONLY euler fix + mass=2.0, 保持原 body_pos) — 运行中
```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/run_hdmi.py \
    task=move_box023 +data_id=0 viewer=none save_video=true save_info=true \
    output_dir=workspace/core4d/results/E051/E051b_box023_euler_only \
    use_torch_compile=false euler_convention=XZY
```

### E051 box025 (euler fix YXZ + mass=2.0) — 远程运行中
```bash
# spider-remote GPU0, tmux session e051
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python examples/run_hdmi.py \
    task=move_box025 +data_id=0 viewer=none save_video=true save_info=true \
    output_dir=workspace/core4d/results/E051/E051_box025_euler_fix \
    use_torch_compile=false euler_convention=YXZ
```
