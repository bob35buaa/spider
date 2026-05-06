# E001: 数据管线验证

## Context
将 holosoma 运动学重定向的 CORE4D Box025 数据转换为 SPIDER 格式，验证端到端数据管线。

## Claims
1. holosoma 重定向数据 (qpos(T,43), fps=30) 可无损转换为 SPIDER trajectory_kinematic.npz 格式
2. 生成的场景 XML (G1 + Box025) 在 MuJoCo 中加载正确

## 改动
1. 新建 `spider/process_datasets/core4d.py` — 数据转换脚本
2. 新建 `examples/config/override/core4d_box025.yaml` — SPIDER 运行配置
3. 新建 `examples/config/override/core4d_box025_act.yaml` — 带 contact guidance 配置
4. 生成 `example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene.xml`
5. 生成 `example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene_act.xml`

## 成功标准
- MuJoCo 播放与 holosoma 可视化一致
- scene.xml: nq=43, nv=41, nu=29
- scene_act.xml: nq=42, nv=41, nu=35
- trajectory_kinematic.npz 包含 qpos, qvel, ctrl, contact, contact_pos

## 命令
```bash
uv run python spider/process_datasets/core4d.py \
    --source-npz /home/ubuntu/Workspace/holosoma/workspace/v2/results/retarget_replace_batch_trimmed/20231011-048-person1-Box025_with_obj_original.npz \
    --task box025_person1 --data-id 0 --no-show-viewer --save-video
```
