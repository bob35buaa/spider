# 实验进度日志

## 会话：2026-04-25

### 当前实验
- **Run ID**: R002
- **阶段**: Evaluate (full run in progress)
- **开始时间**: 02:30

### 已执行操作
- 备份 R001 输出到 workspace/hdmi_reproduce/results/R001/
- 从 HuggingFace 恢复被覆盖的 data_id=1 数据 (232KB kinematic)
- 修复 get_reference: 按 MuJoCo 实际 qpos 布局构建参考轨迹
- 修复 setup_env: 用 MuJoCo joint address 映射初始化 sim 状态
- 修复 _diff_qpos: 适配 [obj(7), pelvis(7), joints(29)] 布局
- 修改 output_dir 到 workspace/hdmi_reproduce/results/R002
- process_config 添加 hdmi 分支跳过 model_path 读取
- 短测试 (20步) 验证: pelvis Z 稳定在 0.776-0.830m, opt_steps=1-2（快速收敛）
- **Claim 1 验证通过**: kinematic npz 与 HF 完全一致 (diff = 0)
- 完整轨迹运行中...

### 关键发现
- MuJoCo qpos 布局: [suitcase_root(7), pelvis_root(7), joints(29)]
- HF 数据用逻辑顺序: [pelvis(7), joints(29), object(7)]
- 两者值完全一致，只是排列不同
- 之前机器人倒地的根因：qpos 排列错误导致 sim/ref 不对齐

### 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| joints 顺序错乱 | 1 | 用 mujoco.mj_name2id 逐个映射 motion joint → MuJoCo qpos address |
| object 在 qpos 前面不是后面 | 1 | 查 MuJoCo joint list, suitcase_root at qposadr=0 |
| _diff_qpos 假设 object 在末尾 | 1 | 改用 pelvis_qpos_adr/obj_qpos_adr 参数化 |
