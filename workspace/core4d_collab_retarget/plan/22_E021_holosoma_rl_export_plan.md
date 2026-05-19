# E021 计划：导出 E018b 轨迹给 Holosoma RL（task_afterE018 §4）

> 上游：`task_afterE018.md` §4 — 把 E018b 结果导成 holosoma RL 训练能直接吃的格式。
> 参考：`/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/workspace/pipeline/convert_data_format_mj_p3_for_rl.py`（99 行 batch driver）+ **真正的转换器** `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj.py`（620 行）。

---

## 0. 关键事实（已逐条核对）

1. **Joint 顺序完全一致**：spider `g1_mocap_29dof.xml` 的 29 dof 与 holosoma `_ROBOT_JOINT_NAMES_DEFAULT["g1"]` 一一对应；`dof_index_list` 是 identity，**无须重排**。
2. **Quat 约定一致**：双方均 MuJoCo wxyz。
3. **坐标系一致**：双方 base / object 均 world-frame freejoint。
4. **单位一致**：SI（m, m/s, rad, rad/s）。
5. **采样率差**：spider 60Hz（`sim_dt=0.0166667`），RL 期望 50Hz；**holosoma converter 内置 lerp+slerp 重采样**，spider 只需写正确 `fps` 字段。
6. **E018b 13 NPZ 已就绪**（`results/E018b/E018b_*_canonical_t02.npz`），可直接跑 13 case 转换。

---

## 1. 目标

A. **一个 ~30 行 shim** `scripts/export/spider_to_rl_shim.py`：读 spider `trajectory_mjwp.npz`，重写为 holosoma `convert_data_format_mj.py` 期望的输入（只需 `qpos` + `fps` 两字段）。
B. **一个批量 driver** `scripts/export/export_E018b_to_rl.py`：扫 13 case，调用 holosoma 转换器（subprocess），输出到 `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/workspace/pipeline/results/spider_E018b_for_rl/`。
C. **一份 manifest** `scripts/export/manifest_rl.tsv`：variant → object_name 映射，可 git 跟踪用于复现。
D. **首批端到端验证**：在 `box025_p2`（唯一 paper_generalization_pass）+ `box023_p2`（GT gate pass）跑通转换 → 在 holosoma RL gym 加载验证。

---

## 2. Spider NPZ → Holosoma converter 输入字段映射

| Holosoma 期望（`convert_data_format_mj.py:136-177`） | Spider 字段 | 转换 |
|---|---|---|
| `qpos[:, 0:3]` base pos | `qpos[:, 0, 0:3]` (取 env 0) | flatten |
| `qpos[:, 3:7]` base quat wxyz | `qpos[:, 0, 3:7]` | flatten |
| `qpos[:, 7:36]` 29 dof | `qpos[:, 0, 7:36]` | flatten |
| `qpos[:, -7:-4]` object pos | `qpos[:, 0, 36:39]` | flatten |
| `qpos[:, -4:]` object quat wxyz | `qpos[:, 0, 39:43]` | flatten |
| `fps` 标量 | 写 `1.0/0.0166667 ≈ 60.0` | 新增 |

其他 spider 字段 (`qvel`, `ctrl`, `support_proxy_*`, `ref`) 全部**不输出**，因 holosoma converter 自己用 `torch.gradient` 算速度。

⚠️ **若 spider qpos 是 (T, 43) 而非 (T, 2, 43)**（取决于 num_parallel）：直接用，无需 flatten。Shim 自动检测维度。

---

## 3. Holosoma converter 输出（RL 训练直接消费）

`convert_data_format_mj.py:444-457`：

- `fps: [50]`
- `joint_pos[T, 36]`、`joint_vel[T, 35]` — robot freejoint(7/6) + 29 dof
- `body_pos_w[T, nbody, 3]`、`body_quat_w[T, nbody, 4]`（wxyz）
- `body_lin_vel_w[T, nbody, 3]`、`body_ang_vel_w[T, nbody, 3]`
- `object_pos_w[T, 3]`、`object_quat_w[T, 4]`、`object_lin_vel_w[T, 3]`、`object_ang_vel_w[T, 3]`
- `joint_names: list[str]`、`body_names: list[str]`

---

## 4. 目录结构

```
spider/workspace/core4d_collab_retarget/
├── plan/22_E021_holosoma_rl_export_plan.md            # 本文件
├── log/21_E021_holosoma_rl_export_results.md          # 实施时新建
├── scripts/export/
│   ├── spider_to_rl_shim.py            # 单 case shim
│   ├── export_E018b_to_rl.py           # batch driver
│   ├── manifest_rl.tsv                 # variant → object_name 映射
│   └── verify_rl_load.py               # 在 holosoma 端加载验证
└── results/E021_rl_export_manifest/    # 只保留 manifest + log，不落地大数据
    ├── conversion_log.csv               # 每 case 转换状态
    └── failure_notes.md

holosoma/workspace/pipeline/results/spider_E018b_for_rl/   # 真实落盘位置
├── E018b_box025_p2_canonical_t02.npz   # holosoma RL 格式
├── E018b_box023_p2_canonical_t02.npz
└── ...
```

为什么 RL 输出放 holosoma 侧：(1) RL 训练在 holosoma 侧消费，路径就近；(2) 转换依赖 holosoma MJCF 模型与 Python 环境；(3) spider 侧只保留 driver + manifest 用于复现。

---

## 5. Variant → Object 映射（manifest_rl.tsv）

参考 `holosoma/workspace/v1/scripts/convert_core4d_to_omniretarget.py` 中 `TAG_TO_OBJECT` 映射 + holosoma `models/g1/g1_29dof_w_{obj}.xml` 实际命名（需现场核对大小写）。

| Spider variant | Object name (holosoma) | 验证 |
|---|---|---|
| `E018b_box021_p1_canonical_t02` | `Box021` 或 `box021` | 核 `models/` 目录 |
| `E018b_box023_p1/p2_canonical_t02` | `Box023` | 同 |
| `E018b_box025_p1/p2_canonical_t02` | `Box025` | 同 |
| `E018b_bucket001_p1/p2_canonical_t02` | `Bucket001` | 同 |
| `E018b_bucket005_s2_p1/p2_canonical_t02` | `Bucket005` | 同 |
| `E018b_bucket007_p1/p2_canonical_t02` | `Bucket007` | 同 |
| `E018b_desk021_p1_canonical_t02` | `Desk021` | 同 |

⚠️ **先验证 holosoma `models/{obj_name}/` 7 个物体齐全**，缺则需从 spider `spider/assets/` 同步资产或先做 OmniRetarget 风格的 mesh 导入。

---

## 6. 实施步骤

| Step | 内容 | Verify |
|---|---|---|
| 1 | 写 `spider_to_rl_shim.py`：自动检测 (T,43) vs (T,2,43)；输出 tmp npz with qpos+fps | 单测：`results/E018/E018_box023_p2_canonical_t02.npz` → tmp npz，加载后 shape 正确 |
| 2 | 核对 holosoma 侧 `models/{Box021,Box023,Box025,Bucket001,Bucket005,Bucket007,Desk021}/` 全部存在 | `ls` 输出 7 个目录 |
| 3 | 用 `box025_p2` 跑端到端：`shim → convert_data_format_mj.py → 加载验证` | 输出 `joint_pos/joint_vel/body_pos_w/object_pos_w` 全字段 + shape 正确 |
| 4 | `box023_p2` 第二个 case 验证 | 同上 |
| 5 | 写 `manifest_rl.tsv`（13 行） | 文件存在 |
| 6 | 写 `export_E018b_to_rl.py` batch driver（参考 `holosoma/workspace/v1/scripts/batch_convert_p3_for_rl.py`），跳过 manifest 中 `recommended_for_rl=false` 的 case | 输出 `results/E018b_rl_export_manifest/conversion_log.csv` 含每 case 状态 |
| 7 | 写 `verify_rl_load.py`：从 holosoma 侧加载产物 + render 一帧 | 13/13 加载成功 |
| 8 | 写 `log/21_*.md` 总结 | 跨 13 case 转换成功率、首批 RL 验证状态 |

---

## 7. 首批验证 case 推荐（按 E018b log）

| 排序 | Case | 选用理由 |
|---|---|---|
| **1（必做）** | `box025_p2` | 唯一 paper_generalization_pass：contact 86.9%、deep pen 0%、leg interf 0%、Erot 1.9° — 端到端最干净 |
| **2** | `box023_p2` | GT gate pass，object 跟踪好；contact 28.6% 偏低但视频稳定，可暴露 "object 完美但 contact 中等" 在 RL 端可学性 |
| **3（可选）** | `desk021_p1` | 不同物体类别（desk 而非 box），Epos 0.048/Erot 1.7°，验证物体资产管线覆盖 |

**禁选（先跳过）**：
- 4 个 `robot_fall_visual_fail`（box021_p1/p2、bucket001_p1/p2），pelvis < 0.45m
- 3 个 `artifact_failed` / `push_or_leg_shortcut`（bucket005_s2_p1/p2、bucket007_p1）

manifest_rl.tsv 中给 13 行各加 `recommended_for_rl ∈ {yes, maybe, no}` 列，配 reason。

---

## 8. 成功标准

- ✅ 7 个物体在 holosoma `models/` 全部存在（或已补齐）
- ✅ 至少 6/13 case 转换成功（pass + GT case + desk）
- ✅ holosoma RL gym 能加载首批 2-3 case 并 render
- ✅ `log/21_*.md` 给出 13 case 转换状态 + RL 验证报告

## 9. 已知风险

- **物体资产缺失**：若 holosoma `models/` 没有 bucket005/007/desk021，需要先在 holosoma 侧补 MJCF（超出本计划范围）。
- **double-env npz**：spider 多 env CEM 输出 (T, 2, 43)，shim 取 env 0；若用户希望保留两 env 各自轨迹，需扩展 shim。
- **G1 模型版本差异**：spider 用 `g1_mocap_29dof.xml`，holosoma 用 `g1_29dof.xml`，确认两边 actuator / joint 完全一致后再跑（已逐条核对一致）。

## 10. 时间预算

| Phase | 内容 | 时间 |
|---|---|---|
| A | Shim + 物体资产核对 + box025_p2 端到端 | 0.5 day |
| B | Batch driver + manifest + box023_p2 验证 | 0.5 day |
| C | 13 case 全跑 + RL load 验证 + 报告 | 1 day（依赖 E018b 数据到位） |
| **合计** | | **2 day** |
