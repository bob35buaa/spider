# V{X.X} 实验结果

**日期**: {YYYY-MM-DD}
**实验域 (exp_name)**: `{exp_name}`  <!-- 大实验方向标识，如 core4d、holosoma_hdmi。决定脚本命名、logs 子目录、NPZ 输出路径 -->
**对应Plan**: `workspace/{exp_name}/plan/{NN}_{version}_{topic}_plan.md`
**前置**: `workspace/{exp_name}/log/{prev_NN}_{prev_version}_{prev_topic}.md`

## 1. 背景

{简述实验目的，链接前置实验结论}

---

## 2. R{XXX} ({Version}): {实验名}

**Config**: `exp:{config-name}`
**变化**:
- {关键改动1}
- {关键改动2}

### 运行指令

> **脚本规则**：脚本保存在 `workspace/{exp_name}/scripts/`，确保可复现。若脚本不存在，**必须先创建再运行**。CORE4D 的 launch/pull 真实入口放 `scripts/launch/active/`，评估 shell 入口放 `scripts/eval/wrappers/`。

**训练**（脚本：`workspace/{exp_name}/scripts/train/train_{exp_name}_{version}.sh`）：
```bash
bash workspace/{exp_name}/scripts/train/train_{exp_name}_{version}.sh R{XXX} {GPU}
```

**评估**（脚本：`workspace/{exp_name}/scripts/eval/wrappers/eval_{exp_name}_{version}.sh`）：
```bash
bash workspace/{exp_name}/scripts/eval/wrappers/eval_{exp_name}_{version}.sh R{XXX} {GPU}
```

**验证指令**：
```bash
# 1. 检查训练进程是否正常启动
tail -f logs/{exp_name}/{run_dir}/train.log

# 2. 早停健康检查（~500 iters）：reward > {最低阈值}，ep_len > {最低阈值}
# 若不达标，立即停止并分析

# 3. 收敛检查（~2000 iters）：曲线趋势是否符合预期
# WandB 关注指标：Mean Reward | Mean Ep Length | {关键指标名}
```

### 结果路径

> **路径规则**：实验产出必须在此记录，验证结果写入同一路径下，禁止分散存放。

| 类型 | 本地路径 |
|------|---------|
| 训练模型 / checkpoint | `logs/{exp_name}/{run_dir}/` |
| 训练日志 / metrics | `logs/{exp_name}/{run_dir}/` |
| 评估 / 验证结果 | `logs/{exp_name}/{run_dir}/eval/` |
| 重定向结果（NPZ / pkl 等） | `workspace/{exp_name}/results/` |

```
logs/{exp_name}/
└── {run_dir}/          # 训练产出
    ├── checkpoint/     # 模型 ckpt
    ├── metrics.json    # 量化指标
    └── eval/           # 评估 & 验证结果
workspace/{exp_name}/
└── results/            # 重定向 NPZ / pkl 输出
```

### 可视化

> ⚠️ **强制要求**：只要具备 viewer 条件，**必须执行可视化并记录观察**。禁止仅凭数字指标判断实验成功。

**可视化命令**：
```bash
# 优先使用 rerun（远程可用）
bash workspace/{exp_name}/scripts/eval/wrappers/eval_{exp_name}_{version}.sh R{XXX} {GPU} --viewer rerun

# 本地环境备选
bash workspace/{exp_name}/scripts/eval/wrappers/eval_{exp_name}_{version}.sh R{XXX} {GPU} --viewer mujoco
# 或 --viewer viser
```

**视频结果分析**（若输出了视频文件）：
```
/video-frames logs/{exp_name}/{run_dir}/eval/{video_file}
```
提取的关键帧描述与 GUI 观察等效，写入下方「实际观察」中。

**观察清单**（逐项确认）：
- [ ] {关键动作1}（期望：{描述}）
- [ ] {关键动作2}（期望：{描述}）
- [ ] 接触是否稳定，无抖动/穿透
- [ ] 末端执行器轨迹是否连贯

**实际观察**：

{描述 GUI/Rerun 中看到的具体行为，**必须填写**，不得留空}

---

### 训练指标 (@{N} iters)

| 指标 | R{prev} ({prev_version}) | **R{XXX} ({version})** | 变化 |
|------|--------------------------|------------------------|------|
| Mean Reward | | | |
| Mean Ep Length | | | |
| Body Pos Error (m) | | | |
| Body Rot Error (rad) | | | |
| Proximity | | | |
| {其他关键指标} | | | |

### 训练曲线

| Iter | Reward | Ep Len | {关键指标} |
|------|--------|--------|-----------|
| 1000 | | | |
| 2000 | | | |
| 3000 | | | |
| 4000 | | | |

### 分析

{指标分析 + 根因诊断}

```
Logs: logs/{exp_name}/
├── {run_dir}/  # R{XXX}
```

---

## N. Claims 验证

| Claim | 结果 |
|-------|------|
| {claim1} | **通过/未通过** — {简述} |

## N+1. Git 提交

> **规则**：Claims 全部通过后，**必须**提交到 git。未通过则继续迭代，不提交。

**当前分支**: `{git-branch}`（对应大方向 `{exp_name}`）

```bash
git add {changed_files}
git commit -m "exp({exp_name}): R{XXX} {实验名} — {一句话结论}

- {关键改动1}
- {关键改动2}
- Result: reward={X.XX}, {其他关键指标}
- Log: logs/{exp_name}/{run_dir}/"

git push origin {git-branch}
```

> **方向偏离检查**：如果下一步计划与当前 `exp_name` 大方向严重偏离（更换数据集/任务域、实验思路根本性改变、当前方向已完结），**不得自行新建分支**，必须先与用户沟通，确认后再开新分支、重新规划大实验方向。

## N+2. 下一步

{基于本次结果的根因分析，指向下一个实验方向}
