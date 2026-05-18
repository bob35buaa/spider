---
name: experiment-brainstorm-zh
description: 面向 RL 物理重定向研究的头脑风暴系统。分析已有实验结果，生成可证伪假设，按影响×可行性排优先级，写入 HYPOTHESIS_BACKLOG.md，然后移交 experiment-planning-zh 执行。触发词：头脑风暴、下一步方向、探索实验、研究方向、下一步做什么、实验优先级、假设生成、方向分析、新方向
user-invocable: true
allowed-tools: "Read Write Edit Bash Glob Grep"
hooks:
  UserPromptSubmit:
    - hooks:
        - type: command
          command: "EXP_WS=\"${EXPERIMENT_WORKSPACE:-workspace/core4d_collab_retarget}\"; if [ -f \"$EXP_WS/ideas/HYPOTHESIS_BACKLOG.md\" ]; then echo '[brainstorm] 检测到假设库，加载前 40 行入上下文：'; head -40 \"$EXP_WS/ideas/HYPOTHESIS_BACKLOG.md\"; fi"
  Stop:
    - hooks:
        - type: command
          command: "EXP_WS=\"${EXPERIMENT_WORKSPACE:-workspace/core4d_collab_retarget}\"; if [ -d \"$EXP_WS/ideas\" ]; then echo '[brainstorm] Session 结束。如果本次产生了新假设或决策，请确认已写入 ideas/HYPOTHESIS_BACKLOG.md 并保存了 ideas/brainstorm_$(date +%Y-%m-%d).md 快照。'; fi"
metadata:
  version: "1.0.0"

---

# 实验头脑风暴系统

面向 RL 物理重定向研究的**假设生成与决策系统**。解决「下一步试什么」的问题，然后移交给 `experiment-planning-zh` 解决「怎么执行」的问题。

---

## 第一步：始终先加载上下文

**在生成任何假设之前**，必须先理解当前状态。根据是否已有假设库，读取范围不同：

### 首次运行（`ideas/HYPOTHESIS_BACKLOG.md` 不存在）

从零建立认知，读取所有实验记录（log 已包含结论，不需要读 plan）：

```bash
EXP_WS="${EXPERIMENT_WORKSPACE:-workspace/{exp_name}}"
cat "$EXP_WS/EXPERIMENT_TRACKER.md"
# 读取全部 log 文件
ls -1 "$EXP_WS/log/"*.md | sort | xargs -I{} sh -c 'echo "=== {} ===" && cat "{}"'
cat "$EXP_WS/progress.md" 2>/dev/null
```

### 后续运行（`ideas/HYPOTHESIS_BACKLOG.md` 已存在）

BACKLOG 已积累历史知识，只需读取增量：

```bash
EXP_WS="${EXPERIMENT_WORKSPACE:-workspace/{exp_name}}"
cat "$EXP_WS/ideas/HYPOTHESIS_BACKLOG.md"   # 已有假设库作为基础
cat "$EXP_WS/EXPERIMENT_TRACKER.md"          # 最新实验状态
ls -1 "$EXP_WS/log/"*.md | sort | tail -3 | xargs -I{} sh -c 'echo "=== {} ===" && cat "{}"'
cat "$EXP_WS/progress.md" 2>/dev/null
```

读完后，**用 3-5 条子弹概括**：
- 已知什么（验证过的结论）
- 什么失败了（及失败模式）
- 什么还是开放问题（尚未回答的关键问题）

---

## 核心工作流（8 步）

### Step 1：框定研究问题

用 ≤2 句话明确说出本次头脑风暴要回答的**核心研究问题**。例如：

> "真 freejoint 物体在当前 CEM reward 下无法被搬运，根本原因是物理参数不可行还是 reward/optimizer 本身不足？"

不允许跳过这一步直接生成假设。

---

### Step 2：生成候选假设（≥3 个）

每个假设必须满足以下格式：

```
假设 Hxxx：[具体声明]
- 类别：Physics / Reward / Optimizer / Data / Algorithm
- 可证伪标准：[具体数字阈值，如 "E003 sweep 中 obj_mean < 0.3m 则假设支持"]
- 最小验证实验：[估计算力，如 "2 cases × 500 iter ≈ 10min × 2GPU"]
- 来源：[基于哪个已有实验/论文/失败分析]
```

**不允许写"我们可以试试 X"** — 必须说清楚什么结果意味着 X 成立，什么结果意味着 X 失败。

---

### Step 3：领域框架驱动生成

使用以下框架触发假设生成（每次头脑风暴至少应用 2 个）：

#### 框架一：失败分解
把当前失败归类到哪一层：
- **Physics**：物理参数（质量/摩擦/碰撞几何）使任务本身不可行
- **Reward**：目标函数没有正确引导机器人行为
- **Optimizer**：CEM/采样策略不足以搜索到正确解
- **Data**：参考轨迹质量/格式有问题
- **Algorithm**：控制架构/模型本身有设计缺陷

每一层都生成一个候选假设，然后设计能区分它们的实验。

#### 框架二：消融思维
当前架构中哪个组件**验证最少**？
- 列出核心组件列表
- 标记哪些已经有专门实验验证，哪些没有
- 优先针对「未验证的核心组件」生成假设

#### 框架三：文献缺口
对比 SPIDER / DynaRetarget / 相关论文的方法，什么是论文里有但还没在本项目实现的？
- 不要假设「论文里有的我们都有」，要逐条核对
- 特别关注：接触处理、双人交互力建模、freejoint 物体控制方案

#### 框架四：可行性阶梯
能证伪或支持假设的**最小代价实验**是什么？
- 1-2 个 case，而不是全量 sweep
- 短 run（100-200 iter），而不是完整训练
- 单一变量，而不是多因素同时改变

#### 框架五：双假设检验
若两个竞争性解释都能解释当前结果：
- 明确写出两个假设各自的预测
- 设计一个实验，让两个假设做出**不同预测**
- 这种实验的信息价值最高

---

### Step 4：优先级排序

对每个假设打分，输出优先级表：

| # | 假设简述 | 类别 | 预期影响(1-3) | 可行性(1-3) | 综合(乘积) | 最小实验代价 |
|---|---------|------|-------------|-----------|----------|------------|
| H001 | ... | Physics | 3 | 3 | 9 | 2 cases, 500 iter |
| H002 | ... | Reward | 2 | 2 | 4 | 4 cases, 1000 iter |

**打分说明**：
- 预期影响：1 = 小改进/边际探索，2 = 中等改进，3 = 可能突破性进展或解决核心瓶颈
- 可行性：1 = 需要大量新代码/外部依赖，2 = 中等工作量，3 = 已有基础设施可直接复用

---

### Step 5：逐一提问（硬规则）

如果需要用户输入来确定方向：
- **每条消息只问一个问题**
- 优先多选题形式（列出 2-4 个选项）
- 不要把「选哪个假设」和「还有没有其他想法」放在同一条消息里

---

### Step 6：确认假设（硬门控）

**在用户明确表示同意之前，不得进入 Step 7**。

确认内容：
1. 选定了哪个假设（Hxxx）
2. 该假设的可证伪标准是什么
3. 最小验证实验是什么

---

### Step 7：自审（写入 BACKLOG 前）

在写入 `HYPOTHESIS_BACKLOG.md` 之前，检查：

- [ ] 假设是否有**具体数字阈值**可证伪？（不是「更好」，而是「obj_mean < X m」）
- [ ] 是否有**最小可行实验**（不是全量 sweep）？
- [ ] 是否存在**更廉价的预筛**（先 smoke，再 full）？
- [ ] 是否记录了被**否决的假设**及原因？（同样重要！）

---

### Step 8：写入 BACKLOG 并移交

1. 将选定假设写入 `workspace/{exp_name}/ideas/HYPOTHESIS_BACKLOG.md`
2. 将本次完整 session（包括所有候选假设）写入 `workspace/{exp_name}/ideas/brainstorm_YYYY-MM-DD.md`
3. 输出 `init-experiment.sh` 调用命令，明确告诉用户下一步运行 `experiment-planning-zh`

```bash
# 移交命令示例
bash workspace/{exp_name}/scripts/init-experiment.sh v{X.X} {topic}
# 然后在 experiment-planning-zh 中把 H{XXX} 的假设写入 plan 的 Context 节
```

---

## 过程规则

| 规则 | 说明 |
|------|------|
| 先生成 ≥3 个，再收敛 | 不允许单刀直入推一个方向 |
| 区分诊断实验 vs 系统实验 | 诊断实验建立知识（可能不提升指标），系统实验目标是超越 baseline |
| 廉价消融优先 | 全量 sweep 之前必须有诊断 run |
| 记录被否决的理由 | 这和选定方向同样有价值，防止未来重复讨论 |
| 假设写入 BACKLOG 前必须有数字阈值 | 「更好」不是可证伪标准 |
| 绝不跳过假设直接写代码 | 连接 experiment-planning-zh 才能进入执行 |

---

## 假设类别说明

| 类别 | 说明 | 典型诊断方法 |
|------|------|------------|
| **Physics** | 物理参数或几何约束使任务本身不可行 | 参数 sweep，检查物理可行性上界 |
| **Reward** | 目标函数设计不正确，机器人优化到了错误行为 | 固定动作检查 reward 值域，消融各项 reward |
| **Optimizer** | CEM/采样策略探索不足或过早收敛 | 调整 sigma/温度/iter 数量，看是否突破瓶颈 |
| **Data** | 参考轨迹有缺陷（格式错误、噪声、无法执行） | 可视化参考轨迹，检查物理合理性 |
| **Algorithm** | 控制架构本身有设计缺陷（如 freejoint 控制口径不匹配） | 代码审计，对照论文实现 |

---

## 工作区文件结构

```
workspace/{exp_name}/
└── ideas/
    ├── HYPOTHESIS_BACKLOG.md          # 持续维护的假设库（所有假设，含状态）
    └── brainstorm_YYYY-MM-DD.md       # 每次 session 快照（含被否决的假设）
```

---

## 模板

- [templates/hypothesis_backlog.md](templates/hypothesis_backlog.md) — 假设库初始化模板
- [templates/brainstorm_session.md](templates/brainstorm_session.md) — 单次 session 快照模板

---

## 反模式

| 不要这样做 | 应该这样做 |
|-----------|-----------|
| "我们试试改 reward 权重" | "假设：增加 contact reward 权重可让 obj_mean 从 0.7m 降至 0.4m 以下，可证伪标准：E00X 实验中 obj_mean ≥ 0.5m 则假设失败" |
| 生成一个假设就停下来 | 先生成 ≥3 个，排优先级，再选一个 |
| 跳过诊断直接 full sweep | 先 2-case smoke → 确认方向 → full sweep |
| 只记录选定方向 | 连被否决的假设也要写入 brainstorm session |
| 遇到物理瓶颈只想算法解 | 先用 Physics 框架做根因分析 |
| 多个问题同时问用户 | 每条消息只问一个问题 |
