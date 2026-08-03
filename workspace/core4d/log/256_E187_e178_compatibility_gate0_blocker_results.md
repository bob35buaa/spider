# E187 E178 compatibility Gate S0 阻塞结果

日期：2026-08-02
实验：E187
阶段：S0（authority freeze + E178 backward compatibility）
结论：静态隔离通过；本地 RTX5090 formal replay 的 exact 与 semantic 门失败；Ada/Full 启动数为 0

## Context

E187 计划只修改 canonical distance reward `R`，继续冻结 E186 的 physics `P`、
candidate gate `G`、keep22 与三个 object-specific CoACD collider。计划要求任何 E187
reward shadow、canary 或 Full 前，必须先证明 E178 legacy 路径可复现。

本轮按预注册协议选择三条代表 case，其中历史上由本地 RTX5090 执行的
`bucket007_20231020_055_p1`作为 same-device golden。正式命令保持原 E178 override、
scene、trajectory、contact mask、`1024 samples × 32 iterations × seed0`，结果只写入
E187 独立 root。

## Authority 与静态合同

| 检查 | 结果 |
|---|---:|
| E178 Full manifest SHA | `de9a3d165301318049da208aad52b1f5bc4d3ed671631f0097958734a0f022a8`，PASS |
| Full27 / keep22 / drop5 | 27 / 22 / 5，顺序与投影 PASS |
| E178 artifact inventory | `27×10+6=276` 行，PASS |
| 三代表 scene snapshot | actual E178 scene SHA逐项一致 |
| authority tests | 3/3 PASS |
| legacy reward regression | 5/5 PASS（含旧score逐值exact） |
| E186 production grid backend | PASS |
| E182 query-tape default-off | 5/5 PASS |
| compat runner static tests | 2/2 PASS |
| E178 source isolation final recheck | PASS，历史 manifest/artifact SHA未变 |

static wrapper 的末行 `E187_GATE_S0_REPLAY=PENDING`只表示该入口不评估动态 replay；
不能覆盖下述 direct-core evaluator 已给出的 formal FAIL。

## Formal same-device replay

### 运行

| 项目 | 值 |
|---|---|
| Case | `bucket007_20231020_055_p1` |
| Device | local GPU0, NVIDIA GeForce RTX 5090 |
| Budget | `1024×32`, seed0 |
| Runtime | 166/166 sim steps，887.45 s |
| Output | finite，root/outdir NPZ均约5.7 MiB |
| Final object error log | pos `0.1441`，quat `0.1078` |
| Status | `run_complete_pending_eval` |

### Gate 结果

| Gate | 结果 | 证据 |
|---|---:|---|
| effective config whitelist | PASS | old/new keys 406/423；只新增17个default-only字段；旧字段只变output/video path |
| 12门 decision match | PASS | 历史与replay均为numeric FAIL |
| same-device exact `≤1e-5` | **FAIL** | 456个公共numeric/bool arrays中236项失败 |
| cross-run semantic tolerance | **FAIL** | 8项仅2项通过 |
| compat row | **FAIL** | exact与semantic均未闭合 |

关键全轨迹差异：

| Array | max abs |
|---|---:|
| qpos | 0.237400 |
| ctrl | 0.347998 |
| qvel | 9.252299 |
| reward mean | 2.019397 |
| CEM gate valid fraction | 0.885742 |

历史 NPZ 没有 direct selected-index array，因此不能伪称对历史 selected index 做了直接
逐值验证；当前比较覆盖 qpos/ctrl 及所有公共同shape numeric/bool arrays。

### Semantic tolerance

| Metric | historical | replay | abs delta | tolerance | 结果 |
|---|---:|---:|---:|---:|---:|
| root pos mean (cm) | 25.0620 | 25.0111 | 0.0509 | 0.5 | PASS |
| EEF pos mean (cm) | 20.1006 | 20.7148 | 0.6143 | 0.5 | FAIL |
| object pos mean (cm) | 15.0717 | 15.0181 | 0.0536 | 0.5 | PASS |
| root ori mean (deg) | 19.4878 | 22.3000 | 2.8122 | 0.5 | FAIL |
| EEF ori mean (deg) | 26.4888 | 29.1699 | 2.6810 | 0.5 | FAIL |
| object ori mean (deg) | 7.1450 | 6.3406 | 0.8044 | 0.5 | FAIL |
| in-mask contact fraction | 0.6812 | 0.6522 | 0.0290 | 0.02 | FAIL |
| 3mm penetration fraction | 0.2289 | 0.1928 | 0.0361 | 0.01 | FAIL |

## 分叉诊断

为区分 legacy 实现回归与 MJWarp 运行间非确定性，额外运行到首两个 formal CEM commit
的 `max_sim_steps=16` 短程诊断。所有诊断仍使用本地 RTX5090、同override、同seed0、
`1024×32`，且不覆盖 formal replay。

| 对照 | qpos | ctrl | qvel | reward mean | gate valid frac |
|---|---:|---:|---:|---:|---:|
| E178 source `64f9a33` vs current source | 2.91e-4 | 1.588e-3 | 1.033e-2 | 1.007e-3 | 0.00293 |
| historical Full prefix vs E178 source rerun | 1.365e-3 | 6.909e-3 | 4.011e-2 | 2.176e-3 | 0.02344 |
| current source repeat A vs B | 5.026e-4 | 3.087e-3 | 1.876e-2 | 1.208e-3 | 0.00977 |

三组均在前6个非CEM warmup records近似bit-close，第一次CEM后出现微小差异，第二次
CEM后放大。same-current A/B底噪已大于old/current源码差的多项幅度，因此不能把formal
失败单独归因为E182/E186/E187代码回归；但这也不能用来放宽预注册的`≤1e-5`门。

另发现E182以后 recorder-off 普通NPZ新增`cem_selected_index0`字段。真正query-tape
chunk仍为default-off且没有生成，但普通输出schema已经变化。该字段与MJWarp
非确定性应在后续独立方法学实验中审计，不能在E187失败后现场改门。

## 可视化

使用离线 MuJoCo renderer 生成E187 kinematic-vs-physics MP4，并通过`video-frames`在
0.20/1.66/3.10 s抽取E178/E187固定帧。E187视频为1440×480、83帧、25fps/3.32s，
历史E178为1440×480、166帧、50fps/3.32s。

实际观察：0.20s两版physics均为机器人站在直立桶侧；1.66s躯干朝向、手臂和跨步姿态
已有肉眼差别；3.10s E178机器人俯身/跨靠桶体且仍居中，E187桶明显倾斜、机器人与
肢体大幅移出固定视野。静帧遮挡不足以独立判定穿透，但明确支持轨迹非等价。renderer
对43-DOF kinematic输入采用截断并警告，因此左侧kinematic只作上下文，不用于量化。

## Claims

| Claim | 状态 | 证据 |
|---|---|---|
| C0 authority | PASS | manifest/keep/drop/collider/protocol与inventory闭合 |
| C1 E178 compatibility | **FAIL** | formal same-device exact与semantic均失败 |
| C2 reward definition | PARTIAL | pure-function 5/5 PASS；S1 formal tape未获准执行 |
| C3–C12 | NOT STARTED | Gate S0失败后按计划停止 |
| C13 isolation | PASS | E178 SHA不变；Ada/A100/Full均0；未kill/暂停/抢占既有进程 |

## 结果路径

| 内容 | 路径 |
|---|---|
| authority / execution manifest | `workspace/core4d/results/E187/s0_environment/e178_compat/` |
| formal replay | `workspace/core4d/results/E187/s0_environment/e178_compat/cem/` |
| direct-core eval | `workspace/core4d/results/E187/s0_environment/e178_compat/eval/` |
| source bisect | `workspace/core4d/results/E187/s0_environment/e178_compat/bisect_v1/` |
| same-source repeat | `workspace/core4d/results/E187/s0_environment/e178_compat/repeat_v1/` |
| visual video/frames | `workspace/core4d/results/E187/s0_environment/e178_compat/render/`、`eval/visual/` |
| runtime/static logs | `logs/E187/s0/e178_compat/` |

关键证据SHA：formal NPZ `055cf589...6999`；eval summary `c9b42d82...38d`；
source bisect `a533c5dc...8102`；same-source repeat `b2797ae2...4fcd`；visual MP4
`784152e6...c07d4`。

## 结论与下一步

E187 在第一个动态 stop/go gate 即停止：不得启动bucket003/004 Ada replay、E187 reward
shadow/canary或keep22 Full。远程RTX 6000 Ada两卡虽然可用，但本轮E187远程session和
worker均为0；A100也未使用。

如果继续研究，应新建独立实验审计MJWarp same-seed determinism、recorder-off输出schema
与可实现的golden口径，再预注册新的兼容门。不能用E187当前失败结果回调reward、grid、
collider或事后放宽`1e-5`/semantic阈值。
