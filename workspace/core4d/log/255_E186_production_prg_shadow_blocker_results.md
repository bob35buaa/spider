# E186 production P/R/G shadow 与 Full 阻塞结果

日期：2026-08-02
实验：E186
阶段：S2–S3c
结论：P/runtime通过；G无false-safe；R在bucket003/007阻塞，Full启动数为0

## Context

E186冻结keep22与三个object-specific CoACD collider，以compound-convex做P、同源
canonical grid-SDF `D_C`做R/G、original-mesh `D_M`只做fidelity oracle。本文记录从
production backend、22-case physics，到44-tape、64×4 shadow及formal-budget消歧的结果。

## 冻结输入

- keep/drop：22/5，未重选case或collider；
- physics：bucket003 K=16，bucket004/007 K=8；
- grid v4：5 mm voxel，120 mm requested padding；
- formal Full budget：1024 samples × 32 iterations，seed0；
- Full资源仍冻结为local GPU0 + remote Ada GPU0/1，未启动任何Full worker。

## 结果

### P与production合同

| 项目 | 结果 |
|---|---:|
| v4 grid formal | 3/3 PASS |
| CPU MuJoCo compound scene | 22/22 PASS |
| MJWarp compound scene | 22/22 PASS |
| robot-object pair | bucket003 288；bucket004/007 144 |
| reference/E178-final finite | 44/44 |
| body false-safe | 0 |
| combined false-safe | 0 |

### 64×4 v7 shadow

| object | combined valid | geometry active | rho | status |
|---|---:|---:|---:|---|
| bucket003 | 0/64 | 0/64 | 1.0（trivial） | FAIL |
| bucket004 | 64/64 | 64/64 | 0.999817 | PASS |
| bucket007 | 0/64 | 0/64 | 1.0（trivial） | FAIL |

v7使用reward实际MJWarp transforms，bucket004 surface trace reproduction max为
`9.36e-6`，排除了此前CPU forward与reward transform错位造成的假失败。

### S3b formal-budget feasibility

| object | record step | combined valid | surface active | wall | peak total GPU memory |
|---|---:|---:|---:|---:|---:|
| bucket003 | 12 | 1024/1024 | 0/1024 | 30.38 s | 2329 MiB |
| bucket007 | 22 | 1024/1024 | 1024/1024 | 127.02 s | 2131 MiB |

两object均在正式预算恢复valid，证明64×4 combined-valid门对这两条轨迹是预算
false-negative，不是compound physics不可运行。

### S3c formal-budget exact fidelity

| object | active | grid/exact valid | false-safe | rho | top-k overlap | selected0 |
|---|---:|---:|---:|---:|---:|---|
| bucket003 | 0/1024 | 1024/1024 | 0 | 1.0（trivial） | 1.0 | match |
| bucket007 | 1024/1024 | 1024/1024 | 0 | 0.802378 | 0.578431 | mismatch |

bucket003的grid/exact min hand SDF分别为45.8/48.6 mm，远离`[-1,3] mm` surface
hard band。`D_C`与`D_M`一致认为没有reward support，因此不是grid漏检；真实C表面上的
R捕获域缺失。

bucket007虽然support与gate均健康，但5 mm grid误差相对1.5 mm sigma和4 mm hard band
过大：geometry delta p99=0.0810，导致total reward排序和elite集合显著变化。G安全门仍
无false-safe，但R fidelity未通过。

## Claims

| Claim | 状态 | 证据 |
|---|---|---|
| C0–C4 authority/C/P/physics | PASS | keep22、三collider、grid v4、22/22 CPU/MJWarp |
| C5 R audit | FAIL | bucket003无捕获域；bucket007 rho=0.802 |
| C6 G audit | PARTIAL PASS | false-safe=0；正式预算valid恢复，但R排序失败 |
| C7 canary | BLOCKED | 64×4是预算false-negative，尚未修订并执行recorder-off canary |
| C8 Full closure | NOT STARTED | Full worker=0 |

## 可视化

本阶段没有生成新MP4：S3b/S3c是bounded candidate-query审计，outer trajectory只有7–12
个control ticks，且正式命令预注册`save_video=false`以隔离recorder成本。CoACD mesh/
collision的3D与2D overlay已在E181 visual diagnostic中完成；本阶段新增结论来自候选级
reward/gate/exact数值，不能由短outer视频替代。后续R改动进入canary时必须恢复视频审查。

## 结果路径

| 内容 | 路径 |
|---|---|
| reference/final audit | `workspace/core4d/results/E186/s3_prg_audit/reference_final/` |
| 64×4 authority shadow | `workspace/core4d/results/E186/s3_prg_audit/shadow64x4_v7/` |
| formal-budget feasibility | `workspace/core4d/results/E186/s3_prg_audit/fullbudget_probe_v1/` |
| formal-budget fidelity | `workspace/core4d/results/E186/s3_prg_audit/fullbudget_fidelity_v1/` |

## 结论与下一步

不能按当前冻结R启动Full。可行路线只有两类：

1. 保留C与G，重做R：给bucket003加入基于`D_C`的宽域approach/continuation reward，再把
   近场收缩到narrow surface band；同时将bucket007的R改为更细grid或连续soft score，
   消除5 mm grid与1.5 mm sigma/hard boundary的不匹配。
2. 显式丢弃阻塞object/cases，只对已通过的子集继续；这会改变已冻结keep22与C8口径，
   必须由用户批准并新建authority，不能在E186内静默执行。

建议路线1，并新开实验版本/计划；P的CoACD collider和G的保守下界可继续复用。
