# E187 Gate S0 预注册合同与 evaluator 实现审计

日期：2026-08-02
实验：E187
阶段：S0 contract audit
结论：修正 evaluator 的 local/cross-GPU 路由；historical query/selected 证据缺失，Gate S0 仍 FAIL；Ada/Full 启动数 0

## Context

[log 256](256_E187_e178_compatibility_gate0_blocker_results.md) 冻结了首次本地 RTX5090
formal replay 的运行事实、all-array 差异、semantic 差异、短程非确定性诊断和可视化。
本审计不修改该历史日志，只核对 [E187 plan 205](../plan/205_E187_canonical_distance_continuation_reward_plan.md)
的 Gate S0 原文是否被 direct-core evaluator 正确实现。

## 预注册合同审计

| 证据 | plan 205 原文 | 旧 evaluator 实现 | 审计结论 |
|---|---|---|---|
| local same-device | 关键 query qpos/reward `≤1e-5`；valid/selected 一致 | 所有共同同shape numeric/bool arrays `≤1e-5` | 实现范围过宽，all-array 只能作 diagnostic |
| cross-GPU | tracking位置/姿态、contact、penetration semantic tolerance | 每个row均计算且强制 semantic | 应只门控 Ada cross-GPU row |
| row pass | local 或 cross-GPU 各走对应门 | 每row同时要求 exact 与 semantic | 与预注册合同不一致，已修正 |

历史运行日志直接确认：bucket003/004 的 E178 Full 在 A100 上运行，bucket007 在本地
RTX5090 上运行。因此冻结路由应为：003→Ada GPU0 semantic、004→Ada GPU1 semantic、
007→local RTX5090 same-device golden。

## Artifact 可验证性

E178 historical root/outdir NPZ 各有 456 个字段；E187 replay 各有 457 个字段。E178
NPZ包含最终轨迹 `qpos/qvel/ctrl`、reward/dist统计、valid mask的
`max/min/median/mean`聚合、gate valid/fallback/selected-valid fraction，但不包含：

- 完整 CEM query qpos；
- 每个 sample 的 query reward；
- 逐样本 body/hand/leg/posture/combined valid mask；
- selected indices。

E182 之后的普通 NPZ 新增 `cem_selected_index0`，但 E178 historical NPZ 生成于该字段之前。
完整 `qpos/rewards/selected_indices/sample_*_valid_mask` 只存在于显式开启 recorder 后的独立
query-tape chunk；E178 没有该 tape 或 manifest。最终轨迹与聚合统计不能冒充预注册的
query/selected golden，因此 same-device 证据必须 fail-closed。

## Evaluator 修正与复评

direct-core evaluator 已修正为：

1. local same-device 只接受 paired query-tape 的 qpos/rewards、五类valid mask和selected
   indices；缺字段或manifest时 `evidence_complete=false`；
2. cross-GPU row只门控8项semantic tolerance；
3. 原456-array exact保留为 `all_common_array_diagnostic_*`，不再替代golden合同；
4. 明确审计case-to-device route；当前首次manifest把三row都写成local5090，003/004
   device contract FAIL；
5. 新输出写入versioned目录，不覆盖log256引用的旧summary。

定向验证：ruff lint/format、diff-check与direct-main tests `5/5 PASS`。新增测试覆盖
local/cross路由互斥、missing historical query evidence fail-closed，以及当前003/004
manifest device错配显式化。

复评结果：

| 项目 | 结果 |
|---|---:|
| evaluated / not_ready / errors | `1 / 2 / 0` |
| bucket007 config / 12门 decision match | PASS / PASS |
| bucket007 same-device evidence | **FAIL：historical query-tape manifest缺失** |
| all-array exact / local semantic | FAIL / FAIL，均为diagnostic |
| manifest device contract | **FAIL：003/004仍为旧local placeholder** |
| status | `INCOMPLETE_OR_FAIL` |

旧summary SHA保持
`c9b42d8271193bff032c9d2b068697848516bcedd8d42a88123b2101dc4ce38d`；
新contract-audit summary SHA为
`70a1fe8ec164fbd9ce4e286e80d7e3450a866c9555883da69978783c8d5f83c8`。

## Determinism 结论边界

log256 的短程证据仍成立：same-current-source A/B 底噪不小于 old/current source 差异，
所以不能把轨迹分叉单归因于E182/E186/E187代码；同样也不能由此证明具体根因就是某个
Warp kernel，或事后放宽E187阈值。即使修正 evaluator 的路由，历史query/selected证据仍
不可追溯，不能通过重跑补造“historical golden”。

若要研究 deterministic replay 或制定未来可实现的 golden artifact 合同，必须新建独立
后续实验（建议E188）并在运行前预注册，不修改E187 Gate S0。

## Claims 与 stop/go

| Claim | 状态 | 证据 |
|---|---|---|
| C0 authority | PASS | E178/keep22/collider authority仍闭合 |
| C1 E178 compatibility | **FAIL** | same-device预注册证据不可由historical artifact验证 |
| C2 reward definition | PARTIAL | pure-function通过；S1 formal未获准 |
| C3–C12 | NOT STARTED | Gate S0失败后继续停止 |
| C13 isolation | PASS | 旧eval/log未覆盖；Ada/A100/Full 0；未操作既有进程 |

E187维持STOP：不启动bucket003/004 Ada replay、reward shadow、canary或keep22 Full。
修正设备manifest本身不能补回historical query/selected证据，因此本轮不创建远程worker或
部署入口。

## 结果路径

| 内容 | 路径 |
|---|---|
| 旧formal evaluator（冻结） | `workspace/core4d/results/E187/s0_environment/e178_compat/eval/` |
| contract-audit v2 | `workspace/core4d/results/E187/s0_environment/e178_compat/eval_contract_audit_v2/` |
| evaluator | `workspace/core4d/scripts/eval/runners/eval_E187_e178_compat.py` |
| evaluator tests | `workspace/core4d/scripts/experiments/E187/test_e178_compat_eval.py` |
