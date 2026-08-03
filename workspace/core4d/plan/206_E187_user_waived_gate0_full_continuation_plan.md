# E187 amendment：用户豁免 Gate S0 后的 Full continuation

_Core4D Phase 50 · 2026-08-02 · 补充且不覆盖 plan 205_

上游权威：

- [E187 原计划](205_E187_canonical_distance_continuation_reward_plan.md)
- [首次 Gate S0 负结果](../log/256_E187_e178_compatibility_gate0_blocker_results.md)
- [Gate S0 合同审计](../log/257_E187_gate0_contract_audit_results.md)

## 1. Context 与用户授权

用户于 2026-08-02 明确要求：跳过 E178 historical query/selected golden 缺口，恢复 E187，
进入 Full CEM。该决定构成本 amendment 的执行授权。

授权的精确边界：

1. 只豁免 plan205 Gate S0 中无法由既有 E178 artifact 追溯的 query qpos/reward、逐样本
   valid mask 与 selected-index golden；
2. log256/257、旧/新 evaluator summary 与 C1 technical FAIL 保持不可变，不回写为 PASS；
3. 后续表格必须同时报告 `technical_gate_status=FAIL` 与
   `progression_authority=USER_WAIVED`；
4. 不豁免 authority/source isolation、config whitelist、legacy pure reward、runtime finite、
   output isolation，也不豁免 S1–S4 的 reward/grid/P/G/canary/效率安全门；
5. 不授权 A100，不授权 kill、暂停、抢占或修改任何已有本地/远程进程。

因此 E187 可以从 S1 继续，但不能把 waiver 写成“证明了 E178 可复现”。最终 paired 结果
必须带 `GATE0_USER_WAIVED` 限定；若没有完成后续门，不启动 keep22 Full。

## 2. 冻结科学变量

除上述 waiver 外，plan205 全部科学变量继续有效：

| Axis | Frozen value |
|---|---|
| keep authority | E186 keep22，顺序不变 |
| CEM | `1024 samples × 32 iterations × seed0` |
| P | 三个 object-specific CoACD compound collider |
| G | body/hand/leg/posture原阈值；object gate=`D_C-epsilon_grid` |
| R | plan205双尺度 continuation：`0.25/0.75, 50/15mm, delta=1mm` |
| grid tree | 每object `5.0→2.5→1.25mm`，冻结最粗全门通过者 |
| padding | 120mm，实际support最低110mm |
| Full devices | local RTX5090 GPU0 + `spider-remote` RTX 6000 Ada GPU0/1 |
| prohibited | A100；减samples/iterations/hulls；改keep22；操作既有进程 |

## 3. Claims

| Claim | 最低证据 |
|---|---|
| W0 waiver authority | 用户授权文本、plan206 SHA、log256/257 SHA均冻结；technical FAIL与waiver并列记录 |
| W1 evidence honesty | evaluator/report不把缺失golden写成PASS；所有后续row携带`GATE0_USER_WAIVED` |
| C2 reward definition | plan205 continuation公式/参数/temporal gate逐值一致；legacy纯函数回归不变 |
| C3 canonical D_C | 三object各按5→2.5→1.25mm冻结唯一grid；payload/source/manifest SHA闭合 |
| C4 P/G invariance | keep22 CPU/MJWarp compile；pair count与P/G阈值不变；false-safe=0 |
| C5–C7 R/G fidelity | 003 capture、007 rho/top-k/regret、三object formal shadow全部过plan205原门 |
| C8 canary | 三设备各1条`1024×32 seed0`完整、finite、无OOM/覆盖、视频可读 |
| C9 efficiency | same-hardware plan-time ratio≤1.50；R/G kernel ratio≤1.25；增量显存≤6GiB |
| C10 Full closure | keep22 `completed+terminal_failed=22`且missing=0；正常目标22/22 completed |
| C11–C12 result | paired 12门/连续指标/bootstrap/效率/22视频闭合，结论带waiver限定 |
| C13 isolation | E178/E181–E186不覆盖；A100=0；无kill/暂停/抢占 |

W0/W1 是新增的 waiver claims；C2–C13 沿用 plan205。C1 保持 technical FAIL，不参与
“通过”计数，但由用户授权允许 progression。

## 4. 阶段与门禁

### A0：waiver freeze

- 生成 `results/E187/s0_environment/gate0_user_waiver/waiver_manifest.json`；
- 冻结用户授权摘要、plan205/log256/log257 SHA、旧/新summary SHA；
- 更新 Tracker 为 `Gate S0 USER_WAIVED；S1–S4 pending`；
- 验证 E178 authority/source isolation仍不变。

**Gate A0**：W0/W1 PASS 才继续。

### A1：S1/S2 reward-grid lock

1. 重跑 continuation/legacy纯函数回归；
2. 使用 E186 reward-aligned tapes完成003/007，补004 production-fixed contact query；
3. 每object按5→2.5→1.25mm决策树做50k smoke、1M exact-C、CPU/CUDA、padding、
   false-safe、rho/top-k/regret与throughput；
4. 冻结 `reward_grid_lock.json`，之后不得根据canary/Full调参。

**Gate A1**：C2/C3/C5/C6/C7及plan205 §5.1全部通过。

### A2：S3 production integration

- 生成22个E187 opt-in overrides；
- keep22 CPU/MJWarp compile与pair矩阵闭合；
- 44条reference/E178-final query复核G false-safe=0；
- recorder-off；config/scene/grid/collider/input SHA fail-closed。

**Gate A2**：C4、22-row authority与override whitelist全过。

### A3：S4 三卡 canary

| Worker | Device | Case |
|---|---|---|
| local-0 | local RTX5090 GPU0 | bucket003 representative |
| remote-0 | RTX 6000 Ada GPU0 | bucket004 representative |
| remote-1 | RTX 6000 Ada GPU1 | bucket007 representative |

启动前后只做被动GPU/process snapshot。远程使用immutable source snapshot和独立run root，
不修改共享checkout。三条 scientific config 与最终 lock SHA一致时才可promote为Full前三行。

**Gate A3**：C8/C9全过；`>2.0×`或OOM停止，`1.5–2.0×`只允许一次数值不变优化。

### A4：S5 keep22 Full CEM

按canary wall做LPT，生成local/remote0/remote1互斥queue；三卡并行、卡内串行。只允许：

```text
local GPU0                         NVIDIA GeForce RTX 5090
spider-remote GPU0 / GPU1         NVIDIA RTX 6000 Ada Generation
```

每row输出唯一；complete自动skip；running/complete禁止覆盖。外部共存导致OOM时不操作外部
进程、不降budget；只有阻塞条件真实变化时允许一次相同科学配置recover。

**Gate A4**：正常目标22/22 completed；最低closure为completed+terminal_failed=22、missing=0。

### A5：S6 paired evaluation / visual

只join keep22与E178冻结baseline；完成12门、连续指标、10k bootstrap、object-stratified
sensitivity、同硬件效率、22/22视频和盲化人工review。结果分级沿用plan205，但标题和
machine-readable summary必须包含`GATE0_USER_WAIVED`。

## 5. 代码与 artifact

新增/修改范围：

| 文件/目录 | 用途 |
|---|---|
| `scripts/experiments/E187/` | waiver builder、reward/grid freeze、override/queue builders、validators |
| `scripts/eval/runners/eval_E187_*.py` | direct-core fidelity、canary、paired evaluator |
| `scripts/eval/wrappers/eval_E187_*.sh` | 固化A0–A5评测入口 |
| `scripts/launch/active/run_E187_local.sh` | local canary/full worker |
| `scripts/launch/active/run_E187_remote_a6000.sh` | Ada双worker与immutable deploy |
| `scripts/launch/active/pull_E187_remote_a6000_results.sh` | manifest-driven pull/SHA校验 |
| `scripts/launch/active/watch_E187_full.sh` | 只读监控与闭合评测 |
| `scripts/train/train_core4d_E187.sh` | canonical CEM入口，首步scene snapshot |
| `results/E187/s0_environment/gate0_user_waiver/` | waiver authority |
| `results/E187/s2_canonical_grid_sdf/` | grid候选与最终lock |
| `results/E187/s3_prg_audit/` | keep22 integration evidence |
| `results/E187/s4_canary/` | 三卡canary与效率证据 |
| `results/E187/s6_downstream/` | Full、eval、render与handoff |

不修改或覆盖E178/E181–E186 artifact、log256、log257和旧compat summary。

## 6. Canonical commands

所有命令必须先由对应脚本实现并通过static preflight，禁止从本文复制裸命令绕过manifest：

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E187_gate0_user_waiver.sh
bash workspace/core4d/scripts/eval/wrappers/eval_E187_reward_grid_fidelity.sh --require-all
bash workspace/core4d/scripts/eval/wrappers/eval_E187_compound_regression.sh --require-all

bash workspace/core4d/scripts/launch/active/run_E187_local.sh canary
bash workspace/core4d/scripts/launch/active/run_E187_remote_a6000.sh canary
bash workspace/core4d/scripts/launch/active/pull_E187_remote_a6000_results.sh canary

bash workspace/core4d/scripts/launch/active/run_E187_local.sh full
bash workspace/core4d/scripts/launch/active/run_E187_remote_a6000.sh full
bash workspace/core4d/scripts/launch/active/pull_E187_remote_a6000_results.sh full
bash workspace/core4d/scripts/eval/wrappers/eval_E187_paired_full.sh --require-all
```

## 7. Stop rules

1. 用户只豁免historical golden缺口；A0 authority/isolation或A1–A3任一失败即停止Full；
2. 不得把waiver改名为technical PASS，不得删除/覆盖负结果；
3. reward/grid lock后禁止调R/grid/C/keep22；Full出现后禁止任何科学参数回调；
4. 不减少hull、samples或iterations救效率；
5. 不使用A100，不kill/暂停/抢占现有进程；
6. 任一新物理运行前scene snapshot；远程只用immutable source snapshot；
7. Claims未全闭合不commit/push；全部闭合后按实验规则提交并推送。
