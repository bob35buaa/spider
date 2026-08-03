# E187 amendment：用户接受约25s并豁免C9后继续Full

_Core4D Phase 50 · 2026-08-03 · 补充且不覆盖plan205/206_

## Context

E187 A1/A2已冻结，三设备production canary按`1024×32 seed0`运行。local RTX5090
bucket003 canary PASS，但与同卡E178短baseline比较得到：

- E187 optimized median：`24.9860s`；
- E178 optimized median：`11.0563s`；
- same-device ratio：`2.259888×`，C9 technical status=`FAIL`；
- Full已启动行数：`0`。

用户于2026-08-03明确回复：`没事, 25s可以接受,继续吧`。本amendment将该文本解释为
只豁免plan206 C9 plan-time ratio停止门并授权继续A3/A4，不把C9测量改写成PASS。

## 授权边界

1. 保持Gate0 `technical_gate_status=FAIL`、`progression_authority=USER_WAIVED`；
2. 保持C9 `technical_status=FAIL`、ratio=`2.259888×`，新增
   `c9_progression_authority=USER_WAIVED`；
3. 不豁免三条canary的finite、OOM、video、config/scene/grid/A2 SHA、recorder-off；
4. 不豁免A1 reward/grid、A2 production integration、keep22 authority和输出隔离；
5. 仍只允许local RTX5090 GPU0与`spider-remote` RTX6000 Ada GPU0/1；A100=0；
6. 仍禁止kill、暂停、抢占、等待idle、覆盖incomplete/complete row；
7. 不回调reward/grid/P/G/samples/iterations/seed，不修改log256/257与历史负证据；
8. Claims未闭合前不commit/push。

## Claims与成功标准

| Claim | 成功标准 |
|---|---|
| W2 C9 waiver authority | 用户文本、local C9 failure artifact、plan207 SHA进入immutable waiver manifest |
| W3 evidence honesty | A3/Full/report同时保留C9 technical FAIL与USER_WAIVED progression |
| C8 canary | 三设备manifest PASS、finite、video可读、peak≤6GiB、recorder-off、SHA闭合 |
| C10 Full closure | keep22 completed+terminal_failed=22、missing=0；正常目标22 completed |
| C13 isolation | A100=0、共享checkout未改、无kill/暂停/抢占、输出不覆盖 |

## 执行阶段

1. 冻结`c9_user_waiver_manifest.json`，引用原始FAIL artifact，不删除或改写；
2. pull并验证remote004/007 canary及007 baseline；补004 baseline仅用于报告，不再阻断progression；
3. 生成A3 lock：`c8_pass=true`、`c9_technical_pass=false`、
   `c9_progression_authority=USER_WAIVED`；
4. 生成三worker LPT keep22 queue，三条canary原子登记为前三个completed row；
5. 部署新的immutable Full snapshot并启动剩余19行，卡间并行、卡内串行；
6. manifest-driven pull、22-row closure、视频与paired evaluation。

## Canonical commands

```bash
bash workspace/core4d/scripts/launch/active/pull_E187_remote_a6000_results.sh canary
bash workspace/core4d/scripts/launch/active/pull_E187_efficiency_baseline.sh
bash workspace/core4d/scripts/launch/active/run_E187_local.sh full
bash workspace/core4d/scripts/launch/active/run_E187_remote_a6000.sh full
bash workspace/core4d/scripts/launch/active/pull_E187_remote_a6000_results.sh full
```

## Stop rules

- 任一C8 technical gate失败仍停止，不由本waiver覆盖；
- Full snapshot/queue/row SHA不一致、OOM、non-finite、video不可读或输出冲突立即停止对应E187 worker；
- 不操作外部进程，不用A100，不降低预算，不原地覆盖失败row；
- C9后续Ada测量无论数值如何都如实记录，但由本用户授权不再阻断A4。
