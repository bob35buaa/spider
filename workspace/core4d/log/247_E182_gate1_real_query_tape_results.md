# E182 阶段日志：dev3 真实 P/R/G query tape 与 Gate1

_Core4D Phase 45 · 2026-08-01 · plan
[200](../plan/200_E182_task_conditioned_coacd_full_cem_plan.md)_

## 0. 一句话结论

E182 Gate1 已正式 `PASS`：三个 dev case 的 recorder-on replay、same-run payload、
object-local P/R/G tape、runtime inputs 和默认关闭/mock 合同全部通过；唯一 selection
tape 固定为 `on_a`，selection set SHA 为
`3781ad96505ee597222f145fbcaa943be198a9473fb4f0faf72b56fed247c8a5`。heldout24
仍未访问，因此允许进入 S2 dev-only task-conditioned candidate audit。

## 1. 范围与不可变边界

- dev3=`bucket003_20231018_001_p1`、`bucket004_20231002_021_p1`、
  `bucket007_20231020_055_p1`；不读取 heldout24 candidate evidence；
- replay 固定 E178 override/task/seed，shadow budget=`64×4 seed0`；
- `on_a` 是唯一 selection distribution，`on_b` 仅用于 CUDA divergence 诊断；
- production Full 仍强制 recorder-off；S1 结果不宣称 downstream 质量改善；
- 资源为本机 GPU0 + 远程 RTX 6000 Ada GPU0/1，与已有任务叠加；两个远程 worker
  均自然结束，未 kill、暂停、抢占或修改已有进程，也未使用 A100。

## 2. Replay 与 same-run 结果

| Case | on_a/on_b chunks | mismatch | nonfinite | on_a content SHA |
|---|---:|---:|---:|---|
| bucket003 | 202/202 | 0/0 | 0/0 | `561ea037...394f` |
| bucket004 | 139/139 | 0/0 | 0/0 | `68378948...22fd` |
| bucket007 | 77/77 | 0/0 | 0/0 | `ab73e0ee...0b93e` |

same-run hard gate逐 chunk核对 reward max/min/median/mean、selected index、所有 1-D
sample summaries、shape、finite 和 SHA。真实 MJWarp 跨进程不 bitwise，因此
off/on_a 与 on_a/on_b divergence 按计划仅 report-only，不用于否决 Gate1。

### Shadow replay wall time

| Case | off | on_a | on_b | 说明 |
|---|---:|---:|---:|---|
| bucket003 / Ada0 | 660.28s | 660.55s | 662.00s | 与已有 SUGAR 共存 |
| bucket004 / Ada1 | 474.98s | 480.84s | 479.62s | 与已有 SUGAR 共存 |
| bucket007 / local0 | 107.25s | 106.44s | 107.06s | 原 canary manifest未保留 wall field，数值来自冻结 canary记录 |

远程共存使单 active step 明显慢于本机，但三条 run 均无 OOM、NaN 或重启。该数据只
描述 S1 recorder shadow，不替代 S4 的同机 E178/E182 production-density probe。

## 3. Object-local P/R/G tape

| Case | total chunks | Stored | Expanded equivalent | Ratio | P geoms | R/G points |
|---|---:|---:|---:|---:|---:|---:|
| bucket003 | 204 | 925.20 MiB | 11.59 GiB | 12.83× | 18 | 1669 |
| bucket004 | 141 | 636.73 MiB | 7.98 GiB | 12.83× | 18 | 1669 |
| bucket007 | 79 | 352.83 MiB | 4.42 GiB | 12.83× | 18 | 1669 |
| Total | 424 | 1.87 GiB | 23.99 GiB | 12.83× | — | — |

每条 tape 包含一块 reference、一块 E178-final 与全部 `on_a` CEM chunks。reference
通过 frozen config + `load_data` 和 E027b 43→42 body-frame conversion恢复；final
来自 E178 NPZ 的 `(record, ctrl_step, 42)` flatten。P 保留完整 qpos 作 full-scene
physics replay；R/G 使用无损 `geom pose + fixed local offsets/radius/mask` 表示，按需
materialize object-body-local 查询点，不保存 24 GiB 展开张量。

三条 tape 的逐 chunk/source SHA、array schema、finite、pose axis、aggregate size 和
raw `on_a` content SHA 均通过 verifier，mismatch=`0`。

## 4. Gate1 aggregate

| Evidence | Result |
|---|---|
| Runtime input closure | `50/50 PASS`，missing/mismatch=`0/0` |
| Dev3 same-run aggregate | `3/3 PASS` |
| Factored PRG case verifier | `3/3 PASS` |
| Selection policy | `ON_A_ONLY_ON_B_DIAGNOSTIC` |
| Heldout access | `NOT_ACCESSED_DEV3_ONLY` |
| Direct-main tests | `5/5 groups PASS` |
| Gate1 artifact SHA | `a1177e14ae16ae1252a9f31aeefef3ebb6ab1a67044327498f4b11904919f6f5` |

direct-main evidence覆盖：config default-off、deterministic mock on/off exact、recorder
schema/finalize、same-run/relocated pull、runtime immutable closure、P/R/G source schema
和 Gate0 preflight。aggregate auditor同时冻结自身、PRG builder、replay auditor/runner
与各 test source SHA。

## 5. Claims 状态

| Claim | 当前证据 | 状态 |
|---|---|---|
| C0 authority | Gate0 的 27/3/24 authority 与 runtime SHA保持 exact | PASS |
| C2 real-query tape | dev3 ref/final/CEM、P/R/G provenance与逐 chunk verifier完整 | PASS |
| C3 task-conditioned fidelity | 尚未运行真实 54-candidate audit | PENDING S2 |
| C1/C4–C9 | production C/D_C、canary、Full、paired eval尚未执行 | PENDING |

Gate1 PASS只证明真实查询分布已被可靠冻结，不说明任何 CoACD candidate 已通过 task
launch floor，也不说明 E182 优于 E178。

## 6. 可视化

S1 是 observational instrumentation/integrity stage，没有生成新 physics 视频，也不对
碰撞体外观或轨迹可用度作视觉结论。三维/二维 candidate heatmap与 P contact replay
属于 S2，27-case paired 视频与盲审属于 S7；这些阶段仍按 plan 的可视化硬门执行。

## 7. 验证与已修问题

- query recorder `3/3`、pipeline `4/4`、runtime/launcher `4/4`、PRG tape `3/3`、
  preflight `5/5`、Gate1 auditor `2/2`、S2 synthetic geometry `3/3` PASS；
- E182 ruff、format-check、launchers `bash -n`、`git diff --check` PASS；
- 修复 remote pull 后绝对 chunk path失效：增加 sibling+SHA resolver；
- 修复 Ada0/Ada1 case manifest覆盖：改为 case-scoped manifest；
- 修复 explicit MuJoCo pair 被 collision mask漏掉：P inventory恢复为18 geoms；
- expanded tape约24 GiB，改为无损 factored layout后降为1.87 GiB。

## 8. 结果路径

| Artifact | Path |
|---|---|
| Gate1 aggregate | `results/E182/s1_query_tape/gate1_audit.json` |
| Same-run aggregate | `results/E182/s1_query_tape/same_run_integrity_audit.json` |
| Raw replays/chunks | `results/E182/s1_query_tape/{replays,raw_chunks}/` |
| Factored P/R/G tapes | `results/E182/s1_query_tape/prg_query_tape/` |
| Runtime manifest | `results/E182/s1_query_tape/runtime_inputs_manifest.json` |
| Remote logs | `logs/E182/s1/` |

以上相对 `workspace/core4d/`，远程 immutable source root 为
`/home/xiayb/pHRI_workspace/e182_runs/e182_832e7fe0f61193a1/spider`。

## 9. 下一步

1. Tracker 状态更新为 `✅ Gate1 PASS；🚧 S2 task audit`；
2. 仅在 frozen dev3 tape 上评估 bucket003/004/007 各18个 K8/K16/K32 candidate；
3. 分离报告 `D_M vs exact-C` decomposition error 与后续 `exact-C vs D_C` grid error；
4. 在 S5 production SHA 冻结前继续禁止读取 heldout24 的 E182-dependent evidence。
