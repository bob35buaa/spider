# CORE4D 当前进度

> 完整备份：
> [E167–E179 初始恢复](progress_archive/E167_E179_20260724_full_backup.md) ·
> [E179 执行过程](progress_archive/E179_20260724_execution_full_backup.md) ·
> [E179 最终收尾](progress_archive/E179_20260725_closeout_full_backup.md) ·
> [E180–E181 完整执行](progress_archive/E180_E181_20260731_full_backup.md) ·
> [E182 Gate0–S1 canary](progress_archive/E182_gate0_s1_canary_20260801_full_backup.md)
>
> 本文件只保留 E182 当前可靠结论与下一执行入口。

## 2026-08-01：E182 资源与不可变口径

- 计划：`plan/200_E182_task_conditioned_coacd_full_cem_plan.md`；Tracker 为
  `✅ Gate0 PASS；🚧 S1 query tape`。
- Full authority 与 E178 exact：27 case、`1024×32 seed0`；K 只测
  `8/16/32`，不测 K4。
- heldout24 在 production SHA 冻结前 selection-forbidden，冻结后仅
  evaluation-only；不得按 heldout/Full 结果反选碰撞体。
- 资源只用本机 GPU0 + `spider-remote` RTX 6000 Ada GPU0/1；三卡允许与
  已有任务叠加。禁止 kill、暂停、抢占或修改已有进程，不使用 A100。

## 2026-08-01：S0 Gate0 PASS

- authority=`27/3/24`，case snapshot files=`108`，E181 candidates=`54`，
  source snapshot files=`5083`。
- remote frozen root：
  `/home/xiayb/pHRI_workspace/e182_runs/e182_fbe65a69fa8a0b70/spider`。
- remote source verification=`0 mismatch`；独立 CoACD dependency layer
  `10/10 SHA PASS`，未修改 shared checkout/venv/现有进程。
- completion audit=`13/13 PASS`；direct-main preflight tests=`5/5 PASS`。

## 2026-08-01：S1 query tape canary

- E178 历史 NPZ 不含 1024 samples 的原始 qpos/query tensor，不能伪恢复；
  dev3 使用 exact E178 override/task/seed 做 `64×4 seed0` shadow replay。
- 已实现默认-off recorder、chunked qpos/reward/G payload、runner、finalizer、
  same-run auditor 与 direct-main tests。公共代码只增加真实语义 diff：
  `run_mjwp +1`、`config +5`、`sampling +48/-1`；误格式化噪声已清理。
- bucket007 canary：off/on_a/on_b wall=`107.25/106.44/107.06s`；on_a/on_b
  各 `77` chunks、约 `40MiB`，manifest 均 `COMPLETE`。
- frozen content SHA：on_a=`ab73e0ee1b9b...`，on_b=`7804dacc8bab...`；
  finalize 幂等，重复 runner 为 `SKIPPED_COMPLETE`。
- same-run exact audit：on_a/on_b 均 PASS，mismatch/nonfinite=`0/0`；
  reward max/min/median/mean、selected index 与全部 1-D sample summaries exact。
- deterministic CPU mock on/off exact，default-off 无 hidden qpos；recorder tests
  `3/3`、pipeline `3/3`、preflight `5/5`、ruff/py_compile/diff-check PASS。
- 真实 MJWarp 跨进程不 bitwise：off/off 与 on/on 均自然分叉，qpos capture 还会
  改变 CUDA 调度并放大闭环差异。plan Gate1 已实证修订：跨 run divergence
  report-only；硬门为 mock/default-off/same-run/COMPLETE/frozen SHA。
- 唯一 selection tape 使用 on_a；on_b 只作诊断。recorder-on shadow 不能宣称
  downstream 改善；production Full 强制 recorder-off。

## 下一执行入口

0. progress 全文已归档到
   `progress_archive/E182_gate0_s1_canary_20260801_full_backup.md`（283 行）；活跃文件
   已压缩为 63 行。archive/链接、bash syntax、无 kill 命令与 diff-check 均 PASS。
1. git checkpoint 已创建：`exp(core4d): E182 Gate0 and query-tape canary`；未 push。
2. 对 bucket003/bucket004 跑 off/on_a/on_b，并完成 dev3 same-run audit。
3. 构建 ref、E178-final、CEM sample 的 object-local P/R/G query tape 与 provenance。
4. Gate1 全量 PASS 后进入 S2：在冻结 dev3 tape 上审计 K8/K16/K32，不读取 heldout。
5. production K/grid/config SHA 冻结后，才启动本机0 + Ada0/1 的 27-case Full；
   launcher 只叠加运行，绝不处理现有进程。
