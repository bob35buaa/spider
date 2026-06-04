# E133 Holosoma Ref-Object-Contact Env Probe Plan

日期：2026-06-03

## Context

E131 为 E126 的两个 `box021_035_p1/p2` paired fragment exports 生成了 Holosoma-compatible `object_contact (T,2)` proxy masks。E132 在 Holosoma `MotionLoader` CPU 层验证了 E126 missing-mask 与 E131 proxy-mask 的结构差异：E131 rows `has_object_contact=true`，shape 为 `214x2`，但尚未证明 IsaacSim env stepping 后 `motion_command.ref_object_contact` runtime view 能读到非零 mask。

E133 承接 E128 no-debug replay startup pattern 和 E132 loader contract，做 bounded env startup/stepping probe。该实验只验证 runtime command plumbing，不启动 CEM、PPO、Holosoma training、checkpoint 或 remote jobs。

## Claims

1. E131 proxy exports 在 Holosoma/IsaacSim env 中加载后，`motion_command.motion.has_object_contact=True`。
2. `motion_command.motion.object_contact` 保持非零 `(T,2)` contact mask，且 p2 row 是稳定 positive guard。
3. bounded replay stepping 内，`motion_command.ref_object_contact` 至少对 p2 row 出现非零样本，证明 command runtime view 与 motion loader plumbing 连通。
4. E131 proxy masks 仍是 geometry proxy，不是 raw/trimmed semantic contact mask；因此 `semantic_ref_mask_ready=false`、`rl_ready=false`，不得启动 PPO 或发布 main-case release claim。

## Inputs

- E131 p1 proxy export:
  `workspace/core4d/results/E131/holosoma_object_contact_proxy/exports/E126_box021_035_p1_with_partner_box021_035_p2_object_contact_proxy5cm.npz`
- E131 p2 proxy export:
  `workspace/core4d/results/E131/holosoma_object_contact_proxy/exports/E126_box021_035_p2_with_partner_box021_035_p1_object_contact_proxy5cm.npz`
- Holosoma Box021 object URDF:
  `/home/ubuntu/Workspace/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/Box021/Box021.urdf`
- Holosoma config alias:
  `exp:g1-29dof-wbt-w-object-r135-box021-handbox-exp0601-v4-3`

## Implementation

- Add `workspace/core4d/scripts/E133/holosoma_ref_object_contact_no_debug_probe.py`.
  - Mirrors E128 no-debug env startup.
  - Disables debug marker drawing.
  - Optionally monkeypatches replay sleep to keep bounded probe short.
  - Logs full-motion `object_contact` stats and sampled `ref_object_contact` stats.
  - Emits `E133_PROBE_SUMMARY <json>` for machine parsing.
- Add fixed eval entry:
  `workspace/core4d/scripts/eval/eval_E133_holosoma_ref_object_contact_env_probe.sh`.
  - Runs both E131 proxy exports under Holosoma source setup.
  - Writes manifest, summary JSON, summary Markdown, and logs under:
    `workspace/core4d/results/E133/holosoma_ref_object_contact_env_probe/`.

## Success Criteria

- Both rows complete no-debug startup without timeout or exception.
- Both rows report `has_object=True`, `has_partner=True`, `has_object_contact=True`.
- Both rows report nonzero full-motion `object_contact`.
- p2 positive guard reports `ref_contact_total > 0` during bounded stepping.
- Summary status is `pass`.
- `training_launched=false`, `cem_launched=false`, `remote_jobs_launched=false`, `rl_ready_rows=0`.

## Commands

```bash
bash workspace/core4d/scripts/eval/eval_E133_holosoma_ref_object_contact_env_probe.sh
```

Configurable environment variables:

- `GPU_ID` default `0`
- `E133_MAX_STEPS` default `250`
- `E133_TIMEOUT_SECONDS` default `600`
- `HOLOSOMA_ROOT` default `/home/ubuntu/Workspace/holosoma`

## Stop/No-Go Rules

- Do not run PPO or CEM from E133 evidence.
- Do not treat fragment `box021_035_p1/p2` rows as main `box021_029_p2` release evidence.
- If p2 `ref_object_contact` remains zero despite full-motion contact being nonzero, inspect command time-step/default-transition behavior before any reward-side claim.
