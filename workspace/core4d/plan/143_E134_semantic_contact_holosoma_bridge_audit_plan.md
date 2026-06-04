# E134 Semantic Contact to Holosoma Bridge Audit Plan

Date: 2026-06-03

## Context

E133 proved a narrow structural/runtime claim: E131 geometry-proxy
`object_contact` can be loaded by Holosoma and observed through
`motion_command.ref_object_contact` during bounded no-debug env stepping.
However, E133 did not prove semantic/raw contact readiness, PPO readiness, or
main-case release readiness.

The contact-improvement plan and E111-E114 already established the Spider/MJWP
side contact chain:

- E111 added S1 raw-contact artifacts and S6 contact alignment.
- E112/E113 ran contact-aware CEM with raw-mask/hold-band variants.
- E114 blocked RL handoff because no E113 row satisfied the stricter contact
  alignment handoff gate.

E134 audits whether those existing semantic/raw-contact artifacts can be bridged
into Holosoma `object_contact` for the E126/E131/E133 fragment exports without
inventing a new proxy or silently resizing a mask across incompatible time
axes.

## Claims

1. Existing semantic/raw-contact mask files for Box021 can be located for the
   current fragment/main cases.
2. The audit can distinguish exact time-axis compatibility from diagnostic-only
   mask availability.
3. A Holosoma semantic `object_contact` export is allowed only when a semantic
   mask has an exact compatible time axis and a valid person/hand axis.
4. If the mask and Holosoma export time axes do not match exactly, E134 must
   report `timeline_bridge_blocked` and keep `semantic_holosoma_mask_ready=false`.
5. E134 does not launch CEM, PPO, Holosoma training, checkpoints, or remote jobs.

## Inputs

Semantic/contact-side artifacts:

- `workspace/core4d/results/E079/contact_masks/box021_person1/raw_contact_mask_3cm.npz`
- `workspace/core4d/results/E082/contact_masks/d003_box021_20231011_035_p2/raw_contact_mask_3cm.npz`
- `workspace/core4d/results/E084/contact_masks/d003_box021_20231018_029_p2/raw_contact_mask_3cm.npz`
- `workspace/core4d/results/E100/fingertip_targets/d003_box021_20231011_035_p2/spider_contact_target_object_local.npz`
- `workspace/core4d/results/E100/fingertip_targets/d003_box021_20231018_029_p2/spider_contact_target_object_local.npz`

Holosoma-side artifacts:

- E126 paired fragment exports under
  `workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports/`
- E131 proxy exports under
  `workspace/core4d/results/E131/holosoma_object_contact_proxy/exports/`
- E107 Holosoma RL export paths where available, for diagnostic comparison only.

## Implementation

Add:

```text
workspace/core4d/scripts/E134/audit_semantic_contact_holosoma_bridge.py
workspace/core4d/scripts/eval/eval_E134_semantic_contact_holosoma_bridge_audit.sh
```

The script will:

- inspect selected semantic mask NPZ keys, shapes, person axes, active fractions,
  and provenance labels;
- inspect Holosoma export frame counts and existing `object_contact` source;
- compare mask time-axis lengths to Holosoma export lengths without resizing;
- write a manifest, summary JSON, and summary Markdown under
  `workspace/core4d/results/E134/semantic_contact_holosoma_bridge_audit/`;
- mark rows as `bridge_ready` only for exact mask/export length compatibility.

## Success Criteria

- Fixed eval entry runs locally and writes all summary artifacts.
- Every inspected mask/export row has explicit `semantic_mask_available`,
  `timeline_exact_match`, `semantic_holosoma_mask_ready`, and `failure_mode`.
- Any mismatch is reported as a blocker rather than silently converted.
- `training_launched=false`, `cem_launched=false`,
  `remote_jobs_launched=false`, and `rl_ready_rows=0`.

## Commands

```bash
bash workspace/core4d/scripts/eval/eval_E134_semantic_contact_holosoma_bridge_audit.sh
```

## No-Go Rules

- Do not resize, pad, or nearest-neighbor map semantic masks into Holosoma
  `object_contact` in E134.
- Do not treat E131 geometry proxy masks as semantic masks.
- Do not launch CEM/PPO/Holosoma training from E134.
- Do not claim main `box021_029_p2` release readiness from fragment evidence.
