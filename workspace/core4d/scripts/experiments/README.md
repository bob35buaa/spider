# CORE4D experiment script archive

This directory is the target layout for experiment-local scripts and manifests.

Current Phase 6 policy:

- `legacy/E###` contains archived historical experiment directories up to E081.
- `E082+` directories are structural copies while the old
  `workspace/core4d/scripts/E###` directories remain the executable compatibility
  paths.
- Many copied scripts still compute the repo root from
  `Path(__file__).resolve().parents[...]`; running those copied scripts directly
  from this deeper directory can change that calculation.

Until a script has been migrated to `workspace/core4d/scripts/eval/runners`,
`workspace/core4d/scripts/eval/reports`, or updated to use
`workspace/core4d/scripts/common/paths.py`, run it through the old compatibility
path under `workspace/core4d/scripts/E###`.
