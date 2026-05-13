# Experiment Guidelines

## 1. Structured Logging & Output Paths

- Training runs MUST use structured, named output paths: `logs/{experiment_name}/{run_id}/`
- Naming should be descriptive and traceable — easy to locate and compare across runs.
- Use W&B for metric tracking when applicable:
  ```bash
  # Login if needed
  export WANDB_API_KEY=d3e840a3d07879a92dec0417d82acc5e68e53e52
  ```

## 2. Experiment Records

Every experiment MUST have a record documenting:
- **Purpose**: hypothesis being tested
- **Parameters**: key config changes, hyperparameters, reward weights
- **Run command**: exact CLI invocation for reproduction
- **Result**: metrics, observations
- **Conclusion**: what was learned, next steps

Format: commit messages, dedicated experiment logs, or PR descriptions.

## 3. Config-Driven Control (Decoupling)

- Prefer **config/override** over modifying core code — use Hydra overrides, YAML overlays, or CLI flags.
- Core modules must remain decoupled: one experiment's changes must NOT silently break another.
- If core code must be changed, ensure the change is **isolated and reversible** — and MUST be tracked by git for traceability and reproducibility.

## 4. Git Discipline

- Use git to manage all experiment code systematically.
- Branch when necessary: `feat/E{NNN}-{short-description}`
- Commit promptly upon experiment completion or feature implementation.
- Commit messages: `exp(context): E{NNN} description`
- Every modification must be reproducible via git history.

## 5. Evaluation Criteria & Rigor

Every experiment MUST define **clear, quantitative evaluation criteria** before running.

### Evaluation Standards

- **Explicit success/failure criteria**: No vague "looks okay" conclusions. Define concrete numeric thresholds or comparison baselines.
- **No lenient thresholds**:
  - Do NOT cherry-pick the best rollout as the result — report mean + std + worst case.
  - Do NOT rely on a single metric — evaluate across multiple dimensions (tracking accuracy, contact fidelity, stability, task completion).
  - Do NOT accept "slightly better than before" — require statistically meaningful improvements.
- **Traceable evaluation basis**:
  - Document ground truth sources (mocap reference, human annotation, physics constraints).
  - If using proxy metrics, justify their correlation with the true objective.
  - Baselines MUST be compared under **identical conditions** — no confounding variables.

### Visual Evaluation

Numeric metrics alone cannot fully capture physics simulation quality. **Visual evaluation is mandatory, not optional.**

- For motion/manipulation experiments, MUST record video and inspect manually:
  - Use `/video-frames` skill to extract frames at critical moments (grasp, contact, transitions).
  - Check for penetration, floating, jitter, unnatural poses, and other visual artifacts.
  - NEVER conclude from reward curves alone — reward hacking is often obvious on video.
- For comparative experiments, side-by-side A/B video of the same clip is far more convincing than numbers alone.

### Common Evaluation Pitfalls

| Pitfall | Symptom | Correct Approach |
|---------|---------|-----------------|
| Lenient threshold | "80% success" but "within 10cm" counts as success | Tighten criteria; report success at multiple thresholds |
| Selective reporting | Only showing the best 5 rollouts | Report full statistical distribution over all N trials |
| Metric deception | High reward but robot is jittering on video | MUST visually confirm behavior is physically plausible |
| Missing baseline | "Loss decreased" with no comparison target | Always include baseline (prior version / ablation / literature) |
| Time-window cropping | Only evaluating first 2s (stable phase), ignoring later collapse | Evaluate the full episode duration |

## 6. Workspace Organization

- Keep experiment working directories organized by task/dataset/phase.
- Temporary outputs (checkpoints, debug dumps) belong in gitignored paths.
- No scattered artifacts in the repo root.

## 7. Scene/Data XML Reproducibility (Dual Safeguards)

`example_datasets/` is in `.gitignore`, but scene XMLs (collision boxes, object poses, euler conventions) get edited as experiments progress. mtime alone is not reproducible. **Two safeguards are required**:

**Safeguard 1 — Active cases tracked in main git**: scene XML for any case under active experimentation MUST be force-added to git (`git add -f`). When activating a new case, force-add its `scene.xml`, `scene_act.xml`, and adjacent JSON metadata (`scene_act_meta.json`, `task_info.json`) BEFORE running the first experiment on it.

**Safeguard 2 — Per-experiment snapshot**: every experiment that runs physics simulation MUST snapshot the exact XML files used into `workspace/{exp_name}/results/E0NN/scene_snapshot/` BEFORE training starts. Snapshot must include:
- All scene XML files for every case the experiment touches
- A `manifest.txt` recording git HEAD + sha256 of each file

The snapshot is committed alongside the experiment log. Even if main-git XMLs are later overwritten, E0NN's exact training state remains recoverable.

**Helper**: `workspace/{exp_name}/scripts/convert/snapshot_scenes.sh <EXP_ID> <case1> [case2 ...]`

**Train script integration**: training scripts (`scripts/train/train_E0NN.sh`) MUST call snapshot_scenes.sh as the first step. The experiment log's "改动文件" / "Files changed" table MUST list the `scene_snapshot/` path.

**Multi-stage experiments**: if an experiment modifies XMLs between stages (e.g., snap → CEM, or margin sweep), each stage that re-uses the modified XML must re-snapshot.

