# 00 环境

v3 要求 SPIDER 与 OmniRetarget/hsretargeting 环境分离。不要把两个环境强行合并；只需要在 wrapper 中把路径、环境变量和版本记录清楚。

## 必需路径

| 变量 | 说明 |
|---|---|
| `SPIDER_REPO` | spider repo 绝对路径。默认可由当前工作目录推断。 |
| `HOLOSOMA_REPO` | holosoma repo 绝对路径。 |
| `CORE4D_RAW_ROOT` | 用户提供的 `CORE4D_Real` 根目录。 |
| `DATA_CONSTRUCTION_RUN_ROOT` | v3 run 输出根目录。默认 `${HOLOSOMA_REPO}/workspace/v3/data_construction_v3_runs`。 |
| `PYTHON_BIN` | SPIDER Python 解释器，通常为 `.venv/bin/python`。 |
| `RETARGET_PYTHON_BIN` | 可选，OmniRetarget/hsretargeting Python 解释器。未设置时 Stage2b wrapper 会 source `scripts/source_retargeting_setup.sh` 后优先使用 `$CONDA_PREFIX/bin/python`。 |

`CORE4D_RAW_ROOT` 至少应包含：

```text
CORE4D_Real/
  human_object_motions/
  object_models/
```

## 必需工具

- SPIDER Python 环境；
- OmniRetarget/hsretargeting 环境，由 `HOLOSOMA_REPO/scripts/source_retargeting_setup.sh` 激活；Stage2b retargeting 阶段必须使用 retargeting 环境的 Python，不能误用 SPIDER `.venv`；
- MuJoCo，可在 headless 环境中使用 EGL；
- ffmpeg / ffprobe；
- CUDA/GPU，取决于 CEM/RL 阶段；
- `numpy`, `scipy`, `trimesh`, `mujoco`, `smplx`, `torch` 等 Python 包。

## 环境检查命令

最小环境检查入口：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s0_environment/check_environment.py \
  --holosoma-repo "$HOLOSOMA_REPO" \
  --core4d-raw-root "$CORE4D_RAW_ROOT"
```

输出：

```text
environment_check.json
environment_check.md
```

检查项至少包括：

- raw path 存在；
- repo path 存在；
- output root 可写；
- `CORE4D_Real/human_object_motions` 与 `object_models` 存在；
- MuJoCo 能 load 一个 guard scene；
- ffmpeg/ffprobe 可执行；
- OmniRetarget retargeting Python 可导入 `smplx` 与 `holosoma_retargeting`；
- spider 与 holosoma git sha、dirty 状态；
- Python executable 和关键 package version。

初始化一个 v3 run：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/init_workspace.sh <run_id> \
  --holosoma-repo "$HOLOSOMA_REPO" \
  --run-root "$DATA_CONSTRUCTION_RUN_ROOT" \
  --core4d-raw-root "$CORE4D_RAW_ROOT"
```

该命令会创建标准目录、写入 S0 环境检查、初始化 `retarget_variant_registry` 和空的 `case_state_registry`。

## 禁止项

- 不要把输出写入 `workspace/v3/data_construction` 或 `workspace/v3/data_construction_v2`。
- 不要在脚本中硬编码 `/home/ubuntu/...` 或 `/mnt/...`。
- 不要把运行结果目录加入 git。
