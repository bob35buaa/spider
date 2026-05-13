# HDMI Reproduce 环境配置

基于 `feat/hdmi-reproduce-v2` 分支，使用 uv 管理 Python 环境。

## 前置条件

- Python 3.12+
- CUDA 12.8+ (已验证 CUDA 12.9 + NVIDIA L20Y)
- uv (`~/.local/bin/uv`，确保在 PATH 中)
- IsaacLab v2.2.0 源码: `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/IsaacLab_v2.2`
- HDMI 源码: `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/Loco-Manipulation/HDMI`

## 安装步骤

所有命令在 spider 项目根目录下执行。

### 1. 基础依赖

`pyproject.toml` 中 nvidia 源 (`pypi.nvidia.com`) 在内网不可达，需要先注释掉：

```toml
# [[tool.uv.index]]
# name = "nvidia"
# url = "https://pypi.nvidia.cn/"
# explicit = true

# [tool.uv.sources]
# warp-lang = { index = "nvidia" }
```

然后安装：

```bash
uv sync --index-url http://pypi.devops.xiaohongshu.com/simple/ \
  --trusted-host pypi.devops.xiaohongshu.com
```

### 2. PyTorch (CUDA 12.8)

直接用 PyTorch 官方 cu128 索引会因为 nvidia 子依赖走 `pypi.nvidia.com` 而超时，
需要同时指定小红书镜像作为 extra-index：

```bash
uv pip install --python .venv/bin/python3 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url http://pypi.devops.xiaohongshu.com/simple/ \
  --trusted-host pypi.devops.xiaohongshu.com
```

### 3. TorchRL + TensorDict

```bash
uv pip install --python .venv/bin/python3 \
  torchrl==0.7.0 tensordict==0.7.0 einops \
  --index-url http://pypi.devops.xiaohongshu.com/simple/ \
  --trusted-host pypi.devops.xiaohongshu.com
```

> torchrl 0.7.0 适配 PyTorch 2.7，与 2.8.0 存在 C++ bindings 兼容性 warning，
> 核心功能不受影响，仅 prioritized replay buffer 可能异常。

### 4. IsaacLab v2.2.0

克隆（如未下载）：

```bash
git clone https://github.com/isaac-sim/IsaacLab.git \
  --branch v2.2.0 /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/IsaacLab_v2.2
```

用 uv 以 `--no-deps` 模式安装各扩展模块（跳过 Isaac Sim 等不可用依赖）：

```bash
PYTHON=".venv/bin/python3"
for ext in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_mimic; do
  uv pip install --python "$PYTHON" --no-deps \
    -e "/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/IsaacLab_v2.2/source/$ext" \
    --index-url http://pypi.devops.xiaohongshu.com/simple/ \
    --trusted-host pypi.devops.xiaohongshu.com
done
```

补装 IsaacLab 运行时依赖：

```bash
uv pip install --python .venv/bin/python3 \
  flatdict==4.1.0 prettytable toml gymnasium \
  --index-url http://pypi.devops.xiaohongshu.com/simple/ \
  --trusted-host pypi.devops.xiaohongshu.com
```

#### 修复：omni 模块缺失

IsaacLab 的 `isaaclab.utils.math` 在顶层 `import omni.log`，该模块属于 NVIDIA Isaac Sim，
我们的环境中没有安装完整的 Isaac Sim，但 HDMI 流程只用到了 `omni.log.warn` 做日志输出。

解决方案：在 venv 的 site-packages 下创建 omni stub 包：

```bash
SITE_PKG=".venv/lib/python3.12/site-packages"
mkdir -p "$SITE_PKG/omni"

cat > "$SITE_PKG/omni/__init__.py" << 'EOF'
# Stub for omni package (Isaac Sim dependency)
EOF

cat > "$SITE_PKG/omni/log.py" << 'EOF'
"""Stub for omni.log (Isaac Sim dependency)."""
import logging

_logger = logging.getLogger("omni.log")


def warn(msg, *args, **kwargs):
    _logger.warning(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    _logger.info(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    _logger.error(msg, *args, **kwargs)
EOF
```

### 5. HDMI 包

```bash
uv pip install --python .venv/bin/python3 --no-deps \
  -e /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/Loco-Manipulation/HDMI \
  --index-url http://pypi.devops.xiaohongshu.com/simple/ \
  --trusted-host pypi.devops.xiaohongshu.com
```

## 运行

### 直接运行（绕过 uv sync）

使用 `.venv/bin/python` 而非 `uv run`，避免 uv 自动同步依赖时访问不可达的 nvidia 源：

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python examples/run_hdmi.py \
  task=move_suitcase +data_id=1 viewer=none save_video=false save_info=true \
  max_sim_steps=250 num_samples=1024 max_num_iterations=32 \
  output_dir=workspace/hdmi_reproduce/results/R008 use_torch_compile=false
```

### 渲染可视化视频

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python \
  workspace/hdmi_reproduce/scripts/render_trajectory_video.py \
  --scene /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/Loco-Manipulation/HDMI/active_adaptation/assets_mjcf/g1_29dof_nohand/g1_29dof_nohand-suitcase.xml \
  --kin <results_dir>/trajectory_kinematic.npz \
  --phys <results_dir>/trajectory_hdmi.npz \
  --output <results_dir>/comparison.mp4 \
  --fps 10
```

### 已知运行时问题

1. **`save_video=true` 报 EGL 错误** — MuJoCo Renderer 的 `MUJOCO_EGL_DEVICE_ID` 在 warp 初始化后被改为 1，但 EGL 只有 1 个设备（index 0）。暂用 `save_video=false`，后续用单独脚本渲染视频。
2. **`torch.compile` 失败** — triton 编译时 gcc 链接 `-lcuda` 报错。暂用 `use_torch_compile=false` 绕过（性能慢 ~2-3x）。

## 验证

```bash
.venv/bin/python workspace/hdmi_reproduce/scripts/test_hdmi_fast.py
```

预期输出（末尾）：

```
ALL PASSED
```

## 已安装包版本

| 包 | 版本 |
|---|---|
| torch | 2.8.0+cu128 |
| torchrl | 0.7.0 |
| tensordict | 0.7.0 |
| isaaclab | 0.44.9 |
| mujoco | 3.7.0 |
| mujoco-warp | 3.7.0.1 |
| warp-lang | 1.12.1 |
| spider | 0.1.0 |
| active-adaptation (HDMI) | 0.0.0 |

## huggingface数据集下载
```bash
python /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/tools/download_hf_data.py \
    --repo_id retarget/retarget_example \
    --output_dir /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider/example_datasets/ \
    --allow_patterns "*move_suitcase/*"
```

## git相关

### 1. 配置代理
```bash
# 设置 git 全局代理
git config --global http.proxy http://10.140.15.68:3128
git config --global https.proxy http://10.140.15.68:3128
```

### 2. 配置信息
```
user.name: bob35buaa
user.email: 643692325@qq.com
remote: https://github.com/bob35buaa/holosoma.git
```

### 3. 保护本地修改不被误 commit (skip-worktree)

`pyproject.toml` 的 nvidia 注释（见本文 §1）和 `uv sync` 后的 `uv.lock` 都是**本机专用**修改，**不能** 推到 git（其他机器需要 canonical 的 nvidia 启用版才能 install）。

`.gitignore` 对**已 tracked 的文件无效**，所以靠 .gitignore 是挡不住的。正确做法是 `git update-index --skip-worktree`：

```bash
# 一次性设置（每次新 clone 都要重新跑）
git update-index --skip-worktree pyproject.toml uv.lock

# 验证（应看到两个文件前面有 S 标记）
git ls-files -v | grep -E "^S (pyproject\.toml|uv\.lock)"
```

设置后：
- 本地 `pyproject.toml` 可以随便 nvidia 注释、`uv sync` 改 `uv.lock`，**`git status` 都不会显示**
- `git add` / `git add -A` / `git commit -a` 都不会捎带这两个文件
- git 推过来的 canonical 版本也不会覆盖本地（pull 时 git 会跳过它们）

**注意点 1 — skip-worktree 是 per-clone 设置**（保存在 `.git/index`，不进 git 历史）。换机器或重新 clone 后必须重新跑一遍：

```bash
git clone <repo>
cd <repo>
# 本地需要的修改 (例如注释 nvidia 索引)
sed -i 's/^\[\[tool\.uv\.index\]\]/# [[tool.uv.index]]/' pyproject.toml  # 等
# 然后立刻保护
git update-index --skip-worktree pyproject.toml uv.lock
```

**注意点 2 — 真要修改 canonical 内容时（加新依赖等），必须先临时关掉 skip-worktree**：

```bash
git update-index --no-skip-worktree pyproject.toml
# 改 pyproject.toml: 比如 uv add new-dep ...
git add pyproject.toml uv.lock
git commit -m "deps: add new-dep"
git push
git update-index --skip-worktree pyproject.toml uv.lock  # 重新保护
```

如果忘了关 skip-worktree 就改 + commit，git 会**静默跳过**你的修改，commit 里只有 message 没有内容（容易让人误以为推上去了实际没有）。
