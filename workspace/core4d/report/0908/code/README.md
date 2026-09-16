# CORE4D 论文可视化渲染

`box021_20231011_037` 系列的三套渲染脚本，用于论文配图/视频。每个脚本都输出
**两版**：纯白背景（`*_white`）+ 带背景（`*_bg`，暖米色瓷砖地板 + 柔和阴影 + 环境）。
所有脚本都支持 **任意 CORE4D 序列**，数据根目录与 SMPLX model 路径均可通过命令行配置。

## 脚本一览

| 脚本 | 内容 | 渲染后端 |
|---|---|---|
| `viz_mixed_robot_smplx.py` | p2 机器人 + p1 SMPLX 人体 + 物体（交付） | pyrender + MuJoCo 取网格 |
| `viz_smplx_reference.py` | 两个人的 SMPLX 真值 + 物体（交付） | pyrender（Phong 柔和着色） |
| `viz_methods_compare.py` | **四方法 p2/G1 对比**（SPIDER / OmniRetarget / SBTO / GMR），仅 p2、仅 G1 + 物体 | pyrender + MuJoCo 取网格 |
| `viz_dual_robot_clean.py` | 两个人的重定向 G1 机器人 + 共享物体（不再作为交付，函数被 compare 复用） | MuJoCo 原生渲染 |
| `render_style.py` | 共享样式：MuJoCo 场景美化、pyrender 柔光/瓷砖地板、配色常量、CLI 公共参数 | — |
| `smplx_min.py` | 纯 NumPy 的最小 SMPLX 前向（LBS），用于按需重摆姿 | — |

**视角一致性（本轮更新）**：`mixed` 与 `smplx` 交付的相机已对齐到同一 3/4 正面视角
（p2 面向镜头）。`mixed` 相机方位角由 p2 机器人的平均朝向自动推出（`facing_yaw + --cam-az-offset`，
默认 +45°），不再固定 135°（旧值导致 p2 背对镜头）；`smplx` 的相机方向/FOV/距离系数默认与
`mixed` 对齐（`--cam-dir -1,0.3,1`、`--yfov-deg 45`、`--dist-mult 1.15`），且 p2（蓝）面向镜头。
`robot_dual` 不再作为交付渲染。

设计要点：
- **机器人 + 物体**贴近 OmniRetarget 论文配图：暖米色瓷砖地板、反射、柔和阴影、三点布光、渐变天空；机器人保留真实银/黑材质。
- **物体统一板岛蓝**：MuJoCo 用 `0.40,0.50,0.60`；pyrender 光照更强，用更深的 `0.24,0.36,0.54`，视觉上一致。
- **SMPLX 参考 CARI4D 观感**：Phong 柔和着色（point + 双 directional，柔和顶光避免过曝）+ 方向光阴影。默认用 **betas=0 的中性平均体型**重摆姿，修掉原始 baked betas 偏瘦/肋骨凸出的问题（加 `--baked-shape` 可切回原始体型）。

## 环境

**无需改动现有共享 venv**（`.venv/`）。本次工作没有 `uv sync`、没有安装任何新包，
`pyproject.toml` / `uv.lock` 均无 git 改动。所需包已全部就绪：

| 包 | 版本 |
|---|---|
| numpy | 2.4.4 |
| trimesh | 4.11.5 |
| pyrender | 0.1.45 |
| imageio | 2.37.3 |
| mujoco | 3.7.0 |
| scipy | 1.17.1 |
| lxml | 6.0.2 |
| pillow | 12.2.0 |
| PyOpenGL | 3.1.5 |

说明：
- **无需 GPU、无需 `nvdiffrast` / `pytorch3d`**。CARI4D 的渲染依赖 nvdiffrast + pytorch3d 且贴合真实视频；这里只借鉴其观感，并用 `smplx_min.py`（纯 NumPy LBS）替代 `smplx` 包做默认体型重摆姿，因此不引入任何重依赖。
- pyrender 0.1.45 使用了 NumPy 2.0 已移除的 `np.infty`，脚本顶部已加 `np.infty = np.inf` 兼容 shim（仅代码层，不改环境）。
- 全部离屏渲染，走 **osmesa**（无显示器）：pyrender 用 `PYOPENGL_PLATFORM=osmesa`，MuJoCo 用 `MUJOCO_GL=osmesa`（脚本内已 `setdefault`，命令行再显式传更保险）。

## 用法

默认渲染 `box021_20231011_037`（与现有产物一致）。快速预览加 `--preview N` 只渲染 N 帧。

```bash
cd /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider

# 混合：p2 机器人 + p1 SMPLX 人体 + 物体（两版，交付）
PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa .venv/bin/python \
  workspace/core4d/report/0908/code/viz_mixed_robot_smplx.py

# SMPLX 真值（两版，交付）—— 用 --window 裁到与 g1 一致的段（box021 为 mocap [109,217)）
PYOPENGL_PLATFORM=osmesa .venv/bin/python \
  workspace/core4d/report/0908/code/viz_smplx_reference.py --window 109,217

# 四方法 p2/G1 对比（SPIDER/OmniRetarget/SBTO/GMR，各出 bg+white + 逐帧）
PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa .venv/bin/python \
  workspace/core4d/report/0908/code/viz_methods_compare.py
```

### 四方法对比（`viz_methods_compare.py`）

仅渲染 **p2、仅 G1 机器人 + 物体**，四方法风格一致、逐帧时序对齐，用于并排对比。

- **数据源**（都是 42 列 qpos = 7 base + 29 G1 关节 + 6 物体[trans3+欧拉ZYX]，统一跑在
  `scene_act_E170_lowerbody_physics.xml` 上）：
  - SPIDER：`workspace/core4d/results/E170/s6_downstream/cem/full/E170_<case>_p2_PRG.npz`（论文 box 组 SPIDER-CEM）
  - OmniRetarget：`paper_results/omni_scene_act_qpos/<case>_p2_omnirt_scene_act_qpos.npz`（对比表复算用的同一条）
  - SBTO：`<sbto>/paper_results/_sbto_scene_qpos/<case>_p2_omnirt_scene_act_qpos.npz`
  - GMR：`<GMR>/out/core4d_g1/paper_metrics/qpos/<case>_p2.qpos.npz`（按 30fps）
- **对齐**：以 SPIDER 为参考。时序上，同一 trim 的 SPIDER/OmniRetarget/SBTO 按帧号对齐（lag 0），
  GMR（独立帧、131 帧）按物体轨迹相关性求整数 lag。空间上做 **物体对齐**（yaw+平移把每个方法的
  箱子摆到同一世界位置），机器人各自落在方法产出的相对位置。**GMR 是纯身体重定向、不感知物体，
  机器人手全程距箱 ≥1.4m**，因此对齐后其机器人会明显偏离箱子——这是它接触率≈0 的如实呈现。
- **相机**：共享距离/地面（统一比例），逐帧跟随各方法 robot+object 中点；方位角由 SPIDER 朝向
  推出使机器人正面朝向镜头。
- **输出**：`paper_results/viz/<case>_p2/methods_compare/<method>/{frames_bg,frames_white,<method>_{bg,white}.mp4}`，
  method ∈ `spider/omniretarget/sbto/gmr`。

### 渲染任意序列 / 自定义数据源

用 `--case <object>_<date>_<seq>` 指定序列；`--data-root`、`--smplx-model` 配置数据源：

```bash
PYOPENGL_PLATFORM=osmesa .venv/bin/python \
  workspace/core4d/report/0908/code/viz_smplx_reference.py \
  --case bucket010_20231003_059 \
  --data-root /path/to/CORE4D_Real \
  --smplx-model /path/to/SMPLX_NEUTRAL.npz \
  --out /path/to/output_dir
```

公共参数（`render_style.add_common_args`，三个脚本都有）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--case` | `box021_20231011_037` | 序列 id：`<object>_<date>_<seq>` |
| `--data-root` | `/mnt/.../CORE4D/CORE4D_Real` | CORE4D_Real 根目录 |
| `--smplx-model` | `/mnt/.../smplx/SMPLX_NEUTRAL.npz` | SMPLX_NEUTRAL.npz |
| `--out` | `<viz>/<case>_<primary>` | 输出目录 |
| `--res` | `1080` | 渲染分辨率 |
| `--fps` | `20` | 输出视频帧率 |
| `--preview` | `0` | 只渲染 N 帧均匀采样（0=全长） |

各脚本额外参数：

- **viz_smplx_reference.py**：`--primary`（输出目录人物 tag，默认 p2）、`--window lo,hi`、`--baked-shape`（用原始体型）、`--object-mesh`（覆盖物体 obj 路径）、`--cam-dir`（默认 `-1,0.3,1`，与 mixed 对齐使 p2 面向镜头）、`--yfov-deg`（默认 45）、`--dist-mult`（默认 1.15）。
- **viz_mixed_robot_smplx.py**：`--primary/--partner`（默认 p2/p1）、`--exp`（默认 E170）、`--cem-dir`、`--processed-root`、`--scene-prefix` + `--window lo,hi`、`--baked-shape`、`--p1-trim-start`（默认 109）、`--p1-to-p2`（默认 15）、`--cam-az-offset`（默认 45，相机方位角=p2朝向+此偏移）、`--cam-el`（默认 12）。
- **viz_methods_compare.py**：`--case`、`--scene`（默认 E170 单机器人场景）、`--out`、`--res/--fps/--preview`、`--az`（默认按 SPIDER 朝向推出）、`--az-offset`（默认 45）、`--el`（默认 12）。
- **viz_dual_robot_clean.py**（已不作交付）：`--primary/--partner`（默认 p2/p1）、`--exp`（默认 E170）、`--cem-dir`、`--processed-root`、`--scene-prefix`（默认 `dcv3_omnirt_v1_ref_fk`）。

> ⚠️ `--p1-trim-start` / `--p1-to-p2` 是 **paired-export 特定** 的对齐偏移（当前 E170 box021
> 序列的 p1↔mocap、p1↔p2 时间对齐）。换序列时需按该序列的配对导出重新确定，否则人体
> 与机器人/物体会错位。

## 数据布局（预期）

`--data-root`（CORE4D_Real）：
```
human_object_motions/<date>/<seq>/person1_poses.npz   # 含 vertices/joints/betas/body_pose/...
                                  person2_poses.npz
                                  smooth_objposes.npy   # (T,4,4) 物体位姿
object_models/<category>/<object>_m.obj                # category = object 去掉末尾数字，如 box021→box
```

机器人 / 混合脚本额外依赖 SPIDER 重定向产物：
```
<repo>/workspace/core4d/results/<exp>/s6_downstream/cem/full/<exp>_<case>_<person>_PRG.npz
<processed-root>/<scene-prefix>_<case>_<person>/scene_act_<exp>_lowerbody_physics.xml
```
因此机器人/混合脚本只能渲染 **已完成重定向 + CEM** 的序列；纯 `viz_smplx_reference.py`
对任意有 mocap 的 CORE4D 序列都可直接渲染。

## 输出

默认写到 `<viz>/<case>_<primary>/`（`<viz>` = `workspace/core4d/report/0908/paper_results/viz`）：

```
robot_frames_bg/   robot_frames_white/     robot_dual_bg.mp4      robot_dual_white.mp4
smplx_frames_bg/   smplx_frames_white/     smplx_reference_bg.mp4 smplx_reference_white.mp4
mixed_frames_bg/   mixed_frames_white/     mixed_robot_smplx_bg.mp4  mixed_robot_smplx_white.mp4
```

SMPLX 脚本先渲染到本地 `/tmp/smplx_stage_<case>/` 再批量拷回（网络盘逐帧写会卡）。
```
