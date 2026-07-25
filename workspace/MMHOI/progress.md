# MMHOI 当前进度

## 2026-07-25：R003 / E004 稀疏 RGB 视频 QC

### 已完成

- 从完整解压目录选择 3 个 `C_2+box` 和 2 个 `C_8` 代表 case；
- camera-0 RGB 覆盖为 385/385，每个 case 的 frame-id gap 全为 30；
- 生成 5 个原分辨率 1 fps H.264 视频，一张 sparse RGB 对应一帧；
- 为用户示例 `20240412_personA_personB/C_8` 额外生成 30 fps 快速预览：
  91 输入/输出帧、3.034 秒、无插值；
- 按用户反馈追加同一 91 帧序列的 10/15/20 fps 版本，时长分别为
  9.100/6.067/4.550 秒，均保持 91 输入帧 = 91 输出帧；
- 将 10 fps 无插帧视频扩展到全部 24 个 `C_2/C_8` capture：
  `C_2=12 case/636帧`、`C_8=12 case/653帧`，合计 1,289 帧、128.9 秒；
- 全量 batch 的 24 个 MP4 均通过 ffprobe：H.264、2048×1536、10/1 fps，
  每个 case 的 input/output frame count 完全相等；
- 已从 24 个视频各解码一张中间帧，并人工检查 `C_2/C_8` 两张 12-case
  总览；`C_2` 可见 box 搬运，`C_8` 可见 chair/table 操作；
- 生成 per-frame TSV、render/run manifest 和中间帧总览；
- 5 个视频工具单元测试和 `py_compile` 通过；
- ffprobe 验证每段视频帧数、帧率、分辨率和时长。

### 结论口径

- 1 fps 版本用于按真实 sparse 时间尺度判断动作可读性；
- 30 fps 版本只是将稀疏帧加速 30 倍，不是 30 Hz motion reconstruction；
- 全量 10 fps 版本同样是 10 倍快速预览，不改变约 1 Hz source 事实；
- E004 的 RGB visual QC 子项完成，human/object representation 全量 gate
  仍待执行。

### 产物

```text
workspace/MMHOI/results/E004/s0b_sparse_rgb_visual_qc/
workspace/MMHOI/log/03_E004_sparse_rgb_video_qc.md
```

## 2026-07-25：R002 / E002 `C_2/C_8` 范围修订与时间审计

### 已完成

- 用户范围已冻结为 `C_2/C_8`（Moving heavy stuffs），旧的
  `C_9/C_10/C_9_r2` production scope 已取消。
- 首批实验固定为 `C_2 + active box`。
- Inventory 已扩展主范围、box pilot 和 archive temporal audit，10 个
  `unittest` 与 `py_compile` 通过。
- 完整 ZIP 已重新解析；E002 机器证据写入
  `results/E002/s0_scope_inventory/`。
- `C_2/C_8` 实数：24 captures、14 sequence roots、1,289 samples、6 类
  active objects；split 为 602/308/314/65。
- 首批 box 实数：460 active samples、12 captures；split 为
  223/79/116/42；365 个 box cooperative samples。
- 已临时解压
  `20240412_personA_personB/20240412__C_2__30skip`：12 个 frame folders
  的相邻 id 全差 30，逐 sparse frame 各有两人 JSON 和 `final/box.ply`，
  没有目录外连续 trajectory。
- 完整 archive 的 `C_2/C_8` 也得到 `gap 30: 1,265/1,265`；1,289 个
  sample 均有两人 JSON 和 final object mesh，whole archive 无
  `.npy/.npz/.pkl`、视频或 mocap 文件。
- 结论已冻结：Kinect capture 是 30 fps，但 released SMPL-X/object GT
  是 30skip、约 1 Hz。
- 已重写 `data_stat.md`，新增 v1 全局方案
  `plan/03_v1_c2_c8_box_global_adaptation_plan.md` 和 R002 日志。

### 当前状态

```text
S0 scope/inventory:          pass
S0b sparse representation:  probe pass, full gate pending
S0b 30 Hz temporal source:  blocked_temporal_density
S1-S6 production:           not started
```

### 下一步

1. E003 获取未 skip 的官方 30 Hz GT，或可独立验证的 30 Hz source；
2. E004 先完成 sparse human-world / box 6DoF 全量 gate；
3. E003/E004 同时 pass 后，构建第一个完整 `C_2+box` dense window。

### 风险

- 460 个 active-box samples 是跨 12 captures 的稀疏标注点，不能拼接成
  连续轨迹。
- `capture_fps=30` 不等于 `released_gt_fps=30`。
- 仅从 1 Hz 做 linear/SLERP 插值不得进入 production。

## 2026-07-25：R001 Collaborative work 数据统计与适配设计

### 已完成

- 完整读取 `experiment-planning-zh` 与 `data-construction-v3-zh` 的 `SKILL.md`。
- 因输入包含论文 PDF 且输出是技术调研文档，选用 `pdf`、`deep-reading-analyst`、`markdown-mermaid-writing`；已完整读取 PDF 与 Markdown/Mermaid 主说明，正在补读 deep-reading 尾部和本任务必需的参考页。
- Deep-reading 采用面向技术落地的组合：SCQA/5W2H 定义论文与数据要回答的问题，critical thinking 区分论文规模声明与本地实数，inversion/systems thinking 用于适配风险与阶段依赖。
- 已完整读取上述 deep-reading 参考框架及 Markdown 样式规范；交付中将把论文表述标为“论文声明”，把磁盘枚举标为“本地审计”，避免统计口径混淆。
- 已完整读取 Mermaid 样式、flowchart 指南和项目文档模板；全局方案将用一个可访问的阶段总览图配合阶段 contract 表，不用图替代精确 gate 描述。
- 已定位 Core4D v3 的规范文档、当前 tracker/progress 与 E161 后续修订；适配基线不能把 E161 的 aggregate CEM 结论直接当成 RL-safe 默认，MMHOI 方案必须保留 per-case raw contact 和 S6 RL 独立证据。
- 已完整复核 Core4D v3 `README.md` 与 `02_pipeline_stages.md`：MMHOI 适配应复用阶段语义和状态隔离，但 S1 原始 reader、S2 object scene、S3 source conversion 必须按 MMHOI contract 重写/扩展，不能仅改 raw root 名称。
- 已完整复核 manifest/schema、template、retarget、diagnostic、completion/release、hand-collision 规范；MMHOI registry 需要在 Core4D case 字段之外增加 dataset split/activity/take、双人 subject identity、多人/多物体 slot 与 source-frame provenance。
- 已读取 Core4D tracker/progress。当前可继承的工程默认应以 E170–E179 已冻结/复核的 `omnirt_v1→v2 rescue + ref_fk + rubber_hull + PRG` 证据为参考，而不是仅按分支名把 E161 reward 直接设为 MMHOI production 默认。
- 核验启动分支为 `experiment/E161-surface-release-ablation`，commit 为 `67cef0b84d81107128a082f36804caa16a15251c`。
- 创建并切换到 `experiment/MMHOI-data-adaptation`。
- 确认论文文件存在于 `paper/Kogashi 等 - 2025 - MMHOI Modeling Complex 3D Multi-Human Multi-Object Interactions.pdf`。
- 发现并保留启动前已有的用户工作树改动：
  - `workspace/core4d/log/232_E172_box004_screening_full_cem.md`
  - `workspace/core4d/log/233_E173_box024_box023_box001_screening_full_cem.md`
  - `workspace/core4d/scripts/data_construction_v3/stages/s6_downstream/finalize_reused_partner_rl.py`
  - `workspace/exp_MMHOI.md`

### 当前步骤

- 读取论文、MMHOI 原始目录和 Core4D v3 相关 contract。
- 建立可复跑的 Collaborative work inventory/stat 口径。

### 数据审计首轮发现

- 原始目录可读，但本地占用仅约 `23M`，共见 `36 .ply / 10 .json / 7 .csv / 61 .jpg / 4 .png / 2 .obj / 1 .md`；这显然不是论文所述约 60 万帧四相机 RGB-D 全量影像落盘，需要确认它是官方 annotation/sample 子集还是未完整下载。
- 顶层包含 `object/`、`sequences/`、`splits/`；`object/` 可枚举 22 个编号物体 mesh，和论文的 22 类一致。
- `sequences/` 当前有多个采集日/人员组合目录；首个目录明确出现 `C_1...C_10` 命名，需要通过 README/JSON/CSV 确认 `C` 是否对应 Collaborative work，而不能仅按首字母猜测。
- 论文声明：12 场景、约 60 万帧、13 名参与者、22 个物体、78 个 human-object action classes；supplement 将 Collaborative work 划为 5 个 scenario。上述是论文全数据口径，不等于本地协作子集统计。
- 在同级路径发现完整归档 `/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/MMHOI_release.zip`，文件大小 `93,682,766,723` bytes；README 声明解压约 `173 GB`。后续以 ZIP central directory + 归档内 split/action CSV 做完整 inventory，无需先解压 173GB。
- 当前 `MMHOI/` 目录只有一个 `20240412_personA_personB/C_5` 样例（2 个 frame folder、98 个序列文件），另外 14 个 sequence 目录为空；因此该目录不能直接代表完整数据量。
- 完整 ZIP central directory 可正常读取：`782,626` entries；归档内存在 `MMHOI/splits/train_val_test_split.json`。
- action 标注是 frame-folder 级 `PARAM/action.csv`，每行语义为 camera/body bbox + person + object + verb；`final/C_X.csv` 是 interaction body-part/object-pair 标注，不能与 action 行数混为一种“序列数”。
- 环境没有 `jq`，首轮 JSON key 检查失败 1 次；后续使用 Python 标准库读取，不安装新依赖、不重复同一失败命令。
- 归档 split 与 central directory 对齐：`15` 个 sequence 根、`135` 个实际 scenario capture 目录、`8,071` 个标注 sample/frame folder、`8,071` 个 `PARAM/action.csv`、`6,796` 个 `final/C_X.csv`。论文“约 60 万帧”与这里的 sample folder 是不同层级，最终文档会同时给出定义。
- 已读取本仓库 common/python 规则；统计器将仅用 Python 标准库、完整类型标注和显式输入校验，并提供测试与可复跑输出。
- 按 TDD 先写 inventory 单元/小型 ZIP 集成测试；尝试 `python3 -m pytest` 失败 1 次，原因是当前 Python 未安装 `pytest`。不安装新依赖，测试改用标准库 `unittest`，避免重复该失败命令。
- `unittest` RED 阶段已按预期完成：测试因 `inventory_mmhoi` 尚未实现而失败；下一步实现最小统计器使其转绿。
- 已实现只读 ZIP inventory 统计器 `scripts/data_inventory/inventory_mmhoi.py`；8 个标准库 `unittest`（含小型 ZIP 集成测试）全部通过。
- 首次真实运行失败 1 次：`20240412_personA_personC/C_1` 在归档有 `150` 个 sample，而官方 split 声明 `75+30+35=140`。这是真实数据/split 不一致，不能丢弃 10 个样本；修复策略是保留全部 archive sample，把超出 split 的部分标为 `unspecified`，并输出 split mismatch audit。已先添加对应回归测试。
- split overflow 回归测试已完成 RED；实现已改为保留 archive authority 并单独记录 `split_mismatches.tsv`，不再因 split 声明偏小而丢样本。
- split overflow 修复后 9 个测试转绿；第二次真实运行又发现独立问题：归档含 `20240510_personH_personI/C_2`，但官方 split JSON 完全缺该 key。第 2 次运行按硬校验停止。下一策略：缺 key 的 archive samples 全部标 `unspecified`，audit 原因标 `missing_split_key`，不重复现有失败路径。
- missing split key 回归测试已完成 RED；准备实现显式 `missing_split_key` audit 分支。
- 已实现 missing-key 保留策略；10 个测试与 `py_compile` 全部通过。下一次真实运行将同时审计 count mismatch 与 missing key。
- 第三次真实 inventory 成功：完整解析 `8,071/8,071` 个 action CSV；Collaborative work 得到 `55` 个 scenario capture、`3,452` 个 annotated sample、`10` 个 active object types。机器可读证据写入 `results/E001/s0_inventory/`。
- 协作细分已复核：`C_2/C_8/C_9/C_10` 各 12 个 capture，`C_9_r2` 7 个，共 55；sample 分布分别为 `636/653/790/742/631`。动作证据与场景映射一致：C2 为 box/stool/suitcase 协作搬堆，C8 为 chair/table 协作搬堆，C9/C9_r2 为 desk/keyboard/monitor meeting，C10 为 backpack/suitcase 搬运/传递。
- 完整归档 Collaborative work：sequence 目录名可辨识 12 个身份，2 人 sample `2,821`、3 人 sample `631`；20,678 条 person-object action row，11 个原始 verb 字符串（含 `no-interaction`），48 个 `(object, verb)` 类，其中 active 为 38 类；1,004 个 sample 存在至少两人同物体的 `* together` 标签。
- split audit 共 8 个不一致，其中 4 个影响 Collaborative work；最终 split 为 train `1,660` / val `848` / test `858` / unspecified `86`。这些 86 个不会被静默并入 train/test。
- 已核验严格双人口径：Collaborative work 的 `3,452` 个 sample 中，`2,821` 个为实际双人 sample（`C_2/C_8/C_9/C_10`），`631` 个为三人 `C_9_r2`。后续 production scope 使用前者，`C_9_r2` 单列为多人扩展 backlog。
- 时间轴审计完成：55 个 capture 内共有 `3,397` 个相邻 sample 间隔，全部恰为 `30` 个源 frame id。结合论文 30 fps，发布标注的有效采样率约为 1 Hz，不能直接冒充 Core4D/OmniRetarget 使用的 30 Hz 连续 mocap。
- 对象表示探针完成：10 类 Collaborative work 物体各抽取一个 archive sample，将 `object/<id>_<name>.ply` 与 `final/<name>.ply` 按同序顶点做刚体配准，RMS 为 `8.6e-9–6.8e-8 m`、scale≈1，可精确恢复逐帧 6DoF；后续必须将变换与残差写入 manifest，不能只消费 final mesh。
- 人体表示探针完成：`PARAM/person*.json` 包含 `pose_53(159)`、`pose_22(66)`、`betas/betas_new(10)`、`j3d_127(381)`；其中 j3d 与 camera-0 mesh 同坐标。抽取样例中 `0.person*.ply → final/person*.ply` 的逐顶点刚体配准 RMS 为 `3.9e-8/5.0e-8 m`，可用该变换把 `j3d_127` 送入统一 Y-up 世界坐标，再复用 Core4D converter 的前 22 joints → OmniRetarget contract。
- Core4D converter contract 已定位：Holosoma commit `04ca515d7030916cccc6966d04cb0b3b6577d18a` 的 `convert_core4d_to_omniretarget.py` 消费 `(T,127,3)` joints 和 `(T,10)` betas，输出 Z-up `global_joint_positions(T,22,3)`、height、`object_poses(T,7)`；MMHOI 需要 dataset-specific S1/S3 adapter，不能只替换 raw root。
- 已生成 `data_stat.md`：以严格双人 2,821 samples 为主口径，另列完整 Collaborative work 3,452 samples、动作/物体/split/时间/表示审计及复跑命令。
- 已生成 `plan/02_v0_global_adaptation_plan.md`：固定工作区、source/case/variant contract、S0/S0b/S1–S6 输入输出与 hard gates、canary 顺序、实验编排、release gate 和 Definition of Done。
- 已生成 `log/01_v0_dataset_adaptation.md`，并将 R001/E001 tracker 状态更新为完成；E002 固定为全量 representation audit + dense source availability。
- 最终验证通过：10 个 `unittest`、`py_compile`、TSV/JSON 数字一致性、12 个 Markdown 内部链接、Mermaid/fence、尾随空白、`git diff --check` 和 branch merge-base。

### 风险/错误

- 用户给出的解压目录可读但只含一个完整 `C_5` 示例；完整统计必须继续以同级 `MMHOI_release.zip` 为 archive authority。
- MMHOI release 的约 1 Hz 标注不足以直接形成 30 Hz 控制/reference 轨迹。取得未 skip 的连续标注、或用 canary 证明重建方案满足速度/接触连续性 gate 之前，S3 production 必须保持 blocked。
- `PARAM/j3d_127` 不是最终世界坐标，人体 world transform 要从 camera-0/final 同拓扑 mesh 恢复并做残差/朝向/地面审计。
- `workspace/exp_MMHOI.md` 是启动前已有的未跟踪文件；本轮不修改，避免覆盖用户内容。
