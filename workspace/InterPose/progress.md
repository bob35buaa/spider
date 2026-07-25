# InterPose 当前进度

_R001 · 2026-07-25_

---

## 📍 当前状态

- 已从 `experiment/E161-surface-release-ablation` 的提交
  `67cef0b84d81107128a082f36804caa16a15251c` 创建
  `experiment/InterPose-data-adaptation`
- 使用独立 linked worktree `/home/ubuntu/Workspace/spider-interpose`，
  避免带入原工作树 `experiment/MMHOI-data-adaptation` 的未提交改动
- R001 全量审计、代码、统计报告、适配方案和结果日志均已完成
- 当前结论：发布包 `confirmed H-O-H = 0`，后续从 W0 来源/许可和原视频
  joint-support pilot 开始
- 下一步：完成 Markdown/link/代码/git 检查后提交并推送

## 🔍 已确认约束

- 只关心两个人共同与同一个物体交互的时序数据
- 本地实际统计优先于论文规模
- 目标数据必须能进入 SPIDER 数据构建 gate，而不只是生成模型训练样本
- 本轮为纯分析，不触发 scene 快照、训练或可视化强制项

## 📊 初步数据审计

- 数据根目录约 `21G`，包含 `Charades/`、`hdvila/`、`kinetics/`、
  `online_video/` 和 `texts/`
- 文件总数 `147636`：`73818` 个 `.npz` 与 `73818` 个 `.txt`；
  尚需验证 basename 是否严格一一对应
- 抽样 NPZ 的统一键为 `poses`、`trans`、`betas`、`num_betas`、
  `gender`、`mocap_frame_rate`、`text`
- 抽样 `poses` 为 `(T, 165)`、`trans` 为 `(T, 3)`，每个 NPZ 只显式保存
  一套 SMPL-X 人体时序；未见 object pose、object mesh、object identity 或
  第二个人体数组
- 这意味着“双人协作”不能仅凭单个 NPZ schema 直接成立；需继续检查同源
  clip/track 的配对关系、caption 与构建代码导出命名
- 论文 PDF 位于原工作树 `paper/`，E161 基线未跟踪该文件；将以只读方式引用
- 官方数据构建仓库说明其链路基于 WHAM 与 HaMeR，并支持网络检索/下载、
  Qwen-VL 筛选、人体/手部估计与文本标注
- 论文 Table 1 报告 `73814` clips、`15.7M` frames、`148.74h`，而本地为
  `73818` NPZ；最终文档必须解释 `+4` 的版本/打包差异，不能强行对齐
- 论文明确写出只提取人体运动、不提取物体运动；multi-person collaboration
  出现在 HOI-Agent 应用章节，不能当作本地数据的成对标注
- 论文说明多人视频会按先前 ViTPose++ bbox 裁剪后“逐人”生成文本；这支持
  同一视频片段可能存在多个独立人体 track，但不保证跨 track 的共同坐标、
  同步范围或共享物体 identity
- 本地 `.txt` 与 NPZ 内 `text` 抽样一致；`.txt` 还附带 NLP token/词性和
  两个 `0.0` 字段，未看到论文所述 action/object class 作为结构化发布字段
- 抽样 caption 能提及另一个人（例如 conversation/exchange），但这不是
  “两个人共同搬运同一物体”的充分条件
- 构建代码 `post_process()` 把结果键命名为
  `<person_idx>_<segment_idx>`，AMASS 导出文件名保留该后缀；因此发布文件
  最后两个整数可恢复 person/segment ID，同一前缀可组成 clip group
- 导出 NPZ 不保存 `frame_ids`、bbox、原视频路径或 track provenance；
  即使同一 clip group 有两个人，也无法从发布包恢复两段的原始时间对齐
- WHAM 对同一视频的所有 track 估计 world translation，理论上同源 track
  共享 SLAM 坐标框架；但发布包缺少 frame ID，必须重新处理原视频才能验证
  两人的同步和共同物体关系
- 已先写统计核心单元测试，覆盖 exporter 后缀、HumanML3D 文本行解析、
  object/action/relation label 提取和三档 pair candidate 分类
- TDD 红灯已确认：实现前测试因缺少 `audit_interpose` 模块而 collection fail
- 初版实现后发现 `passes` 未触发 relation cue；改为同时使用 VERB lemma，
  现有 `7/7` 单测通过
- 首轮覆盖率只有 `40%`，尚未满足本地规则；下一步增加小型合成 release
  integration test，覆盖 NPZ header、全量 CLI、聚合与落盘
- 合成四来源 integration test 已覆盖：正确/错误 exporter suffix、
  text missing、同 clip 双 track、共享 ball + relation cue、输出 TSV/JSON
- 测试中先暴露测试文件漏导入 `json`，修复后 `8/8 PASS`，auditor
  statement coverage=`91%`
- `ruff` 首轮报告 43 个可自动修复项（import、docstring spacing、format）；
  已全部机械修复并通过 lint/format check
- formatter 将内置 fallback vocabulary 展开后主脚本达到 `1219` 行，
  超过本地单文件上限；功能验证后需拆分 core/CLI，不能以当前形态收尾
- R001 全量扫描完成：`73,818` 个 NPZ、`15,623,848` 帧、
  `147.5008556 h`、`21,206,081,784 bytes`，`73,818/73,818` schema OK
- 文本严格配对为 `73,817 MATCH + 1 TXT_MISSING`；唯一缺失项是
  `kinetics/kinetics_lb8oht6c1rE_part_106_20_0.npz`
- 四来源序列数为 Charades `6,974`、HD-VILA `10,044`、Kinetics
  `39,466`、Online video `17,334`；相较论文 `73,814`，本地多 `4`
  条（HD-VILA `+2`、Kinetics `+2`）
- 文件名重建得到 `29,673` 个 clip groups，其中 `12,678` 个多 track
  group、`5,450` 个恰含两个 person ID；全部可能 person pairs 为
  `181,434`
- 文本筛选得到 `5,890` 个 strict pair、`11,388` 个仅共享物体 pair，
  合计 `17,278`；其余 `164,156` 个 pair 无共享物体文本证据
- strict 文本 cue 仍有高误报风险：例如 `pass` 会命中 “passing by a
  table”；即使补充要求“transfer cue + 显式另一人”或“joint cue +
  显式另一人”，也只得到 `725` 个高优先级筛选 pair（`263` groups），
  不能升级为真值
- 发布包仍缺少 object identity/mesh/SE(3)、原始 `frame_ids` 和跨 track
  同步证据，因此最终结论保持 `confirmed_target_hoh_count = 0`
- 实际对象文本标签 top-10：ball `9,983`、table `3,691`、racket
  `3,595`、box `3,163`、chair `1,952`、bag `1,288`、door `1,225`、
  tool `1,135`、bat `991`、instrument `845`
- 实际动作文本标签 top-10：move `38,929`、hold `29,944`、bend
  `21,252`、position `15,043`、adjust `13,168`、turn `8,130`、place
  `6,912`、pick up `6,126`、lift `3,999`、control `3,769`
- 审计器最终拆成 CLI（163 行）、core（611 行）、output（108 行）、
  summary（233 行）和 vocab（286 行），所有文件均低于 800 行
- 拆分后 `ruff check/format`、`py_compile`、`8/8 pytest` 全部通过；
  聚合 statement coverage 从 `91%` 提升为 `93%`
- `scan_manifest.json` 的 `auditor_sha256` 已改为五个审计模块的联合哈希，
  并新增 `auditor_modules` 列表，保证 provenance 覆盖实际统计逻辑
- 首次拆分后全量复扫于 `12:43:31 +08:00` 完成，仍得到 `73,818`
  sequences、`12,678` multi groups、`17,278` shared-object pairs
- 为避免把 `pass` 等单一 cue 称为强证据，先添加 integration 断言并确认
  RED（缺少 high-priority 字段），再实现“显式第二人 cue + transfer/joint
  cue”的二级筛选；GREEN 后 `8/8` tests、coverage `93%`
- 二级筛选属于优先人工复核队列，不改变任何 `confirmed=0` 判定
- 最终代码审查进一步拆出 summary/provenance/record builders；production
  文件 `<800` 行、函数 `≤50` 行，8/8 tests 与 coverage `93%`
- 最终全量复扫于 `13:06:36 +08:00` 完成；summary invariants、6 个 TSV
  行数、来源总和和 manifest/current auditor hash 全部断言通过
- 已完成 `data_stat.md`：本地总量、来源、schema、FPS、动作/物体排名、
  multi-track 分布、候选分层和直接适配阻塞项
- 已完成 `adaptation_plan.md`：W0–W5 原视频 H-O-H 重建、SPIDER S0–S6、
  registry/role axes、双人到同一 mesh 的 3cm/5cm contact、非 box review、
  Pilot 0–3 与风险停线条件
- 已完成 `log/01_v0.1_dataset_audit.md` 并将 R001 tracker 状态更新为完成
- Markdown 结构/link/footnote/Mermaid accessibility 自检通过：6 份文档均
  只有一个 H1、H2 emoji 合规、内部链接存在、脚注已引用
- 使用 Mermaid CLI 实际渲染 `data_stat.md` 和 `adaptation_plan.md`，
  两张 SVG 均成功生成；5 个外部 arXiv/GitHub URL 均返回 HTTP 200
- 首次 Markdown 检查脚本漏导入 `Counter` 导致 `NameError` 1 次；补导入后
  同一检查通过，没有重复失败
- 代码审查发现 `_build_summary` 超过 50 行；已提取
  `interpose_audit_summary.py`，同时拆分 provenance、sequence builder 和
  synthetic fixture，最终所有生产文件 `<800` 行、所有函数 `≤50` 行
- 最终复扫时间 `13:04:21–13:06:36 +08:00`；五模块联合 hash 为
  `6c1ec35a89d6da776002925396c4d0700678ce5abea50fedb4dff25c4d019b93`
- 最终验证：Ruff、format、py_compile、8/8 pytest、93% coverage、summary
  invariants、TSV row counts、Markdown links/footnotes/Mermaid contract 全通过
- 根 `.gitignore` 的通用 `data` 规则会忽略 `scripts/data/`；已仅对 6 个明确
  审计源码/测试文件使用 `git add -f`，results/cache/pyc 仍保持忽略
- Bandit 静态安全扫描为 `0 high / 0 medium / 3 low`；3 个 low 均指向
  `subprocess.run(["git", "-C", validated_repo, "rev-parse", "HEAD"])`，
  无 shell 或字符串命令拼接，作为只读 provenance 调用接受

## ⚠️ 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
| --- | ---: | --- |
| E161 worktree 内找不到论文 PDF | 1 | 从原工作树的只读 `paper/` 路径读取，不复制二进制 |
| 首次 `rg` 命令 shell 引号未闭合 | 1 | 改用双引号包裹 regex 后成功定位 NPZ 导出代码 |
| 当前 Python 环境没有 `pytest` 命令/模块 | 1 | 使用已安装的 `uv run --with pytest --with pytest-cov` 创建临时测试环境 |
| 全 home 搜索 pytest 可执行文件耗时过长 | 1 | 已中止搜索；不再扫描无关目录 |
| Markdown 检查脚本漏导入 `Counter` | 1 | 补充 `collections.Counter` 后检查通过 |
