# CORE4D experiment scripts 归档目录

这个目录是实验局部脚本和 manifest 的目标结构。

当前 Phase 6 策略：

- `legacy/E###` 存放 E081 及以前的历史实验归档。
- E001-E081 的原 `workspace/core4d/scripts/E###` 路径已删除；需要查看历史脚本时使用 `workspace/core4d/scripts/experiments/legacy/E###`。
- `E082+` 目录是结构副本；旧的 `workspace/core4d/scripts/E###` 真实目录仍保留，作为兼容执行入口。
- 很多复制过来的脚本仍通过 `Path(__file__).resolve().parents[...]` 计算 repo root；如果直接从更深的 `experiments/E###` 路径运行，可能改变这个计算结果。

在某个脚本迁到 `workspace/core4d/scripts/eval/runners`、
`workspace/core4d/scripts/eval/reports`，或者改用
`workspace/core4d/scripts/common/paths.py` 之前，E082+ 仍建议通过旧兼容路径
`workspace/core4d/scripts/E###` 运行。
