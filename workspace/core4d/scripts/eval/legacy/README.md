# eval 历史归档

本目录存放 `workspace/core4d/scripts/eval/` 根目录中 E142 及以前的评测、审计、诊断和抽帧脚本。

Phase 7 迁移规则：

- E142 及以前视为历史文档归档，原 `scripts/eval/<file>` 路径删除。
- E143 及以后视为仍可能被复现实验引用的活跃入口，真实实现迁到 `scripts/eval/runners/` 或 `scripts/eval/wrappers/`，原根路径保留 thin wrapper。
- 结果目录 `workspace/core4d/results/E###` 不随本迁移移动。

如果需要恢复某个历史根路径，优先通过 `git revert <phase-7-commit>` 回退本阶段，或显式新增一个兼容 wrapper。
