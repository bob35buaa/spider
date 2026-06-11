# CORE4D launch scripts

本目录整理 `workspace/core4d/scripts/` 根目录中的实验启动、远程回收和远程同步入口。

- `active/`：E143 及以后仍可能被当前复现实验引用的 launch/pull 入口。原根路径保留 thin wrapper。
- `legacy/`：E142 及以前的历史 launch/pull/sync/wait 入口。原根路径删除，只保留 git-tracked 归档文件。

结果目录 `workspace/core4d/results/E###` 不随本迁移移动。
