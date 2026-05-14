# Git 相关记录

## 2026-05-14：从当前分支移除已跟踪的 Core4D 结果产物

### 背景

仓库的 `.gitignore` 已经包含下面这条规则：

```gitignore
results/
```

但在这次清理之前，`workspace/core4d/results/` 下有一些生成结果已经被加入
git 跟踪。`.gitignore` 只会阻止未跟踪的新文件被加入 git；对于已经 tracked
的文件无效。

当时的问题是：已经 push 到远程的实验结果中包含了大文件产物，尤其是
`workspace/core4d/results/E060*` 下的 `.mp4`、`.npz`、`.jpg`、`.png`
文件。

### 已执行操作

从当前分支的 git 索引中移除了 `workspace/core4d/results/` 下所有已跟踪的
以下扩展名文件：

```bash
git rm --cached --ignore-unmatch -- \
  ':(glob)workspace/core4d/results/**/*.mp4' \
  ':(glob)workspace/core4d/results/**/*.png' \
  ':(glob)workspace/core4d/results/**/*.jpg' \
  ':(glob)workspace/core4d/results/**/*.npz'
```

生成并推送了以下提交：

```text
11d80bf chore: stop tracking core4d result artifacts
```

推送目标：

```text
origin/feat/dual-robot-retarget
```

远程分支更新记录：

```text
4c66f83..11d80bf  feat/dual-robot-retarget -> feat/dual-robot-retarget
```

### 影响范围

这次清理从当前分支删除了 124 个已跟踪的结果产物文件。涉及的扩展名是：

- `.mp4`
- `.png`
- `.jpg`
- `.npz`

这次提交只包含上述结果产物的删除记录，没有把其它本地未提交改动一起提交。
保留在本地工作区中的未提交改动包括实验文档、脚本改动，以及部分
`scene_snapshot` XML/JSON 文件的本地删除状态。

### 验证方式

提交并推送后，下面这个命令已经查不到任何 tracked 文件：

```bash
git ls-files -- \
  ':(glob)workspace/core4d/results/**/*.mp4' \
  ':(glob)workspace/core4d/results/**/*.png' \
  ':(glob)workspace/core4d/results/**/*.jpg' \
  ':(glob)workspace/core4d/results/**/*.npz'
```

### 重要限制

这次清理只会让这些文件从当前分支的最新版本中消失，因此 GitHub 当前分支的
普通文件视图不会再显示它们。

但这不会从旧提交或 git object 历史中彻底删除这些文件。也就是说，历史提交中
仍然可能包含这些大文件 blob。

如果需要彻底清理历史 blob，需要使用 `git filter-repo` 或 BFG 之类的工具重写
历史，然后 force push。这个操作会改写共享历史，执行前必须协调好。
