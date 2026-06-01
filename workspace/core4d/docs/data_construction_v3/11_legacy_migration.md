# 11 legacy 迁移

v3 不删除旧目录，也不继续往旧目录写结果。

## legacy 目录

| 目录 | 定位 |
|---|---|
| `${HOLOSOMA_REPO}/workspace/v3/data_construction` | legacy v1，包含旧 inventory/raw-contact/Stage2b 脚本和历史结果 |
| `${HOLOSOMA_REPO}/workspace/v3/data_construction_v2` | legacy v2，包含 E091-E107 工作区和结果 |
| `workspace/core4d/data_preprocess` | legacy Stage2b 入口，包含 E077-E080 pipeline |

## 默认规则

- 不删除；
- 不改写；
- 不隐式读取；
- 不作为新流程 output root；
- 不作为 canonical 入口。

## 允许使用 legacy 的方式

### 显式 import

```text
legacy path -> imported snapshot -> v3 registry
```

manifest 必须记录：

- original path；
- import script；
- import time；
- imported files；
- known risks；
- `source_type=legacy_import`。

当前入口：

```bash
workspace/core4d/scripts/data_construction_v3/migration/import_legacy_snapshot.py \
  --input-tsv "$LEGACY_OR_MANUAL_STATE_TSV" \
  --out-dir "$RUN_DIR/imported_snapshots" \
  --snapshot-id <snapshot_id> \
  --legacy-root "$LEGACY_ROOT" \
  --known-risk "manual review required before treating as positive"
```

该命令不会直接修改当前 registry，只会写入：

```text
imported_snapshots/<snapshot_id>/
  <input basename>
  imported_case_state_registry.tsv
  imported_case_state_registry.json
  import_manifest.json
  import_manifest.md
```

审查通过后，再显式合并：

```bash
workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
  --registry-dir "$RUN_DIR/registries" \
  --input-tsv "$RUN_DIR/imported_snapshots/<snapshot_id>/imported_case_state_registry.tsv"
```

`imported_case_state_registry.tsv` 中的 `source_ref` 默认保留 legacy root 或原始输入路径。除非确实要覆盖 provenance，否则合并时不要再传 `--source-ref`。

### Canonical existing cases seed

已确认可信的历史结果整理为：

```text
workspace/core4d/data_construction_v3/existing_cases.tsv
```

这份 seed 是 v3 `case_state_registry` schema，可直接走上述 `import_legacy_snapshot.py -> update_case_state_registry.py` 流程。它只纳入可复查的真实结果：Box004 正例、E105-E107 clean 结果，以及 E079-E081 的 Box023/Box025 legacy CEM cache。E103 template/inertial bug 修复前的 Box021/Box026 污染结论不作为可信 seed。

### 显式 legacy path

短期必须直接引用旧路径时，必须在 command/config/run manifest 中暴露，不允许代码内隐式查找。

## `workspace/core4d/data_preprocess`

可迁移内容：

- trim window 工具；
- target scene patcher；
- basic verify；
- contact mask proxy 逻辑。

必须修正后才能成为 v3 长期入口：

- 输出 root 参数化；
- contact mask 3cm/5cm 正确命名；
- `REPLACE_WRIST_WITH_FINGERTIP` 变为显式 variant；
- 补 E103 inertial/collision audit；
- 增加 registry/stage manifest；
- 不再使用旧 `cases_*.tsv` 作为自动候选源。
