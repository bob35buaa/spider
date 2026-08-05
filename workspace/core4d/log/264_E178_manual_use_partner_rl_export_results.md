# E178 最终人工 USE + partner RL 资产导出结果

日期：2026-08-05  
实验：E178  
阶段：Phase 41 S6 manual-authority RL export / Holosoma downstream validation  
结论：最终人工 `USE=13` 全部完成真实 opposite-person partner 配对；Holosoma 正式生成 `26/26` paired motions，并全部通过 registry `pre-train` gate。本结论只表示 RL 输入与加载合同 ready，不表示 RL policy 已训练或成功。

## Authority 与选择

唯一人工选择 authority：

`workspace/core4d/results/E178/s6_downstream/eval/full/user_manual_review_filled.tsv`

SHA256：`d430a8ef125117027bc54e0b7bd9bbeafe5c7a0a2e2a25f17a6cea0fd57c6c9f`

最终计数：

| 项目 | 数量 |
|---|---:|
| reviewed | 27/27 |
| USE | 13 |
| DO_NOT_USE | 14 |
| PENDING | 0 |
| CLEAN（导出集） | 8 |
| MINOR_ACCEPTABLE（导出集） | 5 |
| numeric release PASS / FAIL（导出集） | 8 / 5 |

旧 log241 是更早的 `23 reviewed / USE=12 / DNU=11 / pending=4` 快照。本轮遵循用户明确指定的当前 filled TSV，不修改旧历史，只在新的 S6 export 中冻结最终 authority。

导出范围为 bucket003=`5`、bucket004=`1`、bucket007=`7`。Numeric gate不删除人工 `USE`，但逐门结果与failure modes完整保留为风险 metadata。

## Source 与 partner 合同

新增入口：

- `workspace/core4d/scripts/experiments/E178/export_manual_use_partner_rl.py`
- `workspace/core4d/scripts/launch/active/run_E178_manual_use_partner_rl_export.sh`

Source 侧冻结并验证：

- `source_exp_id=E178`；
- `spider_method_id=E178_semantic_bucket_contact_aligned_top_segment_union_r1`；
- `hand_collision_variant_id=rubber_hull`，没有套用 E187 的 COACD 配置；
- 13/13 的 scene、trajectory、contact mask、CEM result、video 均存在并记录 SHA256；
- DO_NOT_USE 零进入。

Partner 侧复用 canonical E174 Stage2b adapter，每条均解析为相同 `(object,date,seq)` 的另一 person：

- 13/13 partner 为 passing `omnirt_v1/ref_fk`；
- trimmed NPZ、OmniRetarget output、trim window、Stage2b provenance 与 SHA 全部写入 manifest；
- 13/13 使用 `common_raw_window`，共同窗口范围为 `76–201` 输入帧；
- 禁止 source-only、零 partner 或复制 source actor fallback。

Core4D 发布结果：source=`13`、partner=`13`、pair complete=`13`、alignment ready=`13`。

## Holosoma 正式资产

固定 exporter 先逐 case 运行 `pre-export --target-source both`，再按 object 使用原始 mesh half-extents 导出：

| Object | cases | motions | output frames（每个case，CEM/trajectory一致） |
|---|---:|---:|---|
| bucket003 | 5 | 10 | 334, 334, 209, 204, 159 |
| bucket004 | 1 | 2 | 207 |
| bucket007 | 7 | 14 | 125, 189, 189, 154, 222, 174, 174 |
| total | 13 | 26 | — |

每条 case 产生 `cem` 与 `trajectory` 两个 target source；partner 均来自真实 opposite-person OmniRetarget motion。每个最终 NPZ 包含：

- `partner_hand_pos_w=(T,2,3)`；
- `partner_hand_quat_w=(T,2,4)`；
- `object_contact=(T,2)`。

独立审计结果：26/26 `export_pass`，26 个 `(case_id,target_source)` 唯一，manifest/output frame一致，必需字段完整，所有数值数组 finite，NaN count=`0`。

## Registry readiness

三份 manifest 已注册到 Holosoma `workspace/v3/registry/motion_manifests.tsv`，owner scope 为 `E178_manual_use_partner_rl`。Registry rebuild 结果：`upstream=136`、`motions=81`。

- bucket003/004/007 三份 manifest 的 `post-export` 全部 PASS；
- E178 motion ID 精确 26 条；
- `pre-train` 逐 motion 检查 `26/26 PASS`。

这里的 `pre-train PASS` 只表示路径、provenance、partner、frame、schema 和 finite 合同满足训练入口要求。本阶段没有建立 RL run ID、没有启动训练、没有产生 policy 指标。

## 结果路径与 SHA256

| 内容 | 路径 | SHA256 |
|---|---|---|
| final manual authority | `workspace/core4d/results/E178/s6_downstream/eval/full/user_manual_review_filled.tsv` | `d430a8ef125117027bc54e0b7bd9bbeafe5c7a0a2e2a25f17a6cea0fd57c6c9f` |
| source RL input | `workspace/core4d/results/E178/s6_downstream/rl_export/rl_export_input.tsv` | `de01f208535a06fd207bb92655d7a511955a18c65bbf95f1d35a78644a5e3fff` |
| partner manifest | `workspace/core4d/results/E178/s6_downstream/rl_export/partner_omnirt/rl_partner_omnirt_manifest.tsv` | `e0616ec883eac2790a2a92f22bce6510cf33856c7464b97c7145b04a5586f194` |
| paired RL input | `workspace/core4d/results/E178/s6_downstream/rl_export/paired_rl_export_input.tsv` | `7cee1db7fc3fdeb9f1a818123c17377cae6ac778966a99e427d2e5c34fba2e2b` |
| partner alignment audit | `workspace/core4d/results/E178/s6_downstream/rl_export/partner_resolution_audit.tsv` | `bfa76226908fd01d25559ec250f488a88ce9d597470155689e5f117b8f351d2a` |
| export validation | `workspace/core4d/results/E178/s6_downstream/rl_export/validation_report.json` | `a9cd47aee0d0ccccf892085635652ed841bc07f3c41c31a831ad388cb276fa9d` |
| downstream validation | `workspace/core4d/results/E178/s6_downstream/rl_export/holosoma_downstream_validation.json` | `cf6fd1bef515fe77f243c0ebb66c92ff7adb2a4effdf78085085a7620c8a40ca` |
| Holosoma bucket003 manifest | `/home/ubuntu/Workspace/holosoma/workspace/v3/data/E178_manual_use_partner_rl/bucket003/manifest.tsv` | `3469fa0176a836eeacfbcdc482d211f4fd0a0e372072af71872436c7ddf557d1` |
| Holosoma bucket004 manifest | `/home/ubuntu/Workspace/holosoma/workspace/v3/data/E178_manual_use_partner_rl/bucket004/manifest.tsv` | `c4b1dfb6a7c32d75c0a75bfe1b62c71608ebf33db06ce3ceeb687d6258059abc` |
| Holosoma bucket007 manifest | `/home/ubuntu/Workspace/holosoma/workspace/v3/data/E178_manual_use_partner_rl/bucket007/manifest.tsv` | `fc21e4aa0df0ed694a820bc4b5694ee1b5a503d54c413adf191fe5df8bdccaf6` |

## Governance

- 本阶段只增加 E178 S6 export/downstream evidence，不修改 E178 S1–S5、CEM 或 eval 历史；
- 人工 USE 与 numeric PASS 的语义保持分离，5条 numeric FAIL 没有被掩盖；
- partner 信息为硬门，13条均为真实 opposite-person motion；
- `RL_EXPORT_READY` / `DOWNSTREAM_RL_INPUT_VALIDATION_PASS` 不等于 RL policy 成功；
- 未执行 commit/push。
