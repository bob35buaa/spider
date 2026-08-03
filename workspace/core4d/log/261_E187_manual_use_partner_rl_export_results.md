# E187 人工 USE + partner RL export 与下游验证结果

日期：2026-08-03
实验：E187
阶段：S6 manual-authority RL export / Holosoma downstream input validation
结论：最终人工 `USE=14` 全部完成真实 partner 配对；Holosoma 正式生成 `28/28` paired motions，并通过 registry `pre-train` gate。该结论只表示 RL 输入与加载链 ready，不表示 RL policy 已训练成功。

## Authority 与选择

本阶段执行 [plan 211](../plan/211_E187_manual_use_partner_rl_export_plan.md)。唯一人工 authority 为：

`results/E187/s6_downstream/eval/full/user_manual_review_filled.tsv`

SHA256：`42dd0ef48cb47dde6cd35d88db0a4e46932e89bbe123227eae2b98f5252f431b`

最终计数：

| 决策/质量 | 数量 |
|---|---:|
| USE | 14 |
| DO_NOT_USE | 8 |
| PENDING | 0 |
| CLEAN（导出集） | 7 |
| MINOR_ACCEPTABLE（导出集） | 7 |

导出规则是“全部且仅人工 `USE`”。Numeric gate不再删除用户已批准的case，但它的状态和failure modes完整保留在source TSV中。14条USE里numeric PASS仅`5`条，numeric FAIL为`9`条；因此本次是用户waiver下的下游验证，不是numeric/C9翻转。

## Source export

新增canonical wrapper：

`scripts/experiments/E187/export_manual_use_partner_rl.py`

它在发布前执行以下fail-closed检查：

1. 人工authority SHA与14/8/0计数精确匹配；
2. manual、E187 evaluation manifest与metrics的22-case集合相等；
3. 14条source都有`scene_act`、`trajectory`、`contact_mask`、CEM result与video；
4. `source_exp_id=E187`、`target_variant_id=ref_fk`、`spider_method_id=E187_canonical_distance_continuation_reward_r1`、`hand_collision_variant_id=object_specific_coacd_compound`；
5. 人工decision/quality、numeric pass/failure、逐门状态、`C9=FAIL/USER_WAIVED`全部进入metadata；
6. DO_NOT_USE零进入。

Source scope：bucket003=`3`、bucket004=`2`、bucket007=`9`，共14条。

## Partner 解析与对齐

Partner不是从E187 CEM `qpos[:,1]`伪造，也不是复制source。wrapper复用共享
`finalize_reused_partner_rl.py`合同：

- partner是同一object/date/sequence的另一位CORE4D person；
- 从E174 passing Stage2b证据中选择`ref_fk`；
- variant preference为`omnirt_v1`后`omnirt_v2`；
- 实际14条均唯一解析为passing `omnirt_v1/ref_fk` partner；
- converted/OmniRetarget/retargeted/trimmed/trim-window全部记录路径与SHA256。

`partner_resolution_audit.tsv`同时比较：

- source trajectory帧数；
- E187 CEM帧数；
- source contact-mask帧数；
- source Stage2b trim window；
- partner trimmed NPZ与trim window；
- 双方raw窗口交集、crop offset与共同帧数。

结果为`14/14 RL_EXPORT_READY`，全部使用`common_raw_window`；共同输入长度范围`76–201`帧，无missing、ambiguous或alignment blocked case。

## Holosoma 正式转换

固定入口：

`workspace/v3/scripts/data/export_rl_motion_from_spider_tsv.py`

先用Holosoma `registry_gate.py pre-export`验证14/14 source/partner，再对每个case正式导出两种target：

- `cem`：E187 Full CEM的`qpos[:,0,:]`；
- `trajectory`：冻结OmniRetarget target trajectory；
- partner：同一条真实opposite-person OmniRetarget motion；
- `--include-contact-mask`：最终NPZ包含`object_contact=(T,2)`。

按object分包是因为fixed exporter的`--object-half-extents`是单值参数。half-extents从原始OBJ顶点精确计算，只用于partner-hand surface-distance诊断：

| Object | cases | motions | original-mesh half-extents (m) |
|---|---:|---:|---|
| bucket003 | 3 | 6 | `0.270393755, 0.381407245, 0.23287703` |
| bucket004 | 2 | 4 | `0.161566625, 0.231058755, 0.15230013` |
| bucket007 | 9 | 18 | `0.273275865, 0.2869532, 0.284911265` |
| total | 14 | 28 | — |

实际输出帧数按case分别为：

- bucket003：`334, 209, 204`；
- bucket004：`237, 207`；
- bucket007：`125, 125, 189, 145, 155, 154, 222, 174, 174`。

每个case的CEM与trajectory输出帧数一致。

### 转换中发现并闭合的两个合同问题

1. Holosoma exporter把相对`--out-dir`传给更深层CWD的converter，导致第一次bucket003试跑找不到刚生成的converter input。正式命令改用绝对out-dir后闭合；没有发布失败manifest。
2. E174的bucket007 `object_name`大小写混用。Holosoma实际资产为`models/Bucket007/Bucket007.obj`，lowercase路径不存在。E187 wrapper只对下游metadata规范化为`Bucket007`，未修改mesh、轨迹、scene或case集合；随后9/9 case转换通过。

## 独立下游审计

独立脚本逐个读取28个正式NPZ与三份manifest，结果：

| 检查 | 结果 |
|---|---|
| unique `(case_id,target_source)` | 28/28 |
| `decision=export_pass` | 28/28 |
| manifest/output frame一致 | 28/28 |
| `partner_hand_pos_w=(T,2,3)` | 28/28 |
| `partner_hand_quat_w=(T,2,4)` | 28/28 |
| `object_contact=(T,2)` | 28/28 |
| numeric NaN count | 0/28有NaN |

三份manifest已登记到Holosoma `workspace/v3/registry/motion_manifests.tsv`，owner scope为`E187_manual_use_partner_rl`。registry rebuild成功：`upstream=115`、`motions=73`。随后：

- `post-export`：三份manifest全部PASS（`6+4+18=28` rows）；
- registry中E187 motion ID：精确28条；
- `pre-train`：`28/28 PASS`。

这里的`pre-train PASS`表示路径、manifest、case、target/partner provenance、帧和NaN合同满足训练入口要求；本阶段没有创建RL run ID、没有启动完整RL训练，也没有产生policy指标。

## Governance

- E187 C9仍为technical `FAIL`，progression authority仍为`USER_WAIVED`；
- 人工USE不等于numeric PASS；9/14仍带numeric failure；
- `RL_EXPORT_READY`、`DOWNSTREAM_RL_INPUT_VALIDATION_PASS`不等于RL成功；
- S1–S5事实未修改；
- E178共享代码与历史artifact未修改，E187通过导入canonical partner adapter保持兼容；
- `results/`仍不进入git。

## 结果路径与SHA256

| 内容 | 路径 | SHA256 |
|---|---|---|
| final manual authority | `workspace/core4d/results/E187/s6_downstream/eval/full/user_manual_review_filled.tsv` | `42dd0ef48cb47dde6cd35d88db0a4e46932e89bbe123227eae2b98f5252f431b` |
| source RL input | `workspace/core4d/results/E187/s6_downstream/rl_export/rl_export_input.tsv` | `42b2964b5894fa47716d63a37d7da6b7ae6a512949cb2954e67ce00f77c112d6` |
| partner manifest | `workspace/core4d/results/E187/s6_downstream/rl_export/partner_omnirt/rl_partner_omnirt_manifest.tsv` | `92175d182a89c38632185ff82e4950f7c7d933aa3bcec53386c6133426a152e8` |
| paired RL input | `workspace/core4d/results/E187/s6_downstream/rl_export/paired_rl_export_input.tsv` | `7a380f430fdaf12a49c319e5bd7ff81399f1c9b3fd13019b694b400dbc76c0b1` |
| partner alignment audit | `workspace/core4d/results/E187/s6_downstream/rl_export/partner_resolution_audit.tsv` | `c491344397b9043b3651d440cd132ae2255241d73fa4678ac78e3a2453574051` |
| export validation | `workspace/core4d/results/E187/s6_downstream/rl_export/validation_report.json` | `c3e23e167757a52d8aec013205386ce7cebdde5121cf3ec3759bbc659e039893` |
| downstream validation | `workspace/core4d/results/E187/s6_downstream/rl_export/holosoma_downstream_validation.json` | `762ee6cffe761610d09330643919ba0dbbe923153004c8550a951235aa607007` |
| Holosoma bucket003 manifest | `/home/ubuntu/Workspace/holosoma/workspace/v3/data/E187_manual_use_partner_rl/bucket003/manifest.tsv` | `8c36db32aa97ba1d2d7a813024562eb8e7f390d1850e1e1f2c7e509ec1a212ce` |
| Holosoma bucket004 manifest | `/home/ubuntu/Workspace/holosoma/workspace/v3/data/E187_manual_use_partner_rl/bucket004/manifest.tsv` | `ea4d119a5f8c3a56c405ad4414c06ef7f918007a6106f76309b73eeda760d5fc` |
| Holosoma bucket007 manifest | `/home/ubuntu/Workspace/holosoma/workspace/v3/data/E187_manual_use_partner_rl/bucket007/manifest.tsv` | `50688804e486eb08fc746484e4fd615d5bd2b8d035bb719dcc73836a34a9f1b5` |

## 下一步

如果要比较E187与E178在RL层面的真实效果，应另写训练/评估计划并冻结：

1. run ID、训练代码/config SHA、seed与GPU分配；
2. 14个case是全部进同一训练集还是按object/case独立训练；
3. CEM target与trajectory target是对照arm还是只选CEM；
4. RL成功指标、E178 baseline checkpoint/数据与配对统计口径；
5. 9条numeric FAIL人工USE的安全监测和停止条件。

在这些训练实验变量冻结前，本阶段停在`DOWNSTREAM_RL_INPUT_VALIDATION_PASS`是正确边界。
