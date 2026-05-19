# E017 Plan: anchor audit + face-cluster selection

日期：2026-05-19

## Context

E016 在 13 个 case 上保持 `config_ok=13/13`、SPIDER/Dyna object success `13/13`、transport success `13/13`，但 robot-side OmniRetarget-style 指标没有过门：contact preservation ok `3/13`、deep penetration ok `9/13`、generalization pass `0/13`。

后续人工复查发现：E014 的成功 anchor 是手工指定的 object-local support point，不是从 contact mask 自动推断；E016 继承了 E014 的 B-only soft-weld 结构，但把 anchor 改成了 `mask_active_ref_palm_centroid_surface_clamp`。因此 E016 同时测试了两个变量：weld 结构泛化和 anchor 自动选择。`box023_p2` 是明确反例：E014 手工 anchor 为 `[0.16, 0.0, 0.10]`，E016 centroid anchor 为 `[0.030, 0.157, 0.150]`，视频中 robot-side 姿态明显变坏。

2026-05-19 追加语义修正：E016 与 E017 auto 的代码路径都从 `variants.tsv` 的 `person_idx` 读取 active palm mask，因此算法输入是 **selected-person contact mask**，不是显式的 counterpart/partner-side contact。E017 audit 必须拆成两层：

1. 对已知 GT case，优先用 E014 手工 support anchor (`box023_p2`、`box025_p2`) 判断 E016/E017 auto 是否对齐 support 语义。
2. 对无 GT case，新增 counterpart-person audit 通道：用同一份双人 contact mask 的另一维和对应 `*_person1/person2` 轨迹估计另一侧 dominant face，但该通道只作为弱证据，不能覆盖 E014 GT。

## Claims

| Claim | 验证方式 |
|-------|----------|
| C0 E016/E017 auto 是否真的使用另一侧 contact | 静态读代码并在 audit 中记录 `anchor_algorithm_contact_side=selected_person_contact_mask`；另算 counterpart-person face distribution 作为 support-side 弱证据 |
| C1 anchor audit 能在重定向前识别 E016 centroid anchor 的低置信 case | 对 13 个 E016 case 分别计算 selected-person active palm face distribution、counterpart-person face distribution、centroid face、current anchor face、face confidence、bimodal/opposed-face flag |
| C2 face-cluster selector 能避免 centroid cancellation 导致的 unsupported-face anchor | 对低置信 case 生成 top-face cluster candidate，而不是使用全局 centroid |
| C3 anchor 相关失败能与纯 robot-side contact 失败分离 | 将 E016 结果诊断与 anchor audit 对齐，列出 `likely_anchor_wrong` / `ambiguous_anchor` / `likely_non_anchor` |
| C4 至少在 `box023_p2` 上生成 E014-consistent 或 face-consistent 候选 | 新 anchor face 不再是 E016 的 `+Y` unsupported face；优先生成 `+X` 候选用于 quick 验证 |

## 改动

1. 新增 E017 anchor audit/selector 脚本：
   - 输入 E016 manifest 和 source trajectory/contact masks。
   - 输出 `results/E017/anchor_audit.csv`、`anchor_audit_summary.json`、`manifest.tsv`。
   - 记录 E016 current anchor 与 proposed anchor。
   - 记录 selected-person contact 通道与 counterpart-person contact 通道，避免把 selected-person top face 误称为 partner/support face。
2. 新增 E017 asset/override/preprocess 脚本：
   - 基于 E016 的 derived freejoint scene 逻辑，但 scene 命名为 `scene_e017_jointB_*`。
   - 支持 selector policy：`centroid`、`face_cluster`、`manual_seed`。
3. 先跑 audit，不立即覆盖 E016 结果。
4. 第一批 quick 验证只跑疑似 anchor 错误的 subset，优先：
   - `E017_box023_p2_face_cluster`
   - `E017_box025_p2_face_cluster`
   - audit 识别出的其他 low-confidence case。

## 成功标准

| Gate | 目标 |
|------|------|
| Audit coverage | 13/13 case 有 anchor audit 记录 |
| Pre-retarget flag | `box023_p2` 被标记为 `likely_anchor_wrong` 或 `low_confidence_centroid` |
| Candidate validity | 生成的 scene `nq=43`、`nv=41`、`nu=29`，无 object actuator/direct wrench |
| Quick target | anchor subset quick 至少不低于 E016 object success；若 contact preservation 或 visual posture 改善，则进入 E017b 13-case run |

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E017_preprocess.sh --force
.venv/bin/python workspace/core4d_collab_retarget/scripts/E017/audit_select_anchors.py --write
bash workspace/core4d_collab_retarget/scripts/train/train_E017.sh quick_subset 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E017.py --all
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/render_E017_visuals.py --force
```

若 subset 数量超过 3 且需要重跑 full quick，按远程规则拆到本地 1 卡 + 远程 2 卡。

## 结果路径

| 产物 | 路径 |
|------|------|
| Plan | `workspace/core4d_collab_retarget/plan/17_E017_anchor_audit_selection_plan.md` |
| Results | `workspace/core4d_collab_retarget/results/E017/` |
| Audit | `workspace/core4d_collab_retarget/results/E017/anchor_audit.csv` |
| Manifest | `workspace/core4d_collab_retarget/results/E017/manifest.tsv` |
| Log | `workspace/core4d_collab_retarget/log/17_E017_anchor_audit_selection_results.md` |
