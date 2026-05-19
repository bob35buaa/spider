# E016 Plan: E014 paper-aligned metrics + multi-case generalization

日期：2026-05-19

## 背景

E014 已证明 COLA-B kinematic support + soft weld/equality 在 `box025_person2` main 和 `box023_person2` guard 上成立：6/6 full 通过 true-freejoint parity、非 COM oracle、无 direct wrench，4/4 main 通过 E013 soft target，2/2 guard stable。

E015 将 support 升级为 dynamic 6-DoF body + PD 后没有复现 E014，失败模式是 support target lag、PD saturation 和数值不稳。因此本轮先不继续 E015b 参数 sweep，而是把 E014 B-only 配置作为当前 work candidate，补齐论文对齐评测指标，并在更多 CORE4D case 上做泛化验证。

## 指标对齐依据

| 来源 | 对齐指标 | 本轮实现 |
|------|----------|----------|
| SPIDER | object `E_pos`、`E_rot`；success = object mean translation error `<0.1m` 且 rotation error `<0.5rad`；humanoid 场景还报告 joint / pelvis / end-effector tracking | 增加 case-window/full object position mean/max、per-frame object rotation error mean/max、SPIDER success gate、pelvis/body proxy、hand contact |
| DynaRetarget | `Epos < 10cm` + `Erot < 25deg` success；smoothness `S = sum ||qddot||_1` 并归一到 reference；RL 侧报告 success、MPKPE、object pos/ori error | 增加 Dyna success gate、relative smoothness、object final/xy/z error、carry progress ratio |
| OmniRetarget | kinematic quality: penetration duration/max depth、foot skating duration/max velocity、contact preservation；下游 RL success rate | 增加 robot-object penetration proxy、foot skating proxy、mask-based contact preservation、transport success gate |
| Holosoma v2 eval | carry progress、XY displacement/progress、z height、object xy/z/final error、body error、per body-group contact | 复用 carry/height/error 分解和 hand/leg/floor contact 分解，作为协作搬运诊断 |

## Claims

| Claim | 验证方式 |
|-------|----------|
| C1 E014 paper-aligned 指标能在已有 E014 full 结果上稳定产出 | 对 `results/E014/*.npz` 生成新增 `paper_*` 字段和 aggregate |
| C2 E014 的评价不只看 object error，还能识别 push/floor/leg/foot-skating/contact-preservation 失败模式 | 新增 comparison CSV 和 per-case JSON 包含 OmniRetarget-style artifact metrics |
| C3 E014 B-only 的 scene/override 生成可泛化到 10+ CORE4D case，不限于 box025/box023 person2 | 生成 13 个 E016 derived freejoint task + E014-style soft-weld scene + overrides |
| C4 10+ case 验证中，至少能区分“结构泛化失败”和“CEM/robot contact 失败” | aggregate 输出 config pass、paper success、transport success、diagnostic class 分布 |

## Case 集合

优先选择已有 3cm contact mask 的 E079/E080 case，覆盖 box / bucket / desk 和 person1/person2：

`box021_person1`, `box021_person2`, `box023_person1`, `box023_person2`, `box025_person1`, `box025_person2`, `bucket001_person1`, `bucket001_person2`, `bucket005_s2_person1`, `bucket005_s2_person2`, `bucket007_person1`, `bucket007_person2`, `desk021_person1`。

所有 case 都派生为 `{source}_freejoint_legobj_e016`，保持 object true-freejoint、robot `nu=29`、object 最后 7 qpos，并加入 leg/foot-object collision pairs。

## 实现计划

1. 新增 reusable paper metrics helper：
   - `scripts/eval/paper_metrics.py`
   - 输入 summary / model / qpos / qpos_ref / contact mask / leg-object rows
   - 输出 `paper_*` 指标：SPIDER/Dyna object errors、smoothness、Omni-style penetration/foot skating/contact preservation、Holosoma-style carry/height/error split。

2. 扩展 E014 eval：
   - 在 `eval_E014.py` 中调用 helper。
   - 重新评估已有 E014 full 结果，确认新增指标写入 `comparison.csv`。

3. 新增 E016 资产与脚本：
   - `scripts/E016/variants.tsv`
   - `scripts/E016/generate_e016_assets.py`：复制 freejoint task、加入 leg-object pairs、按参考手-物局部位形自动估计 non-COM support point、生成 E014-style soft weld scene。
   - `scripts/E016/generate_e016_overrides.py`
   - `scripts/run_E016_preprocess.sh`
   - `scripts/train/train_E016.sh`
   - `scripts/train/train_E016_remote_tmux.sh`
   - `scripts/run_E016_remote.sh`
   - `scripts/pull_E016_remote_results.sh`
   - `scripts/eval/eval_E016.py`

4. 运行顺序：
   - 静态检查 + preprocess。
   - 13 case smoke 验证 wiring。
   - 泛化验证：先跑 quick wave（低 samples/iters，但完整 case-window），如时间允许再对失败/代表 case 跑 full。
   - 全量 paper-aligned eval + log/tracker/progress 更新。

## 成功标准

Metrics implementation：

- 既有 E014 6 条结果新增 `paper_*` 字段完整，无 NaN/缺字段。
- E016 13 case 均通过 scene/config parity：`contact_guidance=false`、`nq_obj=7`、`nu=29`、object actuator empty、support weld anchor 非 COM。

Generalization quick validation：

- `num_cases >= 10`。
- 每个 case 产出 NPZ + eval summary。
- aggregate 至少报告：`num_paper_spider_success`、`num_paper_dyna_success`、`num_contact_preservation_ok`、`num_transport_success`、`diagnostic_classes`。

Interpretation：

- 若多数 case object paper success 通过但 contact preservation / foot skating / floor/leg 指标差：E014 是 object oracle-like candidate，需要下游 RL/robot-side contact 处理。
- 若多数 case config/parity 失败：先修泛化资产生成，不做算法结论。
- 若多数 case object tracking 和 artifact 指标都通过：E014 B-only 可作为 pipeline candidate。
