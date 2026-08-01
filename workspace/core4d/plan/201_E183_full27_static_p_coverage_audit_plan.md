# E183 实验计划：E178 Full27 object-specific CoACD static-P coverage audit

_Core4D Phase 46 · 2026-08-01 · CPU-only execution authorized_

## Context

E182-v9只在`bucket003_20231018_001_p1`的reference与E178-final两段轨迹上检查
882个pose，6个candidate为`0/6 PASS`。用户指出单case证据不足，批准把相同static-P
口径扩展到E178全部27 case，并要求先按实测结论使用CPU并行，不占用或干预GPU任务。

E178 Full authority为27 case，按物体分布`bucket003/004/007=9/4/14`。E181已有三个
物体各18个标准CoACD candidate；E182-v9另有bucket003的6个最新candidate。不同物体的
碰撞体不可交叉使用。

本实验是新的full27 coverage audit，不修改E182-v9已经冻结的negative result。因为会
访问原heldout24，E183结果明确标记为`evaluation/coverage only`：不得据此移动plane、
新增K/threshold或把同一27 case重新宣称为独立heldout selection evidence。

## Candidate与query冻结口径

| 来源 | bucket003 | bucket004 | bucket007 | 合计 |
|---|---:|---:|---:|---:|
| E181标准CoACD (`t×K×v`) | 18 | 18 | 18 | 54 |
| E182-v9 double-plane | 6 | 0 | 0 | 6 |
| Total candidates | 24 | 18 | 18 | 60 |

- E181 family：threshold=`5/10/20mm`，K=`8/16/32`，max vertices=`32/64`；
- E182-v9：threshold=`5/10/20mm`，K=`16/32`，actual hulls=`16/32/16/32/16/24`；
- query：每个case的完整reference + E178-final object-local P collision points；
- oracle：相应物体E181 cleaned original mesh `D_M`；
- contact定义：`min(signed_distance - geom_radius) <= 0`；
- gate：沿用E182的precision和recall各`>=0.70`，不使用不严格的固定phantom计数替代；
- 汇总：逐case、逐物体pooled、逐candidate macro、全pose pooled同时报告；
- 并行：本机4个CPU worker，允许与既有任务叠加，禁止GPU/kill/pause/preempt。

## Claims

| Claim | 最低证据 |
|---|---|
| C0 authority | full27 manifest SHA冻结；27 unique cases；物体分布exact=`9/4/14`；trajectory/E178-result/scene/config SHA全部校验 |
| C1 candidate immutability | 60/60 manifest与ordered asset SHA校验；candidate仅匹配同object case；protocol在score前冻结 |
| C2 static query closure | 27/27 tape COMPLETE；reference/final均存在；总pose=`14542`；non-finite=`0` |
| C3 regression | bucket003 dev case上v9六行TP/phantom/missed exact复现log250 |
| C4 score closure | object-matched candidate-case rows=`24×9 + 18×4 + 18×14 = 540`；missing/duplicate=`0` |
| C5 metric integrity | 每行confusion总数=pose数；pooled confusion等于逐case求和；macro不被长轨迹加权 |
| C6 runtime | 4 CPU workers完成；记录query build、score、aggregate wall、worker RSS；GPU访问=`0` |
| C7 evidence | 每物体candidate排名、case pass矩阵、worst phantom/missed case；3D与2D representative诊断各至少1份并记录实际观察 |

Claims只要求审计闭合，不预设candidate必须通过。允许科学结论为全FAIL、object-specific
mixed或存在full27可用candidate。

## 实现改动

### 1. Full27 static query与协议

**文件**：`workspace/core4d/scripts/experiments/E183/audit_full27_static_p.py`

- 只读join E182 full27 authority与E178 source manifest；
- 复用E182已验证的MuJoCo qpos→object-local query实现，只写reference/final；
- score前冻结authority、candidate、code SHA；已有protocol只允许validation-only resume。

### 2. CPU并行评分与汇总

- 以candidate为任务，每个worker加载一次candidate scene并串行评分该object全部cases；
- `ProcessPoolExecutor(max_workers=4)`；
- 原子写逐candidate JSON，最终生成`case_candidate_metrics.tsv`、
  `candidate_summary.tsv`、`aggregate.json`与runtime telemetry。

### 3. Tests与固化入口

| 文件 | 内容 |
|---|---|
| `workspace/core4d/scripts/experiments/E183/test_audit_full27_static_p.py` | authority、candidate隔离、metric/aggregate、resume/tamper与回归合同 |
| `workspace/core4d/scripts/eval/wrappers/eval_E183_full27_static_p.sh` | 固化CPU-only protocol/query/score/visual/validate命令 |

## 结果路径

```text
workspace/core4d/results/E183/full27_static_p/
  protocol_manifest.json
  query_tape/<case_id>/{reference,e178_final}.npz
  query_tape/<case_id>/manifest.json
  scores/<source>__<object>__<candidate>.json
  case_candidate_metrics.tsv
  candidate_summary.tsv
  aggregate.json
  visual/
```

本实验为纯离线分析，不运行物理训练，按技能规则不要求scene XML训练快照。

## 执行命令

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E183_full27_static_p.sh
```

入口固定环境变量`OMP/OPENBLAS/MKL/NUMEXPR/TBB=1`，但Open3D内部仍可能自动线程化；
worker固定为4，runtime报告实际CPU/wall而不声称线性加速。

## Stop/go

- authority/candidate SHA、dev回归或540-row closure任一失败：停止，不解释质量；
- 审计闭合后如无candidate通过：保留负结果，不调参、不启动Full；
- 如存在通过candidate：只报告其full27 coverage，后续是否freeze/Full需另行批准；
- 本实验不访问R/G CEM query、不启动Full、不使用任何GPU。
