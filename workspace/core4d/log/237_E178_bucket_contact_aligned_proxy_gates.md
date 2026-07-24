# E178 结果日志：Bucket Contact-Aligned Top Segment 本地 Gates

_Core4D Phase 41 · 2026-07-24 · plan [194](../plan/194_E178_bucket_contact_aligned_top_segment_plan.md)_

## 0. 一句话结论

E177 的 27-case ref-contact gate 暴露 bucket003 borderline 与 bucket007 主接触
端段欠覆盖；E178 在不加 lid、不增加 geom 的前提下，仅调整第 5 段 X/Z 截面，
使 27/27 compile/pair/authority、三对象双向几何 gate 和 ref-contact p90≤8cm
全部通过。新 overlay/截面已生成，用户于 2026-07-24 明确批准；canary/Full
结果不在本日志中。

## 1. E177 Contact Gate 结果

E177 evaluator：

```text
workspace/core4d/scripts/eval/runners/eval_E177_contact_fidelity.py
workspace/core4d/scripts/eval/wrappers/eval_E177_contact_fidelity.sh
```

结果：

| Object | Cases | Active rows | Contact→proxy p90 | Gate |
|---|---:|---:|---:|---|
| bucket003 | 9 | 1885 | 8.059cm | FAIL（超 0.59mm） |
| bucket004 | 4 | 575 | 7.480cm | PASS |
| bucket007 | 14 | 1677 | 11.258cm | FAIL（超 3.26cm） |

27/27 可计算、0 errors，但 object gate 只有 bucket004 通过，因此严格停止在
Gate B，没有启动 E177 canary。

## 2. 根因定位

- bucket007 的 1677 个 active target rows 中，1665 个最近
  `object_collision_body_004`，即第 5 段；
- contact local-Y p05/p95=`0.1965/0.2879m`，说明 ref 接触集中在 `+Y`
  主接触端段；
- `581/1677=34.65%` rows 相对 visual mesh under-cover 超过 3cm；
- 左右手均存在失配，排除单手 mask 或单 case 偶然性；
- 所有 object boxes 均已进入 90 个 physics pairs 与 PRG union，排除
  multi-geom 接入遗漏；
- 主因是 bucket007 的 top XZ 仍使用 E177 全身 inward scale 0.82，而不是
  五段接缝、lower four segments 或 PRG/physics geom 盲区。

## 3. E178 Geometry

| Object | Lower 4 XZ scale | Top X/Z scale | Geoms | Pairs |
|---|---:|---:|---:|---:|
| bucket003 | 0.94 | 0.95 / 0.95 | 5 | 90 |
| bucket004 | single mesh AABB | same | 1 | 18 |
| bucket007 | 0.82 | 0.97 / 0.885 | 5 | 90 |

仍保持：

- local-Y 五段实心 boxes；
- 无独立 lid/lid-strip；
- 4mm segment overlap；
- physics pair 与 PRG SDF union 使用同一组 geoms；
- `object_collision_sdf_mode=union`；
- `object_collision_sdf_batch_groups=true`。

bucket007 最初的 top 等比 0.94 候选虽然 contact p90=7.993cm，但
exposed proxy→mesh p90=4.031cm，未过线。最终分轴值来自预注册细网格可行域，
用于降低圆形端部的 box 角点外扩。

## 4. 本地 Gate 结果

### 4.1 Scene/authority contract

- full rows=`27`，canary rows=`3`；
- object distribution=`9/4/14`；
- 27/27 MuJoCo scene audit PASS；
- geoms=`5/1/5`；
- robot–object pairs=`90/18/90`；
- E178 使用独立 `scene_act_E178_contactAlignedTop`、override 与 results；
- E177 regression PASS，默认 E177 geometry/metrics 未漂移。

### 4.2 双向 exposed-union geometry

| Object | Mesh→proxy p90 | Proxy→mesh p90 | 4cm gate |
|---|---:|---:|---|
| bucket003 | 2.56cm | 2.25cm | PASS |
| bucket004 | 2.21cm | 3.03cm | PASS |
| bucket007 | 3.31cm | 3.95cm | PASS |

### 4.3 正式 ref-contact fidelity

| Object | Active rows | Contact→proxy p90 | Gate |
|---|---:|---:|---|
| bucket003 | 1885 | 7.828cm | PASS |
| bucket004 | 575 | 7.480cm | PASS |
| bucket007 | 1677 | 7.913cm | PASS |

总计 27/27 evaluated、0 errors、4137 active rows。

## 5. 可视化

证据：

- 四视角 montage：
  `workspace/core4d/results/E178/scene_snapshot/semantic_bucket_proxy/proxy_visual_evidence/e178_3_object_proxy_montage.png`
- XY/XZ/YZ 与端面截面：
  `workspace/core4d/results/E178/scene_snapshot/semantic_bucket_proxy/proxy_visual_evidence/e178_bucket_cross_section_montage.png`

Codex 实际观察：

- bucket003 只轻微扩大第 5 段，未见明显台阶突变；
- bucket004 与 E177 相同；
- bucket007 top rectangle 约为 `±0.261×±0.250m`，覆盖圆形端部主轮廓；
- 四角存在 box 近似必然的 phantom，但未形成整圈明显外扩、贯通 bridge 或
  segment gap；量化 exposed proxy→mesh p90=3.95cm；
- 用户于 2026-07-24 回复“没问题，继续”，三对象 review 已落为
  `clean_reviewed/approve_clean`。

## 6. Claims

| Claim | 当前结果 |
|---|---|
| C1 scope 27，9/4/14 | PASS |
| C2 geoms 5/1/5、pairs 90/18/90 | PASS |
| C3 双向 union p90≤4cm | PASS |
| C4 三对象 contact p90≤8cm | PASS |
| C5 bucket007 端段归因与改善 | PASS |
| C6 visual 人工重新批准 | PASS |
| C7 canary 3/3、median≤3s | 未执行；记录到后续 canary log |

## 7. 结果路径

| 类型 | 路径 |
|---|---|
| E177 contact failure evidence | `workspace/core4d/results/E177/s2_proxy/contact_fidelity/` |
| E178 root | `workspace/core4d/results/E178/` |
| E178 full manifest | `workspace/core4d/results/E178/s6_downstream/manifests/semantic_bucket_full_manifest.tsv` |
| E178 canary manifest | `workspace/core4d/results/E178/s6_downstream/manifests/semantic_bucket_canary_manifest.tsv` |
| E178 contact evidence | `workspace/core4d/results/E178/s2_proxy/contact_fidelity/` |
| E178 visual review | `workspace/core4d/results/E178/scene_snapshot/semantic_bucket_proxy/proxy_visual_review.tsv` |

## 8. 下一步

视觉 gate 已通过。下一步运行三条 `64×4` canary；只有 3/3 runtime PASS 且
逐 case median plan time≤3s，才允许在 A100 GPUs `2,3,6,7` 启动 27-case
`1024×32` Full CEM。
