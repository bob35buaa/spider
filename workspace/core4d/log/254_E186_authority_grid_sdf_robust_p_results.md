# E186 阶段日志：22-case authority、canonical grid-SDF v3 与 robust-P复核

_Core4D Phase 49 · 2026-08-01 · plan
[204](../plan/204_E186_22case_object_specific_prg_full_plan.md)_

## 0. 一句话结论

E186 Gate0与S1已通过：三个object-specific CoACD collider和keep22/drop5已不可变冻结；
canonical grid-SDF最终采用解析convex-part half-space sign的v3，3/3通过每物体100万点
验证。发现并修复Open3D/trimesh ray sign在boolean-union上的系统性错误后，使用同一
E183真实query与oracle labels重算P，原22/5选择exact不变，robust-P状态为`CONFIRMED`。

这只授权继续实现compound-convex physics与R/G backend；尚未运行canary或Full CEM，
不能声明轨迹质量已经改善。

## 1. 冻结authority

| 项目 | 结果 |
|---|---|
| E178 authority | 27行，SHA `de9a3d16...022a8` |
| keep / drop | `22 / 5` |
| keep by object | bucket003/004/007=`5/4/13` |
| collider actual hulls | bucket003/004/007=`16/8/8` |
| collider-set SHA | `1bc7efe5...22e62` |
| keep22 SHA | `35d028f3...eb535` |
| drop5 SHA | `baa2dc11...a604` |

固定collider：

```text
bucket003 E181__bucket003__t020_k16_v032
bucket004 E181__bucket004__t005_k08_v064
bucket007 E181__bucket007__t020_k08_v064
```

drop5继续为：

```text
bucket003_20231018_001_p1
bucket003_20231018_005_p1
bucket003_20231020_068_p1
bucket003_20231018_003_p2
bucket007_20231003_2_021_p2
```

## 2. Grid-SDF sign根因与修复

### 2.1 被拒绝的sign backend

v1使用Open3D `compute_signed_distance(nsamples=5)`。bucket003正式1M验证出现5个
>2h sign mismatch，最坏点在union AABB外却被判inside，exact/grid分别为
`-129.995/+130.019mm`。进一步在AABB内20万点/object与trimesh contains比较：

| Object | sign mismatch | >1cm deep mismatch |
|---|---:|---:|
| bucket003 | 47,671 | 30,893 |
| bucket004 | 55,922 | 38,056 |
| bucket007 | 42,843 | 31,470 |

增加Open3D ray数不稳定；v2改用trimesh ray contains后，bucket003 50k仍有2,281个
deep mismatch，p99误差74.73mm。v1正式artifact保留`GRID_REJECTED`，v2仅保留临时
失败证据，两者均不得被runtime消费。

### 2.2 v3解析sign

`C`是凸part并集，因此采用无ray的解析合同：

```text
unsigned magnitude = distance to boolean-union external surface
inside(C) = OR_i(all convex half-space inequalities of part_i pass)
signed distance = -magnitude if inside(C), else +magnitude
```

SciPy ConvexHull生成part half-space，part AABB用于预筛。该方法直接绑定32个frozen
ordered parts，不依赖boolean mesh ray parity。

## 3. 正式v3结果

| Object | Voxel | Payload | p99 error | epsilon | deep sign mismatch | CUDA max | Manifest SHA |
|---|---:|---:|---:|---:|---:|---:|---|
| bucket003 | 5mm | 10.0MiB | 1.223mm | 2.875mm | 0 | 0.112µm | `f3682e48...f6914` |
| bucket004 | 2.5mm | 24.0MiB | 0.394mm | 1.386mm | 0 | 0.067µm | `c7c797db...1a5da` |
| bucket007 | 5mm | 9.1MiB | 0.971mm | 2.494mm | 0 | 0.119µm | `d4f87a03...f6094` |

bucket004从5mm细化到2.5mm，是因为预冻结smoke中5mm单点max为12.73mm；003/007
保留5mm以减少cache/memory。该选择发生在R/G与Full前，没有使用downstream结果。

runtime实现已提供float32 trilinear query、candidate/grid SHA fail-closed、
`D_C-epsilon`保守gate query，以及grid外object-AABB距离下界。公共loader默认拒绝
PENDING/REJECTED grid。

## 4. robust-P复核

复核严格保留E183的：

- 27 case、14542 poses；
- reference与E178-final points/radii；
- original-mesh oracle contact labels；
- 三个固定candidate与0.70 zero-aware gate。

只把candidate contact的ray sign换为解析half-space union sign。结果：

| 指标 | Old | Robust |
|---|---:|---:|
| keep count | 22 | 22 |
| bucket003/004/007 | 5/4/13 | 5/4/13 |
| selection changed | — | **0** |

正式case evidence SHA=`1eaaf50d...9985`，aggregate状态=`CONFIRMED`。因此无需重写
用户冻结authority，且已消除旧P实现风险。

## 5. Claims状态

| Claim | 状态 | 说明 |
|---|---|---|
| C0 authority | PASS | 22/5/27、输入SHA与顺序闭合 |
| C1 collider freeze | PASS | 3 manifest与32 parts SHA闭合 |
| C2 P selection | PASS | robust re-audit exact确认同22/5 |
| C3 canonical D_C | PASS | v3 3/3、1M/object、CPU/CUDA parity |
| C4 compound physics | PENDING | sidecar/pair/MuJoCo/MJWarp尚未实现完成 |
| C5–C10 | PENDING | R/G、canary、Full、paired eval尚未运行 |

## 6. 可视化与归因边界

三collider的2D/3D外观已在E183 representative visuals实际审查；本阶段新增的是数值sign
与grid一致性验证，没有生成轨迹视频。未完成physics/R/G canary前，禁止把S1 PASS解释为
Full轨迹可用度改善。

## 7. 下一步

1. 在default-off下接入production grid distance backend，保持E178 legacy逐值不变；
2. 构建22行compound-convex sidecar与`18 x K` pairs，验证mass/inertia/friction/solver；
3. 运行22-case reference/E178-final R/G audit；
4. 三物体`64 x 4` recorder-off canary与同机效率probe；
5. 全部门通过后才启动本地GPU0 + 远程Ada0/1的22-case Full。
