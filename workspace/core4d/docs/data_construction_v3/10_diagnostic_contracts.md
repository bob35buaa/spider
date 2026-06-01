# 10 诊断 contract

本文件说明 E098-E101 如何进入 v3。核心原则：

- E098 是全路线基础 contract；
- E099-E101 是 `target_variant_id=fingertip_aware` 的 route-specific contract；
- 默认 `ref_fk` route 不要求强制跑完 E099-E101。

## E098: 全路线基础 contract

所有 route 必须继承：

1. face label 使用全 3D 六面 argmax；
2. `anchor_face_review=true` 且未 refit 必须 hard block；
3. `contact_pos` 是 G1 FK palm site，不是 raw fingertip；
4. replay gate 记录 `pelvis_end_z`、`lie_on_box_frac`、`pelvis_tilt_end`；
5. `pelvis_tilt_end` 只作为 diagnostic，不单独作为 hard fail。

v3 落点：

- S1/S4 公共几何工具：`workspace/core4d/scripts/data_construction_v3/lib/geometry.py`；
- S4 replay metrics；
- S5 failure taxonomy；
- registry diagnostic 字段。

`geometry.py` 当前固定：

- OBJ vertex 读取与 AABB extents；
- 六面 3D argmax face label：`+x/-x/+y/-y/+z/-z`；
- `contact_pos_source` 标准值：`fk_palm_site` / `raw_fingertip` / `external_target`。

因此 E098 可以纳入默认数据构建路径。`ref_fk`、`adaptive`、`fingertip_aware` 都继承这一层检查。

## E099: fingertip-aware 前置审计

仅 `fingertip_aware` route 必填：

- raw fingertip vote；
- palm/FK vote；
- `face_changed_l/r`；
- quat audit；
- `disable_world_up`；
- raw contact 3D visual evidence。

对 `ref_fk` route，这些可作为 diagnostic 缓存，但不是进入 CEM/RL 的硬门槛。

v3 执行口径：只有显式选择 `target_variant_id=fingertip_aware` 时，S3 才把这些字段当作硬门槛。默认 `omnirt_v1/ref_fk` 不读取也不等待 E099。

## E100: fingertip-aware target 生成

`fingertip_aware` route 必须记录：

- target NPZ path；
- `spider_contact_target_object_local` shape/hash；
- active mask；
- target gap；
- vote face；
- quat audit；
- guard case face_changed=False 时 target swap = 0 的证据。

如果下游暂不读取 active mask，也必须在 manifest 保留。

v3 执行口径：这些字段进入 route diagnostic manifest。`target_active_mask_status` 未通过时，`fingertip_aware` 不进入 Stage2b。

注意：route diagnostics 只证明 `fingertip_aware` 具备进入该 route 的数据证据。当前 legacy Stage2b execute adapter 仍只支持 `ref_fk`；`fingertip_aware` 真正执行需要后续补 target adapter，不能复用旧 `ref_fk` pipeline 输出冒充。

## E101: route-level evidence

E101 的结论是：

- `fingertip_aware` 不退化 box004 guard；
- `fingertip_aware` 不能自动救 Box021 D003；
- Box021 D003 的该 route 不能因为 face target 修复而自动升为 positive；
- 失败应归入 downstream posture/upright/motion-level binding 风险。

v3 处理：

- 把 E101 作为 `fingertip_aware` route 的 negative prior；
- 不影响 `ref_fk` route 的默认 positive 判定；
- 不把 E101 旧 polluted-template 结果当作 clean-source hard label。

v3 执行口径：`e101_route_evidence_status=pass` 是 `fingertip_aware` 的 route contract 字段之一；它不影响 `ref_fk` / `adaptive` 的默认判定。
