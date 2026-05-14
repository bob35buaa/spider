# E074 前置分析: base 定义、E060-E067 影响、E062 palm normal

## 状态

本日志记录 E074 之前的讨论与决策，不包含新训练结果。目的有三点：

1. 明确 E060-E067 里哪些改动会影响后续实验。
2. 明确 E074 应该以哪个实验作为 base。
3. 解释 E062 `palm_normal` 的含义、代码路径和对 E074 的影响。

## 1. E060-E067 是否有直接代码改动

结论：有直接共享代码改动，但对 E074 的默认影响可控。

| 来源 | 改动类型 | 对 E074 的影响 |
|------|----------|----------------|
| E060/E061 附近 scene/robot 几何 | 曾尝试 3-box hand collision，后来回退为 sphere；保留 box023 margin 0.90 | 当前 E071/E073 已经在这个几何基础上运行，因此它是 base 环境的一部分，不是 E074 新变量 |
| E062 | 新增 `compute_palm_normal.py` 和 `core4d_e062_box023.yaml` | 没改共享 runtime code，但 E073 通过继承链使用了 E062 的 box023 palm normal |
| E065 | 修改 `spider/config.py` 和 `spider/simulators/mjwp.py`，新增 `task_obj_use_exp` 等开关 | 默认 `task_obj_use_exp=False`，所以 E074 不显式开启时不改变 E073 行为 |
| E066 | 新增 soft actuator 相关 YAML/脚本/结果 | 不被 E073/E074 继承，默认无影响 |
| E067 | 新增 narrow body partition / HDMI clone YAML/脚本/结果 | 不被 E073/E074 继承，默认无影响 |

因此，后续实验不应把 E065-E067 的失败结论直接纳入 E074 主线；这些实验发生在 E071 ctrl mapping 修复之前，对 early lunge 的解释被污染。可保留的是“开关和机制”，但如果要使用，必须在 E073/E071 修复后的 base 上重新验证。

## 2. E074 的 base 定义

E074 应以 E073 为直接 base：

```text
E074 base = core4d_e073_box023
           -> core4d_e071w02_box023
           -> core4d_e062_box023
           -> core4d_e041c
```

这个 base 包含：

- E071 `scene_act` ctrl mapping 修复：保留 raw 29-dim robot ctrl，并由 scene_act conversion 补齐 object actuator ctrl。
- E071W02 的 `warmup_steps: 0.20`。
- E062 box023 自动 palm normal：左右手均为 `[1,0,0]`。
- E073 `contact_hdmi_target_uses_eef_offset: true`。
- E041c legacy contact/object/local-frame reward stack。

E074 不应默认继承：

- E065 exp task object reward。
- E066 HDMI soft actuator gains。
- E067 narrow body partition。

这些若要重试，应作为 E074 之后的独立 ablation，而不是混入首个 hold/contact 修复实验。

## 3. E062 palm normal 是什么

`contact_hdmi_palm_normal_left/right` 不是 box 表面的 normal，也不是新的接触点位置。它是 contact_hdmi orientation reward 使用的 **wrist-local 朝向向量**。

在 reward 中，它的含义是：

1. 从手腕局部坐标系取一个候选方向，例如 `[1,0,0]`。
2. 用当前手腕姿态转到世界系，得到 `palm_world`。
3. 计算从当前 hand contact point 指向 target point 的方向 `dir_to_target`。
4. 用 `dot(palm_world, dir_to_target)` 衡量手的指定轴是否朝向目标点。
5. 在 E041c 的 additive ori mode 中，该方向项与位置接近项按权重混合。

对应代码路径：

```text
spider/simulators/mjwp.py
  contact_hdmi_rew
    palm_world = quat_apply(eef_quat, palm_local)
    dir_to_target = normalize(target_world - contact_point)
    ori_rew = clamp(dot(palm_world, dir_to_target), min=0)
    pos_rew = (1 - w) * pos_rew + w * ori_rew
```

## 4. E062 怎么得到 box023 的 `[+x,+x]`

E062 新增的 `workspace/core4d/scripts/convert/compute_palm_normal.py` 会读取 ref motion，对每只手测试六个 wrist-local 候选轴：

```text
+x, -x, +y, -y, +z, -z
```

每帧计算候选轴转到世界系后，与 `object_center - wrist_pos` 的单位方向做 dot，取平均 dot 最大的轴。

E062 输出：

```text
box025:
  left  -> [0,-1,0]
  right -> [0,+1,0]

box023:
  left  -> [+1,0,0]
  right -> [+1,0,0]
```

也就是说，E041c 默认的 `[0,-1,0] / [0,+1,0]` 对 box025 自洽，但对 box023 更像是 box025 motion fingerprint；box023 的 ref motion 更支持双手 `[+1,0,0]`。

## 5. 对 E074 的影响判断

E062 palm normal 的正向意义：

- 它移除了 E041c 中明显 case-specific 的 box025 朝向先验。
- 对 box025，自计算结果等于默认值，说明算法 self-consistency 通过。
- 对 box023，它给 orientation reward 提供了更符合 ref motion 的方向项。

风险和限制：

- 它会让 CEM 更积极地把 wrist-local `+x` 轴朝向物体目标点，可能强化伸手/前扑倾向。
- E062 当时看到 box023 出现 “尝试搬箱 -> 摔倒 -> 爬起” 的 mixed 行为，说明它本身不能解决稳定持箱。
- E062-E067 结果发生在 E071 ctrl mapping bug 修复之前，所以不能用当时的摔倒现象直接否定 palm normal。

当前决策：

- **E074 主线保留 E062 palm normal**。
- 理由：E071/E073 可信结果已经包含 E062 `[+x,+x]`，移除它会破坏与 E073 的对照。
- 若要验证 palm normal 的独立影响，应作为单独 ablation，例如：
  - `E073 + reset palm_normal to E041c default`
  - `E073 + contact_hdmi_ori_weight=0`

该 ablation 不应挡在 E074A/E074C 前面。当前首要问题仍是 E073 的 post-2s hold/contact continuity，而不是 palm normal 本身。

## 6. 下一步实验约束

E074 第一批实验应保持 base 不变，只加一个新变量：

| Run | Base | 单一变量 |
|-----|------|----------|
| E074A | E073 | robot ctrl trust-region guard |
| E074C | E073 | hold/contact continuity reward |

评估必须保留 E073 对照口径：

- yaw err t=0.017/0.033。
- B1 pre-contact max foot z。
- first obj_err >25cm。
- first zero contact。
- frame100-145 contact。
- post2 contact。
- post2 obj_err max。
- post2 pelvis_z min。
- f100/f115/f130/f145/f160/f166/f180 视觉复核。

成功标准不能只看 pelvis_z；必须要求 f130/f145 视觉上仍保持持箱/托箱关系。
