# CORE4D 当前进度

> 历史完整备份：
> [progress_full_backup_through_E176_20260724.md](progress_archive/progress_full_backup_through_E176_20260724.md)
>
> 本文件只保留最近活跃实验的关键决策、结果和未决项；纯轮询记录已从
> 当前视图移除。

## E175：高保真 multi-geom 接入修复

- 修复了两个真实缺陷：
  1. robot–object 物理 pair 覆盖全部 `object_collision*`；
  2. PRG SDF 使用全部 collision geom 的 union。
- 高密度 surface-voxel proxy 为 `41–167 geoms`、
  `738–3006 pairs`；A100 Full 预计 52–66 小时。
- E175 Full 已停止，canary/部分 Full 证据保留；不修改 E174 完成日志和
  artifacts。

## E176：个位数非 box proxy

### 用户冻结约束

- bucket/desk 均使用粗粒度表面建模，精度可降低，但每个物体 collision
  geom 必须为个位数。
- A100 固定 GPUs `2,3,6,7`。
- Full case set 与 E174 完全相同（39 条），预算 `1024×32`。
- Full 只在 6/6 runtime、geom≤9、pair≤162、6/6 median plan
  time≤3s 全部满足后启动。

### Proxy 与 correctness

- geoms：`7/6/7/6/7/9`；pairs：
  `126/108/126/108/126/162`。
- 39/39 scene compile、pair matrix、authority parity PASS。
- 6/6 visual review approved；bucket 空腔/顶沿和 desk 桌下开口保留。
- ref-FK contact fidelity 39/39 PASS；六对象 grouped p90 均≤8cm。
- 运行时完整记录 `object_collision_sdf_mode=union` 和
  `object_collision_sdf_batch_groups=true`。

### 等价执行优化

- tick-local exact tuple cache：相同 robot-geom tuple 每 tick 只算一次。
- per-geom group batching：一次计算本 tick 所需 robot geoms，再按原组
  取 min；默认关闭，仅 E176 显式开启。
- sphere/capsule/mesh 与 legacy 逐 world 数值完全相等；旧 cache、
  E175 chunked-SDF、E176 lowgeom 回归全部 PASS。
- Prod3 singleton composition 因 frozen config 中 tuple 已相同而零命中，
  核实后立即停止；该结果不计入正式 gate。

### A100 canary 结果

| Case | Prod1 no-cache | Prod2 exact-cache | Prod4 group-batch | Gate |
|---|---:|---:|---:|---|
| bucket003 | 3.7189s | 3.0756s | 2.9865s | PASS |
| bucket004 | 3.7202s | 3.1359s | 3.0079s | FAIL |
| bucket007 | 3.5832s | 3.0912s | 2.9659s | PASS |
| bucket009 | 3.6031s | 3.0876s | 2.9798s | PASS |
| bucket010 | 3.6694s | 3.1308s | 3.0572s | FAIL |
| desk007 | 3.6163s | 3.0745s | 2.9629s | PASS |

- Prod4 runtime gate：PASS 6/6。
- Prod4 throughput gate：FAIL 4/6；相对 prod1 提速16.68–19.69%。
- 6/6 runtime config 的 geom/mode/batch flag 均与 manifest 一致。
- Full launcher 的阻断测试按预期 exit=1；Full 从未启动。

### 决策与证据

- 按三次失败协议停止继续微调，不改 3s 阈值。
- 正式日志：
  [235_E176_lowgeom_proxy_canary_results.md](log/235_E176_lowgeom_proxy_canary_results.md)
- 计划：
  [192_E176_lowgeom_nonbox_proxy_plan.md](plan/192_E176_lowgeom_nonbox_proxy_plan.md)
- baseline：
  `results/E176/baselines/{no_cache_prod1,cache1_prod2,batch1_prod4}/`
- 当前未决：保持 3s 门则需新的 fused/kernel 路线；若要放行 Full，需用户
  显式修改吞吐准入策略。

## 2026-07-24 收尾验证

- E176 production builder `--preflight --require-review` PASS：
  39/39、review pending=0、max geom=9、max pairs=162。
- log INDEX 已收录 235；Tracker 已更新为 Full blocked。
- Prod4 exact pull 完成，远端无本轮 CEM/tmux 残留。
- 最终 runtime validator 复跑仍为PASS 6/6；`workspace/core4d/results`
  保持项目级软链接。GPU2/6/7=`0–4MiB`，GPU3仅原有`635MiB`进程。
- 工作树中的 `viser_review_player.py`、RL export 等用户既有改动未触碰；
  Claims未全部通过，按技能规则未 commit/push。
- 归档前完整 progress 为1610行，SHA256：
  `dab9db1912c5deab8fa9c6287f002827275e87b0f10fdcd8c8918ba510694454`。

## 2026-07-24 本地 geom 可视化复核

- 按用户要求执行纯本地可视化，不启动 CEM。已打开 E176 六对象四视角
  montage：蓝色为真实 mesh，红色为 collision boxes，计数仍为
  `7/6/7/6/7/9`。
- 初看 desk007 的桌面、左右支撑和桌下通道均被保留；bucket proxy 为允许
  的粗盒近似，需继续查看横截面确认视觉上的叠层没有填实中心空腔。
- 横截面 montage 复核：5/5 bucket 在 mid/upper-Z 的中心均存在连续白色
  空腔，红色 boxes 主要落在底板、侧壁和顶沿；没有实心填桶。粗化最明显
  的是 bucket003/007，局部红盒越过圆弧外轮廓，属于已记录 phantom 覆盖。
- desk007 单物体四视角复核：9 boxes 覆盖大平板、两侧竖向支撑和横梁；
  中央桌下开口贯通，没有用大 box 桥接。局部脚/横梁有外扩，但没有新增
  第10个 geom。

## E177：三种 bucket 语义低 geom proxy（规划中）

- 用户新决策按 case 数收敛 scope。因用户后句明确保留 bucket007，当前按
  前句笔误解释为弃掉 bucket009(1条)+bucket010(2条)，desk 暂不考虑；
  E177 authority 为 bucket003(9)+bucket004(4)+bucket007(14)=27条。
- 候选设计：bucket004 使用单 box；bucket003/007 使用“盖子1 box +
  桶身3–4 boxes”。关键约束是桶身 boxes 应组成侧壁/分段外壳，而不是用
  一个实心大 box 填掉空腔；下一步先量化 mesh 主轴/AABB与E176现有box，
  再冻结 E177 计划，当前不启动CEM。
- Mesh/现有proxy量化：bucket003 mesh extents约
  `0.541×0.763×0.466m`，bucket004约`0.323×0.462×0.305m`，
  bucket007约`0.547×0.574×0.570m`。E176当前分别7/6/7 boxes；
  003/007横截面中段是完整外周，若桶身只用3 boxes会留下一个大碰撞缺口，
  因此建议冻结为“盖子1 + 四侧壁4 = 5 geoms”；四侧壁可带轻微倾角拟合
  梯形，仍保留空腔。bucket004按用户指定用mesh AABB单box。
- 已核对 scene-template policy：非box新proxy必须保持
  `manual_review_required`，完成 object-only mesh/collision overlay 后才
  能 `approve_clean`；MuJoCo load不能替代人工审查。现有E175 solid-AABB
  helper和E176 renderer均可复用，但E177必须写新sidecar/manifest，不能
  覆盖E176。
- 新增 E177 计划193并加入Tracker：scope冻结为27条
  `bucket003=9/bucket004=4/bucket007=14`；geom冻结为`5/1/5`，其中
  003/007为盖子+四个可倾斜侧壁，004为用户指定实心AABB。当前只授权
  Gate A/B 本地建模、contact audit和视觉review，不启动CEM。
- 20万mesh surface sample截面确认：bucket003中段外轮廓近似直壁
  `~0.526×0.754m`；bucket007沿z明显收放，x宽从lower约0.41m增至
  mid约0.545m再收至upper约0.45m，而y深约0.56m稳定。单个倾斜x侧壁只能
  拟合一段线性梯形，E177将优先覆盖手接触更可能出现的中上段，并把
  近底部收缩视为允许的粗近似；视觉/contact gate负责否决过大phantom。
- 结合用户允许bucket004单实心box及E174腿进入空心proxy的失败机制，将
  E177语义冻结修正为“分层实心桶身”：bucket003盖1+body3=4 geoms，
  bucket007盖1+body4=5 geoms，bucket004=1 geom；对应pairs=`72/90/18`
  （按003/007/004顺序）。该选择会填桶腔，是为阻止腿穿入的显式物理取舍。
- 已实现 E177 semantic proxy 与本地回归入口：003/007用确定性20万表面采样
  按z分层拟合robust XY实心截面，层间4mm overlap，顶段独立lid；004使用
  visual mesh精确AABB。所有proxy限制在mesh整体AABB±5mm，且明确要求覆盖
  mesh中心，防止无意退回空心wall方案。尚未生成sidecar/CEM。
- E177 proxy单测首轮PASS：geom=`4/1/5`；mesh→proxy p90：
  bucket003=`5.28cm`、bucket004=`2.17cm`、bucket007=`4.28cm`，均低于
  8cm；proxy→mesh p90约`9.43/2.99/9.76cm`，实心阶梯内部/层间人工表面会
  抬高该方向指标，后续以overlay和ref-contact为主要gate。
- 新增 E177 27-case production builder：只从E174 39-row authority筛选
  003/004/007，写独立sidecar/override/snapshot/full+3-case canary
  manifests；硬断言distribution=`9/4/14`、compiled pair=`18×N`、
  override parity和非proxy轴signature。review初始保持pending，不串联CEM。
- 2026-07-24续跑确认：按用户后半句继续保留bucket007，将前句解释为弃掉
  bucket009/010；E177仍只做本地preflight、visual/contact gate，不启动CEM。
  当前先验证builder运行与27-case分布，再生成003/004/007三对象overlay供人工审查。
- E177本地语法、单测及27-case builder preflight均PASS：分布严格为
  `bucket003=9/bucket004=4/bucket007=14`，scene compile `27/27`，
  geom=`4/1/5`，robot-object pair=`72/18/90`，override/case-set审计通过；
  三对象review仍为pending，未启动任何canary或Full CEM。
- E177 overlay首跑失败1次：复用的E175 renderer在main中硬编码
  `expected six object review rows`，三对象输入被拒绝；根因是renderer输入
  contract而非scene/proxy。下一步将其改为输入驱动，同时维持E175六对象兼容，
  然后重跑；不重复相同命令。
- 已将软件overlay renderer改为非空输入驱动：montage行数按对象数动态计算，
  6对象时保留原E175文件名，其他数量使用`N_object`命名；PASS条件改为所有
  输入对象render成功。尚待E177三对象实际重跑及E175六对象回归。
- E177三对象overlay重跑PASS并已人工打开总览。bucket004单AABB与mesh外包络
  基本一致；bucket003/007的分层实心proxy都跟随纵向截面收放，未回退为单个
  大AABB。总览可见层间水平人工面和圆弧角点phantom，需结合横截面图继续判断
  是否在用户允许的粗精度范围内；review仍保持pending。
- 横截面人工审查否决当前003/007版本：两者真实截锥轴均为object-local `+Y`，
  不是Z；当前沿Z分层虽然p90过线，但语义错误，不能送CEM。20万surface samples
  的Y五段截面显示003的XZ宽度约`0.343→0.524m`、007约`0.402→0.528m`，
  且+Y末段采样显著增多，符合盖子在+Y端。下一版改为沿Y分3/4段桶身并在
  +Y端单独拟合lid，geom总数仍为4/1/5。
- 已先修订E177计划193再改代码：Geometry/C5/Gate A均明确真实桶轴为`+Y`、
  body拟合robust XZ、lid位于+Y末端，并修正renderer CLI参数名。该视觉否决
  作为Gate B证据保留，不会把错误Z分层版本标记approve。
- E177 proxy已改为沿local Y分层：003/007分别3/4个body slabs，+Y末端各
  1个25mm lid，robust拟合XZ且4mm overlap；单测PASS，geom仍`4/1/5`。
  新mesh→proxy p90为`4.83/2.17/3.96cm`，较错误Z轴版003/007的
  `5.28/4.28cm`均改善；003 proxy→mesh也由约9.43cm降至6.87cm。
- 使用`--overwrite`只重建E177派生sidecar后，27-case preflight再次PASS：
  compile `27/27`、分布`9/4/14`、pair=`72/18/90`、最大5 geoms；
  新proxy→mesh p90为`6.77/3.03/9.59cm`。review仍为3个pending，
  现需覆盖旧Z轴overlay并重新人工审查。
- 新overlay总览已覆盖并人工复核：003/007不再是Z向“千层饼”，body boxes
  沿真实桶轴Y由小到大排列，+Y端独立薄lid可见；004仍为单AABB。整体轮廓
  与截锥趋势一致，主要剩余误差是box角点包住圆形截面的预期phantom区域，
  下一步看截面图确认层间无轴向漏缝。
- 横截面复核：003/007的Y向相邻body层有4mm overlap，无轴向碰撞漏缝；
  25mm lid覆盖+Y端，XZ截面外包圆弧产生的角点phantom属于用户已接受的低精度
  box近似。暂不自动`approve_clean`，等待用户看图确认。renderer的E175六对象
  兼容回归也PASS（6/6），且相关代码/计划`git diff --check`通过。

## 2026-07-24 E177视觉review第二轮否决

- 用户明确否决4/1/5版本：盖子建模差、003/007的3/4段桶身不足、box整体比
  mesh偏大。该判断与量化一致：当前proxy→mesh p90为003约6.8cm、007约
  9.6cm，主要来自圆截面外接box角点与单块lid AABB；review继续pending。
- 初步排查发现E176 runtime validator仍硬限制所有object collision geom必须
  为`mjGEOM_BOX`。因此在把主体改为变半径cylinder前，必须先确认PRG union SDF
  对cylinder的解析与group batching均正确；不能只因MuJoCo本身能加载cylinder
  就放行。当前不启动CEM。
- PRG源码确认cylinder路线当前不可直接用：`spider/config.py`的union resolver
  显式拒绝non-box，`mjwp.py`后续调用的也是batched `geom_box_union_sdf`，
  E175单测还断言sphere/non-box必须报错。若现在改cylinder会同时扩大到PRG解析、
  batched SDF、terminal gate及runtime validator，不适合作为本轮低geom修补。
  因此下一候选保持box-only，目标上限调整为8–9 geoms并通过inward fit消除外扩。
- 首次box候选网格搜索失败1次（进程143，约40s无输出）：组合数过多且每个候选
  都调用`trimesh.closest_point_naive`，计算复杂度过高。按三次失败协议不原样
  重跑；下一次缓存mesh采样并先用较小候选集/更少union surface samples筛选，
  再仅对top候选做高精度双向距离。
- KD-tree两阶段筛选首阶段完成。冻结8-geom折中而非顶到9：
  bucket003=`6 body + 2 lid strips`，body/lid inward scale=`0.94/0.88`、
  lid厚18mm，近似双向union p90约2.0cm；bucket007=`5 body + 3 lid strips`，
  scale=`0.82/0.82`、lid厚12mm，近似mesh→proxy/proxy→mesh p90约
  `3.3/3.8cm`。已先更新计划193，目标counts/pairs改为`8/1/8`和
  `144/18/144`，尚待高精度实现与视觉验证。
- 已实现8/1/8候选：object-specific body inward scale、2/3个Z向lid strips、
  薄lid与独立metadata；builder新增真正union外表面双向fidelity gate，过滤
  相邻box overlap中的内部面，003/007要求双向p90均≤4cm。manifest/summary
  同步记录union指标，proxy variant更新；尚未运行单测。
- 8/1/8高精度单测与27-case重建均PASS：compile `27/27`，pair精确为
  `144/18/144`，最大8 geoms。union mesh→proxy/proxy→mesh p90：
  bucket003=`1.82/1.92cm`，bucket004=`2.21/3.03cm`，
  bucket007=`3.31/3.76cm`；review仍pending，未启动CEM。
- 8/1/8四视角总览已生成并人工打开：003六段梯形更连续，2-strip lid不再是
  整块外接AABB；007五段主体明显inset，圆截面四角外扩大幅减少，3-strip lid
  能看到分段轮廓。局部蓝色mesh略伸出红proxy是主动inward fit的结果，
  需横截面确认没有形成大面积接触盲区。
- 现有横截面显示主体inward效果符合预期，但`XZ · mid Y`只切到桶身，没有
  单独展示+Y端lid；用户本轮重点正是lid，因此当前图证据仍不充分。将renderer
  扩展为2×3，新增`XZ · +Y lid`端面截面后再做人工判断，不据现图approve。
- renderer已扩为2×3截面：新增`XZ · -Y base`与`XZ · +Y lid`，保留mid-body
  和XY/YZ截面；E177三对象重渲染PASS。下一步直接检查lid端面分条是否显著
  改善圆角外扩，并继续保持manual review pending。
- 新端面图人工复核：003的2-strip lid与近矩形端面贴合，007的3-strip lid
  相比整块AABB显著削掉圆角phantom；007中段inward box在圆周cardinal方向约
  3cm欠覆盖，是消除原9cm级外扩的显式折中。E175 PRG multi-box union单测、
  E175/E177 py_compile及`git diff --check`全部PASS；未改PRG源码，未启动CEM。
- 为更充分响应用户“3–4段不够”，将007从候选的5 body进一步提高到
  `6 body + 3 lid strips = 9 geoms`，正好保持个位数上限；003仍为
  `6+2=8`。计划先更新为counts=`8/1/9`、pairs=`144/18/162`后再改代码，
  尚待最后一轮compile/visual。
- 最终8/1/9单测和27-case preflight PASS：compile `27/27`，pair为
  `144/18/162`。007 union mesh→proxy/proxy→mesh p90=`3.36/3.71cm`，
  仍低于4cm gate；003/004指标不变。review保持pending，准备最终重渲染。
- 最终8/1/9总览重渲染并人工打开：007六段轴向阶梯连续，三段lid仍清楚可辨，
  未出现新增大外扩；003与004保持上一轮表现。当前图与量化均完成，但按非box
  policy不自动approve，等待用户确认新版视觉后才进入contact fidelity gate。
- 最终2×3截面确认：003/007各六段body沿Y无漏缝；+Y lid端面分别由2/3条
  inward strips覆盖，圆角phantom较整块AABB显著缩小。2×3 renderer对E175
  六对象兼容回归6/6 PASS，最终`git diff --check`通过。E177本轮停在视觉
  review pending，未运行contact gate、canary或Full CEM。

## 2026-07-24 E177最终五段无盖方案

- 用户最终决定取消003/007独立lid，并将桶身从6段统一改为5段；目标geom
  counts=`5/1/5`，pairs=`90/18/90`。已先更新计划193：五段body覆盖完整
  local-Y（第5段包含+Y端面），保留object-specific inward XZ scale，
  不创建任何lid geom。尚未修改实现或启动CEM。
- 首次实现patch失败1次：builder的`proxy_variant`现有括号/缩进与补丁上下文
  不一致，apply_patch原子失败，未产生部分代码修改。下一步拆分为semantic、
  test、builder/renderer四个小patch，避免重复同一大patch。
- 分拆修改已开始：semantic constants改为`5/1/5`，删除lid参数常量；五段body
  的Y edges现覆盖完整mesh bounds，末段不再预留lid厚度，相邻段仍保留4mm
  overlap。lid生成代码与旧metadata尚待删除。
- semantic中的lid strip生成逻辑和相关metadata已全部删除；003/007 policy
  改为`semantic_five_solid_body_steps_no_lid`，004明确
  `has_separate_lid=false`。下一步更新单测、builder method/variant和renderer
  的“+Y end”标注。
- 单测contract已改为：003/007恰好5个连续`body_000..004`、无独立lid、
  首末段覆盖完整Y bounds。builder检查发现旧`proxy_variant`块还有历史
  额外缩进，下一patch将一并规范化并改为`five_body_steps_no_lid`。
- builder method/variant已改为five-step/no-lid r2，renderer标注改为
  `+Y end`。5/1/5语法与单测PASS；union mesh→proxy/proxy→mesh p90：
  003=`2.43/2.31cm`、004=`2.21/2.99cm`、007=`3.36/3.67cm`，均≤4cm。
  代码仅剩`has_separate_lid=false`语义字段，无lid geom生成路径。
- 27-case five-step/no-lid重建PASS：scene compile `27/27`，分布`9/4/14`，
  geom=`5/1/5`，pair=`90/18/90`，最大pair从上一版162降至90；review仍
  pending。三对象overlay也已覆盖重渲染PASS，下一步人工看最终图。
- 最终总览与2×3截面已人工打开：003/007均只显示5个沿Y连续body boxes，
  `+Y end`各仅由最后一个body覆盖，图中无独立lid/strip；inward拟合仍保留，
  007圆截面角点没有重新回到整块外接AABB。下一步只剩回归与diff检查。
- five-step/no-lid最终回归完成：E175 PRG multi-box union PASS、通用renderer
  六对象兼容6/6 PASS、`git diff --check` PASS。E177保持review pending，
  未启动contact gate、canary或Full CEM。
- 用户已显式批准E177 five-step/no-lid最终版作为后续版本。将新增log 236，
  Tracker改为“本地proxy批准、CEM未启动”，并把E177 project-specific bucket
  production profile写入`data-construction-v3-zh/SKILL.md`；该批准不等于
  contact gate/CEM成功。
- 审批落盘口径冻结：三对象review将写为`clean_reviewed/approve_clean`，
  reviewer=`user+codex`，时间`2026-07-24T11:03:21+08:00`，evidence指向
  E177 overlay目录；log 236必须把C6 ref-contact与C9 runtime保留为未执行，
  不把proxy批准夸大成CEM通过。
- 按用户澄清，不把E177固定参数当作skill通用规则。已新增log 236记录最终
  `5/1/5`无盖proxy、27/27 compile、双向p90、视觉批准及未执行的contact/CEM。
  `data-construction-v3-zh`升至v1.1.0，新增通用low-geom box准则：个位数优先、
  禁止明显外扩、毫米级接缝、双向union距离、overlay截面、physics/PRG一致；
  E177只作为实例引用。
- Tracker E177已指向log 236；三对象review正式落为
  `clean_reviewed/approve_clean`。builder以`--require-review`重跑PASS，
  `review_pending=[]`，仍为27 rows、`5/1/5` geoms、`90/18/90` pairs。
  log INDEX已重建，skill/log/tracker相关`git diff --check` PASS。
