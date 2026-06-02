# Core4D 数据构建侧科研贡献：让下游学习建立在可信数据上

_半页 PPT 口径 · 面向项目汇报_

---

## 一句话

我们围绕 Core4D 人-物交互数据，建立了一套从原始 mocap 到下游 CEM/RL 的数据构建方法：既能自动筛选高质量样本，也能构建可扩展的物体场景，并把数据质量证据传递给下游算法。

## 一张图

```mermaid
flowchart LR
    accTitle: Core4D 数据构建科研贡献
    accDescr: 该图展示 Core4D 数据构建从原始动作到下游学习样本的科研逻辑：先控制数据有效性，再形成可解释候选，最后进入 CEM/RL 验证。

    raw["原始 CORE4D<br/>mocap + 物体 mesh"] --> pipeline["统一数据构建 pipeline<br/>raw → retarget → handoff"]
    pipeline --> filter["数据清洗与过滤<br/>接触 / 姿态 / 碰撞"]
    filter --> scene["场景数据构建<br/>box / bucket / 非规则物体"]
    scene --> downstream["下游算法闭环<br/>CEM / RL"]

    pipeline --> c1["可复现"]
    filter --> c2["可解释"]
    scene --> c3["可扩展"]
    downstream --> c4["可评估"]

    classDef main fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef contribution fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    classDef output fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12

    class raw,pipeline,filter,scene main
    class c1,c2,c3,c4 contribution
    class downstream output
```

## 核心贡献

- **完整数据构建 pipeline**：从原始 mocap、物体 mesh、OmniRetarget、SPIDER 输入到 CEM/RL handoff，形成可复现的数据链路。
- **数据清洗/过滤方法**：用接触距离、多阈值候选、姿态安全、下肢-物体干涉和视觉审查过滤低质量样本，避免把数据问题传给算法。
- **场景数据构建能力**：重建可信 source scene template，并支持不同物体形态的场景变种；box 可自动构建，bucket/非规则物体可生成可审查 proxy。
- **下游算法闭环**：为 CEM/RL 提供带证据的候选库和统一 handoff，使下游结果能反向用于分析数据质量和样本可用性。

## 可以汇报的结论

当前最重要的进展不是“多跑了多少 case”，而是把 Core4D 数据构建变成了一个可复现、可解释、可扩展的研究基础设施：后续可以系统扩充物体类型和样本规模，并开展更公平的 retargeting / RL 算法比较。
