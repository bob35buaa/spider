// 面向人机协作搬运的动力学重定向 (E167A) — 汇报 PPT
// 风格参考 tmp/irip-template-blue.pptx 的蓝色主题
const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.defineLayout({ name: "W", width: 13.333, height: 7.5 });
p.layout = "W";

// ---- palette (from template theme1.xml) ----
const NAVY = "2D2D89";      // accent6 深蓝
const BLUE = "333399";      // accent2 蓝
const LBLUE = "A7C6E5";     // accent1 浅蓝
const ICE = "D0DFEF";       // accent5 冰蓝
const TEAL = "009999";      // hlink 强调
const INK = "1A1A2E";       // 正文深色
const GRAY = "6B6B7B";      // 次要
const WHITE = "FFFFFF";
const FH = "Microsoft YaHei"; // 标题
const FB = "Microsoft YaHei"; // 正文

const W = 13.333, H = 7.5, M = 0.7;

// helper: content slide with left accent bar + title
function head(s, title, kicker) {
  s.background = { color: WHITE };
  s.addShape(p.ShapeType.rect, { x: 0, y: 0, w: 0.18, h: H, fill: { color: NAVY } });
  if (kicker) s.addText(kicker, { x: M, y: 0.42, w: 11, h: 0.3, fontFace: FB, fontSize: 12, color: TEAL, bold: true, charSpacing: 2 });
  s.addText(title, { x: M, y: 0.66, w: 12, h: 0.7, fontFace: FH, fontSize: 28, color: NAVY, bold: true });
  s.addShape(p.ShapeType.line, { x: M, y: 1.5, w: 12.0, h: 0, line: { color: ICE, width: 1.5 } });
}
function foot(s, n) {
  s.addText("CORE4D 人机协作动力学重定向 · E167A", { x: M, y: H - 0.42, w: 8, h: 0.3, fontFace: FB, fontSize: 9, color: GRAY });
  s.addText(String(n), { x: W - 1.0, y: H - 0.42, w: 0.5, h: 0.3, fontFace: FB, fontSize: 9, color: GRAY, align: "right" });
}

// ============ Slide 1 — Title ============
let s = p.addSlide();
s.background = { color: NAVY };
s.addShape(p.ShapeType.rect, { x: 0, y: 5.55, w: W, h: 0.12, fill: { color: TEAL } });
s.addShape(p.ShapeType.rect, { x: 0, y: 5.67, w: W, h: 0.06, fill: { color: LBLUE } });
s.addText("PHYSICS-INFORMED RETARGETING FOR HUMAN–ROBOT CO-CARRYING", { x: M, y: 1.7, w: 11.5, h: 0.4, fontFace: FB, fontSize: 13, color: LBLUE, bold: true, charSpacing: 2 });
s.addText("面向人机协作搬运的动力学重定向", { x: M, y: 2.2, w: 12, h: 1.0, fontFace: FH, fontSize: 40, color: WHITE, bold: true });
s.addText("从抓取式采样 MPC 到协作式 Loco-Manipulation 的方法扩展", { x: M, y: 3.35, w: 12, h: 0.6, fontFace: FH, fontSize: 20, color: ICE });
s.addText([
  { text: "拟定版本 E167A", options: { bold: true, color: WHITE } },
  { text: "   |   基线 OmniRetarget · 方法 SPIDER(SBMPC)   |   CORE4D · G1 · SUGAR/Holosoma RL", options: { color: LBLUE } },
], { x: M, y: 6.05, w: 12, h: 0.5, fontFace: FB, fontSize: 14 });
s.addText("2026-06", { x: W - 2.0, y: 1.7, w: 1.3, h: 0.3, fontFace: FB, fontSize: 13, color: LBLUE, align: "right" });

// ============ Slide 2 — 背景与问题 ============
s = p.addSlide(); head(s, "背景与问题：为什么要改造 SPIDER", "BACKGROUND"); foot(s, 2);
const probs = [
  ["上游运动学重定向不够", "OmniRetarget 纯运动学，接触丰富的协作搬运下穿模/脚滑/接触失真；G1 较矮臂短，手只够到箱底、0% 侧面夹持。"],
  ["抓取式 SPIDER 不适配", "原 SPIDER/HDMI 面向单体灵巧抓取（把手点目标、无释放/姿态/穿透安全），直接迁到双臂大箱协作会在多处失效。"],
  ["我们的做法", "不换 SBMPC 优化器，重构其面向协作的任务建模栈：接触/穿透/释放/姿态/运动学/下游对齐六轴改造。"],
];
let yy = 1.85;
probs.forEach((it, i) => {
  s.addShape(p.ShapeType.roundRect, { x: M, y: yy, w: 0.55, h: 0.55, rectRadius: 0.27, fill: { color: i === 2 ? TEAL : BLUE } });
  s.addText(String(i + 1), { x: M, y: yy, w: 0.55, h: 0.55, align: "center", valign: "middle", fontFace: FH, fontSize: 22, color: WHITE, bold: true });
  s.addText(it[0], { x: M + 0.8, y: yy - 0.05, w: 11, h: 0.4, fontFace: FH, fontSize: 18, color: NAVY, bold: true });
  s.addText(it[1], { x: M + 0.8, y: yy + 0.38, w: 11.2, h: 0.8, fontFace: FB, fontSize: 14, color: INK });
  yy += 1.55;
});

// ============ Slide 3 — 方法：E167A = 5 组件 ============
s = p.addSlide(); head(s, "方法：E167A 由五个组件构成", "METHOD"); foot(s, 3);
s.addText("在 SBMPC(CEM) 之上逐步加入；前四项构成成熟参考层 E163，第五项对齐下游", { x: M, y: 1.55, w: 12, h: 0.35, fontFace: FB, fontSize: 13, color: GRAY, italic: true });
const comps = [
  ["①", "面接触带奖励", "窄对称 SDF 带，取代抓取点目标，适配平面大箱接触", "E158→E163"],
  ["②", "采样层穿透门 gateA", "CEM 采样层拒绝穿透样本，压住压入式深穿透", "E152/E153"],
  ["③", "释放相位衰减", "horizon 上对释放窗口衰减接触奖励，解决“放不开手”", "E155/E161"],
  ["④", "参考相对姿态重排序", "按相对蹲姿参考偏差筛选精英，修“贴箱即摔”", "E160"],
  ["⑤", "z-only 精修", "上游跟踪/惩罚改为仅 z，对齐下游 Holosoma 终止门", "E167A"],
];
const cw = 3.86, gap = 0.34, x0 = M;
comps.slice(0, 3).forEach((c, i) => card(s, x0 + i * (cw + gap), 2.1, cw, 2.0, c));
[comps[3], comps[4]].forEach((c, i) => card(s, x0 + (i + 0.5) * (cw + gap), 4.35, cw, 2.0, c));
function card(s, x, y, w, h, c) {
  s.addShape(p.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: "F4F8FC" }, line: { color: ICE, width: 1 } });
  s.addShape(p.ShapeType.roundRect, { x, y, w: 0.9, h: 0.62, rectRadius: 0.08, fill: { color: NAVY } });
  s.addText(c[0], { x, y, w: 0.9, h: 0.62, align: "center", valign: "middle", fontFace: FH, fontSize: 24, color: WHITE, bold: true });
  s.addText(c[3], { x: x + w - 1.7, y: y + 0.12, w: 1.6, h: 0.3, align: "right", fontFace: FB, fontSize: 10, color: TEAL, bold: true });
  s.addText(c[1], { x: x + 0.2, y: y + 0.7, w: w - 0.4, h: 0.5, fontFace: FH, fontSize: 16, color: BLUE, bold: true });
  s.addText(c[2], { x: x + 0.2, y: y + 1.2, w: w - 0.4, h: 0.7, fontFace: FB, fontSize: 12.5, color: INK });
}

// ============ Slide 4 — 消融主表 ============
s = p.addSlide(); head(s, "分步消融：逐组件量化（同一 3-case）", "ABLATION"); foot(s, 4);
const hdr = ["配置（逐步加组件）", "引入实验", "raw接触↑", "clean3↑", "穿透3mm↓", "几何穿透↓", "关节°↓", "EEFcm↓"];
const rows = [
  ["OmniRetarget(运动学基线)", "—", "0.961", "0.027", "0.606", "0.615", "0.00*", "0.00*"],
  ["spider-rubberhand", "E147/8", "0.607", "0.118", "0.283", "0.294", "3.86", "12.75"],
  ["+gateA  ②穿透门", "E152/3", "0.610", "0.122", "0.279", "0.274", "3.79", "12.65"],
  ["+面接触带 ①", "E158/9", "0.616", "0.460", "0.105", "0.030", "4.56", "15.97"],
  ["+姿态重排序 ④", "E160", "0.668", "0.519", "0.106", "0.038", "4.24", "14.65"],
  ["+释放衰减 ③", "E161", "0.716", "0.517", "0.137", "0.047", "4.49", "14.87"],
  ["+窄对称带 = E163", "E163", "0.750", "0.566", "0.123", "0.045", "4.30", "14.60"],
  ["+z-only = E167A（拟定）", "E167A", "0.778", "0.644", "0.097", "0.042", "4.29", "14.05"],
];
const colW = [3.0, 1.0, 1.25, 1.1, 1.2, 1.3, 1.0, 1.1];
const tbl = [hdr.map((t, i) => ({ text: t, options: { fill: { color: NAVY }, color: WHITE, bold: true, fontSize: 12, align: i === 0 ? "left" : "center", valign: "middle" } }))];
rows.forEach((r, ri) => {
  const last = ri === rows.length - 1;
  tbl.push(r.map((t, ci) => ({
    text: t,
    options: {
      fill: { color: last ? ICE : (ri % 2 ? "F4F8FC" : WHITE) },
      color: last ? NAVY : INK, bold: last || ci === 0 ? true : false,
      fontSize: 12, align: ci === 0 ? "left" : "center", valign: "middle",
      fontFace: FB,
    },
  })));
});
s.addTable(tbl, { x: M, y: 1.75, w: 12.0, colW, rowH: 0.46, border: { type: "solid", color: "DCE6F2", pt: 0.5 } });
s.addText("读法：面接触带把接触刷上去（clean3 0.12→0.46+），代价是 EEF 升到 23cm；姿态/释放/窄带逐步把 EEF 拉回 14.6cm 并清干净释放；E167A 参考层再优于 E163。  * 自评偏置",
  { x: M, y: 6.55, w: 12, h: 0.6, fontFace: FB, fontSize: 11.5, color: GRAY, italic: true });

// ============ Slide 5 — 结果① 参考层 ============
s = p.addSlide(); head(s, "结果①：参考层物理指标明确超越 OmniRetarget", "RESULT · REFERENCE"); foot(s, 5);
s.addText("clean8 统一物理口径（E156）·  0 fall · tracking 8/8", { x: M, y: 1.55, w: 12, h: 0.35, fontFace: FB, fontSize: 13, color: GRAY, italic: true });
const stats = [
  ["手-物物理穿透", "0.576", "0.202", "↓ −0.375"],
  ["in-mask 真实接触", "0.034", "0.153", "↑ ×4.5"],
  ["E167A vs E163 接触", "0.566", "0.644", "↑ clean3"],
];
stats.forEach((st, i) => {
  const x = M + i * 4.1;
  s.addShape(p.ShapeType.roundRect, { x, y: 2.2, w: 3.8, h: 3.3, rectRadius: 0.1, fill: { color: i === 1 ? NAVY : "F4F8FC" }, line: { color: ICE, width: 1 } });
  const fg = i === 1 ? WHITE : NAVY, sub = i === 1 ? LBLUE : GRAY;
  s.addText(st[0], { x: x + 0.2, y: 2.45, w: 3.4, h: 0.5, align: "center", fontFace: FH, fontSize: 15, color: fg, bold: true });
  s.addText("OmniRT  " + st[1], { x: x + 0.2, y: 3.15, w: 3.4, h: 0.4, align: "center", fontFace: FB, fontSize: 13, color: sub });
  s.addText(st[2], { x: x + 0.2, y: 3.6, w: 3.4, h: 0.9, align: "center", fontFace: FH, fontSize: 44, color: i === 1 ? TEAL : BLUE, bold: true });
  s.addText("SPIDER", { x: x + 0.2, y: 4.5, w: 3.4, h: 0.3, align: "center", fontFace: FB, fontSize: 11, color: sub });
  s.addText(st[3], { x: x + 0.2, y: 4.85, w: 3.4, h: 0.45, align: "center", fontFace: FH, fontSize: 16, color: i === 1 ? WHITE : TEAL, bold: true });
});
s.addText("OmniRetarget 的“接触”大半是压入式穿透；SPIDER 把穿透压低一个量级、真实接触提高 4–5 倍，且不牺牲跟踪与稳定。", { x: M, y: 5.8, w: 12, h: 0.5, fontFace: FB, fontSize: 13, color: INK });

// ============ Slide 6 — 结果② 下游 ============
s = p.addSlide(); head(s, "结果②：下游 RL 两个独立项目聚合均胜", "RESULT · DOWNSTREAM"); foot(s, 6);
// SUGAR
s.addText("SUGAR（refiner RL，7-case 同口径，Holosoma-like success / 448）", { x: M, y: 1.7, w: 12, h: 0.35, fontFace: FH, fontSize: 15, color: BLUE, bold: true });
const sugar = [["OmniRetarget", "13", GRAY], ["spider E163", "56", TEAL], ["spider E167A", "51", TEAL]];
sugar.forEach((d, i) => {
  const x = M + i * 4.1;
  s.addShape(p.ShapeType.roundRect, { x, y: 2.15, w: 3.8, h: 1.5, rectRadius: 0.08, fill: { color: "F4F8FC" }, line: { color: ICE, width: 1 } });
  s.addText(d[1] + " /448", { x: x + 0.2, y: 2.3, w: 3.4, h: 0.8, align: "center", fontFace: FH, fontSize: 36, color: d[2], bold: true });
  s.addText(d[0], { x: x + 0.2, y: 3.15, w: 3.4, h: 0.4, align: "center", fontFace: FB, fontSize: 13, color: NAVY });
});
s.addText("SPIDER 两版本均 ≈ 4× 于 OmniRetarget", { x: M, y: 3.75, w: 12, h: 0.35, fontFace: FB, fontSize: 12.5, color: GRAY, italic: true });
// Holosoma
s.addText("Holosoma（WBT RL，handbox case-relative 5-case 平均 success）", { x: M, y: 4.35, w: 12, h: 0.35, fontFace: FH, fontSize: 15, color: BLUE, bold: true });
s.addShape(p.ShapeType.roundRect, { x: M, y: 4.8, w: 12.0, h: 1.45, rectRadius: 0.08, fill: { color: NAVY } });
s.addText([{ text: "32.5%", options: { color: LBLUE, fontSize: 34, bold: true } }, { text: "  →  ", options: { color: WHITE, fontSize: 26 } }, { text: "48.1%", options: { color: TEAL, fontSize: 40, bold: true } }],
  { x: M + 0.4, y: 4.95, w: 7, h: 1.15, align: "left", valign: "middle", fontFace: FH });
s.addText([{ text: "+15.6pp", options: { color: WHITE, fontSize: 30, bold: true } }, { text: "\nOmniRetarget → SPIDER（主胜点 Box021）", options: { color: LBLUE, fontSize: 12 } }],
  { x: M + 7.3, y: 4.95, w: 4.5, h: 1.15, align: "right", valign: "middle", fontFace: FB });
s.addText("仍 case-dependent：上游 E163→E167A 在 SUGAR 上为重分布（聚合 56→51）。Holosoma 以 handbox 为准（rubberhand 口径有数据完整性问题，不采信）。",
  { x: M, y: 6.45, w: 12, h: 0.5, fontFace: FB, fontSize: 11.5, color: GRAY, italic: true });

// ============ Slide 7 — 结论 ============
s = p.addSlide();
s.background = { color: NAVY };
s.addShape(p.ShapeType.rect, { x: 0, y: 0, w: W, h: 0.12, fill: { color: TEAL } });
s.addText("结论", { x: M, y: 0.7, w: 11, h: 0.7, fontFace: FH, fontSize: 30, color: WHITE, bold: true });
const concl = [
  ["双维度全胜", "参考层物理指标（穿透↓、真实接触↑）+ 两个独立下游 RL 项目聚合，均超越 OmniRetarget。"],
  ["方法论洞察", "可恢复性不对称：参考层更优不保证下游单 case 更优——上游赢≠下游赢是结构性的。"],
  ["下一步", "把消费端物理、离硬门动态余量、抬升语义放进上游目标与评测；与 DynaRetarget 长 horizon 优化器正交可叠加。"],
];
let cy = 2.0;
concl.forEach((c, i) => {
  s.addShape(p.ShapeType.roundRect, { x: M, y: cy, w: 0.5, h: 0.5, rectRadius: 0.25, fill: { color: TEAL } });
  s.addText(String(i + 1), { x: M, y: cy, w: 0.5, h: 0.5, align: "center", valign: "middle", fontFace: FH, fontSize: 18, color: WHITE, bold: true });
  s.addText(c[0], { x: M + 0.75, y: cy - 0.05, w: 11, h: 0.45, fontFace: FH, fontSize: 19, color: WHITE, bold: true });
  s.addText(c[1], { x: M + 0.75, y: cy + 0.42, w: 11.3, h: 0.8, fontFace: FB, fontSize: 14, color: ICE });
  cy += 1.5;
});
s.addText("E167A = ①面接触带 + ②穿透门 + ③释放衰减 + ④姿态重排序 + ⑤z-only", { x: M, y: 6.75, w: 12, h: 0.4, fontFace: FB, fontSize: 13, color: LBLUE, bold: true, italic: true });

p.writeFile({ fileName: "/mnt/public/usr/yancilin/work_dir/embodied/spider/workspace/core4d/report/0624/E167A_汇报.pptx" }).then(f => console.log("wrote", f));
