#!/usr/bin/env python3
"""Build E167A ablation xlsx (dependency-free; no openpyxl/pandas needed).

All numbers are sourced from workspace/core4d/results/*/eval/*_method_summary.tsv
(strict eval, source-of-truth), the SUGAR 7-case comparison doc, and E166 log 218.
Source paths are written into each sheet's footer rows.
"""
import zipfile, xml.sax.saxutils as sx, os

NA = "—"

# ---------------------------------------------------------------- sheet data
# Sheet 1: OmniRetarget vs spider_E163 vs spider_E167A (reference 3-case + downstream SUGAR 7-case)
sheet1 = [
    ["版本", "raw接触↑", "clean3接触↑", "物理穿透3mm↓", "几何穿透2mm↓", "关节误差°↓", "EEF误差cm↓",
     "SUGAR成功/448↑", "completion↑", "搬运距离m↑"],
    ["OmniRetarget", 0.961, 0.027, 0.606, 0.615, 0.00, 0.00, 13, 0.103, 0.164],
    ["spider_E163", 0.750, 0.566, 0.123, 0.045, 4.30, 14.60, 56, 0.203, 0.443],
    ["spider_E167A", 0.778, 0.644, 0.097, 0.042, 4.29, 14.05, 51, 0.263, 0.608],
    [],
    ["注", "参考层指标取同一 3-case（box023_person2/box021_029_p2/box004_083_p2）；E163 取 E163 full 重评，E167A 取 E167 cem_metrics eval 同 3-case 子集；下游取 SUGAR 7-case"],
    ["注", "OmniRetarget 的 raw接触含大量压入式穿透（physPen3=0.606）；其 关节/EEF 误差≈0 是 self-eval 偏置(qpos_ref=qpos)，非真实优势"],
    ["注", "E167A 参考层亦优于 E163（raw 0.750→0.778、clean3 0.566→0.644、physPen3 0.123→0.097、EEF 14.60→14.05），但下游为重分布（聚合 56→51）"],
    ["注", "Holosoma handbox 5-case（聚合 SPIDER）：OmniRetarget 32.5% vs SPIDER 48.1%（+15.6pp），未拆 E163/E167A"],
    ["来源", "results/E163/.../e163_method_summary.tsv ；results/E167/holosoma_zonly/eval/cem_metrics/full/e167_arm_metrics.tsv ；"
             "SUGAR-private/docs/log/CORE4D_OMNIRETARGET_VS_SPIDER_HOLOSOMA_LIKE_COMPARISON_CN.md"],
]

# Sheet 2: 主消融表 — 同一 3-case，逐步加组件
sheet2 = [
    ["配置（逐步加组件）", "引入实验", "raw接触↑", "clean3接触↑", "物理穿透3mm↓", "几何穿透2mm↓",
     "关节误差°↓", "EEF误差cm↓", "组件"],
    ["OmniRetarget(运动学基线)", "基线(E156)", 0.961, 0.027, 0.606, 0.615, 0.00, 0.00, "—"],
    ["spider-rubberhand(无门/无带)", "E147/E148", 0.607, 0.118, 0.283, 0.294, 3.86, 12.75, "基线"],
    ["+gateA", "E152/E153", 0.610, 0.122, 0.279, 0.274, 3.79, 12.65, "②穿透门"],
    ["+surfaceBand-A(宽单边带)", "E158", 0.683, 0.518, 0.120, 0.018, 5.02, 23.42, "①面接触带"],
    ["+surfaceBand-A2(去penalty)", "E159", 0.616, 0.460, 0.105, 0.030, 4.56, 15.97, "①"],
    ["+postureRerank", "E160", 0.668, 0.519, 0.106, 0.038, 4.24, 14.65, "④姿态重排序"],
    ["+releaseDecay", "E161", 0.716, 0.517, 0.137, 0.047, 4.49, 14.87, "③释放衰减"],
    ["+narrowBand = E163", "E163", 0.750, 0.566, 0.123, 0.045, 4.30, 14.60, "①窄对称带"],
    ["+z-only = E167A（拟定版本）", "E167A", 0.778, 0.644, 0.097, 0.042, 4.29, 14.05, "⑤z-only"],
    [],
    ["注", "rubberhand→E163 各行取自 E163 full 统一重评（同 3-case：box023_person2/box021_029_p2/box004_083_p2，口径 core4d-e154-physics-contact-v1）；E167A 行取自 E167 cem_metrics eval 同 3-case 子集（baseline 行与 E163 完全一致，验证可比）"],
    ["注", "OmniRetarget 关节/EEF≈0 为 self-eval 偏置；面接触带使 EEF 误差升高(12.6→23.4cm)即接触换跟踪的代价，E163 收敛到 14.6cm"],
    ["注", "E167A 参考层亦优于 E163(raw 0.750→0.778, clean3 0.566→0.644, physPen3 0.123→0.097, EEF 14.60→14.05)，但下游为重分布(SUGAR 7-case 聚合 56→51，见 sheet「F_zonly下游」)"],
    ["注", "jerk 指标判别力弱(类似 EEF)已移出主表；其数据见 sheet「G_zonly参考层E167eval」与「H_平滑度E166」"],
    ["来源", "results/E163/.../e163_method_summary.tsv ；results/E167/holosoma_zonly/eval/cem_metrics/full/e167_arm_metrics.tsv"],
]

# Sheet 3: B_gateA穿透门 (多 case 集)
sheet3 = [
    ["B. 采样层穿透门 gateA（组件②）", "", "", "", ""],
    ["口径/case数", "对比", "物理接触/clean3", "深穿透(<-5mm)/physPen3", "fall / strict"],
    ["3-case (E152)", "baseline → +gateA", "物理接触 0.349→0.311", "深穿透 0.553→0.347 (−0.206)", "0/0"],
    ["3-case 扫描 (E153)", "门关→门开 sweet spot(−0.010,0.10)", "clean3 接触 −0.017(近零损失)", "深穿透 −0.172", "3/3 strict"],
    ["3-case (E156)", "rubberhand → +gateA", "clean3 0.118→0.122", "physPen3 0.283→0.279", "0/3"],
    ["clean8 (E156,8case)", "rubberhand → +gateA", "clean3 0.145→0.153", "physPen3 0.210→0.202", "0/8"],
    [],
    ["读法", "gateA 主效在深穿透：3-case 深穿透(<-5mm) 0.553→0.347；接触近零损失。clean8 仅 physPen3 0.210→0.202 但 8-case 不与主表 3-case 可比"],
    ["来源", "results/E152/axis1_hand_object_physics_gate/eval/full/e152_method_summary.tsv"],
    ["来源", "results/E153/gate_threshold_sweep/eval/full/e153_combo_summary.tsv"],
    ["来源", "results/E156/clean8_gate_decay/eval/full/e156_method_metrics.tsv (按 3-case 过滤)"],
]

# Sheet 4: C_姿态重排序 (E160 3-case)
sheet4 = [
    ["C. 参考相对姿态重排序（组件④）— 同 3-case", "tracked", "fall", "pz末端误差↓", "clean3接触", "释放误接触"],
    ["gateA+surfaceBand-A2", "2/3", 1, 0.172, 0.460, 0.083],
    ["+postureRerank", "3/3", 0, 0.017, 0.519, 0.271],
    [],
    ["读法", "修复 fall(1→0)、末端高度误差 0.172→0.017、保接触；但 release 副作用 0.083→0.271 → 引入组件③释放衰减"],
    ["来源", "results/E160/posture_rerank/eval/full/e160_method_summary.tsv"],
]

# Sheet 5: D_释放衰减 (E161 clean8)
sheet5 = [
    ["D. 释放相位衰减（组件③）— clean8 (8case)", "释放误接触3mm↓", "clean3接触", "physPen3", "tracked/fall"],
    ["M0 postureRerankA", 0.188, 0.477, 0.116, "8/8, 0"],
    ["M1 +releaseDecay", 0.030, 0.470, 0.128, "8/8, 0"],
    ["M2 strictMask(仅当前帧,对照)", 0.160, 0.427, 0.150, "8/8, 0"],
    [],
    ["读法", "horizon 级衰减(M1)释放误接触 −0.158 且接触几乎不损；仅当前帧 gate(M2)不足(残留0.160) → 衰减须作用于 horizon 释放帧"],
    ["来源", "results/E161/surface_release_ablation/eval/full/e161_method_summary.tsv"],
]

# Sheet 6: E_窄带RLsafe (E163 clean8)
sheet6 = [
    ["E. 窄对称带收尾 RL-safe 验证（组件①）— clean8 (8case)", "raw接触↑", "clean3接触↑", "释放误接触↓", "通过"],
    ["spider-rubberhand", 0.481, 0.149, 0.000, "—"],
    ["+releaseDecay(E161链)", 0.637, 0.470, 0.030, "—"],
    ["narrowBand = E163", 0.714, 0.544, 0.012, "7/8"],
    [],
    ["读法", "窄对称带把 raw in-mask 接触 0.637→0.714、clean3 0.470→0.544、释放误接触 0.030→0.012；8 case 中 7/8 过 raw-contact RL-safe 硬门"],
    ["注", "唯一未过：box004_082_p1（raw −0.098，低于 −0.05 门）"],
    ["来源", "results/E163/narrow_surface_band/eval/clean8/e163_method_summary.tsv"],
]

# Sheet 7: F_zonly下游 (SUGAR 7-case)
sheet7 = [
    ["F. z-only 精修（组件⑤）— 下游 SUGAR 7-case (64 attempts/case, Holosoma-like success)", "E163", "E167A"],
    ["box004_083_p2", 0.078, 0.016],
    ["box004_082_p1", 0.000, 0.156],
    ["box004_083_p1", 0.000, 0.312],
    ["box021_029_p2", 0.734, 0.047],
    ["box021_035_p1", 0.062, 0.266],
    ["box021_035_p2", 0.000, 0.000],
    ["box023_person2", 0.000, 0.000],
    ["合计 (/448)", 56, 51],
    [],
    ["读法", "z-only 是重分布：改善 box004_082/083_p1、box021_035_p1，但 box021_029_p2 回退；聚合 56→51 基本持平，E167A completion/搬运距离最高(0.263/0.608m)"],
    ["注", "z-only 设计目标是对齐 Holosoma z 终止门(BadTrackingZOnly@0.25m)，参考层非其主战场"],
    ["来源", "SUGAR-private/docs/log/CORE4D_OMNIRETARGET_VS_SPIDER_HOLOSOMA_LIKE_COMPARISON_CN.md"],
]

# Sheet (new): z-only 参考层消融 (E167 cem_metrics eval；baseline=E163 vs E167A/B1/B2；含 jerk)
sheet_zref = [
    ["z-only 参考层消融（E167 cem_metrics eval）", "case集", "raw接触↑", "clean3↑", "physPen3↓", "geom2↓",
     "关节°↓", "EEFcm↓", "qpos_jerk_p95↓", "trackbody_jerk_p95↓"],
    ["baseline (=E163)", "3-case", 0.750, 0.566, 0.123, 0.045, 4.30, 14.60, 3038, 569],
    ["baseline (=E163)", "7-case", 0.720, 0.534, 0.128, 0.049, 4.22, 14.82, 3524, 561],
    ["E167A (z-only,拟定版本)", "3-case", 0.778, 0.644, 0.097, 0.042, 4.29, 14.05, 3182, 580],
    ["E167A (z-only,拟定版本)", "7-case", 0.760, 0.587, 0.124, 0.057, 4.20, 14.12, 3597, 540],
    ["E167A+B1 (z-only CEM平滑)", "3-case", 0.793, 0.578, 0.140, 0.025, 4.31, 15.07, 2929, 513],
    ["E167A+B1 (z-only CEM平滑)", "7-case", 0.773, 0.541, 0.154, 0.031, 4.22, 14.85, 3331, 546],
    ["E167A+B2 (z-only后处理)", "3-case", 0.773, 0.644, 0.094, 0.046, 4.29, 14.05, 3160, 548],
    ["E167A+B2 (z-only后处理)", "7-case", 0.758, 0.587, 0.123, 0.059, 4.20, 14.12, 3579, 511],
    [],
    ["读法", "E167A 相对 baseline(E163) 在 3-case 参考层全面更优(raw↑/clean3↑/physPen3↓/EEF↓)，jerk 持平；B1/B2 进一步降 jerk（B1 qjerk 2929 最低）"],
    ["注", "7-case = clean8 去 box026_139_p1；baseline 行 == E163，验证与主表 E163 一致(3-case raw 0.750/clean3 0.566)"],
    ["来源", "results/E167/holosoma_zonly/eval/cem_metrics/full/e167_arm_metrics.tsv (+ e167_arm_summary.tsv)"],
]

# Sheet 8: G_平滑度E166 (jerk 实际所在；E166 自己的 3-case，排除 box023)
sheet8 = [
    ["G. 平滑度消融（E166 foot-smooth, 3-case: box004_082_p1/box004_083_p2/box021_035_p2, 排除box023）",
     "pass", "raw接触", "clean3接触", "pen3", "qpos_jerk_p95↓", "trackbody_jerk_p95↓", "ankle_acc_max↓", "foot_slip"],
    ["baseline", "3/3", 0.653, 0.445, 0.140, 4388.1, 579.0, 46.0, 0.969],
    ["B1(CEM平滑)", "3/3", 0.654, 0.503, 0.105, 4494.4, 495.1, 47.8, 1.105],
    ["B2(后平滑only)", "2/3", 0.625, 0.457, 0.115, 1987.8, 315.4, 31.4, 1.014],
    ["A(CEM足约束)", "3/3", 0.748, 0.548, 0.131, 4308.5, 519.6, 48.5, 1.120],
    ["A_B2_postSmooth", "3/3", 0.777, 0.618, 0.101, 1907.0, 299.1, 31.5, 1.122],
    ["AplusB(A+B1)", "3/3", 0.730, 0.594, 0.095, 4536.3, 575.4, 59.9, 1.048],
    [],
    ["读法", "这是 qpos/trackbody jerk 唯一被计算的地方（E166 分支）。A_B2_postSmooth 降抖最强(jerk 4388→1907,↓57%)且接触最高，但其 3-case 与主消融表不同(排除box023)，且 E166 不在 E167A 内(并列分支)"],
    ["注", "下游：A_B2_postSmooth SUGAR 任务成功仅 1/4(box021_029_p2 0.672→1.000，box021_035_p1 回退) → 残差为下游 3D ee_body 门 → 催生 z-only(组件⑤)"],
    ["来源", "workspace/core4d/log/218_E166_foot_smooth_cem_results.md (Summary 表)"],
]

# Sheet 9: H_探索未纳入E167A
sheet9 = [
    ["H. 探索但未纳入 E167A 的组件（诚实记录）", "结论", "证据/来源"],
    ["双手 AND 约束 (E164)", "过约束，raw 接触硬门 0/3，回退已过 case", "log/210_E164（负结果，未启用）"],
    ["后平滑 A_B2 (E166)", "上游 jerk↓57% 但下游任务仅 1/4，残差为 3D ee_body 门", "log/218,219（混合，不在 E167A）"],
    ["rubber-hand 代理 (E147/E148)", "手穿透 0.446→0.288 但物理接触 0.403→0.278，几何权衡", "log/187,188（E167A 用其几何，但非净胜组件）"],
    ["E155_decay (放手段衰减替代)", "clean8 接触最高但 release/穿透回升，不设默认", "log/197_E156（C2/C4 不成立）"],
]

SHEETS = [
    ("1_OmniRT_vs_E163_vs_E167A", sheet1),
    ("2_主消融表", sheet2),
    ("3_B_gateA穿透门", sheet3),
    ("4_C_姿态重排序", sheet4),
    ("5_D_释放衰减", sheet5),
    ("6_E_窄带RLsafe", sheet6),
    ("7_F_zonly下游", sheet7),
    ("8_G_zonly参考层E167eval", sheet_zref),
    ("9_H_平滑度E166", sheet8),
    ("10_I_探索未纳入", sheet9),
]

# ---------------------------------------------------------------- xlsx writer
def col_letter(n):  # 1->A
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

def cell_xml(col, row, val, header):
    ref = f"{col_letter(col)}{row}"
    if val is None or val == "":
        return ""
    style = ' s="1"' if header else ""
    if isinstance(val, bool):
        val = str(val)
    if isinstance(val, (int, float)):
        return f'<c r="{ref}"{style}><v>{val}</v></c>'
    txt = sx.escape(str(val))
    return f'<c r="{ref}"{style} t="inlineStr"><is><t xml:space="preserve">{txt}</t></is></c>'

def sheet_xml(rows):
    out = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
           '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>']
    for ri, row in enumerate(rows, start=1):
        header = (ri == 1)
        cells = "".join(cell_xml(ci, ri, v, header) for ci, v in enumerate(row, start=1))
        out.append(f'<row r="{ri}">{cells}</row>')
    out.append('</sheetData></worksheet>')
    return "".join(out)

STYLES = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
 '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
 '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
 '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
 '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
 '<borders count="1"><border/></borders>'
 '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
 '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
 '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
 '</styleSheet>')

def build(path):
    n = len(SHEETS)
    ctypes = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
              '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
              '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
              '<Default Extension="xml" ContentType="application/xml"/>',
              '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
              '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
    for i in range(1, n + 1):
        ctypes.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
    ctypes.append('</Types>')

    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>')

    sheets_xml = "".join(f'<sheet name="{sx.escape(nm)}" sheetId="{i}" r:id="rId{i}"/>'
                         for i, (nm, _) in enumerate(SHEETS, start=1))
    workbook = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{sheets_xml}</sheets></workbook>')

    wb_rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
               '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    for i in range(1, n + 1):
        wb_rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>')
    wb_rels.append(f'<Relationship Id="rId{n+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>')
    wb_rels.append('</Relationships>')

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "".join(ctypes))
        z.writestr("_rels/.rels", root_rels)
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", "".join(wb_rels))
        z.writestr("xl/styles.xml", STYLES)
        for i, (_, rows) in enumerate(SHEETS, start=1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", sheet_xml(rows))

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "E167A_ablation.xlsx")
    build(out)
    print("wrote", out)
