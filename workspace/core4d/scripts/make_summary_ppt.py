#!/usr/bin/env python3
"""Generate a single-slide summary PPT for CORE4D dual-robot retargeting experiments."""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---------- Constants ----------
W, H = Inches(13.333), Inches(7.5)  # 16:9
BG = RGBColor(0x1A, 0x1A, 0x2E)  # dark navy
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT = RGBColor(0x00, 0xD2, 0xFF)  # cyan
GREEN = RGBColor(0x00, 0xE6, 0x96)
ORANGE = RGBColor(0xFF, 0xA5, 0x00)
RED = RGBColor(0xFF, 0x55, 0x55)
GRAY = RGBColor(0xAA, 0xAA, 0xBB)
LIGHT_BG = RGBColor(0x24, 0x24, 0x3E)

OUT = Path("/home/ubuntu/Workspace/spider/workspace/core4d/SUMMARY.pptx")


def add_text_box(slide, left, top, width, height, text, font_size=12, bold=False, color=WHITE, alignment=PP_ALIGN.LEFT, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_rich_text_box(slide, left, top, width, height, lines, font_name="Calibri"):
    """lines: list of (text, font_size, bold, color, alignment)"""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, (text, font_size, bold, color, alignment) in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = text
        p.font.size = Pt(font_size)
        p.font.bold = bold
        p.font.color.rgb = color
        p.font.name = font_name
        p.alignment = alignment
        p.space_after = Pt(2)
    return txBox


def add_rounded_rect(slide, left, top, width, height, fill_color):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def make_table(slide, left, top, width, rows_data, col_widths, header_color=ACCENT):
    """rows_data: list of lists of strings. First row = header."""
    n_rows = len(rows_data)
    n_cols = len(rows_data[0])
    table_shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, Inches(0.28 * n_rows))
    table = table_shape.table

    for ci, cw in enumerate(col_widths):
        table.columns[ci].width = cw

    for ri, row in enumerate(rows_data):
        for ci, cell_text in enumerate(row):
            cell = table.cell(ri, ci)
            cell.text = cell_text
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(8)
            p.font.name = "Calibri"
            p.alignment = PP_ALIGN.CENTER

            if ri == 0:
                p.font.bold = True
                p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0x2A, 0x2A, 0x4E)
            else:
                p.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0x1E, 0x1E, 0x38)

            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    return table_shape


def build():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H

    # Blank layout
    layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(layout)

    # Background
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG

    # ============================== TITLE BAR ==============================
    add_rounded_rect(slide, Inches(0.3), Inches(0.2), Inches(12.7), Inches(0.7), LIGHT_BG)

    add_text_box(slide, Inches(0.5), Inches(0.25), Inches(9), Inches(0.4),
                 "CORE4D → Dual G1 Physics-Based Retargeting", 22, True, ACCENT)
    add_text_box(slide, Inches(0.5), Inches(0.58), Inches(9), Inches(0.3),
                 "SPIDER SBMPC + Connect + Task-Space Rewards  |  E001–E018  |  2026-04-30 → 05-06", 10, False, GRAY)

    # ============================== LEFT COLUMN: Phase Timeline ==============================
    col1_x = Inches(0.3)
    y_cur = Inches(1.1)

    add_text_box(slide, col1_x, y_cur, Inches(4), Inches(0.3),
                 "实验路线 (4 Phases → 18 Experiments)", 11, True, WHITE)
    y_cur += Inches(0.35)

    phases = [
        ("P0 数据管线", "E001", "holosoma → SPIDER NPZ + scene XML", GREEN, "✓"),
        ("P1 单人物理搬运", "E002–E009", "全败 (obj_z ≤ 0.31m)  根因: 臂展 < box 长", RED, "✗"),
        ("P2 + Mocap Partner", "E010–E012", "E012 导出 hybrid 轨迹 (pelvis_err=0.083m)", ORANGE, "△"),
        ("P3 架构修复", "E013–E014", "Intra-mocap 修复但 CEM 高方差", ORANGE, "△"),
        ("P4 双机器人", "E015–E018", "E017 connect 突破 → E018 task-space 最优", GREEN, "★"),
    ]

    for phase_name, exps, desc, color, icon in phases:
        add_rich_text_box(slide, col1_x + Inches(0.05), y_cur, Inches(4.2), Inches(0.50), [
            (f"{icon}  {phase_name}  ({exps})", 9, True, color, PP_ALIGN.LEFT),
            (f"    {desc}", 8, False, GRAY, PP_ALIGN.LEFT),
        ])
        y_cur += Inches(0.45)

    # ============================== CENTER: Key Results Table ==============================
    tbl_x = Inches(0.3)
    y_cur += Inches(0.05)
    add_text_box(slide, tbl_x, y_cur, Inches(5), Inches(0.25),
                 "关键指标演进 (E016 → E017 → E018)", 11, True, WHITE)
    y_cur += Inches(0.30)

    table_data = [
        ["Run", "方法", "obj_err ↓", "obj_z_max", "obj>0.40", "R1 / R2\nstable"],
        ["E016-a", "Gibbs CEM", "—", "0.488", "瞬间", "100% / 100%"],
        ["E017-d", "+ Soft 2-connect", "0.599m", "0.597", "98%", "86% / 89%"],
        ["E018-d2 ✅", "+ Task-space\n+ Interaction", "0.308m", "0.598", "83%", "95% / 100%"],
        ["—对照—", "E018 no-connect", "0.624m", "0.475", "11%", "77% / 56%"],
    ]
    col_widths = [Emu(680000), Emu(1000000), Emu(680000), Emu(620000), Emu(530000), Emu(680000)]
    tbl = make_table(slide, tbl_x, y_cur, Inches(4.5), table_data, col_widths)
    # highlight best row
    for ci in range(6):
        cell = tbl.table.cell(3, ci)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(0x00, 0x3D, 0x2E)

    y_cur += Inches(1.55)

    # ============================== RIGHT COLUMN: Insights ==============================
    col2_x = Inches(4.9)
    y2 = Inches(1.1)

    add_text_box(slide, col2_x, y2, Inches(4), Inches(0.3),
                 "核心洞察", 11, True, WHITE)
    y2 += Inches(0.32)

    insights = [
        ("Connect 约束 = SBMPC 的物理触手", "CEM 短 horizon (1.6s) 无法采样有效接触力 → 把手焊到箱面, CEM 只优化身体, 箱子自动跟随"),
        ("Task-space 奖励: tracking ↓49%", "DynaRetarget 启发 (obj=40, torso=30, hand=5, foot=10), 无需重写 optimizer"),
        ("Interaction reward 维持协作", "Harmanoid Eq.15: 双 pelvis/torso/wrist 相对距离约束, 防止穿透"),
        ("Stability ↔ Tracking tradeoff", "obj-only 拉爆 pelvis; base_pos=15 是平衡点"),
        ("必须 qpos + 视频双验证", "E006/E008 教训: reward 字段不可信"),
    ]

    for title, desc in insights:
        add_rounded_rect(slide, col2_x, y2, Inches(4.1), Inches(0.52), LIGHT_BG)
        add_rich_text_box(slide, col2_x + Inches(0.1), y2 + Inches(0.02), Inches(3.9), Inches(0.50), [
            (f"▸ {title}", 8.5, True, ACCENT, PP_ALIGN.LEFT),
            (f"  {desc}", 7.5, False, GRAY, PP_ALIGN.LEFT),
        ])
        y2 += Inches(0.58)

    # ============================== FAR RIGHT: Architecture + Next ==============================
    col3_x = Inches(9.2)
    y3 = Inches(1.1)

    add_text_box(slide, col3_x, y3, Inches(4), Inches(0.3),
                 "架构 (E018 最佳配置)", 11, True, WHITE)
    y3 += Inches(0.32)

    arch_lines = [
        ("双 G1 机器人 (nq=79, nu=58)", 8.5, False, WHITE, PP_ALIGN.LEFT),
        ("  + Soft 2-Connect 焊接约束", 8.5, False, GREEN, PP_ALIGN.LEFT),
        ("      solref=-200 -30, base_pos=15", 7.5, False, GRAY, PP_ALIGN.LEFT),
        ("  + Task-Space Body Rewards", 8.5, False, GREEN, PP_ALIGN.LEFT),
        ("      obj_pos=40, torso=30, hand=5, foot=10", 7.5, False, GRAY, PP_ALIGN.LEFT),
        ("  + Interaction Reward (Harmanoid)", 8.5, False, GREEN, PP_ALIGN.LEFT),
        ("      σ=5, pairs: pelvis/torso/wrist", 7.5, False, GRAY, PP_ALIGN.LEFT),
        ("  + Gibbs Sampling CEM Optimizer", 8.5, False, GREEN, PP_ALIGN.LEFT),
        ("      交替优化 Robot1 → Robot2", 7.5, False, GRAY, PP_ALIGN.LEFT),
    ]
    add_rounded_rect(slide, col3_x, y3, Inches(3.9), Inches(2.2), LIGHT_BG)
    add_rich_text_box(slide, col3_x + Inches(0.1), y3 + Inches(0.05), Inches(3.7), Inches(2.1), arch_lines)
    y3 += Inches(2.35)

    # Outputs
    add_text_box(slide, col3_x, y3, Inches(4), Inches(0.3),
                 "产出文件", 11, True, WHITE)
    y3 += Inches(0.30)
    outputs = [
        "✅ E018d2_box025_dual_taskspace_basepos15.npz",
        "✅ E018d2_box025_dual_taskspace_basepos15.mp4",
        "✅ E018d2_mpc{0,3,5,8,10}.png (关键帧)",
        "✅ hybrid baseline: box025_p1_holosoma_w_partner.npz",
    ]
    add_rounded_rect(slide, col3_x, y3, Inches(3.9), Inches(1.0), LIGHT_BG)
    add_rich_text_box(slide, col3_x + Inches(0.1), y3 + Inches(0.05), Inches(3.7), Inches(0.9),
                      [(o, 7.5, False, GREEN, PP_ALIGN.LEFT) for o in outputs])
    y3 += Inches(1.1)

    # Next Steps
    add_text_box(slide, col3_x, y3, Inches(4), Inches(0.3),
                 "下一步", 11, True, WHITE)
    y3 += Inches(0.30)
    nexts = [
        ("E019", "长 horizon SBMPC (2s → 4s)"),
        ("E020", "完整 SBTO (DynaRetarget pipeline)"),
        ("RL", "E018-d2 → Holosoma RL finetune"),
    ]
    add_rounded_rect(slide, col3_x, y3, Inches(3.9), Inches(0.85), LIGHT_BG)
    next_lines = [(f"→ {n}: {d}", 8, False, ORANGE, PP_ALIGN.LEFT) for n, d in nexts]
    add_rich_text_box(slide, col3_x + Inches(0.1), y3 + Inches(0.05), Inches(3.7), Inches(0.8), next_lines)

    # ============================== Bottom: P1 root cause ==============================
    add_text_box(slide, Inches(0.5), Inches(6.9), Inches(8), Inches(0.3),
                 "Phase 1 根因: G1 臂展 0.5m < box025 长 0.61m，单人对夹几何不可解 — 需要协作建模 → Phase 4 双机器人方案",
                 8, False, RGBColor(0x88, 0x88, 0x99))

    # Try adding keyframe images
    img_dir = Path("/home/ubuntu/Workspace/spider/workspace/core4d/results")
    imgs = ["E018d2_mpc0.png", "E018d2_mpc3.png", "E018d2_mpc5.png", "E018d2_mpc8.png", "E018d2_mpc10.png"]
    img_y = y_cur + Inches(0.05)
    img_w = Inches(0.85)
    img_x_start = Inches(0.3)

    # Add keyframes label
    add_text_box(slide, Inches(0.3), y_cur - Inches(0.05), Inches(4.5), Inches(0.25),
                 "E018-d2 关键帧 (MPC step 0/3/5/8/10)", 9, True, ACCENT)

    for i, img_name in enumerate(imgs):
        img_path = img_dir / img_name
        if img_path.exists():
            slide.shapes.add_picture(str(img_path), img_x_start + img_w * i + Inches(0.05) * i,
                                     img_y, img_w, img_w)

    prs.save(str(OUT))
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    build()
