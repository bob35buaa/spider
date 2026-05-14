#!/usr/bin/env python3
"""
CORE4D 动力学重定向进展报告 PPT 生成
参考 holosoma progress_report 风格
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor

# 配色方案（参考 holosoma PPT）
DARK_BLUE = RGBColor(0x1B, 0x3A, 0x5C)
BLUE_GRAY = RGBColor(0x53, 0x73, 0x93)
DARK_GRAY = RGBColor(0x22, 0x22, 0x22)
MID_GRAY = RGBColor(0x55, 0x55, 0x55)
LIGHT_GRAY = RGBColor(0xCC, 0xCC, 0xCC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xCF, 0x22, 0x2E)

prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(5.625)

def add_title_slide(title, subtitle, meta=""):
    """添加标题页"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # 空白布局
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE

    # 添加顶部分割线
    line = slide.shapes.add_shape(1, Inches(1.5), Inches(0.45), Inches(7), Inches(0))
    line.line.color.rgb = BLUE_GRAY
    line.line.width = Pt(2)
    line.fill.background()

    # 标题
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(1.2))
    title_frame = title_box.text_frame
    title_frame.word_wrap = True
    p = title_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(48)
    p.font.bold = True
    p.font.color.rgb = DARK_BLUE
    p.font.name = 'Arial'

    # 副标题
    subtitle_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.9), Inches(9), Inches(0.6))
    subtitle_frame = subtitle_box.text_frame
    p = subtitle_frame.paragraphs[0]
    p.text = subtitle
    p.font.size = Pt(24)
    p.font.color.rgb = MID_GRAY
    p.font.name = 'Arial'

    # 元数据
    if meta:
        meta_box = slide.shapes.add_textbox(Inches(0.5), Inches(4.8), Inches(9), Inches(0.5))
        meta_frame = meta_box.text_frame
        p = meta_frame.paragraphs[0]
        p.text = meta
        p.font.size = Pt(11)
        p.font.italic = True
        p.font.color.rgb = MID_GRAY
        p.font.name = 'Arial'

def add_content_slide(title, content_items):
    """添加内容页（bullet list 风格）"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE

    # 添加标题
    title_box = slide.shapes.add_textbox(Inches(1.2), Inches(0.1), Inches(8.5), Inches(0.45))
    title_frame = title_box.text_frame
    p = title_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = DARK_BLUE
    p.font.name = 'Arial'

    # 添加分割线
    line = slide.shapes.add_shape(1, Inches(1.2), Inches(0.58), Inches(7.6), Inches(0))
    line.line.color.rgb = BLUE_GRAY
    line.line.width = Pt(1.5)
    line.fill.background()

    # 添加内容
    content_box = slide.shapes.add_textbox(Inches(0.7), Inches(0.85), Inches(8.6), Inches(4.5))
    text_frame = content_box.text_frame
    text_frame.word_wrap = True

    for idx, item in enumerate(content_items):
        if idx > 0:
            p = text_frame.add_paragraph()
        else:
            p = text_frame.paragraphs[0]

        p.text = item
        p.font.size = Pt(14)
        p.font.color.rgb = DARK_GRAY
        p.font.name = 'Arial'
        p.space_before = Pt(6)
        p.space_after = Pt(6)
        p.level = 0

        # 添加 bullet
        p.bullet = True

def add_two_column_slide(title, left_items, right_items, left_title="", right_title=""):
    """添加两列对比页"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE

    # 标题
    title_box = slide.shapes.add_textbox(Inches(1.2), Inches(0.1), Inches(8.5), Inches(0.45))
    title_frame = title_box.text_frame
    p = title_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = DARK_BLUE
    p.font.name = 'Arial'

    # 分割线
    line = slide.shapes.add_shape(1, Inches(1.2), Inches(0.58), Inches(7.6), Inches(0))
    line.line.color.rgb = BLUE_GRAY
    line.line.width = Pt(1.5)
    line.fill.background()

    # 左列标题
    if left_title:
        left_title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.75), Inches(4.5), Inches(0.3))
        left_title_frame = left_title_box.text_frame
        p = left_title_frame.paragraphs[0]
        p.text = left_title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = DARK_BLUE
        p.font.name = 'Arial'

    # 左列内容
    left_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.2) if left_title else Inches(0.85), Inches(4.2), Inches(4))
    left_frame = left_box.text_frame
    left_frame.word_wrap = True
    for idx, item in enumerate(left_items):
        if idx > 0:
            p = left_frame.add_paragraph()
        else:
            p = left_frame.paragraphs[0]
        p.text = item
        p.font.size = Pt(12)
        p.font.color.rgb = DARK_GRAY
        p.font.name = 'Arial'
        p.level = 0
        p.bullet = True
        p.space_before = Pt(4)
        p.space_after = Pt(4)

    # 右列标题
    if right_title:
        right_title_box = slide.shapes.add_textbox(Inches(5.3), Inches(0.75), Inches(4.2), Inches(0.3))
        right_title_frame = right_title_box.text_frame
        p = right_title_frame.paragraphs[0]
        p.text = right_title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = RED
        p.font.name = 'Arial'

    # 右列内容
    right_box = slide.shapes.add_textbox(Inches(5.3), Inches(1.2) if right_title else Inches(0.85), Inches(4.2), Inches(4))
    right_frame = right_box.text_frame
    right_frame.word_wrap = True
    for idx, item in enumerate(right_items):
        if idx > 0:
            p = right_frame.add_paragraph()
        else:
            p = right_frame.paragraphs[0]
        p.text = item
        p.font.size = Pt(12)
        p.font.color.rgb = DARK_GRAY
        p.font.name = 'Arial'
        p.level = 0
        p.bullet = True
        p.space_before = Pt(4)
        p.space_after = Pt(4)

# ============ 生成幻灯片 ============

# Slide 1: 标题页
add_title_slide(
    "CORE4D 动力学重定向",
    "诊断驱动的参数自动化方向",
    "66 Experiments (E001-E066) | SPIDER Framework | 2026-05-14"
)

# Slide 2: 项目概述
add_content_slide("项目概述", [
    "问题：G1 机器人臂展不足 (63cm < 箱子长 61cm)，运动学重定向完全失败",
    "方案：SPIDER 采样 MPC 物理重定向，替代纯运动学方案",
    "周期：2 周 × 66 个实验，从基础验证到诊断驱动",
    "关键转折：发现数值指标完全欺骗，转向诊断 + 自动化参数",
    "关键产出：诊断工具体系、21 case 碰撞盒标准化、数值指标误导机制理解"
])

# Slide 3: 关键发现 #1
add_content_slide("关键发现 #1：数值指标欺骗", [
    "ObjPos=14cm (E041c box023)：数值显示好，实际机器人摔倒",
    "Contact=93% (HDMI box023)：数值显示接触，实际物体方向错 178度",
    "Stability=100% (E041c desk005)：数值显示稳定，实际丢下物体自己走",
    "根本原因：reward 字段设计局限，看不到整体物理行为",
    "解决：必须用视频逐帧对比 ref vs sim，才能判断真实结果"
])

# Slide 4: 关键发现 #2
add_content_slide("关键发现 #2：物理约束矛盾", [
    "CEM 同时优化 body tracking + contact 两个互斥目标",
    "三次 reward 调参三种不同失败模式：",
    "  • E062 Deep Fall (pelvis=0.058m)：stability_penalty 关闭",
    "  • E063 Superman Lunge (姿态平躺)：stability 约束只约束高度不约束姿态",
    "  • E064 Prone Kneel (单膝跪)：threshold 过严导致新局部最优",
    "结论：3-strike rule 触发，同框架调参已达极限，缺新约束维度"
])

# Slide 5: 关键发现 #3
add_two_column_slide(
    "关键发现 #3：3-Box 手部结构回归",
    [
        "pelvis_min: 0.672m",
        "stable%: 100%",
        "结论：✅ 通过"
    ],
    [
        "pelvis_min: 0.253m",
        "stable%: 61.7%",
        "结论：❌ 回归 -0.42m"
    ],
    "Sphere (回退)",
    "3-Box (回滚)"
)

# Slide 6: 关键发现 #4
add_content_slide("关键发现 #4：参数过拟合", [
    "E041c box025 参数 (palm_normal=[0,∓1,0], eef_offset=[0.05,0,0]) 仅对单 case 有效",
    "box023 同参数：pelvis_min 摔倒，contact 下降",
    "根本原因：hand-tuned 常数依赖特定物体 + 特定 geometry",
    "转向 X1+X2 自动参数化：",
    "  • X1 = auto palm_normal (从 ref motion 导出手掌朝向)",
    "  • X2 = auto eef_offset (从手部几何自动计算)",
    "目标：实现跨 case 泛化，而非单 case 调通"
])

# Slide 7: 诊断工具与方法
add_content_slide("诊断工具与方法", [
    "Hand-Snap IK (E055-E057)：把手部投影到物体表面，验证接触几何",
    "6-Face 握姿分类 (E056)：自动识别握姿类型，排除无效案例 (box021 移出)",
    "Palm Normal 自动化 (E062)：从 ref 运动计算手掌朝向，替代 hardcoded",
    "全帧对比诊断 (E062 Diagnosis)：ref vs sim 物体轨迹全帧对齐，发现真根因",
    "Scene XML Dual Safeguard：force-add 入 git + 实验快照，保证再现性"
])

# Slide 8: 阶段成果
add_content_slide("阶段成果（E001-E066）", [
    "✅ 管线搭建 (E001-E005)：holosoma ↔ SPIDER 格式无损转换、21 case XML",
    "✅ 基础验证 (E002-E024)：协作伙伴方案、几何分析、接触上限发现",
    "✅ 方法探索 (E025-E053)：奖励函数、dual-robot、collision margin sweep",
    "✅ 诊断工具 (E054-E061)：hand-snap、多 case 分级、sphere baseline PASS",
    "⚠️ 泛化突破 (E062-E066)：X1 auto palm_normal、body partition 诊断（进行中）"
])

# Slide 9: 下一步行动
add_content_slide("下一步行动（优先级）", [
    "P1 (本周)：E067 port HDMI body partition (lower 12→6 bodies)；E065 task_obj_rew 形式消融",
    "成功标准：≥2 case 达到 pelvis_min ≥ 0.30m + stable ≥ 80%",
    "P2 (本周末)：E067 结果评估；修复 3-box eef_offset 或回滚 sphere",
    "P3 (下周)：X1+X2 完整泛化测试（4 cases）；考虑分层优化或后置约束",
    "预期产出：确认诊断驱动方向是否可行；打开多物体泛化的可能"
])

# Slide 10: 总结
add_content_slide("总结与价值", [
    "核心洞察：前 50 个实验的'成功'都是数值幻觉，系统性诊断揭示深层矛盾",
    "已验证：诊断框架可复用、数据质量改善、工具链 ready、根本认知深化",
    "预期贡献：自动参数化突破 + 多物体泛化铺路 + 论文素材 + 方法论总结",
    "投入：再 1 周集中投入确认方向，2 周可收口或转向新思路",
    "关键决策：P1 如果通过 → 继续 X1+X2 泛化；失败 → 需要方法学转向"
])

# 保存
output_path = "/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider/workspace/core4d/report/E001-E067/CORE4D_Progress_Report.pptx"
prs.save(output_path)
print(f"✅ PPT 生成成功: {output_path}")
