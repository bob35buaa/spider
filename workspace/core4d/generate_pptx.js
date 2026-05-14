const PptxGenJS = require('pptxgenjs');

const prs = new PptxGenJS();
prs.defineLayout({ name: 'MASTER', width: 13.333, height: 7.5 });
prs.defineLayout({ name: 'BLANK', width: 13.333, height: 7.5 });
prs.layout = 'MASTER';

// 配色方案：深蓝 + 青色 + 白色
const colors = {
  navy: '1E2761',
  teal: '028090',
  mint: '00A896',
  dark: '212121',
  light: 'F5F5F5',
  white: 'FFFFFF',
  accent: 'FF6B6B'
};

// 辅助函数：添加标题页
function addTitleSlide(title, subtitle) {
  const slide = prs.addSlide();
  slide.background = { color: colors.navy };

  // 标题
  slide.addText(title, {
    x: 0.5, y: 2.5, w: 12.333, h: 1.5,
    fontSize: 48, bold: true, color: colors.white,
    align: 'center', fontFace: 'Arial'
  });

  // 副标题
  slide.addText(subtitle, {
    x: 0.5, y: 4.2, w: 12.333, h: 1,
    fontSize: 24, color: colors.mint,
    align: 'center', fontFace: 'Arial'
  });

  // 装饰线
  slide.addShape(prs.ShapeType.rect, {
    x: 5, y: 2.2, w: 3.333, h: 0.08,
    fill: { color: colors.teal }, line: { type: 'none' }
  });
}

// 辅助函数：内容页标题
function addContentSlide(title) {
  const slide = prs.addSlide();
  slide.background = { color: colors.white };

  // 背景色块
  slide.addShape(prs.ShapeType.rect, {
    x: 0, y: 0, w: 13.333, h: 1.2,
    fill: { color: colors.navy }, line: { type: 'none' }
  });

  // 标题
  slide.addText(title, {
    x: 0.5, y: 0.2, w: 12.333, h: 0.8,
    fontSize: 36, bold: true, color: colors.white,
    align: 'left', fontFace: 'Arial'
  });

  return slide;
}

// Slide 1: 标题
addTitleSlide('CORE4D 动力学重定向', '阶段性进展汇报 | 2026-05-14');

// Slide 2: 项目概述
(() => {
  const slide = addContentSlide('项目概述');

  const items = [
    { title: '问题', desc: 'G1 机器人臂展不足 (63cm < 箱子长 61cm)，运动学方案完全失败' },
    { title: '方案', desc: 'SPIDER 采样 MPC 物理重定向，替代纯运动学方案' },
    { title: '周期', desc: '2 周 × 66 个实验，从基础验证到诊断驱动' },
    { title: '关键转折', desc: '发现数值指标完全欺骗，转向诊断+ 自动化参数' }
  ];

  items.forEach((item, idx) => {
    const y = 1.8 + idx * 1.3;

    // 左侧圆点
    slide.addShape(prs.ShapeType.ellipse, {
      x: 0.6, y: y + 0.15, w: 0.15, h: 0.15,
      fill: { color: colors.teal }, line: { type: 'none' }
    });

    // 标题
    slide.addText(item.title, {
      x: 1.1, y: y, w: 2, h: 0.4,
      fontSize: 14, bold: true, color: colors.navy,
      fontFace: 'Arial'
    });

    // 描述
    slide.addText(item.desc, {
      x: 1.1, y: y + 0.4, w: 11, h: 0.7,
      fontSize: 12, color: colors.dark,
      fontFace: 'Arial'
    });
  });
})();

// Slide 3: 关键发现 #1
(() => {
  const slide = addContentSlide('关键发现 #1：数值指标欺骗');

  const cases = [
    { metric: 'Object Position Error', number: '14cm', color: colors.teal, real: '机器人摔倒 ❌' },
    { metric: 'Contact Rate', number: '93%', color: colors.mint, real: '物体方向错 178° ❌' },
    { metric: 'Stability', number: '100%', color: colors.accent, real: '丢下物体自己走 ❌' }
  ];

  cases.forEach((c, idx) => {
    const x = 1 + idx * 4;

    // 背景卡片
    slide.addShape(prs.ShapeType.rect, {
      x, y: 1.8, w: 3.5, h: 3.8,
      fill: { color: colors.light }, line: { color: c.color, width: 3 }
    });

    // 数值
    slide.addText(c.number, {
      x, y: 2.2, w: 3.5, h: 0.8,
      fontSize: 40, bold: true, color: c.color,
      align: 'center', fontFace: 'Arial'
    });

    // 指标名
    slide.addText(c.metric, {
      x: x + 0.2, y: 3.1, w: 3.1, h: 0.6,
      fontSize: 11, bold: true, color: colors.navy,
      align: 'center', fontFace: 'Arial'
    });

    // 分割线
    slide.addShape(prs.ShapeType.rect, {
      x: x + 0.3, y: 3.8, w: 2.9, h: 0.04,
      fill: { color: c.color }, line: { type: 'none' }
    });

    // 真实情况
    slide.addText(c.real, {
      x: x + 0.2, y: 4, w: 3.1, h: 1.4,
      fontSize: 10, color: colors.dark,
      align: 'center', fontFace: 'Arial'
    });
  });
})();

// Slide 4: 关键发现 #2
(() => {
  const slide = addContentSlide('关键发现 #2：物理约束矛盾');

  slide.addText('单机器人 × 多维度同时优化 = CEM 无解', {
    x: 0.8, y: 1.5, w: 11.5, h: 0.5,
    fontSize: 18, bold: true, color: colors.navy,
    fontFace: 'Arial'
  });

  // 三次失败模式
  const modes = [
    { name: '失败模式 1', desc: 'Deep Fall', icon: '📉', color: colors.teal },
    { name: '失败模式 2', desc: 'Superman Lunge', icon: '🏃', color: colors.mint },
    { name: '失败模式 3', desc: 'Prone Kneel', icon: '🧎', color: colors.accent }
  ];

  modes.forEach((mode, idx) => {
    const x = 0.8 + idx * 4;
    const y = 2.5;

    slide.addText(mode.icon, {
      x, y, w: 1, h: 1,
      fontSize: 48, align: 'center'
    });

    slide.addText(mode.name, {
      x: x - 0.2, y: y + 1.1, w: 1.4, h: 0.4,
      fontSize: 12, bold: true, color: mode.color,
      align: 'center', fontFace: 'Arial'
    });

    slide.addText(mode.desc, {
      x: x - 0.2, y: y + 1.5, w: 1.4, h: 1,
      fontSize: 10, color: colors.dark,
      align: 'center', fontFace: 'Arial'
    });
  });

  // 底部结论
  slide.addShape(prs.ShapeType.rect, {
    x: 0.8, y: 5.8, w: 11.5, h: 1.2,
    fill: { color: colors.light }, line: { color: colors.navy, width: 2 }
  });

  slide.addText('3-strike rule 触发：同框架内调参已达极限，缺新约束维度', {
    x: 1.2, y: 5.95, w: 10.7, h: 1,
    fontSize: 13, bold: true, color: colors.navy,
    align: 'center', fontFace: 'Arial'
  });
})();

// Slide 5: 关键发现 #3
(() => {
  const slide = addContentSlide('关键发现 #3：3-Box 手部结构回归');

  // 对比
  slide.addShape(prs.ShapeType.rect, {
    x: 0.8, y: 1.5, w: 5.5, h: 4.5,
    fill: { color: colors.light }, line: { color: colors.teal, width: 2 }
  });

  slide.addText('Sphere (回退)', {
    x: 1, y: 1.7, w: 5.1, h: 0.35,
    fontSize: 16, bold: true, color: colors.teal,
    fontFace: 'Arial'
  });

  const sphereStats = [
    { label: 'pelvis_min', value: '0.672m', good: true },
    { label: 'stable%', value: '100%', good: true },
    { label: '结论', value: '✅ 通过', good: true }
  ];

  sphereStats.forEach((stat, idx) => {
    const y = 2.3 + idx * 1;
    slide.addText(stat.label + ':', {
      x: 1.2, y, w: 2, h: 0.35,
      fontSize: 12, bold: true, color: colors.navy,
      fontFace: 'Arial'
    });
    slide.addText(stat.value, {
      x: 3.2, y, w: 2, h: 0.35,
      fontSize: 12, color: stat.good ? colors.mint : colors.accent,
      fontFace: 'Arial'
    });
  });

  // 右侧
  slide.addShape(prs.ShapeType.rect, {
    x: 6.8, y: 1.5, w: 5.5, h: 4.5,
    fill: { color: colors.light }, line: { color: colors.accent, width: 2 }
  });

  slide.addText('3-Box (回滚)', {
    x: 7, y: 1.7, w: 5.1, h: 0.35,
    fontSize: 16, bold: true, color: colors.accent,
    fontFace: 'Arial'
  });

  const boxStats = [
    { label: 'pelvis_min', value: '0.253m', good: false },
    { label: 'stable%', value: '61.7%', good: false },
    { label: '结论', value: '❌ 回归 -0.42m', good: false }
  ];

  boxStats.forEach((stat, idx) => {
    const y = 2.3 + idx * 1;
    slide.addText(stat.label + ':', {
      x: 7.2, y, w: 2, h: 0.35,
      fontSize: 12, bold: true, color: colors.navy,
      fontFace: 'Arial'
    });
    slide.addText(stat.value, {
      x: 9.2, y, w: 2, h: 0.35,
      fontSize: 12, color: stat.good ? colors.mint : colors.accent,
      fontFace: 'Arial'
    });
  });
})();

// Slide 6: 关键发现 #4
(() => {
  const slide = addContentSlide('关键发现 #4：参数过拟合');

  slide.addText('hand-tuned 常数仅对单 case 有效，泛化完全失败', {
    x: 0.8, y: 1.5, w: 11.5, h: 0.5,
    fontSize: 16, bold: true, color: colors.navy,
    fontFace: 'Arial'
  });

  // 参数对比表
  const params = [
    { name: 'palm_normal', box025: '[0, -1, 0] ✅', box023: '[+1, 0, 0] ❌' },
    { name: 'eef_offset', box025: '[0.05, 0, 0] ✅', box023: '错配 12.5cm ❌' },
    { name: '结果', box025: '0.688m ✅', box023: '摔倒 ❌' }
  ];

  const startY = 2.3;
  params.forEach((param, idx) => {
    const y = startY + idx * 1.1;

    slide.addText(param.name, {
      x: 1, y, w: 2.5, h: 0.35,
      fontSize: 12, bold: true, color: colors.navy,
      fontFace: 'Arial'
    });

    slide.addText(param.box025, {
      x: 3.8, y, w: 4, h: 0.35,
      fontSize: 11, color: colors.mint,
      fontFace: 'Arial'
    });

    slide.addText(param.box023, {
      x: 8, y, w: 4, h: 0.35,
      fontSize: 11, color: colors.accent,
      fontFace: 'Arial'
    });
  });

  // 转向方案
  slide.addShape(prs.ShapeType.rect, {
    x: 0.8, y: 5.2, w: 11.5, h: 1.5,
    fill: { color: colors.navy }, line: { type: 'none' }
  });

  slide.addText('转向 X1+X2 自动参数化：X1 = auto palm_normal (从 ref 导出)；X2 = auto eef_offset (从几何计算)', {
    x: 1.2, y: 5.4, w: 10.7, h: 1.1,
    fontSize: 13, bold: true, color: colors.white,
    align: 'center', fontFace: 'Arial'
  });
})();

// Slide 7: 诊断工具与方法
(() => {
  const slide = addContentSlide('诊断工具：从混乱到清晰');

  const tools = [
    { name: 'Hand-Snap IK', desc: 'E055-E057：把手部投影到物体表面，验证接触几何', icon: '🤖' },
    { name: '6-Face 分类', desc: 'E056：自动识别握姿类型，排除无效案例', icon: '✋' },
    { name: 'Palm Normal 自动化', desc: 'E062：从 ref 运动计算手掌朝向，替代 hardcoded', icon: '📐' },
    { name: '全帧对比诊断', desc: '用户洞察：ref vs sim 物体轨迹全帧对齐，发现真根因', icon: '📊' }
  ];

  tools.forEach((tool, idx) => {
    const y = 1.6 + idx * 1.3;

    slide.addText(tool.icon, {
      x: 0.8, y: y - 0.05, w: 0.6, h: 0.4,
      fontSize: 28, align: 'center'
    });

    slide.addText(tool.name, {
      x: 1.6, y, w: 2.8, h: 0.35,
      fontSize: 13, bold: true, color: colors.navy,
      fontFace: 'Arial'
    });

    slide.addText(tool.desc, {
      x: 1.6, y: y + 0.35, w: 10.5, h: 0.6,
      fontSize: 11, color: colors.dark,
      fontFace: 'Arial'
    });
  });
})();

// Slide 8: 阶段成果
(() => {
  const slide = addContentSlide('阶段成果');

  const stages = [
    { phase: '管线搭建', exp: 'E001-E005', status: '✅ 完成', color: colors.mint },
    { phase: '基础验证', exp: 'E002-E024', status: '✅ 完成', color: colors.mint },
    { phase: '方法探索', exp: 'E025-E053', status: '✅ 完成', color: colors.mint },
    { phase: '诊断工具', exp: 'E054-E061', status: '✅ 完成', color: colors.mint },
    { phase: '泛化突破', exp: 'E062-E066', status: '⚠️ 进行中', color: colors.accent }
  ];

  const startX = 1;
  const blockWidth = 2;
  const blockHeight = 0.8;

  stages.forEach((stage, idx) => {
    const x = startX + idx * 2.2;
    const y = 2;

    slide.addShape(prs.ShapeType.rect, {
      x, y, w: blockWidth, h: blockHeight,
      fill: { color: stage.color }, line: { type: 'none' }
    });

    slide.addText(stage.phase, {
      x: x + 0.1, y: y + 0.08, w: blockWidth - 0.2, h: 0.3,
      fontSize: 11, bold: true, color: colors.white,
      align: 'center', fontFace: 'Arial'
    });

    slide.addText(stage.exp, {
      x: x + 0.1, y: y + 0.38, w: blockWidth - 0.2, h: 0.35,
      fontSize: 9, color: colors.white,
      align: 'center', fontFace: 'Arial'
    });
  });

  // 成果清单
  const achievements = [
    '✅ holosoma ↔ SPIDER 转换流水线',
    '✅ 21 case 碰撞盒标准化',
    '✅ Hand-snap + 多 case 诊断体系',
    '✅ 数值指标欺骗系统理解',
    '✅ Scene XML dual safeguard'
  ];

  slide.addText('关键产出：', {
    x: 0.8, y: 3.2, w: 2, h: 0.3,
    fontSize: 12, bold: true, color: colors.navy,
    fontFace: 'Arial'
  });

  achievements.forEach((ach, idx) => {
    slide.addText(ach, {
      x: 1.1, y: 3.6 + idx * 0.55, w: 11, h: 0.45,
      fontSize: 11, color: colors.dark,
      fontFace: 'Arial'
    });
  });
})();

// Slide 9: 下一步行动
(() => {
  const slide = addContentSlide('下一步行动（优先级）');

  const actions = [
    {
      pri: 'P1',
      title: 'E067: port HDMI body partition',
      desc: '改进 reward.mean() 稀释问题，预计本周',
      color: colors.accent
    },
    {
      pri: 'P2',
      title: 'E065: task_obj_rew 形式消融',
      desc: '验证 unbounded L2 vs exp 形式，下周',
      color: colors.teal
    },
    {
      pri: 'P3',
      title: 'X1+X2 泛化到 4 cases',
      desc: '自动 palm_normal + eef_offset，验证 week 2',
      color: colors.mint
    }
  ];

  actions.forEach((action, idx) => {
    const y = 1.8 + idx * 1.65;

    // 优先级标签
    slide.addShape(prs.ShapeType.ellipse, {
      x: 0.8, y: y - 0.05, w: 0.45, h: 0.45,
      fill: { color: action.color }, line: { type: 'none' }
    });

    slide.addText(action.pri, {
      x: 0.8, y: y - 0.05, w: 0.45, h: 0.45,
      fontSize: 12, bold: true, color: colors.white,
      align: 'center', valign: 'middle', fontFace: 'Arial'
    });

    // 内容
    slide.addText(action.title, {
      x: 1.4, y, w: 10.5, h: 0.4,
      fontSize: 13, bold: true, color: colors.navy,
      fontFace: 'Arial'
    });

    slide.addText(action.desc, {
      x: 1.4, y: y + 0.42, w: 10.5, h: 0.7,
      fontSize: 11, color: colors.dark,
      fontFace: 'Arial'
    });
  });

  // 成功标准
  slide.addShape(prs.ShapeType.rect, {
    x: 0.8, y: 6, w: 11.5, h: 0.8,
    fill: { color: colors.light }, line: { color: colors.navy, width: 1 }
  });

  slide.addText('成功标准：≥2 case 达到 pelvis_min ≥ 0.30m + stable ≥ 80%', {
    x: 1.2, y: 6.15, w: 10.7, h: 0.5,
    fontSize: 12, bold: true, color: colors.navy,
    align: 'center', fontFace: 'Arial'
  });
})();

// Slide 10: 项目价值与总结
(() => {
  const slide = addContentSlide('总结');

  // 左侧：已完成
  slide.addText('已验证', {
    x: 0.8, y: 1.5, w: 5.5, h: 0.35,
    fontSize: 14, bold: true, color: colors.teal,
    fontFace: 'Arial'
  });

  const completed = [
    '✅ 诊断框架可复用',
    '✅ 数据质量改善',
    '✅ 工具链 ready',
    '✅ 根本认知深化'
  ];

  completed.forEach((item, idx) => {
    slide.addText(item, {
      x: 1, y: 2 + idx * 0.6, w: 5, h: 0.45,
      fontSize: 11, color: colors.dark,
      fontFace: 'Arial'
    });
  });

  // 右侧：预期贡献
  slide.addText('预期贡献（P1 达成后）', {
    x: 7, y: 1.5, w: 5.5, h: 0.35,
    fontSize: 14, bold: true, color: colors.accent,
    fontFace: 'Arial'
  });

  const expected = [
    '🎯 自动参数化突破',
    '🎯 多物体泛化铺路',
    '🎯 论文素材整理',
    '🎯 方法论总结'
  ];

  expected.forEach((item, idx) => {
    slide.addText(item, {
      x: 7.2, y: 2 + idx * 0.6, w: 5, h: 0.45,
      fontSize: 11, color: colors.dark,
      fontFace: 'Arial'
    });
  });

  // 底部结论
  slide.addShape(prs.ShapeType.rect, {
    x: 0.8, y: 5, w: 11.5, h: 1.8,
    fill: { color: colors.navy }, line: { type: 'none' }
  });

  slide.addText('诊断驱动的转向：放弃单 case 调参，改为系统性诊断 + 自动化参数化', {
    x: 1.2, y: 5.2, w: 10.7, h: 0.6,
    fontSize: 14, bold: true, color: colors.mint,
    align: 'center', fontFace: 'Arial'
  });

  slide.addText('如果 P1 通过 → X1+X2 泛化有望；失败 → 需要方法学转向', {
    x: 1.2, y: 5.95, w: 10.7, h: 0.6,
    fontSize: 12, color: colors.white,
    align: 'center', fontFace: 'Arial'
  });
})();

prs.writeFile({ fileName: '/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider/workspace/core4d/PRESENTATION.pptx' });
console.log('PPT 生成成功！');
