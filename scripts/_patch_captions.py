"""
Patch Paper.md:
  F6/T7  — convert all Chinese figure/table captions to MDPI English format
  refs   — update in-text 图N → Figure N and 表N → Table N
  T4     — compress Table 1 last column
  T5     — split Table 4 into 4a / 4b
  T6     — merge Table 6 last 2 columns
"""
import re, shutil, pathlib

ROOT = pathlib.Path(__file__).parent.parent
src = ROOT / 'Paper.md'
shutil.copy(src, ROOT / 'Paper_backup_captions.md')

text = src.read_text(encoding='utf-8')

# ──────────────────────────────────────────────────────────
# 1. Figure captions  (F6)
# ──────────────────────────────────────────────────────────
fig_caps = [
    ('**图 1：模型导出与设备端验证闭环**',
     '**Figure 1.** Model Export and On-Device Validation Closed Loop'),
    ('**图 2：资源受限 Android 多任务视觉推理系统总体架构（结构示意）**',
     '**Figure 2.** Overall System Architecture for Multi-Task Vision Inference on Resource-Constrained Android Devices'),
    ('**图 3：SafeHat 从训练到资源受限 Android 部署的闭环流程（流程示意）**',
     '**Figure 3.** End-to-End Closed-Loop Pipeline from SafeHat Training to Android Deployment'),
    ('**图 4：资源受限 Android 设备部署故障诊断流程（方法示意）**',
     '**Figure 4.** Fault Diagnosis Workflow for Deployment on Resource-Constrained Android Devices'),
    ('**图 5：预处理修复前后输入形态与检测结果对比（几何示意）**',
     '**Figure 5.** Comparison of Input Patterns and Detection Results Before and After Preprocessing Fix'),
    ('**图 6：单次与重复 Sigmoid 对背景分数的影响（数值示意）**',
     '**Figure 6.** Effect of Single vs. Repeated Sigmoid Application on Background Scores'),
    ('**图 7：传统网格解码与 E2E 已解码坐标的语义对比（坐标示意）**',
     '**Figure 7.** Semantic Comparison of Legacy Grid Decoding and E2E Pre-Decoded Coordinates'),
    ('**图 8：主案例训练基础与五任务 Android 真机运行证据**',
     '**Figure 8.** Training Baseline and Five-Task On-Device Deployment Evidence'),
]
for old, new in fig_caps:
    assert old in text, f'NOT FOUND: {old!r}'
    text = text.replace(old, new)

# ──────────────────────────────────────────────────────────
# 2. Table captions  (T7)  — Tables 4/6 handled in T5/T6 below
# ──────────────────────────────────────────────────────────
tbl_caps = [
    ('**表 1：五类任务的统一调度与差异化解析**',
     '**Table 1.** Unified Scheduling and Task-Specific Parsing for Five Task Types'),
    ('**表 2：部署各阶段工具、输入与验证目标**',
     '**Table 2.** Tools, Inputs, and Validation Goals for Each Deployment Stage'),
    ('**表 3：SafeHat 数据集类别与应用语义**',
     '**Table 3.** SafeHat Dataset Classes and Application Semantics'),
    ('**表 5：预处理修复前后背景分数统计**',
     '**Table 5.** Background Score Statistics Before and After Preprocessing Fix'),
    ('**表 7：验证环境与主要依赖**',
     '**Table 7.** Validation Environment and Main Dependencies'),
    ('**表 8：SafeHat 部署关键验证信息**',
     '**Table 8.** Key Validation Items for SafeHat Deployment'),
    ('**表 9：部署故障汇总——现象、根因与修复结果**',
     '**Table 9.** Fault Summary: Symptoms, Root Causes, and Fix Outcomes'),
    ('**表 10：Android 设备各任务推理延迟（纯推理调用，不含绘图，CPU 后端）**',
     '**Table 10.** On-Device Inference Latency by Task (Pure Inference, No Rendering, CPU Backend)'),
]
for old, new in tbl_caps:
    assert old in text, f'NOT FOUND: {old!r}'
    text = text.replace(old, new)

# ──────────────────────────────────────────────────────────
# 3. Image alt-text  (F6)
# ──────────────────────────────────────────────────────────
alt_replacements = [
    ('![图 8(a) SafeHat 微调过程中的损失与检测指标曲线]',
     '![Figure 8(a) SafeHat fine-tuning loss and detection metric curves]'),
    ('![图 8(b) SafeHat 验证批次预测样例]',
     '![Figure 8(b) SafeHat validation batch prediction examples]'),
    ('![图 8(c) 检测、分割、姿态、分类与 OBB 五任务真机视频抽帧组合证据图]',
     '![Figure 8(c) Five-task on-device frame montage: Detection, Segmentation, Pose, Classification, and OBB]'),
]
for old, new in alt_replacements:
    assert old in text, f'NOT FOUND: {old!r}'
    text = text.replace(old, new)

# ──────────────────────────────────────────────────────────
# 4. T4 — Compress Table 1 last column  (rows start with |, safe here)
# ──────────────────────────────────────────────────────────
old_t1_rows = (
    '| 0 | 检测 | yolo26n_safehat（兼容别名 yolo26n_e2e） | loadModel(taskid=0)；640×640 letterbox | SafeHat 主案例为 out0≈8400×14（O2M）；原始通用检测模型可为 out0≈300×6（E2E） | SafeHat 主案例采用 O2M 解码、概率校准、per-class Top-K / NMS 与语义约束；通用 E2E 模型则直接解析框、类别与置信度 |\n'
    '| 1 | 分割 | yolo26n_seg_e2e | loadModel(taskid=1)；640×640 letterbox | out0=300×38；out1=mask prototypes | 解析 bbox、类别与 32 维 mask coeffs；重建掩码并叠加轮廓 |\n'
    '| 2 | 姿态 | yolo26n_pose_e2e | loadModel(taskid=2)；640×640 letterbox | out0 宽度为 57（E2E）或 out0/out1 双分支 | 解析 bbox、17 个关键点与可见性；绘制骨骼与关键点 |\n'
    '| 3 | 分类 | yolo26n_cls | loadModel(taskid=3)；224×224 resize | out0 为类别分数向量 | Top-5 partial sort；显示类别名称与概率 |\n'
    '| 4 | OBB | yolo26n_obb_e2e | loadModel(taskid=4)；640×640 letterbox | out0 宽度为 7（E2E）或 bbox/class/angle 分支 | 解析旋转框参数与角度；绘制 RotatedRect 与标签 |'
)
new_t1_rows = (
    '| 0 | 检测 | yolo26n_safehat（兼容别名 yolo26n_e2e） | loadModel(taskid=0)；640×640 letterbox | SafeHat 主案例为 out0≈8400×14（O2M）；原始通用检测模型可为 out0≈300×6（E2E） | O2M: Top-K/NMS + 校准；E2E: 直接解析 |\n'
    '| 1 | 分割 | yolo26n_seg_e2e | loadModel(taskid=1)；640×640 letterbox | out0=300×38；out1=mask prototypes | 重建 32 维 mask，叠加掩码轮廓 |\n'
    '| 2 | 姿态 | yolo26n_pose_e2e | loadModel(taskid=2)；640×640 letterbox | out0 宽度为 57（E2E）或 out0/out1 双分支 | 解析 17 关键点，绘制骨骼 |\n'
    '| 3 | 分类 | yolo26n_cls | loadModel(taskid=3)；224×224 resize | out0 为类别分数向量 | Top-5 排序，显示类名与概率 |\n'
    '| 4 | OBB | yolo26n_obb_e2e | loadModel(taskid=4)；640×640 letterbox | out0 宽度为 7（E2E）或 bbox/class/angle 分支 | 解析旋转框与角度，绘制 RotatedRect |'
)
assert old_t1_rows in text, 'T4: Table 1 rows not found'
text = text.replace(old_t1_rows, new_t1_rows)

# ──────────────────────────────────────────────────────────
# 6. T5 — Split Table 4 into 4a / 4b
# ──────────────────────────────────────────────────────────
old_t4 = (
    '**表 4：五阶段诊断流程——各阶段定义**\n\n'
    '| 阶段 | 名称 | 触发条件 | 抽象操作（方法层） | 本研究实例化（项目层） | 退出判据 |\n'
    '|------|------|----------|-------------------|----------------------|----------|\n'
    '| S1 | 异常现象记录 | 设备端输出异常：分数爆炸、置信度固定、几何形态缺失或标签混乱 | 从设备端日志采集可量化异常信号，建立可复现问题基线 | logcat 过滤 + diagnose_background_scores.py 采集 p50、p99 与 frac_gt090 分数统计 | 现象可复现且已数值化描述 |\n'
    '| S2 | 中间输出检查 | 现象为设备特有但根因层尚未定位 | 在推理链各节点提取中间张量，比较转换前后输出形状与数值分布 | verify_ncnn_output.py 解析 NCNN param 并打印各层输出维度 | 偏差已定位至特定输出维度或数值区间 |\n'
    '| S3 | 跨后端对照 | 偏差出现在设备端但桌面端推理表现正常 | 以相同输入在桌面参考后端与设备端并行推理，通过数值对比定位差异引入层 | trace_conversion_pipeline.py + diagnose_model.py 在 PyTorch、Python-NCNN、Android-NCNN 三端联测 | 确定故障来源于导出端还是设备端实现 |\n'
    '| S4 | 结构追踪 | 故障通过 S3 后仍无法被输出数值规律解释 | 对推理图算子序列与解析器源码进行结构级对照，确认列偏移或激活语义差异 | conversion_detail_check.py 追踪 NCNN param 文件中的 concat 链 | 定位到产生错误值的具体分支、算子索引或解析假设 |\n'
    '| S5 | 回归验证 | 已应用修复方案 | 对修复前后场景执行量化回归，验证修复收益且无新引入退化 | test_letterbox_hypothesis.py / test_yolo26_e2e.py 重跑相关场景 | 分数分布回归预期范围；此前通过的场景无新引入问题 |'
)
new_t4 = (
    '**Table 4a.** Five-Stage Diagnosis Workflow: Stage Overview\n\n'
    '| 阶段 | 名称 | 触发条件 | 抽象操作（方法层） | 退出判据 |\n'
    '|------|------|----------|--------------------|----------|\n'
    '| S1 | 异常现象记录 | 设备端输出异常：分数爆炸、置信度固定、几何形态缺失或标签混乱 | 从设备端日志采集可量化异常信号，建立可复现问题基线 | 现象可复现且已数值化描述 |\n'
    '| S2 | 中间输出检查 | 现象为设备特有但根因层尚未定位 | 在推理链各节点提取中间张量，比较转换前后输出形状与数值分布 | 偏差已定位至特定输出维度或数值区间 |\n'
    '| S3 | 跨后端对照 | 偏差出现在设备端但桌面端推理表现正常 | 以相同输入在桌面参考后端与设备端并行推理，通过数值对比定位差异引入层 | 确定故障来源于导出端还是设备端实现 |\n'
    '| S4 | 结构追踪 | 故障通过 S3 后仍无法被输出数值规律解释 | 对推理图算子序列与解析器源码进行结构级对照，确认列偏移或激活语义差异 | 定位到产生错误值的具体分支、算子索引或解析假设 |\n'
    '| S5 | 回归验证 | 已应用修复方案 | 对修复前后场景执行量化回归，验证修复收益且无新引入退化 | 分数分布回归预期范围；此前通过的场景无新引入问题 |\n'
    '\n'
    '**Table 4b.** Five-Stage Diagnosis Workflow: Project-Level Instantiation\n\n'
    '| 阶段 | 名称 | 本研究实例化（项目层） |\n'
    '|------|------|------------------------|\n'
    '| S1 | 异常现象记录 | logcat 过滤 + diagnose_background_scores.py 采集 p50、p99 与 frac_gt090 分数统计 |\n'
    '| S2 | 中间输出检查 | verify_ncnn_output.py 解析 NCNN param 并打印各层输出维度 |\n'
    '| S3 | 跨后端对照 | trace_conversion_pipeline.py + diagnose_model.py 在 PyTorch、Python-NCNN、Android-NCNN 三端联测 |\n'
    '| S4 | 结构追踪 | conversion_detail_check.py 追踪 NCNN param 文件中的 concat 链 |\n'
    '| S5 | 回归验证 | test_letterbox_hypothesis.py / test_yolo26_e2e.py 重跑相关场景 |'
)
assert old_t4 in text, 'T5: Table 4 block not found'
text = text.replace(old_t4, new_t4)

# ──────────────────────────────────────────────────────────
# 7. T6 — Merge Table 6 last 2 columns + English caption
# ──────────────────────────────────────────────────────────
old_t6 = (
    '**表 6：OBB 输出列布局假设与实际布局对照**\n\n'
    '| 列区间 | 解析器初始假设 | 实际 NCNN 输出语义 | 错误后果 | 修复后处理 |\n'
    '|--------|--------------|-------------------|----------|------------|\n'
    '| 0–3 | bbox | bbox | 无 | 保留原解析 |\n'
    '| 4 | angle | class 0 probability | 第一个类别被误读为角度 | 将 angle 移至末列或 out1 |\n'
    '| 5…4+num_class-1 | classes[0…] | classes_sigmoided[1…] | 类别索引整体偏移 | 按真实类别起始列重排 |\n'
    '| 末列或 out1 | 未单独处理 / 被忽略 | angle_raw | 方向估计错误 | 单独读取角度分支并做必要变换 |'
)
new_t6 = (
    '**Table 6.** OBB Output Column Layout: Assumed vs. Actual\n\n'
    '| 列区间 | 解析器初始假设 | 实际 NCNN 输出语义 | 错误后果 / 修复策略 |\n'
    '|--------|--------------|-------------------|-----------------------|\n'
    '| 0–3 | bbox | bbox | 无 / 保留原解析 |\n'
    '| 4 | angle | class 0 probability | 类别误读为角度 / 将 angle 移至末列或 out1 |\n'
    '| 5…4+num_class-1 | classes[0…] | classes_sigmoided[1…] | 类别索引偏移 / 按真实起始列重排 |\n'
    '| 末列或 out1 | 未单独处理 / 被忽略 | angle_raw | 方向估计错误 / 单独读取角度分支 |'
)
assert old_t6 in text, 'T6: Table 6 block not found'
text = text.replace(old_t6, new_t6)

# ──────────────────────────────────────────────────────────
# 8. In-text cross-references  (figure sub-refs first, then general)
# ──────────────────────────────────────────────────────────
text = text.replace('图 8(a)', 'Figure 8(a)')
text = text.replace('图 8(b)', 'Figure 8(b)')
text = text.replace('图 8(c)', 'Figure 8(c)')

lines = text.splitlines()
result_lines = []
for line in lines:
    s = line.strip()
    if (s.startswith('|') or s.startswith('![')
            or s.startswith('**Table') or s.startswith('**Figure')):
        result_lines.append(line)
        continue
    line = re.sub(r'图\s*(\d+)', r'Figure \1', line)
    line = re.sub(r'表\s*(\d+)', r'Table \1', line)
    result_lines.append(line)
text = '\n'.join(result_lines)

# ──────────────────────────────────────────────────────────
# 9. Update "Table 4" cross-ref in body text after split
# ──────────────────────────────────────────────────────────
text = text.replace(
    'Figure 4 与 Table 4 互补：图示给出流程顺序，Table 4 给出每阶段触发条件、实例化脚本与退出判据。',
    'Figure 4 与 Tables 4a/4b 互补：图示给出流程顺序，Table 4a 给出每阶段触发条件与退出判据，Table 4b 列出实例化脚本。'
)

# ──────────────────────────────────────────────────────────
# Write and verify
# ──────────────────────────────────────────────────────────
src.write_text(text, encoding='utf-8')

# Sanity checks
remaining_fig = [m for m in re.findall(r'图\s*\d', text)
                 if '图 8' not in m or True]  # all should be gone
remaining_tbl = re.findall(r'(?<!字段|统计|样例|指标|数据|说明|证据|流程|实例|映射|分析|汇)表\s*\d', text)
print('Remaining 图N refs:', len(remaining_fig), remaining_fig[:10])
print('Remaining 表N refs:', len(remaining_tbl), remaining_tbl[:10])
print('All done — Paper.md updated.')
