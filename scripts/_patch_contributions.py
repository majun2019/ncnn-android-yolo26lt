"""Patch Paper.md §1 contributions: rewrite C1/C2/C3 and remove defensive tail sentence."""
import pathlib, sys

p = pathlib.Path("Paper.md")
text = p.read_text(encoding="utf-8")

old = (
    "1. 构建一种面向资源受限 Android 设备多任务视觉推理的部署架构，打通 Java、JNI 与 NCNN 的协同链路，并在同一应用内承载五类异构视觉任务。\n"
    "2. 提出一套面向模型导出与设备端解析偏差的部署一致性诊断流程，可对预处理、激活、坐标语义与输出布局等典型故障进行结构化定位与修复。\n"
    "3. 通过 SafeHat 主案例与五任务设备端运行覆盖，给出模型加载、空场景误检归零、关键点恢复和 OBB 方向修复等证据，说明该框架能够在资源受限 Android 设备上完成从部署到诊断再到回归验证的闭环。\n"
    "\n"
    "据此，本文聚焦于资源受限 Android 设备上的多任务视觉推理部署与一致性诊断，而不试图就单一模型精度或跨框架性能优劣展开算法比较。"
)

new = (
    "1. 提出一种面向资源受限 Android 设备的多任务视觉推理统一部署架构，"
    "通过 Java–JNI–C++–NCNN 分层协同，在单一应用内支持检测、分割、姿态、分类和 OBB 五类任务的独立加载与统一调度，"
    "解决多输出路径（E2E、One-to-Many、Legacy）共存条件下的资产管理与接口一致性问题。\n"
    "2. 提出一套多任务移动部署的一致性诊断流程，系统覆盖预处理不匹配、冗余激活、坐标语义误解和 OBB 布局误判四类典型故障，"
    "通过异常记录、中间输出检查、跨后端对照和回归验证形成结构化定位与修复闭环。\n"
    "3. 以 SafeHat 10 类 PPE 检测为主案例，提供从硬负例挖掘、模型微调到设备端诊断与回归的端到端验证记录，"
    "并建立 Kirin 970 平台五任务 CPU 推理延迟基准（检测至 OBB 任务均值 321–430 ms @ 640×640），"
    "为资源受限 Android 设备上的多任务部署研究提供可复现的参照数据。"
)

if old not in text:
    print("ERROR: target block not found", file=sys.stderr)
    sys.exit(1)

text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")
print("Done. Verifying...")

result = p.read_text(encoding="utf-8")
assert "Java\u2013JNI\u2013C++\u2013NCNN" in result, "C1 not found"
assert "四类典型故障" in result, "C2 not found"
assert "可复现的参照数据" in result, "C3 not found"
assert "而不试图就单一模型精度" not in result, "defensive sentence still present"
print("All assertions passed.")
