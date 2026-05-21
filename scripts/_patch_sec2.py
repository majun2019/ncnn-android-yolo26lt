import pathlib, re, sys

p = pathlib.Path("Paper.md")
text = p.read_text(encoding="utf-8")

old1 = (
    "这些工作大多验证了单任务检测在智能手机上的可运行性。"
    "对资源受限 Android 设备而言，进一步的问题在于不同任务输出能否在统一代码路径下被稳定解析。"
    "因此，多任务移动部署不能被简化为\u201c同一模型在终端上运行\u201d，"
    "而更接近一种共享基础链路与任务特定解析并存的系统问题。"
    "本文关注的正是这种共享部署链路下的稳定解析，而不是把五类任务并列为等强度的算法 benchmark。"
)
new1 = (
    "这些工作大多验证了单任务检测在智能手机上的可运行性。"
    "然而，当部署对象扩展至涵盖检测、分割、姿态、分类和 OBB 的多任务模型集合时，"
    "核心挑战在于不同任务输出格式差异显著——掩码原型、关键点布局、旋转框参数——"
    "如何在共享基础链路上被统一调度并稳定解析，这一问题在现有工作中尚缺乏系统讨论。"
)

old2 = (
    "相比之下，TensorFlow Lite、MNN 和 ONNX Runtime Mobile 同样是重要参照框架，"
    "但现有文献与技术资料更多关注框架接口、硬件适配或单步模型转换，"
    "而较少讨论跨层协同条件下的部署偏差如何在资源受限 Android 设备上被系统化发现、定位与修复。"
    "\n\n基于上述差异，本文不试图提出新的通用视觉模型，"
    "也不通过新增实验比较 NCNN 与其他移动推理框架的普遍优劣。"
    "本文关注的是，当多任务模型共享同一部署链路时，"
    "如何通过导出、结构检查、资产集成、跨后端对照、修正与回归形成可执行的闭环，"
    "并在出现偏差时完成定位与恢复。"
    "对低成本、广泛可得的 Android 终端而言，这一闭环直接决定模型能否从导出结果变成可运行的设备端系统。"
)
new2 = (
    "而现有文献与技术资料更多关注框架接口、硬件适配或单步模型转换，"
    "较少提供跨层协同条件下部署偏差的系统化定位与修复方法 [11\u201317]。\n\n"
    "基于上述研究空白，本文的贡献在于：为多任务模型在资源受限 Android 设备上的统一部署提供"
    "一套从导出、结构检查、资产集成到跨后端对照与回归验证的可执行闭环，"
    "并通过四类代表性故障模式说明该闭环能够实现结构化定位与修复。"
    "对低成本、广泛可得的 Android 终端而言，这一闭环直接决定模型能否从导出结果变成可运行的设备端系统。"
)

if old1 not in text:
    print("ERROR: old1 not found in text", file=sys.stderr)
    sys.exit(1)
if old2 not in text:
    print("ERROR: old2 not found in text", file=sys.stderr)
    sys.exit(1)

text = text.replace(old1, new1, 1)
text = text.replace(old2, new2, 1)

p.write_text(text, encoding="utf-8")
print("Done. Verifying...")

result = p.read_text(encoding="utf-8")
assert "尚缺乏系统讨论" in result, "patch1 verification failed"
assert "基于上述研究空白" in result, "patch2 verification failed"
assert "不试图提出新的通用视觉模型" not in result, "old defensive text still present"
assert "本文关注的正是这种共享部署链路下的稳定解析" not in result, "old defensive text still present"
print("All assertions passed.")
