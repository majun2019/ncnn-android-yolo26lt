# 面向资源受限 Android 设备的多任务 YOLO26 一致性部署：基于 Java–JNI–C++–NCNN 架构的校准化后处理与可量化故障诊断

---

## 摘要
针对资源受限 Android 设备上多任务视觉模型在导出、集成与设备端解析过程中易出现部署偏差的问题，本文提出面向资源受限 Android 设备的多任务视觉推理统一部署与一致性诊断框架。该框架以 yolo26n 系列模型资产覆盖检测、分割、姿态、分类和 OBB 五类任务，解决 E2E、One-to-Many（O2M）和 Legacy 多输出路径并存条件下的资产命名冲突、接口语义分歧与任务调度耦合问题，在 Java–JNI–C++–NCNN 协同链路下实现统一加载、统一调度与一致性接口管理，并在统一架构内给出概率校准（$T = 6$、$\gamma = 1.6$）、per-class Top-K（$K_c = 50$）、互斥类抑制、对象依赖关联（重叠阈值 0.15 / 0.20）与 PPE 框锁定（0.35:0.65 加权）共五类运行时常驻的后处理一致性算法。针对预处理不匹配、冗余激活、坐标语义误解和 OBB 布局误判四类典型故障，本文进一步提出由异常记录、中间输出检查、跨后端对照、结构追踪和回归验证组成的五阶段一致性诊断流程，并给出可量化的修复证据：预处理偏差使空场景 $p_{50}$ 由 0.84–0.96 降至 < 0.0001；冗余激活修复后模型原始概率 $p$ 由固定约 0.5 恢复为多峰分布（实测 $p_{50}\approx 0.34$、$p_{95}\approx 0.74$、上限约 0.95），UI 显示置信度 $\tilde p$ 经 §3.3 校准压缩后稳定在 0.35–0.73。以 SafeHat 10 类 PPE 检测为主案例，本文对上述部署架构与诊断流程给出从硬负例挖掘、模型微调、NCNN 导出、资产集成到 Android 真机回归的端到端闭环验证，并完成统一应用内五任务运行覆盖；针对单参数（$T$、$\gamma$、$K_c$、关联重叠、PPE 锁定权重）的系统性消融留待后续工作。在 Kirin 970 CPU 路径下，检测、分割、姿态与 OBB 四类 640×640 任务的推理均值为 321.3–430.5 ms，分类任务在 224×224 输入下的均值为 33.6 ms。上述结果说明，所提出的部署架构、后处理算法集合与诊断流程能够为依赖低成本、广泛可得 Android 设备的边缘智能系统提供从导出、部署到诊断的一体化实现路径。

**关键词：** Android；资源受限 Android 设备；NCNN；部署一致性；边缘智能；多任务视觉推理；故障诊断

**English Title:** Consistency-Aware Deployment of Multi-Task YOLO26 on Resource-Constrained Android Devices: A Java–JNI–C++–NCNN Architecture with Calibrated Post-Processing and Quantitative Fault Diagnosis

**Abstract:** This paper addresses deployment deviations that frequently arise when multi-task vision models are exported, integrated, and parsed on resource-constrained Android devices. We present an NCNN-based framework that integrates Java, JNI, C++, and NCNN for multi-task vision deployment and consistency diagnosis. YOLO26 is treated as an engineering-managed family of yolo26n assets covering detection, segmentation, pose estimation, classification, and oriented bounding box (OBB) detection, with support for E2E, One-to-Many (O2M), and Legacy output paths. The framework resolves asset naming conflicts, interface semantic divergence, and task-scheduling coupling across these output paths within the Java–JNI–C++–NCNN pipeline, and within the same architecture supplies a runtime-resident, parameterizable post-processing consistency algorithm set—probability calibration ($T = 6$, $\gamma = 1.6$), per-class Top-K ($K_c = 50$), mutually exclusive class suppression, object-dependent association (overlap thresholds 0.15 / 0.20), and PPE box locking (0.35:0.65 weighting)—that gives on-device thresholding and multi-class competition an interpretable, auditable mathematical form. We further propose a five-stage diagnosis workflow consisting of anomaly logging, intermediate-output inspection, cross-backend comparison, structure tracing, and regression validation, instantiated on four representative fault classes (preprocessing mismatch, redundant activation, coordinate semantics mismatch, and OBB layout misinterpretation), and produce quantitative repair evidence: preprocessing correction lowers the empty-scene $p_{50}$ from 0.84–0.96 to < 0.0001; after the redundant-activation fix the model raw probability $p$ recovers from a degenerate value near 0.5 to a multi-modal distribution (measured $p_{50}\approx 0.34$, $p_{95}\approx 0.74$, upper bound $\approx 0.95$), while the UI-displayed score $\tilde p$, after the §3.3 calibration and compression, stabilizes within 0.35–0.73. Using SafeHat 10-class PPE detection as the main case, we provide an end-to-end closed-loop validation from hard-negative mining and model fine-tuning to NCNN export, asset integration, and on-device regression, and verify runtime coverage of all five tasks within a single Android application; systematic single-parameter ablations ($T$, $\gamma$, $K_c$, association overlap, PPE locking weights) are left to future work. On the Kirin 970 CPU path, the four 640×640 tasks averaged 321.3–430.5 ms per inference, while classification averaged 33.6 ms at 224×224. These results indicate that the proposed deployment architecture, post-processing algorithm set, and diagnosis workflow provide a practical path from export and deployment to diagnosis for edge AI applications built on low-cost, widely available Android terminals.

**Keywords:** Android deployment; resource-constrained Android devices; NCNN; multi-task vision inference; deployment consistency; fault diagnosis; edge AI

---

## 1. 引言

随着低时延响应、隐私保护和自主运行需求的持续增长，深度视觉推理正在从云端服务器加速迁移到边缘设备。在建筑施工安检、工业安全巡查、移动机器人、智能终端和现场监测等场景中，端侧推理能够在本地完成图像采集、模型执行与结果反馈，从而减少通信开销并提升系统响应效率。这一趋势一方面推动了轻量化检测器和移动视觉骨干的持续优化 [1–6]，另一方面也推动了 TensorFlow Lite、MNN、ONNX Runtime Mobile 和 NCNN 等移动端推理框架的发展 [7–10]。对于嵌入式系统、移动计算和边缘 AI 场景，如何在资源受限 Android 设备上稳定承载视觉模型，已从工程实现层面上升为系统设计层面的核心命题。本文所指资源受限 Android 设备，是使用通用移动 CPU 核心、不具备专用神经网络加速路径、在 fp32 精度下对 640×640 输入的推理吞吐量显著低于实时帧率阈值的在役移动设备，以低成本、广泛可得的中低端机型为典型代表。尤其在预算受限、网络条件不稳定或算力基础设施不足的应用环境中，低成本、广泛可得的 Android 设备往往是边缘智能系统最现实的硬件载体。因此，围绕资源受限 Android 设备展开部署研究，不仅面向工程实现，也关系到边缘智能能力能否以较低门槛进入更广泛的实际场景。已有研究已在农学表型识别与视觉辅助等具体任务中完成了基于 YOLOv8 或轻量化 YOLO 网络的 Android 设备部署，并以智能手机为主要载体验证了资源受限条件下本地视觉推理的现实可行性 [11,12]。这些结果说明资源受限条件下的单任务本地视觉推理已经具备工程可行性。然而，当部署对象扩展为本文统一管理的 YOLO26 系列模型资产时，问题不再只是单一模型能否在设备上运行，而转化为多任务统一承载、E2E、One-to-Many 与 Legacy 多输出路径并存，以及预处理、输出布局和设备端解析一致性更难保障的系统挑战。

然而，对 Android 部署来说，桌面端“模型可用”并不等于设备端“系统可落地”。一个在 Python 环境中运行正常的模型，仍可能因为预处理不一致、输出布局误判、转换链语义变化或工具链兼容性问题而在移动设备上失效。这类故障通常不会直接体现在常规 benchmark 指标中，却会直接决定模型能否进入真实应用流程。部署一致性及其诊断方法是正式研究对象，其出现的故障无法通过附属于模型训练的零散排障来稳定解决。

本文所称的 YOLO26 是本项目在工程上统一管理的 yolo26n 系列模型资产，包含 E2E、One-to-Many 和 Legacy 三类输出路径，以及检测、分割、姿态、分类和 OBB 五类任务模型，并非新的通用模型家族。

围绕这一问题，本文将研究目标明确限定为：面向资源受限 Android 设备，建立一条可统一承载多任务模型、并能够对部署偏差进行结构化诊断与恢复的系统路径。为此，本文提出一套基于 Java、JNI、C++ 与 NCNN 的设备端部署框架，使目标检测、实例分割、姿态估计、图像分类和旋转框检测五类任务能够在同一应用流程内被统一加载、调度和观察；并进一步提出由异常记录、中间输出检查、跨后端对照、结构追踪与回归验证构成的五阶段一致性诊断流程，用于对部署偏差进行结构化定位与恢复。在此基础上，本文以 SafeHat 安全帽检测为主案例，完成从模型微调、NCNN 导出、资产替换、真机运行到故障修复和回归验证的闭环验证。

本文的主要贡献如下：

1. **统一部署架构与后处理一致性算法。** 提出面向资源受限 Android 设备的 Java–JNI–C++–NCNN 四层多任务部署架构，解决 E2E、One-to-Many 与 Legacy 多输出路径并存条件下五类任务的资产命名冲突、接口语义分歧与调度耦合问题；并在统一架构内给出可参数化、运行时常驻的后处理一致性算法集合——概率校准（$T = 6$、$\gamma = 1.6$）、per-class Top-K（$K_c = 50$）、互斥类抑制、对象依赖关联（重叠阈值 0.15 / 0.20）与 PPE 框锁定（0.35:0.65 加权），使设备端阈值过滤与多类别竞争具有可解释、可复核的数学形式。
2. **可量化的五阶段一致性诊断流程。** 提出由异常记录、中间输出检查、跨后端对照、结构追踪和回归验证组成的五阶段诊断流程，并以预处理不匹配、冗余激活、坐标语义误解和 OBB 布局误判四类典型故障实例化；其中预处理偏差给出 $p_{50}: 0.84\text{–}0.96 \to < 0.0001$ 的修复前后量化对照，冗余激活给出模型原始概率 $p$（实测 $p_{50}\approx 0.34$、$p_{95}\approx 0.74$、上限约 0.95）与 UI 显示值 $\tilde p \in 0.35\text{–}0.73$ 的双层口径，使设备端故障定位从经验排障升级为可复现的数值证据。
3. **SafeHat 端到端闭环验证与设备端运行基线。** 以 SafeHat 10 类 PPE 检测为主案例，给出从硬负例挖掘、模型微调到设备端诊断与回归的端到端闭环验证；统一应用内五任务真机运行覆盖与上述后处理算法集合的运行时承载由 Figure 9 与 Table 8 行 4 共同证据化，对单参数（$T$、$\gamma$、$K_c$、关联重叠、PPE 锁定权重）的系统性消融留待后续工作；并在 Kirin 970 CPU 路径下报告五任务运行量级基线（四类 640×640 任务均值 321.3–430.5 ms，分类 224×224 均值 33.6 ms），为资源受限 Android 设备的部署评估提供可复现参照。

本文其余部分组织如下。第 2 节讨论资源受限 Android 设备部署与边缘推理系统相关工作，并说明本文的研究范围与现有工作的差异。第 3 节给出资源受限 Android 设备统一部署架构及其部署流程。第 4 节提出资源受限 Android 设备部署一致性诊断方法，并分析四类代表性故障模式。第 5 节通过 SafeHat 主案例、五任务运行覆盖和设备端延迟表征对系统进行验证，并讨论适用范围与局限性。第 6 节总结全文并给出后续工作方向。

---

## 2. 资源受限 Android 设备部署与边缘推理系统相关工作

现有工作主要涉及三类内容：面向资源受限场景的轻量化视觉模型与高效推理结构、移动端推理框架与 Android 原生集成、以及模型转换后的一致性验证与设备端诊断。第一类工作关注 YOLO、EfficientDet、MobileNet、ShuffleNet 等模型如何通过骨干压缩、特征融合优化和计算量控制提升低算力、低功耗设备上的可部署性 [1–6,13–15]。第二类工作聚焦 TensorFlow Lite、MNN、ONNX Runtime Mobile 和 NCNN 等移动端推理框架，它们在平台适配、算子覆盖、模型转换路径和硬件加速支持上形成不同实现路线 [7–10]。第三类工作体现为 Android NDK、相机接口、OpenCV、Vulkan、NCNN Wiki 与 PNNX 等技术资料和工具链实践 [16–21]，它们为原生部署提供了组件级基础，但较少把模型导出、真机部署和回归验证放在同一系统闭环中讨论。

从模型侧看，轻量化检测器与端到端检测的发展并不只是提升推理速度，也改变了部署端需要面对的输出语义。检测分支往往受益于更直接的 E2E 输出，而分割、姿态、分类和 OBB 任务仍保留掩码原型、关键点、类别向量和旋转参数等不同结构 [22–27]。近期也已有工作将 YOLOv8 或轻量化 YOLO 网络部署到 Android 终端，并在稻粒识别与视觉辅助等单任务场景中完成验证 [11,12]。这些工作大多验证了单任务检测在智能手机上的可运行性。然而，当部署对象扩展至涵盖检测、分割、姿态、分类和 OBB 的多任务模型集合时，核心挑战在于不同任务输出格式差异显著——掩码原型、关键点布局、旋转框参数——如何在共享基础链路上被统一调度并稳定解析，这一问题在现有工作中尚缺乏系统讨论。

从系统侧看，移动部署的难点也不只在于选择某个推理框架，而在于如何把模型导出、原生推理、图像处理、相机接入和设备端可观测性组织成一致的执行链路。对资源受限 Android 设备而言，这一问题还叠加了成本、功耗、算力和网络依赖等现实约束，使“能否在普及度高、替换门槛低的终端上稳定运行”本身成为系统设计目标。NCNN 之所以适合作为本文的实现环境，在于其轻量级 C++ 运行时与成熟 ARM CPU 适配，能够与 Android 原生相机访问、OpenCV 图像处理以及 JNI 应用集成形成较高透明度的协同路径；Vulkan GPU 路径在框架层面保留，但本文实验均在 CPU 路径下完成 [10,16–21]。而现有文献与技术资料更多关注框架接口、硬件适配或单步模型转换，较少提供跨层协同条件下部署偏差的系统化定位与修复方法 [10,16–21]。

基于上述研究空白，本文的贡献在于：提出面向资源受限 Android 设备的多任务视觉推理统一部署架构，并在该架构内给出五类运行时常驻的后处理一致性算法（概率校准、per-class Top-K、互斥类抑制、对象依赖关联与 PPE 框锁定）；进一步提出由异常记录、中间输出检查、跨后端对照、结构追踪与回归验证构成的五阶段一致性诊断流程，并以 raw $p$ 与 UI $\tilde p$ 的双层量化口径承载证据；SafeHat 主案例与统一应用内五任务真机运行覆盖用于对架构、算法集合与诊断流程进行联合验证。对低成本、广泛可得的 Android 终端而言，这一部署与诊断组合直接决定模型能否从导出结果变成可运行的设备端系统。

---

## 3. 资源受限 Android 设备统一部署架构

针对多任务模型在资源受限 Android 设备上统一承载与一致性诊断的挑战，本文提出 Java–JNI–C++–NCNN 四层协同的统一部署架构，将模型导出、设备端部署与回归验证组织为一条可观测的闭环路径（Figure 1）。

**Figure 1.** Model Export and On-Device Validation Closed Loop

```text
训练或微调权重
	PyTorch 多任务权重（检测 / 分割 / 姿态 / 分类 / OBB）
						↓
模型导出环节
	- 选择 E2E / One-to-Many / Legacy 输出路径
	- Ultralytics export（format=ncnn；内部经 PyTorch → ONNX → onnx2ncnn）
						↓
中间与最终模型资产
	ONNX 中间表示 与 NCNN param/bin 文件对
						↓
结构验证环节
	- 检查输出张量形状、分支组织、列布局
	- 跨后端数值一致性比对
						↓
Android 资产部署
	统一命名的 *.ncnn.{param,bin} 资产
						↓
设备端加载与运行
	模型加载 → 相机输入 → 渲染输出（Java–JNI–C++–NCNN 协同）
						↓
日志与跨后端对照
	设备端日志采集 + 结构追踪 + Python 端参考实现对照
						↓
回归验证与结果汇总
	延迟与覆盖度回归 → 验证证据
```

*Figure 1 强调：从 PyTorch 权重到 Android 真机结果之间并非单步转换，而是“导出 – 结构验证 – 资产部署 – 对照诊断 – 回归验证”的闭环；图中各环节以功能模块刻画，对应的具体脚本与资产命名见第 5 节 SafeHat 实现部分（Figure 3）。*

### 3.1 设计目标与系统需求

本文的系统设计目标不是单纯让模型在 Android 终端上运行，而是在资源受限 Android 设备上实现多任务视觉推理的统一承载、统一调度和统一诊断。围绕这一目标，本文对系统提出四项基本要求：其一，多个任务应共享尽可能一致的模型加载入口、相机输入路径和结果显示机制，以降低跨任务维护成本；其二，设备端预处理必须与训练和导出时的假设保持一致，否则不同任务的解析可靠性将受到破坏；其三，系统需要保留足够的日志与中间状态可观测性，使部署偏差能够被定位而不是被动接受；其四，统一链路不应抹平任务差异，系统必须允许检测、分割、姿态、分类和 OBB 在输出解析层面保留各自语义。

在上述约束下，本文将 YOLO26 视为一组工程统一管理的模型资产，而不是新的模型方法本身。系统设计聚焦于统一入口、统一调度与一致性恢复，SafeHat 检测则作为完整闭环主案例承载训练、导出和设备端验证证据。

### 3.2 Java–JNI–C++–NCNN 四层协同架构

为在单一应用内统一承载五类任务并保留设备端可观测性，本文设计了 Java–JNI–C++–NCNN 四层分层架构，以资源受限 Android 设备上的稳定运行和可观测诊断为前提。Java 层负责界面交互、任务切换、阈值调节和渲染控制；JNI 层承担模型加载、参数传递与原生对象生命周期管理；C++ 层封装检测、分割、姿态、分类和 OBB 等任务的核心推理逻辑；底层依赖层则由 NCNN、OpenCV-Mobile、Android NDK Camera 与可选的 Vulkan GPU 路径共同构成。之所以采用 JNI 与 NCNN 组合，而不是将部署逻辑完全停留在更高层封装之中，原因在于 JNI 可以提供更直接的参数控制和日志插桩能力，NCNN 则能够提供更透明的原生执行环境，使资源受限 Android 设备上的异常行为更易被结构化分析。

一个关键设计点在于预处理一致性。除分类任务外，其余四类任务均采用 640×640 的 letterbox 预处理：先按比例缩放使长边匹配目标尺寸，再以 pad_value = 114 对称补边至 640×640 正方形张量。分类任务则按模型资产要求使用 224×224 resize。这样的区分使训练假设、导出假设与设备端执行保持一致，也为后续诊断中识别预处理偏差奠定了基线。

**Figure 2.** Overall System Architecture for Multi-Task Vision Inference on Resource-Constrained Android Devices

```text
Java / UI 层 [Java]
	MainActivity
	- 任务切换: detect / segment / pose / classify / obb
	- 后端切换: CPU / GPU(Vulkan/Turnip)
	- 阈值调节: prob / nms
	- 相机切换 + SurfaceView 渲染
							↓
JNI 桥接层 [Java native 声明  ←→  C++ JNIEXPORT 实现]
	Java 侧: YOLO26Ncnn.java
	- loadModel() / setDetectThresholds()
	- openCamera() / closeCamera() / setOutputWindow()
	C++ 侧: yolo26ncnn.cpp 中的 JNIEXPORT 入口
	- 类型转换、JavaVM/JNIEnv 句柄管理
	- 原生对象生命周期与异常透传
							↓
C++ 原生调度层 [C++]
	yolo26ncnn.cpp / yolo26.h
	- taskid=0..4 分发到五类任务实现
	- 管理 CPU 与 GPU 路径
	- 连接相机回调、日志与绘制
							↓
C++ 任务推理层 [C++]
	yolo26_det.cpp / seg.cpp / pose.cpp / cls.cpp / obb.cpp
	- 共享: 模型加载、阈值、预处理、绘制入口
	- 差异: bbox / mask / keypoints / Top-K / angle 解析
							↓
一致性算法层（运行时常驻，横切五类任务，C++ 实现，详见 §3.3）
	输入侧 — 预处理一致性约束
	- 640×640 letterbox（pad_value = 114，对称补边到正方形张量）
	- 224×224 resize（分类任务）
	- 反映射时使用与补边一致的整数 (p_x, p_y, r)
	输出侧 — 后处理一致性算法集合
	- 概率校准（温度 T、幂次 γ）
	- per-class Top-K（每类候选上限 K_c）
	- 语义约束 NMS（per-class + 互斥类抑制）
	- 对象依赖关联（PPE↔Person 重叠阈值）
	- PPE 框锁定（位置/尺寸加权）
							↓
底层依赖 [C/C++ 原生库]
	NCNN + OpenCV-Mobile + Android NDK Camera + Vulkan GPU 组件
```

*对应实现：Java 侧 MainActivity.java、YOLO26Ncnn.java；C++ 侧 yolo26ncnn.cpp（含 JNIEXPORT 入口）、yolo26.h 与五个 yolo26_*.cpp。一致性算法层的具体公式与参数取值见 §3.3，针对具体输出路径（E2E / O2M / Legacy）的实例化与量化证据见第 5 节。*

### 3.3 设备端预处理约束与后处理算法

针对设备端输入分布偏移与多类别混淆问题，本文提出预处理一致性约束与后处理校准机制，作为第 4 节四类故障的分析基础。

**预处理一致性约束。** 设备端输入预处理必须与训练时的假设严格对齐，否则输入分布偏移将直接造成置信度异常。本文所有 640×640 任务采用统一 letterbox 预处理，设原始图像宽高为 W 和 H，目标输入边长为 S，则缩放系数 r、缩放后尺寸 W' 和 H' 以及对称补边 p_x 和 p_y 分别为：

$$
r = \min\left(\frac{S}{W}, \frac{S}{H}\right), \quad W' = \lfloor rW \rfloor, \quad H' = \lfloor rH \rfloor
$$

$$
p_x = \left\lfloor\frac{S - W'}{2}\right\rfloor, \quad p_y = \left\lfloor\frac{S - H'}{2}\right\rfloor
$$

其中 $p_x, p_y$ 取整至最近整数，与实现中 pad\_left / pad\_top 的实际值保持一致，反映射时直接使用同一整数补边量。

部署端检测框从补边坐标系反映射回原图坐标系时，采用：

$$
x = \frac{\hat{x} - p_x}{r}, \quad y = \frac{\hat{y} - p_y}{r}, \quad w = \frac{\hat{w}}{r}, \quad h = \frac{\hat{h}}{r}
$$

预处理必须补边至严格的 target_size × target_size 正方形张量，而不是仅补齐到步长对齐尺寸。对非正方形相机输入（如 1920×1080），步长对齐补边会产生 640×384 的宽高失配张量；该失配对背景分数分布的影响将在第 4.2 节作为故障模式 I 进行量化分析。

**概率校准。** 当检测模型采用 One-to-Many 输出路径时，输出候选中常出现概率高分饱和现象（大量候选集中在 0.99 附近），使阈值调节和多类别竞争的可分性下降。设备端后处理采用基于 logit 的温度缩放并叠加幂次压缩进行概率校准，设模型给出的类别概率为 p，经校准后的显示概率 $\tilde{p}$ 为：

$$
\tilde{p} = \left[\sigma\!\left(\frac{\log \frac{p}{1-p}}{T}\right)\right]^{\gamma}, \quad T = 6,\ \gamma = 1.6
$$

其中 $\sigma(\cdot)$ 为 Sigmoid 函数，T = 6 为温度参数，$\gamma$ = 1.6 为后置幂次压缩指数。该变换保持候选排序关系不变，但能够压缩过饱和概率并放大高分区差异（示例：0.999→0.66、0.995→0.59、0.950→0.43、0.900→0.36），使阈值调节和多类别竞争更具可分性。对候选框集合 $\mathcal{B}$ 完成排序后，设备端为每个类别单独保留前 $K_c$ 个候选（per-class Top-K，本项目 SafeHat 实现中 $K_c = 50$），以避免高分背景类挤占目标类别的候选空间。

**语义约束 NMS。** 对于类别间存在合法共存或语义互斥关系的检测任务，agnostic NMS 会误删真实共存目标。对于两个候选框 $B_i$ 和 $B_j$，其交并比定义为：

$$
\operatorname{IoU}(B_i, B_j) = \frac{|B_i \cap B_j|}{|B_i \cup B_j|}
$$

本文采用 per-class NMS：仅在同类候选之间执行抑制，保留不同类别在同一空间位置合法共存的可能性。对于真正存在语义互斥的类别对（如 Hardhat 与 No-Hardhat），额外引入互斥类别抑制规则：若两个互斥类别框的 IoU 超过阈值，则保留概率更高者、抑制概率更低者。

**对象依赖约束。** 对于 PPE 等天然依附于人员主体的类别，若场景中不存在人员候选，则相关 PPE 类别应被直接抑制；若存在人员候选，则仅保留与某个 Person 框满足最小关联重叠阈值的 PPE 候选。对 PPE 框 $B_\text{ppe}$ 与 Person 框 $B_\text{person}$，定义基于 PPE 框面积的关联重叠率为：

$$
\operatorname{overlap}(B_\text{ppe}, B_\text{person}) = \frac{|B_\text{ppe} \cap B_\text{person}|}{|B_\text{ppe}|}
$$

本项目实现对正向 PPE 类别（Hardhat、Mask、Safety Vest）采用 0.15 的关联重叠阈值，对负向 PPE 类别（No-Hardhat、No-Mask、No-Safety Vest）采用略严的 0.20 阈值，以减少负样本误关联。

若满足最小重叠阈值，设备端进一步将 PPE 框锁定到匹配到的 Person 框上，以减少悬浮框和随机漂移。设 PPE 框与 Person 框分别表示为 $(x, y, w, h)$ 形式（左上角坐标加宽高），则更新后的 PPE 框各分量分别为：

$$
\begin{aligned}
x' &= 0.35\, x_\text{ppe} + 0.65\, x_\text{person}, & y' &= 0.35\, y_\text{ppe} + 0.65\, y_\text{person} \\
w' &= 0.35\, w_\text{ppe} + 0.65\, w_\text{person}, & h' &= 0.35\, h_\text{ppe} + 0.65\, h_\text{person}
\end{aligned}
$$

该更新对位置和尺寸均按 0.35:0.65 加权，使 PPE 框在视觉上靠近匹配 Person 框的几何范围，从而抑制纹理漂移导致的孤立小框。

上述预处理约束与后处理算法共同构成设备端部署的算法基础，独立于任何具体任务实现而成立；其针对具体输出路径（E2E / One-to-Many / Legacy）的实例化形式、对应资产命名以及量化验证结果由第 5 节统一承载。

### 3.4 统一任务调度与解析路径

为避免形成彼此割裂的任务专属应用，本文框架采用统一调度机制。系统通过任务标识符选择对应模型与解析模块，而相机输入路径、加载接口和显示入口尽量保持共享。因此，同一应用可以在相同资源约束条件下切换五类任务，这不仅提升了代码复用性，也为不同任务之间的对照诊断提供了基础。

尽管运行路径统一，各任务的输出语义并不相同。检测任务直接解析目标位置、类别与置信度；分割任务需要在检测输出之上结合原型和系数恢复掩码；姿态任务还要进一步解释关键点坐标与可见性；分类任务关注类别分数向量的 Top-K 排序；OBB 任务则需恢复旋转框参数与方向。因此，本文强调的是统一调度与统一承载，而不是语义同构化。

**Table 1.** Unified Scheduling and Task-Specific Parsing for Five Task Types

| 任务ID | 任务类型 | 模型资产 | 输入与预处理 | 主要输出语义 | 任务特定解析与显示 |
|--------|----------|----------|--------------|--------------|--------------------|
| 0 | 检测 | yolo26n_safehat（兼容别名 yolo26n_e2e） | loadModel(taskid=0)；640×640 letterbox | SafeHat 主案例为 out0≈8400×14（O2M）；原始通用检测模型可为 out0≈300×6（E2E） | O2M: Top-K/NMS + 校准；E2E: 直接解析 |
| 1 | 分割 | yolo26n_seg_e2e | loadModel(taskid=1)；640×640 letterbox | out0=300×38；out1=mask prototypes | 重建 32 维 mask，叠加掩码轮廓 |
| 2 | 姿态 | yolo26n_pose_e2e | loadModel(taskid=2)；640×640 letterbox | out0 宽度为 57（E2E）或 out0/out1 双分支 | 解析 17 关键点，绘制骨骼 |
| 3 | 分类 | yolo26n_cls | loadModel(taskid=3)；224×224 resize | out0 为类别分数向量 | Top-5 排序，显示类名与概率 |
| 4 | OBB | yolo26n_obb_e2e | loadModel(taskid=4)；640×640 letterbox | out0 宽度为 7（E2E）或 bbox/class/angle 分支 | 解析旋转框与角度，绘制 RotatedRect |

### 3.5 模型导出、资产集成与设备端部署流程

本文将面向资源受限 Android 设备的部署流程组织为模型准备、NCNN 导出、结构验证、资产部署和真机回归五个阶段。在模型准备阶段，系统可使用公开预训练权重，也可使用微调后的任务权重；导出阶段负责将任务模型转换为 NCNN 的 param/bin 文件对；结构验证阶段用于检查输出维度、分支组织和张量列布局；资产部署阶段将模型文件集成到 Android 工程；真机回归阶段则用来确认设备端加载、解析逻辑与 Python 侧参考行为保持一致。该流程并非附属性工程操作说明，而是资源受限 Android 设备部署闭环的有机组成。

**Table 2.** Tools, Inputs, and Validation Goals for Each Deployment Stage

| 阶段 | 输入 / 资产 | 主要工具或脚本 | 阶段输出 | 验证目标 |
|------|------------|----------------|----------|----------|
| 模型准备 | safehat.yaml、data/、yolo26n.pt / best.pt | yolo_26_train.py、Ultralytics | 微调权重 best.pt | 类别定义正确；训练与验证流程可复现 |
| NCNN 导出 | PyTorch 权重 | export_yolo26_ncnn.py（Ultralytics export，format=ncnn）| param/bin 文件对 | E2E、One-to-Many 与任务分支导出成功 |
| 结构验证 | param/bin、原始模型 | verify_ncnn_output.py、conversion_detail_check.py | 输出形状、分支和列布局说明 | 确认 out0/out1 等张量语义与解析器一致 |
| 资产部署 | app/src/main/assets/、Android 工程 | export_best_safehat_to_assets.py、Gradle/NDK | Android 可加载模型资产 | loadModel 无报错；JNI/资源路径正确 |
| 真机回归 | APK、摄像头输入、logcat | trace_conversion_pipeline.py、diagnose_model.py、collect_latency.py | 功能日志、故障定位结果、延迟统计 | 部署结果与 Python 参考一致；修复后无回归 |

在五类任务中，SafeHat 检测构成了最完整的案例研究。该数据集覆盖工业场景下与安全帽和个人防护装备相关的十个类别，整个流程包括微调训练、权重筛选、模型导出、资产替换和设备端验证。之所以将其作为全文主线，不是因为它代表五任务中的唯一算法贡献，而是因为它将训练、部署、排障和回归验证连接成了一条最完整的系统闭环。SafeHat 主案例在导出阶段保留 One-to-Many 输出路径（out0 形状为 8400×(nc+4)），而非通用 E2E 路径（out0 形状为 300×6）；这一选择由结构验证阶段确认输出维度与类别数后确定，并在 §3.3 的设备端后处理中实例化应用。检测主案例采用语义名为 yolo26n_safehat 的 NCNN 资产，工程中兼容别名 yolo26n_e2e；文件名中的 e2e 为历史兼容别名，不代表当前 SafeHat 主案例采用了端到端检测头。

**Table 3.** SafeHat Dataset Classes and Application Semantics

| 类别ID | 类别名 | 语义分组 | 应用含义 |
|--------|--------|----------|----------|
| 0 | Hardhat | PPE 合规 | 人员正确佩戴安全帽 |
| 1 | Mask | PPE 合规 | 人员正确佩戴口罩 |
| 2 | No-Hardhat | PPE 违规 | 人员未佩戴安全帽 |
| 3 | No-Mask | PPE 违规 | 人员未佩戴口罩 |
| 4 | No-Safety Vest | PPE 违规 | 人员未穿安全背心 |
| 5 | Person | 人员主体 | 人员目标本体 |
| 6 | Safety Cone | 场景安全设施 | 施工或隔离锥桶 |
| 7 | Safety Vest | PPE 合规 | 人员正确穿着安全背心 |
| 8 | Machinery | 施工设备 | 机械设备或工程车辆部件 |
| 9 | Vehicle | 交通 / 作业车辆 | 道路或工地车辆目标 |

**Figure 3.** End-to-End Closed-Loop Pipeline from SafeHat Training to Android Deployment

```text
safehat.yaml + data/train|valid|test
			↓
yolo_26_train.py / Ultralytics
			↓
runs/detect/runs/train/yolo26n_safehat/weights/best.pt
			↓
NCNN 导出与结构验证
	export_yolo26_ncnn.py；verify_ncnn_output.py
			↓
Android 资产生成
	export_best_safehat_to_assets.py
			↓
app/src/main/assets/yolo26n_safehat.ncnn.{param,bin}
	或兼容别名 yolo26n_e2e.ncnn.{param,bin}
			↓
Android App: loadModel() → openCamera() → setOutputWindow()
			↓
logcat + diagnose_* + collect_latency.py
			↓
Figure 9、Table 8、Table 9、Table 10
```

*训练侧证据见第 5 节 Figure 8(a) 与 Figure 8(b)；部署侧证据见 Figure 9、Table 8、Table 9 和 Table 10。*

---

## 4. 资源受限 Android 设备部署一致性诊断方法

### 4.1 诊断流程

资源受限 Android 设备上的部署故障通常不表现为直接崩溃，而是体现为异常置信度分布、空输出、错误可视化或工程行为不一致，且往往横跨模型导出、预处理、张量解析与平台实现等多个层次，使零散修补难以形成稳定方法。为此，本文提出一套五阶段一致性诊断流程，包括异常现象记录、中间输出检查、跨后端对照、结构追踪与回归验证，并以 SafeHat 案例加以验证。该流程依次回答五个问题：问题首先表现为什么、偏差最早出现在链路哪个环节、故障来源于导出端还是设备端、具体是哪个张量分支或解析假设导致异常，以及修复后是否真正恢复了部署一致性。

**Figure 4.** Fault Diagnosis Workflow for Deployment on Resource-Constrained Android Devices

```text
异常现象
	分数爆炸 / 固定 0.5 / 几何缺失 / 标签混乱
						↓ S1
日志采集与现象量化
	logcat + 分数统计脚本
						↓ S2
中间输出检查
	输出形状、数值范围、异常层位
						↓ S3
跨后端对照
	PyTorch ↔ Python-NCNN ↔ Android-NCNN
						↓ S4（当 S3 仍不足以解释时）
结构追踪
	param concat 链 / 解析器列布局 / 激活语义
						↓ S5
回归验证
	复现场景重跑 + 既有用例不退化
```

*Figure 4 与 Tables 4a/4b 互补：图示给出流程顺序，Table 4a 给出每阶段触发条件与退出判据，Table 4b 列出实例化脚本。*

**Table 4a.** Five-Stage Diagnosis Workflow: Stage Overview

| 阶段 | 名称 | 触发条件 | 抽象操作（方法层） | 退出判据 |
|------|------|----------|--------------------|----------|
| S1 | 异常现象记录 | 设备端输出异常：分数爆炸、置信度固定、几何形态缺失或标签混乱 | 从设备端日志采集可量化异常信号，建立可复现问题基线 | 现象可复现且已数值化描述 |
| S2 | 中间输出检查 | 现象为设备特有但根因层尚未定位 | 在推理链各节点提取中间张量，比较转换前后输出形状与数值分布 | 偏差已定位至特定输出维度或数值区间 |
| S3 | 跨后端对照 | 偏差出现在设备端但桌面端推理表现正常 | 以相同输入在桌面参考后端与设备端并行推理，通过数值对比定位差异引入层 | 确定故障来源于导出端还是设备端实现 |
| S4 | 结构追踪 | 故障通过 S3 后仍无法被输出数值规律解释 | 对推理图算子序列与解析器源码进行结构级对照，确认列偏移或激活语义差异 | 定位到产生错误值的具体分支、算子索引或解析假设 |
| S5 | 回归验证 | 已应用修复方案 | 对修复前后场景执行量化回归，验证修复收益且无新引入退化 | 分数分布回归预期范围；此前通过的场景无新引入问题 |

**Table 4b.** Five-Stage Diagnosis Workflow: Project-Level Instantiation

| 阶段 | 名称 | 本研究实例化（项目层） |
|------|------|------------------------|
| S1 | 异常现象记录 | logcat 过滤 + diagnose_background_scores.py 采集 p50、p99 与 frac_gt090 分数统计 |
| S2 | 中间输出检查 | verify_ncnn_output.py 解析 NCNN param 并打印各层输出维度 |
| S3 | 跨后端对照 | trace_conversion_pipeline.py + diagnose_model.py 在 PyTorch、Python-NCNN、Android-NCNN 三端联测 |
| S4 | 结构追踪 | conversion_detail_check.py 追踪 NCNN param 文件中的 concat 链 |
| S5 | 回归验证 | test_letterbox_hypothesis.py / test_yolo26_e2e.py 重跑相关场景 |

### 4.2 故障模式 I：预处理不匹配

本节按 S1→S3→S2→S5 路径对预处理不匹配故障进行诊断与修复。

**现象（S1）：** SafeHat 部署早期，Android 应用在空场景下仍持续输出多个高置信目标，大量候选分数异常高于 0.9。经 diagnose_background_scores.py 量化，p50 = 0.84–0.96，frac_gt090 = 0.41–0.58，现象可复现。

**诊断过程：** S3 跨后端对照确认同一模型在 PyTorch 与 Python 侧 NCNN 中表现正常，排除训练权重与导出链路问题；S2 中间输出检查发现设备端实际接收的输入张量形状为 640×384，而非预期的 640×640 正方形，将故障定位至设备端预处理。

**根因：** 原生预处理代码只将输入补齐到步长对齐尺寸，而非训练时使用的 640×640 正方形输入。对于 1920×1080 等相机画面，模型实际接收到的是 640×384 张量，造成输入分布偏移与背景分数系统性抬升。

**修复动作：** 将设备端补边逻辑改为始终生成 target_size × target_size 的 letterbox 输入，并统一使用 pad_value = 114。

**S5 回归验证：** 修复后重跑诊断脚本，背景分数回落至合理区间（p50 < 0.0001，frac_gt090 < 0.000001），空场景误检消失，SafeHat 检测恢复正常，既有通过场景无新引入退化。

**Figure 5.** Comparison of Input Patterns and Detection Results Before and After Preprocessing Fix

```text
修复前
相机帧 1920×1080
	→ 按长边缩放
	→ 仅补到步长对齐尺寸
	→ 输入张量 = 640×384
	→ 背景分数爆炸，空场景出现 10 类伪检出

修复后
相机帧 1920×1080
	→ 按长边缩放
	→ 使用 pad_value=114 对称补边
	→ 输入张量 = 640×640
	→ 背景分数回落到接近 0，空场景误检消失
```

*量化结果见 Table 5。*

**Table 5.** Background Score Statistics Before and After Preprocessing Fix

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 中位数分数（p50，空场景） | 0.84–0.96 | < 0.0001 |
| 第 99 百分位分数（p99） | ~1.0 | 0.0006–0.0029 |
| 分数 > 0.9 的 anchor 比例 | 0.41–0.58 | < 0.000001 |
| 空场景误检类别数 | 10 类全部检出 | 0 |
| 有目标场景 | Person + PPE 正确检出 | Person + PPE 正确检出 |

### 4.3 故障模式 II：冗余激活

本节按 S1→S3→S4→S5 路径对冗余激活故障进行诊断与修复。

**现象（S1）：** 在分割、姿态和 OBB 模式下，大量候选的置信度固定在约 0.5，表面现象类似阈值设置不当；降低置信度阈值不能改变该分布，故障持续可复现。

**诊断过程：** S3 跨后端对照确认 Python 侧 NCNN 在相同输入下置信度分布正常，排除模型权重问题；S4 结构追踪通过 conversion_detail_check.py 枚举 NCNN param 文件中的 Sigmoid 节点，确认 NCNN 图内部已含显式 Sigmoid，而 C++ 解析器存在重复调用。

**根因：** 端到端分支的类别相关输出在 NCNN 图中已经包含 Sigmoid，但 C++ 解析器仍再次执行了一次 Sigmoid，导致原本极低的背景概率被重新映射到约 0.5，形成密集伪候选。

**修复动作：** 去除解析器中的冗余激活，仅对任务特定分支保留必要变换。

**S5 回归验证：** 修复后置信度分布恢复正常，背景不再被系统性抬升，分割、姿态和 OBB 候选密度回落；既有通过任务无新引入退化。

**Figure 6.** Effect of Single vs. Repeated Sigmoid Application on Background Scores

```text
若模型图内部已经输出概率 p：

背景位置
	正常：p = 0.0006
	再做一次 sigmoid：sigmoid(0.0006) ≈ 0.5001

前景位置
	正常：p = 0.9000
	再做一次 sigmoid：sigmoid(0.9000) ≈ 0.7109

结果
	背景被抬升到约 0.5
	前景动态范围被压缩
	全图出现密集伪候选
```

*该示意说明“固定在约 50%”并非阈值问题，而是概率分支被重复激活。*

### 4.4 故障模式 III：坐标语义不匹配

本节按 S1→S2→S4→S5 路径对坐标语义不匹配故障进行诊断与修复。

**现象（S1）：** 在姿态和 OBB 模式下，边框可以显示，但关键点连线和旋转框几何形态无法被正确绘制，绘制结果超出图像边界或完全缺失。

**诊断过程：** S2 中间输出检查对原始输出张量数值进行审查，发现坐标量级达到数百至数千，显著超出 640×640 图像范围；S4 结构追踪回溯 E2E 导出路径，确认端到端分支输出已是绝对像素坐标，导出阶段内部已完成解码。

**根因：** 导出的 E2E 输出已经表示为绝对像素坐标，而设备端解析器仍将其视为网格相对值，继续执行基于 stride 的还原，导致坐标被推到图像范围之外，关键点与旋转框顶点无法正确映射。

**修复动作：** 为 E2E 输出引入独立解析分支，先判断输出是否已解码，再决定直接使用还是保留传统解码路径。

**S5 回归验证：** 修复后 Pose 关键点和 OBB 几何形态恢复正常，显示结果与任务语义一致；检测等已通过任务无新引入退化。

**Figure 7.** Semantic Comparison of Legacy Grid Decoding and E2E Pre-Decoded Coordinates

```text
传统 One-to-Many / Legacy
	网络输出：grid-relative / stride-relative
	解析器动作：乘 stride + 网格还原
	得到：像素坐标

E2E 输出
	网络输出：已经是像素坐标
	正确动作：直接使用
	错误动作：再次乘 stride / 还原
	后果：关键点和 OBB 顶点被推到图像范围之外
```

*该差异是 Pose 与 OBB“边框正常但几何形态缺失”的直接原因。*

### 4.5 故障模式 IV：OBB 布局误判

本节按 S1→S3→S4→S5 路径对 OBB 布局误判故障进行诊断与修复。

**现象（S1）：** OBB 推理结果初期出现所有框标签相同、置信度集中在约 50% 且方向估计错误的现象；单独调整阈值或类别数均无效，故障持续可复现。

**诊断过程：** 置信度固定在 50% 与故障模式 II 特征部分重叠，S3 跨后端对照首先排除模型权重问题；S4 对 NCNN param 文件执行 concat 链结构追踪（conversion_detail_check.py），枚举各输出列的来源分支，发现解析器假设 [bbox, angle, classes] 与实际输出 [bbox, classes_sigmoided, angle_raw] 的列布局存在系统性不匹配。

**根因：** 列布局假设错误使第一个类别分数被误读为角度值，后续类别索引整体偏移，并与故障模式 II 的冗余激活效应叠加，共同导致类别标签、置信度和方向均异常。

**修复动作：** 调整类别起始列、类别数量与角度列位置，并单独处理角度分支。

**S5 回归验证：** 修复后 OBB 的类别标签、置信度与方向估计恢复可解释性，旋转框显示符合预期；其他已通过任务无新引入退化。

**Table 6.** OBB Output Column Layout: Assumed vs. Actual

| 列区间 | 解析器初始假设 | 实际 NCNN 输出语义 | 错误后果 / 修复策略 |
|--------|--------------|-------------------|-----------------------|
| 0–3 | bbox | bbox | 无 / 保留原解析 |
| 4 | angle | class 0 probability | 类别误读为角度 / 将 angle 移至末列或 out1 |
| 5…4+num_class-1 | classes[0…] | classes_sigmoided[1…] | 类别索引偏移 / 按真实起始列重排 |
| 末列或 out1 | 未单独处理 / 被忽略 | angle_raw | 方向估计错误 / 单独读取角度分支 |

上述四个案例说明，五个阶段并不要求对每一类故障都机械执行完整顺序。每个阶段更像一个决策门：若某阶段已经足以定位故障，后续阶段可以按需裁剪。在本研究中，S3 的跨后端对照是最有效的前置筛选手段，因为一次 Python 侧与 Android 侧数值对比就足以区分导出端故障与设备端实现故障；S4 仅在 S3 之后仍无法用数值规律解释问题时才启用；S5 始终作为强制性的收尾阶段，以确保修复不会在此前通过的场景中引入新的退化。

本方法的适用前提包括三点：其一，导出路径遵循 PyTorch → ONNX → NCNN（经 Ultralytics export / onnx2ncnn）的标准流程；其二，设备端使用包含显式张量布局假设的原生 C++ 解析器；其三，存在可用于跨后端对照的 Python 侧参考环境。若 Python 侧参考不可用，S2 与 S4 将承担更多分析工作，需要更多依赖离线 param 文件检查与基于 logcat 的张量日志替代直接数值对比。

---

## 5. 案例研究与设备端验证

### 5.1 SafeHat 主案例

SafeHat 主案例作为贡献 3，通过端到端闭环对统一部署架构（贡献 1）与五阶段诊断流程（贡献 2）进行实例化验证。本节从主案例训练基础、五任务运行覆盖和资源受限 Android 设备基线下的运行量级三个层面报告验证结果。SafeHat 检测模型构成全文最完整的主案例，它提供了从数据、训练、导出到真机部署和故障修复的闭环证据。训练侧采用 Ultralytics 工作流完成模型加载、微调、验证与 NCNN 导出，具体为：以 yolo26n 预训练权重为初始模型，在 SafeHat 数据集上完成基础微调；随后通过 hard negative mining 自动收集无目标场景中的高置信误检图像，以空标签样本形式回注训练集，继续执行 confcal_v3 微调，以降低背景纹理、反光区域和人形设备边缘的伪激活概率。部署侧基于 Android Studio、Android Gradle Plugin、NDK、NCNN 和 OpenCV-Mobile 构建统一 Android 应用。测试设备为 HUAWEI P20 Pro，当前主要报告 CPU 路径结果，并保留基于 Vulkan 的 GPU 路径用于后续扩展对比。选取单一真机并非以最新高端 SoC 条件下的最优性能为目标，而是为资源受限 Android 设备上的统一部署提供可复现的设备端基线。

**Table 7.** Validation Environment and Main Dependencies

| 类别 | 项目 | 配置 / 版本 |
|------|------|-------------|
| 训练侧 | 训练框架 | Ultralytics YOLO（Python 3.12） |
| 训练侧 | 导出工具链 | PyTorch → ONNX → NCNN（Ultralytics export，format=ncnn；内部经 onnx2ncnn） |
| 训练侧 | 默认输入尺寸 | 640 × 640 |
| 训练侧 | SafeHat 数据集 | 10 类 PPE / Person；Roboflow 导出；含增强 |
| 部署侧 | 推理引擎 | NCNN 20260113（保留 Vulkan GPU 路径；本文报告 CPU 路径） |
| 部署侧 | 图像处理库 | OpenCV-Mobile 4.13.0（Android 移动精简版） |
| 部署侧 | Android NDK | r29（29.0.14206865） |
| 部署侧 | Android Gradle Plugin | 8.7.3 |
| 部署侧 | compileSdk / minSdk | 33 / 24（Android 7.0+） |
| 部署侧 | 推理后端 | NCNN CPU，fp32，arm64-v8a |
| 部署侧 | 输入预处理 | 检测 / 分割 / 姿态 / OBB 采用 640×640 letterbox；分类采用 224×224 resize |
| 测试设备 | 设备型号 | HUAWEI P20 Pro（CLT-AL00） |
| 测试设备 | Android 版本 | Android 10（API 29）；build CLT-AL00 10.0.0.175(C00E175R1P4) |
| 测试设备 | CPU / SoC | HiSilicon Kirin 970（adb getprop=kirin970；/proc/cpuinfo 为 4× Cortex-A73 + 4× Cortex-A53） |

从主案例训练侧证据来看，SafeHat 微调过程形成了稳定收敛曲线，验证批次预测中 PPE 与 Person 相关类别也已被正确区分。mAP50 = 0.796 和 Precision = 0.858 为主案例在进入导出与设备端验证前的训练阶段基线，此处不作进一步的精度分析。

**Figure 8.** SafeHat Fine-Tuning Training Curves and Validation Batch Prediction Examples

![Figure 8 – (a) training curves; (b) validation batch 2×4 grid](runs/paper_figures/figure8_ab.png)

*（a）SafeHat 微调过程中的损失与检测指标曲线，给出主案例训练收敛情况。（b）SafeHat 验证批次预测样例（2×4 网格，共 8 幅），给出主案例在验证集上的类别学习结果。*

### 5.2 五任务运行覆盖验证

本节验证结果回证了贡献 1 中统一加载与统一调度的声明。以下报告统一系统在同一资源约束条件下完成五任务加载、切换与运行覆盖的结果。Figure 9 给出了检测、分割、姿态、分类和 OBB 的真机抽帧组合证据，Table 8 汇总了主案例部署关键验证项，Table 9 则总结了故障模式、根因与修复结果。三者分别对应运行覆盖、关键验证项和故障修复结果。

**Figure 9.** Five-Task On-Device Frame Montage: Detection, Segmentation, Pose, Classification, and OBB

![Figure 9 – five-task on-device frame montage](runs/paper_figures/figure8c_five_task_montage.png)

*检测、分割、姿态、分类与 OBB 五任务真机视频抄帧组合证据图。统一编号条 01–05 依次对应 Detection、Segmentation、Pose、Classification 与 OBB。该图用于证明统一应用内五类任务均已完成真机加载与运行覆盖。其中 (e) OBB 子图取自公路场景实测，显示 DOTA 预训练模型在应用内正确渲染旋转框（roundabout 73.9%），印证 OBB 路径引擎与方向解析均正常；上游工地场景阈值内无激活，决于 DOTA 模型的训练域与该场景的目标分布差异，不影响 OBB 任务切换与接口完整性验证（详见 Table 8 与 Table 9）。*

**Table 8.** Key Validation Items for SafeHat Deployment

| 序号 | 验证项 | 预期标准 | 验证结果 |
|------|--------|----------|----------|
| 1 | SafeHat 检测模型加载（首选 yolo26n_safehat，兼容 yolo26n_e2e） | param/bin 正常解析，logcat 无报错 | ✓ |
| 2 | 输出类别数与目标类别一致 | = 10（SafeHat PPE 类别） | ✓ |
| 3 | 预处理修复后空场景误检消除 | 置信度 p50 < 0.0001；0 误检 | ✓ |
| 4 | 有目标场景正确检出 PPE / Person | UI 显示置信度 $\tilde p \in 0.35–0.73$（同帧原始 $p \le 0.95$） | ✓ |
| 5 | 设备端阈值调节功能 | setDetectThresholds JNI 调用后过滤生效 | ✓ |
| 6 | 多任务切换（5 类任务） | 切换后各任务独立运行，渲染输出正常 | ✓ |
| 7 | Pose E2E 坐标修复后关键点渲染 | 关键点与骨骼线正确叠加于视频帧 | ✓ |
| 8 | OBB 布局修复后方向与类别正确 | 类别标签、置信度、旋转框方向均符合预期 | ✓ |

**Table 9.** Fault Summary: Symptoms, Root Causes, and Fix Outcomes

| 序号 | 故障类型 | 可观察现象 | 根因 | 修复策略 | 恢复结果 |
|------|----------|-----------|------|----------|----------|
| 1 | 预处理不匹配 | 空场景 10 类全检出；p50 = 0.84–0.96 | 步长对齐补边生成 640×384 而非 640×640 | 补边至 target_size × target_size 正方形 | p50 < 0.0001；零误检 |
| 2 | 冗余激活 | 所有置信度固定在约 50.0%；出现密集候选 | 模型图已含 Sigmoid；C++ 再次执行 | 去除冗余 sigmoid()；仅保留必要变换 | 模型原始概率 $p$ 分布恢复（实测 $p_{50}\approx 0.34$、$p_{95}\approx 0.74$、上限约 0.95），经 §3.3 中 $\tilde p = [\sigma(\operatorname{logit}(p)/6)]^{1.6}$ 压缩后 UI 显示值 $\tilde p \in 0.35–0.73$ |
| 3 | E2E 坐标语义误解 | Pose 无骨骼线；OBB 几何形态缺失 | E2E 输出为绝对像素坐标；解析器再次做 grid×stride 解码 | 引入已解码输出解析分支并自动检测 | 关键点和旋转框正确渲染 |
| 4 | OBB 布局误判 | 所有框标签和方向异常 | 假设 [bbox, angle, classes]；实际为 [bbox, classes_sigmoided, angle_raw] | 通过 NCNN concat 链追踪修正列索引 | 类别、置信度、方向均正确 |

从系统角度看，关键结果不是单一精度指标，而是四类故障在修复后都出现了可观测变化。预处理修正后，空场景 p50 从 0.84–0.96 降至 < 0.0001；去除冗余激活后，端到端分支的模型原始概率 $p$ 分布从固定约 0.5 恢复为跨场景变化的多峰分布（2026-05-07 SafeHat 真机抽帧 1009 样本：$p_{50}\approx 0.34$、$p_{95}\approx 0.74$、上限约 0.95），经 §3.3 概率校准公式压缩后设备端 UI 显示值 $\tilde p$ 分布为 0.35–0.73（例如同帧 $p \approx 0.95$ 对应 $\tilde p \approx 0.43$，与 §3.3 示例一致）；修正坐标语义和 OBB 布局解析后，关键点、旋转框和类别标签重新与任务语义一致。这些变化说明本文提出的诊断方法不仅能够解释故障来源，也能给出可复核的修复结果。

需要说明的是，§3.3 提出的概率校准、per-class NMS、互斥类抑制、对象依赖约束与 PPE 框锁定共五类后处理机制，是 SafeHat 主案例运行时的常驻组件，其运行覆盖证据由 Figure 9（五任务真机抽帧）与 Table 8 行 4（有目标场景正确检出 PPE / Person、UI 显示置信度 $\tilde p \in 0.35–0.73$）共同承载。本文以"统一部署架构 + 一致性诊断流程"为方法论核心，不再对上述每一项后处理机制单独执行消融实验；针对单类别阈值与权重参数（$T = 6$、$\gamma = 1.6$、$K_c = 50$、关联重叠 0.15 / 0.20、PPE 框锁定 0.35 : 0.65）的系统性消融留待后续工作。

### 5.3 设备端延迟表征

在运行效率方面，Table 10 报告五类任务在同一真机、同一路径和同一资源受限 Android 设备基线下的运行量级与波动范围。分类任务输入为 224×224 resize，检测、分割、姿态和 OBB 任务输入为 640×640 letterbox，两类数值不具备直接可比性。

**Table 10.** On-Device Inference Latency by Task (Pure Inference, No Rendering, CPU Backend)

| 任务 | 模型 | 后端 | 样本数 | 均值 (ms) | P5 (ms) | P95 (ms) | FPS | 备注 |
|------|------|------|--------|-----------|---------|----------|-----|------|
| 检测 | yolo26n_safehat（兼容别名 yolo26n_e2e） | CPU | 148 | 321.3 | 272.2 | 412.2 | 3.1 | 640×640 letterbox |
| 分割 | yolo26n_seg_e2e | CPU | 108 | 430.5 | 359.3 | 517.5 | 2.3 | 640×640 letterbox |
| 姿态 | yolo26n_pose_e2e | CPU | 113 | 369.4 | 342.0 | 427.6 | 2.7 | 640×640 letterbox |
| 分类 | yolo26n_cls | CPU | 928 | 33.6 | 25.9 | 41.8 | 29.8 | 224×224 resize |
| OBB | yolo26n_obb_e2e | CPU | 120 | 336.1 | 308.4 | 386.3 | 3.0 | 640×640 letterbox ^a |

^a OBB 行来自独立采集日志，均值与 FPS 取 120 帧 SUMMARY，P5 / P95 由转录中可见 FRAME 子样本估计。检测、分割、姿态和分类四项由 collect_latency.py 对 YOLO26BENCH FRAME 日志统一统计，跳过前 10 帧预热。所有数值均为 2026-05-07 真机 CPU 路径测量。

Table 10 显示，统一部署框架已经在单设备 CPU 路径上完成五类任务推理：分类任务均值为 33.6 ms，检测、分割、姿态和 OBB 的均值为 321.3–430.5 ms。Table 10 给出资源受限 Android 设备条件下的运行量级刻画，为后续 GPU、量化和跨设备评测提供基线；该结果不应被外推至所有 Android 平台。
Kirin 970 于 2017 年发布，采用通用 ARM 大小核架构（4× Cortex-A73 + 4× Cortex-A53），NCNN 在该设备上沿 CPU 路径运行时不启用专用神经网络加速器。如 Table 10 所示，640×640 输入下检测、分割、姿态与 OBB 四类任务的 CPU 推理均值为 321–430 ms，对应吞吐量约 2–3 FPS，远低于实时帧率阈值，实证确认该设备满足本文所界定的资源受限条件。分类任务的 29.8 FPS 来自 224×224 输入，因输入规模与其他四类任务差异显著，不适宜作为资源充裕与否的独立判据。
### 5.4 适用范围与局限性讨论

尽管本文系统已经在统一应用内完成五任务运行覆盖，并通过主案例展示了从训练到修复再到回归验证的完整路径，但当前工作仍存在清晰边界。首先，最完整的定量验证主要集中于 SafeHat 检测案例，其余四类任务更多体现为统一部署链路下的运行覆盖和解析正确性验证，而非与主案例同等强度的任务级 benchmark。其次，本文当前主要报告单一 Android 设备与 CPU 路径结果，尚未形成跨设备、跨芯片与跨后端的标准化评测矩阵。再次，本文未引入与 TensorFlow Lite、MNN 或 ONNX Runtime Mobile 的实测对比，因此不应将当前结果外推为对所有移动推理框架的普适性能判断。

在上述边界下，本文仍报告了三类可复现结果：Figure 9 给出五任务真机运行覆盖，Table 8 列出 SafeHat 部署关键验证项，Table 9 记录四类故障的根因与修复结果。这些结果覆盖了运行、验证和修复三个环节。对于依赖低成本、广泛可得的 Android 终端的场景，这意味着边缘智能应用可以在有限算力和弱网络条件下完成实际部署。

---

## 6. 结论

本文提出面向资源受限 Android 设备的多任务视觉推理统一部署架构，解决五类任务模型在 E2E、One-to-Many 和 Legacy 多输出路径并存条件下出现的资产命名冲突、接口语义分歧与任务调度耦合问题，在 Java–JNI–C++–NCNN 协同链路下实现统一加载、统一调度与一致性接口管理，并完成五任务真机加载、切换和运行覆盖。该架构以低成本、广泛可得的 Android 设备为主要承载对象，通过统一链路覆盖从模型导出到设备端验证的全流程。

本文提出的五阶段一致性诊断流程通过异常现象记录、中间输出检查、跨后端对照、结构追踪与回归验证，为预处理不匹配、冗余激活、坐标语义误解和 OBB 布局误判四类典型故障提供了结构化定位与修复路径；其中 S3 跨后端对照作为最有效的前置筛选手段，一次数值对比即可区分导出端故障与设备端实现故障。

SafeHat 主案例给出了从硬负例挖掘、模型微调到设备端诊断与回归的端到端闭环验证，对上述部署架构与诊断流程进行了完整的实例化验证：预处理修正后空场景误检归零，去除冗余激活后模型原始概率 $p$ 恢复为跨场景变化的多峰分布（上限约 0.95）、设备端 UI 显示置信度 $\tilde p$ 分布稳定在 0.35–0.73，修正坐标语义和 OBB 布局后关键点与旋转框恢复正常渲染。

此外，在 Kirin 970 的 CPU 路径下，检测、分割、姿态与 OBB 四类 640×640 任务的推理均值为 321.3–430.5 ms，分类任务在 224×224 输入下的均值为 33.6 ms，为资源受限 Android 设备上的运行量级刻画提供了可复现基线。

本文仍有明确的工程边界。当前最完整的主案例仍是 SafeHat 检测任务，设备端结果主要来自单一 Android 设备和 CPU 路径，且尚未开展跨框架实测比较。因此，本文验证的是所提出的统一部署架构与五阶段一致性诊断流程在单设备 CPU 路径下的可行性，而非对多模型、多设备或多推理框架性能最优性进行全面评估。

未来工作可沿面向资源受限 Android 设备的三条扩展路线展开。第一，在统一代码路径下引入 INT8 量化、结构压缩和其他面向部署的轻量化策略，以进一步降低中低端 SoC 条件下的时延、内存占用与能耗开销。第二，在不同 Android 芯片平台、不同价格带终端、不同档位 SoC 以及 CPU/GPU 路径上开展分层跨设备评测，以建立更具代表性的资源受限 Android 设备基线，并验证系统在更广泛资源约束条件下的稳定性与可移植性。第三，继续完善自动化验证链路，把更多导出检查、日志对照和回归测试纳入持续化部署流程，并增强弱网络或离线条件下的资产替换与部署复核能力，提升多任务移动视觉系统在低资源场景中的可维护性与可复现性。

---

## 数据可用性声明

本文实验所用部署框架代码已公开发布于 [https://github.com/majun2019/ncnn-android-yolo26lt](https://github.com/majun2019/ncnn-android-yolo26lt)（Zenodo DOI: [https://doi.org/10.5281/zenodo.20137180](https://doi.org/10.5281/zenodo.20137180)）。训练数据集 SafeHat 基于 Roboflow 公开数据集构建，原始来源见数据集说明文件。

---

## 参考文献

[1] Z. Ge, S. Liu, F. Wang, Z. Li, and J. Sun, "YOLOX: Exceeding YOLO series in 2021," arXiv preprint arXiv:2107.08430, 2021.

[2] M. Tan, R. Pang, and Q. V. Le, "EfficientDet: Scalable and efficient object detection," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2020, pp. 10781-10790.

[3] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770-778.

[4] A. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Q. V. Le, and H. Adam, "Searching for MobileNetV3," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2019, pp. 1314-1324.

[5] X. Zhang, X. Zhou, M. Lin, and J. Sun, "ShuffleNet: An extremely efficient convolutional neural network for mobile devices," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 6848-6856.

[6] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 4700-4708.

[7] A. Contributors, "TensorFlow Lite guide," 2023. [Online]. Available: https://www.tensorflow.org/lite/guide

[8] Alibaba, "MNN: A blazing fast, lightweight deep learning framework," 2023. [Online]. Available: https://github.com/alibaba/MNN

[9] Microsoft, "ONNX Runtime," 2023. [Online]. Available: https://onnxruntime.ai/

[10] nihui, "ncnn: A high-performance neural network inference framework optimized for the mobile platform," 2023. [Online]. Available: https://github.com/Tencent/ncnn

[11] C. Liu, L. Zhong, J. Wang, J. Huang, Y. Wang, M. Guan, X. Li, H. Zheng, X. Hu, X. Ma, and S. Tan, "Grain-YOLO: An improved lightweight YOLO v8 and its android deployment for rice grains detection," *Computers and Electronics in Agriculture*, vol. 237, Art. no. 110757, 2025, doi: 10.1016/j.compag.2025.110757.

[12] S. S. More, N. Patil, V. B. Lobo, N. Shet, D. Goswami, P. Rane, and P. N. Kumar, "Empowering the visually impaired: YOLOv8-based object detection in Android applications," *Procedia Computer Science*, vol. 252, pp. 457-469, 2025, doi: 10.1016/j.procs.2025.01.005.

[13] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: Unified, real-time object detection," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 779-788.

[14] C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2023, pp. 7464-7475.

[15] G. Jocher, J. Qiu, and A. Chaurasia, "Ultralytics YOLO," version 8.0, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

[16] G. Bradski, *Learning OpenCV: Computer Vision with the OpenCV Library*. Sebastopol, CA, USA: O'Reilly Media, 2008.

[17] Google, "Android NDK documentation," 2023. [Online]. Available: https://developer.android.com/ndk

[18] Google, "CameraX overview," 2023. [Online]. Available: https://developer.android.com/training/camerax

[19] Khronos Group, "Vulkan specification," 2023. [Online]. Available: https://www.khronos.org/vulkan/

[20] nihui, "ncnn wiki: how-to-use-and-faq," 2023. [Online]. Available: https://github.com/Tencent/ncnn/wiki

[21] PNNX Contributors, "PNNX: PyTorch neural network exchange," 2023. [Online]. Available: https://github.com/pnnx/pnnx

[22] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, "End-to-end object detection with transformers," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2020, pp. 213-229.

[23] Y. Zhao, W. Lv, S. Xu, J. Wei, G. Wang, Q. Dang, Y. Liu, and J. Chen, "DETRs beat YOLOs on real-time object detection," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2024, pp. 16965-16974.

[24] W. Lv, S. Xu, Y. Zhao, G. Wang, J. Wei, C. Cui, Y. Du, Q. Dang, and Y. Liu, "DETRs beat YOLOs on real-time object detection," arXiv preprint arXiv:2304.08069, 2023.

[25] Y. Li, S. Yang, P. Liu, S. Zhang, Y. Wang, Z. Wang, W. Yang, and S.-T. Xia, "SimCC: A simple coordinate classification perspective for human pose estimation," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2022, pp. 89-106.

[26] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin Transformer: Hierarchical vision transformer using shifted windows," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2021, pp. 10012-10022.

[27] C.-Y. Wang, H.-Y. M. Liao, Y.-H. Wu, P.-Y. Chen, J.-W. Hsieh, and I.-H. Yeh, "CSPNet: A new backbone that can enhance learning capability of CNN," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, 2020, pp. 390-391.