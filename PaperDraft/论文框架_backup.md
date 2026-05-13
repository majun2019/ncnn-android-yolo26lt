**Title**  
Deployment and Diagnostic Framework for YOLO26-Based Multi-Task Visual Inference on Android Devices

**中文题目**  
面向 Android 设备的基于 YOLO26 的多任务视觉推理部署与诊断框架

---

## Abstract / 摘要

This study develops an NCNN-based Android deployment framework for YOLO26-based multi-task visual inference, supporting object detection, instance segmentation, pose estimation, classification, and oriented bounding box (OBB) detection within a unified application. A SafeHat detection case study is used to build a complete workflow from fine-tuning on a 10-class PPE dataset (mAP50 = 0.796, Precision = 0.858) to NCNN export and real-device validation. Several critical deployment inconsistencies are identified and systematically resolved, including preprocessing mismatch, redundant activation, coordinate re-decoding, and output-layout misinterpretation. A reusable diagnostic methodology is summarized covering abnormal-behavior recording, intermediate-output inspection, cross-backend comparison, structural tracing, and device-side regression. Experimental results confirm that the framework stably supports five visual tasks on Android and that the diagnostic workflow effectively restores deployment consistency.

本文构建了一套基于 NCNN 的 Android 部署框架，用于支持 YOLO26 的多任务视觉推理，在统一应用中实现目标检测、实例分割、姿态估计、图像分类和旋转框检测。本文以 SafeHat 检测为重点案例，在 10 类 PPE 数据集上完成微调（mAP50 = 0.796，Precision = 0.858），并建立从导出到真机验证的完整闭环。部署中发现并系统修复了多类关键不一致性问题，包括预处理尺寸不匹配、重复激活、坐标重复解码和输出布局误判。本文总结出一套可复用的诊断方法，覆盖异常记录、中间输出检查、跨后端对照、结构追踪和设备端回归验证。实验表明，该框架可稳定支持五类视觉任务在 Android 上运行，诊断流程能有效恢复部署一致性。

**Keywords:** YOLO26; NCNN; Android; edge inference; deployment consistency; multi-task visual inference; model conversion

**关键词：** YOLO26；NCNN；Android；边缘推理；部署一致性；多任务视觉推理；模型转换

---

## 1. Introduction / 1. 引言

Deep visual inference is increasingly shifting from cloud servers to edge devices due to the growing demand for low latency, privacy preservation, and autonomous operation in real-world environments. In industrial safety inspection, mobile robotics, smart terminals, and field monitoring scenarios, on-device inference allows image acquisition, model execution, and result feedback to be completed locally, thereby reducing communication overhead and improving responsiveness. This trend has motivated extensive research on lightweight models, mobile inference engines, and edge-oriented deployment strategies [1–6].

由于低时延、隐私保护和自主运行需求的持续增长，深度视觉推理正逐步从云端服务器迁移至边缘设备。在工业安全巡检、移动机器人、智能终端和现场监测等场景中，端侧推理能够在本地完成图像采集、模型执行和结果反馈，从而降低通信开销并提升响应效率。这一趋势推动了轻量化模型、移动端推理引擎和边缘部署策略等方向的大量研究 [1–6]。

Among real-time vision models, the YOLO family has remained highly influential because of its favorable trade-off between accuracy and inference speed. Recent developments in YOLO-like architectures have not only improved the backbone and feature aggregation design, but have also simplified the inference pathway by reducing the dependence on post-processing-heavy dense candidate paradigms. Such evolution is especially valuable for mobile deployment, where implementation simplicity and runtime consistency are often as important as raw detection accuracy.

在实时视觉模型中，YOLO 系列因其在精度和速度之间具备较好的平衡而长期占据重要地位。近年来，YOLO 类架构的发展不仅体现在骨干网络和特征融合设计的持续优化，也体现在通过减少对重后处理密集候选范式的依赖来简化推理路径。这种演进对移动端部署尤为重要，因为在移动设备上，实现简洁性与运行一致性往往与检测精度同样关键。

For Android deployment, model availability alone is insufficient. A model that works correctly in a desktop Python environment may still fail on a mobile device due to preprocessing mismatch, tensor-layout misunderstanding, conversion-side semantic changes, or toolchain-related compatibility issues. These failures are often not reflected by conventional benchmark metrics, yet they directly determine whether a model can be used in practice. Therefore, deployment consistency and debugging methodology should be treated as first-class research topics rather than purely auxiliary engineering concerns.

对于 Android 部署而言，仅仅“模型可用”并不意味着系统可落地。一个在桌面端 Python 环境中运行正常的模型，仍可能因为预处理不一致、张量布局误判、转换链语义变化或工具链兼容性问题而在移动设备上失效。这类故障往往无法通过常规 benchmark 指标直接反映，但它们却直接决定模型是否能够在实际场景中投入使用。因此，部署一致性与调试方法应被视为正式研究问题，而不只是附属工程细节。

To address this gap, this work develops a unified Android deployment framework for YOLO26-based multi-task visual inference using Java, JNI, C++, and NCNN. The framework supports five task types, namely object detection, instance segmentation, pose estimation, image classification, and oriented bounding box detection, within the same application pipeline. More importantly, a SafeHat detection case study is used to build a complete workflow from model fine-tuning and export to real-device validation and failure analysis.

为弥补上述研究空白，本文构建了一套基于 Java、JNI、C++ 与 NCNN 的统一 Android 部署框架，用于支持基于 YOLO26 的多任务视觉推理。该框架在同一应用流程中承载了目标检测、实例分割、姿态估计、图像分类和旋转框检测五类任务。更重要的是，本文以 SafeHat 安全帽检测为重点案例，建立了从模型微调、导出到真机验证和故障分析的完整研究链路。

The main contributions of this study are summarized as follows:

1. A unified Android deployment framework is developed for YOLO26-based multi-task visual inference, supporting five heterogeneous visual tasks on a single Android device.
2. A complete deployment workflow is established through the SafeHat case, covering the full path from fine-tuning on a 10-class PPE dataset to on-device regression with verified mAP50 of 0.796.
3. A reusable diagnostic methodology is summarized, resolving four categories of deployment inconsistencies (preprocessing, activation, coordinate semantics, and output layout) that rendered the system unusable before correction.

本文的主要贡献概括如下：

1. 构建了一套面向 YOLO26 多任务视觉推理的统一 Android 部署框架，在单一 Android 设备上支持五类异构视觉任务。
2. 通过 SafeHat 案例建立了完整部署闭环，覆盖从 10 类 PPE 数据集微调到设备端回归验证的全路径，验证集 mAP50 达 0.796。
3. 总结出一套可复用的诊断方法，修复了四类部署不一致性问题（预处理、激活函数、坐标语义、输出布局），修复前系统完全不可用。

The remainder of this paper is organized as follows. Section 2 reviews the relevant technical background and related work. Section 3 introduces the proposed Android deployment framework and the SafeHat case study. Section 4 presents the failure analysis and diagnostic methodology. Section 5 reports experimental validation and result analysis. Section 6 concludes the paper and discusses future work.

本文其余部分组织如下。第 2 节介绍相关技术背景与已有研究。第 3 节给出所提出的 Android 部署框架及 SafeHat 案例。第 4 节重点分析部署故障及其诊断方法。第 5 节展示实验验证与结果分析。第 6 节总结全文并讨论未来工作。

[Table 1 here: Overall paper structure and section objectives / 表 1 插入此处：全文结构与各节目标]

---

## 2. Related Work and Technical Background / 2. 相关工作与技术背景

The YOLO family has evolved from early single-stage real-time detectors to more deployment-friendly architectures that better balance efficiency and output simplicity [1–3]. Meanwhile, end-to-end detection paradigms represented by transformer-based detectors and real-time variants have demonstrated that reducing post-processing complexity can improve inference clarity and system integration [4–6]. In the context of this study, the deployment value of YOLO26 lies not only in its real-time capability, but also in its more direct detection output format, which is particularly helpful for mobile-side implementation.

YOLO 系列已经从早期的单阶段实时检测器逐步演进为更利于部署的架构，在效率和输出简洁性之间取得了更好的平衡 [1–3]。与此同时，以基于 Transformer 的检测器及其实时变体为代表的端到端检测范式表明，减少后处理复杂度有助于提升推理链路的清晰性和系统集成便利性 [4–6]。在本文场景中，YOLO26 的部署价值不仅体现在实时性上，也体现在更直接的检测输出形式上，这对于移动端实现尤其重要。

However, the present work does not assume that all five supported tasks share the same output semantics. While the detection branch benefits from an end-to-end style output with simplified post-processing, segmentation, pose, classification, and OBB models still carry task-specific output structures such as mask prototypes, keypoint coordinates, class vectors, and rotation parameters. Therefore, this study treats multi-task support as a demonstration of framework generality, while keeping the detailed deployment analysis focused on the detection-centered workflow and its extensions.

然而，本文并不假设所支持的五类任务具有完全相同的输出语义。虽然检测分支受益于端到端风格输出以及简化的后处理流程，但分割、姿态、分类和 OBB 模型仍包含各自特有的输出结构，例如掩码原型、关键点坐标、类别向量和旋转参数。因此，本文将多任务支持视为框架通用性的体现，而将详细部署分析聚焦于以检测为核心并向其他任务扩展的工作流。

Several mobile inference frameworks have been developed for edge deployment, including TensorFlow Lite, MNN, ONNX Runtime Mobile, and NCNN [8–11]. These frameworks differ in operator coverage, platform integration, model conversion pathways, runtime customization, and hardware acceleration support. For Android-oriented native deployment, NCNN is particularly attractive because of its lightweight C++ implementation, mature ARM adaptation, and practical Vulkan support. Such characteristics make it well suited to scenarios that require close interaction with native camera access, OpenCV-based image processing, and JNI-based application integration.

当前已有多种面向边缘部署的移动端推理框架，包括 TensorFlow Lite、MNN、ONNX Runtime Mobile 和 NCNN [8–11]。这些框架在算子覆盖范围、平台集成方式、模型转换路径、运行时可定制性以及硬件加速支持等方面各有差异。对于面向 Android 的原生部署，NCNN 具有轻量级 C++ 实现、成熟 ARM 适配和实用 Vulkan 支持等优势，因此特别适合需要与原生相机访问、基于 OpenCV 的图像处理以及 JNI 应用集成深度协同的场景。

Android deployment is inherently a cross-layer engineering problem rather than a simple model-format migration task. A practical visual inference application usually involves Java-level interaction logic, JNI bridge code, native inference modules, image-processing utilities, camera access interfaces, and optional GPU backends [12–17]. As a result, consistency problems may emerge from any layer in the deployment chain, making it necessary to study system design together with model behavior.

Android 部署本质上是跨层协同的工程问题，而非简单的模型格式迁移任务。一个可用的视觉推理应用通常涉及 Java 层交互逻辑、JNI 桥接代码、原生推理模块、图像处理工具、相机接入接口以及可选的 GPU 后端 [12–17]。因此，部署链条中的任一层都可能引发一致性问题，这使得系统设计与模型行为必须被一并研究。

Despite growing interest in mobile deployment, most existing studies focus on model compression, knowledge distillation, or accuracy–speed trade-offs [25–27], while relatively few works systematically examine the deployment consistency issues that arise during the conversion and integration process on real Android devices. This gap motivates the present study.

尽管移动端部署的研究关注度持续提升，但现有工作多集中于模型压缩、知识蒸馏或精度–速度权衡 [25–27]，而较少系统地审视在真实 Android 设备上模型转换与集成过程中出现的部署一致性问题。这一空白构成了本文的研究动机。

从工程部署视角看，模型导出也是引发不一致的重要来源。即使使用相同的预训练或微调模型，不同的导出路径、张量布局、内置激活、坐标表示方式以及预处理假设，仍可能导致桌面端推理与设备端执行之间出现显著偏差。因此，部署不应被视为一次性的格式转换，而应被视为由导出、验证、比较、修正和回归构成的闭环过程。

[Figure 1 here: Model export and deployment verification workflow / 图 1 插入此处：模型导出与部署验证流程]

---

## 3. Proposed Android Deployment Framework / 3. 所提出的 Android 部署框架

The proposed system adopts a layered architecture composed of a Java interaction layer, a JNI bridge layer, a native C++ inference layer, and a low-level dependency layer. The design objective is not limited to running a model on a phone, but to support unified multi-task inference while preserving enough observability and controllability for failure diagnosis. In this architecture, the Java layer manages task selection, camera interaction, threshold adjustment, and UI rendering, while JNI serves as the gateway for model loading, parameter passing, and native object lifecycle management.

所提出的系统采用由 Java 交互层、JNI 桥接层、原生 C++ 推理层和底层依赖层组成的分层架构。该设计目标并不仅仅是让模型在手机上运行，而是在支持统一多任务推理的同时，为故障诊断保留足够的可观测性与可控性。在该架构中，Java 层负责任务选择、相机交互、阈值调节和界面渲染，JNI 则承担模型加载、参数传递和原生对象生命周期管理的网关角色。

The native C++ layer encapsulates the core inference logic for detection, segmentation, pose, classification, and OBB tasks. Shared components are used for model initialization, threshold configuration, image preprocessing, and output drawing, whereas task-specific parsers are retained for different output semantics. This design allows the system to maintain a unified runtime path without erasing the structural differences among the supported tasks.

原生 C++ 层封装了检测、分割、姿态、分类和 OBB 任务的核心推理逻辑。系统在模型初始化、阈值配置、图像预处理和结果绘制等方面采用共享组件，而在输出解析阶段保留各任务的专门实现。该设计使系统能够在统一运行路径下保持框架一致性，同时又不抹平各类任务在输出结构上的差异。

The low-level dependency layer mainly consists of NCNN for neural network execution, OpenCV-Mobile for image conversion and visualization, Android NDK camera interfaces for image acquisition, and Vulkan or Turnip-related backends for optional GPU-side acceleration. This stack is chosen because it provides a relatively transparent engineering environment, enabling the insertion of logs, tensor inspection, and component-level troubleshooting when abnormal behavior appears on mobile devices.

底层依赖层主要包括用于神经网络执行的 NCNN、用于图像转换与可视化的 OpenCV-Mobile、用于图像采集的 Android NDK Camera 接口，以及作为可选 GPU 加速后端的 Vulkan 或 Turnip 相关组件。之所以选择这一技术栈，是因为它能提供相对透明的工程环境，便于在移动设备出现异常行为时插入日志、检查张量并开展模块级排查。

[Figure 2 here: Overall architecture of the Android multi-task visual inference system / 图 2 插入此处：Android 多任务视觉推理系统总体架构]

To avoid fragmented task-specific applications, the proposed framework adopts a unified scheduling mechanism. A task identifier is used to select the corresponding inference module, while the camera input pathway, preprocessing pipeline, and display entry remain largely shared. As a result, the same application can switch among five task types under consistent device settings, which is useful not only for code reuse but also for comparative diagnosis across tasks.

为避免形成彼此割裂的任务专属应用，本文框架采用统一调度机制。系统通过任务标识符选择对应推理模块，而相机输入路径、预处理流程和显示入口则尽量保持共享。因此，同一应用可以在一致的设备条件下切换五类任务，这不仅提升了代码复用性，也为不同任务之间的对照诊断提供了便利。

Although the runtime path is unified, the output semantics remain task-dependent. Detection outputs are directly parsed into target location, class, and confidence entries. Segmentation requires additional mask reconstruction from prototypes and coefficients. Pose estimation further parses keypoint coordinates and visibility cues. Classification focuses on top-k ranking of class probabilities. OBB prediction additionally reconstructs geometry from box parameters and orientation values. Therefore, the framework emphasizes unified orchestration rather than semantic homogenization.

尽管运行路径是统一的，各任务的输出语义仍然依赖于具体任务类型。检测任务直接解析目标位置、类别和置信度。分割任务还需要根据原型和系数重建掩码。姿态估计则需进一步解析关键点坐标和可见性信息。分类任务主要关注类别概率的 Top-K 排序。OBB 预测还需要基于边框参数与朝向值恢复几何形态。因此，本文框架强调的是统一调度，而不是语义同构化。

[Table 2 here: Unified scheduling and task-specific output parsing across five tasks / 表 2 插入此处：五类任务的统一调度与差异化解析]

The deployment workflow is organized into five phases: model preparation, NCNN export, structural verification, asset deployment, and real-device regression. During model preparation, either public pretrained weights or fine-tuned custom-task weights are used. The export stage converts task-specific models into NCNN param-bin pairs. Structural verification then checks output dimensions, branch organization, and file integrity. After that, the assets are deployed to the Android project, followed by device-side regression to ensure that the loaded model and the parser remain consistent.

本文将部署流程组织为五个阶段，即模型准备、NCNN 导出、结构验证、资产部署和真机回归。在模型准备阶段，系统可以使用公开预训练权重，也可以使用微调后的自定义任务权重。导出阶段将任务相关模型转换为 NCNN 的 param-bin 文件对。随后通过结构验证检查输出维度、分支组织和文件完整性。最后将模型资产部署至 Android 工程，并通过真机回归确保加载模型与解析逻辑保持一致。

[Table 3 here: Deployment stages, tools, inputs, and verification targets / 表 3 插入此处：部署各阶段工具、输入与验证目标]

Among the five supported tasks, the SafeHat detection scenario forms the most complete case study. The dataset covers ten classes related to helmets and personal protective equipment in industrial environments, and the workflow includes fine-tuning, weight selection, export, asset replacement, and on-device validation. This case serves as the central experimental thread of the paper because it connects model development, deployment, troubleshooting, and performance observation into a single reproducible loop.

在五类任务中，SafeHat 检测场景构成了最完整的案例研究。该数据集覆盖工业场景下与安全帽和个人防护装备相关的十个类别，整个流程包括微调训练、权重筛选、模型导出、资产替换和设备端验证。之所以将该案例作为全文主线，是因为它将模型开发、部署、排障和效果观察连接成了一个可复现的完整闭环。

[Table 4 here: SafeHat dataset classes and application semantics / 表 4 插入此处：SafeHat 数据集类别与应用语义]  
[Figure 3 here: End-to-end SafeHat workflow from training to Android deployment / 图 3 插入此处：SafeHat 从训练到 Android 部署的闭环流程]

---

## 4. Deployment Failure Analysis and Diagnostic Methodology / 4. 部署故障分析与诊断方法

The most distinctive contribution of this work lies in the analysis of deployment failures that emerged during real-device execution. In contrast to offline accuracy evaluation, mobile deployment failures often appear as abnormal confidence patterns, empty outputs, incorrect visualization, or inconsistent engineering behavior rather than direct crashes. Such issues may span model export, preprocessing, tensor parsing, and platform-specific implementation, making ad hoc fixes ineffective. To address this challenge, this study summarizes a five-stage diagnostic workflow: abnormal behavior recording, intermediate-output inspection, cross-backend comparison, structural tracing, and regression verification.

本文最具辨识度的贡献，在于对真机运行过程中暴露出的部署故障进行了系统分析。不同于离线精度评估，移动端部署故障往往表现为异常置信度分布、空输出、错误可视化或工程行为不一致，而不是直接崩溃。这类问题通常横跨模型导出、预处理、张量解析和平台实现等多个层次，使得零散修补往往无效。为应对这一挑战，本文总结出一套五阶段诊断流程，包括异常现象记录、中间输出检查、跨后端对照、结构追踪和回归验证。

[Figure 4 here: Deployment failure diagnosis workflow / 图 4 插入此处：部署故障诊断流程]

The first major failure was severe false positives caused by preprocessing inconsistency. In the early SafeHat deployment stage, the Android application produced multiple high-confidence detections even in empty scenes. Statistical inspection further showed that an abnormally large proportion of candidate scores exceeded 0.9. Since the same model behaved normally in both PyTorch and Python-side NCNN inference, the problem was unlikely to originate from model training or weight corruption.

第一个主要故障是由预处理不一致引发的严重高置信误检。在 SafeHat 部署早期阶段，Android 应用在空场景下仍持续输出多个高置信目标。进一步统计发现，大量候选分数异常地高于 0.9。由于相同模型在 PyTorch 和 Python 侧 NCNN 推理中均表现正常，因此问题不太可能来自模型训练本身或权重损坏。

Further investigation revealed that the native preprocessing code only padded the input to a stride-aligned shape instead of the square 640 × 640 input used during training. For camera frames such as 1920 × 1080, the actual input tensor became 640 × 384 rather than the expected square form. This mismatch distorted the feature distribution seen by the classifier head and triggered score explosion in background regions. After changing the device-side padding logic to always produce a target_size × target_size input, the abnormal false positives disappeared and the score distribution returned to a reasonable range.

进一步排查表明，原生预处理代码只是将输入补齐到步长对齐尺寸，而非训练时所使用的 640 × 640 正方形输入。对于 1920 × 1080 等相机画面，模型实际接收到的是 640 × 384 张量，而不是预期的正方形输入。这种偏差破坏了分类头所依赖的特征分布，进而导致背景区域的分数爆炸。将设备侧补边逻辑改为始终生成 target_size × target_size 输入后，异常误检现象消失，分数分布也恢复到合理区间。

[Figure 5 here: Comparison of input form and detection results before and after preprocessing correction / 图 5 插入此处：预处理修复前后输入形态与检测结果对比]  
[Table 5 here: Background score statistics before and after preprocessing correction / 表 5 插入此处：预处理修复前后背景分数统计]

The second failure involved duplicated Sigmoid operations in end-to-end branches. In segmentation, pose, and OBB deployment, many outputs displayed nearly fixed confidence values around 0.5, which initially looked like a thresholding problem. However, the traced NCNN graph showed that class-related branches had already included Sigmoid operations, while the C++ parser still applied another Sigmoid on top of them. This caused very small background probabilities to be remapped toward 0.5, leading to dense false candidates across the image.

第二个故障与端到端分支中的重复 Sigmoid 有关。在分割、姿态和 OBB 的部署中，大量输出显示为接近 0.5 的固定置信度，表面上看像是阈值设置问题。然而，追踪后的 NCNN 图结构表明，类别相关分支在模型内部已经包含了 Sigmoid，而 C++ 解析器仍对其再次执行了一次 Sigmoid。这使得原本极小的背景概率被重新映射到接近 0.5，从而在整张图像中产生密集伪候选。

After removing redundant activation in the parser and keeping only necessary transforms for task-specific branches such as angle values, the confidence distribution became normal again. This case indicates that deployment parsers must follow the actual exported graph semantics rather than inherit assumptions from previous model versions.

在解析器中去除冗余激活，并仅对角度等任务特有分支保留必要变换后，置信度分布恢复正常。该案例表明，部署端解析逻辑必须严格遵循实际导出图的语义，而不能简单继承旧版本模型的假设。

[Figure 6 here: Effect of single vs. duplicated Sigmoid on background scores / 图 6 插入此处：单次与重复 Sigmoid 对背景分数的影响]

The third failure was caused by repeated decoding of already-decoded coordinates. In pose and OBB modes, bounding boxes were visible but keypoints or rotated boxes were not correctly rendered. The underlying reason was that the exported end-to-end outputs already represented absolute pixel coordinates, while the device-side parser still treated them as grid-relative values and applied stride-based restoration again. As a result, the reconstructed coordinates were pushed far outside the image region.

第三个故障来源于对已解码坐标的重复还原。在姿态和 OBB 模式下，边框可以显示，但关键点或旋转框无法被正确绘制。其根本原因在于，导出的端到端输出已经表示为绝对像素坐标，而设备侧解析器仍将其视为网格相对值，再次进行基于 stride 的还原。这样会把重构坐标推到远超图像范围的位置。

To solve this issue, an independent parser branch was introduced for end-to-end outputs. Instead of blindly applying legacy decoding equations, the parser first checks whether the numeric pattern matches already-decoded coordinates. If so, the values are directly consumed in image space; otherwise, the legacy decoding path is retained for compatibility. This correction restored valid pose keypoints and OBB geometry.

为解决该问题，系统为端到端输出引入了独立解析分支。解析器不再盲目套用旧的解码公式，而是先判断数值模式是否符合已解码坐标；若符合，则直接在图像空间中使用这些值，否则保留旧解码路径以兼容传统输出。修复后，姿态关键点和 OBB 几何形态均恢复正常。

[Figure 7 here: Semantic difference between legacy grid decoding and already-decoded E2E coordinates / 图 7 插入此处：传统网格解码与 E2E 已解码坐标的语义对比]

The fourth failure concerned incorrect interpretation of the OBB output layout. The original parser assumed the output order to be bbox, angle, and then classes. Structural tracing of the NCNN concat chain later revealed that the actual order was bbox, class probabilities, and then the raw angle value. This mismatch caused the first class score to be misread as an angle, shifted all following class indices, and further amplified the previous double-Sigmoid problem.

第四个故障与 OBB 输出列布局的错误理解有关。最初的解析器假设输出顺序为 bbox、angle 和 classes。随后对 NCNN concat 链的结构追踪表明，真实顺序实际上是 bbox、类别概率，最后才是原始角度值。这种误判使第一个类别分数被误读为角度，并导致后续所有类别索引发生偏移，同时进一步放大了此前的双重 Sigmoid 问题。

Once the class start index, class count, and angle column were corrected, the OBB results became interpretable again, with reasonable class labels, confidence values, and orientation estimation. This failure emphasizes that tensor layout verification should be considered a standard deployment step, especially for multi-branch tasks such as OBB and pose estimation.

在修正类别起始列、类别数量和角度列位置之后，OBB 结果恢复了可解释性，类别标签、置信度和方向估计都回到了合理状态。该故障说明，对于 OBB 和姿态估计这类多分支任务，张量布局核查应成为标准部署步骤，而不应等到出错后才被动使用。

[Table 6 here: OBB output layout assumption vs. actual layout / 表 6 插入此处：OBB 输出列布局假设与实际布局对照]

The final issue involved cross-platform engineering compatibility. On Windows-based development environments, IDE-level analysis occasionally reported undefined Android logging symbols or misleading standard-library errors, even though the NDK compilation path itself was valid. Although such issues did not necessarily break the APK build, they significantly affected maintainability, readability, and debugging efficiency.

最后一个问题涉及跨平台工程兼容性。在基于 Windows 的开发环境中，IDE 层面的分析有时会报告 Android 日志符号未定义或标准库相关误报，尽管 NDK 实际编译路径本身是有效的。虽然这类问题不一定直接导致 APK 构建失败，但它们会显著影响项目的可维护性、可读性和调试效率。

To mitigate this problem, compatibility wrappers and more robust calls were introduced where necessary, without compromising the actual Android build path. This engineering adjustment may look less algorithmically important than output parsing, but it is essential for sustaining a reproducible and maintainable deployment project.

为缓解这一问题，项目在必要位置引入了兼容封装和更稳健的调用方式，同时不破坏真实的 Android 构建路径。与输出解析问题相比，这类改动看起来不那么“算法化”，但它们对于维持一个可复现、可维护的部署项目同样至关重要。

[Figure 8 here: Example of cross-platform compatibility issues before and after adjustment / 图 8 插入此处：跨平台兼容性问题修复前后示例]

---

## 5. Experiments and Results / 5. 实验与结果

The experimental section is designed to validate three aspects of the study: whether the framework can support unified multi-task execution, whether the SafeHat case can form a complete real-device deployment loop, and whether the proposed debugging methodology can effectively correct deployment failures. Rather than constructing a large-scale benchmark matrix across all possible devices and model sizes, the experiments focus on reproducible engineering evidence.

本节实验旨在验证三方面内容，即框架是否能够支撑统一多任务执行，SafeHat 案例是否能够形成完整真机部署闭环，以及所提出的调试方法是否能够有效修复部署故障。本文并不试图构建覆盖所有设备和模型规模的大规模 benchmark 矩阵，而是聚焦于可复现的工程证据。

The training side uses the Ultralytics workflow for model loading, fine-tuning, validation, and NCNN export with a default input size of 640 × 640. The deployment side is based on Android Studio, Android Gradle Plugin, NDK toolchains, NCNN, and OpenCV-Mobile. The system supports CPU inference as well as optional Vulkan/Turnip-related backends for future runtime comparison. This environment is sufficient to validate the complete path from model preparation to device-side execution.

训练侧采用 Ultralytics 工作流完成模型加载、微调、验证与 NCNN 导出，默认输入尺寸为 640 × 640。部署侧基于 Android Studio、Android Gradle Plugin、NDK 工具链、NCNN 和 OpenCV-Mobile 构建。系统支持 CPU 推理，并保留 Vulkan 和 Turnip 相关后端用于后续运行对比。这一环境足以验证从模型准备到设备侧执行的完整路径。

[Table 7 here: Experimental environment and major dependencies / 表 7 插入此处：实验环境与主要依赖]

The first result is the successful functional validation of the unified framework. Under the same application and interaction pipeline, the system can switch among detection, segmentation, pose estimation, classification, and OBB tasks, while preserving task-specific visualization outputs. This confirms that the proposed layered design is capable of supporting heterogeneous visual tasks within a single Android deployment framework.

第一项结果是统一框架的功能验证成功。在同一应用和交互流程下，系统可以在检测、分割、姿态估计、分类和 OBB 任务之间切换，并保留各任务对应的可视化输出。这表明，所提出的分层设计能够在单一 Android 部署框架中支撑异构视觉任务。

[Figure 9 here: Multi-task visual results on Android devices / 图 9 插入此处：Android 端多任务运行结果]

The SafeHat case provides the most complete deployment evidence. The detection model is fine-tuned on a ten-class dataset related to helmets and personal protective equipment, then exported to NCNN and deployed as Android assets. Real-device execution confirms that the model can be correctly loaded and that the output shape matches the target number of classes. Threshold adjustment is also functional on the device side, which indicates that the deployment path is not merely static but interactively controllable.

SafeHat 案例提供了最完整的部署证据。该检测模型在一个包含十个安全帽和个人防护装备类别的数据集上完成微调，随后导出为 NCNN 并部署到 Android 资产目录。真机运行结果表明，模型能够被正确加载，且输出形状与目标类别数保持一致。设备端阈值调节功能同样有效，这说明部署路径并非静态展示，而是具备交互可控性。

[Figure 10 here: Real-device SafeHat detection results / 图 10 插入此处：SafeHat 真机检测效果]  
[Table 8 here: Key validation items of SafeHat deployment / 表 8 插入此处：SafeHat 部署验证关键信息]

The most important experimental findings come from before-versus-after comparisons of failure correction. After the preprocessing correction, abnormal high-confidence detections in empty scenes were removed and background score statistics dropped sharply. After removing redundant Sigmoid operations, the confidence distribution of E2E branches became significantly more realistic. After correcting E2E coordinate handling and OBB layout parsing, keypoints, rotated boxes, and class labels became consistent with the expected task semantics. These observations jointly verify the effectiveness of the proposed diagnostic methodology.

实验中最重要的发现来自故障修复前后的对比。预处理修正后，空场景中的异常高置信检测被消除，背景分数统计显著下降。去除冗余 Sigmoid 后，端到端分支的置信度分布明显回归合理。修正 E2E 坐标处理和 OBB 布局解析后，关键点、旋转框和类别标签重新与预期任务语义保持一致。这些现象共同验证了本文诊断方法的有效性。

[Table 9 here: Comparison of failure symptoms and recovery results / 表 9 插入此处：故障现象与修复结果对比]

In terms of runtime efficiency, this work intentionally avoids claiming overly specific performance gains without a fully standardized timing matrix. Still, the engineering implication is clear: the end-to-end detection branch simplifies the mobile inference path by reducing dense-candidate filtering and NMS-related implementation burden. The current framework also preserves the possibility of future comparisons across CPU and GPU backends under unified code paths.

在运行效率方面，本文有意避免在缺少严格标准化测速矩阵的情况下给出过于具体的性能增益结论。尽管如此，其工程意义依然明确，即端到端检测分支通过减少密集候选筛选和 NMS 相关实现负担，简化了移动端推理路径。当前框架也保留了在统一代码路径下对 CPU 和 GPU 后端开展未来对比的可能性。

[Table 10 here: Suggested runtime and resource evaluation template / 表 10 插入此处：运行效率与资源评测模板]

Overall, the experiments demonstrate that the value of this work lies not in reporting a new state-of-the-art detector, but in proving that a YOLO26-based visual inference system can be deployed, debugged, and stabilized on Android devices through a structured engineering methodology.

总体而言，本节实验说明，本文的价值并不在于报告一个新的最优检测器，而在于证明：通过结构化工程方法，基于 YOLO26 的视觉推理系统可以在 Android 设备上被成功部署、诊断并稳定运行。

---

## 6. Conclusions and Future Work / 6. 结论与展望

This study presented a deployment and diagnostic framework for YOLO26-based multi-task visual inference on Android devices. By combining Java interaction, JNI bridging, native C++ inference, and NCNN-based execution, the framework supports five visual task types within a unified application. A SafeHat case study was further used to establish a complete workflow from fine-tuning and model export to asset deployment and real-device validation.

本文提出了一套面向 Android 设备的 YOLO26 多任务视觉推理部署与诊断框架。通过结合 Java 交互、JNI 桥接、原生 C++ 推理以及基于 NCNN 的执行路径，该框架在统一应用内支持五类视觉任务。进一步地，本文以 SafeHat 案例建立了从模型微调、模型导出到资产部署和真机验证的完整闭环。

The main finding of the paper is that deployment consistency is a decisive factor in mobile visual inference. The observed failures were not caused by a single source, but by mismatches across preprocessing, activation placement, coordinate semantics, tensor layout interpretation, and engineering environment handling. By organizing these issues into a reusable debugging workflow, the study turns fragmented deployment experience into a more general diagnostic methodology.

本文的主要结论在于，部署一致性是移动视觉推理成败的关键因素。所观察到的故障并非来自单一源头，而是由预处理、激活函数位置、坐标语义、张量布局理解和工程环境处理等多方面不匹配共同造成。通过将这些问题组织为可复用的调试流程，本文把零散的部署经验提升为更具通用性的诊断方法。

Nevertheless, several limitations remain. Although five task types are supported within the same framework, the most complete experimental validation is still centered on the SafeHat detection case. In addition, standardized runtime comparisons across model scales, device types, and hardware backends are not yet sufficiently complete. These aspects should be strengthened in future work.

尽管如此，本文仍存在若干局限。虽然框架支持五类任务，但最完整的实验验证仍主要集中在 SafeHat 检测案例上。此外，围绕不同模型规模、不同设备类型和不同硬件后端的标准化运行对比仍不够充分。这些方面均应在后续研究中进一步加强。

Future work may proceed along three directions. First, INT8 quantization and other deployment-oriented compression strategies can be explored to further reduce device-side cost. Second, cross-device evaluation on different Android chip platforms should be performed to examine stability and portability. Third, hybrid edge-cloud collaboration and online model update mechanisms can be integrated to support more complex real-world visual applications.

未来工作可沿三个方向展开。第一，可研究 INT8 量化及其他面向部署的压缩策略，以进一步降低设备端开销。第二，应在不同 Android 芯片平台上开展跨设备评测，以验证系统的稳定性和可移植性。第三，可结合边云协同和在线模型更新机制，以支撑更复杂的真实视觉应用场景。
