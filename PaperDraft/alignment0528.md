# MethodsX 转投对齐方案（2026-05-28）

## 1. 总体判断

当前稿件 `Paper20260527.md` 不再适合继续包装成“高水平视觉检测/边缘 AI 性能论文”。在不新增实验数据的前提下，最稳妥的转向是投 MethodsX，并将文章重新定义为：

> 一个面向 YOLO-to-NCNN Android 部署链的、可复现的部署一致性诊断方法与验证协议。

也就是说，文章主贡献不再是“提出一个多任务 PPE 检测系统”，而是“给出一套可复用的方法，用于发现、定位和修复 Android-NCNN 部署中的一致性偏差”。SafeHat、五任务运行、Kirin 970/659 延迟和 48 张图敏感性分析都作为方法验证证据，而不是作为大规模算法性能证据。

MethodsX 官方定位是发表可复现的方法文章，接受新方法、已有方法的新应用或已有方法的定制化版本，并要求方法细节足够完整、能让读者复现，同时提供方法有效性的验证证据。参考：

- MethodsX Guide for Authors: https://www.sciencedirect.com/journal/methodsx/publish/guide-for-authors
- MethodsX journal page: https://www.sciencedirect.com/journal/methodsx

## 2. 新定位

### 原稿定位

原稿更像：

- Android-NCNN 多任务部署框架；
- PPE monitoring 应用系统；
- YOLO26/yolo26n 工程资产说明；
- 部署故障诊断案例；
- 两台设备 CPU 路径延迟测试。

这个定位投 Applied Sciences 会被审稿人自然要求：

- 更大测试集；
- 更多检测指标；
- 与其他模型/框架对比；
- 消融实验；
- 更充分的五任务性能评估。

这些都需要新增实验，不符合当前约束。

### MethodsX 目标定位

建议改成：

- **Method object**：YOLO-based model export and Android-NCNN deployment consistency diagnosis workflow。
- **Method use case**：SafeHat PPE detection and five YOLO task assets。
- **Validation goal**：证明该方法能定位常见部署偏差、修复后产生可观测改进，并能在真实 Android CPU 设备上执行。
- **Not the goal**：不证明检测器 SOTA，不证明实时性能最优，不证明所有视觉任务都有完整 benchmark。

核心句式建议：

> This article describes a reproducible workflow for diagnosing deployment inconsistency in YOLO-to-NCNN Android inference pipelines. The method integrates backend comparison, intermediate tensor inspection, graph-structure tracing, and regression validation. A PPE monitoring deployment is used as the demonstration case.

## 3. 推荐题目

首选题目：

> A reproducible workflow for diagnosing YOLO-to-NCNN deployment inconsistencies on resource-constrained Android devices

备选题目：

> A method for consistency diagnosis in Android-NCNN deployment of YOLO-based visual inference models

更应用化题目：

> A reproducible Android-NCNN deployment diagnosis workflow for YOLO-based PPE monitoring

不建议继续使用原题：

> An Android-NCNN Deployment and Consistency Diagnosis Framework for Multi-Task YOLO-Based PPE Monitoring on Resource-Constrained Devices

原因：原题会让审稿人期待“完整多任务系统论文”和“PPE monitoring 性能论文”，从而引出大规模实验要求。

## 4. 摘要改写方向

MethodsX 摘要应围绕“方法是什么、如何执行、如何验证、为什么有用”。不要把摘要写成 Applied Sciences 风格的工程系统结果摘要。

建议摘要结构：

1. 背景问题：YOLO 模型从桌面训练后端导出到 Android-NCNN 时，常出现前处理、激活、坐标语义、输出布局等部署偏差。
2. 方法：提出五阶段诊断流程：anomaly logging, intermediate-output inspection, cross-backend comparison, structure tracing, regression validation。
3. 实施细节：给出 Java-JNI-C++-NCNN 管线、脚本、日志、张量检查和回归验证步骤。
4. 验证：用四类真实故障展示方法有效性；空场景分数从 0.84-0.96 降到 <0.0001；冗余 sigmoid 修复后概率分布恢复；坐标和 OBB 布局错误被定位。
5. 可复用性：方法适用于其他 YOLO-to-NCNN Android 部署场景，尤其是资源受限设备。

需要弱化：

- “multi-task framework”；
- “five-task coverage”；
- “PPE monitoring system baseline”；
- “latency characterization”。

这些可以出现在 validation，但不应是摘要主线。

## 5. 贡献重写

原稿三条贡献应重写为 MethodsX 风格。

### 建议贡献 1

> A stepwise diagnosis method for YOLO-to-NCNN Android deployment consistency, combining anomaly logging, intermediate-output inspection, cross-backend comparison, graph-structure tracing, and regression validation.

### 建议贡献 2

> A reproducible implementation protocol that connects model export, asset replacement, JNI/native inference, log collection, tensor-shape checking, and repair validation.

### 建议贡献 3

> A validated demonstration on four real deployment fault modes: preprocessing mismatch, redundant activation, coordinate semantic misinterpretation, and OBB output-layout misjudgment.

### 建议贡献 4（可选）

> Open-source code, scripts, logs, figures, and frozen test samples are provided to support reuse of the method.

不要再把“SafeHat 10-class PPE detection performance”或“五任务统一框架”放在贡献核心位置。

## 6. 文章结构建议

MethodsX 可以保留研究论文叙事，但必须明显变成 method article。建议结构如下。

### 1. Introduction

目标：解释为什么 Android-NCNN 部署一致性诊断是一个方法问题。

应包含：

- 边缘部署常见转换链：PyTorch / ONNX / NCNN / Android JNI；
- 错误不一定表现为 crash，而是表现为高置信背景、50% 置信塌缩、坐标错位、类别错读；
- 现有工具多解决“能否导出/能否推理”，缺少系统诊断流程；
- 本文提供可复现工作流。

删减：

- 大量 YOLO SOTA 背景；
- 对轻量 backbone 的泛泛介绍；
- “resource-constrained devices” 的过度定义。

### 2. Method Overview

建议由原稿 Figure 1 和 Figure 4 合并形成主图。

包含：

- 输入：PyTorch/Ultralytics YOLO model, NCNN param/bin, Android app, test images/logs；
- 输出：fault category, root cause, repair action, regression evidence；
- 五阶段流程；
- 决策门：什么时候进入 S2、S3、S4、S5。

这一节是全文核心。

### 3. Method Details

建议拆成 5 个小节。

#### 3.1 Environment and assets

来自原稿 Table 7，但要压缩。

保留：

- Python/Ultralytics；
- NCNN version；
- OpenCV-Mobile；
- Android NDK；
- device information；
- model input sizes；
- scripts。

删除或降级：

- 过多 Gradle/compileSdk 细节，除非与复现直接相关。

#### 3.2 Export and asset deployment protocol

来自原稿 Table 2。

写成 protocol：

1. Train or obtain YOLO weights。
2. Export to NCNN param/bin。
3. Verify output tensor shapes。
4. Replace Android assets。
5. Run logcat-based regression。

重点是“步骤可执行”，不是工程故事。

#### 3.3 Android-NCNN inference path

来自原稿 Java-JNI-C++-NCNN 四层架构。

保留：

- Java UI / JNI bridge / C++ parser / NCNN runtime；
- letterbox preprocessing；
- task-specific parser；
- diagnostic logging points。

弱化：

- “统一多任务框架”作为创新点。

#### 3.4 Five-stage diagnosis workflow

来自原稿 Table 4a/4b，是 MethodsX 最重要内容。

建议把每一阶段写成固定模板：

- Input；
- Operation；
- Diagnostic signal；
- Exit criterion；
- Script/tool；
- Expected artifact。

例如：

| Stage | Input | Operation | Output artifact |
|---|---|---|---|
| S1 | Android logcat / screenshots | collect confidence/shape/render anomalies | anomaly report |
| S2 | NCNN param/bin | inspect output tensors | tensor-shape table |
| S3 | same image across backends | compare PyTorch, Python-NCNN, Android-NCNN | backend consistency table |
| S4 | NCNN graph + parser code | trace activation/layout/coordinate semantics | root-cause note |
| S5 | repaired app + frozen scenes | rerun regression | before-after evidence |

#### 3.5 Fault-specific repair templates

把四类故障写成 reusable templates，而不是单次案例。

模板建议：

- Symptom；
- Diagnostic signature；
- Likely cause；
- Inspection command/script；
- Repair rule；
- Regression criterion。

四个模板：

1. Preprocessing mismatch；
2. Redundant activation；
3. Coordinate semantic misinterpretation；
4. OBB layout misjudgment。

### 4. Method Validation

这是 MethodsX 的关键章节。不要叫 “Results and Discussion”，建议叫 “Method validation”。

现有数据可全部保留，但叙事要改。

#### 4.1 Validation case and available data

说明 SafeHat 是 demonstration case，不是 benchmark dataset。

包括：

- 10 类 PPE/Person；
- training baseline mAP50/Precision 只是确认模型可用；
- 48 frozen images 用于参数敏感性；
- logcat / scripts / figures / repository。

#### 4.2 Validation on four fault modes

来自原稿 Fault Mode I-IV 和 Table 9。

重点表达：

- 方法能否定位问题；
- 定位依据是什么；
- 修复后信号如何变化。

不要强调“检测性能提升”，强调“diagnostic signal recovery”。

#### 4.3 Reproducibility and regression artifacts

新增一个不需要实验的表，整理现有材料。

建议表：

| Artifact | Path / source | Purpose |
|---|---|---|
| Android source | GitHub repo | reproduce deployment |
| NCNN assets | app/src/main/assets | model loading |
| diagnosis scripts | scripts/*.py | tensor/log analysis |
| frozen images | data/testdata/ | parameter sweep |
| latency log | logcat.txt | runtime statistics |
| table10.csv | runs/paper_figures/table10.csv | sensitivity results |

#### 4.4 Runtime feasibility evidence

来自原稿 Table 11。

表述方式：

- “The method was exercised on two real Android CPU devices.”
- “Latency results indicate the operating scale of the validation environment.”
- 不写成性能 benchmark。

#### 4.5 Parameter sensitivity as protocol demonstration

来自原稿 Table 10。

需要修改结论：

- T 和 gamma 是 display-score compression 参数，不是模型置信度性能提升；
- Kc 和 overlap 是候选保留/抑制策略；
- 48 张图只是 protocol demonstration；
- 不要说 selected setting “optimal”，只说 “conservative setting used in the demonstration”。

## 7. 图表处理建议

### 建议保留并重排

| 原稿图表 | MethodsX 中的角色 | 处理建议 |
|---|---|---|
| Figure 1 | Method pipeline | 保留，可作为 graphical abstract 候选 |
| Figure 2 | Implementation architecture | 压缩，放 Method details |
| Figure 3 | Deployment protocol | 可合并到 Figure 1，避免图太多 |
| Figure 4 | Diagnosis workflow | 必须保留，是核心方法图 |
| Figure 5 | Fault validation | 保留 |
| Figure 6 | Fault validation | 保留 |
| Figure 7 | Fault validation | 保留 |
| Figure 8 | Demonstration case background | 降级，缩小 |
| Figure 9 | Runtime smoke test | 保留，但不要作为主证据 |
| Table 1 | Task assets | 压缩或移到补充材料 |
| Table 2 | Protocol | 保留，改成 reproducible protocol |
| Table 3 | SafeHat classes | 保留但缩短 |
| Table 4a/4b | Core method | 必须保留并增强 |
| Table 5 | Validation evidence | 保留 |
| Table 6 | Repair template | 保留 |
| Table 7 | Environment | 保留但压缩 |
| Table 8 | Deployment checklist | 改成 validation checklist |
| Table 9 | Fault summary | 保留 |
| Table 10 | Protocol sensitivity demo | 保留但降调 |
| Table 11 | Runtime feasibility | 保留但降调 |

### 建议新增 2 个表

不需要新增实验，只整理已有信息。

#### 新表 A：Method Reproducibility Checklist

列：

- Step；
- Required input；
- Script/tool；
- Expected output；
- Failure signal。

#### 新表 B：Reusable Fault Diagnosis Templates

列：

- Fault mode；
- Symptom；
- Diagnostic signal；
- Root-cause check；
- Repair rule；
- Regression criterion。

这两个表能显著增强 MethodsX 匹配度。

## 8. 内容降档策略

### 必须降级的表述

把：

> proposed framework

多处改为：

> proposed method / workflow / protocol

把：

> multi-task YOLO-based PPE monitoring system

改为：

> demonstration deployment case using PPE detection and related YOLO task assets

把：

> results position the work as a reproducible deployment-system baseline

改为：

> validation results demonstrate that the workflow can expose and repair representative deployment inconsistencies

把：

> five-task runtime coverage validates the framework

改为：

> five-task runtime coverage confirms that the diagnosis protocol can be exercised across heterogeneous output formats

### 必须避免的表述

避免：

- state-of-the-art；
- optimal；
- universal benchmark；
- real-time；
- comprehensive multi-task evaluation；
- detector performance improvement；
- superior to existing methods。

建议：

- reproducible；
- protocol；
- diagnostic workflow；
- implementation-level validation；
- representative fault modes；
- reusable deployment checks；
- demonstration case。

## 9. 参考文献修正

当前参考文献错配较严重，转投前必须修。

### 现有明显问题

- 文中 `[1,2]` 被用于 real-time detection architectures，但 [2] 是 MNN。
- `[3,4]` 被用于 lightweight backbone，但 [4] 是 TensorFlow Lite。
- `[7,8]` 被用于 Android 单任务部署，但 [7][8] 是 ResNet/DenseNet。
- 结果部分用 `[27]` 指 Ultralytics workflow，但 [27] 是 OpenCV 书。

### MethodsX 参考文献策略

参考文献不需要堆很多，建议控制在 20-30 篇。

必须覆盖：

1. YOLO / Ultralytics source；
2. NCNN；
3. ONNX / model conversion；
4. Android NDK / JNI / camera or mobile deployment；
5. TFLite / MNN / ONNX Runtime Mobile 作为背景；
6. PPE / safety helmet deployment application；
7. Reproducible software/method reporting。

引用目标不是证明算法先进，而是证明方法背景完整、工具链来源清楚。

## 10. MethodsX 审稿风险与应对

### 风险 1：方法创新性不够

应对：

不要声称单个技术新，而要强调组合后的 workflow 可复用。

可写：

> The novelty of the method lies in integrating backend-level comparison, tensor-shape inspection, graph-structure tracing, and Android-side regression validation into a reproducible deployment-diagnosis protocol.

### 风险 2：实验数据规模小

应对：

MethodsX 不要求 SOTA benchmark，但要求方法验证。强调四类真实故障的 before-after 诊断证据。

可写：

> The validation is not intended as a detector benchmark. It demonstrates whether the method can reveal, localize, and verify repairs for representative deployment inconsistencies.

### 风险 3：过于应用特定

应对：

把 SafeHat 写成 demonstration case，把四类故障写成可迁移模板。

说明这些故障不依赖 PPE 场景：

- letterbox mismatch；
- redundant activation；
- decoded-vs-grid coordinate mismatch；
- output column layout mismatch。

### 风险 4：更像软件，不像方法

应对：

在 Method Details 中给出明确步骤、输入、输出、判断标准和回归准则。代码只是载体，workflow 才是论文对象。

## 11. 建议重写后的目录

```text
Title
Abstract
Keywords

1. Introduction
2. Method Overview
   2.1 Problem definition
   2.2 Inputs and outputs of the workflow
   2.3 Five-stage diagnosis logic
3. Method Details
   3.1 Environment and software assets
   3.2 YOLO-to-NCNN export and asset deployment protocol
   3.3 Android-NCNN inference and logging path
   3.4 Five-stage consistency diagnosis procedure
   3.5 Fault-specific repair templates
4. Method Validation
   4.1 Demonstration case: SafeHat Android deployment
   4.2 Validation on four representative fault modes
   4.3 Reproducibility artifacts and regression checklist
   4.4 Runtime feasibility on two Android CPU devices
   4.5 Parameter sensitivity demonstration
5. Method Characteristics and Limitations
6. Conclusions

Ethics Statements
Declaration of Competing Interest
Data Availability
Acknowledgments
References
```

## 12. 可直接执行的修改清单

### 第一轮：定位和结构

- [ ] 改题目；
- [ ] 重写摘要；
- [ ] 重写 Introduction，删除 SOTA 式背景；
- [ ] 把 Deployment Architecture 改为 Method Details；
- [ ] 把 Diagnosis Workflow 提升为全文核心；
- [ ] 把 Results and Discussion 改为 Method Validation；
- [ ] 把 Limitations 改为 Method Characteristics and Limitations。

### 第二轮：证据重排

- [ ] Table 4a/4b 增强为核心 protocol 表；
- [ ] Table 9 改成 reusable fault templates；
- [ ] Table 10 改成 parameter sensitivity demonstration；
- [ ] Table 11 改成 runtime feasibility evidence；
- [ ] Figure 1/Figure 4 合并或明确分工；
- [ ] Figure 9 降级为 runtime smoke test。

### 第三轮：语言降档

- [ ] 删除 detector benchmark 语气；
- [ ] 删除或弱化 multi-task system 语气；
- [ ] 删除 optimal / superior / comprehensive 等词；
- [ ] 所有“performance result”改成“validation evidence”；
- [ ] 所有“framework contribution”改成“method/protocol contribution”。

### 第四轮：参考文献和格式

- [ ] 修复引用编号错配；
- [ ] 补 Ultralytics、NCNN、ONNX、Android/NDK、TFLite/MNN/ONNX Runtime Mobile；
- [ ] 补 2-4 篇 PPE/helmet Android/edge deployment 应用文献；
- [ ] 检查 MethodsX ethics/data/competing interest 声明；
- [ ] 检查图表是否能独立理解；
- [ ] 确认 GitHub/Zenodo 链接可访问。

## 13. 推荐投稿版本的一句话定位

最终投稿版应让编辑一眼看到：

> 这不是一篇追求检测精度 SOTA 的论文，而是一篇可复现的方法论文，提供了 Android-NCNN 部署一致性诊断的完整流程、脚本化检查点、故障模板和真实设备验证证据。

如果全文能稳定围绕这句话展开，MethodsX 的匹配度会明显高于 Applied Sciences。
