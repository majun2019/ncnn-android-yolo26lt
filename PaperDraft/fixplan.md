# Applied Sciences 转投最小补救方案

## 0. 当前判断

Electronics 的拒稿属于编辑部 desk reject，核心信号是 `discipline, novelty and general significance`。这说明文章目前被视为 AI / edge intelligence 研究稿时，算法新意和普适意义不足；但如果改成工程应用型、部署验证型论文，价值仍然成立。

转投 Applied Sciences 的策略不是继续提高理论包装，而是降低声称、强化应用闭环，把文章定位为：

> 面向低成本 Android 终端的施工现场 PPE 视觉监测部署系统，以及一套可复现的 NCNN 部署一致性诊断流程。

最小补救目标：

1. 避免再次被认为是“只是一篇部署记录”。
2. 避免审稿人抓住“无消融、单设备、无对比”直接否定。
3. 不大改工程、不重训大模型，只补能支撑投稿的最低限度证据。
4. 把论文口吻从“提出通用 AI 方法”改为“提出并验证一个应用部署框架”。

---

## 1. 目标期刊与栏目定位

### 1.1 推荐期刊

目标期刊：Applied Sciences

建议栏目方向：

- Computer Science and Engineering
- Artificial Intelligence
- Applied Computer Vision
- Smart Cities / Intelligent Systems 相关 Special Issue，如果主题明确包含 edge computing、mobile vision、construction safety，则优先考虑。

### 1.2 投稿定位

不要再突出“YOLO26 是一个新模型族”。应明确说明：

- YOLO26 是本项目中工程管理的一组 yolo26n 资产，不是新的通用模型系列。
- 本文贡献不是新检测算法，而是资源受限 Android 设备上的多任务部署、输出一致性恢复和故障诊断闭环。
- SafeHat PPE 检测是主案例，其余检测、分割、姿态、分类、OBB 是统一部署框架的运行覆盖证明。

### 1.3 降低声称后的核心卖点

Applied Sciences 更适合接受以下叙事：

1. 一个低成本 Android 终端上的施工安全视觉监测原型系统。
2. 一个 Java-JNI-C++-NCNN 的多任务部署框架。
3. 一个针对实际部署偏差的五阶段诊断流程。
4. 一组真实故障的 before/after 定量修复证据。
5. 一个开源代码和 Zenodo DOI 支撑的可复现工程案例。

---

## 2. 标题、摘要和关键词最小修改

### 2.1 当前标题问题

当前标题：

> Consistency-Aware Deployment of Multi-Task YOLO26 on Resource-Constrained Android Devices: A Java-JNI-C++-NCNN Architecture with Calibrated Post-Processing and Quantitative Fault Diagnosis

问题：

- 太像“自创 YOLO26 方法论文”，容易触发 novelty 审查。
- 标题过长，概念过多。
- `YOLO26` 在标题中太显眼，审稿人会追问模型创新。
- `Quantitative Fault Diagnosis` 声称偏大，实际更像 deployment diagnosis。

### 2.2 建议标题

优先标题：

> An Android-NCNN Deployment and Consistency Diagnosis Framework for Multi-Task YOLO-Based PPE Monitoring on Resource-Constrained Devices

备选标题 1：

> Resource-Constrained Android Deployment of Multi-Task YOLO-Based Vision Models for Construction-Site PPE Monitoring

备选标题 2：

> A Java-JNI-C++-NCNN Framework for On-Device PPE Monitoring and Deployment Consistency Diagnosis on Low-Cost Android Devices

建议使用优先标题。它把重点从“YOLO26”转移到“Android-NCNN deployment”和“PPE monitoring”。

### 2.3 摘要修改原则

摘要需要做三处收缩：

1. 第一段不要说“multi-task vision models”太泛，改成“YOLO-based mobile vision models for PPE monitoring and related visual tasks”。
2. 不要把五个任务都说成同等贡献，明确 SafeHat 是主案例。
3. 把“systematic ablations are reserved for future work”从摘要中删掉。摘要主动暴露弱点不利于送审。

摘要应保留的硬证据：

- empty-scene median score: 0.84-0.96 -> < 0.0001
- redundant activation 修复后的 raw probability 分布
- UI confidence 0.35-0.73
- five task runtime coverage
- Kirin 970 CPU latency: 321.3-430.5 ms, classification 33.6 ms
- open-source repository and reproducibility

### 2.4 关键词修改

建议关键词：

- Android deployment
- NCNN
- PPE monitoring
- construction-site safety
- edge AI
- deployment consistency
- resource-constrained devices
- mobile computer vision

---

## 3. 最小补救实验

最小补救实验只做三类，按优先级排序。原则是不大改训练，只补最容易被审稿人追问的证据。

## 3.1 实验 A：参数敏感性小表，必须补

### 目的

解决审稿人最可能质疑的问题：

> T = 6、gamma = 1.6、Kc = 50、overlap 0.15 / 0.20、PPE locking 0.35:0.65 是否只是经验拍脑袋？

### 最小做法

只在 SafeHat 主案例上做，不需要五任务全做。

选择 20-50 张代表性图片或若干 logcat 帧：

- target-present scenes：有人、有 PPE、有安全帽/无安全帽。
- empty/background scenes：无目标或背景干扰。
- hard-negative scenes：反光、机械边缘、人体相似纹理。

### 建议参数组合

只做一维扫描，不做全组合网格，避免工作量爆炸。

| Group | Parameter | Values | Fixed settings |
|------|-----------|--------|----------------|
| A1 | Temperature T | 1, 3, 6, 10 | gamma=1.6, Kc=50 |
| A2 | Gamma | 1.0, 1.3, 1.6, 2.0 | T=6, Kc=50 |
| A3 | Per-class Top-K Kc | 20, 50, 100 | T=6, gamma=1.6 |
| A4 | PPE overlap threshold | 0.10, 0.15, 0.20, 0.30 | T=6, gamma=1.6, Kc=50 |

PPE locking 权重可以不做表格实验，只在 limitation 中说明目前采用经验权重，未来会系统评估。若时间允许，再补：

| Group | Parameter | Values |
|------|-----------|--------|
| A5 | PPE locking weight | 0.25:0.75, 0.35:0.65, 0.50:0.50 |

### 最小指标

不要求完整 mAP。使用部署相关指标即可：

| Metric | Meaning |
|--------|---------|
| FP_empty | empty scenes 中误检数量 |
| frac_gt090 | score > 0.9 的候选比例 |
| TP_keep | target-present scenes 中 Person/PPE 是否保留 |
| mean_candidates | NMS 前/后候选数量 |
| UI_score_range | UI 显示分数范围 |

### 论文中新增位置

在 Section 5.2 后新增一个小节：

> 5.3 Parameter Sensitivity of Runtime Post-Processing

原 5.3 latency 改成 5.4。

### 新增表格建议

新增 Table 10，把原 latency 表后移为 Table 11。

表名：

> Table 10. Parameter sensitivity of runtime post-processing in the SafeHat main case.

表格可以简化为：

| Setting | FP_empty | frac_gt090 | TP_keep | Mean candidates after filtering | Comment |
|---------|----------|------------|---------|--------------------------------|---------|
| T=1 | ... | ... | ... | ... | insufficient compression |
| T=3 | ... | ... | ... | ... | moderate compression |
| T=6 | ... | ... | ... | ... | selected setting |
| T=10 | ... | ... | ... | ... | over-compressed display scores |
| gamma=1.0 | ... | ... | ... | ... | weak display separation |
| gamma=1.6 | ... | ... | ... | ... | selected setting |
| Kc=20 | ... | ... | ... | ... | may drop small PPE candidates |
| Kc=50 | ... | ... | ... | ... | selected setting |
| overlap=0.15/0.20 | ... | ... | ... | ... | selected setting |

### 重要提醒：不要做半成品

这张表要么不做，要么做完整。**禁止只填 2-3 个数值、其他用文字描述代替**——审稿人看到半成品表会直接判定不严肃，比没有表更糟。

最小可接受标准：

- A1（Temperature T）必须做满 4 个值 × 至少 3 个指标（FP_empty、frac_gt090、TP_keep）= 12 个数字。
- A2（gamma）、A3（Kc）、A4（overlap）可以缩到每组 3 个值，但每个值的指标必须全部填齐。
- 如果时间不允许做完 4 组，宁可只做 A1+A2 两组，其他参数在 Methods 末尾一句话写明“selected based on deployment validation; systematic sensitivity is left for future work”。

### 实施前置条件

参数敏感性不是“跑几帧就出表”。需要先准备：

1. 离线评测脚本：把 logcat 帧或代表性图片喂回 Python 端，按参数重算 FP_empty / frac_gt090 / TP_keep。
2. 测试集冻结：20-50 张图片或帧固定，三类（target / empty / hard-negative）比例固定，后续所有参数共用。
3. 评测脚本入仓库：作为可复现实验的一部分，写进 GitHub README。

这一步**单独估时 2-3 天，不是 1 天**。

---

## 3.2 实验 B：第二台 Android 设备（条件允许才做）

### 重要原则

此实验从“强烈建议补”降级为“if available”。理由：

1. 不是“插上跑一下就行”，需要重新签包、APK 兼容性调试、CameraX 在不同 Android 版本上的差异、logcat tag 过滤重设。
2. 如果第二台手机跑出来的数据质量差（CameraX 接入不顺、采样不稳定），反而成为审稿人攻击点。
3. 干净承认单设备 > 勉强凑出半可信数据。

**判断准则**：手边有可立即接入的第二台 Android 手机且能在 1 天内完成跑通，则做；否则直接跳到“替代写法”。

### 目的

当前只有 HUAWEI P20 Pro / Kirin 970，容易被质疑“单设备个例”。

### 最小做法

找任意一台更普通或更新的 Android 手机，不需要同样详细：

- Snapdragon 中低端机，例如 Snapdragon 7 系、6 系、778G、870 均可。
- 或者另一台旧机，例如 Kirin 980、Snapdragon 845、Dimensity 中端机。

只跑 detection 和 classification 两个任务即可。

### 最小指标

| Device | SoC | Android version | Detection mean latency | Classification mean latency | Loading status | Notes |
|--------|-----|-----------------|------------------------|-----------------------------|----------------|-------|
| HUAWEI P20 Pro | Kirin 970 | Android 10 | existing | existing | OK | baseline |
| Device 2 | ... | ... | ... | ... | OK | portability check |

如果能跑五任务更好，但不是必须。最小目标是证明框架不只绑定一台手机。

### 论文中新增位置

放在 latency 小节中，作为扩展表或补充段落：

> To avoid treating the Kirin 970 result as the only evidence of portability, a lightweight cross-device check was additionally conducted on a second Android phone...

### 替代写法：单设备 + 干净承认

如果不补第二台设备，**不要含糊带过**，直接在 Section 5.4（Applicability, Limitations）开头写：

> Validation in the present study is restricted to a single Kirin 970 device on the CPU path. The paper therefore does not claim cross-device generalization, and the reported latency values should be interpreted as an implementation-level feasibility baseline rather than a portability benchmark. A systematic cross-device and cross-SoC evaluation is explicitly listed as future work.

这种写法的好处：审稿人看到作者主动承认边界，反而会减少“你为什么不补”的追问。**勉强凑一台跑不通的数据，比这种诚实承认更糟。**

---

## 3.3 实验 C：轻量对比基线（不再列为正式实验项）

### 重要降级说明

此项从“建议补但可选”降级为“Limitation 段落即可，不进结果表”。理由：

1. Kirin 970 的 Mali-G72 在 NCNN Vulkan 路径上很多算子不支持或退化为 CPU 兜底，实测有可能比 CPU 还慢。一旦出现 Vulkan 慢于 CPU 的数据，反而被审稿人抓住“你的方案 GPU 不如 CPU”。
2. TFLite 跑 YOLOv8n demo 不是 apples-to-apples，结果价值有限。
3. 跨框架对比是 future work 的标准话术，绝大多数应用型期刊不会因为缺少这个对比就拒稿。

### 推荐写法

直接在 Limitations 段落写：

> A direct head-to-head comparison with TensorFlow Lite, MNN, and ONNX Runtime Mobile is intentionally omitted in the present study because an apples-to-apples comparison would require re-exporting the SafeHat detection model under each runtime with matched preprocessing, post-processing, and quantization settings. NCNN was selected for its mature ARM CPU support and transparent native-layer instrumentation, which are central to the consistency-diagnosis workflow proposed here. A standardized cross-framework benchmark is left for future work.

Vulkan 路径同样处理：

> The NCNN Vulkan path was retained in the implementation but is not reported as a quantitative result, because operator coverage on the test device’s Mali-G72 GPU is incomplete and a fair GPU/CPU comparison requires deeper engineering work that is outside the present scope.

### 目的（仅供参考，不必做）

解决“为什么用 NCNN，而不是 TFLite / MNN / ONNX Runtime Mobile”的问题。

### 最小可行方案

不需要完整移植同一个 YOLO26，只做一个引用或小型实测：

方案 1：同机 NCNN CPU vs NCNN Vulkan

- 工作量最低。
- 如果 Vulkan 已保留，只需要跑 detection 或 classification。
- 即使 Vulkan 未优化，也可以报告为 future optimization baseline。

方案 2：同机 TFLite YOLOv8n / YOLOv5n 官方 demo 延迟

- 不要求同模型，只作为 mobile runtime reference。
- 要在文字中说明不是严格 apples-to-apples，只是 runtime context。

方案 3：不做实测，只加强相关工作和 limitation

- 最省事，但审稿风险最高。

### 推荐选择

优先做方案 1：NCNN CPU vs NCNN Vulkan。

表格格式：

| Backend | Device | Task | Input size | Mean latency | Status | Comment |
|---------|--------|------|------------|--------------|--------|---------|
| NCNN CPU | Kirin 970 | Detection | 640 | existing | stable | reported baseline |
| NCNN Vulkan | Kirin 970 | Detection | 640 | ... | ... | optional acceleration path |

如果 Vulkan 跑不通，也可以诚实写入 limitation，不作为结果表。

**最终建议：本次投稿不做实验 C，全部走 Limitations 文字处理。**

---

## 4. 论文结构最小调整

## 4.1 Introduction

需要做的修改：

1. 第一段减少模型文献堆砌，增加施工现场 PPE 安全监测和低成本 Android 终端的应用需求。
2. 把“resource-constrained Android devices”的定义保留，但不要写得像概念创新。
3. 更早承认本文不是提出新 YOLO 算法。
4. Contributions 改成应用系统贡献，而不是算法贡献。

建议贡献重写为三点：

1. A reproducible Android-NCNN deployment framework for YOLO-based PPE monitoring and related visual tasks on low-cost Android devices.
2. A deployment consistency diagnosis workflow that localizes preprocessing, activation, coordinate-semantics, and tensor-layout deviations through logs, backend comparison, and graph tracing.
3. A SafeHat PPE case study with before/after repair evidence, runtime post-processing sensitivity, five-task coverage, and on-device latency characterization.

## 4.2 Related Work

需要做的修改：

1. 减少 DETR、Swin、Oriented R-CNN 等偏算法文献的篇幅。
2. 增加三类相关工作：
	- Android / edge deployment of YOLO models
	- construction-site PPE detection
	- mobile inference frameworks and consistency verification
3. 明确差异不是“检测精度更高”，而是“部署一致性诊断和真实 Android 工程闭环”。

## 4.3 Methods / Architecture

需要做的修改：

1. 保留 Java-JNI-C++-NCNN 四层结构。
2. 把 YOLO26 统一改成 “YOLO-based assets” 或 “project-managed yolo26n assets”。
3. 公式保留，但口吻降低：不要说这些机制是通用最优方法，而说是 runtime consistency mechanisms used in the SafeHat implementation。
4. 每个参数后加一句：参数选择来自部署验证，后续通过新增敏感性小节给出最小验证。

## 4.4 Diagnosis Section

这个部分是文章最有特色的地方，建议保留，但要降低“方法论”口吻。

修改方向：

- 从 “Proposed Consistency Diagnosis Method” 改成 “Deployment Consistency Diagnosis Workflow”。
- 强调该流程是工程诊断 workflow，不是通用 fault diagnosis theory。
- 四个 fault mode 要保留，因为这是文章最真实的 evidence。

## 4.5 Results and Discussion

需要新增或调整：

1. 新增参数敏感性小节。
2. 若补了第二台设备，新增 cross-device sanity check。
3. 若补了 Vulkan 或 runtime 对比，新增 backend note。
4. 原 limitation 提前说清楚，避免审稿人觉得作者过度声称。

## 4.6 Conclusion

结论要收缩：

- 不要说“provides a reproducible path for smart-city edge AI systems”太大。
- 改成“provides an implementation-level feasibility baseline and diagnosis workflow for low-cost Android PPE monitoring and related YOLO-based mobile vision tasks”。

---

## 5. 文风和声称降级清单

全文搜索并替换或弱化以下表达。

### 5.1 建议替换词

| 当前表达 | 建议表达 |
|---------|----------|
| YOLO26 family | project-managed YOLO-based assets / yolo26n assets |
| proposed method | proposed deployment framework / workflow |
| fault diagnosis method | deployment consistency diagnosis workflow |
| generalizable framework | reproducible implementation framework |
| smart-city edge AI systems | construction-site and field-monitoring edge applications |
| quantitative fault diagnosis | quantitative deployment diagnosis / repair evidence |
| validates the method | demonstrates feasibility / provides implementation evidence |
| systematic | structured / reproducible |

### 5.2 必须避免的说法

不要暗示：

- YOLO26 是新的算法模型。
- 本文性能优于其他移动推理框架。
- 单台 Kirin 970 能代表所有 resource-constrained Android devices。
- 五个任务都有同等强度的 benchmark。
- post-processing 参数已经系统最优。

### 5.3 应主动承认的边界

建议在 Discussion 中明确写：

1. SafeHat detection 是最完整主案例。
2. 其他四个任务主要验证统一加载、切换和解析路径。
3. 当前性能结果主要来自 CPU path。
4. 如果只补一台设备，则结果是 feasibility baseline，不是广泛泛化结论。
5. 参数敏感性是最小部署验证，不是完整 ablation study。

---

## 6. Applied Sciences 投稿前检查表

## 6.1 必须完成

- [ ] 改标题，弱化 YOLO26。
- [ ] 改摘要，删除“future ablation reserved”这类主动暴露弱点的句子。
- [ ] Introduction 重写 contribution。
- [ ] Related Work 增加 PPE monitoring / Android deployment 文献。
- [ ] 新增 SafeHat post-processing 参数敏感性小表。
- [ ] Discussion 明确 single-device / CPU-path / SafeHat-main-case 边界。
- [ ] 检查所有图表编号，新增表格后重新排序。
- [ ] 检查 References 是否有 Applied Sciences 偏好的最新应用文献。

## 6.2 强烈建议完成

- [ ] 增加第二台 Android 设备的 detection/classification latency。
- [ ] 跑一次 NCNN Vulkan 或说明 Vulkan 未纳入本轮实测的原因。
- [ ] 在 GitHub README 中补充最小复现实验命令。
- [ ] 确认 Zenodo DOI 可访问，且仓库版本与论文一致。

## 6.3 可选增强

- [ ] 补 3-5 张 hard-negative 示例图。
- [ ] 增加一张 deployment diagnosis workflow 与实际脚本对应关系图。
- [ ] 增加 Supplementary Table：logcat tags、diagnosis scripts、expected outputs。
- [ ] 对 Table 10 latency 增加 standard deviation。

---

## 7. 工作量时间表（两阶段，10-14 天）

**重要提醒**：上一版的“5 天”估时严重低估，实际按学术写作正常节奏需要 10-14 天。下面按两阶段拆分。

### 第一阶段：定位重写 + 参数敏感性（约 1 周）

**Day 1-2：定位重写**

任务：

1. 改标题、摘要、关键词。
2. 重写 Introduction 的最后 3-5 段。
3. 重写 contributions。
4. 全文替换过强表述（参考 §5.1）。
5. Methods 章节 YOLO26 全部改为 “project-managed yolo26n assets”。

产出：

- 新标题、新摘要、新 contributions
- 降低声称后的 Introduction 和 Methods

**Day 3-5：参数敏感性实验（核心工作量）**

任务：

1. 写离线评测脚本：把 logcat 帧或代表性图片喂回 Python 端，按参数重算 FP_empty / frac_gt090 / TP_keep。
2. 冻结 20-50 张测试集，三类比例固定。
3. 至少跑 A1（T）+ A2（gamma）完整数据，时间允许再补 A3+A4。
4. 生成 Table 10。
5. 新增 5.3 小节并写分析段落。
6. 评测脚本提交到 GitHub 仓库。

产出：

- 参数敏感性表（完整数值，无半成品）
- 评测脚本入库
- 5.3 小节文字

### 第二阶段：补充证据 + 文献 + 投稿包（约 1 周）

**Day 6-7：第二台设备（可选）+ Limitations 文字**

任务：

1. 如果第二台 Android 手机可立即接入且 1 天能跑通，做 detection/classification 跑通。
2. 否则直接按 §3.2 的“单设备 + 干净承认”模板写 Limitations。
3. 实验 C 不做，按 §3.3 模板写 Limitations 中的跨框架和 Vulkan 段落。

产出：

- cross-device sanity check 表（如果做了）
- 干净的 Limitations 段落

**Day 8-9：Related Work 和 Discussion 重写**

任务：

1. 找 PPE 监测、Android YOLO 部署的 2024-2026 最新文献（3-5 篇足够）。
2. 删除或压缩 DETR、Swin、Oriented R-CNN 等偏算法背景。
3. 重写 Discussion 和 Conclusion，明确边界。

产出：

- 更贴 Applied Sciences 的 Related Work
- 收紧的 Discussion 和 Conclusion

**Day 10-12：模板转换 + 图表 + 润色**

任务：

1. 转成 Applied Sciences / MDPI 模板。
2. 检查所有图表编号、引用、表名重新对齐。
3. 检查参考文献格式。
4. 英文润色一遍（重点是新加的 5.3 小节和重写的 Discussion）。

**Day 13-14：投稿包**

任务：

1. 准备 cover letter（参考 §8）。
2. 确认 Data Availability、Code Availability、Conflicts of Interest。
3. 确认 Zenodo DOI 与论文版本一致。
4. Highlights / Graphical Abstract（如期刊要求）。
5. Suggested reviewers（如果系统要求）。

产出：

- 可投稿版本
- cover letter
- submission checklist

### 时间表风险说明

上述 10-14 天假设：

- 离线评测脚本能在 2 天内写出来（如果工程基础不熟，可能要 3-4 天）。
- 没有大规模返工。
- 英文润色不外包。

现实里如果触发任一返工，整体周期可能拉到 3 周。建议规划时按 3 周预留缓冲。

---

## 8. Cover Letter 核心说法

### 是否提及 Electronics 拒稿：judgment call

两种做法都合理，需要自己权衡：

- **不提（默认保守）**：Electronics 是 desk reject，未送审，不构成必须 disclose 的同行评议结果。直接当全新投稿处理，避免给编辑负面锚定印象。
- **主动提一句**：MDPI 集团内部存在 transfer 机制，主动说明“previously submitted to Electronics and returned without external review”反而显得诚实，有时编辑会直接把稿子按 transfer 走，减少送审延迟。

建议默认走“不提”，除非投稿系统明确询问是否曾投过 MDPI 其他期刊。

### 建议强调：

> This manuscript presents an applied Android-NCNN deployment framework for YOLO-based PPE monitoring on resource-constrained mobile devices. The study focuses on deployment consistency, real-device diagnosis, and reproducible engineering validation rather than proposing a new detection architecture. The framework is validated through a SafeHat PPE monitoring case, five-task on-device runtime coverage, and quantitative before/after repair evidence for four deployment faults. The code and diagnosis scripts are publicly available with an archived DOI.

可以突出 Applied Sciences 匹配点：

- applied mobile computer vision
- construction-site safety monitoring
- edge AI deployment
- reproducible software implementation
- real-device validation

---

## 9. 风险评估

### 9.1 仍可能被拒的原因

1. 审稿人认为工程贡献不足，不构成学术论文。
2. 单设备结果仍然太弱。
3. 没有与 TFLite / MNN / ONNX Runtime Mobile 的严格对比。
4. YOLO26 命名仍让人误解为自创模型。
5. 参数敏感性不够完整。

### 9.2 对策

| 风险 | 最小对策 |
|------|----------|
| 工程贡献不足 | 强调 open-source framework、diagnosis workflow、before/after repair evidence |
| 单设备 | 至少补一台设备的 detection/classification |
| 无框架对比 | 增加 NCNN CPU/Vulkan 或在 limitation 中明确不做跨框架性能主张 |
| YOLO26 误解 | 标题和摘要中弱化，只在方法中解释为 project-managed assets |
| 参数经验化 | 补最小 sensitivity table |

---

## 10. 真正的最小可投版本（再砍一刀）

如果时间或资源紧张，**实际最最小版本只需要这三件事**：

1. **改标题 + 摘要 + contributions**：弱化 YOLO26，定位为 PPE monitoring 应用部署系统。零实验成本。
2. **补 T 和 gamma 两组敏感性表**（A1 + A2，约 24 个数字）：必须完整数值，不要半成品。其他参数在 Methods 末尾一句话承认 selected based on deployment validation。
3. **重写 Discussion / Limitations**：单设备、CPU path、SafeHat 主案例、五任务 coverage、未做跨框架对比——全部边界一次性说清楚。

\u00a73.2 第二台设备和 \u00a73.3 后端对比都是 nice-to-have，不是必需。

### 投稿概率的诚实判断

完成上述三项后：

- Applied Sciences 进入送审环节的概率**明显高于** Electronics（因为期刊定位口径更宽，且 desk reject 风险点已经被针对性处理）。
- 但**不能保证一审通过**，大概率仍会有一轮 major revision，可能的反馈点：
  - 要求补充更多设备
  - 要求与 TFLite / MNN 做严格对比
  - 要求扩展 ablation 覆盖所有参数
  - 要求把 YOLO26 命名进一步澄清
- **不能排除 desk reject**：Applied Sciences 近两年标准在升，纯部署案例稿仍可能被刷。

### 现实预期

这篇稿子的本质是工程实现 + 故障诊断闭环 + 单案例验证。即使按本方案改完，**它依然是一篇工程应用型论文，不会变成强 novelty 稿**。本方案只是把它放到合适的赛道，而不是改变它的级别。

如果 Applied Sciences 仍被拒，下一站可考虑 **SoftwareX**（强调开源代码与可复现实现）或 **PeerJ Computer Science**（对工程性 case study 友好），不再回头投 AI 算法导向的期刊。
