
---

### 一、最高优先级：实验部分证据不足，审稿人大概率会要求补数据

这是当前文稿被拒的最大风险点。

**问题 1：第 5 节没有一张填了真实数字的表。** 所有表都还是占位状态（"Table 7 here""Table 9 here"）。Electronics 是工程型期刊，审稿人对"有框架无数据"的文章容忍度很低。你不需要做 COCO 级别的大规模 benchmark，但至少要有三组硬数据：

| 必须补的表 | 数据来源 | 你现在已有的基础 |
|-----------|---------|----------------|
| **修复前后对比表**（最核心） | Bugfix.md 里已经有完整的分数统计：`p50=0.84~0.96, frac_gt090=0.41~0.58`，修复后恢复正常 | 直接搬，只需格式化成表 |
| **五任务功能验证表** | 逐任务记录：模型文件名、输出 shape、是否正确渲染、是否需要 NMS | 你的代码里五个 cpp 文件都在，跑一遍截图 + 记录即可 |
| **SafeHat 训练与部署关键参数表** | safehat.yaml 10 类、quick_finetune_confcal_v3.py epochs=12, batch=4, imgsz=640 | 已有，只需要再补一组 mAP/precision/recall |

**问题 2：运行效率没有任何数字。** 你的代码里其实已经有 FPS 计数器（yolo26ncnn.cpp，10 帧滑动平均），只要在真机上跑每种任务时截取 FPS 读数就行。哪怕只报一个设备上的 CPU 模式 FPS，也比现在"intentionally avoids claiming"要好得多。Electronics 审稿人不会因为你只测了一台设备就拒你，但会因为你一个数字都没有而质疑"这个框架到底能不能用"。

**建议行动：** 在手机上跑 5 种任务各 30 秒，记录屏幕上显示的平均 FPS，再用 Ultralytics 的 `model.val()` 跑一次 SafeHat 验证集拿到 mAP50。这两组数据加上 Bugfix 里已有的分数统计，就够撑起第 5 节。

---

### 二、高优先级：文稿结构和表述需要进一步收紧

**问题 3：文首和文尾残留了不属于论文正文的内容。** 文档开头的"按 Electronics 风格，建议把全文收束成 6 个正式章节……"和文末的"直接替换建议"段落是写作过程中的工作备注，不是论文内容。正式投稿前必须删除。

**问题 4：摘要偏长。** 当前摘要三段加起来大约 280 个英文单词。Electronics 的摘要一般建议 200 词以内（MDPI 官方模板要求 "no more than 200 words"）。现在第二段里五个故障类型全部在摘要里展开，太细了。建议压缩成一句："Several critical deployment inconsistencies are identified and systematically resolved, including preprocessing mismatch, redundant activation, coordinate re-decoding, and output-layout misinterpretation."

**问题 5：Keywords 可以更精准。** 现在的关键词是 `YOLO26; NCNN; Android; on-device inference; deployment debugging; multi-task visual inference`。建议把 `deployment debugging` 改成 `deployment consistency`，把 `on-device inference` 改成 `edge inference`，这两个词在 Electronics 的检索里更常见。另外可以加一个 `model conversion`，因为这是你文章核心故障的来源之一。

**问题 6：引言里 contributions 的表述可以更硬。** 现在的三条贡献都是"构建了""建立了""总结了"，缺少动词后面的"效果量化"。即使不给精确数字，也可以加限定：
- "...supporting five heterogeneous visual tasks on a single Android device"
- "...covering the full path from fine-tuning to on-device regression with a 10-class PPE dataset"
- "...resolving four categories of deployment inconsistencies that rendered the system unusable before correction"

---

### 三、中等优先级：技术细节的严谨性

**问题 7：第 4 节第五个故障（跨平台兼容性）在 Electronics 里显得偏弱。** 前四个故障都有明确的"技术根因→修复→效果验证"逻辑链，但第五个"Windows IDE 报未定义符号"本质上是开发环境配置问题，不是部署一致性问题。审稿人可能会质疑它的学术价值。我建议两种处理方式：
- **方案 A**：把它从正文降级到第 5 节实验环境描述里，用一句话带过。
- **方案 B**：保留但弱化篇幅，从目前的两段压缩成一段，并把小节标题从平行并列改成"此外"式过渡，明确它是辅助性发现而非核心故障。

**问题 8：第 2 节 Related Work 里缺少"移动端部署研究"这个细分方向的文献引用。** 你现在引了 YOLO 系列、DETR 系列、四个推理框架、Android NDK 官方文档，但没有引任何一篇"把 YOLO 部署到手机上"的已有工作。Electronics 审稿人会问："别人有没有做过类似的事？你和他们的区别是什么？" 建议至少补 2-3 篇移动端 YOLO 部署或 edge AI 部署的论文，然后在 Related Work 末尾加一段"gap analysis"，明确说现有工作大多关注模型压缩或精度对比，而较少系统讨论部署链路中的一致性问题和诊断方法。

**问题 9：第 3 节 Framework 部分缺一个关键的技术细节——预处理的具体实现。** 你的文章核心故障之一就是预处理不一致，但第 3 节在描述系统设计时没有明确写出"预处理采用 letterbox padding 到 640×640 正方形输入"这个规范。建议在 3.1 或 3.2 里加一段，把你最终确定的正确预处理流程写清楚，这样第 4 节的故障分析才有对照基准。

---

### 四、投稿格式层面

**问题 10：双语写法不适合直接投 Electronics。** Electronics 是纯英文期刊，投稿时只需要英文正文。中文段落在写作阶段可以保留作为参考，但最终投稿版必须删除所有中文内容，只保留英文。建议你现在继续用双语版做打磨，但最后导出投稿版时要有一个"去中文"的步骤。

**问题 11：Electronics 使用 MDPI 模板。** MDPI 有自己的 LaTeX 和 Word 模板，章节编号、参考文献格式、图表标注方式都有具体要求。你现在这个 Markdown 只是内容底稿，最终需要转到 MDPI 模板里。几个要注意的点：
- 参考文献要用 MDPI 自己的格式（不完全是 GB/T 7714，也不是标准 IEEE），一般由模板自动处理
- 图表必须有英文 caption，格式为 "Figure 1. xxx" / "Table 1. xxx"
- 作者信息、Funding、Data Availability Statement、Conflicts of Interest 等都是必填项

**问题 12：图的数量偏多。** 当前文稿标了 Figure 1 到 Figure 10，共 10 张图。对一篇 Electronics 文章来说偏多了，通常 6-8 张比较合理。建议合并或删减：
- Figure 5（预处理修复前后对比）和 Table 5（分数统计）可以只保留一个，因为它们说明的是同一个问题
- Figure 8（跨平台兼容性）可以删，改用文字描述即可
- Figure 9（五任务运行结果）如果做成 2×3 子图拼接，可以和 Figure 10（SafeHat 结果）合并成一张

---

### 五、提升竞争力的可选建议

**问题 13：如果你能补一组 CPU vs Vulkan 的 FPS 对比，文章的实验价值会明显提升。** 你的代码里已经有 Vulkan 支持（yolo26.cpp），只需要分别在 CPU 模式和 GPU 模式下各跑一轮 FPS。即使只测 detection 一种任务、一台设备，也能构成一张有说服力的小表。

**问题 14：可以考虑把诊断方法抽象成一张 decision tree 或 flowchart。** 你现在第 4 节开头提了五阶段诊断流程，但正文里是线性叙述。如果把它画成一张"症状→排查路径→根因定位"的流程图（就是 Figure 4 的位置），会让审稿人更直观地看到"这不是零散修 bug，而是有方法论的"。

**问题 15：标题可以微调。** 当前标题 "Deployment and Diagnostic Study of YOLO26-Based Multi-Task Visual Inference on Android Devices" 里 "Study" 这个词在 Electronics 里偏中性、偏综述感。建议改成更有动作感的词：
- "...Multi-Task Visual Inference **Framework** for Android Devices"
- 或 "...Deployment **Framework** and Failure Diagnosis for..."

这样更像在报告一个系统贡献，而不是一个观察性研究。

---

### 总结：按优先级排的行动清单

| 优先级 | 行动 | 预计工作量 |
|-------|------|-----------|
| **P0** | 补真机 FPS 数据（5 任务各跑 30 秒读 FPS） | 半小时 |
| **P0** | 补 SafeHat mAP50（跑一次 `model.val()`） | 几分钟 |
| **P0** | 把 Bugfix.md 里的分数统计格式化成正式表格填入第 5 节 | 半小时 |
| **P1** | 压缩摘要到 200 词以内 | 20 分钟 |
| **P1** | 删除文首工作备注和文末"直接替换建议" | 5 分钟 |
| **P1** | 补 2-3 篇移动端 YOLO 部署相关文献到 Related Work | 1 小时 |
| **P1** | 把第五个故障（IDE 兼容性）降级处理 | 15 分钟 |
| **P2** | 合并/精简图到 6-8 张以内 | 30 分钟 |
| **P2** | 补 CPU vs Vulkan FPS 对比（如环境允许） | 1 小时 |
| **P2** | 把诊断流程画成 decision-tree 图 | 1 小时 |
| **P3** | 转 MDPI LaTeX/Word 模板 | 2-3 小时 |
