# NCNN/YOLO 集成与测试实战手册（Android）

本手册用于快速完成：
- 自训练 YOLO26 模型导出并接入 Android NCNN；
- 运行期验证输出分支是否正确；
- 定位“满屏框/框爆炸”类问题。

---

## 1) 先决条件

1. Android Gradle Plugin 8.7.3 需要 **JDK 11+**（建议 JDK 17）。
2. 已正确放置 ncnn 与 opencv-mobile 到 JNI 目录。
3. 模型文件位于 assets 且命名与加载逻辑一致。

当前代码中的检测模型默认文件名由 JNI 拼接得到：
- param: `yolo26n_e2e.ncnn.param`
- bin: `yolo26n_e2e.ncnn.bin`

参考实现：
- [app/src/main/jni/yolo26ncnn.cpp](../app/src/main/jni/yolo26ncnn.cpp#L202-L203)

---

## 2) 导出与文件放置

### A. E2E（One-to-One，通常无需 NMS）

```python
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="ncnn", end2end=True)
```

期望输出形状（运行时日志）：
- `w=6, h=300`（每个检测 6 个值）

### B. O2M（One-to-Many，需要 NMS）

```python
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="ncnn", end2end=False)
```

期望输出形状（运行时日志）：
- 常见 `w=8400, h=84`（或转置后 `w=84, h=8400`）

将导出的 `.param/.bin` 复制到：
- [app/src/main/assets](../app/src/main/assets)

---

## 3) Android 侧关键验证点

### 3.1 输出分支判定

检测逻辑已兼容多分支，优先看日志：
- `Output shape: w=... h=... c=...`

参考实现：
- [app/src/main/jni/yolo26_det.cpp](../app/src/main/jni/yolo26_det.cpp#L670)
- [app/src/main/jni/yolo26_det.cpp](../app/src/main/jni/yolo26_det.cpp#L677-L678)

### 3.2 阈值联调（UI -> JNI -> C++）

检测阈值可动态调节：
- `prob_threshold`（置信度）
- `nms_threshold`（NMS 阈值）

关键调用链：
- [app/src/main/java/com/tencent/yolo26ncnn/MainActivity.java](../app/src/main/java/com/tencent/yolo26ncnn/MainActivity.java#L57-L58)
- [app/src/main/jni/yolo26ncnn.cpp](../app/src/main/jni/yolo26ncnn.cpp#L257-L265)

---

## 4) “满屏框”快速排查（优先顺序）

### Step 1: 看三条日志

1. `setDetectThresholds prob=... nms=...`（确认阈值已下发）
2. `Total detected: ...`（阈值后候选数）
3. `After NMS: proposals=... picked=... agnostic=...`

对应实现：
- [app/src/main/jni/yolo26ncnn.cpp](../app/src/main/jni/yolo26ncnn.cpp#L265)
- [app/src/main/jni/yolo26_det.cpp](../app/src/main/jni/yolo26_det.cpp#L546)
- [app/src/main/jni/yolo26_det.cpp](../app/src/main/jni/yolo26_det.cpp#L780)

### Step 2: 判定问题类型

- 若 `Total detected` 始终数千（即使 prob 提高）：
  - 常见是训练/导出侧置信度分布异常，单纯调阈值不够。
- 若 `picked` 常年顶到上限：
  - 属于后处理压力过大，需进一步压制候选。

### Step 3: 当前工程内的抑制策略

已内置以下防刷屏策略：
- pre-NMS top-k: 300
- max_det: 50
- 过密场景自动 `agnostic NMS`
- 几何过滤：极小框 / 面积过小 / 极端长宽比

参考代码：
- [app/src/main/jni/yolo26_det.cpp](../app/src/main/jni/yolo26_det.cpp#L762-L772)
- [app/src/main/jni/yolo26_det.cpp](../app/src/main/jni/yolo26_det.cpp#L777-L780)

建议现场调参起点：
- `prob = 0.80 ~ 0.90`
- `nms = 0.30 ~ 0.40`

---

## 5) 标准测试流程（建议每次模型替换都执行）

1. **Python 侧 sanity check**：
	- 同一张图分别跑 `.pt` 与导出的 `ncnn_model`，确认目标数量量级一致。
2. **Android 启动日志检查**：
	- 记录 `Output shape`、`Total detected`、`After NMS`。
3. **阈值滑条回归**：
	- 拉高 prob 后，`Total detected` 应明显下降。
4. **三类场景截图**：
	- 稀疏场景 / 中密度 / 高密度，比较误检与漏检。
5. **性能记录**：
	- CPU 与 GPU 各测一次，记录 FPS 与稳定性。

---

## 6) 常见失败与处理

### 构建失败：JDK 版本过低

若出现 “Dependency requires at least JVM runtime version 11. This build uses a Java 8 JVM.”：
- 将 Gradle JDK 切换到 17（或至少 11）；
- 命令行构建前确认 `java -version` 不是 1.8。

### 模型能加载但检测异常

优先检查：
- 类别数与标签表是否一致；
- 导出模式是否与运行分支匹配（E2E vs O2M）；
- 训练集标注质量（错标/漏标/类别污染）。

---

## 7) 建议提交问题时附带的信息

请至少附：
1. 一段包含三条关键日志的 logcat；
2. 当前 `prob/nms` 值；
3. 一张原图 + 一张检测截图；
4. 模型导出命令（含 `end2end` 参数）；
5. 输出形状（`w/h/c`）。

---

## 8) 本地实机验证记录（2026-02-28）

已在真机上完成一次完整验证（安装 -> 启动 -> 抓日志）：

- 阈值下发生效：`setDetectThresholds prob=0.85 nms=0.35`
- 输出分支：`Output shape: w=8400 h=84 c=1`（O2M）
- 高密度场景下后处理生效：
	- `Post-process info ... pretrim=3508~5811 topk=120`
	- `After NMS ... proposals=120 picked=30 agnostic=1`

	---

	## 13) 训练侧审计结论（YOLOv11 工程）

	对训练工程（`E:/projects/AndroidPro/YOLOv11`）做了直接审计，结论如下：

	1. 数据集配置与标签质量正常
		- `nc=10` 与 `names` 数量一致；
		- 标签扫描无坏行、无越界。

	2. 之前“全是 100%”并非单一训练崩坏结论
		- 训练侧 `best.pt` 在验证集抽样统计，置信度并非全部接近 1；
		- 但在手机实时场景中，仍可能出现大量高分候选（域偏移 + 后处理压力）。

	3. 已完成落地修复
		- 使用训练工程 `best.pt` 重新导出并覆盖 Android assets；
		- C++ 已支持 `10` 类 O2M 分支正确解析（`h=14`）；
		- 绘制文本改为两位小数，降低 `100.0%` 视觉误判；
		- 10 类模型使用 SafeHat 类名，而非 COCO 80 类名。

	关键文件：
	- [app/src/main/jni/yolo26_det.cpp](../app/src/main/jni/yolo26_det.cpp)
	- [scripts/export_best_safehat_to_assets.py](../scripts/export_best_safehat_to_assets.py)
	- [scripts/preflight_safehat_pipeline.py](../scripts/preflight_safehat_pipeline.py)

	---

	## 14) 已执行的“置信度校准微调”与结果

	在训练工程执行了短程微调（8 epochs）：
	- 基础模型：`E:/projects/AndroidPro/YOLOv11/best.pt`
	- 结果目录：`E:/projects/AndroidPro/YOLOv11/runs/detect/runs/calib/safehat_confcal_v1`
	- 关键超参：较低 `lr0`、`cos_lr`、较高 `weight_decay`、`warmup`

	随后已将该微调权重导出并覆盖 Android assets。

	实机日志确认：
	- 输出仍为 10 类 O2M：`w=8400 h=14`；
	- `model_prob` 已呈现明显分布（低/中/高都有），不再是“单一全100%”的假象；
	- 仍会在部分帧出现高分噪声，这是场景域偏移 + O2M 候选密度导致，需继续通过训练数据与采样策略优化。

结论：
- JNI/C++ 新策略已正确生效（不是旧包缓存）；
- 但模型端仍表现为“高分候选过密”，后处理只能做工程止损，无法替代训练质量修复。

---

## 9) 关键注意：JNI改动后如何确保真机拿到新so

若你发现日志仍是旧值（例如 `topk=300`），请执行“强制原生重编译”再安装：

- 先执行 `externalNativeBuildCleanDebug`
- 再执行 `externalNativeBuildDebug` + `assembleDebug`
- 然后重新安装 APK

否则可能出现 Java 更新了、但旧 native so 仍在包里的情况。

---

## 10) 训练/导出侧根因预检（已落地脚本）

新增脚本：
- [scripts/preflight_safehat_pipeline.py](../scripts/preflight_safehat_pipeline.py)

它会自动检查：
1. `safehat.yaml` 的 `nc` 与 `names` 是否一致；
2. 训练/验证/测试标签是否有越界或坏行；
3. assets 中 NCNN 检测模型真实输出形状；
4. 模型类别数是否与数据集一致。

本次实测的关键根因信号：
- 旧 assets 模型输出 `w=8400 h=84` => 推断 `80` 类；
- 数据集是 `nc=10`；
- 属于“部署模型与训练数据类别不一致”，高风险导致误检泛滥。

---

## 11) 一键导出 best.pt 到 Android assets（已落地脚本）

新增脚本：
- [scripts/export_best_safehat_to_assets.py](../scripts/export_best_safehat_to_assets.py)

默认行为：
- 从 `runs/detect/runs/train/yolo26n_safehat/weights/best.pt` 导出 NCNN；
- 默认导出 O2M（`end2end=False`）；
- 自动覆盖 `app/src/main/assets/yolo26n_e2e.ncnn.*`（沿用当前 JNI 加载路径）；
- 自动打印新模型输出形状。

当前实测：
- 新 assets 模型输出已变为 `w=8400 h=14`；
- 对应 `10` 类（`14 = 4 + 10`），已与数据集一致。

---

## 12) 代码侧兼容修复（自定义类别数 O2M）

当 `feature_dim=14`（10类）时，必须走 YOLO26 无DFL分支，而不是 YOLO26 Legacy DFL 分支。

已修复于：
- [app/src/main/jni/yolo26_det.cpp](../app/src/main/jni/yolo26_det.cpp)

修复后日志应出现：
- `Output shape: w=8400 h=14 c=1`
- `Using YOLO26 processing (no DFL, transposed format, custom classes supported)`
- `inferred_classes=10`

---

## 15) 第二轮校准微调（v2）与实机回归（2026-02-28）

已继续完成第二轮训练 -> 导出 -> 部署 -> 日志回归闭环：

1. 训练输出
	- 目录：`E:/projects/AndroidPro/YOLOv11/runs/calib/safehat_confcal_v2`
	- 验证集摘要：`P=0.824, R=0.739, mAP50=0.790, mAP50-95=0.470`

2. 导出与替换
	- 已将 `safehat_confcal_v2/weights/best.pt` 导出为 NCNN；
	- 已覆盖 Android assets；
	- 形状校验：`w=8400 h=14 c=1`。

3. 实机日志（debug）
	- `Output shape: w=8400 h=14 c=1`（类别与分支正确）；
	- `model_prob` 已有低/中/高分布（如 `0.005 ~ 0.999`）；
	- `Total detected` 仍常见 `3.7k ~ 6.8k`；
	- `After NMS` 约 `21 ~ 30`，后处理控制稳定。

结论：
- v2 已确认集成正确且可运行；
- 仍存在“高分候选过密”场景，下一步应优先做 hard negatives（无目标/反光/密集背景）与误标清洗，而非继续单纯增加 epoch。

---

## 16) 清单实跑结论（已执行，2026-03-01）

你授权后，已按“转换/部署排查清单”逐项实跑，关键结论如下：

1. 训练数据与标签
	- `nc=10` 与 `names=10` 一致；
	- 标签扫描：`malformed=0`、`out_of-range=0`。

2. NCNN导出与类别一致性
	- assets 检测模型输出：`w=8400 h=14 c=1`；
	- 由输出反推类别数 `14-4=10`，与数据集一致。

3. PT vs NCNN 一致性（同批验证图）
	- `PT`: min=2 / med=5 / max=61 / avg=10.0
	- `NCNN`: min=2 / med=5 / max=61 / avg=10.0
	- 逐图均值置信度平均：`PT=0.7092`, `NCNN=0.7092`

判定：
- **自训练模型“转NCNN后语义错误”已排除**（输出语义与数量分布一致）；
- 当前剩余问题属于**部署场景域偏移导致的高分候选过密**，不是“转换坏了”。

4. 已落实的工程侧收敛措施（本次新增）
	- 默认阈值提升：`prob=0.90`, `nms=0.30`；
	- 极密场景分级压制：`topk=60/90/120/250`，`max_det=20/25/30/50`；
	- 极密场景几何过滤更严格：最小面积和长宽比上限动态收紧。

5. 真机回归（强制原生重编后）
	- 日志确认新参数已生效：`setDetectThresholds prob=0.90 nms=0.30`；
	- `pretrim` 很高时已触发 `topk=60/90`；
	- `After NMS` 稳定在约 `10~25`，明显收敛。

---

## 17) 转换链路逐项取证结论（2026-03-01）

本节只回答一个问题：**是否是 PT -> NCNN 转换导致 Android 异常**。

### 17.1 同一模型文件一致性（源码 assets vs APK 内 assets）

已对 `yolo26n_e2e.ncnn.param/.bin` 做 MD5 比对：
- 源码 assets 与 APK assets 完全一致（param/bin 均相同）。

判定：
- 不是“打包拿错模型”或“APK 未更新模型文件”。

### 17.2 运行时精度与打包布局排查

在 [app/src/main/jni/yolo26.cpp](../app/src/main/jni/yolo26.cpp) 中已禁用：
- `use_fp16_packed`
- `use_fp16_storage`
- `use_fp16_arithmetic`
- `use_bf16_storage`
- `use_packing_layout`

并在 [app/src/main/jni/yolo26_det.cpp](../app/src/main/jni/yolo26_det.cpp#L672) 打印：
- `elempack`
- `elembits`
- `cstep`

实机日志确认：
- `elempack=1`, `elembits=32`。

判定：
- 不是“FP16/pack 读取错误”导致的分数异常。

### 17.3 相机链路隔离：离线合成图探针

在 [app/src/main/jni/yolo26ncnn.cpp](../app/src/main/jni/yolo26ncnn.cpp) 增加离线探针（合成图，非相机帧）：
- `OfflineProbe: begin synthetic frame detect`
- `OfflineProbe: end synthetic frame detect ...`

离线探针实机统计（Android）：
- `p50≈0.512`
- `p90≈0.9999`
- `p99=1.0000`
- `frac_gt090≈0.2698`

对同一合成图在 PC 端（同一 param/bin）复算：
- `p50≈0.5345`
- `p90=1.0`
- `p99=1.0`
- `frac_gt090≈0.2054`

判定：
- 同输入下 PC 与 Android 分布同量级，**转换语义一致**。
- 先前“PC 低分、Android 高分”的冲突，根因是输入样本不一致（验证集静态图 vs 实时相机/合成图），不是转换链路损坏。

### 17.4 最终结论

1. `PT -> NCNN` 转换链路未发现语义损坏证据。
2. 当前问题主因不在格式转换；属于输入域差异下模型对非训练分布的高置信响应（候选过密）。
3. 后续治理重点：
	- 采集并加入 hard negatives（无目标、反光、纹理噪声、密集背景）；
	- 增加离线固定图回归集（Android 与 PC 同图同口径）；
	- 继续用 `RawScoreStats` 做版本对比，避免“感觉式”回归。

---

## 18) 下一步落地：Hard Negatives 回灌 + v3 快速微调

已新增两份可直接执行的脚本：

1. hard negatives 挖掘
	- [scripts/mine_hard_negatives.py](../scripts/mine_hard_negatives.py)

2. v3 快速微调
	- [scripts/quick_finetune_confcal_v3.py](../scripts/quick_finetune_confcal_v3.py)

### 18.1 先挖 hard negatives（自动空标签）

示例：

```bash
python scripts/mine_hard_negatives.py \
	--model runs/calib/safehat_confcal_v2/weights/best.pt \
	--images-root data/train/images \
	--labels-root data/train/labels \
	--output-root data/hardneg_pool/v3 \
	--min-max-conf 0.75 --min-count 2 --topk 600 --inject-train
```

产物：
- `data/hardneg_pool/v3/images`（挑出的误检图）
- `data/hardneg_pool/v3/labels`（对应空标签）
- `data/hardneg_pool/v3/hardneg_report.json`
- `data/hardneg_pool/v3/hardneg_report.md`

若显存较小（如 4GB）出现 OOM，建议加：
- `--device cpu --batch 2`

若带 `--inject-train`，会自动注入：
- `data/train/images/hardneg_v3`
- `data/train/labels/hardneg_v3`

### 18.2 再跑 v3 短程微调

示例：

```bash
python scripts/quick_finetune_confcal_v3.py \
	--base-model runs/calib/safehat_confcal_v2/weights/best.pt \
	--data-yaml scripts/safehat.yaml \
	--epochs 12 --batch 4 --imgsz 640
```

可选：训练后直接覆盖 Android assets：

```bash
python scripts/quick_finetune_confcal_v3.py \
	--base-model runs/calib/safehat_confcal_v2/weights/best.pt \
	--data-yaml scripts/safehat.yaml \
	--epochs 12 --batch 4 --imgsz 640 \
	--export-assets
```

### 18.3 回归判定标准（建议）

1. Android `RawScoreStats`：
	- `frac_gt090` 相比 v2 明显下降；
	- `p50` 下移但真实目标仍可检出。

2. Android 业务指标：
	- `Total detected` 中位数下降；
	- `After NMS picked` 保持稳定（不明显漏检）。

3. PC/Android 同图一致性：
	- 固定离线图集上分布变化方向一致。

---

## 19) 针对“识别率还是 100%”的运行期修正（2026-03-01）

已在推理后处理中加入概率温度校准（O2M 分支）：
- 新增 `calibrate_probability()`，对模型概率做保序压缩；
- 阈值筛选改为使用 `cal_prob`；
- 绘制文本做上限显示（`99.90%`），避免界面长期显示 `100.00%`。

关键代码：
- [app/src/main/jni/yolo26_det.cpp](../app/src/main/jni/yolo26_det.cpp#L214)
- [app/src/main/jni/yolo26_det.cpp](../app/src/main/jni/yolo26_det.cpp#L530)
- [app/src/main/jni/yolo26_det.cpp](../app/src/main/jni/yolo26_det.cpp#L1017)

同时默认阈值从 `0.90` 提升到 `0.93`：
- [app/src/main/java/com/tencent/yolo26ncnn/MainActivity.java](../app/src/main/java/com/tencent/yolo26ncnn/MainActivity.java#L58)
- [app/src/main/jni/yolo26.cpp](../app/src/main/jni/yolo26.cpp#L20)

实机回归（同场景）显示：
- `setDetectThresholds prob=0.93 nms=0.30` 已生效；
- `Total detected` 典型值下降到约 `1.8k~2.6k`（个别帧更高）；
- `After NMS picked` 维持在可用范围（约 `11~30`）。

说明：
- 这一步主要解决“显示长期 100% + 候选过密”体验问题；
- 根因仍是场景域偏移，后续需继续依赖 hard negatives + 完整 v3 训练收敛。

