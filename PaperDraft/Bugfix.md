# Bugfix: NCNN 非正方形输入导致分类分数爆炸

> **日期**: 2026-03-02  
> **影响范围**: 所有基于自定义数据集微调的 YOLO26/YOLO26 Legacy 模型部署到 Android (NCNN)  
> **严重程度**: Critical — 空场景下所有类别全部误检，应用完全不可用  
> **修复文件**: `app/src/main/jni/yolo26_det.cpp`

---

## 一、问题现象

将基于 SafeHat 数据集（10 类：Hardhat / Mask / No-Hardhat / No-Mask / No-Safety Vest / Person / Safety Cone / Safety Vest / Machinery / Vehicle）微调的 YOLO26n 模型以 NCNN 格式部署到 Android 手机后：

1. **空场景下 10 个类别全部被检出**，识别框随机漂浮在画面各处
2. 原始分数统计极度异常：`p50=0.84~0.96, frac_gt090=0.41~0.58`（8400 个 anchor 中超过一半分数 >0.9）
3. 尤其是 **最后 3~4 个类别通道**（Safety Vest / Machinery / Vehicle）分数飙升到 1.0

而**同一模型在 PC 端（PyTorch / Python-ncnn）运行完全正常**，检测准确、框位精准。

---

## 二、排查过程

### 2.1 最初误判：NCNN 导出精度问题

早期怀疑 NCNN 导出存在精度丢失。但用户指出 **COCO 原始模型（yolo26n 及其 Legacy 变体）使用相同导出流程转换后在手机上工作正常**，排除了通用导出问题。

### 2.2 第二次误判：训练不充分

曾怀疑 12 epoch 训练不够导致背景抑制能力不足。但用户明确指出 **训练好的模型在 PC 上检测完全正确**，说明模型本身的背景抑制能力没有问题。

### 2.3 全链路追踪定位

编写 `scripts/trace_conversion_pipeline.py` 对 **640×640 正方形空白图** 逐级验证：

| 阶段 | max_score | >0.50 数量 | 结论 |
|------|-----------|-----------|------|
| PyTorch (best.pt) | 0.004840 | 0 | ✅ 正常 |
| NCNN Ultralytics 导出 | 0.004841 | 0 | ✅ 正常 |
| NCNN PNNX 导出 | 0.004504 | 0 | ✅ 正常 |
| 已部署到 assets 的模型 | 0.002828 | 0 | ✅ 正常 |
| COCO yolo26n 基线 | 0.000209 | 0 | ✅ 正常 |

**所有阶段在 640×640 输入下全部正常！** 问题不在模型、不在导出、不在转换。

### 2.4 关键突破：非正方形输入假设

回溯发现：早期诊断脚本 `diagnose_background_scores.py` 使用 **480×640 非正方形图片** 时 NCNN 出现异常分数，而 `trace_conversion_pipeline.py` 使用 **640×640 正方形图片** 时一切正常。手机摄像头输出也是非正方形（1920×1080 或 1080×1920）。

编写 `scripts/test_letterbox_hypothesis.py` 系统验证：

| 预处理方式 | 640×640 | 480×640 | 1080×1920 | 1920×1080 |
|-----------|---------|---------|-----------|-----------|
| **直接 resize 到 640×640** | ✅ 0.003 | ✅ 0.003 | ✅ 0.003 | ✅ 0.003 |
| **letterbox 保持比例 (非正方形)** | ✅ 0.003 | ❌ 0.974 / 6115 | ❌ 0.977 / 6926 | ❌ 1.000 / 5422 |
| **letterbox 填充到 640×640** | ✅ 0.003 | ✅ 0.007 | ✅ 0.041 | ✅ 0.023 |

**结论：当 NCNN 模型接收非 640×640 的输入张量时，最后 3~4 个类别通道分数爆炸。**

---

## 三、根因分析

### 3.1 C++ 预处理代码（修复前）

```cpp
// yolo26_det.cpp — YOLO26_det::detect()
int w = img_w, h = img_h;
float scale = 1.f;
if (w > h) { scale = (float)target_size / w; w = target_size; h = h * scale; }
else       { scale = (float)target_size / h; h = target_size; w = w * scale; }

ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);

// ❌ BUG: 只 pad 到 32 的倍数，而不是 640×640
int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
ncnn::copy_make_border(in, in_pad, hpad/2, hpad-hpad/2, wpad/2, wpad-wpad/2, BORDER_CONSTANT, 114.f);
```

以手机横屏 1920×1080 为例：
- `scale = 640/1920 = 0.333`，缩放后 `w=640, h=360`
- `hpad = ceil(360/32)*32 - 360 = 384 - 360 = 24`
- **实际输入模型的张量尺寸：640×384（而非 640×640）**

### 3.2 为什么非正方形输入会导致分数爆炸

YOLO26 采用 anchor-free 架构，**没有显式的 objectness 分支**。背景抑制完全依赖分类头的 sigmoid 输出在负样本上被 BCE loss 推向 0。

训练时模型**只见过 640×640 正方形输入**。当推理时输入一个 640×384 的非正方形张量：

1. **特征图尺寸异常**：三个检测头输出的 anchor 网格与训练时不同
2. **位置编码错乱**：模型内部学到的空间先验在非正方形输入上失效
3. **最后几个通道最敏感**：可能与 concat 层的通道排列和卷积核的边界效应有关
4. sigmoid 输出不再被正确抑制，大量 anchor 的分类分数飙升到 0.9~1.0

### 3.3 为什么 COCO 原始模型不受影响

COCO 预训练模型（80 类）经过大规模数据、充分 epoch 训练，对各种输入尺寸的鲁棒性远高于 12 epoch 微调的小数据集模型。COCO 模型在非正方形输入上虽然也不理想，但分数不会爆炸到不可用的程度。微调模型的分类头是全新初始化的，只在 640×640 上训练了很少轮次，对输入尺寸极度敏感。

---

## 四、修复方案

### 4.1 代码修改（仅两行）

```cpp
// yolo26_det.cpp — YOLO26_det::detect()

// ✅ 修复: pad 到 target_size × target_size 正方形
int wpad = target_size - w;
int hpad = target_size - h;
ncnn::copy_make_border(in, in_pad, hpad/2, hpad-hpad/2, wpad/2, wpad-wpad/2, BORDER_CONSTANT, 114.f);
```

这保证不管摄像头输出什么尺寸，模型永远接收 640×640 正方形输入，与训练时一致。

### 4.2 坐标还原

原有的坐标反变换代码已经使用 `wpad/2` 和 `hpad/2` 作为偏移量，**无需额外修改**：

```cpp
float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
```

### 4.3 修复效果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| p50 (空场景) | 0.84~0.96 | **0.0000** |
| p99 (空场景) | ~1.0 | **0.0006~0.0029** |
| frac_gt090 | 0.41~0.58 | **0.000000** |
| 空场景误检数 | 10 类全检出 | **0** |
| 有人场景 | Person + PPE 正确检出 | Person + PPE 正确检出 ✅ |

---

## 五、同源性问题防范清单

此 bug 的本质是 **C++ 推理预处理与 Python 训练预处理不一致**。以下场景可能触发同类问题：

### ✅ 必须确保一致的预处理环节

| 环节 | 训练时 (Ultralytics) | C++ 推理时 | 一致性要求 |
|------|---------------------|-----------|-----------|
| 输入尺寸 | 640×640 正方形 | **必须 pad 到 640×640** | ⭐ 本次 bug |
| 填充值 | 114 (灰色) | 114.f | ✅ 已一致 |
| 归一化 | /255.0 | `norm_vals = 1/255.f` | ✅ 已一致 |
| 通道顺序 | RGB | `PIXEL_RGB` | ✅ 已一致 |
| letterbox 对齐 | stride=32 倍数 | 已通过 640×640 保证 | ✅ |

### ⚠️ 高风险场景

1. **更换 target_size**（如改为 320/416/1280）：需同步修改 C++ 中的 `det_target_size`
2. **更换模型架构**（如 YOLOv8/v9/v10）：需确认其预处理流程是否一致
3. **动态输入尺寸**：某些 NCNN 模型支持动态 shape，但微调模型可能不耐受
4. **非 letterbox 的 resize 策略**：直接 resize 到 640×640（不保持比例）也能工作，但会引入形变

### 🔧 诊断工具

如遇到类似"空场景全误检"的问题，可使用以下脚本快速定位：

- `scripts/trace_conversion_pipeline.py` — 全链路逐级验证（PyTorch→ONNX→NCNN）
- `scripts/test_letterbox_hypothesis.py` — 测试不同输入尺寸 × 不同预处理方式的分数表现

---

## 六、时间线

| 时间 | 事件 |
|------|------|
| 2026-02-28 | 模型训练完成，NCNN 导出，部署到手机 |
| 2026-02-28 | 发现空场景 10 类全误检，开始 C++ 后处理调试（9+ 轮迭代） |
| 2026-03-01 | Person 检出恢复，但空场景误检仍存在 |
| 2026-03-01 | 用户指出 PC 端完全正常，问题仅在 Android |
| 2026-03-02 | 全链路追踪发现 640×640 输入全部正常 |
| 2026-03-02 | letterbox 非正方形假设验证成功，**根因确认** |
| 2026-03-02 | 两行代码修复，部署验证通过 ✅ |

---

# Bugfix #2: E2E 模型双重 Sigmoid 导致 OBB/Seg/Pose 置信度异常

> **日期**: 2026-03-03  
> **影响范围**: yolo26_obb.cpp / yolo26_seg.cpp / yolo26_pose.cpp 的 `generate_proposals_yolo26()` 函数  
> **严重程度**: Critical — OBB 所有检测框显示 50.0% 置信度，Seg/Pose 置信度同样失真  
> **修复文件**: `yolo26_obb.cpp`, `yolo26_seg.cpp`, `yolo26_pose.cpp`

---

## 一、问题现象

OBB 模式下所有检测框的置信度**精确显示 50.0%**，且画面中出现大量密集的 DOTAv1 类别标签（swimming pool、baseball diamond 等）重叠覆盖。Seg 和 Pose 模式也存在类似的置信度异常。

## 二、根因分析

### 2.1 模型内置 Sigmoid

所有 YOLO26 E2E 模型在 `.ncnn.param` 中已经将 Sigmoid 操作内置到模型图中：

| 模型 | Sigmoid 层 | 作用 |
|------|-----------|------|
| yolo26n_e2e (det) | `sigmoid_162` (line 275) | 类别分数 |
| yolo26n_seg_e2e | `sigmoid_214` (line 340) | 类别分数 |
| yolo26n_pose_e2e | `sigmoid_201` (line 341) | 目标置信分数 |
| yolo26n_pose_e2e | `sigmoid_202` (line 350) | 关键点可见性分数 |
| yolo26n_obb_e2e | `sigmoid_201` (line 353) | 类别分数 |

### 2.2 C++ 代码再次应用 Sigmoid（双重 Sigmoid）

`generate_proposals_yolo26()` 函数中对分数又调用了 `sigmoid(score)`：

```cpp
// OBB/Seg: 类别分数
score = sigmoid(score);    // ❌ 模型已经做过 sigmoid！

// Pose: 目标置信度  
float score = sigmoid(pred_grid[4]);  // ❌ 双重 sigmoid

// Pose: 关键点可见性
keypoint.prob = sigmoid(pred_points_grid.row(k)[2]);  // ❌ 双重 sigmoid
```

### 2.3 为什么所有分数都是 50.0%

对于大多数背景框，模型输出的 sigmoid 概率接近 0（如 0.001）。
对这个值再次应用 sigmoid：

$$\text{sigmoid}(0.001) = \frac{1}{1 + e^{-0.001}} \approx 0.50025 \approx 50.0\%$$

所有低置信度框都被映射到 ~50%，全部通过 0.35 的置信度阈值，导致 8400 个框全部显示。

### 2.4 Det 模型为何不受影响

`yolo26_det.cpp` 在之前的修复中已经注释掉了 sigmoid：
```cpp
// score = sigmoid(score);  // 已移除 - 避免双重sigmoid
```
该修复未同步到 OBB/Seg/Pose 文件。

## 三、修复方案

在 `generate_proposals_yolo26()` 函数中移除对已经过 sigmoid 处理的分数的二次 sigmoid 调用：

| 文件 | 修改行 | 修改内容 |
|------|--------|---------|
| `yolo26_obb.cpp` | ~L253 | `score = sigmoid(score)` → 注释掉 |
| `yolo26_seg.cpp` | ~L249 | `score = sigmoid(score)` → 注释掉 |
| `yolo26_pose.cpp` | ~L233 | `sigmoid(pred_grid[4])` → `pred_grid[4]` |
| `yolo26_pose.cpp` | ~L259 | `sigmoid(pred_points_grid.row(k)[2])` → `pred_points_grid.row(k)[2]` |

**注意**：Legacy `generate_proposals()` 函数（带 DFL/reg_max_1=16，兼容 YOLO26 Legacy 非 E2E 模型）中的 sigmoid **保持不变**，因为非 E2E 模型输出的是原始 logits。

OBB 的角度值 `sigmoid(pred_angle...)` 也**保持不变**，因为模型对角度通道未内置 sigmoid。

---

# Bugfix #3: E2E 模型输出已解码像素坐标，C++ 仍做网格解码（Pose 无骨骼线 / OBB 无检测）

> **日期**: 2026-03-03  
> **影响范围**: `yolo26_pose.cpp`, `yolo26_obb.cpp` 的 `generate_proposals_yolo26()` 路径  
> **严重程度**: Critical — Pose 模式无骨骼线显示，OBB 模式无任何检测框  
> **修复文件**: `yolo26_pose.cpp`, `yolo26_obb.cpp`

---

## 一、问题现象

1. **Pose 模式**：能检测到人体框，但**完全没有骨骼连线和关键点**显示
2. **OBB 模式**：画面上**什么也不显示**，无任何检测框

## 二、根因分析

### 2.1 E2E 模型 vs 非 E2E 模型的输出格式差异

| 模型类型 | bbox 输出 | 关键点/角度输出 | 需要的后处理 |
|---------|----------|---------------|------------|
| 非 E2E（传统 PNNX） | DFL 编码的网格偏移量 | 相对网格偏移 | 需要 DFL 解码 + stride × grid 变换 |
| **E2E（`_e2e` 后缀）** | **已解码的绝对像素坐标** | **已解码的绝对像素坐标** | **直接使用，无需变换** |

### 2.2 Pose 的问题

E2E Pose 模型输出格式为 **(8400, 56)**：

```
[cx, cy, w, h, score, kp0_x, kp0_y, kp0_vis, kp1_x, kp1_y, kp1_vis, ..., kp16_x, kp16_y, kp16_vis]
```

其中 `kp_x`、`kp_y` 已经是**输入图像空间的绝对像素坐标**（如 `kp_x=320.5, kp_y=180.2`）。

但原始 C++ 代码 `generate_proposals_yolo26()` 对关键点做了网格解码变换：

```cpp
// ❌ 错误：对已经是像素坐标的值做了 stride * 2 + anchor 偏移
float kp_x = (pred_points_grid.row(k)[0] * 2.f + (j - 0.5f)) * stride;
float kp_y = (pred_points_grid.row(k)[1] * 2.f + (i - 0.5f)) * stride;
```

当 `kp_x=320.5` 被乘以 `stride=8`（最小层）后变成 `~5128`，远超图像尺寸，关键点全部被画到画面外面，导致骨骼线不可见。

### 2.3 OBB 的问题

E2E OBB 模型输出的 bbox 已经是绝对像素坐标（`cx=400, cy=300, w=80, h=40`），但代码仍然执行：

```cpp
// ❌ 错误：将像素坐标当作网格偏移量再次变换
float pb_cx = (pred_grid[0] * 2.f + (j - 0.5f)) * stride;
float pb_cy = (pred_grid[1] * 2.f + (i - 0.5f)) * stride;
float pb_w  = powf(pred_grid[2] * 2.f, 2) * stride;
float pb_h  = powf(pred_grid[3] * 2.f, 2) * stride;
```

`cx=400` 变成 `(400*2+j)*stride` → 数值爆炸，所有框的尺寸和位置完全错误，经 NMS 后无任何有效框。

## 三、修复方案

为 Pose 和 OBB 各新增一个 **`process_decoded_*_proposals()`** 函数，专门处理 E2E 已解码输出：

### 3.1 自动检测是否为 E2E 解码格式

```cpp
// 检查 bbox 值范围：E2E 输出的像素坐标 >> 1.0，而网格偏移量 < 4.0
float max_bbox_val = 0.f;
for (int i = 0; i < std::min(100, num_proposals); i++) {
    for (int c = 0; c < 4; c++)
        max_bbox_val = std::max(max_bbox_val, fabsf(feat_blob.row(i)[c]));
}
bool is_decoded = (max_bbox_val > target_size * 0.1f);
```

当 `max_bbox_val > 64`（对于 `target_size=640`），说明值是像素坐标而非网格偏移。

### 3.2 Pose — `process_decoded_pose_proposals()`

```cpp
// 直接使用像素坐标，无需 stride/grid 变换
float pb_cx = row[0];  // 已是绝对像素 cx
float pb_cy = row[1];  // 已是绝对像素 cy
float pb_w  = row[2];  // 已是像素宽度
float pb_h  = row[3];  // 已是像素高度
float score = row[4];  // 已经过 sigmoid

// 关键点也直接使用
keypoint.p.x = row[5 + k*3 + 0];  // 绝对像素 x
keypoint.p.y = row[5 + k*3 + 1];  // 绝对像素 y
keypoint.prob = row[5 + k*3 + 2];  // 已经过 sigmoid 的可见性
```

### 3.3 OBB — `process_decoded_obb_proposals()`

```cpp
// 直接使用像素坐标
float pb_cx = row[0];
float pb_cy = row[1];
float pb_w  = row[2];
float pb_h  = row[3];
// 类别和角度见 Bugfix #4 的列布局修正
```

### 3.4 调度逻辑

在 `detect()` 入口处根据 `is_decoded` 标志自动选择路径：

```cpp
if (is_decoded) {
    process_decoded_pose_proposals(feat_blob, objects_all, prob_threshold);  // E2E 路径
} else {
    generate_proposals_yolo26(...);  // 传统 PNNX 路径（保留向后兼容）
}
```

## 四、修复效果

| 模式 | 修复前 | 修复后 |
|------|--------|--------|
| Pose | 有人体框但无骨骼线/关键点 | 骨骼线和关键点正确显示 ✅ |
| OBB | 完全无检测 | 旋转框正确显示（但见 Bugfix #4 置信度问题） |

---

# Bugfix #4: OBB E2E 模型列布局解析错误（所有检测均显示 "plane 50.0%"）

> **日期**: 2026-03-03  
> **影响范围**: `yolo26_obb.cpp` 的 `process_decoded_obb_proposals()` 函数  
> **严重程度**: Critical — 所有检测框均显示同一类别 "plane" 且置信度固定为 50.0%  
> **修复文件**: `yolo26_obb.cpp`

---

## 一、问题现象

在 Bugfix #3 修复 E2E 解码后，OBB 模式可以显示检测框了，但**所有框的标签全部是 "plane 50.0%"**，密密麻麻覆盖整个画面。

## 二、根因分析

### 2.1 假定的列布局 vs 实际列布局

初始代码假设 OBB E2E 输出格式为：

```
[bbox(4), angle(1), classes(15)] — 共 20 列
```

但通过追踪 `yolo26n_obb_e2e.ncnn.param` 的计算图发现**实际列布局**为：

```
[bbox(4), classes_sigmoided(15), angle_raw(1)] — 共 20 列
```

关键的 `Concat` 操作链：

```
cat_24: [decoded_bbox(4)] + [sigmoid(class_scores)(15)] → 19 列
cat_25: [19列] + [raw_angle(1)] → 20 列
```

### 2.2 错误的列解析导致的连锁反应

| 数据列 | 代码假设 | 实际内容 | 后果 |
|--------|---------|---------|------|
| col 4 | angle | **class_0 (plane) 的 sigmoid 概率** | 被当作角度 |
| col 5~18 | classes 0~13 | classes 1~14 | 类别索引全部偏移 |
| col 19 | class 14 | **angle (raw, 未过 sigmoid)** | 被当作类别分数 |

由于 col 4 的值是 plane 的 sigmoid 概率（接近 0 的小数），被当作角度后旋转角度几乎为 0。

更关键的是，代码对"类别分数"（实际是 col 5~19 = classes 1~14 + raw_angle）执行了 `sigmoid()`：

```cpp
score = sigmoid(score);  // 对已经过 sigmoid 的概率再做一次 sigmoid
```

已经过 sigmoid 的概率（接近 0）再次 sigmoid → **全部变成 ~0.50**，与 Bugfix #2 相同的双重 sigmoid 效应。而 col 4（plane 类）因为在新的偏移位置被最先读到，所有框都显示 "plane 50.0%"。

### 2.3 ncnn.param 计算图追踪

```
# 在 yolo26n_obb_e2e.ncnn.param 中的关键操作：

Sigmoid   sigmoid_201   1 1   splitncnn_35_0 sigmoid_201   # 类别 sigmoid
Concat    cat_24        2 1   box_decode_out sigmoid_201 cat_24  0  # [bbox(4), cls_sig(15)]
Sigmoid   sigmoid_202   1 1   splitncnn_35_1 sigmoid_202   # 角度的 sigmoid（仅用于 cat 前一个分支）
Concat    cat_25        2 1   cat_24 angle_raw cat_25  0           # [bbox(4), cls_sig(15), angle(1)]
```

## 三、修复方案

修正 `process_decoded_obb_proposals()` 中的列索引：

```cpp
// ✅ 正确的列布局: [bbox(4), classes_sigmoided(15), angle_raw(1)]
int num_class = w - 5;  // 15 = 20 - 4(bbox) - 1(angle)

// 类别分数：cols 4 ~ w-2（已过 sigmoid，直接使用）
for (int class_idx = 0; class_idx < num_class; class_idx++) {
    float score = row[4 + class_idx];  // ✅ 不再重复 sigmoid
    // ...
}

// 角度：最后一列（raw，需要 sigmoid）
float angle_raw = row[w - 1];
float angle = sigmoid(angle_raw) * M_PI - M_PI / 2;
```

### 关键修正点

| 项目 | 修复前 | 修复后 |
|------|--------|--------|
| 类别起始列 | col 5（跳过 angle） | **col 4**（紧接 bbox） |
| 类别数量 | `w - 5` | `w - 5`（不变，15 类） |
| 类别 sigmoid | `sigmoid(score)` 双重应用 | **直接使用**（已是 sigmoid 输出） |
| 角度位置 | col 4 | **col w-1**（最后一列） |
| 角度处理 | 直接使用 | **`sigmoid(raw)` → 映射到 [-π/2, π/2]** |

## 四、修复效果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 类别标签 | 全部 "plane" | 正确识别各 DOTAv1 类别 ✅ |
| 置信度 | 全部 50.0% | 正常分布（0.35~0.95） ✅ |
| 检测数量 | 8400 个全显示 | 仅显示有效目标 ✅ |
| 旋转角度 | 接近 0°（错误） | 与目标朝向匹配 ✅ |

## 五、教训

**永远不要假设模型输出的列布局** — 必须从 `.ncnn.param` 的 `Concat` 操作链追踪实际拼接顺序。不同任务头（Det/Seg/Pose/OBB）的 E2E 输出列布局可能完全不同。

---

# Bugfix #5: yolo26_seg.cpp IntelliSense 报错（Windows IDE 红色波浪线）

> **日期**: 2026-03-03  
> **影响范围**: `yolo26_seg.cpp` 在 VS Code + C/C++ 扩展中的 IntelliSense 解析  
> **严重程度**: Low — 不影响 NDK 实际编译和运行，仅 IDE 显示问题  
> **修复文件**: `yolo26_seg.cpp`

---

## 一、问题现象

`yolo26_seg.cpp` 在 VS Code 中显示 **11 处红色波浪线**错误：

1. **7 处** `__android_log_print` / `ANDROID_LOG_DEBUG` / `ANDROID_LOG_ERROR` 标识符未定义
2. **4 处** `std::max` / `std::min` 调用显示 "expected an identifier"

同样的代码在 `yolo26_pose.cpp` 和 `yolo26_obb.cpp` 中**不报错**（虽然它们也没有显式 `#include <android/log.h>`）。

## 二、根因分析

### 2.1 Android Log 未定义

Windows 上的 IntelliSense 引擎（MSVC 模式）无法解析 Android NDK 的 `<android/log.h>` 头文件。在其他文件中，IntelliSense 碰巧通过 ncnn/OpenCV 头文件的传递包含链找到了 `android/log.h`，但在 `yolo26_seg.cpp` 中因包含顺序不同导致传递路径断裂。

直接添加 `#include <android/log.h>` 反而会导致 IntelliSense **完全禁用**该翻译单元（报 "cannot open source file"），使所有语法检查失效。

### 2.2 `std::max`/`std::min` 报错

Windows 平台的 `<windows.h>` 或 MSVC 标准库头文件中定义了 `max`/`min` 宏，与 `std::max`/`std::min` 模板函数冲突。IntelliSense 在预处理时将 `std::max(...)` 中的 `max` 展开为宏，导致语法错误。

## 三、修复方案

### 3.1 Android Log — 条件编译 Fallback

```cpp
#if defined(__ANDROID__) || defined(ANDROID)
#include <android/log.h>
#else
// Fallback stubs for IDE IntelliSense on non-Android hosts
#define ANDROID_LOG_DEBUG 3
#define ANDROID_LOG_ERROR 6
static inline int __android_log_print(int, const char*, const char*, ...) { return 0; }
#endif
```

- Android NDK 编译时（`__ANDROID__` 已定义）：正常包含系统头文件
- Windows IntelliSense 解析时：使用 stub 定义消除未定义错误
- **不影响实际编译产物**

### 3.2 `std::max`/`std::min` — 括号化防宏展开

```cpp
// 修复前
x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);

// 修复后 — 括号阻止宏展开
x0 = (std::max)((std::min)(x0, (float)(img_w - 1)), 0.f);
```

`(std::max)` 的括号使预处理器不再将 `max` 视为函数式宏调用，从而正确解析为 `std::max` 模板函数。这是 Windows C++ 开发中的标准惯用法。

## 四、修复效果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| IntelliSense 错误数 | 11 | **0** ✅ |
| NDK 编译 | 正常 | 正常（无影响） |
| 运行时行为 | 正常 | 正常（无影响） |

---

# 修复总览

| # | 问题 | 根因 | 影响 | 修复文件 |
|---|------|------|------|---------|
| 1 | 空场景 10 类全误检 | 非正方形输入未 pad 到 640×640 | Det Critical | `yolo26_det.cpp` |
| 2 | OBB/Seg/Pose 置信度 50% | E2E 已内置 sigmoid，C++ 双重应用 | Critical | `yolo26_obb/seg/pose.cpp` |
| 3 | Pose 无骨骼线 / OBB 无检测 | E2E 输出已解码像素坐标，C++ 仍做网格解码 | Critical | `yolo26_pose/obb.cpp` |
| 4 | OBB 全显示 "plane 50.0%" | OBB 列布局假设错误 | Critical | `yolo26_obb.cpp` |
| 5 | yolo26_seg.cpp IDE 报错 | Windows IntelliSense 缺少 NDK 头 + 宏冲突 | Low | `yolo26_seg.cpp` |
