# ncnn-android-yolo26LT

**A consistency-aware Java–JNI–C++–NCNN framework for deploying and diagnosing multi-task YOLO26 assets on resource-constrained Android devices.**

---

## Overview

This project accompanies the paper:

> **Consistency-Aware Deployment of Multi-Task YOLO26 on Resource-Constrained Android Devices: A Java–JNI–C++–NCNN Architecture with Calibrated Post-Processing and Quantitative Fault Diagnosis**

In this repository, YOLO26 denotes an engineering-managed family of `yolo26n` model assets rather than a new general-purpose model family. A single Android application loads five heterogeneous vision tasks — detection, segmentation, pose estimation, classification, and OBB detection — through a unified Java–JNI–C++–NCNN pipeline. The project focuses on unified deployment under coexisting E2E, One-to-Many (O2M), and Legacy output paths, runtime post-processing consistency, and a five-stage diagnosis workflow. The SafeHat 10-class PPE detection scenario serves as the main closed-loop validation case for smart-city edge contexts such as construction-site safety monitoring and engineering inspection patrols.

---

## 项目简介

本项目配套论文《Consistency-Aware Deployment of Multi-Task YOLO26 on Resource-Constrained Android Devices: A Java–JNI–C++–NCNN Architecture with Calibrated Post-Processing and Quantitative Fault Diagnosis》。本文中的 YOLO26 指本项目工程化管理的一组 `yolo26n` 模型资产，而不是新的通用模型族。项目在同一 Android 应用内通过 Java–JNI–C++–NCNN 分层协同承载检测、分割、姿态估计、分类和 OBB 五类视觉任务，重点覆盖 E2E、One-to-Many (O2M) 与 Legacy 输出路径共存时的统一部署、后处理一致性和五阶段故障诊断。SafeHat 10 类 PPE 检测作为施工安全监测、工程巡检等智能城市场景中的主要闭环验证案例。

---

## Supported Tasks

| Task ID | Task | Model Asset | Input | Output semantics and parser |
|---------|------|-------------|-------|-----------------------------|
| 0 | Detection / SafeHat PPE | `yolo26n_safehat.ncnn.*` preferred; compatible alias `yolo26n_e2e.ncnn.*` | 640×640 letterbox | SafeHat main case: O2M, `out0≈8400×14`; O2M Top-K/NMS + calibration. Original general detection can use E2E, `out0≈300×6`. |
| 1 | Segmentation | `yolo26n_seg_e2e.ncnn.*` | 640×640 letterbox | `out0=300×38`, `out1=mask prototypes`; reconstruct 32-dimensional masks and overlay contours. |
| 2 | Pose Estimation | `yolo26n_pose_e2e.ncnn.*` | 640×640 letterbox | E2E output or dual branches; parse 17 keypoints and draw skeletons. |
| 3 | Classification | `yolo26n_cls.ncnn.*` | 224×224 resize | class-score vector with Top-5 ranking. |
| 4 | OBB Detection | `yolo26n_obb_e2e.ncnn.*` | 640×640 letterbox | E2E or bbox/class/angle branches; parse oriented boxes and angles. |

All reported experiments run on the **NCNN CPU fp32 path** (arm64-v8a). Vulkan is retained by the build but is not used in the reported measurements. Experiments were conducted on HUAWEI P20 Pro (Kirin 970, Android 10).

---

## 支持任务

所有报告结果均在 NCNN CPU fp32 路径下运行（arm64-v8a），实验设备为华为 P20 Pro（Kirin 970，Android 10）。注意：SafeHat 主案例采用 One-to-Many 输出路径，`out0≈8400×14`，`yolo26n_e2e` 在该场景中是兼容历史别名，不能据此判断当前 SafeHat 使用 E2E 检测头。

---

## On-Device Inference Latency

Measured on Kirin 970 (4× Cortex-A73 + 4× Cortex-A53), NCNN CPU fp32, no NPU/GPU. Values are pure inference latency without rendering.

| Task | Model | Samples | Mean (ms) | P5 (ms) | P95 (ms) | FPS | Note |
|------|-------|---------|-----------|---------|----------|-----|------|
| Detection | `yolo26n_safehat` (compatible alias `yolo26n_e2e`) | 148 | 321.3 | 272.2 | 412.2 | 3.1 | 640×640 letterbox |
| Segmentation | `yolo26n_seg_e2e` | 108 | 430.5 | 359.3 | 517.5 | 2.3 | 640×640 letterbox |
| Pose Estimation | `yolo26n_pose_e2e` | 113 | 369.4 | 342.0 | 427.6 | 2.7 | 640×640 letterbox |
| Classification | `yolo26n_cls` | 928 | 33.6 | 25.9 | 41.8 | 29.8 | 224×224 resize |
| OBB Detection | `yolo26n_obb_e2e` | 120 | 336.1 | 308.4 | 386.3 | 3.0 | 640×640 letterbox |

---

## 设备端推理延迟

在 Kirin 970（4× Cortex-A73 + 4× Cortex-A53）上测量，NCNN CPU fp32，不启用 NPU/GPU。检测、分割、姿态估计和 OBB 四个 640×640 任务的平均延迟为 321.3–430.5 ms，分类任务在 224×224 输入下平均为 33.6 ms。数据见上表。

---

## Consistency Diagnosis Workflow

The framework addresses four representative fault classes through a structured five-stage workflow: anomaly logging → intermediate-output inspection → cross-backend comparison → structure tracing → regression validation. In practice, S3 cross-backend comparison is the main front-end filter, S4 is used when value patterns alone cannot explain the fault, and S5 remains mandatory after repair.

| Fault Class | Observable phenomenon | Root cause | Repair / validation evidence |
|-------------|-----------------------|------------|------------------------------|
| Preprocessing mismatch | all 10 classes detected in empty scenes; `p50 = 0.84–0.96` | stride-aligned padding generated 640×384 instead of the required 640×640 square tensor | pad to `target_size × target_size`; empty-scene `p50 < 0.0001`, zero false detections |
| Redundant activation | confidences fixed around 50%; dense candidates appear | the NCNN graph already contains sigmoid, but the C++ parser applies sigmoid again | remove redundant `sigmoid()`; raw probability distribution is restored and calibrated UI scores remain in 0.35–0.73 |
| Coordinate semantic misinterpretation | Pose skeleton missing; OBB geometry absent | E2E outputs are already absolute pixel coordinates, but the parser decodes them again as `grid × stride` | add a parser branch for already decoded outputs; keypoints and oriented boxes render correctly |
| OBB layout misjudgment | class labels, confidences, and directions are abnormal | parser assumes `[bbox, angle, classes]`; actual output is `[bbox, classes_sigmoided, angle_raw]` | correct column indices through NCNN concat-chain tracing; class, confidence, and direction become correct |

---

## 一致性诊断流程

框架通过异常记录、中间输出检查、跨后端对照、结构追踪和回归验证五个步骤，系统覆盖预处理不匹配、冗余激活、坐标语义误解和 OBB 布局误判四类代表性故障。其中 S3 跨后端对照用于优先区分导出侧问题和设备端实现问题，S4 仅在数值模式不足以解释故障时进一步追踪结构，S5 回归验证是每次修复后的必要步骤。

---

## Build Instructions

> **Requirement:** Android Studio, JDK 17+, Android Gradle Plugin 8.7.3, NDK r29 (29.0.14206865), compileSdk 33, minSdk 24

### Step 1 — NCNN library

Download `ncnn-YYYYMMDD-android-vulkan.zip` from [ncnn releases](https://github.com/Tencent/ncnn/releases).  
Extract into `app/src/main/jni/` and verify the path in `CMakeLists.txt`:

```cmake
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20260113-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
```

### Step 2 — OpenCV Mobile

Download `opencv-mobile-XYZ-android.zip` from [opencv-mobile releases](https://github.com/nihui/opencv-mobile/releases).  
Extract into `app/src/main/jni/` and verify:

```cmake
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.13.0-android/sdk/native/jni)
```

### Step 3 — Model assets

Place the following files in `app/src/main/assets/`:

```
yolo26n_e2e.ncnn.bin / .param
yolo26n_seg_e2e.ncnn.bin / .param
yolo26n_pose_e2e.ncnn.bin / .param
yolo26n_cls.ncnn.bin / .param
yolo26n_obb_e2e.ncnn.bin / .param
yolo26n_safehat.ncnn.bin / .param
```

Export from PyTorch using the Ultralytics NCNN path. E2E task branches and the SafeHat O2M main case should be exported with different output-path settings and then verified before asset replacement:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="ncnn", imgsz=640, half=False, end2end=True)   # E2E task branches

safehat = YOLO("best.pt")
safehat.export(format="ncnn", imgsz=640, half=False, end2end=False) # SafeHat O2M main case
```

Project scripts such as `scripts/export_yolo26_ncnn.py`, `scripts/export_best_safehat_to_assets.py`, `scripts/verify_ncnn_output.py`, and `scripts/conversion_detail_check.py` are used to export, copy, and verify the concrete assets described in the paper.

### Step 4 — Build

Open in Android Studio → Build → Run on device (compileSdk 33, minSdk 24, targetSdk 35).

---

## 构建步骤

> **环境要求：** Android Studio，JDK 17+，Android Gradle Plugin 8.7.3，NDK r29 (29.0.14206865)，compileSdk 33，minSdk 24

1. 从 [ncnn releases](https://github.com/Tencent/ncnn/releases) 下载 `ncnn-YYYYMMDD-android-vulkan.zip`，解压至 `app/src/main/jni/`，并按实际路径修改 `CMakeLists.txt` 中的 `ncnn_DIR`。
2. 从 [opencv-mobile releases](https://github.com/nihui/opencv-mobile/releases) 下载 `opencv-mobile-XYZ-android.zip`，解压至同目录，修改 `OpenCV_DIR`。
3. 将 NCNN 模型文件（`.bin` + `.param`）放入 `app/src/main/assets/`。E2E 任务分支可通过 `model.export(format="ncnn", imgsz=640, half=False, end2end=True)` 导出；SafeHat 主案例采用 O2M 输出路径，应使用 `end2end=False` 并在替换资产前检查输出形状和列布局。
4. 用 Android Studio 打开项目，编译并部署到设备（compileSdk 33，minSdk 24，targetSdk 35）。

---

## Project Structure

```
ncnn-android-yolo26LT/
├── app/src/main/
│   ├── assets/                     # NCNN model files (.bin + .param)
│   ├── java/com/tencent/yolo26ncnn/
│   │   ├── MainActivity.java       # UI + task dispatch
│   │   └── YOLO26Ncnn.java        # JNI wrapper
│   └── jni/
│       ├── yolo26.h / yolo26.cpp  # Base inference class
│       ├── yolo26_det.cpp         # Detection parser
│       ├── yolo26_seg.cpp         # Segmentation parser
│       ├── yolo26_pose.cpp        # Pose parser
│       ├── yolo26_cls.cpp         # Classification parser
│       ├── yolo26_obb.cpp         # OBB parser
│       ├── yolo26ncnn.cpp         # JNI interface
│       ├── ndkcamera.cpp          # Android NDK Camera2
│       └── CMakeLists.txt
├── scripts/                        # Export, diagnosis, and benchmark scripts
├── data/                           # Label files and dataset metadata
├── test/                           # Integration testing notes
├── runs/paper_figures/             # Paper figures and runtime validation montages
└── PaperEnglish.md                 # Electronics-style manuscript draft
```

---

## Notes

- Android NDK Camera2 API is used; HAL3 support required (API 24+).
- All models run on the CPU path; Vulkan is available in the NCNN build but not activated in experiments.
- SafeHat is the main quantitative case and uses the O2M output path; other task branches primarily validate runtime coverage and parsing correctness under the unified pipeline.
- Model assets in `assets/` are the NCNN-converted files (~10 MB each); original `.pt` weights are excluded from this repository.

---

## 注意事项

- 使用 Android NDK Camera2 接口，需设备支持 HAL3（API 24+）。
- 所有模型均在 CPU 路径下运行，NCNN 构建中包含 Vulkan 路径，但实验中未启用。
- SafeHat 是主要定量验证案例并采用 O2M 输出路径；其他任务分支主要用于验证统一部署管线下的运行覆盖和解析正确性。
- `assets/` 中为 NCNN 转换后的模型文件（单文件约 10 MB），原始 `.pt` 权重不包含在本仓库中。

---

## Related Paper

> *Consistency-Aware Deployment of Multi-Task YOLO26 on Resource-Constrained Android Devices: A Java–JNI–C++–NCNN Architecture with Calibrated Post-Processing and Quantitative Fault Diagnosis*  
> Submitted to **Electronics** (MDPI).

---

## 相关论文

> 《Consistency-Aware Deployment of Multi-Task YOLO26 on Resource-Constrained Android Devices: A Java–JNI–C++–NCNN Architecture with Calibrated Post-Processing and Quantitative Fault Diagnosis》  
> 投稿至 **Electronics**（MDPI）。

---

## Dependencies

| Library | Version | Link |
|---------|---------|------|
| ncnn | 20260113 | https://github.com/Tencent/ncnn |
| opencv-mobile | 4.13.0 | https://github.com/nihui/opencv-mobile |
| ultralytics | latest | https://github.com/ultralytics/ultralytics |

---

## License

BSD 3-Clause License
