# ncnn-android-yolo26LT

**An NCNN-based framework for multi-task vision deployment and consistency diagnosis on resource-constrained Android devices.**

---

## Overview

This project accompanies the paper:

> **An NCNN-Based Framework for Multi-Task Vision Deployment and Consistency Diagnosis on Resource-Constrained Android Devices**

A single Android application loads five heterogeneous YOLO26 vision tasks — detection, segmentation, pose estimation, classification, and OBB detection — via a unified Java–JNI–C++–NCNN pipeline. Beyond deployment, the project provides a five-stage consistency diagnosis workflow that covers four representative fault classes, with the SafeHat 10-class PPE detection scenario serving as the end-to-end validation case.

---

## 项目简介

本项目配套论文《面向资源受限 Android 设备的基于 NCNN 的多任务视觉推理部署与一致性诊断框架》。在同一 Android 应用内通过 Java–JNI–C++–NCNN 分层协同统一承载 YOLO26 的五类视觉任务，并提供一套从异常记录到回归验证的部署一致性诊断流程。主验证案例为 SafeHat 10 类安全帽/PPE 检测。

---

## Supported Tasks

| Task | Model Asset | Input | Output Path |
|------|-------------|-------|-------------|
| Detection | `yolo26n_e2e.ncnn.*` | 640×640 | E2E (no NMS) |
| Segmentation | `yolo26n_seg_e2e.ncnn.*` | 640×640 | E2E |
| Pose Estimation | `yolo26n_pose_e2e.ncnn.*` | 640×640 | E2E |
| Classification | `yolo26n_cls.ncnn.*` | 224×224 | Softmax |
| OBB Detection | `yolo26n_obb_e2e.ncnn.*` | 640×640 | E2E |
| SafeHat (PPE) | `yolo26n_safehat.ncnn.*` | 640×640 | E2E (fine-tuned) |

All tasks run on the **CPU path** (ARM fp32, arm64-v8a). Experiments were conducted on HUAWEI P20 Pro (Kirin 970, Android 10).

---

## 支持任务

所有任务均在 CPU 路径下运行（ARM fp32，arm64-v8a），实验设备为华为 P20 Pro（Kirin 970，Android 10）。任务列表见上表。

---

## On-Device Inference Latency

Measured on Kirin 970 (4× Cortex-A73 + 4× Cortex-A53), NCNN CPU fp32, no NPU/GPU.

| Task | Input | Avg Latency | FPS |
|------|-------|-------------|-----|
| Detection | 640×640 | ~321 ms | ~3.1 |
| Segmentation | 640×640 | ~380 ms | ~2.6 |
| Pose Estimation | 640×640 | ~430 ms | ~2.3 |
| OBB Detection | 640×640 | ~355 ms | ~2.8 |
| Classification | 224×224 | ~34 ms | ~29.8 |

---

## 设备端推理延迟

在 Kirin 970（4× Cortex-A73 + 4× Cortex-A53）上测量，NCNN CPU fp32，不启用 NPU/GPU。数据见上表。

---

## Consistency Diagnosis Workflow

The framework addresses four representative fault classes through a structured five-stage workflow (anomaly logging → intermediate-output inspection → cross-backend comparison → structure tracing → regression validation):

| Fault Class | Description | Diagnosis Step |
|-------------|-------------|----------------|
| Preprocessing mismatch | letterbox / normalization inconsistency between Python and C++ | Cross-backend comparison |
| Redundant activation | Sigmoid applied twice on E2E output | Intermediate-output inspection |
| Coordinate semantics error | Pose keypoint confidence threshold truncation | Anomaly logging + structure tracing |
| OBB layout misinterpretation | Angle encoding offset in rotated-box parser | Regression validation |

---

## 一致性诊断流程

框架通过异常记录、中间输出检查、跨后端对照、结构追踪和回归验证五个步骤，系统覆盖上述四类代表性故障：预处理不匹配、冗余激活、坐标语义误解和 OBB 布局误判。

---

## Build Instructions

> **Requirement:** Android Studio, JDK 17+, Android Gradle Plugin 8.7.3, NDK r25+

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

Export from PyTorch using:

```python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
model.export(format="ncnn", end2end=True)
```

### Step 4 — Build

Open in Android Studio → Build → Run on device (minSdk 24, targetSdk 35).

---

## 构建步骤

> **环境要求：** Android Studio，JDK 17+，Android Gradle Plugin 8.7.3，NDK r25+

1. 从 [ncnn releases](https://github.com/Tencent/ncnn/releases) 下载 `ncnn-YYYYMMDD-android-vulkan.zip`，解压至 `app/src/main/jni/`，并按实际路径修改 `CMakeLists.txt` 中的 `ncnn_DIR`。
2. 从 [opencv-mobile releases](https://github.com/nihui/opencv-mobile/releases) 下载 `opencv-mobile-XYZ-android.zip`，解压至同目录，修改 `OpenCV_DIR`。
3. 将 NCNN 模型文件（`.bin` + `.param`）放入 `app/src/main/assets/`。模型可通过 `model.export(format="ncnn", end2end=True)` 从 PyTorch 权重导出。
4. 用 Android Studio 打开项目，编译并部署到设备（minSdk 24，targetSdk 35）。

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
└── Paper.md                        # Manuscript draft
```

---

## Notes

- Android NDK Camera2 API is used; HAL3 support required (API 24+).
- All models run on the CPU path; Vulkan is available in the NCNN build but not activated in experiments.
- Model assets in `assets/` are the NCNN-converted files (~10 MB each); original `.pt` weights are excluded from this repository.

---

## 注意事项

- 使用 Android NDK Camera2 接口，需设备支持 HAL3（API 24+）。
- 所有模型均在 CPU 路径下运行，NCNN 构建中包含 Vulkan 路径，但实验中未启用。
- `assets/` 中为 NCNN 转换后的模型文件（单文件约 10 MB），原始 `.pt` 权重不包含在本仓库中。

---

## Related Paper

> *An NCNN-Based Framework for Multi-Task Vision Deployment and Consistency Diagnosis on Resource-Constrained Android Devices*  
> Submitted to **Electronics** (MDPI).

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
