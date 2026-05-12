# ncnn-android-yolo26

![download](https://img.shields.io/github/downloads/nihui/ncnn-android-yolo26/total.svg)

The YOLO26 object detection - End-to-End (E2E) mode

基于最新的YOLO26架构，采用端到端推理模式，无需NMS后处理。

This is a sample ncnn android project, it depends on ncnn library and opencv

https://github.com/Tencent/ncnn

https://github.com/nihui/opencv-mobile

https://github.com/nihui/mesa-turnip-android-driver (mesa turnip driver)

## ✨ Features

- ✅ **YOLO26 E2E (One-to-One Head)**: 端到端推理，无需NMS后处理
- ✅ **43% CPU速度提升**: 相比传统One-to-Many模式
- ✅ **无DFL模块**: 更轻量化的模型架构
- ✅ **5种任务支持**: Detection、Segmentation、Pose、Classification、OBB
- ✅ **Vulkan GPU加速**: 支持GPU和Turnip驱动

## 📊 Supported Tasks

| Task | Model | Description |
|------|-------|-------------|
| Detection | yolo26n_e2e | COCO 80类目标检测 |
| Segmentation | yolo26n_seg_e2e | 实例分割 |
| Pose | yolo26n_pose_e2e | 人体姿态估计 |
| Classification | yolo26n_cls | ImageNet 1000类分类 |
| OBB | yolo26n_obb_e2e | 旋转框检测 |

## 🚀 YOLO26 E2E Architecture

```
输入图像 (640x640)
    ↓
YOLO26 Backbone + Neck
    ↓
One-to-One Head (端到端)
    ↓
输出: (300, 6) = [x_center, y_center, width, height, class_id, confidence]
    ↓
无需NMS，直接获得最终结果
```

**E2E输出格式**:
- 最多300个检测结果
- 每个检测包含6个值: `[x_center, y_center, width, height, class_id, confidence]`
- 无需非极大值抑制(NMS)后处理

## how to build and run

> Build requirement: Android Gradle Plugin `8.7.3` needs **JDK 11+** (recommended JDK 17).

For practical model integration + on-device debug workflow, see:
- [test/integration_testing.md](test/integration_testing.md)

### step1
https://github.com/Tencent/ncnn/releases

* Download ncnn-YYYYMMDD-android-vulkan.zip or build ncnn for android yourself
* Extract ncnn-YYYYMMDD-android-vulkan.zip into **app/src/main/jni** and change the **ncnn_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step2
https://github.com/nihui/opencv-mobile

* Download opencv-mobile-XYZ-android.zip
* Extract opencv-mobile-XYZ-android.zip into **app/src/main/jni** and change the **OpenCV_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step3
https://github.com/nihui/mesa-turnip-android-driver

* Download mesa-turnip-android-XYZ.zip
* Create directory **app/src/main/jniLibs/arm64-v8a** if not exists
* Extract `libvulkan_freedreno.so` from mesa-turnip-android-XYZ.zip into **app/src/main/jniLibs/arm64-v8a**

### step4
* Open this project with Android Studio, build it and enjoy!

## some notes
* Android ndk camera is used for best efficiency
* Crash may happen on very old devices for lacking HAL3 camera interface
* All models are manually modified to accept dynamic input shape
* Most small models run slower on GPU than on CPU, this is common
* FPS may be lower in dark environment because of longer camera exposure time

## screenshot
![](screenshot0.jpg)
![](screenshot1.jpg)
![](screenshot2.jpg)

## 📋 Model Conversion Guide

### Export YOLO26 to NCNN (E2E mode)

```python
from ultralytics import YOLO

# Load YOLO26 model
model = YOLO("yolo26n.pt")

# Export with E2E mode (One-to-One Head, no NMS needed)
model.export(format="ncnn", end2end=True)
```

### Model Files
After export, you will get:
```
yolo26n_e2e.ncnn.param
yolo26n_e2e.ncnn.bin
```

Copy these files to `app/src/main/assets/`

## 📁 Project Structure

```
ncnn-android-yolo26/
├── app/
│   └── src/main/
│       ├── assets/                    # NCNN model files
│       │   ├── yolo26n_e2e.ncnn.*    # Detection
│       │   ├── yolo26n_seg_e2e.ncnn.*# Segmentation
│       │   ├── yolo26n_pose_e2e.ncnn.*# Pose
│       │   ├── yolo26n_cls.ncnn.*    # Classification
│       │   └── yolo26n_obb_e2e.ncnn.*# OBB
│       ├── java/com/tencent/yolo26ncnn/
│       │   ├── MainActivity.java
│       │   └── YOLO26Ncnn.java       # JNI wrapper
│       ├── jni/
│       │   ├── yolo26.h              # Header
│       │   ├── yolo26.cpp            # Base class
│       │   ├── yolo26_det.cpp        # Detection
│       │   ├── yolo26_seg.cpp        # Segmentation
│       │   ├── yolo26_pose.cpp       # Pose estimation
│       │   ├── yolo26_cls.cpp        # Classification
│       │   ├── yolo26_obb.cpp        # OBB detection
│       │   ├── yolo26ncnn.cpp        # JNI interface
│       │   └── CMakeLists.txt
│       └── res/
└── modelback/                         # Backup of training models
```

## ⚡ Performance

| Model | Mode | Input Size | NMS | Speed Improvement |
|-------|------|------------|-----|-------------------|
| YOLO26n | E2E (One-to-One) | 640 | ❌ No | **+43%** |
| YOLO26n | One-to-Many | 640 | ✅ Yes | Baseline |
| YOLO26n (Legacy DFL) | DFL | 640 | ✅ Yes | - |

## 🔧 Technical Details

### YOLO26 vs YOLO26 (Legacy)

| Feature | YOLO26 | YOLO26 (Legacy) |
|---------|--------|--------|
| DFL Module | ❌ Removed | ✅ Present |
| E2E Support | ✅ Native | ❌ No |
| BBox Output | Direct (4 values) | DFL (64 values) |
| Post-process | None (E2E) | NMS required |

### E2E Output Format

```cpp
// Output shape: (300, 6)
// Each detection: [x_center, y_center, width, height, class_id, confidence]

for (int i = 0; i < 300; i++) {
    float x_center = output[i * 6 + 0];
    float y_center = output[i * 6 + 1];
    float width    = output[i * 6 + 2];
    float height   = output[i * 6 + 3];
    float class_id = output[i * 6 + 4];
    float conf     = output[i * 6 + 5];
    
    if (conf > threshold) {
        // Valid detection, no NMS needed!
    }
}
```

## License

BSD 3-Clause License

## Acknowledgments

- [ncnn](https://github.com/Tencent/ncnn) - High-performance neural network inference framework
- [opencv-mobile](https://github.com/nihui/opencv-mobile) - Minimal OpenCV for mobile
- [ultralytics](https://github.com/ultralytics/ultralytics) - YOLO models
