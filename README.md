# ncnn-android-yolo26LT

A reproducible Android-NCNN workflow for diagnosing YOLO-to-NCNN deployment inconsistencies on resource-constrained Android devices.

This repository accompanies the paper:

> **A reproducible workflow for diagnosing YOLO-to-NCNN deployment inconsistencies on resource-constrained Android devices**

The project provides an Android application and supporting scripts for moving YOLO26 assets into an NCNN-based Android runtime, observing silent deployment errors, and validating repairs on real CPU-only Android devices. The main demonstration case is a 10-class SafeHat PPE detector, while the same Android application also exercises segmentation, pose estimation, classification, and oriented bounding-box (OBB) assets.

In this repository, YOLO26 denotes an engineering-managed set of `yolo26n` model assets rather than a new general-purpose model family.

## What This Repository Provides

- A Java-JNI-C++-NCNN Android application for multi-task YOLO26 inference.
- A SafeHat PPE detection deployment case using `yolo26n_safehat`.
- A five-stage diagnosis workflow for silent Android deployment inconsistencies.
- Native task parsers for detection, segmentation, pose estimation, classification, and OBB detection.
- Export, inspection, conversion-check, regression, and latency scripts used by the workflow.
- CPU-path runtime evidence on HUAWEI P20 Pro (Kirin 970) and HUAWEI Nova2 (Kirin 659).

## Diagnosis Workflow

The workflow treats Android inference as an observable chain:

```text
training model -> exported graph -> NCNN param/bin -> Android asset ->
JNI/native inference -> task parser -> post-processing -> rendered output/log
```

It uses five stages:

| Stage | Name | Purpose | Representative tools |
|---|---|---|---|
| S1 | Anomaly logging | Record abnormal scores, missing geometry, label disorder, or loading errors | Android logcat, `diagnose_background_scores.py` |
| S2 | Intermediate-output inspection | Check output names, tensor shapes, class count, and value ranges | `verify_ncnn_output.py`, `diagnose_model.py` |
| S3 | Cross-backend comparison | Compare the same input across desktop and Android paths | `diagnose_model.py`, `conversion_matrix_check.py` |
| S4 | Graph-structure tracing | Trace activations, concat chains, coordinate semantics, and output columns | `conversion_detail_check.py`, parser inspection |
| S5 | Regression validation | Re-run failing and previously passing frozen scenes after repair | `test_letterbox_hypothesis.py`, `test_yolo26_e2e.py` |

The paper validates four representative fault modes:

| Fault mode | Symptom | Repair evidence |
|---|---|---|
| Preprocessing mismatch | Empty scenes triggered high-confidence detections across all 10 classes | Enforcing strict `640 x 640` letterbox reduced empty-scene median scores from `0.84-0.96` to below `0.0001` |
| Redundant activation | Confidence values collapsed near `0.5` | Removing duplicate sigmoid restored scene-dependent probabilities with `p50 ~= 0.34` and `p95 ~= 0.74` |
| Coordinate semantic mismatch | Pose skeletons or OBB geometry were missing or outside view | Adding an E2E parser branch restored decoded keypoints and oriented boxes |
| OBB layout mismatch | Labels, confidences, and directions were systematically wrong | Correcting output-column indices restored class labels, confidence values, and angles |

## Task Assets

| Task | Asset prefix | Input | Parser checkpoint |
|---|---|---|---|
| SafeHat detection | `yolo26n_safehat` | `640 x 640` letterbox | O2M `out0 ~= 8400 x 14`, 10 class channels, Top-K, NMS, display score |
| Segmentation | `yolo26n_seg_e2e` | `640 x 640` letterbox | E2E detections plus mask coefficients/prototypes |
| Pose estimation | `yolo26n_pose_e2e` | `640 x 640` letterbox | 17 keypoints and skeleton rendering |
| Classification | `yolo26n_cls` | `224 x 224` resize | Top-5 class ranking |
| OBB detection | `yolo26n_obb_e2e` | `640 x 640` letterbox | Class score, angle, and rotated rectangle |

The SafeHat demonstration uses 10 classes: `Hardhat`, `Mask`, `No-Hardhat`, `No-Mask`, `No-Safety Vest`, `Person`, `Safety Cone`, `Safety Vest`, `Machinery`, and `Vehicle`.

## Runtime Evidence

Reported measurements use NCNN CPU fp32 only. No NPU, GPU, Vulkan acceleration, or quantization is used in these rows.

| Task | Device | Samples | Mean (ms) | P5/Min (ms) | P95/Max (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|
| Detection | Kirin 970 | 148 | 321.3 | 272.2 | 412.2 | 3.1 |
| Segmentation | Kirin 970 | 108 | 430.5 | 359.3 | 517.5 | 2.3 |
| Pose estimation | Kirin 970 | 113 | 369.4 | 342.0 | 427.6 | 2.7 |
| Classification | Kirin 970 | 928 | 33.6 | 25.9 | 41.8 | 29.8 |
| OBB | Kirin 970 | 120 | 336.1 | 308.4 | 386.3 | 3.0 |
| Detection | Kirin 659 | 30 | 621.5 | 533.5 | 887.4 | 1.6 |
| Segmentation | Kirin 659 | 30 | 829.8 | 619.5 | 1017.4 | 1.2 |
| Pose estimation | Kirin 659 | 30 | 740.2 | 611.9 | 861.9 | 1.4 |
| Classification | Kirin 659 | 150 | 70.8 | 41.4 | 208.5 | 14.1 |
| OBB | Kirin 659 | 60 | 640.6 | 541.1 | 859.8 | 1.6 |

These values document the validation environment rather than a universal Android performance benchmark.

## Project Structure

```text
ncnn-android-yolo26LT/
|-- app/src/main/
|   |-- assets/                         # NCNN model files (.bin + .param)
|   |-- java/com/tencent/yolo26ncnn/
|   |   |-- MainActivity.java           # UI, task switching, threshold controls
|   |   `-- YOLO26Ncnn.java            # JNI wrapper
|   `-- jni/
|       |-- yolo26.h / yolo26.cpp       # Base inference and shared preprocessing
|       |-- yolo26_det.cpp              # SafeHat/detection parser
|       |-- yolo26_seg.cpp              # Segmentation parser
|       |-- yolo26_pose.cpp             # Pose parser
|       |-- yolo26_cls.cpp              # Classification parser
|       |-- yolo26_obb.cpp              # OBB parser
|       |-- yolo26ncnn.cpp              # JNI interface
|       |-- ndkcamera.cpp               # Android NDK Camera2 path
|       `-- CMakeLists.txt
|-- scripts/                            # Export, diagnosis, regression, latency scripts
|-- data/                               # Dataset metadata and label files
|-- best.pt                             # SafeHat demonstration weights
`-- yolo26n.pt                          # YOLO26 reference weights
```

## Build Requirements

- Android Studio
- JDK 17+
- Android Gradle Plugin 8.7.3
- CMake 3.31.5 as configured in `app/build.gradle`
- Android NDK `29.0.14206865`
- Android device with API 24+ and Camera2 HAL3 support

The native build expects these dependency directories under `app/src/main/jni/`:

```cmake
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.13.0-android/sdk/native/jni)
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20260113-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
```

If they are not already present, download:

- NCNN Android Vulkan package from <https://github.com/Tencent/ncnn/releases>
- OpenCV-Mobile Android package from <https://github.com/nihui/opencv-mobile/releases>

Then extract them into `app/src/main/jni/` and keep the paths in `CMakeLists.txt` aligned with the extracted folder names.

## Model Export and Asset Replacement

The Android application loads NCNN `.param` and `.bin` pairs from `app/src/main/assets/`.

E2E task branches and the SafeHat O2M main case should be exported with different output-path settings and verified before asset replacement:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="ncnn", imgsz=640, half=False, end2end=True)

safehat = YOLO("best.pt")
safehat.export(format="ncnn", imgsz=640, half=False, end2end=False)
```

Project scripts used by the workflow include:

| Script | Role |
|---|---|
| `scripts/yolo_26_train.py` | Training or fine-tuning entry point |
| `scripts/export_yolo26_ncnn.py` | YOLO26-to-NCNN export helper |
| `scripts/export_best_safehat_to_assets.py` | SafeHat asset replacement helper |
| `scripts/diagnose_background_scores.py` | Empty-scene score inspection |
| `scripts/diagnose_model.py` | Tensor/value inspection |
| `scripts/verify_ncnn_output.py` | NCNN output check |
| `scripts/conversion_matrix_check.py` | Conversion consistency matrix check |
| `scripts/conversion_detail_check.py` | Graph and output-layout inspection |
| `scripts/test_letterbox_hypothesis.py` | Letterbox repair validation |
| `scripts/test_yolo26_e2e.py` | E2E parser regression validation |

## Minimal Regression Checklist

For a new YOLO-to-NCNN Android deployment:

1. Confirm that the Android app loads the expected `.param` and `.bin` pair.
2. Confirm that the output shape and class count match the native parser.
3. Run an empty-background input and record background score quantiles.
4. Run at least one target-present input and confirm a plausible score range.
5. Compare the same input between a desktop reference and Android-side NCNN when a symptom appears.
6. Inspect graph-level activations and output-column layout if value-range comparison is not enough.
7. Re-run the failing input and at least one previously passing input after every repair.

## Repository Scope

This public repository is intended to expose the Android implementation and reusable diagnosis workflow. Local writing drafts, generated paper figures, logs, temporary validation images, and machine-specific project files are intentionally excluded by `.gitignore` unless they are explicitly released as paper artifacts.

## Related Paper and Archive

- GitHub: <https://github.com/majun2019/ncnn-android-yolo26lt>
- Zenodo DOI: <https://doi.org/10.5281/zenodo.20137180>

Please cite the accompanying paper and archive if this workflow is useful in your deployment or diagnosis work.

## Dependencies

| Library | Version or role | Link |
|---|---|---|
| NCNN | `20260113` Android Vulkan package, CPU path used in measurements | <https://github.com/Tencent/ncnn> |
| OpenCV-Mobile | `4.13.0` Android mobile build | <https://github.com/nihui/opencv-mobile> |
| Ultralytics YOLO | Training and export workflow | <https://github.com/ultralytics/ultralytics> |
| Android NDK/JNI | Native Android integration | <https://developer.android.com/ndk> |

## License

BSD 3-Clause License
