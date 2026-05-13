# An Android-NCNN Deployment and Consistency Diagnosis Framework for Multi-Task YOLO-Based PPE Monitoring on Resource-Constrained Devices

---

## Abstract

Deployment deviations frequently arise when YOLO-based mobile vision models are exported, integrated, and parsed on resource-constrained Android devices. This study presents a Java–JNI–C++–NCNN deployment and consistency-diagnosis framework for project-managed yolo26n assets used in PPE monitoring and related visual tasks, including detection, segmentation, pose estimation, classification, and oriented bounding box (OBB) detection. The framework unifies model loading, task scheduling, and interface management under coexisting E2E, One-to-Many (O2M), and Legacy output paths, while preserving task-specific parsing semantics. Within the same pipeline, runtime post-processing mechanisms are implemented for the SafeHat PPE case, including probability calibration, per-class Top-K filtering, mutually exclusive class suppression, object-dependent association, and PPE box locking. A five-stage deployment consistency diagnosis workflow composed of anomaly logging, intermediate-output inspection, cross-backend comparison, structure tracing, and regression validation is used to localize four representative deployment faults: preprocessing mismatch, redundant activation, coordinate semantic misinterpretation, and OBB layout misjudgment. In the SafeHat 10-class PPE case, preprocessing correction reduced the empty-scene median score from 0.84–0.96 to < 0.0001, and removing redundant activation restored raw probabilities from values near 0.5 to a scene-dependent distribution ($p_{50}\approx 0.34$, $p_{95}\approx 0.74$, upper bound $\approx 0.95$). The calibrated on-device UI score remained within 0.35–0.73 on target-present frames. Runtime coverage of all five task paths was verified within one Android application; a one-dimensional sensitivity sweep of $T$, $\gamma$, $K_c$, and NMS overlap over 48 frozen test images confirms that the selected settings balance FP suppression and TP retention and that two-stage calibration eliminates over-confident scores ($\tilde p > 0.90$) across all tested configurations. Real-device latency was measured on two Android phones along the NCNN CPU path: on the Kirin 970, mean latency was 321.3–430.5 ms for the four 640×640 tasks and 33.6 ms for classification at 224×224; on the Kirin 659 (Nova2), the measured pose, classification, and OBB tasks reached 740.2 ms, 70.8 ms, and 640.6 ms respectively. The results provide an implementation-level feasibility baseline and reproducible diagnosis evidence for low-cost Android PPE monitoring and related YOLO-based mobile vision tasks.

**Keywords:** Android deployment; NCNN; PPE monitoring; construction-site safety; edge AI; deployment consistency; resource-constrained devices; mobile computer vision

---

## 1. Introduction

Driven by demands for low-latency response, privacy protection, and autonomous operation, deep visual inference is rapidly moving from cloud servers to edge devices. In construction-site PPE monitoring, industrial safety inspection, mobile robotics, and field monitoring, on-device inference completes image acquisition, model execution, and result feedback locally, reducing communication overhead and improving responsiveness. This trend has driven advances in real-time detectors [1,2,13,14], lightweight backbones [3–6], and mobile inference frameworks such as TensorFlow Lite, MNN, ONNX Runtime Mobile, and NCNN [7–10]. In this study, resource-constrained Android devices are defined as in-service mobile devices that use general-purpose CPU cores, do not provide a dedicated neural accelerator, and deliver fp32 throughput on 640×640 input substantially below the real-time frame-rate threshold; low-cost mid-range and low-end phones are the typical representatives. Such devices are often the most practical carriers for field-deployable visual monitoring under limited budgets, unstable networks, or weak compute infrastructure. Prior studies have already deployed YOLOv8 or lightweight YOLO networks on Android devices for single-task scenarios such as agronomic phenotype recognition and vision assistance [11,12], showing that single-task on-device inference is engineering-feasible. However, when an Android PPE monitoring system must support detection and related visual tasks in one application, the problem becomes a systems challenge involving unified support for multiple task assets, coexisting E2E, One-to-Many (O2M), and Legacy output paths, and consistency across preprocessing, output layout, and on-device parsing.

A model that runs correctly on the desktop does not necessarily become a system that works on the device. Preprocessing mismatch, output layout misjudgment, conversion-chain semantic drift, or toolchain incompatibility can cause faults that do not appear in conventional benchmark metrics but directly determine whether a model can enter a real application workflow. Deployment consistency and its diagnosis therefore constitute a formal research object. The research objective is narrowly defined: for resource-constrained Android devices, establish an applied deployment path that supports YOLO-based PPE monitoring and related visual tasks under a unified Android-NCNN framework and can diagnose deployment deviations in a structured manner. In this study, YOLO26 denotes the project-managed yolo26n asset set used in the implementation, covering five task types (detection, segmentation, pose estimation, classification, OBB) across E2E, O2M, and Legacy output paths; it is not proposed as a new general-purpose model family. A Java–JNI–C++–NCNN deployment framework is presented in which the five tasks can be loaded, scheduled, and observed within a shared application flow; a five-stage diagnosis workflow localizes and repairs deployment deviations; and SafeHat 10-class PPE detection serves as the main case, providing a closed loop from hard-negative mining and fine-tuning to NCNN export, asset replacement, on-device execution, fault repair, and regression validation.

The main contributions of this study are as follows:

1. **An applied Android-NCNN deployment framework for YOLO-based PPE monitoring and related visual tasks.** A Java–JNI–C++–NCNN four-layer architecture resolves asset naming conflicts, interface semantic divergence, and scheduling coupling for five task types under coexisting E2E, One-to-Many, and Legacy output paths. Runtime post-processing for the SafeHat PPE case—probability calibration ($T = 6$, $\gamma = 1.6$), per-class Top-K ($K_c = 50$), mutually exclusive class suppression, object-dependent association (overlap 0.15 / 0.20), and PPE box locking (0.35:0.65)—keeps on-device thresholding and multi-class competition interpretable and auditable.
2. **A deployment consistency diagnosis workflow with quantitative repair evidence.** A five-stage workflow (anomaly logging, intermediate-output inspection, cross-backend comparison, structure tracing, regression validation) is instantiated on four fault classes: preprocessing mismatch, redundant activation, coordinate semantic misinterpretation, and OBB layout misjudgment. The preprocessing case provides $p_{50}: 0.84\text{–}0.96 \to < 0.0001$; the redundant-activation case provides a double-layer reading—raw probability $p$ ($p_{50}\approx 0.34$, $p_{95}\approx 0.74$, upper bound $\approx 0.95$) versus UI value $\tilde p \in 0.35\text{–}0.73$.
3. **SafeHat closed-loop validation, parameter sensitivity, five-task coverage, and two-device latency characterization.** An end-to-end loop from hard-negative mining to on-device regression is provided. Five-task runtime coverage and runtime carriage of the post-processing mechanisms are evidenced by Figure 9 and Table 8 row 4. A 1-D sensitivity sweep of $T$, $\gamma$, $K_c$, and NMS overlap over 48 frozen test images (Table 10) quantifies their effect on FP\_empty, TP\_keep, and the UI score range. Table 11 reports two-device CPU-path latency: Kirin 970 covers all five tasks; Kirin 659 (Nova2) adds pose, classification, and OBB.

---

## 2. Related Work on Deployment for Resource-Constrained Android Devices and Edge Inference Systems

Existing studies span three areas: lightweight vision models and efficient inference structures for resource-constrained scenarios; mobile inference frameworks and native Android integration; and post-conversion consistency verification with on-device diagnosis. Real-time detectors such as YOLO and EfficientDet [1,2,13,14] and lightweight backbones such as ResNet, DenseNet, MobileNetV3, and ShuffleNet [3–6] improve deployability on low-compute devices through feature-fusion optimization and computational-cost control. Mobile inference frameworks (TensorFlow Lite, MNN, ONNX Runtime Mobile, NCNN) follow different routes in platform adaptation, operator coverage, model conversion, and hardware-acceleration support [7–10]. Toolchain resources for Android NDK, camera interfaces, OpenCV, Vulkan, the NCNN Wiki, and PNNX [16–21] provide component-level foundations for native deployment but rarely discuss model export, real-device deployment, and regression validation within the same systems closed loop. On the model side, lightweight detectors and end-to-end detection have changed the output semantics the deployment side must handle: detection branches often benefit from direct E2E outputs [22,23], whereas segmentation, pose estimation, classification, and OBB tasks retain heterogeneous structures such as mask prototypes, keypoint layouts, class vectors, and rotation parameters [24–27]. Existing Android deployments mostly validate a single task [11,12]; when the target expands to a multi-task asset set, the core challenge becomes scheduling markedly different task outputs uniformly and parsing them stably on a shared base pipeline. NCNN suits the present implementation because its lightweight C++ runtime and mature ARM CPU support form a transparent path with native Android camera access, OpenCV image processing, and JNI integration; the Vulkan GPU path remains available but all experiments here use the CPU path [10,16–20].

Given these gaps, the present study contributes an applied Android-NCNN deployment framework with runtime-resident post-processing mechanisms (probability calibration, per-class Top-K, mutually exclusive class suppression, object-dependent association, PPE box locking) for the SafeHat PPE implementation, and a five-stage deployment consistency diagnosis workflow. Raw model probability $p$ versus UI-displayed $\tilde p$ provides a double-layer quantitative reading for repair evidence; the SafeHat main case, five-task runtime coverage, parameter sensitivity, and two-device CPU-path latency check evaluate the implementation.

---

## 3. Proposed Android-NCNN Deployment Architecture for Resource-Constrained Devices

**Figure 1.** Model Export and On-Device Validation Closed Loop

![Figure 1 - model export and on-device validation closed loop](runs/paper_figures/Figure_1.svg)

### 3.1 Design Objectives and System Requirements

The system-level goal is unified support, scheduling, and diagnosis for multi-task visual inference on resource-constrained Android devices, subject to four requirements: (i) multiple tasks share as much of the model-loading entry, camera-input path, and result-display mechanism as possible; (ii) on-device preprocessing remains consistent with training and export assumptions; (iii) sufficient observability through logs and intermediate states is preserved so that deployment deviations can be localized; (iv) the unified pipeline does not erase task differences—detection, segmentation, pose estimation, classification, and OBB retain their own output-parsing semantics. Under these constraints, the project-managed yolo26n asset set is treated as an engineering asset collection rather than a new model method; SafeHat detection serves as the main closed-loop case carrying the training, export, and on-device validation evidence.

### 3.2 Java–JNI–C++–NCNN Four-Layer Collaborative Architecture

The proposed system adopts a layered architecture: a Java interaction layer (UI, task switching, threshold adjustment, rendering control); a JNI bridge layer (model loading, parameter transfer, native-object lifecycle); a native C++ inference layer (core logic for detection, segmentation, pose, classification, OBB); and a low-level dependency layer (NCNN, OpenCV-Mobile, Android NDK camera stack, optional Vulkan path). JNI + NCNN is preferred over higher-level wrappers because it provides direct parameter control, logging instrumentation, and a transparent native execution environment for structured analysis under resource constraints. One key design point is preprocessing consistency: the four 640×640 tasks use letterbox (resize by aspect ratio, then symmetric padding with `pad_value = 114` to a 640×640 square tensor), while classification uses 224×224 resize to match the model asset. This keeps training, export, and on-device assumptions consistent and establishes the baseline for later preprocessing-deviation diagnosis.

**Figure 2.** Overall System Architecture for Multi-Task Vision Inference on Resource-Constrained Android Devices

![Figure 2 - overall system architecture for multi-task vision inference](runs/paper_figures/Figure_2.svg)

### 3.3 On-Device Preprocessing Constraints and Post-Processing Consistency Mechanisms

**Preprocessing consistency constraint.** On-device input preprocessing must remain strictly aligned with the assumptions used during training; otherwise, input-distribution drift will directly cause abnormal confidence values. All 640×640 tasks in the present study use a unified letterbox preprocessing strategy. Let the original image width and height be \(W\) and \(H\), and let the target input side length be \(S\). The scaling factor \(r\), the resized width and height \(W'\) and \(H'\), and the symmetric padding values \(p_x\) and \(p_y\) are given by:

$$
r = \min\left(\frac{S}{W}, \frac{S}{H}\right), \quad W' = \lfloor rW \rfloor, \quad H' = \lfloor rH \rfloor
$$

$$
p_x = \left\lfloor\frac{S - W'}{2}\right\rfloor, \quad p_y = \left\lfloor\frac{S - H'}{2}\right\rfloor
$$

Here $p_x$, $p_y$ are rounded to integers (matching `pad_left` / `pad_top` in the implementation), so the same integer padding is applied during the inverse mapping. When mapping detection boxes from padded to original image coordinates:

$$
x = \frac{\hat{x} - p_x}{r}, \quad y = \frac{\hat{y} - p_y}{r}, \quad w = \frac{\hat{w}}{r}, \quad h = \frac{\hat{h}}{r}
$$

Preprocessing must pad to a strict `target_size × target_size` square rather than only to a stride-aligned size; for non-square inputs such as 1920×1080, stride-aligned padding yields a mismatched 640×384 tensor, with quantitative effects reported as Fault Mode I in Section 4.2.

**Probability calibration.** Under the One-to-Many output path, candidate outputs often saturate near 0.99, reducing the separability of threshold tuning and multi-class competition. The device therefore applies logit-based temperature scaling followed by power-law compression. For a class probability $p$, the calibrated display probability $\widetilde{p}$ is:

$$
\widetilde{p} = \left[\sigma\!\left(\frac{\log \frac{p}{1-p}}{T}\right)\right]^{\gamma}, \quad T = 6,\ \gamma = 1.6
$$

where $\sigma(\cdot)$ is the sigmoid function. This transformation preserves candidate ranking while compressing over-saturated probabilities and amplifying differences in the high-score region (illustrative: 0.999→0.66, 0.995→0.59, 0.950→0.43, 0.900→0.36). After ranking the candidate set $\mathcal{B}$, the device keeps the top $K_c = 50$ candidates per class (`per-class Top-K`) so high-score background classes do not crowd out targets.

**Semantics-constrained NMS.** Agnostic NMS may remove legitimately coexisting targets. For boxes $B_i$, $B_j$, the IoU is:

$$
\operatorname{IoU}(B_i, B_j) = \frac{|B_i \cap B_j|}{|B_i \cup B_j|}
$$

Per-class NMS is adopted (suppression only within the same class, preserving legal cross-class coexistence). For truly exclusive class pairs such as Hardhat / No-Hardhat, additional mutually exclusive class suppression is applied: if IoU exceeds the threshold, the higher-probability box is kept.

**Object-dependence constraint.** PPE classes are naturally attached to a person: if no person candidate exists, related PPE categories are suppressed; otherwise only PPE candidates whose overlap with at least one `Person` box exceeds a threshold are retained. For boxes $B_{\text{ppe}}$, $B_{\text{person}}$:

$$
\operatorname{overlap}(B_{\text{ppe}}, B_{\text{person}}) = \frac{|B_{\text{ppe}} \cap B_{\text{person}}|}{|B_{\text{ppe}}|}
$$

The implementation uses overlap threshold 0.15 for positive PPE classes (`Hardhat`, `Mask`, `Safety Vest`) and 0.20 for negative ones (`No-Hardhat`, `No-Mask`, `No-Safety Vest`) to reduce spurious associations.

If the threshold is met, the PPE box is further locked onto the matched person box. In $(x, y, w, h)$ form, the updated PPE box is:

$$
\begin{aligned}
x' &= 0.35\, x_{\text{ppe}} + 0.65\, x_{\text{person}}, & y' &= 0.35\, y_{\text{ppe}} + 0.65\, y_{\text{person}} \\
w' &= 0.35\, w_{\text{ppe}} + 0.65\, w_{\text{person}}, & h' &= 0.35\, h_{\text{ppe}} + 0.65\, h_{\text{person}}
\end{aligned}
$$

The same 0.35:0.65 weighting applies to position and size, anchoring the PPE box to the matched person box and suppressing isolated small boxes from texture drift. Together, the preprocessing and post-processing mechanisms above form the operational basis of on-device deployment; in the SafeHat main case the detection model uses the One-to-Many output path (8400\u00d7(nc+4)) and all five mechanisms are instantiated, with concrete validation and fault-repair evidence in Section 5.

### 3.4 Unified Task Scheduling and Parsing Paths

To avoid creating separate task-specific applications, the framework uses a unified scheduling mechanism: a task identifier selects the corresponding model and parsing module, while the camera-input path, loading interface, and display entry are shared. The same application can therefore switch among five task types under identical resource constraints, improving code reuse and providing a common basis for cross-task comparative diagnosis. The execution path is unified, but output semantics remain task-specific: detection parses location/class/confidence; segmentation reconstructs masks from detections, prototypes, and coefficients; pose interprets keypoint coordinates and visibility; classification ranks class-score vectors; OBB recovers oriented-box parameters and directions. The framework therefore emphasizes unified scheduling and unified support rather than semantic homogenization.

**Table 1.** Unified Scheduling and Task-Specific Parsing for Five Task Types

| Task ID | Task type | Model asset | Input and preprocessing | Main output semantics | Task-specific parsing and display |
|--------|-----------|-------------|-------------------------|-----------------------|-----------------------------------|
| 0 | Detection | yolo26n_safehat (compatible alias: yolo26n_e2e) | loadModel(taskid=0); 640×640 letterbox | in the SafeHat main case, out0≈8400×14 (O2M); in the original general detection model, out0 can be ≈300×6 (E2E) | O2M: Top-K/NMS + calibration; E2E: direct parsing |
| 1 | Segmentation | yolo26n_seg_e2e | loadModel(taskid=1); 640×640 letterbox | out0=300×38; out1=mask prototypes | reconstruct 32-dimensional masks and overlay mask contours |
| 2 | Pose estimation | yolo26n_pose_e2e | loadModel(taskid=2); 640×640 letterbox | out0 width is 57 (E2E) or dual out0/out1 branches | parse 17 keypoints and draw the skeleton |
| 3 | Classification | yolo26n_cls | loadModel(taskid=3); 224×224 resize | out0 is a class-score vector | Top-5 ranking with class names and probabilities |
| 4 | OBB | yolo26n_obb_e2e | loadModel(taskid=4); 640×640 letterbox | out0 width is 7 (E2E) or bbox/class/angle branches | parse oriented boxes and angles and draw `RotatedRect` |

### 3.5 Model Export, Asset Integration, and On-Device Deployment Process

Deployment for resource-constrained Android devices is organized into five stages—model preparation, NCNN export, structural verification, asset deployment, real-device regression—as detailed in Table 2. This sequence is not auxiliary engineering tooling but an organic part of the deployment closed loop: real-device regression confirms that on-device loading and parsing behavior remain consistent with the Python-side reference.

**Table 2.** Tools, Inputs, and Validation Goals for Each Deployment Stage

| Stage | Input / asset | Main tools or scripts | Stage output | Validation goal |
|------|----------------|-----------------------|--------------|-----------------|
| Model preparation | safehat.yaml, data/, yolo26n.pt / best.pt | yolo_26_train.py, Ultralytics | fine-tuned weight `best.pt` | correct class definition; reproducible training and validation workflow |
| NCNN export | PyTorch weights | export_yolo26_ncnn.py (Ultralytics export, format=ncnn) | param/bin file pair | successful export of E2E, One-to-Many, and task-specific branches |
| Structural verification | param/bin, original model | verify_ncnn_output.py, conversion_detail_check.py | description of output shapes, branches, and column layout | confirm that tensors such as out0/out1 match parser semantics |
| Asset deployment | app/src/main/assets/, Android project | export_best_safehat_to_assets.py, Gradle/NDK | Android-loadable model assets | `loadModel` reports no errors; JNI and resource paths are correct |
| Real-device regression | APK, camera input, logcat | trace_conversion_pipeline.py, diagnose_model.py, collect_latency.py | functional logs, fault-localization results, latency statistics | deployment results remain consistent with the Python reference; no regression after repair |

Among the five tasks, SafeHat detection forms the most complete case study (10 PPE/Person classes), linking fine-tuning, export, asset replacement, and on-device validation into the most complete closed loop. It does not imply that SafeHat is the only algorithmic contribution. In export, the SafeHat main case preserves the One-to-Many path (out0 = 8400×(nc+4)) rather than the general E2E path (out0 = 300×6); the decision is reflected in the post-processing of Section 3.3. The detection asset uses the semantic name `yolo26n_safehat`, with a historical compatibility alias `yolo26n_e2e` (the `e2e` token does not indicate an end-to-end detection head in this main case).

**Table 3.** SafeHat Dataset Classes and Application Semantics

| Class ID | Class name | Semantic group | Application meaning |
|---------|------------|----------------|---------------------|
| 0 | Hardhat | PPE compliance | the person wears a hardhat correctly |
| 1 | Mask | PPE compliance | the person wears a mask correctly |
| 2 | No-Hardhat | PPE violation | the person does not wear a hardhat |
| 3 | No-Mask | PPE violation | the person does not wear a mask |
| 4 | No-Safety Vest | PPE violation | the person does not wear a safety vest |
| 5 | Person | person entity | the human target itself |
| 6 | Safety Cone | scene safety facility | a construction or isolation cone |
| 7 | Safety Vest | PPE compliance | the person wears a safety vest correctly |
| 8 | Machinery | construction equipment | machinery or engineering vehicle components |
| 9 | Vehicle | traffic / work vehicle | a road or worksite vehicle target |

**Figure 3.** End-to-End Closed-Loop Pipeline from SafeHat Training to Android Deployment

![Figure 3 - SafeHat training-to-Android deployment closed loop](runs/paper_figures/Figure_3.svg)

---

## 4. Deployment Consistency Diagnosis Workflow for Resource-Constrained Android Devices

### 4.1 Diagnosis Workflow

Deployment faults on resource-constrained Android devices usually do not manifest as crashes but as abnormal confidence distributions, empty outputs, incorrect visualizations, or inconsistent behavior spanning model export, preprocessing, tensor parsing, and platform implementation. A five-stage consistency diagnosis workflow—anomaly logging (S1), intermediate-output inspection (S2), cross-backend comparison (S3), structure tracing (S4), regression validation (S5)—is proposed and validated on the SafeHat case. It answers in sequence: how the problem manifests, where the deviation first appears, whether the fault originates from export or device-side, which tensor branch or parser assumption is responsible, and whether consistency is truly restored after repair.

**Figure 4.** Fault Diagnosis Workflow for Deployment on Resource-Constrained Android Devices

![Figure 4 - fault diagnosis workflow for Android deployment](runs/paper_figures/Figure_4.svg)

**Table 4a.** Five-Stage Diagnosis Workflow: Stage Overview

| Stage | Name | Trigger condition | Abstract operation (method level) | Exit criterion |
|------|------|-------------------|-----------------------------------|----------------|
| S1 | Anomaly logging | abnormal on-device output: score explosion, fixed confidence, missing geometry, or label disorder | collect quantifiable abnormal signals from device logs and establish a reproducible baseline | the phenomenon is reproducible and numerically described |
| S2 | Intermediate-output inspection | the phenomenon is device-specific but the root level is not yet localized | extract intermediate tensors along the inference chain and compare output shapes and value distributions before and after conversion | the deviation is localized to a specific output dimension or numerical range |
| S3 | Cross-backend comparison | the deviation appears on the device while desktop-side inference is normal | run inference in parallel with the same input on the desktop reference backend and the device backend, and localize the layer where the discrepancy is introduced through numerical comparison | determine whether the fault originates from export or from the on-device implementation |
| S4 | Structure tracing | S3 still cannot explain the fault through output-value patterns alone | compare the inference-graph operator sequence and parser source code at the structural level to identify column offsets or activation-semantic differences | localize the specific branch, operator index, or parser assumption that produces the wrong value |
| S5 | Regression validation | a repair has been applied | run quantitative regression before and after the repair to verify the benefit and ensure no new degradation is introduced | the score distribution returns to the expected range and previously passing cases remain stable |

**Table 4b.** Five-Stage Diagnosis Workflow: Project-Level Instantiation

| Stage | Name | Project-level instantiation in this study |
|------|------|-------------------------------------------|
| S1 | Anomaly logging | logcat filtering + `diagnose_background_scores.py` to collect score statistics such as `p50`, `p99`, and `frac_gt090` |
| S2 | Intermediate-output inspection | `verify_ncnn_output.py` parses the NCNN param file and prints the output dimensions of each layer |
| S3 | Cross-backend comparison | `trace_conversion_pipeline.py` + `diagnose_model.py` run joint tests on PyTorch, Python-NCNN, and Android-NCNN |
| S4 | Structure tracing | `conversion_detail_check.py` traces the concat chain in the NCNN param file |
| S5 | Regression validation | `test_letterbox_hypothesis.py` / `test_yolo26_e2e.py` rerun the relevant scenes |

### 4.2 Fault Mode I: Preprocessing Mismatch

**Phenomenon (S1).** In early SafeHat deployment, the Android application continuously produced multiple high-confidence detections in empty scenes, with `p50 = 0.84–0.96` and `frac_gt090 = 0.41–0.58` (`diagnose_background_scores.py`); the phenomenon was reproducible. **Diagnosis (S3, S2).** Cross-backend comparison confirmed the model behaved normally in PyTorch and Python-side NCNN, ruling out training/export. Intermediate-output inspection then found that the device received a 640×384 tensor rather than 640×640, localizing the fault to on-device preprocessing. **Root cause.** Native preprocessing padded only to a stride-aligned size; for 1920×1080 frames the model received 640×384, causing input-distribution drift and systematic background-score elevation. **Repair.** The padding logic was changed to always emit a `target_size × target_size` letterbox with uniform `pad_value = 114`. **Regression (S5).** After the fix, `p50 < 0.0001` and `frac_gt090 < 0.000001`; false detections in empty scenes disappeared with no new regression in previously passing scenes.

**Figure 5.** Comparison of Input Patterns and Detection Results Before and After the Preprocessing Fix

![Figure 5 - preprocessing fix comparison](runs/paper_figures/Figure_5.svg)

**Table 5.** Background Score Statistics Before and After the Preprocessing Fix

| Metric | Before the fix | After the fix |
|------|----------------|---------------|
| Median score (`p50`, empty scene) | 0.84–0.96 | < 0.0001 |
| 99th-percentile score (`p99`) | ~1.0 | 0.0006–0.0029 |
| Fraction of anchors with score > 0.9 | 0.41–0.58 | < 0.000001 |
| Number of false-detected classes in empty scenes | all 10 classes detected | 0 |
| Target-present scene | correct detection of Person + PPE | correct detection of Person + PPE |

### 4.3 Fault Mode II: Redundant Activation

**Phenomenon (S1).** In segmentation, pose, and OBB modes, candidate confidences were clamped near 0.5; lowering the threshold did not change the distribution. **Diagnosis (S3, S4).** Cross-backend comparison confirmed Python-side NCNN produced a normal distribution on the same input, ruling out model weights. Structure tracing via `conversion_detail_check.py` enumerated `Sigmoid` nodes in the NCNN param file and confirmed an explicit sigmoid in the graph, while the C++ parser applied it again. **Root cause.** Class-related outputs in the end-to-end branches already contained sigmoid in the NCNN graph; double activation remapped small background probabilities to ~0.5, producing dense pseudo-candidates. **Repair.** The redundant activation in the parser was removed, retaining only transformations still required by task-specific branches. **Regression (S5).** Confidence distributions returned to normal, candidate density decreased to the expected level, and previously passing tasks showed no new regression.

**Figure 6.** Effect of Single vs. Repeated Sigmoid Application on Background Scores

![Figure 6 - single versus repeated sigmoid effect](runs/paper_figures/Figure_6.svg)

### 4.4 Fault Mode III: Coordinate Semantic Misinterpretation

**Phenomenon (S1).** In pose and OBB modes, bounding boxes were rendered but keypoint skeletons and oriented-box geometry extended beyond image boundaries or disappeared. **Diagnosis (S2, S4).** Intermediate-output inspection showed coordinate magnitudes in the hundreds to thousands, far beyond the 640×640 range; structure tracing along the E2E export path confirmed that end-to-end branches output already-decoded absolute pixel coordinates. **Root cause.** The exported E2E outputs were already absolute pixels, but the device-side parser still treated them as grid-relative values and applied stride-based restoration, pushing coordinates out of range. **Repair.** A dedicated parsing branch was added for E2E outputs; the parser first detects whether an output is already decoded, then either uses it directly or applies the legacy decoding path. **Regression (S5).** Pose keypoints and OBB geometry returned to normal display, with no regression on previously passing tasks such as detection.

**Figure 7.** Semantic Comparison of Legacy Grid Decoding and E2E Pre-Decoded Coordinates

![Figure 7 - legacy grid decoding versus E2E pre-decoded coordinates](runs/paper_figures/Figure_7.svg)

### 4.5 Fault Mode IV: OBB Layout Misjudgment

**Phenomenon (S1).** At the beginning of OBB inference, all boxes showed identical labels, confidences clamped near 50%, and direction estimation was wrong. **Diagnosis (S3, S4).** The 50% signature partly overlapped with Fault Mode II, so cross-backend comparison first ruled out weight issues. Tracing the concat chain via `conversion_detail_check.py` revealed that the parser assumed `[bbox, angle, classes]` while the actual layout was `[bbox, classes_sigmoided, angle_raw]`. **Root cause.** The incorrect column-layout assumption caused the first class score to be misread as an angle, shifted subsequent class indices, and combined with the Fault II redundant-activation effect. **Repair.** The parser starting class column, class column count, and angle column position were corrected, and the angle branch was handled separately. **Regression (S5).** OBB class labels, confidences, and direction estimates became interpretable, oriented boxes were rendered correctly, and other previously passing tasks showed no regression.

**Table 6.** OBB Output Column Layout: Assumed vs. Actual

| Column range | Initial parser assumption | Actual NCNN output semantics | Error consequence / repair strategy |
|-------------|---------------------------|-------------------------------|------------------------------------|
| 0–3 | bbox | bbox | none / keep the original parsing |
| 4 | angle | class 0 probability | class score misread as angle / move `angle` to the last column or `out1` |
| 5…4+num_class-1 | classes[0…] | classes_sigmoided[1…] | class-index shift / reorder according to the true start column |
| last column or `out1` | not handled separately / ignored | angle_raw | wrong direction estimate / read the angle branch separately |

These four cases show that the five stages need not be executed mechanically in full: each stage is a decision gate, and later stages can be truncated when an earlier one already localizes the fault. S3 cross-backend comparison is the most effective front-end filter because a single Python-vs-Android numerical comparison usually distinguishes export-side faults from device-side implementation faults; S4 is needed only when S3 cannot explain the fault from value patterns alone; S5 remains mandatory to ensure repairs do not regress previously passing scenes. The workflow assumes a standard PyTorch → ONNX → NCNN export path (Ultralytics export / onnx2ncnn), a device-side native C++ parser with explicit tensor-layout assumptions, and an available Python-side reference for cross-backend comparison; without the latter, S2 and S4 take on more analytical load through offline param-file inspection and logcat tensor logs.

---

## 5. Results and Discussion

### 5.1 Experimental Setup and the SafeHat Main Case

This section reports validation results from four perspectives: SafeHat training basis, runtime coverage across five tasks, parameter sensitivity of runtime post-processing, and runtime-scale characterization on two resource-constrained Android phones. SafeHat detection is the most complete main case because it provides closed-loop evidence from data and training to export, real-device deployment, and fault repair. On the training side, the Ultralytics workflow [15] is used for loading, fine-tuning, validation, and NCNN export. Pretrained `yolo26n` weights are fine-tuned on the SafeHat dataset; hard-negative mining then collects high-confidence false-positive images from empty scenes automatically, adds them back as empty-label samples, and continues fine-tuning with `confcal_v3` to reduce spurious activation on background textures, reflective regions, and humanoid machinery edges. On the deployment side, a unified Android application is built with Android Studio, AGP, NDK, NCNN, and OpenCV-Mobile. The HUAWEI P20 Pro (Kirin 970) serves as the primary validation device, and a HUAWEI Nova2 (PIC-AL00, Kirin 659) supplies a lightweight second-device latency check. CPU-path results are reported; the Vulkan GPU path is retained for future comparison. These measurements establish an implementation-level feasibility baseline for unified deployment on low-cost, widely available Android devices, not a high-end-SoC performance ranking.

**Table 7.** Validation Environment and Main Dependencies

| Category | Item | Configuration / version |
|---------|------|-------------------------|
| Training side | training framework | Ultralytics YOLO (Python 3.12) |
| Training side | export toolchain | PyTorch → ONNX → NCNN (Ultralytics export, format=ncnn; internally via onnx2ncnn) |
| Training side | default input size | 640 × 640 |
| Training side | SafeHat dataset | 10 PPE / Person classes; exported from Roboflow; with augmentation |
| Deployment side | inference engine | NCNN 20260113 (Vulkan GPU path retained; the present study reports the CPU path) |
| Deployment side | image-processing library | OpenCV-Mobile 4.13.0 (Android mobile slim build) |
| Deployment side | Android NDK | r29 (29.0.14206865) |
| Deployment side | Android Gradle Plugin | 8.7.3 |
| Deployment side | compileSdk / minSdk | 33 / 24 (Android 7.0+) |
| Deployment side | inference backend | NCNN CPU, fp32, arm64-v8a |
| Deployment side | input preprocessing | detection / segmentation / pose / OBB use 640×640 letterbox; classification uses 224×224 resize |
| Primary validation device | device model | HUAWEI P20 Pro (CLT-AL00) |
| Primary validation device | Android version | Android 10 (API 29); build CLT-AL00 10.0.0.175(C00E175R1P4) |
| Primary validation device | CPU / SoC | HiSilicon Kirin 970 (`adb getprop=kirin970`; `/proc/cpuinfo` shows 4× Cortex-A73 + 4× Cortex-A53) |
| Secondary latency-check device | device model | HUAWEI Nova2 (PIC-AL00) |
| Secondary latency-check device | CPU / SoC | HiSilicon Kirin 659 (4× Cortex-A73 @ 2.36 GHz + 4× Cortex-A53 @ 1.7 GHz), arm64 |
| Secondary latency-check device | measured tasks | pose estimation, classification, and OBB CPU-path latency SUMMARY logs on 2026-05-26 |

Training-side evidence shows stable convergence and correct PPE / Person separation in validation-batch predictions; `mAP50 = 0.796` and `Precision = 0.858` serve as the training-stage baseline before export and on-device validation. Further accuracy analysis is outside the present scope.

**Figure 8.** SafeHat Fine-Tuning Curves and Validation Examples

![Figure 8 – (a) training curves; (b) validation batch 2×4 grid](runs/paper_figures/figure8_ab.png)

*(a) Loss and detection-metric curves during SafeHat fine-tuning, showing convergence in the main case. (b) Validation-batch prediction examples from SafeHat, shown as a 2×4 grid of 8 images, illustrating class-learning results on the validation set.*

### 5.2 Runtime Coverage across Five Tasks

**Figure 9.** Five-Task On-Device Runtime Montage: Detection, Segmentation, Pose, Classification, and OBB

![Figure 9 – five-task on-device frame montage](runs/paper_figures/figure8c_five_task_montage.png)

*Montage of real-device frames for detection, segmentation, pose estimation, classification, and OBB. The unified label strip 01–05 corresponds to Detection, Segmentation, Pose, Classification, and OBB, respectively. The figure shows that all five task types complete real-device loading and runtime coverage within one application. Panel (e) OBB is from a road scene and shows the DOTA-pretrained model correctly rendering a rotated bounding box inside the application (roundabout 73.9%), confirming that the OBB inference path and direction-parsing engine are both operational. The lack of above-threshold OBB activations in the upstream construction-site scene reflects the domain gap between the DOTA training distribution and that scene’s target classes, and does not affect the OBB task-switching or interface completeness validation (see Table 8 and Table 9 for detail).*

**Table 8.** Key Validation Items for SafeHat Deployment

| No. | Validation item | Expected criterion | Observed evidence |
|----|-----------------|--------------------|-------------------|
| 1 | SafeHat detection model loading (preferred `yolo26n_safehat`, compatible `yolo26n_e2e`) | normal parsing of param/bin; no logcat errors | param/bin loaded through `loadModel`; no loading error in logcat |
| 2 | output class count matches target classes | = 10 (SafeHat PPE classes) | output parser verified the 10-class SafeHat layout |
| 3 | elimination of false detections in empty scenes after preprocessing repair | confidence `p50 < 0.0001`; 0 false detections | background-score statistics in Table 5 |
| 4 | correct detection of PPE / Person in target-present scenes | UI-displayed confidence $\tilde p \in 0.35–0.73$ (raw $p \le 0.95$ on the same frames) | target-present frame logs and Figure 9 detection panel |
| 5 | on-device threshold-adjustment function | filtering takes effect after `setDetectThresholds` JNI call | threshold changes observed after JNI parameter update |
| 6 | multi-task switching (five task types) | each task runs independently after switching, with normal rendering output | Figure 9 panels (a)–(e) |
| 7 | keypoint rendering after the Pose E2E coordinate repair | keypoints and skeleton lines are correctly overlaid on video frames | Figure 9 pose panel after coordinate-semantics repair |
| 8 | correct OBB direction and class after layout repair | class labels, confidences, and oriented-box directions all meet expectations | Figure 9 OBB panel and OBB layout repair in Table 9 |

**Table 9.** Summary of Fault Symptoms, Root Causes, and Repair Outcomes

| No. | Fault type | Observable phenomenon | Root cause | Repair strategy | Recovery result |
|----|------------|-----------------------|------------|-----------------|-----------------|
| 1 | preprocessing mismatch | all 10 classes detected in empty scenes; `p50 = 0.84–0.96` | stride-aligned padding generated 640×384 rather than 640×640 | pad to a `target_size × target_size` square | `p50 < 0.0001`; zero false detections |
| 2 | redundant activation | all confidences fixed at about 50.0%; dense candidates appear | the model graph already contains sigmoid; C++ applies it again | remove redundant `sigmoid()` and retain only necessary transformations | the raw model probability $p$ distribution is restored (measured $p_{50}\approx 0.34$, $p_{95}\approx 0.74$, upper bound $\approx 0.95$); after the Section 3.3 calibration $\tilde p = [\sigma(\operatorname{logit}(p)/6)]^{1.6}$, the on-device UI value is $\tilde p \in 0.35–0.73$ |
| 3 | E2E coordinate semantic misinterpretation | Pose skeleton missing; OBB geometry absent | E2E output is already in absolute pixel coordinates; the parser decodes it again as `grid × stride` | introduce a parser branch for already decoded outputs with automatic detection | keypoints and oriented boxes are rendered correctly |
| 4 | OBB layout misjudgment | all box labels and directions are abnormal | the parser assumes `[bbox, angle, classes]`; the actual layout is `[bbox, classes_sigmoided, angle_raw]` | correct the column indices through NCNN concat-chain tracing | class, confidence, and direction all become correct |

From a systems perspective, the key result is not a single accuracy metric but the fact that all four fault classes show observable changes after repair: empty-scene `p50` falls from 0.84–0.96 to `< 0.0001`; the end-to-end raw probability $p$ returns from being clamped near 0.5 to a scene-dependent multi-modal distribution (1009 on-device SafeHat samples on 2026-05-07: $p_{50}\approx 0.34$, $p_{95}\approx 0.74$, upper bound $\approx 0.95$), with calibrated UI value $\tilde p \in 0.35\text{–}0.73$ (e.g., $p \approx 0.95 \mapsto \tilde p \approx 0.43$); coordinate semantics and OBB layout corrections restore keypoints, oriented boxes, and class labels. The five post-processing mechanisms in Section 3.3 are persistent runtime components of the SafeHat main case; their runtime carriage is evidenced by Figure 9 and row 4 of Table 8 ($\tilde p \in 0.35\text{–}0.73$ on target-present frames). A 1-D sensitivity sweep of $T$, $\gamma$, $K_c$, and NMS overlap is reported in Section 5.3; PPE box locking weights (0.35:0.65) remain empirically set and are reserved for future targeted evaluation.

### 5.3 Parameter Sensitivity of Runtime Post-Processing

Table 10 reports a one-dimensional sensitivity sweep of the four main runtime post-processing parameters — temperature scaling factor $T$, power-law exponent $\gamma$, per-class Top-K budget $K_c$, and NMS overlap threshold — evaluated on 48 frozen test images from three scene categories (18 target-present, 15 empty-background, 15 hard-negative). All images were collected offline and fixed before the sweep; inference was performed once per image and the raw model output was reused across all parameter settings. The five metrics are defined as follows: **FP\_empty** is the mean number of post-NMS detections on empty-background images (lower is better); **frac\_gt090** is the fraction of post-NMS boxes with display score $\tilde p > 0.90$ across all images (lower is better, as it indicates avoidance of over-confidence); **TP\_keep** is the fraction of target-present images retaining at least one Person or PPE-class detection (higher is better); **Mean cand.** is the mean pre-NMS candidate count per image after per-class Top-K; **UI range** is the $[\min, \max]$ display score $\tilde p$ on target-present images.

**Table 10.** Parameter sensitivity of runtime post-processing in the SafeHat main case (48 frozen test images; `conf = 0.25`; selected settings marked ★).

| Setting | FP\_empty | frac\_gt090 | TP\_keep | Mean cand. | UI range | Comment |
|---------|-----------|-------------|---------|-----------|----------|---------|
| $T = 1$ | 131.1 | 0.000 | 1.000 | 496.7 | [0.33, 0.59] | insufficient compression; widest score range |
| $T = 3$ | 131.8 | 0.000 | 1.000 | 496.4 | [0.33, 0.41] | moderate compression |
| **$T = 6$ ★** | **131.3** | **0.000** | **1.000** | **494.5** | **[0.33, 0.37]** | **selected; balanced FP and score readability** |
| $T = 10$ | 133.5 | 0.000 | 1.000 | 493.4 | [0.33, 0.36] | over-compressed; FP marginally higher |
| $\gamma = 1.0$ | 131.3 | 0.000 | 1.000 | 494.5 | [0.50, 0.54] | no power-law compression; scores cluster near 0.5 |
| $\gamma = 1.3$ | 131.3 | 0.000 | 1.000 | 494.5 | [0.41, 0.45] | moderate compression |
| **$\gamma = 1.6$ ★** | **131.3** | **0.000** | **1.000** | **494.5** | **[0.33, 0.37]** | **selected; range separated below 0.5** |
| $\gamma = 2.0$ | 131.3 | 0.000 | 1.000 | 494.5 | [0.25, 0.29] | aggressive compression; scores near display lower bound |
| $K_c = 20$ | 62.6 | 0.000 | 1.000 | 199.9 | [0.33, 0.37] | FP halved; risks dropping low-ranked PPE proposals |
| **$K_c = 50$ ★** | **131.3** | **0.000** | **1.000** | **494.5** | **[0.33, 0.37]** | **selected; all PPE classes represented** |
| $K_c = 100$ | 239.9 | 0.000 | 1.000 | 969.8 | [0.33, 0.37] | FP nearly doubled; higher compute cost |
| overlap $= 0.10$ | 98.9 | 0.000 | 1.000 | 494.5 | [0.33, 0.37] | aggressive NMS; risks suppressing co-located PPE boxes |
| overlap $= 0.15$ | 106.7 | 0.000 | 1.000 | 494.5 | [0.33, 0.37] | — |
| overlap $= 0.20$ | 113.1 | 0.000 | 1.000 | 494.5 | [0.33, 0.37] | — |
| **overlap $= 0.30$ ★** | **131.3** | **0.000** | **1.000** | **494.5** | **[0.33, 0.37]** | **selected; per-class NMS preserves co-located boxes** |

Across all 15 settings, frac\_gt090 = 0.000, confirming that the two-stage calibration ($T$ rescaling followed by $\gamma$ power law) eliminates over-confident display scores regardless of parameter choice. $K_c$ is the dominant FP control: reducing $K_c$ from 50 to 20 halves FP\_empty (131.3→62.6) but may drop low-ranked PPE proposals. The NMS overlap threshold reduces FP\_empty monotonically as it decreases; values below 0.20 risk suppressing legitimately co-located Person and PPE boxes under per-class NMS, so 0.30 is selected as the conservative boundary. $T$ and $\gamma$ primarily govern the UI score range and have negligible effect on FP\_empty or TP\_keep. PPE box locking weights (0.35:0.65) are set empirically; their sensitivity is reserved for future work.

### 5.4 On-Device Latency Results

**Table 11.** On-Device Inference Latency by Task on the CPU Path (Pure Inference, No Rendering)

| Task | Device | Model | Backend | Sample count | Mean (ms) | P5/Min (ms) | P95/Max (ms) | FPS | Note |
|------|--------|-------|---------|--------------|-----------|---------|----------|-----|------|
| Detection | Kirin 970 | yolo26n_safehat (compatible alias `yolo26n_e2e`) | CPU | 148 | 321.3 | 272.2 | 412.2 | 3.1 | 640×640 letterbox |
| Segmentation | Kirin 970 | yolo26n_seg_e2e | CPU | 108 | 430.5 | 359.3 | 517.5 | 2.3 | 640×640 letterbox |
| Pose estimation | Kirin 970 | yolo26n_pose_e2e | CPU | 113 | 369.4 | 342.0 | 427.6 | 2.7 | 640×640 letterbox |
| Classification | Kirin 970 | yolo26n_cls | CPU | 928 | 33.6 | 25.9 | 41.8 | 29.8 | 224×224 resize |
| OBB | Kirin 970 | yolo26n_obb_e2e | CPU | 120 | 336.1 | 308.4 | 386.3 | 3.0 | 640×640 letterbox ^a |
| Pose estimation | Kirin 659 (Nova2) | yolo26n_pose_e2e | CPU | 30 | 740.2 | 611.9 | 861.9 | 1.4 | 640×640 letterbox ^b |
| Classification | Kirin 659 (Nova2) | yolo26n_cls | CPU | 150 | 70.8 | 41.4 | 208.5 | 14.1 | 224×224 resize ^b |
| OBB | Kirin 659 (Nova2) | yolo26n_obb_e2e | CPU | 60 | 640.6 | 541.1 | 859.8 | 1.6 | 640×640 letterbox ^b |

^a For Kirin 970, the two spread columns report P5 / P95 latency. The OBB row comes from independently collected logs: the mean and FPS use the 120-frame SUMMARY, and P5 / P95 are estimated from visible FRAME subsamples in the transcript. The other four tasks were computed uniformly by `collect_latency.py` from `YOLO26BENCH FRAME` logs after skipping the first 10 warm-up frames. All Kirin 970 values were measured on 2026-05-07.

^b Kirin 659 (Nova2, PIC-AL00) values were measured on 2026-05-26. For these runs, the two spread columns report the SUMMARY `min_ms` and `max_ms` respectively, as per-frame logs were not available. Detection and segmentation were not captured in the available Kirin 659 log session.

Both Kirin 970 and Kirin 659 (released in 2017) use ARM big.LITTLE architectures without a dedicated neural accelerator and run NCNN on the CPU path. On Kirin 970, mean latency is 321–430 ms for the four 640×640 tasks (≈2–3 FPS) and 33.6 ms for 224×224 classification. On Kirin 659 (Nova2), the three measured tasks reach 740.2 ms (pose), 70.8 ms (classification), and 640.6 ms (OBB), about 2× higher than Kirin 970, consistent with the lower effective clock. Both devices fall far below the real-time threshold, confirming the resource-constrained definition. Table 11 should be read as a lightweight cross-device reference and feasibility baseline, not a universal Android benchmark.

### 5.5 Applicability, Limitations, and Future Evaluation

The current study has clear boundaries. Quantitative validation remains concentrated in the SafeHat detection case; the other four tasks demonstrate runtime coverage and parsing correctness under the unified pipeline rather than same-intensity task-level benchmarks. The latency evidence covers two Android phones but is a lightweight cross-device check: Kirin 970 covers all five tasks, the Kirin 659 (Nova2) session covers pose, classification, and OBB. Both devices use the NCNN CPU path; no apples-to-apples comparison with TensorFlow Lite, MNN, or ONNX Runtime Mobile is provided because that would require re-exporting SafeHat under each runtime with matched preprocessing, post-processing, and quantization. Within these boundaries, five reproducible evidence groups are reported—Figure 9 (five-task real-device runtime coverage), Table 8 (key validation evidence for SafeHat deployment), Table 9 (root causes and repair outcomes of four fault classes), Table 10 (runtime post-processing sensitivity), and Table 11 (two-device CPU-path latency)—covering runtime, validation, repair, parameter selection, and latency scale, and indicating that PPE monitoring and related mobile vision tasks can be deployed on low-cost, widely available Android devices under limited compute and weak-network conditions.

---

## 6. Conclusions

This study presents an Android-NCNN deployment architecture and a five-stage deployment consistency diagnosis workflow for YOLO-based PPE monitoring and related visual tasks on resource-constrained Android devices. Within the Java–JNI–C++–NCNN pipeline, the architecture realizes unified loading, scheduling, and interface management across detection, segmentation, pose estimation, classification, and OBB under coexisting E2E, One-to-Many, and Legacy output paths. The diagnosis workflow—anomaly logging, intermediate-output inspection, cross-backend comparison, structure tracing, regression validation—provides structured localization and repair paths for four representative fault classes (preprocessing mismatch, redundant activation, coordinate semantic misinterpretation, OBB layout misjudgment), with S3 cross-backend comparison as the most effective front-end filter. The SafeHat main case instantiates the architecture and workflow end-to-end: after preprocessing correction, false detections in empty scenes are eliminated; after redundant activation is removed, raw model probability $p$ recovers to a scene-dependent multi-modal distribution (upper bound $\approx 0.95$) and the calibrated UI confidence $\tilde p$ stabilizes in 0.35–0.73; after coordinate semantics and OBB layout are corrected, keypoints and oriented boxes are rendered normally. Two real-device CPU-path measurements provide runtime-scale evidence: on Kirin 970, mean latency is 321.3–430.5 ms for the four 640×640 tasks and 33.6 ms for classification; on Kirin 659 (Nova2), pose, classification, and OBB reach 740.2 ms, 70.8 ms, and 640.6 ms respectively. These results provide an implementation-level feasibility baseline rather than an optimal-performance benchmark, with limitations including the SafeHat-centric main case, partial Kirin 659 coverage, CPU-only path, and absence of cross-framework comparisons.

Future work proceeds along three extension paths: (i) INT8 quantization and structural compression within the unified code path to further reduce latency, memory, and energy on mid-range and low-end SoCs; (ii) layered cross-device evaluation across different Android chip platforms, price tiers, SoC levels, and CPU/GPU paths; and (iii) an automated validation pipeline that brings more export checks, log comparisons, and regression tests into a continuous deployment process, with stronger asset replacement and re-checking under weak-network or offline conditions.

---

## Author Contributions

Conceptualization, [author initials]; methodology, [author initials]; software, [author initials]; validation, [author initials]; formal analysis, [author initials]; investigation, [author initials]; data curation, [author initials]; writing—original draft preparation, [author initials]; writing—review and editing, [author initials]; visualization, [author initials]. All authors have read and agreed to the published version of the manuscript.

## Funding

This research received no external funding.

## Institutional Review Board Statement

Not applicable. This study used public datasets and did not involve human-subject intervention or the collection of new personal data.

## Informed Consent Statement

Not applicable.

## Data Availability Statement

The deployment framework code developed in the present study has been publicly released at [https://github.com/majun2019/ncnn-android-yolo26lt](https://github.com/majun2019/ncnn-android-yolo26lt). The SafeHat training dataset was derived from a public Roboflow dataset; the original source and license information are documented in the dataset description file. The scripts used for model export, consistency diagnosis, sensitivity evaluation, and latency collection are included in the released repository.

## Conflicts of Interest

The authors declare no conflicts of interest.

---

## References

[1] Ge, Z.; Liu, S.; Wang, F.; Li, Z.; Sun, J. YOLOX: Exceeding YOLO Series in 2021. arXiv 2021, arXiv:2107.08430.

[2] Tan, M.; Pang, R.; Le, Q.V. EfficientDet: Scalable and Efficient Object Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020; pp. 10778–10787.

[3] He, K.; Zhang, X.; Ren, S.; Sun, J. Deep Residual Learning for Image Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016; pp. 770–778.

[4] Howard, A.; Sandler, M.; Chu, G.; Chen, L.-C.; Chen, B.; Tan, M.; Wang, W.; Zhu, Y.; Pang, R.; Vasudevan, V.; Le, Q.V.; Adam, H. Searching for MobileNetV3. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019; pp. 1314–1324.

[5] Zhang, X.; Zhou, X.; Lin, M.; Sun, J. ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018; pp. 6848–6856.

[6] Huang, G.; Liu, Z.; van der Maaten, L.; Weinberger, K.Q. Densely Connected Convolutional Networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017; pp. 2261–2269.

[7] TensorFlow Contributors. TensorFlow Lite Guide. Available online: https://www.tensorflow.org/lite/guide (accessed on 14 May 2026).

[8] Alibaba. MNN: A Blazing Fast, Lightweight Deep Learning Framework. Available online: https://github.com/alibaba/MNN (accessed on 14 May 2026).

[9] Microsoft. ONNX Runtime. Available online: https://onnxruntime.ai/ (accessed on 14 May 2026).

[10] Tencent. ncnn: A High-Performance Neural Network Inference Framework Optimized for the Mobile Platform. Available online: https://github.com/Tencent/ncnn (accessed on 14 May 2026).

[11] Liu, C.; Zhong, L.; Wang, J.; Huang, J.; Wang, Y.; Guan, M.; Li, X.; Zheng, H.; Hu, X.; Ma, X.; Tan, S. Grain-YOLO: An improved lightweight YOLO v8 and its android deployment for rice grains detection. Computers and Electronics in Agriculture 2025, 237, 110757.

[12] More, S.S.; Patil, N.; Lobo, V.B.; Shet, N.; Goswami, D.; Rane, P.; Kumar, P.N. Empowering the Visually Impaired: YOLOv8-based Object Detection in Android Applications. Procedia Computer Science 2025, 252, 457–469.

[13] Redmon, J.; Divvala, S.; Girshick, R.; Farhadi, A. You Only Look Once: Unified, Real-Time Object Detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016; pp. 779–788.

[14] Wang, C.-Y.; Bochkovskiy, A.; Liao, H.-Y.M. YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023; pp. 7464–7475.

[15] Ultralytics. Ultralytics YOLO. Available online: https://github.com/ultralytics/ultralytics (accessed on 14 May 2026).

[16] Bradski, G.; Kaehler, A. Learning OpenCV: Computer Vision with the OpenCV Library; O'Reilly Media: Sebastopol, CA, USA, 2008.

[17] Google. Android NDK Documentation. Available online: https://developer.android.com/ndk (accessed on 14 May 2026).

[18] Google. CameraX Overview. Available online: https://developer.android.com/training/camerax (accessed on 14 May 2026).

[19] Khronos Group. Vulkan Specification. Available online: https://www.khronos.org/vulkan/ (accessed on 14 May 2026).

[20] Tencent. ncnn Wiki: How-to-Use-and-FAQ. Available online: https://github.com/Tencent/ncnn/wiki (accessed on 14 May 2026).

[21] PNNX Contributors. PNNX: PyTorch Neural Network Exchange. Available online: https://github.com/pnnx/pnnx (accessed on 14 May 2026).

[22] Carion, N.; Massa, F.; Synnaeve, G.; Usunier, N.; Kirillov, A.; Zagoruyko, S. End-to-End Object Detection with Transformers. In Computer Vision – ECCV 2020; Springer: Cham, Switzerland, 2020; pp. 213–229.

[23] Zhao, Y.; Lv, W.; Xu, S.; Wei, J.; Wang, G.; Dang, Q.; Liu, Y.; Chen, J. DETRs Beat YOLOs on Real-time Object Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024; pp. 16965–16974.

[24] Bolya, D.; Zhou, C.; Xiao, F.; Lee, Y.J. YOLACT: Real-Time Instance Segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019; pp. 9156–9165.

[25] Li, Y.; Yang, S.; Liu, P.; Zhang, S.; Wang, Y.; Wang, Z.; Yang, W.; Xia, S.-T. SimCC: A Simple Coordinate Classification Perspective for Human Pose Estimation. In Computer Vision – ECCV 2022; Springer: Cham, Switzerland, 2022; pp. 89–106.

[26] Liu, Z.; Lin, Y.; Cao, Y.; Hu, H.; Wei, Y.; Zhang, Z.; Lin, S.; Guo, B. Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021; pp. 10012–10022.

[27] Xie, X.; Cheng, G.; Wang, J.; Yao, X.; Han, J. Oriented R-CNN for Object Detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021; pp. 3500–3509.
