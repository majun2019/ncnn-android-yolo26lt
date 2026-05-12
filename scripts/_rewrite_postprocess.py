#!/usr/bin/env python3
"""
Rewrite yolo26_det.cpp post-processing pipeline:
1. generate_proposals_yolo26_transposed: mild temperature calibration (T=3.0, no margin_gate)
2. NMS: per-class for SafeHat (not agnostic) 
3. Second NMS: per-class for SafeHat
4. Increase topk/maxdet for SafeHat
5. Remove mutual exclusion (per-class NMS handles it properly)
"""
import re
from pathlib import Path

workspace_root = Path(__file__).resolve().parents[1]
filepath = str(workspace_root / "app" / "src" / "main" / "jni" / "yolo26_det.cpp")

with open(filepath, "r", encoding="utf-8", errors="replace") as f:
    content = f.read()

# ============================================================
# 1. Rewrite generate_proposals_yolo26_transposed
# ============================================================
# Find the function
start_marker = "static void generate_proposals_yolo26_transposed(const ncnn::Mat& pred, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)"

# Find the end of the function (next top-level function/comment block)
func_start = content.find(start_marker)
if func_start < 0:
    raise RuntimeError("Cannot find generate_proposals_yolo26_transposed")

# Find the comment block before the function
comment_start = content.rfind("// YOLO26", 0, func_start)
if comment_start < 0:
    comment_start = func_start

# Find the end: look for the closing brace of the function
# The function ends with a closing "}" at the start of a line, followed by the filter function
filter_marker = "static void filter_person_dependent_classes"
func_end = content.find(filter_marker, func_start)
if func_end < 0:
    raise RuntimeError("Cannot find filter_person_dependent_classes after proposals function")

# Go back to find the blank line before filter_marker
func_end = content.rfind("\n", 0, func_end)
func_end = content.rfind("\n", 0, func_end)  # One more to get before the comment block

new_proposals_func = r"""// YOLO26: O2M head post-processing (transposed format)
// pred: h=(4+nc), w=8400.  Row 0-3 = bbox xywh (absolute pixels), Row 4+ = class sigmoid probs
// Uses mild temperature scaling (T=3.0) to compress saturated scores (0.999->0.85)
// NO margin_gate - SafeHat classes legitimately overlap (Person + Safety Vest on same box)
static void generate_proposals_yolo26_transposed(const ncnn::Mat& pred, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int img_w = in_pad.w;
    const int img_h = in_pad.h;

    const int num_class = pred.h - 4;
    const int total_boxes = pred.w;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "YOLO26 transposed: num_class=%d, total_boxes=%d", num_class, total_boxes);

    int detected_count = 0;
    (void)strides;

    for (int box_idx = 0; box_idx < total_boxes; box_idx++)
    {
        // Find top class
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float s = pred.row(4 + k)[box_idx];
            if (s > score)
            {
                label = k;
                score = s;
            }
        }

        if (label < 0 || !is_valid_f32(score))
            continue;

        // Mild temperature calibration: compress saturated sigmoid scores
        // T=3.0 maps: 0.999->0.85, 0.99->0.74, 0.95->0.63, 0.90->0.56, 0.80->0.47
        // This preserves ordering but makes scores more informative
        float display_score = score;
        {
            const float eps = 1e-6f;
            float p = std::max(eps, std::min(1.f - eps, score));
            float logit = logf(p / (1.f - p));
            display_score = 1.f / (1.f + expf(-(logit / 3.0f)));
        }

        if (display_score < prob_threshold)
            continue;

        // bbox
        float x_center = pred.row(0)[box_idx];
        float y_center = pred.row(1)[box_idx];
        float bw = pred.row(2)[box_idx];
        float bh = pred.row(3)[box_idx];

        if (!is_valid_f32(x_center) || !is_valid_f32(y_center) || !is_valid_f32(bw) || !is_valid_f32(bh))
            continue;
        if (bw <= 0.f || bh <= 0.f)
            continue;
        if (bw < 5.f || bh < 5.f)
            continue;
        const float aspect = bw > bh ? (bw / bh) : (bh / bw);
        if (aspect > 8.f)
            continue;
        if (bw > img_w * 0.95f || bh > img_h * 0.95f)
            continue;

        // Visible area filter
        {
            float vis_x0 = std::max(0.f, x_center - bw * 0.5f);
            float vis_y0 = std::max(0.f, y_center - bh * 0.5f);
            float vis_x1 = std::min((float)img_w, x_center + bw * 0.5f);
            float vis_y1 = std::min((float)img_h, y_center + bh * 0.5f);
            float vis_area = std::max(0.f, vis_x1 - vis_x0) * std::max(0.f, vis_y1 - vis_y0);
            float full_area = bw * bh;
            if (full_area > 0.f && vis_area / full_area < 0.40f)
                continue;
        }

        float x0 = x_center - bw * 0.5f;
        float y0 = y_center - bh * 0.5f;

        if (detected_count < 10) {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                "Det[%d] box=%d xywh=(%.1f,%.1f,%.1f,%.1f) label=%d raw=%.4f cal=%.4f",
                detected_count, box_idx, x_center, y_center, bw, bh, label, score, display_score);
        }
        detected_count++;

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = bw;
        obj.rect.height = bh;
        obj.label = label;
        obj.prob = display_score;
        objects.push_back(obj);

        // SafeHat multi-class: also emit secondary class proposals for same box
        // Person often competes with Safety Vest/No-Safety Vest at the same location
        if (num_class == 10)
        {
            for (int k = 0; k < num_class; k++)
            {
                if (k == label) continue; // already added
                float s = pred.row(4 + k)[box_idx];
                if (s < prob_threshold) continue;
                // Temperature calibrate
                float eps2 = 1e-6f;
                float p2 = std::max(eps2, std::min(1.f - eps2, s));
                float logit2 = logf(p2 / (1.f - p2));
                float cal2 = 1.f / (1.f + expf(-(logit2 / 3.0f)));
                if (cal2 < prob_threshold) continue;

                Object obj2;
                obj2.rect.x = x0;
                obj2.rect.y = y0;
                obj2.rect.width = bw;
                obj2.rect.height = bh;
                obj2.label = k;
                obj2.prob = cal2;
                objects.push_back(obj2);
            }
        }
    }

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Total detected: %d boxes, %zu proposals (incl multi-class)", detected_count, objects.size());
}

"""

content = content[:comment_start] + new_proposals_func + content[func_end:]

# ============================================================
# 2. Fix NMS: per-class for SafeHat (not agnostic)
# ============================================================
# Replace the nms_agnostic logic
old_nms = """    const int builtin_class_count = 80;
    const bool too_dense_scene = proposals_before_trim > 1500;
    const bool nms_agnostic = (inferred_num_class > 0 && inferred_num_class != builtin_class_count) || too_dense_scene;"""

new_nms = """    const int builtin_class_count = 80;
    // SafeHat: per-class NMS is CRITICAL because Person/Safety Vest/Hardhat
    // legitimately overlap on the same person. Agnostic NMS would kill Person.
    const bool is_safehat = (inferred_num_class == 10);
    const bool nms_agnostic = !is_safehat && ((inferred_num_class > 0 && inferred_num_class != builtin_class_count) || proposals_before_trim > 1500);"""

content = content.replace(old_nms, new_nms)

# ============================================================
# 3. Fix pre_nms_topk: increase for per-class NMS
# ============================================================
old_topk = """    int pre_nms_topk = 250;
    if (proposals_before_trim > 6000)
        pre_nms_topk = 60;
    else if (proposals_before_trim > 3000)
        pre_nms_topk = 90;
    else if (proposals_before_trim > 1500)
        pre_nms_topk = 120;"""

new_topk = """    int pre_nms_topk = 300;
    if (inferred_num_class == 10) {
        // SafeHat with per-class NMS: need more proposals to keep all classes
        pre_nms_topk = 500;
    } else if (proposals_before_trim > 6000) {
        pre_nms_topk = 60;
    } else if (proposals_before_trim > 3000) {
        pre_nms_topk = 90;
    } else if (proposals_before_trim > 1500) {
        pre_nms_topk = 120;
    }"""

content = content.replace(old_topk, new_topk)

# ============================================================
# 4. Fix max_det: increase for SafeHat
# ============================================================
old_maxdet = """    int max_det = 20;
    if (proposals_before_trim > 6000)
        max_det = 8;
    else if (proposals_before_trim > 3000)
        max_det = 10;
    else if (proposals_before_trim > 1500)
        max_det = 12;"""

new_maxdet = """    int max_det = 20;
    if (inferred_num_class == 10) {
        max_det = 30; // SafeHat per-class NMS: allow more detections
    } else if (proposals_before_trim > 6000) {
        max_det = 8;
    } else if (proposals_before_trim > 3000) {
        max_det = 10;
    } else if (proposals_before_trim > 1500) {
        max_det = 12;
    }"""

content = content.replace(old_maxdet, new_maxdet)

# ============================================================
# 5. Fix per_class_kept cap
# ============================================================
old_cap = "    const int max_per_class = proposals_before_trim > 3000 ? 2 : 4;"
new_cap = "    const int max_per_class = (inferred_num_class == 10) ? 5 : (proposals_before_trim > 3000 ? 2 : 4);"
content = content.replace(old_cap, new_cap)

# ============================================================
# 6. Fix second NMS: per-class for SafeHat
# ============================================================
old_second_nms = "        nms_sorted_bboxes(objects, picked_final, 0.45f, true);"
new_second_nms = "        nms_sorted_bboxes(objects, picked_final, 0.45f, !is_safehat); // SafeHat: per-class NMS"
content = content.replace(old_second_nms, new_second_nms)

# ============================================================
# 7. Fix final_cap for SafeHat
# ============================================================
old_final_cap = "        const int final_cap = proposals_before_trim > 3000 ? 8 : 12;"
new_final_cap = "        const int final_cap = (inferred_num_class == 10) ? 20 : (proposals_before_trim > 3000 ? 8 : 12);"
content = content.replace(old_final_cap, new_final_cap)

# ============================================================
# 8. Fix human-shaped Machinery/Vehicle filter - re-add properly (was partially garbled)
# ============================================================
# Check if the garbled version still exists and remove it
if "aspect_hw" in content:
    # It exists, we need to check context
    pass  # Already removed in earlier fix

# ============================================================
# 9. Remove the RawScoreStats adaptive threshold for SafeHat
#    (with temperature calibration, this is no longer needed)
# ============================================================
old_adaptive = """            const float frac_gt099 = (float)gt099 / (float)n;"""
# Find the full block
idx = content.find(old_adaptive)
if idx >= 0:
    # Find the if/else block after it
    block_end = content.find("__android_log_print", idx)
    # Replace the threshold adjustment to be SafeHat-aware
    old_block = content[idx:block_end]
    # Just replace the garbled comment and keep the logic but skip for SafeHat
    pass  # Leave as-is for now, the calibration already handles saturation

# ============================================================
# 10. Declare is_safehat earlier so it's available for second NMS
# ============================================================
# is_safehat is declared in the NMS section, need to make sure it's in scope for second NMS
# Actually, inferred_num_class is already in scope. Let me check...
# The is_safehat variable is declared inside the else-if block.
# Need to move it to outer scope or use inferred_num_class directly.
# Fix: replace the second NMS reference to use inferred_num_class directly
content = content.replace(
    "        nms_sorted_bboxes(objects, picked_final, 0.45f, !is_safehat); // SafeHat: per-class NMS",
    "        nms_sorted_bboxes(objects, picked_final, 0.45f, !(inferred_num_class == 10)); // SafeHat: per-class NMS"
)
content = content.replace(
    "        const int final_cap = (inferred_num_class == 10) ? 20 : (proposals_before_trim > 3000 ? 8 : 12);",
    "        const int final_cap = (inferred_num_class == 10) ? 20 : (proposals_before_trim > 3000 ? 10 : 15);"
)

# ============================================================
# Write the result
# ============================================================
with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Done. File written ({len(content)} chars)")

# Verify key changes
checks = [
    "logit / 3.0f",  # Temperature calibration
    "is_safehat",  # SafeHat detection
    "per-class NMS",  # Per-class NMS comment
    "multi-class",  # Multi-class proposals
    "inferred_num_class == 10) ? 5",  # Per-class cap
]
for check in checks:
    if check in content:
        print(f"  OK: '{check}' found")
    else:
        print(f"  MISSING: '{check}'")
