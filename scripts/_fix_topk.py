from pathlib import Path

workspace_root = Path(__file__).resolve().parents[1]
filepath = str(workspace_root / "app" / "src" / "main" / "jni" / "yolo26_det.cpp")

with open(filepath, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

topk_start = None
topk_end = None
nms_comment_line = None

for i, line in enumerate(lines):
    if "int pre_nms_topk = 300;" in line or "int pre_nms_topk = 250;" in line:
        topk_start = i - 1
    if topk_start and "proposals.resize(pre_nms_topk);" in line:
        topk_end = i + 1
        for j in range(i+1, min(i+5, len(lines))):
            if lines[j].strip() == "}":
                topk_end = j + 1
                break
        break
    if "apply nms with nms_threshold" in line:
        nms_comment_line = i

print(f"topk_start={topk_start}, topk_end={topk_end}, nms_comment_line={nms_comment_line}")
print(f"Lines to replace:")
for i in range(topk_start, topk_end):
    print(f"  {i+1}: {lines[i].rstrip()}")

new_code = """    if (inferred_num_class == 10) {
        const int per_class_topk = 50;
        std::vector<int> class_count(inferred_num_class, 0);
        std::vector<Object> balanced_proposals;
        balanced_proposals.reserve(per_class_topk * inferred_num_class);
        for (size_t i = 0; i < proposals.size(); i++) {
            int cls = proposals[i].label;
            if (cls >= 0 && cls < inferred_num_class && class_count[cls] < per_class_topk) {
                balanced_proposals.push_back(proposals[i]);
                class_count[cls]++;
            }
        }
        proposals.swap(balanced_proposals);
        qsort_descent_inplace(proposals);

        __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
            "SafeHat per-class topk: Person=%d SafeVest=%d Machinery=%d Vehicle=%d Hardhat=%d total=%zu",
            class_count[5], class_count[7], class_count[8], class_count[9], class_count[0],
            proposals.size());
    } else {
        int pre_nms_topk = 300;
        if (proposals_before_trim > 6000)
            pre_nms_topk = 60;
        else if (proposals_before_trim > 3000)
            pre_nms_topk = 90;
        else if (proposals_before_trim > 1500)
            pre_nms_topk = 120;
        if ((int)proposals.size() > pre_nms_topk)
            proposals.resize(pre_nms_topk);
    }
"""

lines[topk_start:topk_end] = [new_code]

with open(filepath, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Done. Per-class topk implemented.")

with open(filepath, "r", encoding="utf-8", errors="replace") as f:
    content = f.read()

for check in ["per_class_topk", "balanced_proposals", "class_count[5]"]:
    if check in content:
        print(f"  OK: '{check}' found")
    else:
        print(f"  MISSING: '{check}'")
