#!/usr/bin/env python3
"""Fix adaptive threshold for SafeHat - don't raise threshold."""
from pathlib import Path

workspace_root = Path(__file__).resolve().parents[1]
filepath = str(workspace_root / "app" / "src" / "main" / "jni" / "yolo26_det.cpp")

with open(filepath, "r", encoding="utf-8", errors="replace") as f:
    content = f.read()

# Find the frac_gt099 > 0.25f block and wrap it in a num_class check
old = """            if (frac_gt099 > 0.25f)
                effective_prob_threshold = std::max(prob_threshold, 0.50f);
            else if (frac_gt099 > 0.15f)
                effective_prob_threshold = std::max(prob_threshold, 0.45f);"""

new = """            // SafeHat (10 classes): do NOT raise threshold - Person has lower scores
            // than Machinery/Vehicle and would be killed by threshold increase
            if (num_class != 10) {
                if (frac_gt099 > 0.25f)
                    effective_prob_threshold = std::max(prob_threshold, 0.50f);
                else if (frac_gt099 > 0.15f)
                    effective_prob_threshold = std::max(prob_threshold, 0.45f);
            }"""

if old in content:
    content = content.replace(old, new)
    print("Fixed adaptive threshold")
else:
    # Try with different whitespace
    import re
    pattern = r'if \(frac_gt099 > 0\.25f\)\s+effective_prob_threshold = std::max\(prob_threshold, 0\.50f\);\s+else if \(frac_gt099 > 0\.15f\)\s+effective_prob_threshold = std::max\(prob_threshold, 0\.45f\);'
    match = re.search(pattern, content)
    if match:
        content = content[:match.start()] + """// SafeHat (10 classes): do NOT raise threshold - Person has lower scores
            if (num_class != 10) {
                if (frac_gt099 > 0.25f)
                    effective_prob_threshold = std::max(prob_threshold, 0.50f);
                else if (frac_gt099 > 0.15f)
                    effective_prob_threshold = std::max(prob_threshold, 0.45f);
            }""" + content[match.end():]
        print("Fixed adaptive threshold (regex)")
    else:
        # Find it line by line
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'frac_gt099 > 0.25f' in line:
                print(f"Found at line {i+1}: {repr(line)}")
                # Replace this block
                # line i: if (frac_gt099 > 0.25f)
                # line i+1: effective_prob_threshold = ...
                # line i+2: else if (frac_gt099 > 0.15f)
                # line i+3: effective_prob_threshold = ...
                indent = '            '
                new_lines = [
                    f"{indent}// SafeHat (10 classes): do NOT raise threshold",
                    f"{indent}if (num_class != 10) {{",
                    f"{indent}    {lines[i].strip()}",
                    f"{indent}        {lines[i+1].strip()}",
                    f"{indent}    {lines[i+2].strip()}",
                    f"{indent}        {lines[i+3].strip()}",
                    f"{indent}}}"
                ]
                lines[i:i+4] = new_lines
                content = '\n'.join(lines)
                print("Fixed via line replacement")
                break

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

# Verify
with open(filepath, "r", encoding="utf-8", errors="replace") as f:
    content = f.read()
if "num_class != 10" in content and "frac_gt099" in content:
    print("Verification OK")
else:
    print("WARNING: fix may not have applied")
