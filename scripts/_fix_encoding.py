#!/usr/bin/env python3
"""Clean up yolo26_det.cpp: remove garbled old comments, fix encoding issues"""
import re
from pathlib import Path

workspace_root = Path(__file__).resolve().parents[1]
filepath = str(workspace_root / "app" / "src" / "main" / "jni" / "yolo26_det.cpp")

with open(filepath, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

# Remove the garbled line 480 (0-indexed 479)
# It contains "YOLO26鐗堟湰" which is the garbled version
cleaned = []
for line in lines:
    # Skip garbled old comment lines
    if "鐗堟湰" in line or "鐞嗚浆缃" in line:
        continue
    cleaned.append(line)

with open(filepath, "w", encoding="utf-8") as f:
    f.writelines(cleaned)

print(f"Cleaned: {len(lines)} -> {len(cleaned)} lines")
