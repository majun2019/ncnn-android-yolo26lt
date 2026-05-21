import re
from pathlib import Path

workspace_root = Path(__file__).resolve().parents[1]
filepath = str(workspace_root / "app" / "src" / "main" / "jni" / "yolo26_det.cpp")

with open(filepath, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

cleaned = []
for line in lines:
    if "鐗堟湰" in line or "鐞嗚浆缃" in line:
        continue
    cleaned.append(line)

with open(filepath, "w", encoding="utf-8") as f:
    f.writelines(cleaned)

print(f"Cleaned: {len(lines)} -> {len(cleaned)} lines")
