#!/usr/bin/env python3
"""
训练/导出侧预检工具（SafeHat 项目）

目标：在把模型放进 Android 前，先回答 4 个问题：
1) 数据集标签是否越界/损坏？
2) 数据配置 nc 与 names 是否一致？
3) NCNN 导出模型实际输出形状是什么？（E2E 还是 O2M）
4) 模型类别数是否和数据集类别数一致？

可选：对比 PT 与 NCNN 在同一批图片上的检测数量分布。

用法示例：
  python scripts/preflight_safehat_pipeline.py --run-compare
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import yaml


@dataclass
class LabelScanResult:
    total_files: int
    total_labels: int
    malformed_lines: int
    out_of_range_labels: int
    class_hist: List[int]


def load_data_yaml(yaml_path: Path) -> dict:
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid yaml: {yaml_path}")
    return cfg


def resolve_dataset_paths(cfg: dict, yaml_path: Path) -> Tuple[Path, Path, Path, Path, int, List[str]]:
    base = (yaml_path.parent / cfg.get("path", "")).resolve()
    train_rel = cfg.get("train", "train/images")
    val_rel = cfg.get("val", cfg.get("valid", "valid/images"))
    test_rel = cfg.get("test", "test/images")

    nc = int(cfg.get("nc", 0))
    names = cfg.get("names", [])
    if not isinstance(names, list):
        raise ValueError("names must be a list")

    return (
        base / train_rel,
        base / val_rel,
        base / test_rel,
        base,
        nc,
        names,
    )


def labels_dir_from_images_dir(images_dir: Path) -> Path:
    # YOLO 常规结构: xxx/images -> xxx/labels
    if images_dir.name == "images":
        return images_dir.parent / "labels"
    return images_dir.parent / "labels"


def iter_label_files(*label_dirs: Path) -> Iterable[Path]:
    for d in label_dirs:
        if d.exists():
            yield from d.rglob("*.txt")


def scan_labels(nc: int, *label_dirs: Path) -> LabelScanResult:
    class_hist = [0 for _ in range(max(nc, 1))]
    total_files = 0
    total_labels = 0
    malformed_lines = 0
    out_of_range_labels = 0

    for lf in iter_label_files(*label_dirs):
        total_files += 1
        try:
            lines = lf.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            malformed_lines += 1
            continue

        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                malformed_lines += 1
                continue
            try:
                cls = int(float(parts[0]))
                _ = [float(v) for v in parts[1:5]]
            except Exception:
                malformed_lines += 1
                continue

            total_labels += 1
            if cls < 0 or cls >= nc:
                out_of_range_labels += 1
            else:
                class_hist[cls] += 1

    return LabelScanResult(
        total_files=total_files,
        total_labels=total_labels,
        malformed_lines=malformed_lines,
        out_of_range_labels=out_of_range_labels,
        class_hist=class_hist,
    )


def detect_ncnn_output_shape(param_path: Path, bin_path: Path, input_size: int = 640) -> Tuple[int, int, int]:
    import ncnn  # 延迟导入，避免无依赖时启动失败

    net = ncnn.Net()
    ret = net.load_param(str(param_path))
    if ret != 0:
        raise RuntimeError(f"load_param failed: {param_path}")
    ret = net.load_model(str(bin_path))
    if ret != 0:
        raise RuntimeError(f"load_model failed: {bin_path}")

    img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    mat = ncnn.Mat.from_pixels_resize(
        img,
        ncnn.Mat.PixelType.PIXEL_BGR,
        input_size,
        input_size,
        input_size,
        input_size,
    )
    norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
    mat.substract_mean_normalize([], norm_vals)

    ex = net.create_extractor()
    ex.input("in0", mat)
    ret, out = ex.extract("out0")
    if ret != 0:
        raise RuntimeError("extract out0 failed")

    return int(out.w), int(out.h), int(out.c)


def infer_head_type_and_classes(w: int, h: int) -> Tuple[str, Optional[int]]:
    # E2E detection 常见 (300,6)
    if w == 6 or h == 6:
        return "E2E", None

    # O2M detection 常见 (8400, nc+4) 或转置
    a, b = max(w, h), min(w, h)
    if a == 8400 and b > 4:
        return "O2M", b - 4

    return "Unknown", None


def pick_sample_images(val_images: Path, limit: int = 12) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in val_images.rglob("*") if p.suffix.lower() in exts]
    imgs.sort()
    return imgs[:limit]


def build_ultralytics_ncnn_temp_model(param_path: Path, bin_path: Path) -> Path:
    # Ultralytics 对 NCNN 目录识别依赖 *_ncnn_model 命名约定
    td = Path(tempfile.mkdtemp(prefix="ncnn_ultra_", suffix="_ncnn_model"))
    shutil.copy2(param_path, td / "model.ncnn.param")
    shutil.copy2(bin_path, td / "model.ncnn.bin")
    return td


def resolve_assets_model_prefix(root: Path, prefix_text: str) -> Path:
    prefix = (root / prefix_text).resolve()
    param = Path(str(prefix) + ".param")
    bin_path = Path(str(prefix) + ".bin")
    if param.exists() and bin_path.exists():
        return prefix

    detect_aliases = {
        "yolo26n_safehat.ncnn": "yolo26n_e2e.ncnn",
        "yolo26n_e2e.ncnn": "yolo26n_safehat.ncnn",
    }
    alias_name = detect_aliases.get(prefix.name)
    if alias_name:
        alias_prefix = prefix.with_name(alias_name)
        alias_param = Path(str(alias_prefix) + ".param")
        alias_bin = Path(str(alias_prefix) + ".bin")
        if alias_param.exists() and alias_bin.exists():
            return alias_prefix

    return prefix


def compare_pt_vs_ncnn(
    pt_model: Path,
    ncnn_param: Path,
    ncnn_bin: Path,
    sample_images: List[Path],
    conf: float = 0.25,
) -> None:
    from ultralytics import YOLO

    if not sample_images:
        print("[COMPARE] 跳过：验证集图片为空")
        return

    temp_ncnn = build_ultralytics_ncnn_temp_model(ncnn_param, ncnn_bin)
    try:
        pt = YOLO(str(pt_model))
        # 某些 ultralytics 版本对 NCNN 目录识别不稳定，按顺序尝试多种入口
        try:
            ncnn_model = YOLO(str(temp_ncnn), task="detect")
        except Exception:
            try:
                ncnn_model = YOLO(str(temp_ncnn / "model.ncnn.param"), task="detect")
            except Exception:
                # 再试一次带目录斜杠形式
                ncnn_model = YOLO(str(temp_ncnn).rstrip("/\\") + "/", task="detect")

        pt_counts, ncnn_counts = [], []
        pt_conf_med, ncnn_conf_med = [], []

        for img in sample_images:
            r_pt = pt.predict(str(img), conf=conf, imgsz=640, verbose=False)[0]
            r_nc = ncnn_model.predict(str(img), conf=conf, imgsz=640, verbose=False)[0]

            c_pt = int(len(r_pt.boxes)) if r_pt.boxes is not None else 0
            c_nc = int(len(r_nc.boxes)) if r_nc.boxes is not None else 0
            pt_counts.append(c_pt)
            ncnn_counts.append(c_nc)

            if c_pt > 0:
                pt_conf_med.append(float(r_pt.boxes.conf.cpu().numpy().mean()))
            if c_nc > 0:
                ncnn_conf_med.append(float(r_nc.boxes.conf.cpu().numpy().mean()))

        def stats(xs: List[int]) -> str:
            if not xs:
                return "n/a"
            return f"min={min(xs)} med={statistics.median(xs):.1f} max={max(xs)} avg={statistics.mean(xs):.1f}"

        print("\n[COMPARE] PT vs NCNN 检测数量统计")
        print(f"  PT   : {stats(pt_counts)}")
        print(f"  NCNN : {stats(ncnn_counts)}")

        if pt_conf_med:
            print(f"  PT 平均置信度(逐图均值再平均): {statistics.mean(pt_conf_med):.4f}")
        if ncnn_conf_med:
            print(f"  NCNN 平均置信度(逐图均值再平均): {statistics.mean(ncnn_conf_med):.4f}")

    except Exception as e:
        print(f"[COMPARE][SKIP] PT vs NCNN 对比失败: {e}")
        print("[COMPARE][TIP] 不影响前面的核心结论（类别数/输出形状）。")
    finally:
        shutil.rmtree(temp_ncnn, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="SafeHat 训练/导出预检")
    parser.add_argument("--data-yaml", default="scripts/safehat.yaml")
    parser.add_argument("--assets-model-prefix", default="app/src/main/assets/yolo26n_safehat.ncnn")
    parser.add_argument("--pt-model", default="runs/detect/runs/train/yolo26n_safehat/weights/best.pt")
    parser.add_argument("--run-compare", action="store_true", help="运行 PT vs NCNN 快速对比")
    parser.add_argument("--compare-images", type=int, default=12)
    args = parser.parse_args()

    root = Path.cwd()

    data_yaml = (root / args.data_yaml).resolve()
    cfg = load_data_yaml(data_yaml)
    train_img, val_img, test_img, data_root, nc, names = resolve_dataset_paths(cfg, data_yaml)

    print("=" * 72)
    print("[A] 数据配置检查")
    print("=" * 72)
    print(f"data.yaml : {data_yaml}")
    print(f"data root : {data_root}")
    print(f"train img : {train_img}")
    print(f"val img   : {val_img}")
    print(f"test img  : {test_img}")
    print(f"nc        : {nc}")
    print(f"names.len : {len(names)}")

    if nc != len(names):
        print("[FAIL] nc 与 names 数量不一致")
    else:
        print("[OK] nc 与 names 数量一致")

    train_lbl = labels_dir_from_images_dir(train_img)
    val_lbl = labels_dir_from_images_dir(val_img)
    test_lbl = labels_dir_from_images_dir(test_img)

    print("\n" + "=" * 72)
    print("[B] 标签质量扫描")
    print("=" * 72)
    scan = scan_labels(nc, train_lbl, val_lbl, test_lbl)
    print(f"label files         : {scan.total_files}")
    print(f"total label entries : {scan.total_labels}")
    print(f"malformed lines     : {scan.malformed_lines}")
    print(f"out-of-range labels : {scan.out_of_range_labels}")
    if scan.class_hist:
        top = sorted(enumerate(scan.class_hist), key=lambda x: x[1], reverse=True)[:5]
        print("top classes by count:")
        for i, c in top:
            name = names[i] if i < len(names) else str(i)
            print(f"  - {i:2d} ({name}): {c}")

    print("\n" + "=" * 72)
    print("[C] NCNN 输出形状与头类型")
    print("=" * 72)
    prefix = resolve_assets_model_prefix(root, args.assets_model_prefix)
    param_path = Path(str(prefix) + ".param")
    bin_path = Path(str(prefix) + ".bin")
    print(f"param: {param_path}")
    print(f"bin  : {bin_path}")

    if not param_path.exists() or not bin_path.exists():
        print("[FAIL] assets NCNN 模型文件不存在")
        return

    w, h, c = detect_ncnn_output_shape(param_path, bin_path)
    head, cls_from_shape = infer_head_type_and_classes(w, h)
    print(f"out shape: w={w} h={h} c={c}")
    print(f"head type: {head}")

    if cls_from_shape is not None:
        print(f"classes from shape: {cls_from_shape}")
        if cls_from_shape != nc:
            print("[FAIL] 模型类别数与数据集 nc 不一致（这是高风险根因）")
        else:
            print("[OK] 模型类别数与数据集 nc 一致")
    else:
        print("[INFO] E2E 输出不直接携带 nc，需结合训练配置确认")

    if args.run_compare:
        print("\n" + "=" * 72)
        print("[D] PT vs NCNN 快速对比")
        print("=" * 72)
        pt_model = (root / args.pt_model).resolve()
        if not pt_model.exists():
            print(f"[SKIP] PT 模型不存在: {pt_model}")
        else:
            imgs = pick_sample_images(val_img, limit=max(1, args.compare_images))
            compare_pt_vs_ncnn(pt_model, param_path, bin_path, imgs)

    print("\n" + "=" * 72)
    print("预检完成")
    print("=" * 72)


if __name__ == "__main__":
    main()
