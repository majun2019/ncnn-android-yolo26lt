#!/usr/bin/env python3
"""
基于当前模型自动挖掘 hard negatives（空标签负样本）。

用途：
- 在“无目标/反光/密集纹理”图中，挑出模型高置信误检样本；
- 自动生成空标签 txt；
- 可一键注入到 data/train 继续微调。

"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Candidate:
    image: str
    rel: str
    det_count: int
    max_conf: float
    mean_conf: float


def iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def label_path_of(img: Path, images_root: Path, labels_root: Path) -> Optional[Path]:
    try:
        rel = img.relative_to(images_root)
    except Exception:
        return None
    return (labels_root / rel).with_suffix(".txt")


def is_labeled_positive(label_file: Path) -> bool:
    if not label_file.exists():
        return False
    try:
        txt = label_file.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return False
    return len(txt) > 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine hard negatives from high-confidence false positives")
    ap.add_argument("--model", default="runs/calib/safehat_confcal_v2/weights/best.pt")
    ap.add_argument("--images-root", default="data/train/images")
    ap.add_argument("--labels-root", default="data/train/labels")
    ap.add_argument("--output-root", default="data/hardneg_pool/v3")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05, help="inference conf threshold for mining")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="")
    ap.add_argument("--topk", type=int, default=500)
    ap.add_argument("--min-max-conf", type=float, default=0.75)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--max-images", type=int, default=0, help="0 means no limit")
    ap.add_argument("--include-labeled", action="store_true", help="also consider images with non-empty labels")
    ap.add_argument("--inject-train", action="store_true", help="copy selected negatives into data/train/images/labels")
    args = ap.parse_args()

    root = Path.cwd()
    model_path = (root / args.model).resolve()
    images_root = (root / args.images_root).resolve()
    labels_root = (root / args.labels_root).resolve()
    output_root = (root / args.output_root).resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"model not found: {model_path}\n"
            f"tip: pass --model with your trained checkpoint, e.g. "
            f"E:/projects/AndroidPro/YOLOv11/runs/calib/safehat_confcal_v2/weights/best.pt"
        )
    if not images_root.exists():
        raise FileNotFoundError(f"images-root not found: {images_root}")

    all_images = sorted(iter_images(images_root))
    if args.max_images and args.max_images > 0:
        all_images = all_images[: args.max_images]

    candidates_src: List[Path] = []
    for img in all_images:
        if args.include_labeled:
            candidates_src.append(img)
            continue

        lp = label_path_of(img, images_root, labels_root)
        if lp is None:
            continue
        if is_labeled_positive(lp):
            continue
        candidates_src.append(img)

    if not candidates_src:
        print("No candidate images to mine.")
        return

    print(f"[1/4] load model: {model_path}")
    model = YOLO(str(model_path))

    print(f"[2/4] run inference on candidates: {len(candidates_src)}")
    pred_kwargs = dict(imgsz=args.imgsz, conf=args.conf, verbose=False, stream=True, batch=args.batch)
    if args.device:
        pred_kwargs["device"] = args.device

    rows: List[Candidate] = []
    for r in model.predict([str(p) for p in candidates_src], **pred_kwargs):
        path = Path(r.path)
        boxes = r.boxes
        cnt = int(len(boxes)) if boxes is not None else 0
        if cnt > 0:
            confs = boxes.conf.cpu().numpy().tolist()
            max_conf = float(max(confs))
            mean_conf = float(sum(confs) / len(confs))
        else:
            max_conf = 0.0
            mean_conf = 0.0

        if max_conf < args.min_max_conf:
            continue
        if cnt < args.min_count:
            continue

        rel = path.relative_to(images_root)
        rows.append(
            Candidate(
                image=str(path),
                rel=str(rel).replace("\\", "/"),
                det_count=cnt,
                max_conf=max_conf,
                mean_conf=mean_conf,
            )
        )

    rows.sort(key=lambda x: (x.max_conf, x.det_count, x.mean_conf), reverse=True)
    selected = rows[: args.topk]

    out_imgs = output_root / "images"
    out_lbls = output_root / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    print(f"[3/4] export selected hard negatives: {len(selected)}")
    manifest = []
    for i, c in enumerate(selected):
        src = Path(c.image)
        # avoid collisions
        dst_name = f"hn_{i:05d}_{src.name}"
        dst_img = out_imgs / dst_name
        dst_lbl = out_lbls / (Path(dst_name).stem + ".txt")
        shutil.copy2(src, dst_img)
        dst_lbl.write_text("", encoding="utf-8")
        manifest.append({**asdict(c), "dst_image": str(dst_img), "dst_label": str(dst_lbl)})

    report = {
        "model": str(model_path),
        "images_root": str(images_root),
        "labels_root": str(labels_root),
        "output_root": str(output_root),
        "config": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "topk": args.topk,
            "min_max_conf": args.min_max_conf,
            "min_count": args.min_count,
            "include_labeled": args.include_labeled,
        },
        "summary": {
            "total_images_scanned": len(all_images),
            "candidate_images": len(candidates_src),
            "selected": len(selected),
        },
        "selected": manifest,
    }

    report_json = output_root / "hardneg_report.json"
    report_md = output_root / "hardneg_report.md"
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Hard Negative Mining Report",
        "",
        f"- model: {model_path}",
        f"- scanned: {len(all_images)}",
        f"- candidates: {len(candidates_src)}",
        f"- selected: {len(selected)}",
        "",
        "## Top 20",
        "",
        "| rank | image | max_conf | det_count | mean_conf |",
        "|---:|---|---:|---:|---:|",
    ]
    for i, c in enumerate(selected[:20], start=1):
        md_lines.append(f"| {i} | {c.rel} | {c.max_conf:.4f} | {c.det_count} | {c.mean_conf:.4f} |")
    report_md.write_text("\n".join(md_lines), encoding="utf-8")

    if args.inject_train:
        train_imgs = (root / "data/train/images/hardneg_v3").resolve()
        train_lbls = (root / "data/train/labels/hardneg_v3").resolve()
        train_imgs.mkdir(parents=True, exist_ok=True)
        train_lbls.mkdir(parents=True, exist_ok=True)

        for p in out_imgs.iterdir():
            if p.is_file():
                shutil.copy2(p, train_imgs / p.name)
                (train_lbls / (p.stem + ".txt")).write_text("", encoding="utf-8")

        print(f"[4/4] injected into train set: {train_imgs}")
    else:
        print("[4/4] skip train injection (use --inject-train to enable)")

    print(f"report json: {report_json}")
    print(f"report md  : {report_md}")
    print("done")


if __name__ == "__main__":
    main()
