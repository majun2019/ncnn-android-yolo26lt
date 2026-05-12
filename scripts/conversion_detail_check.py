#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import cv2
import ncnn
import numpy as np
from ultralytics import YOLO


@dataclass
class NcnnStats:
    name: str
    model_dir: str
    ok: bool
    error: str
    n_images: int
    out_w: int
    out_h: int
    inferred_nc: int
    p50: float
    p90: float
    p99: float
    frac_gt_090: float
    frac_gt_099: float


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def pick_images(images_dir: Path, limit: int) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    imgs.sort()
    return imgs[:limit]


def letterbox_bgr(im: np.ndarray, target: int = 640, stride: int = 32) -> np.ndarray:
    h0, w0 = im.shape[:2]
    scale = min(target / w0, target / h0)
    w = int(round(w0 * scale))
    h = int(round(h0 * scale))
    im_r = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    wpad = int(np.ceil(w / stride) * stride - w)
    hpad = int(np.ceil(h / stride) * stride - h)
    top, bottom = hpad // 2, hpad - hpad // 2
    left, right = wpad // 2, wpad - wpad // 2
    return cv2.copyMakeBorder(im_r, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def ncnn_extract_out(model_dir: Path, image_path: Path) -> ncnn.Mat:
    net = ncnn.Net()
    if net.load_param(str(model_dir / "model.ncnn.param")) != 0:
        raise RuntimeError(f"load_param failed: {model_dir}")
    if net.load_model(str(model_dir / "model.ncnn.bin")) != 0:
        raise RuntimeError(f"load_model failed: {model_dir}")

    im = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"bad image: {image_path}")
    im_pad = letterbox_bgr(im, target=640, stride=32)

    mat = ncnn.Mat.from_pixels(im_pad, ncnn.Mat.PixelType.PIXEL_BGR2RGB, im_pad.shape[1], im_pad.shape[0])
    mat.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])

    ex = net.create_extractor()
    ex.input("in0", mat)
    ret, out = ex.extract("out0")
    if ret != 0:
        raise RuntimeError("extract out0 failed")
    return out


def max_class_probs(out: ncnn.Mat) -> tuple[np.ndarray, int]:
    w, h = int(out.w), int(out.h)
    feature_dim = min(w, h)
    if feature_dim <= 4:
        return np.array([], dtype=np.float32), -1
    nc = feature_dim - 4

    probs = []
    if w >= h:
        # shape likely (w=num_boxes, h=4+nc), rows are features
        num_boxes = w
        for i in range(num_boxes):
            cls_max = -1e9
            for c in range(nc):
                s = float(out.row(4 + c)[i])
                if s > cls_max:
                    cls_max = s
            probs.append(cls_max)
    else:
        # shape likely (w=4+nc, h=num_boxes), rows are boxes
        num_boxes = h
        for i in range(num_boxes):
            row = out.row(i)
            cls = [float(row[4 + c]) for c in range(nc)]
            probs.append(max(cls) if cls else 0.0)

    return np.array(probs, dtype=np.float32), nc


def eval_ncnn(name: str, model_dir: Path, images: List[Path]) -> NcnnStats:
    try:
        p50s: list[float] = []
        p90s: list[float] = []
        p99s: list[float] = []
        f90s: list[float] = []
        f99s: list[float] = []
        out_w = out_h = inferred_nc = -1

        for img in images:
            out = ncnn_extract_out(model_dir, img)
            out_w, out_h = int(out.w), int(out.h)
            probs, nc = max_class_probs(out)
            inferred_nc = nc
            if probs.size == 0:
                continue
            p50s.append(float(np.quantile(probs, 0.50)))
            p90s.append(float(np.quantile(probs, 0.90)))
            p99s.append(float(np.quantile(probs, 0.99)))
            f90s.append(float((probs > 0.90).mean()))
            f99s.append(float((probs > 0.99).mean()))

        return NcnnStats(
            name=name,
            model_dir=str(model_dir),
            ok=True,
            error="",
            n_images=len(images),
            out_w=out_w,
            out_h=out_h,
            inferred_nc=inferred_nc,
            p50=statistics.mean(p50s) if p50s else 0.0,
            p90=statistics.mean(p90s) if p90s else 0.0,
            p99=statistics.mean(p99s) if p99s else 0.0,
            frac_gt_090=statistics.mean(f90s) if f90s else 0.0,
            frac_gt_099=statistics.mean(f99s) if f99s else 0.0,
        )
    except Exception as e:
        return NcnnStats(name, str(model_dir), False, str(e), len(images), -1, -1, -1, 0, 0, 0, 0, 0)


def export_ultra_ncnn(pt: Path, out_dir: Path, imgsz: int, half: bool) -> Path:
    model = YOLO(str(pt))
    model.export(format="ncnn", imgsz=imgsz, half=half, end2end=False)
    src = pt.parent / f"{pt.stem}_ncnn_model"
    if not src.exists():
        src = Path.cwd() / f"{pt.stem}_ncnn_model"
    if not src.exists():
        raise RuntimeError("ultralytics ncnn output dir not found")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "model.ncnn.param", out_dir / "model.ncnn.param")
    shutil.copy2(src / "model.ncnn.bin", out_dir / "model.ncnn.bin")
    return out_dir


def export_pnnx_ts(pt: Path, out_dir: Path, imgsz: int, fp16: bool, pnnx_exe: Path) -> Path:
    model = YOLO(str(pt))
    ts = Path(model.export(format="torchscript", imgsz=imgsz))
    out_dir.mkdir(parents=True, exist_ok=True)
    run([
        str(pnnx_exe),
        str(ts),
        f"inputshape=1,3,{imgsz},{imgsz}",
        f"ncnnparam={(out_dir / 'model.ncnn.param').as_posix()}",
        f"ncnnbin={(out_dir / 'model.ncnn.bin').as_posix()}",
        f"fp16={1 if fp16 else 0}",
    ])
    return out_dir


def export_onnx2ncnn(pt: Path, out_dir: Path, imgsz: int, onnx2ncnn_exe: Path, ncnnoptimize_exe: Path, fp16_opt: bool) -> Path:
    model = YOLO(str(pt))
    onnx_path = Path(model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=False, nms=False))
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_param = out_dir / "model_raw.ncnn.param"
    raw_bin = out_dir / "model_raw.ncnn.bin"
    run([str(onnx2ncnn_exe), str(onnx_path), str(raw_param), str(raw_bin)])

    flag = "65536" if fp16_opt else "0"
    run([
        str(ncnnoptimize_exe),
        str(raw_param),
        str(raw_bin),
        str(out_dir / "model.ncnn.param"),
        str(out_dir / "model.ncnn.bin"),
        flag,
        f"1,3,{imgsz},{imgsz}",
    ])
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--pnnx-exe", default="")
    ap.add_argument("--onnx2ncnn-exe", default="")
    ap.add_argument("--ncnnoptimize-exe", default="")
    args = ap.parse_args()

    pt = Path(args.pt).resolve()
    imgs = pick_images(Path(args.images).resolve(), args.n)
    if not imgs:
        raise RuntimeError("no images")

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    results: list[NcnnStats] = []

    # ultra fp32
    m1 = export_ultra_ncnn(pt, workdir / "ultra_fp32", imgsz=args.imgsz, half=False)
    results.append(eval_ncnn("ultra_fp32", m1, imgs))

    # ultra fp16
    m2 = export_ultra_ncnn(pt, workdir / "ultra_fp16", imgsz=args.imgsz, half=True)
    results.append(eval_ncnn("ultra_fp16", m2, imgs))

    if args.pnnx_exe:
        # pnnx fp32
        m3 = export_pnnx_ts(pt, workdir / "pnnx_fp32", args.imgsz, fp16=False, pnnx_exe=Path(args.pnnx_exe))
        results.append(eval_ncnn("pnnx_fp32", m3, imgs))
        # pnnx fp16
        m4 = export_pnnx_ts(pt, workdir / "pnnx_fp16", args.imgsz, fp16=True, pnnx_exe=Path(args.pnnx_exe))
        results.append(eval_ncnn("pnnx_fp16", m4, imgs))

    if args.onnx2ncnn_exe and args.ncnnoptimize_exe:
        m5 = export_onnx2ncnn(
            pt,
            workdir / "onnx2ncnn_opt_fp16",
            args.imgsz,
            onnx2ncnn_exe=Path(args.onnx2ncnn_exe),
            ncnnoptimize_exe=Path(args.ncnnoptimize_exe),
            fp16_opt=True,
        )
        results.append(eval_ncnn("onnx2ncnn_opt_fp16", m5, imgs))

        m6 = export_onnx2ncnn(
            pt,
            workdir / "onnx2ncnn_opt_fp32",
            args.imgsz,
            onnx2ncnn_exe=Path(args.onnx2ncnn_exe),
            ncnnoptimize_exe=Path(args.ncnnoptimize_exe),
            fp16_opt=False,
        )
        results.append(eval_ncnn("onnx2ncnn_opt_fp32", m6, imgs))

    print("\n=== NCNN raw-output stats (conversion detail) ===")
    for r in results:
        print(
            f"[{r.name}] ok={r.ok} shape=({r.out_w},{r.out_h}) nc={r.inferred_nc} "
            f"p50={r.p50:.4f} p90={r.p90:.4f} p99={r.p99:.4f} "
            f"frac>0.90={r.frac_gt_090:.4f} frac>0.99={r.frac_gt_099:.4f} err={r.error}"
        )

    out = workdir / "conversion_detail_stats.json"
    out.write_text(json.dumps([asdict(x) for x in results], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
