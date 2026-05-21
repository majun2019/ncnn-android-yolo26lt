from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import cv2
from ultralytics import YOLO

@dataclass
class EvalStats:
    name: str
    model_path: str
    ok: bool
    error: str
    n_images: int
    det_min: float
    det_med: float
    det_max: float
    det_avg: float
    conf_avg: float

def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)

def pick_images(images_dir: Path, limit: int) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    imgs.sort()
    return imgs[:limit]

def eval_model(name: str, model_path: Path, images: List[Path], conf: float, imgsz: int) -> EvalStats:
    try:
        model = YOLO(str(model_path), task="detect")
        det_counts = []
        conf_means = []
        for img in images:
            r = model.predict(str(img), conf=conf, imgsz=imgsz, verbose=False)[0]
            n = int(len(r.boxes)) if r.boxes is not None else 0
            det_counts.append(float(n))
            if n > 0:
                conf_means.append(float(r.boxes.conf.cpu().numpy().mean()))

        return EvalStats(
            name=name,
            model_path=str(model_path),
            ok=True,
            error="",
            n_images=len(images),
            det_min=min(det_counts) if det_counts else 0.0,
            det_med=statistics.median(det_counts) if det_counts else 0.0,
            det_max=max(det_counts) if det_counts else 0.0,
            det_avg=statistics.mean(det_counts) if det_counts else 0.0,
            conf_avg=statistics.mean(conf_means) if conf_means else 0.0,
        )
    except Exception as e:
        return EvalStats(
            name=name,
            model_path=str(model_path),
            ok=False,
            error=str(e),
            n_images=len(images),
            det_min=0,
            det_med=0,
            det_max=0,
            det_avg=0,
            conf_avg=0,
        )

def copy_ncnn_outputs(src_param: Path, src_bin: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_param, dst_dir / "model.ncnn.param")
    shutil.copy2(src_bin, dst_dir / "model.ncnn.bin")
    return dst_dir

def export_ultralytics_ncnn(pt: Path, out_dir: Path, imgsz: int, half: bool, end2end: bool) -> Path:
    model = YOLO(str(pt))
    model.export(format="ncnn", imgsz=imgsz, half=half, end2end=end2end)
    src = pt.parent / f"{pt.stem}_ncnn_model"
    if not src.exists():
        src = Path.cwd() / f"{pt.stem}_ncnn_model"
    if not src.exists():
        raise FileNotFoundError("Ultralytics ncnn export dir not found")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "model.ncnn.param", out_dir / "model.ncnn.param")
    shutil.copy2(src / "model.ncnn.bin", out_dir / "model.ncnn.bin")
    return out_dir

def export_pnnx_torchscript(pt: Path, out_dir: Path, imgsz: int, fp16: bool, pnnx_exe: Path) -> Path:
    model = YOLO(str(pt))
    ts = model.export(format="torchscript", imgsz=imgsz)
    ts_path = Path(ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    run([
        str(pnnx_exe),
        str(ts_path),
        f"inputshape=1,3,{imgsz},{imgsz}",
        f"ncnnparam={(out_dir / 'model.ncnn.param').as_posix()}",
        f"ncnnbin={(out_dir / 'model.ncnn.bin').as_posix()}",
        f"fp16={1 if fp16 else 0}",
    ])
    return out_dir

def export_onnx2ncnn(pt: Path, out_dir: Path, imgsz: int, onnx2ncnn_exe: Path, ncnnoptimize_exe: Path | None, fp16_opt: bool) -> Path:
    model = YOLO(str(pt))
    onnx_path = Path(model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=False, nms=False))
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_param = out_dir / "model_raw.ncnn.param"
    raw_bin = out_dir / "model_raw.ncnn.bin"
    run([str(onnx2ncnn_exe), str(onnx_path), str(raw_param), str(raw_bin)])

    if ncnnoptimize_exe and ncnnoptimize_exe.exists():
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
    else:
        shutil.copy2(raw_param, out_dir / "model.ncnn.param")
        shutil.copy2(raw_bin, out_dir / "model.ncnn.bin")

    return out_dir

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--images", required=True, help="validation images dir")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--pnnx-exe", default="")
    ap.add_argument("--onnx2ncnn-exe", default="")
    ap.add_argument("--ncnnoptimize-exe", default="")
    args = ap.parse_args()

    pt = Path(args.pt).resolve()
    images = pick_images(Path(args.images).resolve(), args.n)
    if not images:
        raise RuntimeError("No images found")

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    results: list[EvalStats] = []

    results.append(eval_model("PT-baseline", pt, images, args.conf, args.imgsz))

    try:
        mdir = export_ultralytics_ncnn(pt, workdir / "ultra_ncnn_fp32_o2m", args.imgsz, half=False, end2end=False)
        results.append(eval_model("NCNN-ultra-fp32-o2m", mdir, images, args.conf, args.imgsz))
    except Exception as e:
        results.append(EvalStats("NCNN-ultra-fp32-o2m", str(workdir / "ultra_ncnn_fp32_o2m"), False, str(e), len(images), 0, 0, 0, 0, 0))

    try:
        mdir = export_ultralytics_ncnn(pt, workdir / "ultra_ncnn_fp16_o2m", args.imgsz, half=True, end2end=False)
        results.append(eval_model("NCNN-ultra-fp16-o2m", mdir, images, args.conf, args.imgsz))
    except Exception as e:
        results.append(EvalStats("NCNN-ultra-fp16-o2m", str(workdir / "ultra_ncnn_fp16_o2m"), False, str(e), len(images), 0, 0, 0, 0, 0))

    if args.pnnx_exe:
        try:
            mdir = export_pnnx_torchscript(pt, workdir / "pnnx_ts_fp16", args.imgsz, fp16=True, pnnx_exe=Path(args.pnnx_exe))
            results.append(eval_model("NCNN-pnnx-ts-fp16", mdir, images, args.conf, args.imgsz))
        except Exception as e:
            results.append(EvalStats("NCNN-pnnx-ts-fp16", str(workdir / "pnnx_ts_fp16"), False, str(e), len(images), 0, 0, 0, 0, 0))

    if args.onnx2ncnn_exe:
        try:
            mdir = export_onnx2ncnn(
                pt,
                workdir / "onnx2ncnn_opt_fp16",
                args.imgsz,
                onnx2ncnn_exe=Path(args.onnx2ncnn_exe),
                ncnnoptimize_exe=Path(args.ncnnoptimize_exe) if args.ncnnoptimize_exe else None,
                fp16_opt=True,
            )
            results.append(eval_model("NCNN-onnx2ncnn-opt-fp16", mdir, images, args.conf, args.imgsz))
        except Exception as e:
            results.append(EvalStats("NCNN-onnx2ncnn-opt-fp16", str(workdir / "onnx2ncnn_opt_fp16"), False, str(e), len(images), 0, 0, 0, 0, 0))

    print("\n=== Conversion Matrix Results ===")
    for r in results:
        print(f"[{r.name}] ok={r.ok} det(min/med/max/avg)=({r.det_min:.1f}/{r.det_med:.1f}/{r.det_max:.1f}/{r.det_avg:.1f}) conf_avg={r.conf_avg:.4f} err={r.error}")

    out_json = workdir / "conversion_matrix_results.json"
    out_json.write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_json}")

if __name__ == "__main__":
    main()
