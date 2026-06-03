from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

def backup_and_copy(src_param: Path, src_bin: Path, dst_prefix: Path) -> tuple[Path, Path]:
    dst_param = Path(str(dst_prefix) + ".param")
    dst_bin = Path(str(dst_prefix) + ".bin")
    dst_param.parent.mkdir(parents=True, exist_ok=True)

    if dst_param.exists():
        shutil.copy2(dst_param, dst_param.with_suffix(dst_param.suffix + ".bak"))
    if dst_bin.exists():
        shutil.copy2(dst_bin, dst_bin.with_suffix(dst_bin.suffix + ".bak"))

    shutil.copy2(src_param, dst_param)
    shutil.copy2(src_bin, dst_bin)
    return dst_param, dst_bin

def resolve_best_checkpoint(root: Path, run_name: str, model: YOLO, train_results) -> Path:
    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        sd = getattr(trainer, "save_dir", None)
        if sd:
            p = (Path(sd) / "weights" / "best.pt").resolve()
            if p.exists():
                return p

    sd2 = getattr(train_results, "save_dir", None)
    if sd2:
        p = (Path(sd2) / "weights" / "best.pt").resolve()
        if p.exists():
            return p

    candidates = sorted(
        root.glob(f"**/{run_name}/weights/best.pt"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()

    raise FileNotFoundError(
        f"best.pt not found for run '{run_name}'. "
        f"Looked via trainer.save_dir and pattern '**/{run_name}/weights/best.pt'."
    )

def resolve_ncnn_export_dir(root: Path, best_pt: Path, export_result) -> Path:
    if export_result is not None:
        p = Path(str(export_result)).resolve()
        if p.is_dir() and (p / "model.ncnn.param").exists() and (p / "model.ncnn.bin").exists():
            return p

    common = [
        best_pt.parent / f"{best_pt.stem}_ncnn_model",
        root / f"{best_pt.stem}_ncnn_model",
    ]
    for d in common:
        if d.exists() and (d / "model.ncnn.param").exists() and (d / "model.ncnn.bin").exists():
            return d.resolve()

    dirs = []
    for d in root.glob("**/*_ncnn_model"):
        if d.is_dir() and (d / "model.ncnn.param").exists() and (d / "model.ncnn.bin").exists():
            dirs.append(d)
    if dirs:
        dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return dirs[0].resolve()

    raise FileNotFoundError("NCNN export directory not found (expected model.ncnn.param/.bin).")

def main() -> None:
    ap = argparse.ArgumentParser(description="Quick finetune for confcal v3")
    ap.add_argument("--base-model", default="runs/calib/safehat_confcal_v2/weights/best.pt")
    ap.add_argument("--data-yaml", default="scripts/safehat.yaml")
    ap.add_argument("--project", default="runs/calib")
    ap.add_argument("--name", default="safehat_confcal_v3")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of dataset to use, e.g. 0.1")
    ap.add_argument("--lr0", type=float, default=8e-4)
    ap.add_argument("--lrf", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=1.5e-3)
    ap.add_argument("--warmup-epochs", type=float, default=2.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--close-mosaic", type=int, default=8)
    ap.add_argument("--device", default="", help="e.g. cuda:0 / cpu. empty=auto")
    ap.add_argument("--export-assets", action="store_true", help="export ncnn and overwrite app assets at end")
    ap.add_argument("--assets-prefix", default="app/src/main/assets/yolo26n_safehat.ncnn")
    ap.add_argument("--legacy-alias-prefix", default="app/src/main/assets/yolo26n_e2e.ncnn")
    args = ap.parse_args()

    root = Path.cwd()
    base_model = (root / args.base_model).resolve()
    data_yaml = (root / args.data_yaml).resolve()

    if not base_model.exists():
        raise FileNotFoundError(
            f"base model not found: {base_model}\n"
            f"tip: pass --base-model with your trained checkpoint, e.g. "
            f"E:/projects/AndroidPro/YOLOv11/runs/calib/safehat_confcal_v2/weights/best.pt"
        )
    if not data_yaml.exists():
        raise FileNotFoundError(f"data yaml not found: {data_yaml}")

    device = args.device.strip()
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"base model: {base_model}")
    print(f"data yaml : {data_yaml}")
    print(f"device    : {device}")

    model = YOLO(str(base_model))

    train_results = model.train(
        data=str(data_yaml),
        project=args.project,
        name=args.name,
        exist_ok=True,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        fraction=args.fraction,
        device=device,
        amp=False,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        save=True,
        plots=True,
        verbose=True,
    )

    best = resolve_best_checkpoint(root=root, run_name=args.name, model=model, train_results=train_results)

    print(f"best model: {best}")

    print("run validation...")
    YOLO(str(best)).val(workers=args.workers, device=device)

    if args.export_assets:
        print("export ncnn and overwrite app assets...")
        model_best = YOLO(str(best))
        export_result = model_best.export(format="ncnn", imgsz=args.imgsz, half=False, end2end=False)
        export_dir = resolve_ncnn_export_dir(root=root, best_pt=best, export_result=export_result)

        src_param = export_dir / "model.ncnn.param"
        src_bin = export_dir / "model.ncnn.bin"

        dst_prefix = (root / args.assets_prefix).resolve()
        dst_param, dst_bin = backup_and_copy(src_param, src_bin, dst_prefix)

        legacy_prefix = (root / args.legacy_alias_prefix).resolve() if args.legacy_alias_prefix else None
        if legacy_prefix and legacy_prefix != dst_prefix:
            legacy_param, legacy_bin = backup_and_copy(src_param, src_bin, legacy_prefix)
            print(f"legacy alias updated: {legacy_param}")
            print(f"legacy alias updated: {legacy_bin}")

        print(f"assets updated: {dst_param}")
        print(f"assets updated: {dst_bin}")

    print("done")

if __name__ == "__main__":
    main()
