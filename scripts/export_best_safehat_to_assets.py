from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from ultralytics import YOLO

def ncnn_output_shape(param_path: Path, bin_path: Path, input_size: int = 640):
    import ncnn

    net = ncnn.Net()
    assert net.load_param(str(param_path)) == 0, f"load_param failed: {param_path}"
    assert net.load_model(str(bin_path)) == 0, f"load_model failed: {bin_path}"

    img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    mat = ncnn.Mat.from_pixels_resize(
        img,
        ncnn.Mat.PixelType.PIXEL_BGR,
        input_size,
        input_size,
        input_size,
        input_size,
    )
    mat.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])

    ex = net.create_extractor()
    ex.input("in0", mat)
    ret, out = ex.extract("out0")
    assert ret == 0, "extract out0 failed"
    return int(out.w), int(out.h), int(out.c)

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

def main():
    parser = argparse.ArgumentParser(description="导出 best.pt 到 Android assets")
    parser.add_argument("--pt", default="runs/detect/runs/train/yolo26n_safehat/weights/best.pt")
    parser.add_argument("--assets-prefix", default="app/src/main/assets/yolo26n_safehat.ncnn")
    parser.add_argument("--legacy-alias-prefix", default="app/src/main/assets/yolo26n_e2e.ncnn")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--end2end", action="store_true", help="导出 E2E (300x6)")
    args = parser.parse_args()

    root = Path.cwd()
    pt = (root / args.pt).resolve()
    if not pt.exists():
        raise FileNotFoundError(f"best.pt not found: {pt}")

    print(f"[1/4] 加载模型: {pt}")
    model = YOLO(str(pt))

    print(f"[2/4] 导出 NCNN, end2end={args.end2end}, imgsz={args.imgsz}")
    model.export(format="ncnn", imgsz=args.imgsz, half=False, end2end=args.end2end)

    export_dir = pt.parent / f"{pt.stem}_ncnn_model"
    if not export_dir.exists():
        export_dir = root / f"{pt.stem}_ncnn_model"

    src_param = export_dir / "model.ncnn.param"
    src_bin = export_dir / "model.ncnn.bin"
    if not src_param.exists() or not src_bin.exists():
        raise FileNotFoundError(f"export output not found in: {export_dir}")

    print(f"[3/4] 覆盖 assets 模型: {args.assets_prefix}.*")
    dst_prefix = (root / args.assets_prefix).resolve()
    dst_param, dst_bin = backup_and_copy(src_param, src_bin, dst_prefix)

    legacy_prefix = (root / args.legacy_alias_prefix).resolve() if args.legacy_alias_prefix else None
    if legacy_prefix and legacy_prefix != dst_prefix:
        legacy_param, legacy_bin = backup_and_copy(src_param, src_bin, legacy_prefix)
        print(f"legacy alias updated: {legacy_param}")
        print(f"legacy alias updated: {legacy_bin}")

    print("[4/4] 验证新 assets 模型输出形状")
    w, h, c = ncnn_output_shape(dst_param, dst_bin, input_size=args.imgsz)
    print(f"new assets out shape: w={w} h={h} c={c}")

    if args.end2end:
        print("期望 E2E: 常见 (300,6)")
    else:
        print("期望 O2M: 常见 (8400, nc+4)；safehat(10类) 应接近 (8400,14)")

    print("完成。请重新编译并在 Android 端观察 Output shape / Total detected。")

if __name__ == "__main__":
    main()
