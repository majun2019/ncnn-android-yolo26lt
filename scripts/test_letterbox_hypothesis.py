import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
关键假设验证: 非正方形图像的letterbox/padding是否导致NCNN分数爆炸?

旧诊断脚本用480x640图做letterbox → 结果异常
新诊断脚本用640x640图直接输入 → 结果正常

如果这个假设成立，说明问题在于:
手机摄像头输出非正方形图(如1920x1080) → C++端做letterbox → padding区域导致异常
"""
import os
from pathlib import Path
import numpy as np
import cv2

def resolve_detect_asset_paths(workspace_root: Path) -> tuple[str, str]:
    assets_dir = workspace_root / "app" / "src" / "main" / "assets"
    for stem in ("yolo26n_safehat.ncnn", "yolo26n_e2e.ncnn"):
        param = assets_dir / f"{stem}.param"
        bin_path = assets_dir / f"{stem}.bin"
        if param.exists() and bin_path.exists():
            return str(param.resolve()), str(bin_path.resolve())
    return str((assets_dir / "yolo26n_safehat.ncnn.param").resolve()), str((assets_dir / "yolo26n_safehat.ncnn.bin").resolve())

def test_ncnn_with_size(param_path, bin_path, img, label):
    import ncnn
    
    net = ncnn.Net()
    net.load_param(param_path)
    net.load_model(bin_path)
    
    h, w = img.shape[:2]
    target = 640
    
    img_sq = cv2.resize(img, (target, target))
    mat_sq = ncnn.Mat.from_pixels(img_sq, ncnn.Mat.PixelType.PIXEL_BGR2RGB, target, target)
    mat_sq.substract_mean_normalize([], [1/255.0]*3)
    
    ex = net.create_extractor()
    ex.input("in0", mat_sq)
    ret, out = ex.extract("out0")
    arr = np.array(out)
    if arr.ndim == 3: arr = arr[0]
    cls = arr[4:, :]
    max_scores = cls.max(axis=0)
    
    print(f"\n  [{label}] 直接resize到640x640 (无padding)")
    print(f"    cls range: [{cls.min():.6f}, {cls.max():.6f}]")
    print(f"    max_score: p50={np.percentile(max_scores, 50):.6f} max={max_scores.max():.6f}")
    print(f"    >0.35: {(max_scores>0.35).sum()}  >0.50: {(max_scores>0.50).sum()}")
    
    for c in range(cls.shape[0]):
        if cls[c].max() > 0.01:
            print(f"    class {c}: max={cls[c].max():.4f}")
    if cls.max() < 0.01:
        print(f"    ✅ 所有类别 max < 0.01")
    del ex

    scale = min(target/w, target/h)
    new_w, new_h = int(w*scale), int(h*scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    
    pad_w = (new_w + 31) // 32 * 32
    pad_h = (new_h + 31) // 32 * 32
    wpad = pad_w - new_w
    hpad = pad_h - new_h
    
    mat_in = ncnn.Mat.from_pixels(img_resized, ncnn.Mat.PixelType.PIXEL_BGR2RGB, new_w, new_h)
    mat_pad = ncnn.copy_make_border(mat_in, hpad//2, hpad - hpad//2, 
                                     wpad//2, wpad - wpad//2,
                                     ncnn.BorderType.BORDER_CONSTANT, 114.0)
    mat_pad.substract_mean_normalize([], [1/255.0]*3)
    
    ex2 = net.create_extractor()
    ex2.input("in0", mat_pad)
    ret2, out2 = ex2.extract("out0")
    arr2 = np.array(out2)
    if arr2.ndim == 3: arr2 = arr2[0]
    cls2 = arr2[4:, :]
    max_scores2 = cls2.max(axis=0)
    
    actual_input_w = mat_pad.w
    actual_input_h = mat_pad.h
    print(f"\n  [{label}] letterbox (resize={new_w}x{new_h}, pad={wpad},{hpad}, final=??)")
    print(f"    ncnn Mat: w={mat_pad.w}, h={mat_pad.h}, c={mat_pad.c}")
    print(f"    cls range: [{cls2.min():.6f}, {cls2.max():.6f}]")
    print(f"    max_score: p50={np.percentile(max_scores2, 50):.6f} max={max_scores2.max():.6f}")
    print(f"    >0.35: {(max_scores2>0.35).sum()}  >0.50: {(max_scores2>0.50).sum()}")
    
    for c in range(cls2.shape[0]):
        if cls2[c].max() > 0.01:
            print(f"    class {c}: max={cls2[c].max():.4f}")
    if cls2.max() < 0.01:
        print(f"    ✅ 所有类别 max < 0.01")
    del ex2
    
    img_640 = np.full((target, target, 3), 114, dtype=np.uint8)
    top = (target - new_h) // 2
    left = (target - new_w) // 2
    img_640[top:top+new_h, left:left+new_w] = img_resized
    
    mat3 = ncnn.Mat.from_pixels(img_640, ncnn.Mat.PixelType.PIXEL_BGR2RGB, target, target)
    mat3.substract_mean_normalize([], [1/255.0]*3)
    
    ex3 = net.create_extractor()
    ex3.input("in0", mat3)
    ret3, out3 = ex3.extract("out0")
    arr3 = np.array(out3)
    if arr3.ndim == 3: arr3 = arr3[0]
    cls3 = arr3[4:, :]
    max_scores3 = cls3.max(axis=0)
    
    print(f"\n  [{label}] letterbox到640x640 (标准YOLO方式, 114填充)")
    print(f"    cls range: [{cls3.min():.6f}, {cls3.max():.6f}]")
    print(f"    max_score: p50={np.percentile(max_scores3, 50):.6f} max={max_scores3.max():.6f}")
    print(f"    >0.35: {(max_scores3>0.35).sum()}  >0.50: {(max_scores3>0.50).sum()}")
    
    for c in range(cls3.shape[0]):
        if cls3[c].max() > 0.01:
            print(f"    class {c}: max={cls3[c].max():.4f}")
    if cls3.max() < 0.01:
        print(f"    ✅ 所有类别 max < 0.01")
    del ex3
    del net

def main():
    workspace_root = Path(__file__).resolve().parents[1]
    PARAM, BIN = resolve_detect_asset_paths(workspace_root)

    legacy_dir = os.environ.get("LEGACY_NCNN_MODEL_DIR", "")
    PARAM2 = str(Path(legacy_dir) / "model.ncnn.param") if legacy_dir else ""
    BIN2   = str(Path(legacy_dir) / "model.ncnn.bin") if legacy_dir else ""
    
    print("=" * 70)
    print("关键假设验证: 非正方形输入 + letterbox 是否导致分数爆炸?")
    print("=" * 70)
    
    test_cases = [
        ("640x640 正方形", np.full((640, 640, 3), 128, dtype=np.uint8)),
        ("480x640 竖屏",   np.full((640, 480, 3), 128, dtype=np.uint8)),
        ("1080x1920 手机竖屏", np.full((1920, 1080, 3), 128, dtype=np.uint8)),
        ("1920x1080 手机横屏", np.full((1080, 1920, 3), 128, dtype=np.uint8)),
        ("320x240 小图",   np.full((240, 320, 3), 128, dtype=np.uint8)),
    ]
    
    for param, bin_f, model_name in [(PARAM, BIN, "已部署SafeHat模型"), (PARAM2, BIN2, "历史项目SafeHat模型")]:
        if not param or not os.path.exists(param):
            continue
        print(f"\n\n{'#'*70}")
        print(f"# 模型: {model_name}")
        print(f"# {param}")
        print(f"{'#'*70}")
        
        for label, img in test_cases:
            print(f"\n{'='*60}")
            print(f"  输入图像: {label} ({img.shape[1]}x{img.shape[0]})")
            print(f"{'='*60}")
            test_ncnn_with_size(param, bin_f, img, label)

if __name__ == "__main__":
    main()
