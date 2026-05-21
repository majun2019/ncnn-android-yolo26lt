import sys
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import torch
import cv2

ROOT = Path(__file__).resolve().parent.parent
CLASSES = [
    "Hardhat", "Mask", "No-Hardhat", "No-Mask", "No-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "Machinery", "Vehicle"
]

def resolve_detect_asset_paths(root: Path) -> tuple[Path, Path]:
    assets_dir = root / "app" / "src" / "main" / "assets"
    for stem in ("yolo26n_safehat.ncnn", "yolo26n_e2e.ncnn"):
        param = assets_dir / f"{stem}.param"
        bin_path = assets_dir / f"{stem}.bin"
        if param.exists() and bin_path.exists():
            return param, bin_path
    return assets_dir / "yolo26n_safehat.ncnn.param", assets_dir / "yolo26n_safehat.ncnn.bin"

def test_pytorch_model(model_path: str, img_path: str):
    print("=" * 70)
    print(f"[PyTorch] 模型: {model_path}")
    print(f"[PyTorch] 图片: {img_path}")
    print("=" * 70)
    
    model = YOLO(model_path)
    
    results = model.predict(img_path, imgsz=640, conf=0.25, verbose=False)
    
    for r in results:
        boxes = r.boxes
        print(f"\n检测到 {len(boxes)} 个对象:")
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            print(f"  [{i}] {CLASSES[cls_id]:15s} conf={conf:.4f} "
                  f"box=({xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f}) "
                  f"w={w:.0f} h={h:.0f}")
    
    return results

def test_pytorch_raw_output(model_path: str, img_path: str):
    print("\n" + "=" * 70)
    print(f"[PyTorch RAW] 检查 O2M 头原始输出")
    print("=" * 70)
    
    model = YOLO(model_path)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ✗ 无法读取图片: {img_path}")
        return
    
    from ultralytics.data.augment import LetterBox
    letterbox = LetterBox(new_shape=(640, 640), auto=False)
    img_lb = letterbox(image=img)
    img_tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    
    torch_model = model.model
    torch_model.eval()
    
    with torch.no_grad():
        preds = torch_model(img_tensor)
    
    print(f"\n  模型输出类型: {type(preds)}")
    if isinstance(preds, (tuple, list)):
        for i, p in enumerate(preds):
            if isinstance(p, torch.Tensor):
                print(f"  preds[{i}] shape: {p.shape}, dtype: {p.dtype}")
                print(f"    min={p.min().item():.6f} max={p.max().item():.6f} mean={p.mean().item():.6f}")
            elif isinstance(p, (tuple, list)):
                print(f"  preds[{i}] is {type(p).__name__} len={len(p)}")
                for j, pp in enumerate(p):
                    if isinstance(pp, torch.Tensor):
                        print(f"    preds[{i}][{j}] shape: {pp.shape}")
    elif isinstance(preds, torch.Tensor):
        print(f"  preds shape: {preds.shape}, dtype: {preds.dtype}")
    
    head = torch_model.model[-1]
    print(f"\n  检测头类型: {type(head).__name__}")
    print(f"  检测头属性: {[a for a in dir(head) if not a.startswith('_')][:30]}")
    
    if hasattr(head, 'nc'):
        print(f"  nc (num_classes): {head.nc}")
    if hasattr(head, 'nl'):
        print(f"  nl (num_levels): {head.nl}")
    if hasattr(head, 'reg_max'):
        print(f"  reg_max: {head.reg_max}")

def compare_ncnn_output(param_path: str, bin_path: str, img_path: str):
    try:
        import ncnn
    except ImportError:
        print("\n[NCNN] pyncnn 未安装，跳过NCNN推理对比")
        print("  安装方法: pip install ncnn")
        return
    
    print("\n" + "=" * 70)
    print(f"[NCNN] 模型: {param_path}")
    print("=" * 70)
    
    net = ncnn.Net()
    net.load_param(str(param_path))
    net.load_model(str(bin_path))
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ✗ 无法读取图片")
        return
    
    h, w = img.shape[:2]
    scale = min(640/w, 640/h)
    new_w, new_h = int(w*scale), int(h*scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    
    mat_in = ncnn.Mat.from_pixels_resize(img_resized, ncnn.Mat.PixelType.PIXEL_BGR2RGB, new_w, new_h, new_w, new_h)
    
    wpad = (new_w + 31) // 32 * 32 - new_w
    hpad = (new_h + 31) // 32 * 32 - new_h
    mat_pad = ncnn.copy_make_border(mat_in, hpad // 2, hpad - hpad // 2, wpad // 2, wpad - wpad // 2, ncnn.BorderType.BORDER_CONSTANT, 114.0)
    
    norm_vals = [1/255.0, 1/255.0, 1/255.0]
    mat_pad.substract_mean_normalize([], norm_vals)
    
    ex = net.create_extractor()
    ex.input("in0", mat_pad)
    
    ret, out = ex.extract("out0")
    print(f"  输出 shape: w={out.w}, h={out.h}, c={out.c}")
    
    if out.c == 1:
        arr = np.array(out)
        if arr.ndim == 3:
            arr = arr[0]
        print(f"  numpy shape: {arr.shape}")
        
        if out.w > out.h:
            num_boxes = out.w
            num_class = out.h - 4
            print(f"  格式: transposed, boxes={num_boxes}, classes={num_class}")
            
            bbox_row0 = arr[0, :]
            bbox_row1 = arr[1, :]
            bbox_row2 = arr[2, :]
            bbox_row3 = arr[3, :]
            
            print(f"\n  bbox x_center: min={bbox_row0.min():.2f} max={bbox_row0.max():.2f} mean={bbox_row0.mean():.2f}")
            print(f"  bbox y_center: min={bbox_row1.min():.2f} max={bbox_row1.max():.2f} mean={bbox_row1.mean():.2f}")
            print(f"  bbox width:    min={bbox_row2.min():.2f} max={bbox_row2.max():.2f} mean={bbox_row2.mean():.2f}")
            print(f"  bbox height:   min={bbox_row3.min():.2f} max={bbox_row3.max():.2f} mean={bbox_row3.mean():.2f}")
            
            cls_scores = arr[4:, :]
            print(f"\n  类别分数矩阵 shape: {cls_scores.shape}")
            print(f"  类别分数范围: min={cls_scores.min():.6f} max={cls_scores.max():.6f}")
            
            for k in range(num_class):
                row = cls_scores[k, :]
                top5_idx = np.argsort(row)[-5:][::-1]
                print(f"  class[{k}] {CLASSES[k]:15s}: max={row.max():.6f} "
                      f"mean={row.mean():.6f} >0.5 count={np.sum(row > 0.5)}")
            
            max_per_box = cls_scores.max(axis=0)
            top_k = 10
            top_idx = np.argsort(max_per_box)[-top_k:][::-1]
            print(f"\n  Top-{top_k} 检测结果:")
            for rank, idx in enumerate(top_idx):
                cls_id = cls_scores[:, idx].argmax()
                score = cls_scores[cls_id, idx]
                xc, yc, bw, bh = arr[0, idx], arr[1, idx], arr[2, idx], arr[3, idx]
                print(f"    [{rank}] box={idx} {CLASSES[cls_id]:15s} score={score:.6f} "
                      f"xywh=({xc:.1f},{yc:.1f},{bw:.1f},{bh:.1f})")

def check_export_output_format(model_path: str):
    print("\n" + "=" * 70)
    print("[导出检查] 检查模型导出参数")
    print("=" * 70)
    
    model = YOLO(model_path)
    
    print(f"  模型类型: {model.task}")
    print(f"  模型名称: {model.model_name if hasattr(model, 'model_name') else 'N/A'}")
    
    head = model.model.model[-1]
    print(f"  检测头: {type(head).__name__}")
    
    if hasattr(head, 'end2end'):
        print(f"  end2end: {head.end2end}")
    if hasattr(head, 'export'):
        print(f"  export: {head.export}")
    if hasattr(head, 'shape'):
        print(f"  shape: {head.shape}")
    if hasattr(head, 'dynamic'):
        print(f"  dynamic: {head.dynamic}")
    
    import inspect
    fwd_src = inspect.getsource(head.forward)
    has_e2e = 'end2end' in fwd_src or 'one2one' in fwd_src
    has_o2m = 'one2many' in fwd_src
    print(f"\n  forward 包含 end2end/one2one 分支: {has_e2e}")
    print(f"  forward 包含 one2many 分支: {has_o2m}")
    
    lines = fwd_src.split('\n')
    for i, line in enumerate(lines):
        if any(kw in line for kw in ['end2end', 'one2one', 'one2many', 'export', 'self.training', 'return']):
            print(f"    L{i}: {line.rstrip()}")

def main():
    best_pt = ROOT / "runs" / "detect" / "runs" / "calib" / "safehat_confcal_v3" / "weights" / "best.pt"
    ncnn_param, ncnn_bin = resolve_detect_asset_paths(ROOT)
    
    val_dir = ROOT / "data" / "valid" / "images"
    test_images = sorted(val_dir.glob("*.jpg"))[:1]
    
    if not test_images:
        print("未找到测试图片!")
        return
    
    img_path = str(test_images[0])
    print(f"测试图片: {img_path}")
    
    check_export_output_format(str(best_pt))
    
    test_pytorch_model(str(best_pt), img_path)
    
    test_pytorch_raw_output(str(best_pt), img_path)
    
    compare_ncnn_output(str(ncnn_param), str(ncnn_bin), img_path)

if __name__ == "__main__":
    main()
