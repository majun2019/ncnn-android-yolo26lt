#!/usr/bin/env python3
"""
诊断脚本：在空白/无目标图像上测试模型的原始分数分布
验证模型是否具备"物体 vs 背景"区分能力
"""
import sys, os
from pathlib import Path
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

def create_test_images():
    """创建几种无目标的测试图像"""
    images = {}
    
    # 1. 纯灰色图 (模拟墙壁)
    gray = np.full((640, 480, 3), 128, dtype=np.uint8)
    images["pure_gray_wall"] = gray
    
    # 2. 随机噪声图
    noise = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    images["random_noise"] = noise
    
    # 3. 模拟室内场景 - 有纹理的墙壁
    wall = np.full((640, 480, 3), 200, dtype=np.uint8)
    # 添加一些水平线条 (模拟天花板/地板线)
    for y in range(0, 640, 80):
        cv2.line(wall, (0, y), (480, y), (180, 180, 180), 2)
    # 添加垂直线 (模拟门框)
    cv2.rectangle(wall, (150, 100), (330, 500), (160, 160, 160), 3)
    images["indoor_scene"] = wall
    
    return images


def test_pytorch_raw_scores(model_path: str, img: np.ndarray, name: str):
    """用PyTorch模型直接检查原始输出分数分布"""
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    
    print(f"\n{'='*70}")
    print(f"[PyTorch] 测试: {name}  ({img.shape[1]}x{img.shape[0]})")
    print(f"{'='*70}")
    
    model = YOLO(model_path)
    
    # 标准推理（会经过NMS过滤）
    results = model.predict(img, imgsz=640, conf=0.25, verbose=False)
    n_det = len(results[0].boxes) if results else 0
    print(f"  标准推理检测到: {n_det} 个对象 (conf>=0.25)")
    
    if n_det > 0:
        for i, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"    [{i}] {CLASSES[cls_id]:15s} conf={conf:.4f}")
    
    # 获取原始输出 (O2M head)
    letterbox = LetterBox(new_shape=(640, 640), auto=False)
    img_lb = letterbox(image=img)
    img_tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    
    torch_model = model.model
    torch_model.eval()
    
    # 确保tensor与模型在同一设备
    device = next(torch_model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        preds = torch_model(img_tensor)
    
    # preds 通常是 (batch, 4+nc, 8400) 或类似格式
    if isinstance(preds, (tuple, list)):
        raw = preds[0] if isinstance(preds[0], torch.Tensor) else preds
    else:
        raw = preds
    
    if isinstance(raw, torch.Tensor):
        raw = raw.squeeze(0)  # (14, 8400)
        print(f"\n  原始输出 shape: {raw.shape}")
        
        nc = raw.shape[0] - 4
        if nc != 10:
            print(f"  ⚠ nc={nc}, 期望10")
            # 可能输出格式不同, 尝试转置
            if raw.shape[1] - 4 == 10:
                raw = raw.T
                nc = 10
        
        # bbox部分
        bbox = raw[:4, :]
        print(f"  bbox x_center: min={bbox[0].min():.1f} max={bbox[0].max():.1f}")
        print(f"  bbox y_center: min={bbox[1].min():.1f} max={bbox[1].max():.1f}")
        print(f"  bbox width:    min={bbox[2].min():.1f} max={bbox[2].max():.1f}")
        print(f"  bbox height:   min={bbox[3].min():.1f} max={bbox[3].max():.1f}")
        
        # 类别分数部分 - 注意这些是RAW LOGITS (sigmoid之前)
        cls_logits = raw[4:, :].numpy() if not raw.is_cuda else raw[4:, :].cpu().numpy()
        cls_sigmoid = 1.0 / (1.0 + np.exp(-cls_logits))
        
        print(f"\n  === 类别 LOGIT (sigmoid之前) 分布 ===")
        print(f"  全局: min={cls_logits.min():.4f} max={cls_logits.max():.4f} mean={cls_logits.mean():.4f}")
        
        print(f"\n  === 类别 SIGMOID 分数分布 ===")
        print(f"  全局: min={cls_sigmoid.min():.6f} max={cls_sigmoid.max():.6f} mean={cls_sigmoid.mean():.6f}")
        
        # 每个box的最大类别分数
        max_per_box = cls_sigmoid.max(axis=0)
        n_boxes = max_per_box.shape[0]
        
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        pvals = np.percentile(max_per_box, percentiles)
        print(f"\n  max_class_score 分位数 (共{n_boxes}个box):")
        for p, v in zip(percentiles, pvals):
            print(f"    p{p:02d} = {v:.6f}")
        
        # 统计高分比例
        thresholds = [0.1, 0.25, 0.35, 0.5, 0.7, 0.9, 0.99]
        print(f"\n  高分box比例:")
        for t in thresholds:
            count = np.sum(max_per_box > t)
            frac = count / n_boxes
            print(f"    > {t:.2f}: {count:5d} ({frac*100:.1f}%)")
        
        # 每个类别统计
        print(f"\n  === 各类别 sigmoid 分数统计 ===")
        print(f"  {'类别':15s} {'mean':>8s} {'max':>8s} {'p50':>8s} {'p90':>8s} {'>0.5':>6s} {'>0.25':>6s}")
        for k in range(nc):
            row = cls_sigmoid[k, :]
            print(f"  {CLASSES[k]:15s} {row.mean():8.4f} {row.max():8.4f} "
                  f"{np.median(row):8.4f} {np.percentile(row, 90):8.4f} "
                  f"{np.sum(row>0.5):6d} {np.sum(row>0.25):6d}")


def test_ncnn_raw_scores(param_path: str, bin_path: str, img: np.ndarray, name: str):
    """用NCNN模型检查原始输出分数分布"""
    try:
        import ncnn
    except ImportError:
        print(f"\n[NCNN] pyncnn未安装, 跳过")
        return
    
    print(f"\n{'='*70}")
    print(f"[NCNN] 测试: {name}  ({img.shape[1]}x{img.shape[0]})")
    print(f"{'='*70}")
    
    net = ncnn.Net()
    net.load_param(str(param_path))
    net.load_model(str(bin_path))
    
    # letterbox
    h, w = img.shape[:2]
    target = 640
    scale = min(target/w, target/h)
    new_w, new_h = int(w*scale), int(h*scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    
    mat_in = ncnn.Mat.from_pixels(img_resized, ncnn.Mat.PixelType.PIXEL_BGR2RGB, new_w, new_h)
    
    wpad = (new_w + 31) // 32 * 32 - new_w
    hpad = (new_h + 31) // 32 * 32 - new_h
    mat_pad = ncnn.copy_make_border(mat_in, hpad//2, hpad-hpad//2, wpad//2, wpad-wpad//2,
                                     ncnn.BorderType.BORDER_CONSTANT, 114.0)
    
    norm_vals = [1/255.0, 1/255.0, 1/255.0]
    mat_pad.substract_mean_normalize([], norm_vals)
    
    ex = net.create_extractor()
    ex.input("in0", mat_pad)
    ret, out = ex.extract("out0")
    
    print(f"  NCNN输出: w={out.w}, h={out.h}, c={out.c}")
    
    arr = np.array(out)
    if arr.ndim == 3:
        arr = arr[0]
    
    nc = out.h - 4
    n_boxes = out.w
    print(f"  classes={nc}, boxes={n_boxes}")
    
    # NCNN输出已经过sigmoid
    cls_scores = arr[4:, :]  # (10, 8400), 已sigmoid
    
    print(f"\n  === NCNN类别分数分布 (已sigmoid) ===")
    print(f"  全局: min={cls_scores.min():.6f} max={cls_scores.max():.6f} mean={cls_scores.mean():.6f}")
    
    max_per_box = cls_scores.max(axis=0)
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    pvals = np.percentile(max_per_box, percentiles)
    print(f"\n  max_class_score 分位数 (共{n_boxes}个box):")
    for p, v in zip(percentiles, pvals):
        print(f"    p{p:02d} = {v:.6f}")
    
    thresholds = [0.1, 0.25, 0.35, 0.5, 0.7, 0.9, 0.99]
    print(f"\n  高分box比例:")
    for t in thresholds:
        count = np.sum(max_per_box > t)
        frac = count / n_boxes
        print(f"    > {t:.2f}: {count:5d} ({frac*100:.1f}%)")
    
    print(f"\n  === 各类别 sigmoid 分数统计 ===")
    print(f"  {'类别':15s} {'mean':>8s} {'max':>8s} {'p50':>8s} {'p90':>8s} {'>0.5':>6s} {'>0.25':>6s}")
    for k in range(nc):
        row = cls_scores[k, :]
        print(f"  {CLASSES[k]:15s} {row.mean():8.4f} {row.max():8.4f} "
              f"{np.median(row):8.4f} {np.percentile(row, 90):8.4f} "
              f"{np.sum(row>0.5):6d} {np.sum(row>0.25):6d}")


def main():
    best_pt = ROOT / "runs" / "detect" / "runs" / "calib" / "safehat_confcal_v3" / "weights" / "best.pt"
    ncnn_param, ncnn_bin = resolve_detect_asset_paths(ROOT)
    
    if not best_pt.exists():
        print(f"模型不存在: {best_pt}")
        return
    
    # 创建无目标测试图像
    test_images = create_test_images()
    
    # 也加载一张有目标的验证图 (对比用)
    val_dir = ROOT / "data" / "valid" / "images"
    val_imgs = sorted(val_dir.glob("*.jpg"))
    if val_imgs:
        val_img = cv2.imread(str(val_imgs[0]))
        if val_img is not None:
            test_images["val_with_objects"] = val_img
    
    # 测试 PyTorch
    print("\n" + "#"*70)
    print("# Part 1: PyTorch 模型原始输出分析")
    print("#"*70)
    for name, img in test_images.items():
        test_pytorch_raw_scores(str(best_pt), img, name)
    
    # 测试 NCNN
    if ncnn_param.exists() and ncnn_bin.exists():
        print("\n\n" + "#"*70)
        print("# Part 2: NCNN 模型输出分析")
        print("#"*70)
        for name, img in test_images.items():
            test_ncnn_raw_scores(str(ncnn_param), str(ncnn_bin), img, name)
    
    print("\n\n" + "#"*70)
    print("# 诊断总结")
    print("#"*70)
    print("""
    如果空白图像的分数分布为:
      - p50 < 0.05, >0.5比例 < 1%  → 模型背景抑制能力正常, 问题在C++后处理
      - p50 > 0.3, >0.5比例 > 10%  → 模型背景抑制能力不足, 需要重新训练
      - PyTorch正常但NCNN异常      → NCNN导出有问题
    """)


if __name__ == "__main__":
    main()
