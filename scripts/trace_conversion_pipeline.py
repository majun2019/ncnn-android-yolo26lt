import os, sys, json, time
from pathlib import Path
import numpy as np
import cv2

CLASSES_10 = [
    "Hardhat", "Mask", "No-Hardhat", "No-Mask", "No-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "Machinery", "Vehicle"
]

def resolve_detect_asset_paths(root: Path) -> tuple[str, str]:
    assets_dir = root / "app" / "src" / "main" / "assets"
    for stem in ("yolo26n_safehat.ncnn", "yolo26n_e2e.ncnn"):
        param = assets_dir / f"{stem}.param"
        bin_path = assets_dir / f"{stem}.bin"
        if param.exists() and bin_path.exists():
            return str(param.resolve()), str(bin_path.resolve())
    return str((assets_dir / "yolo26n_safehat.ncnn.param").resolve()), str((assets_dir / "yolo26n_safehat.ncnn.bin").resolve())

def create_blank_images():
    imgs = {}
    imgs["gray_128"] = np.full((640, 640, 3), 128, dtype=np.uint8)
    imgs["gray_200"] = np.full((640, 640, 3), 200, dtype=np.uint8)
    imgs["white"]    = np.full((640, 640, 3), 255, dtype=np.uint8)
    imgs["dark_30"]  = np.full((640, 640, 3), 30, dtype=np.uint8)
    np.random.seed(42)
    imgs["noise"]    = np.random.randint(80, 200, (640, 640, 3), dtype=np.uint8)
    imgs["gray_114"] = np.full((640, 640, 3), 114, dtype=np.uint8)
    return imgs

def analyze_scores(cls_data, already_sigmoid=False, label=""):
    if not already_sigmoid:
        from scipy.special import expit
        cls_sig = expit(cls_data)
    else:
        cls_sig = cls_data

    nc = cls_sig.shape[0]
    n_boxes = cls_sig.shape[1]

    max_per_box = cls_sig.max(axis=0)

    stats = {
        "label": label,
        "nc": nc,
        "n_boxes": n_boxes,
        "global_min": float(cls_sig.min()),
        "global_max": float(cls_sig.max()),
        "global_mean": float(cls_sig.mean()),
        "max_score_p50": float(np.percentile(max_per_box, 50)),
        "max_score_p90": float(np.percentile(max_per_box, 90)),
        "max_score_p99": float(np.percentile(max_per_box, 99)),
        "max_score_max": float(max_per_box.max()),
        "n_gt_0.01": int((max_per_box > 0.01).sum()),
        "n_gt_0.10": int((max_per_box > 0.10).sum()),
        "n_gt_0.25": int((max_per_box > 0.25).sum()),
        "n_gt_0.35": int((max_per_box > 0.35).sum()),
        "n_gt_0.50": int((max_per_box > 0.50).sum()),
        "n_gt_0.90": int((max_per_box > 0.90).sum()),
        "per_class": [],
    }

    for c in range(nc):
        row = cls_sig[c, :]
        stats["per_class"].append({
            "class": c,
            "mean": float(row.mean()),
            "max": float(row.max()),
            "p50": float(np.median(row)),
            "gt_0.35": int((row > 0.35).sum()),
            "gt_0.50": int((row > 0.50).sum()),
        })

    return stats

def print_stats(stats):
    label = stats["label"]
    print(f"  [{label}] nc={stats['nc']}, boxes={stats['n_boxes']}")
    print(f"    sigmoid: min={stats['global_min']:.6f} max={stats['global_max']:.6f} mean={stats['global_mean']:.6f}")
    print(f"    max_score: p50={stats['max_score_p50']:.6f} p90={stats['max_score_p90']:.6f} "
          f"p99={stats['max_score_p99']:.6f} max={stats['max_score_max']:.6f}")
    print(f"    >0.01:{stats['n_gt_0.01']:5d}  >0.10:{stats['n_gt_0.10']:5d}  "
          f">0.25:{stats['n_gt_0.25']:5d}  >0.35:{stats['n_gt_0.35']:5d}  "
          f">0.50:{stats['n_gt_0.50']:5d}  >0.90:{stats['n_gt_0.90']:5d}")

    bad = [c for c in stats["per_class"] if c["max"] > 0.01]
    if bad:
        print(f"    异常类别 (max>0.01):")
        for c in bad:
            cname = CLASSES_10[c["class"]] if stats["nc"] == 10 and c["class"] < 10 else f"cls_{c['class']}"
            print(f"      {cname:15s}: mean={c['mean']:.4f} max={c['max']:.4f} "
                  f">0.35={c['gt_0.35']:5d} >0.50={c['gt_0.50']:5d}")
    else:
        print(f"    ✅ 所有类别 max < 0.01 (背景抑制正常)")

def test_pytorch(model_path, images):
    print(f"\n{'='*70}")
    print(f"[Stage 1: PyTorch] {os.path.basename(model_path)}")
    print(f"{'='*70}")

    import torch
    from ultralytics import YOLO

    model = YOLO(model_path)
    nc = model.model.nc
    names = model.names
    print(f"  nc={nc}, names={names}")

    all_stats = {}
    for img_name, img in images.items():
        print(f"\n  --- {img_name} ---")

        results = model(img, conf=0.25, verbose=False)
        n_det = len(results[0].boxes)
        print(f"  标准推理 (conf>0.25): {n_det} 检测")

        im = torch.from_numpy(img.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        device = next(model.model.parameters()).device
        im = im.to(device)

        model.model.eval()
        with torch.no_grad():
            preds = model.model(im)

        if isinstance(preds, (list, tuple)):
            raw = preds[0]
        else:
            raw = preds

        raw = raw.squeeze(0).cpu().numpy()
        print(f"  raw output shape: {raw.shape}")

        if raw.shape[0] == nc + 4:
            cls_data = raw[4:, :]
        elif raw.shape[1] == nc + 4:
            cls_data = raw[:, 4:].T
        else:
            print(f"  ⚠️ 无法解析输出格式 {raw.shape}")
            continue

        val_range = cls_data.max() - cls_data.min()
        is_sigmoid = cls_data.min() >= -0.01 and cls_data.max() <= 1.01
        print(f"  cls范围: [{cls_data.min():.4f}, {cls_data.max():.4f}] {'(已sigmoid)' if is_sigmoid else '(logit)'}")

        stats = analyze_scores(cls_data, already_sigmoid=is_sigmoid, label=f"PT-{img_name}")
        print_stats(stats)
        all_stats[f"PT-{img_name}"] = stats

    return all_stats

def test_onnx_export(model_path, images, export_dir=None):
    print(f"\n{'='*70}")
    print(f"[Stage 2: ONNX Export] {os.path.basename(model_path)}")
    print(f"{'='*70}")

    from ultralytics import YOLO

    model = YOLO(model_path)
    nc = model.model.nc

    print("  导出 ONNX (end2end=False)...")
    onnx_path = model.export(format="onnx", imgsz=640, half=False)
    print(f"  ONNX 文件: {onnx_path}")

    try:
        import onnxruntime as ort
    except ImportError:
        print("  ⚠️ onnxruntime 未安装, 跳过 ONNX 验证")
        return {}, str(onnx_path)

    sess = ort.InferenceSession(str(onnx_path))
    inp = sess.get_inputs()[0]
    outs = sess.get_outputs()
    print(f"  ONNX input: {inp.name}, shape={inp.shape}")
    for o in outs:
        print(f"  ONNX output: {o.name}, shape={o.shape}")

    all_stats = {}
    for img_name, img in images.items():
        print(f"\n  --- {img_name} ---")

        blob = img.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = blob[::-1, :, :].copy()
        blob = np.expand_dims(blob, 0)

        results = sess.run(None, {inp.name: blob})

        for i, (o_info, o_data) in enumerate(zip(outs, results)):
            print(f"  output[{i}] ({o_info.name}): shape={o_data.shape}, "
                  f"range=[{o_data.min():.6f}, {o_data.max():.6f}]")

        out = results[0].squeeze(0)
        if out.shape[0] == nc + 4:
            cls_data = out[4:, :]
        elif out.shape[1] == nc + 4:
            cls_data = out[:, 4:].T
        else:
            print(f"  ⚠️ 无法解析 ONNX 输出 {out.shape}")
            continue

        is_sigmoid = cls_data.min() >= -0.01 and cls_data.max() <= 1.01
        print(f"  cls范围: [{cls_data.min():.4f}, {cls_data.max():.4f}] {'(已sigmoid)' if is_sigmoid else '(logit)'}")

        stats = analyze_scores(cls_data, already_sigmoid=is_sigmoid, label=f"ONNX-{img_name}")
        print_stats(stats)
        all_stats[f"ONNX-{img_name}"] = stats

    return all_stats, str(onnx_path)

def test_ncnn_ultralytics_export(model_path, images):
    print(f"\n{'='*70}")
    print(f"[Stage 3a: NCNN via Ultralytics] {os.path.basename(model_path)}")
    print(f"{'='*70}")

    from ultralytics import YOLO

    model = YOLO(model_path)
    nc = model.model.nc

    print("  导出 NCNN (end2end=False, half=False)...")
    ncnn_path = model.export(format="ncnn", imgsz=640, half=False, end2end=False)
    print(f"  NCNN 目录: {ncnn_path}")

    return test_ncnn_model(str(ncnn_path), images, nc, label_prefix="NCNN-ultra")

def test_ncnn_model(ncnn_dir_or_param, images, expected_nc=None, label_prefix="NCNN"):
    try:
        import ncnn as pyncnn
    except ImportError:
        print("  ⚠️ pyncnn 未安装, 跳过")
        return {}

    if os.path.isdir(ncnn_dir_or_param):
        param_path = os.path.join(ncnn_dir_or_param, "model.ncnn.param")
        bin_path = os.path.join(ncnn_dir_or_param, "model.ncnn.bin")
    else:
        param_path = ncnn_dir_or_param
        bin_path = ncnn_dir_or_param.replace(".param", ".bin")

    if not os.path.exists(param_path):
        print(f"  ⚠️ 找不到 {param_path}")
        return {}

    print(f"  NCNN param: {param_path}")

    net = pyncnn.Net()
    net.load_param(param_path)
    net.load_model(bin_path)

    input_names = net.input_names()
    output_names = net.output_names()
    print(f"  输入: {input_names}, 输出: {output_names}")

    all_stats = {}
    for img_name, img in images.items():
        print(f"\n  --- {img_name} ---")

        mat_in = pyncnn.Mat.from_pixels(img, pyncnn.Mat.PixelType.PIXEL_BGR2RGB, img.shape[1], img.shape[0])
        norm_vals = [1/255.0, 1/255.0, 1/255.0]
        mat_in.substract_mean_normalize([], norm_vals)

        ex = net.create_extractor()
        ex.input(input_names[0], mat_in)

        ret, mat_out = ex.extract(output_names[0])
        if ret != 0:
            print(f"  ❌ 提取失败")
            continue

        arr = np.array(mat_out)
        if arr.ndim == 3:
            arr = arr[0]

        print(f"  output shape: w={mat_out.w}, h={mat_out.h}, c={mat_out.c}")
        print(f"  numpy shape: {arr.shape}, range=[{arr.min():.6f}, {arr.max():.6f}]")

        if arr.shape[0] < arr.shape[1]:
            nc = arr.shape[0] - 4
            cls_data = arr[4:, :]
        elif arr.shape[1] < arr.shape[0]:
            nc = arr.shape[1] - 4
            cls_data = arr[:, 4:].T
        else:
            print(f"  ⚠️ 无法解析")
            continue

        is_sigmoid = cls_data.min() >= -0.01 and cls_data.max() <= 1.01
        print(f"  cls范围: [{cls_data.min():.6f}, {cls_data.max():.6f}] {'(已sigmoid)' if is_sigmoid else '(logit)'}")

        stats = analyze_scores(cls_data, already_sigmoid=is_sigmoid, label=f"{label_prefix}-{img_name}")
        print_stats(stats)
        all_stats[f"{label_prefix}-{img_name}"] = stats

        del ex

    del net
    return all_stats

def test_deployed_ncnn(param_path, bin_path, images, label_prefix="Deployed"):
    print(f"\n{'='*70}")
    print(f"[Stage 4: Deployed NCNN] {os.path.basename(param_path)}")
    print(f"{'='*70}")
    return test_ncnn_model(param_path, images, label_prefix=label_prefix)

def simulate_cpp_reading(param_path, bin_path, images):
    print(f"\n{'='*70}")
    print(f"[Stage 5: 模拟C++数据读取] {os.path.basename(param_path)}")
    print(f"{'='*70}")

    try:
        import ncnn as pyncnn
    except ImportError:
        print("  ⚠️ pyncnn 未安装")
        return {}

    net = pyncnn.Net()
    net.load_param(param_path)
    net.load_model(bin_path)

    for img_name, img in list(images.items())[:2]:
        print(f"\n  --- {img_name} ---")

        mat_in = pyncnn.Mat.from_pixels(img, pyncnn.Mat.PixelType.PIXEL_BGR2RGB, img.shape[1], img.shape[0])
        norm_vals = [1/255.0, 1/255.0, 1/255.0]
        mat_in.substract_mean_normalize([], norm_vals)

        ex = net.create_extractor()
        ex.input(net.input_names()[0], mat_in)
        ret, out = ex.extract(net.output_names()[0])

        w, h, c = out.w, out.h, out.c
        print(f"  ncnn Mat: w={w}, h={h}, c={c}")

        num_class = h - 4
        total_boxes = w
        print(f"  C++视角: num_class={num_class}, total_boxes={total_boxes}")

        print(f"\n  C++方式逐box读取 (前5个box):")
        arr = np.array(out)
        if arr.ndim == 3:
            arr = arr[0]

        for box_idx in range(min(5, total_boxes)):
            scores = []
            for k in range(num_class):
                s = arr[4 + k, box_idx]
                scores.append(s)
            best_k = np.argmax(scores)
            print(f"    box[{box_idx}]: scores={[f'{s:.6f}' for s in scores]} "
                  f"best=class{best_k}({scores[best_k]:.6f})")

        print(f"\n  最后3个类别通道 (SafeVest=7, Machinery=8, Vehicle=9) 在空白图上:")
        for k in [7, 8, 9]:
            if k < num_class:
                row = arr[4 + k, :]
                cname = CLASSES_10[k] if num_class == 10 else f"cls_{k}"
                print(f"    {cname}: min={row.min():.6f} max={row.max():.6f} "
                      f"mean={row.mean():.6f} p50={np.median(row):.6f} "
                      f">0.35={int((row>0.35).sum())} >0.50={int((row>0.50).sum())}")

        del ex
    del net

def main():
    print("=" * 70)
    print("全链路特征值追踪诊断")
    print("PyTorch → ONNX → NCNN 逐环节追踪")
    print("=" * 70)

    workspace_root = Path(__file__).resolve().parents[1]
    SAFEHAT_PT = str((workspace_root / "modelback" / "yolo26n.pt").resolve())
    COCO_PT    = str((workspace_root / "modelback" / "yolo26n.pt").resolve())

    SAFEHAT_NCNN_ULTRA = str((workspace_root / "runs" / "detect" / "runs" / "train" / "yolo26n_safehat" / "weights" / "best_ncnn_model").resolve())
    SAFEHAT_NCNN_PNNX  = str((workspace_root / "modelback" / "yolo26n_ncnn_model" / "model.ncnn.param").resolve())
    COCO_NCNN          = str((workspace_root / "modelback" / "yolo26n_ncnn_model").resolve())
    DEPLOYED_PARAM, DEPLOYED_BIN = resolve_detect_asset_paths(workspace_root)

    images = create_blank_images()
    quick_images = {k: v for k, v in list(images.items())[:3]}

    results_summary = {}

    print("\n\n" + "#" * 70)
    print("# A) SafeHat 训练模型 (10类) 全链路追踪")
    print("#" * 70)

    if os.path.exists(SAFEHAT_PT):
        pt_stats = test_pytorch(SAFEHAT_PT, quick_images)
        results_summary["SafeHat-PT"] = pt_stats

        onnx_stats, onnx_path = test_onnx_export(SAFEHAT_PT, quick_images)
        results_summary["SafeHat-ONNX"] = onnx_stats

        ncnn_ultra_stats = test_ncnn_ultralytics_export(SAFEHAT_PT, quick_images)
        results_summary["SafeHat-NCNN-fresh"] = ncnn_ultra_stats

    if os.path.exists(SAFEHAT_NCNN_ULTRA):
        print(f"\n{'='*70}")
        print(f"[已有 NCNN: best_ncnn_model (Ultralytics export)]")
        print(f"{'='*70}")
        stats = test_ncnn_model(SAFEHAT_NCNN_ULTRA, quick_images, label_prefix="NCNN-ultra-existing")
        results_summary["SafeHat-NCNN-ultra-existing"] = stats

    if os.path.exists(SAFEHAT_NCNN_PNNX):
        print(f"\n{'='*70}")
        print(f"[已有 NCNN: ncnnmodel (PNNX export)]")
        print(f"{'='*70}")
        stats = test_ncnn_model(SAFEHAT_NCNN_PNNX, quick_images, label_prefix="NCNN-pnnx-existing")
        results_summary["SafeHat-NCNN-pnnx-existing"] = stats

    if os.path.exists(DEPLOYED_PARAM):
        stats = test_deployed_ncnn(DEPLOYED_PARAM, DEPLOYED_BIN, quick_images, label_prefix="Deployed")
        results_summary["SafeHat-Deployed"] = stats

    if os.path.exists(DEPLOYED_PARAM):
        simulate_cpp_reading(DEPLOYED_PARAM, DEPLOYED_BIN, quick_images)

    print("\n\n" + "#" * 70)
    print("# B) COCO yolo26n 原模型 (80类) 基线对比")
    print("#" * 70)

    if os.path.exists(COCO_PT):
        coco_pt_stats = test_pytorch(COCO_PT, {"gray_128": quick_images["gray_128"]})
        results_summary["COCO-PT"] = coco_pt_stats

    if os.path.exists(COCO_NCNN):
        print(f"\n{'='*70}")
        print(f"[COCO yolo26n NCNN]")
        print(f"{'='*70}")
        coco_ncnn_stats = test_ncnn_model(COCO_NCNN, {"gray_128": quick_images["gray_128"]}, label_prefix="COCO-NCNN")
        results_summary["COCO-NCNN"] = coco_ncnn_stats

    print("\n\n" + "=" * 70)
    print("总结: 空白图像(gray_128)上各环节的 max_class_score")
    print("=" * 70)
    print(f"{'环节':<35s} {'p50':>10s} {'p90':>10s} {'p99':>10s} {'max':>10s} {'>0.35':>8s} {'>0.50':>8s}")
    print("-" * 95)

    for group_name, group_stats in results_summary.items():
        for key, stats in group_stats.items():
            if "gray_128" in key or "gray_128" in stats.get("label", ""):
                print(f"  {stats['label']:<33s} "
                      f"{stats['max_score_p50']:10.6f} "
                      f"{stats['max_score_p90']:10.6f} "
                      f"{stats['max_score_p99']:10.6f} "
                      f"{stats['max_score_max']:10.6f} "
                      f"{stats['n_gt_0.35']:8d} "
                      f"{stats['n_gt_0.50']:8d}")

    print("\n" + "=" * 70)
    print("诊断结论:")
    print("=" * 70)
    print("""
    对比 PyTorch vs ONNX vs NCNN 的分数分布:
    - 如果 PyTorch 正常, ONNX 也正常, NCNN 异常 → 问题在 ONNX→NCNN 转换
    - 如果 PyTorch 正常, ONNX 已异常            → 问题在 PyTorch→ONNX 导出
    - 如果 PyTorch 已异常                       → 模型训练本身有问题
    - 如果所有NCNN模型(ultra/pnnx)都异常        → 问题是系统性的NCNN转换问题
    - 如果只有某种NCNN转换路径异常              → 那种转换工具有bug
    """)

if __name__ == "__main__":
    main()
