#!/usr/bin/env python3
"""
YOLO26 E2E模式测试脚本

测试内容：
1. 验证E2E模型导出
2. 检查输出格式
3. 对比推理性能
4. 验证检测结果

使用方法：
    python test_yolo26_e2e.py
    python test_yolo26_e2e.py --model yolo26n.pt --image test.jpg
"""

import os
import sys
import time
import argparse
from pathlib import Path

def check_dependencies():
    """检查依赖"""
    try:
        import ultralytics
        print(f"✓ ultralytics 版本: {ultralytics.__version__}")
    except ImportError:
        print("✗ 请先安装 ultralytics: pip install ultralytics")
        sys.exit(1)
    
    try:
        import cv2
        print(f"✓ opencv 版本: {cv2.__version__}")
    except ImportError:
        print("⚠ opencv 未安装，部分测试将跳过")
        return False
    
    try:
        import numpy as np
        print(f"✓ numpy 版本: {np.__version__}")
    except ImportError:
        print("✗ 请先安装 numpy: pip install numpy")
        sys.exit(1)
    
    return True

def test_export_formats(model_path: str):
    """测试导出格式"""
    from ultralytics import YOLO
    
    print("\n" + "="*60)
    print("测试1: 导出格式验证")
    print("="*60)
    
    model = YOLO(model_path)
    model_name = Path(model_path).stem
    
    # 测试 One-to-Many 导出
    print(f"\n[1/2] 导出 One-to-Many 模式...")
    try:
        export_path_many = model.export(format="ncnn", end2end=False)
        print(f"  ✓ 导出成功: {export_path_many}")
    except Exception as e:
        print(f"  ✗ 导出失败: {e}")
        return False
    
    # 测试 E2E 导出
    print(f"\n[2/2] 导出 E2E (One-to-One) 模式...")
    try:
        export_path_e2e = model.export(format="ncnn", end2end=True)
        print(f"  ✓ 导出成功: {export_path_e2e}")
    except Exception as e:
        print(f"  ✗ 导出失败: {e}")
        return False
    
    return True

def test_inference_speed(model_path: str, num_runs: int = 50):
    """测试推理速度对比"""
    from ultralytics import YOLO
    import numpy as np
    
    print("\n" + "="*60)
    print("测试2: 推理速度对比")
    print("="*60)
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    model = YOLO(model_path)
    
    # 预热
    print("\n预热中...")
    for _ in range(5):
        model.predict(test_image, verbose=False)
    
    # One-to-Many 模式测试
    print(f"\n[1/2] One-to-Many 模式 ({num_runs}次推理)...")
    times_many = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model.predict(test_image, verbose=False)
        times_many.append(time.perf_counter() - start)
    
    avg_many = np.mean(times_many) * 1000
    std_many = np.std(times_many) * 1000
    print(f"  平均耗时: {avg_many:.2f} ± {std_many:.2f} ms")
    
    # E2E 模式需要重新加载模型
    # 注意：这里测试的是Python端，Android端的性能提升会更明显
    print(f"\n[2/2] E2E模式 (需要导出后的NCNN模型测试)")
    print(f"  ⚠ Python端测试不反映Android端真实性能")
    print(f"  ⚠ Android端E2E模式预期速度提升: ~43%")
    
    return avg_many

def test_output_format(model_path: str):
    """测试输出格式"""
    from ultralytics import YOLO
    import numpy as np
    
    print("\n" + "="*60)
    print("测试3: 输出格式验证")
    print("="*60)
    
    model = YOLO(model_path)
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    results = model.predict(test_image, verbose=False)
    
    print(f"\n模型类型: {model.task}")
    print(f"检测结果数量: {len(results[0].boxes)}")
    
    if len(results[0].boxes) > 0:
        print(f"\n示例检测框:")
        box = results[0].boxes[0]
        print(f"  坐标 (xyxy): {box.xyxy[0].tolist()}")
        print(f"  置信度: {box.conf[0].item():.4f}")
        print(f"  类别ID: {int(box.cls[0].item())}")
    
    # 预期的Android端输出格式
    print(f"\n预期Android端输出格式:")
    print(f"  One-to-Many: (8400, 84) - 需要NMS后处理")
    print(f"  E2E模式:     (300, 6)  - 直接可用，无需NMS")
    
    return True

def test_detection_accuracy(model_path: str, image_path: str = None):
    """测试检测精度"""
    from ultralytics import YOLO
    import numpy as np
    
    print("\n" + "="*60)
    print("测试4: 检测精度验证")
    print("="*60)
    
    model = YOLO(model_path)
    
    if image_path and os.path.exists(image_path):
        print(f"\n使用测试图像: {image_path}")
        results = model.predict(image_path, verbose=False)
    else:
        # 使用内置测试
        print(f"\n使用COCO验证集样本...")
        try:
            results = model.predict("https://ultralytics.com/images/bus.jpg", verbose=False)
        except Exception as e:
            print(f"  ⚠ 无法加载测试图像: {e}")
            return True
    
    print(f"\n检测结果:")
    print(f"  检测到 {len(results[0].boxes)} 个目标")
    
    # 获取类别名称
    names = model.names
    
    # 打印检测结果
    for i, box in enumerate(results[0].boxes):
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()
        cls_name = names.get(cls_id, f"class_{cls_id}")
        print(f"  [{i+1}] {cls_name}: {conf:.2%}")
    
    return True

def generate_android_test_commands():
    """生成Android测试命令"""
    print("\n" + "="*60)
    print("Android端测试指南")
    print("="*60)
    
    print("""
1. 复制模型文件到assets目录:
   
   # One-to-Many模式
   cp yolo26n_ncnn_model/model.ncnn.param app/src/main/assets/yolo26n.ncnn.param
   cp yolo26n_ncnn_model/model.ncnn.bin app/src/main/assets/yolo26n.ncnn.bin
   
   # E2E模式 (推荐)
    cp yolo26n_ncnn_model/model.ncnn.param app/src/main/assets/yolo26n_safehat.ncnn.param
    cp yolo26n_ncnn_model/model.ncnn.bin app/src/main/assets/yolo26n_safehat.ncnn.bin
    # 如需兼容旧加载链路，可同步复制为 yolo26n_e2e.ncnn.*

2. 编译并运行App

3. 使用adb查看Logcat:
   adb logcat -s ncnn

4. 预期输出:
   
   # One-to-Many模式
   D/ncnn: Output shape: w=84 h=8400 c=1
   D/ncnn: Using YOLO26 processing (no DFL)
   
   # E2E模式
   D/ncnn: Output shape: w=6 h=300 c=1
   D/ncnn: Using YOLO26 E2E processing (no NMS needed)
   D/ncnn: E2E mode: detected X objects above threshold 0.25

5. 性能对比:
   - 切换模型选择器: v26-n-320 vs v26-n-E2E-320
   - 观察FPS变化
   - E2E模式应有明显速度提升
""")

def verify_ncnn_files():
    """验证NCNN文件"""
    print("\n" + "="*60)
    print("测试5: NCNN文件验证")
    print("="*60)
    
    assets_dir = Path("app/src/main/assets")
    
    expected_files = [
        # One-to-Many
        ("yolo26n.ncnn.param", "yolo26n.ncnn.bin", "检测-OneToMany"),
        ("yolo26n_seg.ncnn.param", "yolo26n_seg.ncnn.bin", "分割-OneToMany"),
        ("yolo26n_pose.ncnn.param", "yolo26n_pose.ncnn.bin", "姿态-OneToMany"),
        ("yolo26n_obb.ncnn.param", "yolo26n_obb.ncnn.bin", "旋转框-OneToMany"),
        # E2E
        ("yolo26n_safehat.ncnn.param", "yolo26n_safehat.ncnn.bin", "检测-SafeHat主资产"),
        ("yolo26n_e2e.ncnn.param", "yolo26n_e2e.ncnn.bin", "检测-E2E"),
        ("yolo26n_seg_e2e.ncnn.param", "yolo26n_seg_e2e.ncnn.bin", "分割-E2E"),
        ("yolo26n_pose_e2e.ncnn.param", "yolo26n_pose_e2e.ncnn.bin", "姿态-E2E"),
        ("yolo26n_obb_e2e.ncnn.param", "yolo26n_obb_e2e.ncnn.bin", "旋转框-E2E"),
    ]
    
    print(f"\n检查目录: {assets_dir.absolute()}")
    print()
    
    found_count = 0
    for param, bin_file, desc in expected_files:
        param_path = assets_dir / param
        bin_path = assets_dir / bin_file
        
        param_exists = param_path.exists()
        bin_exists = bin_path.exists()
        
        if param_exists and bin_exists:
            param_size = param_path.stat().st_size / 1024
            bin_size = bin_path.stat().st_size / 1024 / 1024
            print(f"  ✓ {desc}: {param} ({param_size:.1f}KB) + {bin_file} ({bin_size:.1f}MB)")
            found_count += 1
        elif param_exists or bin_exists:
            print(f"  ⚠ {desc}: 文件不完整")
        else:
            print(f"  ✗ {desc}: 未找到")
    
    print(f"\n找到 {found_count}/{len(expected_files)} 组模型文件")
    
    return found_count > 0

def main():
    parser = argparse.ArgumentParser(description="YOLO26 E2E模式测试")
    parser.add_argument("--model", default="yolo26n.pt", help="模型路径")
    parser.add_argument("--image", default=None, help="测试图像路径")
    parser.add_argument("--runs", type=int, default=50, help="速度测试次数")
    parser.add_argument("--skip-export", action="store_true", help="跳过导出测试")
    args = parser.parse_args()
    
    print("="*60)
    print("YOLO26 E2E模式测试套件")
    print("="*60)
    
    # 检查依赖
    has_cv2 = check_dependencies()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"\n⚠ 模型文件不存在: {args.model}")
        print("请先下载YOLO26模型或指定正确的模型路径")
        print("\n下载命令:")
        print("  from ultralytics import YOLO")
        print("  model = YOLO('yolo26n.pt')  # 自动下载")
        
        # 仍然运行部分测试
        verify_ncnn_files()
        generate_android_test_commands()
        return
    
    # 运行测试
    print(f"\n使用模型: {args.model}")
    
    if not args.skip_export:
        test_export_formats(args.model)
    
    test_output_format(args.model)
    test_inference_speed(args.model, args.runs)
    test_detection_accuracy(args.model, args.image)
    verify_ncnn_files()
    generate_android_test_commands()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
    print("""
下一步:
1. 运行导出脚本: python export_yolo26_ncnn.py
2. 复制模型到assets目录
3. 在Android Studio中编译运行
4. 选择E2E模型测试性能提升
""")

if __name__ == "__main__":
    main()
