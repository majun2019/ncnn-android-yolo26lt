import os
import sys
import time
import argparse
from pathlib import Path

def check_dependencies():
    try:
        from ultralytics import YOLO
        import numpy as np
        return True
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请安装: pip install ultralytics numpy")
        return False

def benchmark_pytorch(model_path: str, img_size: int, num_runs: int):
    from ultralytics import YOLO
    import numpy as np
    
    print(f"\n{'='*60}")
    print(f"PyTorch 推理基准测试")
    print(f"{'='*60}")
    print(f"模型: {model_path}")
    print(f"图像尺寸: {img_size}x{img_size}")
    print(f"测试次数: {num_runs}")
    
    model = YOLO(model_path)
    test_image = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    
    print("\n预热中 (10次)...")
    for _ in range(10):
        model.predict(test_image, verbose=False)
    
    print(f"测试中 ({num_runs}次)...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        results = model.predict(test_image, verbose=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{num_runs}")
    
    times = np.array(times) * 1000
    
    print(f"\n结果:")
    print(f"  平均: {np.mean(times):.2f} ms")
    print(f"  最小: {np.min(times):.2f} ms")
    print(f"  最大: {np.max(times):.2f} ms")
    print(f"  标准差: {np.std(times):.2f} ms")
    print(f"  FPS: {1000/np.mean(times):.1f}")
    
    return {
        'mean': np.mean(times),
        'min': np.min(times),
        'max': np.max(times),
        'std': np.std(times),
        'fps': 1000/np.mean(times)
    }

def benchmark_sizes(model_path: str, sizes: list, num_runs: int):
    from ultralytics import YOLO
    import numpy as np
    
    print(f"\n{'='*60}")
    print(f"多尺寸基准测试")
    print(f"{'='*60}")
    
    model = YOLO(model_path)
    results = {}
    
    for size in sizes:
        print(f"\n测试尺寸: {size}x{size}")
        test_image = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        
        for _ in range(5):
            model.predict(test_image, verbose=False)
        
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            model.predict(test_image, verbose=False)
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times) * 1000
        fps = 1000 / avg_time
        
        results[size] = {'time': avg_time, 'fps': fps}
        print(f"  平均: {avg_time:.2f} ms, FPS: {fps:.1f}")
    
    return results

def generate_report(results: dict, model_name: str):
    print(f"\n{'='*60}")
    print(f"性能测试报告")
    print(f"{'='*60}")
    
    print(f"\n模型: {model_name}")
    print(f"\n{'尺寸':^10} | {'耗时 (ms)':^12} | {'FPS':^8}")
    print("-" * 36)
    
    for size, data in sorted(results.items()):
        print(f"{size:^10} | {data['time']:^12.2f} | {data['fps']:^8.1f}")
    
    print("\n" + "="*60)
    print("Android端性能预估 (基于NCNN)")
    print("="*60)
    print("""
注意: 以上为PyTorch CPU推理结果
""")

def main():
    parser = argparse.ArgumentParser(description="YOLO26性能基准测试")
    parser.add_argument("--model", default="yolo26n.pt", help="模型路径")
    parser.add_argument("--size", type=int, default=640, help="图像尺寸")
    parser.add_argument("--runs", type=int, default=50, help="测试次数")
    parser.add_argument("--multi-size", action="store_true", help="测试多个尺寸")
    args = parser.parse_args()
    
    print("="*60)
    print("YOLO26 性能基准测试")
    print("="*60)
    
    if not check_dependencies():
        return
    
    if not os.path.exists(args.model):
        print(f"\n⚠ 模型不存在: {args.model}")
        print("请先下载模型:")
        print("  from ultralytics import YOLO")
        print("  model = YOLO('yolo26n.pt')")
        
        generate_report({}, args.model)
        return
    
    if args.multi_size:
        sizes = [320, 480, 640]
        results = benchmark_sizes(args.model, sizes, args.runs)
        generate_report(results, args.model)
    else:
        benchmark_pytorch(args.model, args.size, args.runs)
        generate_report({args.size: {'time': 0, 'fps': 0}}, args.model)

if __name__ == "__main__":
    main()
