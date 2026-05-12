#!/usr/bin/env python3
"""
YOLO26 NCNN输出格式验证工具

用于验证导出的NCNN模型输出格式是否符合预期。
需要安装 ncnn python 绑定或使用 pyncnn。

使用方法：
    python verify_ncnn_output.py --model yolo26n.ncnn.param
"""

import os
import sys
import argparse
from pathlib import Path

def parse_ncnn_param(param_path: str):
    """解析NCNN param文件，提取输出层信息"""
    print(f"\n解析文件: {param_path}")
    
    if not os.path.exists(param_path):
        print(f"  ✗ 文件不存在")
        return None
    
    layers = []
    output_layers = []
    
    with open(param_path, 'r') as f:
        lines = f.readlines()
    
    # 跳过magic number
    if lines[0].strip() == '7767517':
        lines = lines[1:]
    
    # 第一行是 layer_count blob_count
    header = lines[0].strip().split()
    layer_count = int(header[0])
    blob_count = int(header[1])
    
    print(f"  层数: {layer_count}, Blob数: {blob_count}")
    
    # 解析每一层
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 4:
            layer_type = parts[0]
            layer_name = parts[1]
            input_count = int(parts[2])
            output_count = int(parts[3])
            
            layers.append({
                'type': layer_type,
                'name': layer_name,
                'inputs': input_count,
                'outputs': output_count
            })
            
            # 查找输出层（通常以out开头或是最后几层）
            if layer_name.startswith('out') or 'output' in layer_name.lower():
                output_layers.append(layer_name)
    
    # 找到输入层
    input_layers = [l for l in layers if l['type'] == 'Input']
    print(f"  输入层: {[l['name'] for l in input_layers]}")
    
    # 找到Permute层（通常在输出前）
    permute_layers = [l for l in layers if l['type'] == 'Permute']
    
    return {
        'layer_count': layer_count,
        'blob_count': blob_count,
        'layers': layers,
        'input_layers': input_layers,
        'output_layers': output_layers,
        'permute_layers': permute_layers
    }

def detect_model_type(param_info: dict) -> str:
    """根据网络结构检测模型类型"""
    layer_types = [l['type'] for l in param_info['layers']]
    
    # E2E模型通常有特定的后处理层
    if 'Concat' in layer_types and param_info['layer_count'] > 200:
        # 检查是否有NMS相关层
        layer_names = [l['name'] for l in param_info['layers']]
        has_nms = any('nms' in name.lower() or 'topk' in name.lower() for name in layer_names)
        
        if has_nms:
            return "E2E (One-to-One)"
        else:
            return "One-to-Many"
    
    return "Unknown"

def verify_e2e_output_format(param_path: str):
    """验证E2E输出格式"""
    print("\n" + "="*60)
    print("NCNN模型输出格式验证")
    print("="*60)
    
    param_info = parse_ncnn_param(param_path)
    if not param_info:
        return False
    
    model_type = detect_model_type(param_info)
    print(f"\n检测到模型类型: {model_type}")
    
    # 显示最后几层
    print(f"\n最后10层:")
    for layer in param_info['layers'][-10:]:
        print(f"  {layer['type']:20s} {layer['name']}")
    
    return True

def compare_models(many_param: str, e2e_param: str):
    """对比两种模型"""
    print("\n" + "="*60)
    print("模型对比")
    print("="*60)
    
    many_info = parse_ncnn_param(many_param)
    e2e_info = parse_ncnn_param(e2e_param)
    
    if many_info and e2e_info:
        print(f"\n对比结果:")
        print(f"  One-to-Many 层数: {many_info['layer_count']}")
        print(f"  E2E 层数: {e2e_info['layer_count']}")
        
        # 统计各类型层数
        many_types = {}
        e2e_types = {}
        
        for l in many_info['layers']:
            many_types[l['type']] = many_types.get(l['type'], 0) + 1
        for l in e2e_info['layers']:
            e2e_types[l['type']] = e2e_types.get(l['type'], 0) + 1
        
        # 找出差异
        all_types = set(many_types.keys()) | set(e2e_types.keys())
        print(f"\n层类型统计差异:")
        for t in sorted(all_types):
            many_count = many_types.get(t, 0)
            e2e_count = e2e_types.get(t, 0)
            if many_count != e2e_count:
                print(f"  {t}: {many_count} -> {e2e_count}")

def check_assets_directory():
    """检查assets目录中的模型"""
    print("\n" + "="*60)
    print("检查Assets目录")
    print("="*60)
    
    # 获取脚本所在目录的父目录（项目根目录）
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    assets_dir = project_root / "app" / "src" / "main" / "assets"
    
    if not assets_dir.exists():
        print(f"  ⚠ 目录不存在: {assets_dir}")
        return
    
    param_files = list(assets_dir.glob("*.ncnn.param"))
    
    if not param_files:
        print("  ⚠ 未找到任何NCNN模型文件")
        return
    
    print(f"\n找到 {len(param_files)} 个模型:")
    
    for param_file in sorted(param_files):
        bin_file = param_file.with_suffix('.bin')
        bin_file = param_file.parent / (param_file.stem.replace('.ncnn', '') + '.ncnn.bin')
        
        param_size = param_file.stat().st_size / 1024
        
        if bin_file.exists():
            bin_size = bin_file.stat().st_size / 1024 / 1024
            status = "✓"
        else:
            bin_size = 0
            status = "⚠ (缺少bin文件)"
        
        # 检测是否是E2E模型
        is_e2e = "_e2e" in param_file.name
        mode = "E2E" if is_e2e else "Normal"
        
        print(f"  {status} {param_file.name} ({param_size:.1f}KB) [{mode}]")

def generate_expected_output_info():
    """生成预期输出格式说明"""
    print("\n" + "="*60)
    print("预期输出格式参考")
    print("="*60)
    
    print("""
┌────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ 任务           │ YOLO26 Legacy   │ YOLO26 Many     │ YOLO26 E2E      │
├────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Detection      │ (8400, 144)     │ (8400, 84)      │ (300, 6)        │
│ Segmentation   │ (8400, 176)     │ (8400, 116)     │ (300, 38)       │
│ Pose           │ (8400, 65)      │ (8400, 5)       │ (300, 57)       │
│ OBB            │ (8400, 79)      │ (8400, 19)      │ (300, 7)        │
└────────────────┴─────────────────┴─────────────────┴─────────────────┘

E2E模式输出详解:
  Detection (300, 6):
    [x_center, y_center, width, height, class_id, confidence]
  
  Segmentation (300, 38):
    [x, y, w, h, class_id, conf, mask_coeffs(32)]
  
  Pose (300, 57):
    [x, y, w, h, class_id, conf, keypoints(17*3)]
  
  OBB (300, 7):
    [x_center, y_center, width, height, angle, class_id, confidence]

C++自动检测逻辑:
  if (out.w == 6 && out.h <= 300)  -> E2E Detection
  if (out.w == 84)                  -> YOLO26 One-to-Many
    if (out.w == 144)                 -> YOLO26 Legacy
""")

def main():
    parser = argparse.ArgumentParser(description="NCNN模型输出格式验证")
    parser.add_argument("--model", default=None, help="NCNN param文件路径")
    parser.add_argument("--compare", nargs=2, help="对比两个模型")
    parser.add_argument("--check-assets", action="store_true", help="检查assets目录")
    args = parser.parse_args()
    
    print("="*60)
    print("YOLO26 NCNN输出格式验证工具")
    print("="*60)
    
    generate_expected_output_info()
    
    if args.model:
        verify_e2e_output_format(args.model)
    
    if args.compare:
        compare_models(args.compare[0], args.compare[1])
    
    if args.check_assets or (not args.model and not args.compare):
        check_assets_directory()
    
    print("\n" + "="*60)
    print("验证完成")
    print("="*60)

if __name__ == "__main__":
    main()
