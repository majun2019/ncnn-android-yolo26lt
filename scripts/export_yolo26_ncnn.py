from ultralytics import YOLO
import shutil
import os

def export_yolo26_e2e(model_name="yolo26n", imgsz=640):
    print(f"=" * 60)
    print(f"导出 {model_name} 检测模型 (One-to-One Head / 端到端)")
    print(f"=" * 60)
    
    model = YOLO(f"{model_name}.pt")
    
    model.export(
        format="ncnn",
        imgsz=imgsz,
        half=False,
        end2end=True,
    )
    
    print(f"\n✅ {model_name} 端到端模型导出完成!")
    print(f"输出目录: ./{model_name}_ncnn_model/")
    print(f"输出格式: (300, 6) = [x, y, w, h, class_id, confidence]")
    
    return f"{model_name}_ncnn_model"

def export_yolo26_many(model_name="yolo26n", imgsz=640):
    print(f"=" * 60)
    print(f"导出 {model_name} 检测模型 (One-to-Many Head / 传统格式)")
    print(f"=" * 60)
    
    model = YOLO(f"{model_name}.pt")
    
    model.export(
        format="ncnn",
        imgsz=imgsz,
        half=False,
        end2end=False,
    )
    
    print(f"\n✅ {model_name} 传统格式模型导出完成!")
    print(f"输出格式: (8400, 84)")
    
    return f"{model_name}_ncnn_model"

def export_yolo26_seg_e2e(model_name="yolo26n-seg", imgsz=640):
    print(f"\n{'=' * 60}")
    print(f"导出 {model_name} 分割模型 (端到端)")
    print(f"{'=' * 60}")
    
    model = YOLO(f"{model_name}.pt")
    model.export(format="ncnn", imgsz=imgsz, half=False, end2end=True)
    
    print(f"✅ {model_name} 端到端分割模型导出完成!")
    return f"{model_name.replace('-', '_')}_ncnn_model"

def export_yolo26_pose_e2e(model_name="yolo26n-pose", imgsz=640):
    print(f"\n{'=' * 60}")
    print(f"导出 {model_name} 姿态模型 (端到端)")
    print(f"{'=' * 60}")
    
    model = YOLO(f"{model_name}.pt")
    model.export(format="ncnn", imgsz=imgsz, half=False, end2end=True)
    
    print(f"✅ {model_name} 端到端姿态模型导出完成!")
    return f"{model_name.replace('-', '_')}_ncnn_model"

def export_yolo26_obb_e2e(model_name="yolo26n-obb", imgsz=640):
    print(f"\n{'=' * 60}")
    print(f"导出 {model_name} OBB模型 (端到端)")
    print(f"{'=' * 60}")
    
    model = YOLO(f"{model_name}.pt")
    model.export(format="ncnn", imgsz=imgsz, half=False, end2end=True)
    
    print(f"✅ {model_name} 端到端OBB模型导出完成!")
    return f"{model_name.replace('-', '_')}_ncnn_model"

def export_yolo26_classification(model_name="yolo26n-cls", imgsz=224):
    print(f"\n{'=' * 60}")
    print(f"导出 {model_name} 分类模型")
    print(f"{'=' * 60}")
    
    model = YOLO(f"{model_name}.pt")
    model.export(format="ncnn", imgsz=imgsz, half=False)
    
    print(f"✅ {model_name} 分类模型导出完成!")
    return f"{model_name.replace('-', '_')}_ncnn_model"

def copy_to_assets(src_dir, asset_name, assets_path):
    if not os.path.exists(src_dir):
        print(f"⚠️ 源目录不存在: {src_dir}")
        return False
    
    os.makedirs(assets_path, exist_ok=True)
    
    param_src = os.path.join(src_dir, "model.ncnn.param")
    bin_src = os.path.join(src_dir, "model.ncnn.bin")
    
    param_dst = os.path.join(assets_path, f"{asset_name}.ncnn.param")
    bin_dst = os.path.join(assets_path, f"{asset_name}.ncnn.bin")
    
    if os.path.exists(param_src) and os.path.exists(bin_src):
        shutil.copy2(param_src, param_dst)
        shutil.copy2(bin_src, bin_dst)
        print(f"✅ 已复制到: {assets_path}")
        print(f"   - {asset_name}.ncnn.param")
        print(f"   - {asset_name}.ncnn.bin")
        return True
    else:
        print(f"⚠️ 找不到模型文件")
        return False

def main():
    
    print("\n" + "=" * 60)
    print("  YOLO26 NCNN 模型导出工具")
    print("  方案二：One-to-One Head (端到端，无需NMS)")
    print("=" * 60)
    
    model_size = "n"
    assets_path = "../app/src/main/assets"
    
    print("\n" + "=" * 60)
    print("  导出端到端模型 (One-to-One Head)")
    print("=" * 60)
    
    det_e2e_dir = export_yolo26_e2e(f"yolo26{model_size}")
    
    seg_e2e_dir = export_yolo26_seg_e2e(f"yolo26{model_size}-seg")
    
    pose_e2e_dir = export_yolo26_pose_e2e(f"yolo26{model_size}-pose")
    
    cls_dir = export_yolo26_classification(f"yolo26{model_size}-cls")
    
    obb_e2e_dir = export_yolo26_obb_e2e(f"yolo26{model_size}-obb")
    
    print("\n" + "=" * 60)
    print("复制端到端模型到Android assets目录...")
    print("=" * 60)
    
    copy_to_assets(det_e2e_dir, f"yolo26{model_size}_e2e", assets_path)
    copy_to_assets(seg_e2e_dir, f"yolo26{model_size}_seg_e2e", assets_path)
    copy_to_assets(pose_e2e_dir, f"yolo26{model_size}_pose_e2e", assets_path)
    copy_to_assets(cls_dir, f"yolo26{model_size}_cls", assets_path)
    copy_to_assets(obb_e2e_dir, f"yolo26{model_size}_obb_e2e", assets_path)
    
    print("\n" + "=" * 60)
    print("🎉 所有端到端模型导出完成!")
    print("=" * 60)
    print("\n模型命名说明:")
    print("  - yolo26n_e2e.ncnn.*     : 检测模型（端到端）")
    print("  - yolo26n_seg_e2e.ncnn.* : 分割模型（端到端）")
    print("  - yolo26n_pose_e2e.ncnn.*: 姿态模型（端到端）")
    print("  - yolo26n_obb_e2e.ncnn.* : OBB模型（端到端）")
    print("  - yolo26n_cls.ncnn.*     : 分类模型")
    print("\n下一步:")
    print("1. 编译Android项目")
    print("2. 选择带 'E2E' 标记的模型进行测试")
    print("3. 对比端到端模式与传统模式的速度差异")

if __name__ == "__main__":
    main()
