"""
YOLO26 自定义数据集训练脚本

训练安全帽检测模型 (10类)
- Hardhat, Mask, No-Hardhat, No-Mask, No-Safety Vest
- Person, Safety Cone, Safety Vest, Machinery, Vehicle

使用方法：
    python yolo_26_train.py
"""

from ultralytics import YOLO
import torch
import os


def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 数据集配置文件路径
    data_yaml = os.path.join(script_dir, "safehat.yaml")
    
    # 检查配置文件是否存在
    if not os.path.exists(data_yaml):
        print(f"错误: 找不到数据配置文件: {data_yaml}")
        return
    
    print(f"数据配置文件: {data_yaml}")
    
    # 加载预训练模型
    # 方式1: 直接加载预训练权重 (推荐，会自动下载)
    model = YOLO("yolo26n.pt")
    
    # 方式2: 从yaml配置构建模型后加载权重 (需要有yolo26.yaml)
    # model = YOLO("yolo26.yaml").load("yolo26n.pt")
    
    # 检测设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    if device == "cpu":
        print("警告: 使用CPU训练会非常慢，建议使用GPU")
    
    # 开始训练
    print("\n开始训练...")
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=4,           # 根据显存调整，GPU 8GB可用batch=8
        workers=0,         # Windows下设为0避免多进程问题
        device=device,
        amp=False,         # 混合精度训练，GPU支持时可设为True加速
        patience=20,       # 早停耐心值
        save=True,         # 保存检查点
        project="runs/train",  # 保存目录
        name="yolo26n_safehat",  # 实验名称
        exist_ok=True,     # 覆盖已有实验
        plots=True,        # 生成训练曲线
        verbose=True,
    )
    
    print("\n训练完成!")
    print(f"最佳模型保存在: runs/train/yolo26n_safehat/weights/best.pt")
    
    # 验证模型
    print("\n开始验证...")
    model.val(workers=0)
    
    # 导出为NCNN格式 (用于Android部署)
    print("\n导出NCNN模型...")
    best_model = YOLO("runs/train/yolo26n_safehat/weights/best.pt")
    best_model.export(format="ncnn", imgsz=640)
    
    print("\n全部完成! NCNN模型已导出")


if __name__ == "__main__":
    # Windows 上 DataLoader 多进程的安全入口
    main()