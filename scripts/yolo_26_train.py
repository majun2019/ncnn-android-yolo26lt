from ultralytics import YOLO
import torch
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    data_yaml = os.path.join(script_dir, "safehat.yaml")
    
    if not os.path.exists(data_yaml):
        print(f"错误: 找不到数据配置文件: {data_yaml}")
        return
    
    print(f"数据配置文件: {data_yaml}")
    
    model = YOLO("yolo26n.pt")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    if device == "cpu":
        print("警告: 使用CPU训练会非常慢，建议使用GPU")
    
    print("\n开始训练...")
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=4,
        workers=0,
        device=device,
        amp=False,
        patience=20,
        save=True,
        project="runs/train",
        name="yolo26n_safehat",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    
    print("\n训练完成!")
    print(f"最佳模型保存在: runs/train/yolo26n_safehat/weights/best.pt")
    
    print("\n开始验证...")
    model.val(workers=0)
    
    print("\n导出NCNN模型...")
    best_model = YOLO("runs/train/yolo26n_safehat/weights/best.pt")
    best_model.export(format="ncnn", imgsz=640)
    
    print("\n全部完成! NCNN模型已导出")

if __name__ == "__main__":
    main()