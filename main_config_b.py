"""
主程序入口 - 配置B
- 加载数据 -> 构建模型 -> 训练 -> 评估 -> 保存模型
- 超参数：lr=0.0001, Adam（由 train_config_b.py 定义）
"""

import os
import torch
from dataset import get_dataloaders
from model import build_model
from train_config_b import train       # ← 用组员写的配置B训练函数
from evaluate import evaluate

def main():
    data_dir = './data'
    batch_size = 32
    epochs = 20
    lr = 0.0001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'config_b'

    print(f"使用设备: {device}")
    print(f"配置B: lr={lr}, optimizer=Adam")

    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size)
    model = build_model(num_classes=3)

    history = train(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)

    metrics = evaluate(model, test_loader, device=device, model_name=model_name)

    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}.pth'))
    print(f"模型已保存至 checkpoints/{model_name}.pth")

if __name__ == '__main__':
    main()