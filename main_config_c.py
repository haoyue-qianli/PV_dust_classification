"""
主程序入口 - 配置C
- 加载数据 -> 构建模型 -> 训练 -> 评估 -> 保存模型
- 超参数：lr=0.01, SGD+momentum（由 train_config_c.py 定义）
- 模型结构：model_v3
"""

import os
import torch
from dataset import get_dataloaders
from model_v3 import build_model         # ← 使用 model_v3 结构
from train_config_c import train         # ← 用组员写的配置C训练函数
from evaluate import evaluate

def main():
    data_dir = './data'
    batch_size = 32
    epochs = 20
    lr = 0.01
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'config_c'

    print(f"使用设备: {device}")
    print(f"配置C: lr={lr}, optimizer=SGD+momentum, model=model_v3")

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