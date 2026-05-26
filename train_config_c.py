"""
训练模块 - 配置C
- 使用交叉熵损失和 SGD 优化器（lr=0.01，动量0.9）
- 每个 epoch 记录训练/验证损失和准确率
- 训练结束后自动绘制并保存训练曲线图
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 设置中文字体，解决图表中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import os


def plot_training_history(history, save_dir='checkpoints'):
    """绘制并保存训练损失和准确率曲线"""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # 子图1：损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='训练损失', markersize=4)
    plt.plot(epochs, history['val_loss'], 'r-o', label='验证损失', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练 & 验证 损失曲线 (配置C: lr=0.01+SGD)')
    plt.legend()
    plt.grid(True)

    # 子图2：准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-o', label='训练准确率', markersize=4)
    plt.plot(epochs, history['val_acc'], 'r-o', label='验证准确率', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('训练 & 验证 准确率曲线 (配置C: lr=0.01+SGD)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves_config_c.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"训练曲线图已保存至 {save_path}")


def train(model, train_loader, val_loader, epochs=20, lr=0.01, device='cuda'):
    """
    训练模型（配置C: lr=0.01 + SGD）
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        # 记录
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"[配置C] Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

    plot_training_history(history)
    return history