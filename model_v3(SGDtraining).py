import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """残差块：包含两个卷积层和残差连接"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # 主路径
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        # 残差连接（当维度变化时使用1x1卷积调整）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.main_path(x)
        out += identity  # 残差连接
        out = self.relu(out)
        return out


class ResNetCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetCNN, self).__init__()

        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 224x224 -> 112x112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 112x112 -> 56x56
        )

        # 残差块1: 64 -> 64 (保持通道数)
        self.res_block1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
            nn.MaxPool2d(2)  # 56x56 -> 28x28
        )

        # 残差块2: 64 -> 128 (通道数翻倍，空间尺寸减半)
        self.res_block2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),  # 28x28 -> 14x14
            ResidualBlock(128, 128, stride=1),
            nn.MaxPool2d(2)  # 14x14 -> 7x7
        )

        # 残差块3: 128 -> 256 (通道数翻倍，空间尺寸减半)
        self.res_block3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),  # 7x7 -> 4x4
            ResidualBlock(256, 256, stride=1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 输出尺寸 1x1
            nn.Flatten(),  # 展平为 256 维
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.initial_conv(x)  # 224x224 -> 56x56
        x = self.res_block1(x)  # 56x56 -> 28x28
        x = self.res_block2(x)  # 28x28 -> 7x7
        x = self.res_block3(x)  # 7x7 -> 4x4
        x = self.classifier(x)
        return x


def build_model(num_classes=3):
    """返回模型实例"""
    return ResNetCNN(num_classes=num_classes)


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = build_model(num_classes=3)

    # 创建测试输入 (batch_size=1, 3通道, 224x224)
    test_input = torch.randn(1, 3, 224, 224)

    # 前向传播
    output = model(test_input)

    # 打印结果
    print("模型结构：")
    print(model)
    print(f"\n输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出值: {output}")

    # 验证输出形状
    assert output.shape == (1, 3), f"输出形状应为(1, 3)，但得到{output.shape}"
    print("✅ 测试通过：输出形状正确！")

    import torch.optim as optim
    import torch.nn as nn

    model = build_model(num_classes=3)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义SGD优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,  # 学习率
        momentum=0.9,  # 动量（可选，但推荐）
        weight_decay=1e-4  # 权重衰减（L2正则化，可选）
    )


    # 训练循环
    def train(model, train_loader, criterion, optimizer, num_epochs):
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}')