import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# ==================== 任务1：环境准备 ====================
batch_size = 64        # 批次大小
lr = 0.001             # 学习率
epochs = 10            # 训练轮数
# 自动选择GPU/CPU设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==================== 进阶3：分别定义MNIST、CIFAR-10图像预处理 ====================
# MNIST灰度单通道图预处理：转张量 + 标准化
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# CIFAR-10彩色3通道图预处理：转张量 + 标准化
transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# ==================== 进阶1：定义加入Dropout的CNN网络（防止过拟合） ====================
class CNN_Dropout(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # 卷积+池化+Dropout特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1),  # 卷积层：输入通道→16通道
            nn.ReLU(),                            # 激活函数，引入非线性
            nn.MaxPool2d(2, 2),                   # 最大池化，下采样
            nn.Dropout(0.2),                      # Dropout：随机丢弃20%神经元，抑制过拟合
            nn.Conv2d(16, 32, 3, 1, 1),           # 第二层卷积：16通道→32通道
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )
        # 适配MNIST(1通道)、CIFAR-10(3通道)的全连接输入维度
        fc_in = 32 * 7 * 7 if in_channels == 1 else 32 * 8 * 8
        # 全连接分类层，加入Dropout进一步防过拟合
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, 128),
            nn.ReLU(),
            nn.Dropout(0.5),          # 全连接层随机丢弃50%神经元
            nn.Linear(128, 10)        # 输出10个类别概率
        )

    def forward(self, x):
        x = self.features(x)     # 卷积提取特征
        x = torch.flatten(x, 1)   # 特征展平为一维向量
        x = self.classifier(x)    # 全连接分类输出
        return x

# ==================== 通用训练+验证函数（进阶2优化器对比专用） ====================
def train_val(model, train_loader, val_loader, optimizer, epochs):
    criterion = nn.CrossEntropyLoss()  # 分类任务损失函数
    # 保存训练、验证的损失和准确率，用于绘图
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []

    for epoch in range(epochs):
        # ---------- 训练阶段 ----------
        model.train()  # 切换为训练模式，启用Dropout
        loss_sum, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)  # 数据迁移到对应设备
            out = model(x)                     # 前向传播
            loss = criterion(out, y)            # 计算损失

            optimizer.zero_grad()               # 清空梯度
            loss.backward()                     # 反向传播计算梯度
            optimizer.step()                    # 更新模型参数

            # 累加损失、统计正确预测数
            loss_sum += loss.item()
            _, pred = torch.max(out, 1)         # 取概率最大的类别为预测值
            correct += (pred == y).sum().item()
            total += y.size(0)
        # 计算本轮训练平均损失与准确率
        train_loss = loss_sum / len(train_loader)
        train_acc = correct / total

        # ---------- 验证阶段 ----------
        model.eval()  # 切换为验证模式，关闭Dropout
        loss_sum, correct, total = 0, 0, 0
        with torch.no_grad():  # 关闭梯度计算，节省显存、加速推理
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                loss_sum += loss.item()
                _, pred = torch.max(out, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_loss = loss_sum / len(val_loader)
        val_acc = correct / total

        # 保存数据
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        print(f'Epoch {epoch+1:2d} | Train Loss:{train_loss:.4f} Acc:{train_acc:.4f} | Val Loss:{val_loss:.4f} Acc:{val_acc:.4f}')
    return train_loss_list, train_acc_list, val_loss_list, val_acc_list

# ==================== 通用测试函数，评估模型泛化能力 ====================
def test_model(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss_sum / len(test_loader), correct / total

# ==================== 进阶2：对比Adam、SGD两种优化器（MNIST数据集） ====================
print("===== 【进阶2】Adam优化器训练MNIST =====")
# 加载MNIST数据集，划分训练集、验证集
full_train_mnist = datasets.MNIST('./data', train=True, download=True, transform=transform_mnist)
test_mnist = datasets.MNIST('./data', train=False, download=True, transform=transform_mnist)
train_size = int(0.8 * len(full_train_mnist))
val_size = len(full_train_mnist) - train_size
train_mnist, val_mnist = random_split(full_train_mnist, [train_size, val_size])
# 构建数据加载器
train_loader_mnist = DataLoader(train_mnist, batch_size=batch_size, shuffle=True)
val_loader_mnist = DataLoader(val_mnist, batch_size=batch_size, shuffle=False)
test_loader_mnist = DataLoader(test_mnist, batch_size=batch_size, shuffle=False)

# Adam优化器训练
model_adam = CNN_Dropout(in_channels=1).to(device)
opt_adam = optim.Adam(model_adam.parameters(), lr=lr)
adam_loss, adam_acc, adam_val_loss, adam_val_acc = train_val(model_adam, train_loader_mnist, val_loader_mnist, opt_adam, epochs)
test_loss_adam, test_acc_adam = test_model(model_adam, test_loader_mnist)
print(f"Adam测试结果：Loss:{test_loss_adam:.4f} Acc:{test_acc_adam:.4f}\n")

# SGD优化器训练（带动量）
print("===== 【进阶2】SGD优化器训练MNIST =====")
model_sgd = CNN_Dropout(in_channels=1).to(device)
opt_sgd = optim.SGD(model_sgd.parameters(), lr=lr, momentum=0.9)
sgd_loss, sgd_acc, sgd_val_loss, sgd_val_acc = train_val(model_sgd, train_loader_mnist, val_loader_mnist, opt_sgd, epochs)
test_loss_sgd, test_acc_sgd = test_model(model_sgd, test_loader_mnist)
print(f"SGD测试结果：Loss:{test_loss_sgd:.4f} Acc:{test_acc_sgd:.4f}\n")

# ==================== 进阶3：CIFAR-10彩色数据集训练，与MNIST对比 ====================
print("===== 【进阶3】CIFAR-10数据集训练 =====")
# 加载CIFAR-10数据集，划分训练集、验证集
full_train_cifar = datasets.CIFAR10('./data', train=True, download=True, transform=transform_cifar)
test_cifar = datasets.CIFAR10('./data', train=False, download=True, transform=transform_cifar)
train_size = int(0.8 * len(full_train_cifar))
val_size = len(full_train_cifar) - train_size
train_cifar, val_cifar = random_split(full_train_cifar, [train_size, val_size])
# 构建数据加载器
train_loader_cifar = DataLoader(train_cifar, batch_size=batch_size, shuffle=True)
val_loader_cifar = DataLoader(val_cifar, batch_size=batch_size, shuffle=False)
test_loader_cifar = DataLoader(test_cifar, batch_size=batch_size, shuffle=False)

# 训练CIFAR-10模型
model_cifar = CNN_Dropout(in_channels=3).to(device)
opt_cifar = optim.Adam(model_cifar.parameters(), lr=lr)
cifar_loss, cifar_acc, cifar_val_loss, cifar_val_acc = train_val(model_cifar, train_loader_cifar, val_loader_cifar, opt_cifar, epochs)
test_loss_cifar, test_acc_cifar = test_model(model_cifar, test_loader_cifar)
print(f"CIFAR-10测试结果：Loss:{test_loss_cifar:.4f} Acc:{test_acc_cifar:.4f}\n")

# ==================== 绘制多组训练曲线对比图 ====================
plt.figure(figsize=(14, 5))
# 损失曲线对比
plt.subplot(1, 2, 1)
plt.plot(adam_loss, label='Adam-Train Loss')
plt.plot(adam_val_loss, label='Adam-Val Loss')
plt.plot(sgd_loss, label='SGD-Train Loss')
plt.plot(sgd_val_loss, label='SGD-Val Loss')
plt.plot(cifar_loss, label='CIFAR-10-Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Comparison')

# 准确率曲线对比
plt.subplot(1, 2, 2)
plt.plot(adam_acc, label='Adam-Train Acc')
plt.plot(adam_val_acc, label='Adam-Val Acc')
plt.plot(sgd_acc, label='SGD-Train Acc')
plt.plot(sgd_val_acc, label='SGD-Val Acc')
plt.plot(cifar_acc, label='CIFAR-10-Train Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Comparison')
plt.tight_layout()
plt.show()