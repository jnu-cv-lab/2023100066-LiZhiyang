import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# ==================== 任务1：环境准备 ====================
batch_size = 64
lr = 0.001
epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==================== 任务2：加载图像数据集 ====================
# 图像预处理：转张量 + 归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST训练/测试集
full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# 划分训练集与验证集 8:2
train_size = int(0.8 * len(full_train))
val_size = len(full_train) - train_size
train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

# 构建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 显示8张训练样本图片
def show_samples(dataset, num=8):
    fig, axes = plt.subplots(1, num, figsize=(12, 2))
    for i in range(num):
        img, label = dataset[i]
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.show()

show_samples(train_dataset)

# ==================== 任务3：定义CNN模型 ====================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),  # 卷积：1通道→16通道
            nn.ReLU(),                  # 激活函数
            nn.MaxPool2d(2, 2),         # 池化下采样
            nn.Conv2d(16, 32, 3, 1, 1), # 卷积：16通道→32通道
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 全连接分类层
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)     # 卷积提取特征
        x = torch.flatten(x, 1)  # 展平为一维
        x = self.classifier(x)   # 全连接分类
        return x

model = SimpleCNN().to(device)

# ==================== 任务4：训练模型 ====================
criterion = nn.CrossEntropyLoss()  # 分类损失函数
optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化器

# 记录曲线数据
train_loss_list, train_acc_list = [], []
val_loss_list, val_acc_list = [], []

# 单轮训练函数
def train_one_epoch():
    model.train()  # 切换训练模式
    loss_sum, correct, total = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)             # 前向传播
        loss = criterion(out, y)   # 计算损失

        optimizer.zero_grad()      # 清空梯度
        loss.backward()            # 反向传播
        optimizer.step()           # 更新参数

        # 统计损失与准确率
        loss_sum += loss.item()
        _, pred = torch.max(out, 1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return loss_sum / len(train_loader), correct / total

# ==================== 任务5：验证模型 ====================
def val_one_epoch():
    model.eval()  # 切换验证模式
    loss_sum, correct, total = 0, 0, 0

    with torch.no_grad():  # 关闭梯度计算
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            loss_sum += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return loss_sum / len(val_loader), correct / total

# 训练与验证循环
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = val_one_epoch()

    # 保存历史数据用于绘图
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    print(f'Epoch {epoch+1:2d} | '
          f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
          f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

# ==================== 任务6：测试模型 ====================
def test():
    model.eval()  # 切换测试模式
    loss_sum, correct, total = 0, 0, 0

    with torch.no_grad():  # 关闭梯度，节省显存
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)             # 前向传播预测
            loss = criterion(out, y)   # 计算测试损失

            loss_sum += loss.item()
            _, pred = torch.max(out, 1)  # 取最大概率类别
            correct += (pred == y).sum().item()
            total += y.size(0)

    return loss_sum / len(test_loader), correct / total

# 执行测试并输出结果
test_loss, test_acc = test()
print(f'\nTest Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

# 显示8张测试图：真实标签、预测标签
def show_test_preds(num=8):
    model.eval()
    fig, axes = plt.subplots(1, num, figsize=(12, 2))
    it = iter(test_loader)
    x, y = next(it)
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        out = model(x)
        _, pred = torch.max(out, 1)

    for i in range(num):
        img = x[i].cpu().squeeze()  # 转回CPU并降维
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'T:{y[i]} P:{pred[i]}')  # T=真实，P=预测
        axes[i].axis('off')
    plt.show()

show_test_preds()

# ==================== 任务7：绘制训练曲线 ====================
plt.figure(figsize=(10, 4))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='Train Acc')
plt.plot(val_acc_list, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')

plt.tight_layout()
plt.show()