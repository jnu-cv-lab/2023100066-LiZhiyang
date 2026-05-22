import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 指定使用CPU
device = torch.device("cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# CNN模型定义
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 分类器：全连接层
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平
        return self.classifier(x)

# 完整训练函数
def train_full(optimizer, lr, epochs=10):
    model = SimpleCNN().to(device)
    opt = optimizer(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()  # 损失函数
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        # 训练模式
        model.train()
        t_loss, t_corr = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()  # 清空梯度
            out = model(x)
            loss = criterion(out, y)
            loss.backward()   # 反向传播
            opt.step()        # 更新参数
            t_loss += loss.item()
            t_corr += (out.argmax(1) == y).sum().item()

        # 计算训练集指标
        train_loss = t_loss / len(train_loader)
        train_acc = t_corr / len(train_dataset)

        # 验证模式
        model.eval()
        v_loss, v_corr = 0.0, 0
        with torch.no_grad():  # 禁止梯度
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_loss += criterion(out, y).item()
                v_corr += (out.argmax(1) == y).sum().item()

        # 计算验证集指标
        val_loss = v_loss / len(test_loader)
        val_acc = v_corr / len(test_dataset)

        # 保存历史
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["train_acc"].append(train_acc)
        hist["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1:2d} | TL:{train_loss:.4f} VL:{val_loss:.4f} TA:{train_acc:.4f} VA:{val_acc:.4f}")

    # 最终测试准确率
    model.eval()
    test_corr = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            test_corr += (model(x).argmax(1) == y.to(device)).sum().item()
    test_acc = test_corr / len(test_dataset)
    return model, hist, test_acc

# 绘制loss和accuracy曲线
def plot_hist(hist, title):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(hist["train_loss"], label="Train Loss")
    plt.plot(hist["val_loss"], label="Val Loss")
    plt.title(f"{title} - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(hist["train_acc"], label="Train Acc")
    plt.plot(hist["val_acc"], label="Val Acc")
    plt.title(f"{title} - Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ===================== 任务1 =====================
print("\n===== Task 1: Base Training =====")
model1, hist1, test_acc1 = train_full(optim.Adam, 0.001)
plot_hist(hist1, "Base Training")

# ===================== 任务2 =====================
print("\n===== Task 2: Optimizer Comparison =====")
opts = [
    ("SGD", optim.SGD, 0.001),
    ("SGD+Momentum", lambda p, lr: optim.SGD(p, lr=lr, momentum=0.9), 0.001),
    ("Adam", optim.Adam, 0.001)
]
res2 = []
for name, opt, lr in opts:
    print(f"\n--- {name} ---")
    m, h, ta = train_full(opt, lr)
    res2.append((name, h, ta))
    plot_hist(h, name)

print("\n===== Task 2 Results =====")
print(f"{'Optimizer':<15}{'Train Loss':<12}{'Val Loss':<12}{'Train Acc':<12}{'Val Acc':<12}{'Test Acc':<12}")
for name, h, ta in res2:
    print(f"{name:<15}{h['train_loss'][-1]:<12.4f}{h['val_loss'][-1]:<12.4f}{h['train_acc'][-1]:<12.4f}{h['val_acc'][-1]:<12.4f}{ta:<12.4f}")

# ===================== 任务3 =====================
print("\n===== Task 3: Learning Rate Comparison =====")
lrs = [0.1, 0.01, 0.001]
res3 = []
for lr in lrs:
    print(f"\n--- LR={lr} ---")
    m, h, ta = train_full(optim.Adam, lr)
    res3.append((lr, h, ta))
    plot_hist(h, f"LR={lr}")

# ===================== 任务4 =====================
print("\n===== Task 4: Conv1 Kernels =====")
def plot_kernels(model):
    k = model.features[0].weight.detach().cpu().numpy()
    plt.figure(figsize=(8,4))
    for i in range(8):
        plt.subplot(2,4,i+1)
        plt.imshow(k[i,0], cmap="gray")
        plt.axis("off")
    plt.suptitle("Conv1 Kernels")
    plt.show()
plot_kernels(model1)

# ===================== 任务5 =====================
print("\n===== Task 5: Conv1 Feature Maps =====")
def plot_fmap(model):
    img, _ = test_dataset[0]
    model.eval()
    with torch.no_grad():
        f = model.features[0](img.unsqueeze(0).to(device))
    f = f[0].cpu().numpy()
    plt.figure(figsize=(8,4))
    for i in range(8):
        plt.subplot(2,4,i+1)
        plt.imshow(f[i], cmap="gray")
        plt.axis("off")
    plt.suptitle("Conv1 Feature Maps")
    plt.show()
plot_fmap(model1)

# ===================== 任务6 =====================
print("\n===== Task 6: Wrong Samples =====")
def plot_wrong(model):
    ws, tl, pl = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).argmax(1)
            for i in range(len(y)):
                if pred[i] != y[i]:
                    ws.append(x[i].squeeze().cpu())
                    tl.append(y[i].item())
                    pl.append(pred[i].item())
                    if len(ws)>=8: break
            if len(ws)>=8: break
    plt.figure(figsize=(8,4))
    for i in range(8):
        plt.subplot(2,4,i+1)
        plt.imshow(ws[i], cmap="gray")
        plt.title(f"T:{tl[i]} P:{pl[i]}")
        plt.axis("off")
    plt.suptitle("Wrong Classified Samples")
    plt.show()
plot_wrong(model1)

# ===================== 任务7 =====================
print("\n===== Task 7: Confusion Matrix =====")
def plot_cm(model):
    yt, yp = [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            p = model(x).argmax(1).cpu()
            yt.extend(y.numpy())
            yp.extend(p.numpy())
    cm = np.zeros((10,10), dtype=int)
    for t,p in zip(yt,yp): cm[t,p] += 1
    plt.figure(figsize=(8,6))
    plt.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            plt.text(j,i,cm[i,j], ha="center", va="center")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
plot_cm(model1)