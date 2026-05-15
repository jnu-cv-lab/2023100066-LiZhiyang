# 2023100066-8
2023100066自动化李智阳+实验作业8
# PyTorch 入门与图像分类

## 1. 项目目的
1. 理解卷积神经网络（CNN）在图像分类任务中的基本结构与工作原理。
2. 完成 MNIST 手写数字数据集的加载、预处理、划分与可视化。
3. 搭建基础 CNN 模型，实现训练、验证、测试全流程，并绘制损失与准确率曲线。
4. 通过可视化展示训练样本与测试集预测结果，直观验证模型效果。

## 2. 运行环境
- 操作系统：Linux / Windows
- Python 3
- 安装依赖库：pip install torch torchvision matplotlib numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

## 3. 主要功能
1. 数据集处理MNIST数据集自动下载、归一化、划分训练/验证/测试集。
2. 数据可视化：显示指定数量的训练集样本与测试集预测结果。
3. 基础CNN训练：搭建卷积+池化+全连接网络，完成10轮epoch训练与指标记录。
4. 模型评估：在独立测试集上计算损失与准确率，输出最终结果。
5. 曲线绘制：绘制训练 / 验证损失曲线与准确率曲线，分析收敛状态。
6. 在卷积层与全连接层后加入Dropout，有效降低过拟合。
7. 使用同一模型分别以Adam、SGD训练，对比性能差异。
8. 在MNIST与CIFAR-10上训练，分析图像复杂度对分类任务的影响。

## 4. 核心代码与说明

### 4.1 数据集加载与预处理
```python
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
```

### 4.2 基础 CNN 模型定义
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
```

### 4.3 带 Dropout 的防过拟合网络
```python
class CNN_Dropout(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),  # 随机失活，抑制过拟合
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
```

### 4.4 优化器对比（Adam / SGD）
```python
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
optimizer_sgd = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```

### 4.5 CIFAR-10 数据集训练
```python
# 三通道彩色图像，更复杂，分类难度更高
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
```

### 4.6 训练/验证/测试流程
```python
model.train()       # 训练模式
model.eval()        # 评估模式
with torch.no_grad():  # 测试阶段关闭梯度
```

## 5. 核心参数说明
1. 训练轮数（epochs）：10
2. 批次大小（batch_size）：64
3. 学习率（lr）：0.001
4. 优化器：Adam（默认）、SGD（进阶对比）
5. Dropout概率：卷积层 0.2，全连接层 0.5
6. 损失函数：交叉熵损失（CrossEntropyLoss）
7. 数据集：MNIST（手写数字）、CIFAR-10（彩色物体）


## 6. 运行步骤
1. 配置Python环境，安装PyTorch、Matplotlib等依赖库。
2. 在Ubuntu中输入Linux指令mkdir 实验八，创建本次实验目录。
3. 利用touch指令创建基础任务和进阶任务的Python文件，并利用Python3指令运行文件
4. 程序会自动下载MNIST、CIFAR-10数据集。
5. 依次弹出训练样本图、测试预测图、损失/准确率对比曲线图。
6. 查看控制台输出，观察每轮训练指标与最终测试准确率。
7. 基础任务：观察10轮epoch下CNN对MNIST的分类效果。
8. 进阶任务：查看Dropout、优化器、数据集带来的性能变化。
## 7. 结果与分析
1. 基础任务：CNN 在10轮epoch后，MNIST测试集准确率可达98%~99%，模型收敛正常。
2. 可视化：训练样本清晰可辨，测试集预测标签与真实标签基本完全一致。
3. 曲线趋势：训练损失持续下降，准确率稳步上升，验证集指标平稳，无剧烈波动。
4. 进阶任务1：加入Dropout后，训练与验证准确率差距缩小，过拟合明显减轻。
5. 进阶任务2：Adam收敛更快、准确率更高；SGD收敛较慢，稳定性稍弱。
6. 进阶任务3：MNIST简单易分类；CIFAR-10复杂，相同模型准确率仅65%~70%。
7. 整体结论：CNN适合图像特征提取；Dropout、优化器、数据集复杂度均显著影响分类效果。

## 8. 作者信息
1. 作者：李智阳
2. 日期：2026年5月15日
