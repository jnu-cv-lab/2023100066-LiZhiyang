# 2023100066-9
2023100066自动化李智阳+实验作业9
# 基于 CNN 的优化器与学习率对比实验

## 1. 项目目的
1. 理解卷积神经网络（CNN）在图像分类任务中的基本结构与训练流程。
2. 掌握PyTorch构建、训练、测试模型的完整流程。
3. 实现 SGD、SGD+Momentum、Adam 三种优化器的对比实验，分析收敛速度与精度差异。
4. 完成不同学习率对Adam优化器的影响分析。
5. 可视化训练，验证损失曲线与准确率曲线，直观观察模型收敛过程。
6. 实现卷积核、特征图、错误样本、混淆矩阵的可视化，理解模型行为。

## 2. 运行环境
- 操作系统：Linux / Windows
- Python 3
- 安装依赖库：pip install opencv-python numpy matplotlib

## 3. 主要功能
1. CNN 模型构建：基于卷积、池化、全连接层构建手写数字分类模型。
2. 数据集加载：自动下载并加载MNIST手写数字数据集。
3. 模型训练：支持指定优化器与学习率，自动记录Loss与Accuracy。
4. 优化器对比：对比SGD、SGD+Momentum、Adam的训练效果与收敛速度。
5. 学习率对比：对比不同学习率对Adam训练稳定性与精度的影响。
6. 模型可视化：输出卷积核、特征图、错误分类样本、混淆矩阵。

## 4. 核心代码与说明

### 4.1 CNN模型定义
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 分类全连接层
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
```

### 4.2  训练函数
```python
def train_full(optimizer, lr, epochs=10):
    model = SimpleCNN().to(device)
    opt = optimizer(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # 训练并记录 loss/acc
    ...
    return model, history, test_acc
```

### 4.3 优化器对比实验
```python
opts = [
    ("SGD", optim.SGD, 0.001),
    ("SGD+Momentum", lambda p, lr: optim.SGD(p, lr=lr, momentum=0.9), 0.001),
    ("Adam", optim.Adam, 0.001)
]
```

### 4.4 学习率对比实验
```python
lrs = [0.1, 0.01, 0.001]
for lr in lrs:
    m, h, ta = train_full(optim.Adam, lr)
```

### 4.5 卷积核与特征图可视化
```python
# 绘制第一层卷积核
k = model.features[0].weight.detach().cpu().numpy()
plt.imshow(k[i,0], cmap="gray")

# 绘制特征图
f = model.features[0](img.unsqueeze(0))
```

### 4.6 错误样本与混淆矩阵
```python
# 收集预测错误的图片
ws.append(x[i].squeeze().cpu())

# 计算并绘制混淆矩阵
cm = np.zeros((10,10), dtype=int)
```

## 5. 核心参数说明
1. 训练轮数：10轮
2. 批次大小：64
3. 优化器：SGD、SGD+Momentum、Adam
4. 学习率：0.1、0.01、0.001
5. 损失函数：交叉熵损失CrossEntropyLoss
6. 数据集：MNIST手写数字
7. 可视化：Loss曲线、Acc曲线、卷积核、特征图、错误样本、混淆矩阵

## 6. 运行步骤
1. 直接运行Python文件，自动下载MNIST数据集。
2. 安装依赖：pip install opencv-python numpy matplotlib。
3. 分别利用Linux指令touch CNN_task10.py创建对应任务的Python文件。
4. 在Ubuntu中激活开发环境，并利用python3 task10.py进行下方运行文件的步骤。
5. 基础训练、三种优化器对比、三种学习率对比、卷积核可视化、特征图可视化、错误样本展示、混淆矩阵绘制。
6. 查看控制台输出的指标表格与弹出的图像结果；记录优化器与学习率对比结论，完成实验分析。

## 7. 结果与分析
1. 优化器对比：Adam收敛最快、精度最高；SGD+Momentum居中；SGD收敛最慢。
2. 学习率影响：学习率0.1过大导致模型不收敛；0.01平稳收敛；0.001精度最高。
3. 卷积核：提取边缘、线条、角点等底层视觉特征。
4. 特征图：对数字笔画、轮廓产生不同程度的响应。
5. 错误样本：主要集中在形态相似数字（如：3/8、4/9、2/7）。
6. 混淆矩阵：对角线数值高，分类整体准确；相似数字易混淆。

## 8. 作者信息
1. 作者：李智阳
2. 日期：2026年5月22日
