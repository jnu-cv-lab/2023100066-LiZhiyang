# 2023100066-7
2023100066_李智阳_ML_CV_Assignment
# 传统机器学习方法在手写数字图像分类中的应用

## 1. 项目目的
1. 理解图像如何转换为机器学习模型可以处理的特征向量。
2. 理清训练集和测试集的基本概念。
3. 熟悉常见传统机器学习分类方法的基本使用。
4. 区分不同分类器在同一图像分类任务中的表现差异。
5. 学会使用准确率、混淆矩阵和错误样本分析评价分类结果。

## 2. 运行环境
- 操作系统：Linux / Windows
- Python 3
- 安装依赖库：pip install opencv-python numpy matplotlib

## 3. 主要功能
1. 数据加载与查看，加载sklearn手写数字数据集，展示样本图像。
2. 数据集划分，按75%训练、25%测试比例划分数据集。
3. 特征表示，将8×8图像展平为64维特征向量，理解像素特征特点。
4. 多模型训练，训练 KNN、SVM、逻辑回归、朴素贝叶斯、决策树、随机森林。
5. 模型评估，计算各模型测试准确率，输出对比表格。
6. 结果分析，绘制最优模型混淆矩阵，展示错误分类样本。

## 4. 核心代码与说明

### 4.1  数据准备与可视化
```python
# 加载数据集，提取特征、标签与原始图像
digits = load_digits()
X = digits.data
y = digits.target
images = digits.images
# 绘制并保存前10张样本图像
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(str(y[i]))
    plt.axis('off')
plt.savefig("task1_samples.png")
plt.close()
```

### 4.2 训练集与测试集划分
```python
# 按75%训练、25%测试划分数据，保证实验可复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
```

### 4.3 特征向量说明
```python
# 将8x8原始二维图像展平为64维一维特征向量
# 这是机器学习模型能够接受的输入格式
image = images[0]  # 取第一张图像 (8x8)
feature_vector = image.flatten()  # 展平成特征向量 (64,)
print("原始图像形状：", image.shape)
print("转换后特征向量形状：", feature_vector.shape)
```

### 4.4 多模型定义与训练
```python
# 定义6种经典分类模型
models = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}
# 遍历训练、预测并计算准确率
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)
```

### 4.5 模型准确率对比输出
```python
# 以表格形式输出各模型测试准确率
print("===== 任务5：模型准确率对比表 =====")
print("| 模型名称 | 测试准确率 |")
print("|--------|------------|")
for name, acc in results.items():
    print(f"| {name} | {acc:.4f} |")
```

### 4.6 混淆矩阵与错误样本分析
```python
# 训练SVM并计算混淆矩阵
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
# 绘制并保存混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.savefig("confusion_matrix.png")
plt.close()
# 展示错误分类样本
err_idx = np.where(y_pred != y_test)[0]
plt.figure(figsize=(12, 4))
for i, idx in enumerate(err_idx[:8]):
    plt.subplot(1, 8, i+1)
    plt.imshow(X_test[idx].reshape(8,8), cmap='gray')
    plt.title(f"T:{y_test[idx]}\nP:{y_pred[idx]}")
    plt.axis('off')
plt.savefig("error_samples.png")
plt.close()
```

## 5. 核心参数说明
1. 数据集，sklearn手写数字数据集，共1797张8×8灰度图像。
2. 划分比例，训练集75%，测试集25%。
3. 随机种子，random_state=42，保证结果可复现。
4. 逻辑回归迭代次数，max_iter=10000，确保收敛。
5. 评价指标，测试集分类准确率。
6. 绘图格式，灰度显示，保存为PNG高清图片。

## 6. 运行步骤
1. 安装依赖：pip install opencv-python numpy matplotlib。
2. 分别利用Linux指令touch mission.py创建对应任务的Python文件。
3. 在Ubuntu中激活开发环境source /home/lzy/cv-course/.venv-basic/bin/activate。
4. 利用python3 mission.py进行下方运行文件的步骤。
5. 查看控制台输出与保存的图片结果。
6. 整理准确率表格、混淆矩阵、错误样本图用于报告。

## 7. 结果与分析
1. 数据结构：8×8图像展平为64维向量，可直接输入机器学习模型。
2. 模型表现：KNN与SVM准确率最高，接近99%；朴素贝叶斯与单棵决策树效果较弱。
3. 混淆矩阵：模型主要在外形相似的数字（5/6、7/9、8/9）上产生混淆。
4. 错误样本：错误数量极少，均为手写形态模糊、轮廓接近导致的误判。
5. 特征特点：像素特征简单直接，但对位置、形态变化敏感，缺乏几何不变性。
6. 结论：传统机器学习在小规模手写数字数据集上可实现极高分类精度。

## 8. 作者信息
1. 作者：李智阳
2. 日期：2026年5月8日
