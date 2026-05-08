import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用无界面后端，适配服务器环境
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC

# 设置英文字体，避免绘图乱码
plt.rcParams['font.family'] = 'DejaVu Sans'

# 加载数据集并划分训练集与测试集
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 初始化并训练SVM分类器
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("===== 任务6：错误样本分析 =====")

# 计算混淆矩阵并保存图像
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("SVM Confusion Matrix")
plt.savefig("task6_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# 找出所有预测错误的样本索引
err_idx = np.where(y_pred != y_test)[0]
print("错误样本数量：", len(err_idx))

# 绘制前8个错误样本并保存图像
plt.figure(figsize=(12, 4))
for i, idx in enumerate(err_idx[:8]):
    plt.subplot(1, 8, i+1)
    plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
    plt.title(f"T:{y_test[idx]}\nP:{y_pred[idx]}")
    plt.axis('off')
plt.savefig("task6_error_samples.png", dpi=300, bbox_inches='tight')
plt.close()

print("混淆矩阵已保存为task6_confusion_matrix.png")
print("错误样本已保存为task6_error_samples.png")