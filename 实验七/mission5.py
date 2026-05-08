import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 加载手写数字数据集，并按75%训练、25%测试划分数据
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 构建多种机器学习分类模型，用于对比实验
models = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# 遍历所有模型，依次训练并在测试集上评估准确率
results = {}
for name, model in models.items():

    # 使用训练集完成模型拟合
    model.fit(X_train, y_train)

    # 对测试集样本进行预测
    y_pred = model.predict(X_test)
    
    # 计算并保存每个模型的准确率
    results[name] = accuracy_score(y_test, y_pred)

# 以表格形式输出所有模型的准确率对比结果
print("===== 任务5：模型准确率对比表 =====")
print("| 模型名称 | 测试准确率 |")
print("|--------|------------|")
for name, acc in results.items():
    print(f"| {name} | {acc:.4f} |")