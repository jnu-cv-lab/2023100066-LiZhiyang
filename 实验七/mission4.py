# 导入数据集加载、数据划分、评估指标及各类分类器
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 加载手写数字数据集并提取特征与标签
digits = load_digits()
X = digits.data
y = digits.target

# 按照75%训练集、25%测试集比例划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 定义需要训练的多个分类模型，存入字典方便统一调用
models = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

print("===== 任务4：模型训练与预测 =====")
results = {}

# 遍历所有模型，依次完成训练、预测、计算准确率
for name, model in models.items():
    # 使用训练集训练模型
    model.fit(X_train, y_train)
    
    # 在测试集上进行预测
    y_pred = model.predict(X_test)
    
    # 计算并存储当前模型的准确率
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    # 输出结果
    print(f"{name:20s}  准确率：{acc:.4f}")