from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 加载数据
digits = load_digits()
X = digits.data
y = digits.target

# 划分训练集与测试集 75% / 25%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("===== 任务2：数据划分 =====")
print("训练集特征形状：", X_train.shape)
print("测试集特征形状：", X_test.shape)
print("训练集用途：训练模型参数")
print("测试集用途：评估模型泛化能力")