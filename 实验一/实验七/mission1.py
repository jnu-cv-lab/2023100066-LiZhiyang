# 导入数值计算、绘图、数据集工具
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# 加载手写数字数据集
digits = load_digits()

# 提取特征、标签、原始图像
X = digits.data
y = digits.target
images = digits.images

print("===== 任务1：数据准备 =====")
print("图像总数：", len(images))
print("单张图像大小：", images[0].shape)
print("所有标签：", np.unique(y))

# 创建画布
plt.figure(figsize=(10, 4))

# 绘制前10张样本图像
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(str(y[i]))
    plt.axis('off')

# 保存图像并关闭画布
plt.savefig("task1_samples.png", dpi=300, bbox_inches='tight')
plt.close()

print("样本图像已保存为 task1_samples.png")