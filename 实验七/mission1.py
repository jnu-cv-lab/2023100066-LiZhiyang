import numpy as np
import matplotlib
matplotlib.use('Agg')  # 适配无图形界面环境
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# 加载数据
digits = load_digits()
X = digits.data
y = digits.target
images = digits.images

print("===== 任务1：数据准备 =====")
print("图像总数：", len(images))
print("单张图像大小：", images[0].shape)
print("所有标签：", np.unique(y))

# 创建画布，设置为2行5列，让样本分成两行显示
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)  # 2行5列布局
    plt.imshow(images[i], cmap='gray')
    plt.title(str(y[i]))
    plt.axis('off')

# 保存图片到当前目录
plt.savefig("task1_samples.png", dpi=300, bbox_inches='tight')
plt.close()
print("样本图像已保存为 task1_samples.png")
