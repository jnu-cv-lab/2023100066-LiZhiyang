from sklearn.datasets import load_digits

# 加载数据
digits = load_digits()
images = digits.images
X = digits.data

print("===== 任务3：特征表示 =====")
print("原始图像尺寸：8x8 二维矩阵")
print("展平后特征向量：64维一维向量")
print("原图像 shape：", images[0].shape)
print("特征向量 shape：", X[0].shape)
print("\n像素特征优点：简单直接、无需手工设计")
print("像素特征缺点：对位移/旋转敏感、无空间结构信息")