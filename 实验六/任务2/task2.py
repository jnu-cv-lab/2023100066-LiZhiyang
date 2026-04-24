import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== 任务2：ORB 初始暴力匹配 =====================

# 读取模板图像和场景图像
img_box = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

# 创建 ORB 特征检测器，最多检测1000个特征点
orb = cv2.ORB_create(nfeatures=1000)

# 对两幅图像分别检测关键点并计算描述子
kp_box, des_box = orb.detectAndCompute(img_box, None)
kp_scene, des_scene = orb.detectAndCompute(img_scene, None)

# 创建暴力匹配器
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 对两幅图像的描述子进行匹配
matches = bf.match(des_box, des_scene)

# 将匹配结果按照匹配距离从小到大排序
matches = sorted(matches, key=lambda x: x.distance)

# 输出初始匹配的总数量
print("===== 任务2 初始匹配 =====")
print(f"总匹配数量: {len(matches)}")

# 绘制前50个最佳匹配结果
img_matches = cv2.drawMatches(img_box, kp_box, img_scene, kp_scene, matches[:50], None, flags=2)

# 保存匹配结果图片
cv2.imwrite('orb_match_result.png', img_matches)

# 显示匹配结果图
plt.figure(figsize=(16,6))
plt.imshow(img_matches)
plt.title('ORB 初始匹配')
plt.axis('off')
plt.show()