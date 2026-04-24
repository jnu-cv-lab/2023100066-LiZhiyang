import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== 任务1：ORB 特征点检测 =====================
# ---------------------- 1. 读取图像并转为灰度图 ----------------------
# 读取模板图像 box.png
img_box = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
# 读取场景图像 box_in_scene.png
img_scene = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

# 判断图像是否成功读取
if img_box is None or img_scene is None:
    raise FileNotFoundError("请确保图片在同一目录")

# ---------------------- 2. 创建 ORB 特征检测器 ----------------------
# nfeatures=1000 表示最多检测1000个特征点
orb = cv2.ORB_create(nfeatures=1000)

# ---------------------- 3. 检测关键点，计算描述子 ----------------------
# 对模板图进行特征检测
kp_box, des_box = orb.detectAndCompute(img_box, None)
# 对场景图进行特征检测
kp_scene, des_scene = orb.detectAndCompute(img_scene, None)

# ---------------------- 4. 绘制并保存关键点可视化图 ----------------------
# 在图像上绘制绿色关键点
img_box_kp = cv2.drawKeypoints(img_box, kp_box, None, color=(0, 255, 0))
img_scene_kp = cv2.drawKeypoints(img_scene, kp_scene, None, color=(0, 255, 0))

# 保存可视化结果图片
cv2.imwrite('box_keypoints.png', img_box_kp)
cv2.imwrite('box_in_scene_keypoints.png', img_scene_kp)

# ---------------------- 5. 输出实验结果信息 ----------------------
print("===== 任务1 实验结果 =====")
print(f"box.png 关键点数量: {len(kp_box)}")       # 输出模板图关键点数量
print(f"box_in_scene.png 关键点数量: {len(kp_scene)}")  # 输出场景图关键点数量
print(f"描述子维度: {des_box.shape[1]}")         # 输出 ORB 描述子维度

# ---------------------- 6. 显示结果图像 ----------------------
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(img_box_kp), plt.title('box')
plt.subplot(122), plt.imshow(img_scene_kp), plt.title('scene')
plt.show()