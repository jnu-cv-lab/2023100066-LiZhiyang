import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
img_box = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

# 初始化ORB特征检测器，设置最大特征点数量1000
orb = cv2.ORB_create(nfeatures=1000)
kp_box, des_box = orb.detectAndCompute(img_box, None)
kp_scene, des_scene = orb.detectAndCompute(img_scene, None)

# 暴力匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_box, des_scene)

# 按匹配距离从小到大排序
matches = sorted(matches, key=lambda x: x.distance)

# ===================== 任务3：RANSAC 剔除错误匹配 =====================
# 获取匹配点对的坐标，用于单应矩阵计算
src_pts = np.float32([kp_box[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 基于RANSAC鲁棒算法计算单应矩阵 H
# 能够自动识别并排除外点（错误匹配）
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 将mask展平，用于筛选正确匹配
matchesMask = mask.ravel().tolist()

# 输出RANSAC处理结果
print("===== 任务3 RANSAC结果 =====")
print("总匹配数:", len(matches))                # 初始匹配总数
print("内点数量:", sum(matchesMask))             # 正确匹配数量
print("单应矩阵:\n", H)                         # 输出计算得到的单应矩阵

# 绘制仅保留内点的匹配结果图
img_ransac = cv2.drawMatches(img_box, kp_box, img_scene, kp_scene, 
                             matches, None, matchesMask=matchesMask, flags=2)
cv2.imwrite('orb_ransac_matches.png', img_ransac)

plt.figure(figsize=(16,6))
plt.imshow(img_ransac)
plt.title('RANSAC 去误匹配')
plt.axis('off')
plt.show()

# ---------------------- 任务4：目标定位 ----------------------
# 获取模板图像的四个角点
h, w = img_box.shape
box_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

# 使用单应矩阵将角点投影到场景图像中
scene_corners = cv2.perspectiveTransform(box_corners, H)

# 将灰度图转为彩色图，便于绘制彩色边框
img_scene_color = cv2.cvtColor(img_scene, cv2.COLOR_GRAY2BGR)

# 绘制红色四边形框，标记目标位置
img_scene_with_box = cv2.polylines(img_scene_color, [np.int32(scene_corners)], 
                                   isClosed=True, color=(0, 0, 255), thickness=3)

# 保存并显示定位结果
cv2.imwrite('target_localization.png', img_scene_with_box)

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_scene_with_box, cv2.COLOR_BGR2RGB))
plt.title('目标定位结果')
plt.axis('off')
plt.show()

# 输出定位结果说明
print("\n===== 任务4 目标定位结果 =====")
print("目标物体已成功定位，红色四边形框出了box.png在场景中的位置。")