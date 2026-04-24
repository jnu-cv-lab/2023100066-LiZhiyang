import cv2
import numpy as np
import time

# 读取图像
img_box = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

# ----------------------  SIFT 特征匹配与目标检测 ----------------------
print("===== SIFT 特征匹配 =====")
start_time = time.time()

# 初始化SIFT特征提取器
sift = cv2.SIFT_create()
# 提取关键点与浮点型描述子
kp_sift_box, des_sift_box = sift.detectAndCompute(img_box, None)
kp_sift_scene, des_sift_scene = sift.detectAndCompute(img_scene, None)

print(f"SIFT 模板图关键点数量: {len(kp_sift_box)}")
print(f"SIFT 场景图关键点数量: {len(kp_sift_scene)}")

# 构建暴力匹配器，SIFT描述子使用L2范数度量距离
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
# KNN匹配，每个描述子返回前2个最优匹配
matches_knn = bf.knnMatch(des_sift_box, des_sift_scene, k=2)

# Lowe比率测试，剔除模糊匹配与错误匹配
good_matches = []
for m, n in matches_knn:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"SIFT 筛选后匹配数量: {len(good_matches)}")

# 匹配点数量满足最低要求时，使用RANSAC求解单应矩阵
if len(good_matches) >= 4:
    # 提取匹配对的坐标
    src_pts = np.float32([kp_sift_box[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp_sift_scene[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    # 结合RANSAC鲁棒估计单应性矩阵，剔除外点
    H_sift, mask_sift = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inlier_count_sift = int(mask_sift.sum())
    inlier_ratio_sift = inlier_count_sift / len(good_matches)
    
    # 获取模板图像四角坐标，通过单应矩阵透视变换映射到场景图
    h, w = img_box.shape
    box_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    scene_corners = cv2.perspectiveTransform(box_corners, H_sift)
    # 简单边界判断，验证目标定位是否有效
    located_sift = True
    for p in scene_corners:
        x, y = p[0]
        if x < 0 or x > img_scene.shape[1] or y < 0 or y > img_scene.shape[0]:
            located_sift = False
            break
else:
    inlier_count_sift = 0
    inlier_ratio_sift = 0
    located_sift = False

end_time = time.time()
time_sift = end_time - start_time
print(f"SIFT RANSAC内点数量: {inlier_count_sift}")
print(f"SIFT 内点比例: {inlier_ratio_sift:.4f}")
print(f"SIFT 是否成功定位: {'是' if located_sift else '否'}")
print(f"SIFT 运行时间: {time_sift:.4f} 秒")

# 绘制前50组优质匹配对并保存
img_sift_matches = cv2.drawMatches(img_box, kp_sift_box, img_scene, kp_sift_scene, good_matches[:50], None, flags=2)
cv2.imwrite('sift_matches.png', img_sift_matches)

# ----------------------  ORB 特征匹配（对比实验） ----------------------
print("\n===== ORB 特征匹配 =====")
start_time = time.time()

# 初始化ORB特征检测器，限制最大特征点数量
orb = cv2.ORB_create(nfeatures=1000)
kp_orb_box, des_orb_box = orb.detectAndCompute(img_box, None)
kp_orb_scene, des_orb_scene = orb.detectAndCompute(img_scene, None)

print(f"ORB 模板图关键点数量: {len(kp_orb_box)}")
print(f"ORB 场景图关键点数量: {len(kp_orb_scene)}")

# ORB二进制描述子使用汉明距离，开启交叉匹配提升匹配精度
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf_orb.match(des_orb_box, des_orb_scene)
# 按匹配距离升序排序，优先保留高质量匹配
matches_orb = sorted(matches_orb, key=lambda x: x.distance)
match_count_orb = len(matches_orb)

print(f"ORB 匹配数量: {match_count_orb}")

# 结合RANSAC计算ORB匹配结果的单应矩阵
src_pts_orb = np.float32([kp_orb_box[m.queryIdx].pt for m in matches_orb]).reshape(-1,1,2)
dst_pts_orb = np.float32([kp_orb_scene[m.trainIdx].pt for m in matches_orb]).reshape(-1,1,2)
H_orb, mask_orb = cv2.findHomography(src_pts_orb, dst_pts_orb, cv2.RANSAC, 5.0)
inlier_count_orb = int(mask_orb.sum())
inlier_ratio_orb = inlier_count_orb / match_count_orb

# 利用单应矩阵完成目标四角投影映射
scene_corners_orb = cv2.perspectiveTransform(box_corners, H_orb)
located_orb = True
for p in scene_corners_orb:
    x, y = p[0]
    if x < 0 or x > img_scene.shape[1] or y < 0 or y > img_scene.shape[0]:
        located_orb = False
        break

end_time = time.time()
time_orb = end_time - start_time
print(f"ORB RANSAC内点数量: {inlier_count_orb}")
print(f"ORB 内点比例: {inlier_ratio_orb:.4f}")
print(f"ORB 是否成功定位: {'是' if located_orb else '否'}")
print(f"ORB 运行时间: {time_orb:.4f} 秒")

# 绘制ORB匹配结果
img_orb_matches = cv2.drawMatches(img_box, kp_orb_box, img_scene, kp_orb_scene, matches_orb[:50], None, flags=2)
cv2.imwrite('orb_matches.png', img_orb_matches)
# ===================== SIFT 目标定位可视化输出 =====================
# 灰度图转为彩色画布，用于绘制定位边框
img_scene_color = cv2.cvtColor(img_scene, cv2.COLOR_GRAY2BGR)
# 绘制绿色闭合边框，标记SIFT算法定位目标区域
img_sift_local = cv2.polylines(img_scene_color, [np.int32(scene_corners)], 
                               isClosed=True, color=(0, 255, 0), thickness=3)

cv2.imwrite('sift_localization.png', img_sift_local)

print("\n===== SIFT 目标定位已完成 =====")
print("已生成图片：sift_localization.png")

# ----------------------  两种特征算法指标对比输出 ----------------------
print("\n===== SIFT vs ORB 对比结果 =====")
print("方法 | 匹配数量 | RANSAC内点数 | 内点比例 | 是否成功定位 | 运行速度主观评价")
print("----|----------|-------------|----------|--------------|----------------")
print(f"ORB | {match_count_orb:<8} | {inlier_count_orb:<11} | {inlier_ratio_orb:.4f} | {'是' if located_orb else '否':<10} | 快")
print(f"SIFT | {len(good_matches):<8} | {inlier_count_sift:<11} | {inlier_ratio_sift:.4f} | {'是' if located_sift else '否':<10} | 慢")

