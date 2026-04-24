import cv2
import numpy as np
import matplotlib.pyplot as plt  # 用于显示图片

# ===================== 任务6：ORB 参数 nfeatures 对比实验 =====================

# 读取模板图像和场景图像
img_box = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

# 定义需要对比的 nfeatures 参数：500、1000、2000
nfeatures_list = [500, 1000, 2000]

# 用于存储每组参数的实验结果
results = []

# 遍历每一组参数进行实验
for idx, n in enumerate(nfeatures_list):
    print(f"\n===== 测试 nfeatures = {n} =====")
    
    # 创建ORB检测器，使用当前参数 nfeatures
    orb = cv2.ORB_create(nfeatures=n)
    kp_box, des_box = orb.detectAndCompute(img_box, None)
    kp_scene, des_scene = orb.detectAndCompute(img_scene, None)
    
    # 统计并输出关键点数量
    kp_box_count = len(kp_box)
    kp_scene_count = len(kp_scene)
    print(f"模板图关键点数量: {kp_box_count}")
    print(f"场景图关键点数量: {kp_scene_count}")
    
    # 暴力匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_box, des_scene)
    matches = sorted(matches, key=lambda x: x.distance)
    match_count = len(matches)
    print(f"匹配数量: {match_count}")
    
    # 使用 RANSAC 估计单应矩阵
    src_pts = np.float32([kp_box[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # 统计内点
    inlier_count = int(mask.sum())
    inlier_ratio = inlier_count / match_count if match_count > 0 else 0
    print(f"RANSAC内点数量: {inlier_count}")
    print(f"内点比例: {inlier_ratio:.4f}")
    
    # 目标定位判断
    h, w = img_box.shape
    box_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    scene_corners = cv2.perspectiveTransform(box_corners, H)
    
    located = True
    for p in scene_corners:
        x, y = p[0]
        if x < 0 or x > img_scene.shape[1] or y < 0 or y > img_scene.shape[0]:
            located = False
            break
    print(f"是否成功定位: {'是' if located else '否'}")
    
    # ===================== 输出匹配图 =====================
    # 绘制前50个匹配点
    img_matches = cv2.drawMatches(img_box, kp_box, img_scene, kp_scene, matches[:50], None, flags=2)
    # 保存图片
    cv2.imwrite(f'match_result_n{n}.png', img_matches)
    print(f"已保存匹配图：match_result_n{n}.png")
    
    # 保存结果
    results.append([
        n,
        kp_box_count,
        kp_scene_count,
        match_count,
        inlier_count,
        round(inlier_ratio, 4),
        '是' if located else '否'
    ])

# 打印最终对比结果
print("\n===== 参数对比实验结果 =====")
print("nfeatures | 模板图关键点数 | 场景图关键点数 | 匹配数量 | RANSAC内点数 | 内点比例 | 是否成功定位")
print("-" * 80)
for row in results:
    print(f"{row[0]:<9} | {row[1]:<14} | {row[2]:<14} | {row[3]:<8} | {row[4]:<12} | {row[5]:<8} | {row[6]}")