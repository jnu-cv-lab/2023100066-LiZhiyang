# 2023100066-6
2023100066自动化李智阳+实验作业6
# 基于 OpenCV 的局部特征检测、描述与图像匹配

## 1. 项目目的
1. 理解图像特征点、特征描述子的作用，掌握ORB算法的原理与使用流程。
2. 实现基于ORB的关键点检测、描述子生成与暴力匹配，完成模板与场景图像的特征匹配。
3. 使用RANSAC算法剔除错误匹配点，提高匹配鲁棒性。
4. 基于单应矩阵Homography实现目标定位，在场景中框选出目标物体。
5. 对比ORB与SIFT算法在匹配效果、运行速度、稳定性上的差异。
6. 完成ORB参数nfeatures对比实验，分析参数对关键点数量、匹配效果的影响。

## 2. 运行环境
- 操作系统：Linux / Windows
- Python 3
- 安装依赖库：pip install opencv-python numpy matplotlib

## 3. 主要功能
1. ORB 特征检测：提取模板图与场景图的关键点与二进制描述子。
2. 特征匹配：使用汉明距离暴力匹配，完成描述子配对。
3. RANSAC 去误匹配：自动剔除错误匹配，保留几何一致的内点。
4. 目标定位：通过单应矩阵投影模板四角，在场景中框出目标位置。
5. SIFT 与 ORB 对比：对比两种算法的匹配数、内点比例、定位结果、运行时间。
6. ORB 参数实验：测试不同 nfeatures 对关键点、匹配、定位的影响。

## 4. 核心代码与说明

### 4.1 ORB 特征检测与描述子提取
```python
# 创建 ORB 检测器，提取两幅图像的关键点与 256 位二进制描述子
orb = cv2.ORB_create(nfeatures=1000)
kp_box, des_box = orb.detectAndCompute(img_box, None)
kp_scene, des_scene = orb.detectAndCompute(img_scene, None)
```

### 4.2 暴力匹配与排序
```python
# 使用汉明距离匹配描述子，按匹配质量从优到差排序
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_box, des_scene)
matches = sorted(matches, key=lambda x: x.distance)
```

### 4.3 RANSAC+单应矩阵估计
```python
# 用RANSAC鲁棒求解单应矩阵，标记内点，剔除错误匹配
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()
```

### 4.4 目标定位与框选
```python
# 将模板四角投影到场景，绘制定位边框，标出目标位置
scene_corners = cv2.perspectiveTransform(box_corners, H)
img_scene_with_box = cv2.polylines(img_scene_color, [np.int32(scene_corners)], isClosed=True, color=(0,255,0), thickness=3)
```

### 4.5 ORB参数对比实验
```python
# 对比不同最大关键点数量下 ORB 的表现，输出完整实验表格
nfeatures_list = [500, 1000, 2000]
for n in nfeatures_list:
    orb = cv2.ORB_create(nfeatures=n)
    # 检测、匹配、RANSAC、定位并记录结果
```

### 4.6 SIFT特征检测与匹配
```python
sift = cv2.SIFT_create()
kp_sift_box, des_sift_box = sift.detectAndCompute(img_box, None)
kp_sift_scene, des_sift_scene = sift.detectAndCompute(img_scene, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches_knn = bf.knnMatch(des_sift_box, des_sift_scene, k=2)

good_matches = []
for m, n in matches_knn:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
```

## 5. 核心参数说明
1. ORB nfeatures：最大关键点数量，实验值500、1000、2000。
2. 匹配距离：ORB使用汉明距离SIFT使用L2欧氏距离。
3. RANSAC重投影误差阈值:5.0。
4. Lowe比率阈值：0.75用于SIFT筛选优质匹配。
5. 绘制匹配数量：展示前50对最优匹配。
6. 定位框颜色：绿色（SIFT）、红色（ORB）。

## 6. 运行步骤
1. 将 box.png 和 box_in_scene.png 放入程序目录。
2. 安装依赖：pip install opencv-python numpy matplotlib。
3. 分别利用Linux指令touch task.py创建对应任务的Python文件。
4. 在Ubuntu中激活开发环境，并利用python3 task.py进行下方运行文件的步骤。
5. 运行 ORB 特征匹配与目标定位。
6. 运行 ORB 参数对比实验，查看输出表格与匹配图。
7. 运行 SIFT 与 ORB 对比实验，查看匹配效果、速度、内点比例。
8. 查看保存结果：匹配图、定位图、控制台输出。

## 7. 结果与分析
1. 特征点分布：角点、文字边缘、纹理丰富区域特征点更多。
2. 初始匹配：存在大量误匹配，主要来自重复纹理与相似区域。
3. RANSAC：有效剔除误匹配，使匹配点几何一致性大幅提高。
4. 目标定位：单应矩阵可准确将模板投影到场景，完成目标框选。
5. ORB参数影响：nfeatures越大，关键点越多，但内点比例不一定提升。
6. SIFT vs ORB：SIFT匹配质量更高、更稳定；ORB速度更快、更轻量。

## 8. 作者信息
1. 作者：李智阳
2. 日期：2026年4月24日
