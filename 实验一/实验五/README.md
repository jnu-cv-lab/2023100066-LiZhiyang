# 2023100066-5
2023100066自动化李智阳+实验作业5
# 图像几何变换与透视校正

## 1. 项目目的
1. 理解并实现相似变换、仿射变换、透视变换三种基本几何变换。
2. 掌握基于OpenCV的图像旋转、缩放、倾斜、投影校正方法。
3. 实现文档图像自动角点检测与透视畸变校正，完成倾斜图像的摆正。
4. 对比不同变换的效果，理解其几何性质与适用场景。

## 2. 运行环境
- 操作系统：Linux / Windows
- Python 3
- 安装依赖库：pip install opencv-python numpy matplotlib

## 3. 主要功能
1. 基础几何变换：实现图像相似变换、仿射变换、透视变换，并对比显示效果。
2. 文档透视校正：自动检测文档轮廓与四角点，支持手动选点备选，完成畸变文档校正。
3. 结果可视化：自动保存变换结果，生成对比图，直观展示变换前后差异。
4. 交互：自动检测失败时切换手动鼠标选点，保证校正流程可完成。

## 4. 核心代码与说明

### 4.1 相似变换
```python
def similarity_transform(img):
    # 获取图像高度h和宽度w
    h, w = img.shape[:2]
    # 以图像中心作为旋转中心
    center = (w // 2, h // 2)
    # 生成变换矩阵：中心旋转22度，整体缩放0.85倍
    M = cv2.getRotationMatrix2D(center, 22, 0.85)
    # 执行仿射变换，空白区域填充白色
    return cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))
```

### 4.2 仿射变换
```python
def affine_transform(img):
    h, w = img.shape[:2]
    # 原始图像上的3个关键点
    src_pts = np.float32([[50,50],[w-50,50],[50,h-50]])
    # 变换后对应的目标点
    dst_pts = np.float32([[30,120],[w-80,60],[100,h-80]])
    # 根据3组点计算仿射变换矩阵
    M = cv2.getAffineTransform(src_pts, dst_pts)
    # 执行仿射变换
    return cv2.warpAffine(img, M, (w,h), borderValue=(255,255,255))
```

### 4.3 透视变换
```python
def perspective_transform(img):
    h, w = img.shape[:2]
    # 原图四个角坐标：左上、右上、左下、右下
    src_pts = np.float32([[0,0],[w-1,0],[0,h-1],[w-1,h-1]])
    # 目标投影位置（产生畸变效果）
    dst_pts = np.float32([[100,80],[w-200,150],[50,h-50],[w-50,h-20]])
    # 计算3x3透视变换矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # 执行透视变换
    return cv2.warpPerspective(img, M, (w,h), borderValue=(255,255,255))
```

### 4.4 文档角点自动检测
```python
def find_document_corners(image):
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    # Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    # 提取最外层轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找到面积最大的轮廓（对应文档）
    max_contour = max(contours, key=cv2.contourArea)
    # 多边形逼近，近似出四边形
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    # 对角点排序并返回
    return arrange_corners(approx.reshape(4,2))
```
### 4.5 透视校正
```python
def correct_perspective(image, corners):
    # 设置输出文档宽度为800像素
    width = 800
    # 按A4比例设置高度
    height = int(width * 1.414)
    # 目标标准矩形的四个角点
    dst = np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    # 计算从文档四角到标准矩形的变换矩阵
    M = cv2.getPerspectiveTransform(corners, dst)
    # 执行透视校正，输出正视图
    return cv2.warpPerspective(image, M, (width, height))
```

## 5. 核心参数说明
1. 相似变换：旋转角度 22°，缩放系数 0.85，以图像中心为变换原点。
2. 仿射变换：3 组对应点，实现倾斜、拉伸效果。
3. 透视变换：4 组角点，模拟近大远小投影效果。
4. 角点检测：Canny 阈值 50/150，轮廓逼近精度 0.02 倍周长。
5. 校正输出：宽度 800，高度按 A4 比例 1.414 计算。

## 6. 运行步骤
1. 将测试图、畸变图放入实验目录。
2. 创建Python文件：touch homework5.py和touch jiaozheng.py。
3. 激活虚拟环境：source /home/lzy/cv-course/.venv-basic/bin/activate。
4. 运行基础几何变换：python3 homework5.py。
5. 运行文档透视校正：python3 jiaozheng.py。
6. 查看输出

## 7. 结果与分析
1. 相似变换：图像旋转缩小，形状、角度无畸变。
2. 仿射变换：图像产生倾斜拉伸，平行线仍保持平行。
3. 透视变换：图像呈现投影畸变，平行线不再平行。
4. 透视校正：倾斜文档自动摆正为标准矩形，文字与边缘基本笔直。

## 8. 作者信息
1. 作者：李智阳
2. 日期：2026年4月17日

