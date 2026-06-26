# 2023100066-12

2023100066自动化李智阳+实验作业12

# 使用棋盘格进行相机标定

## 1.项目目的

1. 理解棋盘格标定中三维空间点与二维图像点之间的对应关系。
2. 掌握相机成像模型中内参矩阵、外参矩阵和畸变参数的基本含义。
3. 学会使用OpenCV读取标定图像并检测棋盘格内角点。
4. 掌握`cv2.findChessboardCorners()`检测棋盘格角点的方法。
5. 掌握`cv2.cornerSubPix()`对角点进行亚像素精度优化的方法。
6. 使用`cv2.calibrateCamera()`完成相机内参、畸变参数和外参估计。
7. 计算重投影误差，评价相机标定结果的准确性。
8. 使用`cv2.undistort()`对原始图像进行去畸变处理，并对比去畸变前后的效果。

## 2.运行环境

* 操作系统：Ubuntu/WSLUbuntu/Windows
* Python版本：Python3
* 开发工具：VSCode
* 主要依赖库：

```bash
pip install opencv-python numpy matplotlib
```

说明：本实验主要使用OpenCV完成棋盘格角点检测、亚像素优化、相机标定和图像去畸变处理。

## 3.实验数据说明

1. 棋盘格来源：打印棋盘格。
2. 拍摄设备：vivoX200s手机。
3. 棋盘格内角点数量：9×6。
4. 方格边长：25mm。
5. 待标定图片存放路径：

```text
/home/lzy/cv-course/build/实验12/待标定图像
```

6. 本次实验共处理15张棋盘格图像。

7. 角点检测结果：

   * 成功检测图片数量：14张
   * 检测失败图片数量：1张
   * 检测失败图片：1.jpg

8. 输出结果保存路径：

```text
/home/lzy/cv-course/build/实验12/标定结果
```

## 4.主要功能

1. 图像读取：读取待标定图像文件夹中的`.jpg`、`.jpeg`、`.png`、`.bmp`、`.tif`、`.tiff`等格式图片。
2. 角点检测：使用OpenCV检测每张图像中的9×6棋盘格内角点。
3. 亚像素优化：对检测到的棋盘格角点进行亚像素精度优化，提高角点定位精度。
4. 三维点构建：根据棋盘格内角点数量和实际方格边长建立标定板坐标系下的三维坐标。
5. 相机标定：估计相机内参矩阵K、畸变参数D和每张图像对应的外参。
6. 误差计算：计算每张有效图片的平均重投影误差和总平均重投影误差。
7. 去畸变处理：对一张原始棋盘格图片进行去畸变处理。
8. 结果保存：保存角点检测图、标定参数、去畸变图像和原图对比图。

## 5.文件结构说明

```text
实验12/
├── 待标定图像/                         # 原始棋盘格标定图片
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── ...
│   └── 15.jpg
├── 标定结果/                           # 程序运行后生成的结果文件夹
│   ├── camera_calibration_result.npz   # 相机标定结果数据文件
│   ├── camera_calibration_result.txt   # 相机标定结果文本文件
│   ├── 角点检测结果_01.jpg              # 角点检测绘制结果
│   ├── 角点检测结果_02.jpg
│   ├── ...
│   ├── original_image.jpg              # 用于去畸变的原始图像
│   ├── undistorted_image.jpg           # 去畸变图像
│   ├── undistorted_cropped_image.jpg   # 裁剪黑边后的去畸变图像
│   └── original_vs_undistorted.png     # 原图与去畸变图像对比图
├── task12.py                           # 实验主代码
└── README.md                           # 项目说明文件
```

## 6.核心代码与说明

### 6.1基本参数设置

```python
from pathlib import Path

IMAGE_DIR = Path("/home/lzy/cv-course/build/实验12/待标定图像")
OUTPUT_DIR = Path("/home/lzy/cv-course/build/实验12/标定结果")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 25.0
```

### 6.2构造棋盘格三维坐标

```python
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)

objp[:, :2] = np.mgrid[
    0:CHESSBOARD_SIZE[0],
    0:CHESSBOARD_SIZE[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE
```


### 6.3检测棋盘格角点

```python
ret, corners = cv2.findChessboardCorners(
    gray,
    CHESSBOARD_SIZE,
    None
)
```


### 6.4亚像素精度优化

```python
corners_subpix = cv2.cornerSubPix(
    gray,
    corners,
    winSize=(11, 11),
    zeroZone=(-1, -1),
    criteria=criteria
)
```


### 6.5绘制角点检测结果

```python
img_corners = img.copy()

cv2.drawChessboardCorners(
    img_corners,
    CHESSBOARD_SIZE,
    corners_subpix,
    ret
)

save_path = OUTPUT_DIR / f"角点检测结果_{success_count:02d}_原图{img_path.stem}.jpg"
cv2.imencode(".jpg", img_corners)[1].tofile(str(save_path))
```

### 6.6相机标定

```python
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)
```

说明：

### 6.7重投影误差计算

```python
projected_points, _ = cv2.projectPoints(
    objpoints[i],
    rvecs[i],
    tvecs[i],
    camera_matrix,
    dist_coeffs
)

error = cv2.norm(imgpoints[i], projected_points, cv2.NORM_L2) / len(projected_points)
```

### 6.8图像去畸变

```python
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    dist_coeffs,
    (w, h),
    alpha=0,
    newImgSize=(w, h)
)

undistorted = cv2.undistort(
    img,
    camera_matrix,
    dist_coeffs,
    None,
    new_camera_matrix
)
```


## 7.核心参数说明

1. `CHESSBOARD_SIZE`：棋盘格内角点数量，本实验为`(9,6)`。
2. `SQUARE_SIZE`：棋盘格方格边长，本实验为`25.0mm`。
3. `objpoints`：棋盘格在标定板坐标系中的三维角点坐标。
4. `imgpoints`：图像中检测到的二维角点坐标。
5. `camera_matrix`：相机内参矩阵K。
6. `dist_coeffs`：相机畸变参数D。
7. `rvecs`：每张图像对应的旋转向量。
8. `tvecs`：每张图像对应的平移向量。
9. `criteria`：亚像素优化的终止条件。
10. `alpha`：去畸变图像保留视野的参数。`alpha=0`时黑边较少，`alpha=1`时视野保留更多但黑边较明显。

## 8.运行步骤

### 8.1进入实验目录

```bash
cd "/home/lzy/cv-course/build/实验12"
```

### 8.2安装依赖

```bash
pip install opencv-python numpy matplotlib
```

### 8.3运行相机标定程序


```bash
/home/lzy/cv-course/.venv-basic/bin/python /home/lzy/cv-course/build/实验12/task12.py
```

### 8.4查看输出结果

程序运行完成后，结果文件会保存在：

```text
/home/lzy/cv-course/build/实验12/标定结果
```

主要输出文件包括：

```text
camera_calibration_result.txt
camera_calibration_result.npz
角点检测结果_*.jpg
original_image.jpg
undistorted_image.jpg
undistorted_cropped_image.jpg
original_vs_undistorted.png
```

## 9.实验结果与分析

### 9.1角点检测结果

本实验程序共处理15张棋盘格标定图像，其中14张成功检测出棋盘格内角点，1张检测失败。

```text
成功检测图片数量：14
检测失败图片数量：1
检测失败图片：1.jpg
```

说明大部分图像能够满足棋盘格角点检测要求。检测失败的图片可能存在棋盘格姿态过于极端、图像局部模糊、纸张弯曲、边缘角点不完整或黑白角点对比度不足等问题。

### 9.2相机内参矩阵

本次实验得到的相机内参矩阵K为：

```text
K =
[[3056.51529,    0.00000, 1621.06468],
 [   0.00000, 3031.95836, 1903.60438],
 [   0.00000,    0.00000,    1.00000]]
```

其中：

1. `fx=3056.51529`
2. `fy=3031.95836`
3. `cx=1621.06468`
4. `cy=1903.60438`

fx和fy数值较为接近，说明相机在水平和竖直方向上的等效焦距基本一致。cx和cy表示相机主点坐标，结合原始图像分辨率来看，主点整体位于图像中心区域附近，但仍存在一定偏差。

### 9.3畸变参数

本次实验得到的畸变参数D为：

```text
D = [-0.00457215, 0.00860206, -0.00090259, 0.00428736, -0.7894696]
```

其中：

1. `k1`、`k2`、`k3`表示径向畸变参数。
2. `p1`、`p2`表示切向畸变参数。

该参数用于描述镜头成像过程中产生的畸变，并可用于后续图像去畸变处理。

### 9.4重投影误差

本实验程序手动计算得到的总平均重投影误差为：

```text
Mean reprojection error = 0.7128pixel
```

OpenCV返回的RMS重投影误差为：

```text
RMS reprojection error = 6.5979pixel
```

从每张图像的平均重投影误差来看，大多数图像的误差低于1pixel，说明角点检测结果与重投影点之间整体较为接近。但OpenCV返回的RMS误差相对偏大，说明整体标定仍存在改进空间。造成该现象的原因可能包括部分图像拍摄角度较大、纸张不够平整、局部角点定位不够准确，以及少数误差较高的图像影响了整体标定结果。

### 9.5去畸变结果

完成相机标定后，程序选取一张棋盘格图像进行去畸变处理。去畸变的目的是根据估计得到的相机内参和畸变参数，减小镜头径向畸变和切向畸变对图像造成的影响，使图像中的直线结构更加接近真实成像关系。

去畸变后，图像边缘可能出现一定裁剪或黑边，这是由于图像几何校正时部分区域没有对应的有效像素造成的。使用`alpha=0`可以减少黑边，使结果图像更加适合实验报告展示。

## 10.实验结论

1. 本实验完成了基于棋盘格的相机标定流程。
2. 通过OpenCV成功读取并处理了多张棋盘格标定图像。
3. 本实验采用9×6内角点、25mm方格边长的打印棋盘格作为标定板。
4. 程序成功检测14张图像中的棋盘格角点，并对角点进行了亚像素精度优化。
5. 通过`cv2.calibrateCamera()`成功求得相机内参矩阵、畸变参数和每张图像的外参。
6. 手动计算得到的总平均重投影误差为0.7128pixel，说明标定结果具有一定可靠性。
7. 通过`cv2.undistort()`完成了图像去畸变处理，能够观察到相机畸变校正效果。

## 11.改进方向

1. 将打印棋盘格固定在硬纸板或平整板材上，减少纸张弯曲对角点位置的影响。
2. 拍摄时保证棋盘格完整出现在画面中，避免边缘遮挡。
3. 增加更多不同距离、不同角度、不同位置的标定图像。
4. 尽量让棋盘格覆盖图像中心、边缘和四角区域，提高标定稳定性。


## 12.作者信息

作者：李智阳
学号：2023100066
日期：2026年6月26日
