# OpenCV 基础图像处理实验

## 1. 项目介绍
本项目使用 Python + OpenCV + NumPy 实现基础图像处理功能，包括图片读取、信息查看、灰度转换、图像裁剪、显示与保存等操作。

## 2. 运行环境
- Python3
- Linux/wsl
  
安装依赖库：
pip install opencv-python numpy

## 3. 主要功能
1. 读取本地图片并判断文件是否存在
2. 输出图像尺寸、通道数、数据类型
3. 显示原始图像
4. 将彩色图像转为灰度图并显示
5. 保存灰度图像
6. 获取指定坐标像素值
7. 裁剪图像左上角区域并保存

## 4. 核心代码与说明

### 4.1 图片读取
```python
img_bytes = np.fromfile(img_path, dtype=np.uint8)
img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
```

### 4.2 获取图像信息
```python
height, width = img.shape[:2]
channels = img.shape[2] if len(img.shape) == 3 else 1
dtype = img.dtype
```
### 4.3 转为灰度图
```python
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```
### 4.4 保存图像
```python
_, img_encoded = cv2.imencode(".png", gray_img)
img_encoded.tofile(save_path)
```
### 4.5 图像裁剪
```python
cropped_img = img[0:CROP_HEIGHT, 0:CROP_WIDTH]
```
## 5. 使用说明
1. 可修改 img_path 为自己的图片路径
2. 运行脚本
3. 按任意键关闭图片窗口
4. 灰度图与裁剪图会自动保存到指定路径
## 6. 作者信息
1. 作者：李智阳
2. 日期：2026年3月20日

