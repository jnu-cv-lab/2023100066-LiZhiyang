import cv2
import numpy as np
import os
#  任务1: 读取一张图片
img_path = r"/home/lzy/.venv/bin/test.jpg"

# 检查图片文件是否存在
if not os.path.exists(img_path):
    print(f"错误：文件不存在！路径: {img_path}")
    exit()

# 读取图片字节流
img_bytes = np.fromfile(img_path, dtype=np.uint8)
# 解码字节流为OpenCV可识别的图像格式
img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
# 检查图片是否解码成功
if img is None:
    print("图片读取失败！可能是文件格式不支持")
    exit()
print(" 图片读取成功！")

#  任务2: 输出图像信息 
# 获取图像高度和宽度
height, width = img.shape[:2]
# 判断图像通道数（彩色图3通道，灰度图1通道）
channels = img.shape[2] if len(img.shape) == 3 else 1
# 获取图像像素数据类型
dtype = img.dtype
print("\n===== 图像基本信息 =====")
print(f"图像尺寸（宽×高）: {width} × {height}")
print(f"图像通道数: {channels}")
print(f"像素数据类型: {dtype}")

#  任务3: 显示原图 
# 创建窗口显示原图
cv2.imshow("Original Image", img)
cv2.waitKey(0)
# 关闭所有OpenCV创建的窗口
cv2.destroyAllWindows()

#  任务4: 转灰度图并显示 
# 将彩色图转换为灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建窗口显示灰度图
cv2.imshow("Grayscale Image", gray_img)
cv2.waitKey(0)
# 关闭所有窗口
cv2.destroyAllWindows()

#  任务5: 保存灰度图 
save_path = r"/home/lzy/.venv/bin/gray.png"
# 编码灰度图为PNG格式字节流
_, img_encoded = cv2.imencode(".png", gray_img)
# 将编码后的字节流写入文件
img_encoded.tofile(save_path)
print(f"\n灰度图已保存至: {save_path}")

# 任务6: NumPy 操作 
# 输出指定像素值
pixel = img[502, 205] if channels == 3 else gray_img[100, 100]
print(f"\n坐标 (502,205) 的像素值: {pixel}")

CROP_WIDTH = 400   # 裁剪宽度
CROP_HEIGHT = 400  # 裁剪高度
# 校验裁剪尺寸是否超出原图范围
if CROP_WIDTH > width or CROP_HEIGHT > height:
    print(f"\n裁剪尺寸超出原图范围!")
    # 若超出范围，将裁剪尺寸调整为原图尺寸
    CROP_WIDTH = width
    CROP_HEIGHT = height

# 裁剪左上角区域
cropped_img = img[0:CROP_HEIGHT, 0:CROP_WIDTH]

# 显示裁剪结果并标注信息
print(f"裁剪区域：左上角({CROP_WIDTH}, {CROP_HEIGHT})")
# 输出裁剪后图像的宽高
print(f"裁剪后尺寸：宽{cropped_img.shape[1]} × 高{cropped_img.shape[0]}")
# 创建窗口显示裁剪后的图像
cv2.imshow(f"Cropped Image (Top-left {CROP_WIDTH}x{CROP_HEIGHT})", cropped_img)
# 等待按键输入
cv2.waitKey(0)
# 关闭所有窗口
cv2.destroyAllWindows()
# 裁剪图保存路径
crop_save_path = r"/home/lzy/.venv/bin/cut.png"
# 编码裁剪图为PNG格式字节流
_, crop_encoded = cv2.imencode(".png", cropped_img)
# 将编码后的字节流写入文件
crop_encoded.tofile(crop_save_path)
print(f"裁剪区域已保存至: {crop_save_path}")