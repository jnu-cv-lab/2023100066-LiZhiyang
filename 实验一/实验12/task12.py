import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# =========================
# 1. 基本参数设置
# =========================

# WSL / Ubuntu 下的图片文件夹路径
IMAGE_DIR = Path("/home/lzy/cv-course/build/实验12/待标定图像")

# 输出结果保存文件夹
OUTPUT_DIR = Path("/home/lzy/cv-course/build/实验12/标定结果")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 棋盘格内角点数量
# 注意：这里是“内角点”数量，不是黑白格子的数量
# 常见作业要求为 9 × 6
CHESSBOARD_SIZE = (9, 6)

# 每个小方格的实际边长，单位：mm
# 如果你用的是 25 mm，就写 25.0；如果是 30 mm，就改成 30.0
SQUARE_SIZE = 25.0


# =========================
# 2. 构造棋盘格三维坐标
# =========================

# objp 用来存放棋盘格角点在真实世界中的三维坐标
# 假设棋盘格平面为 Z = 0
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)

# 生成 (0,0,0), (1,0,0), (2,0,0) ... 这样的点
objp[:, :2] = np.mgrid[
    0:CHESSBOARD_SIZE[0],
    0:CHESSBOARD_SIZE[1]
].T.reshape(-1, 2)

# 乘以实际方格边长，得到单位为 mm 的真实坐标
objp *= SQUARE_SIZE


# =========================
# 3. 读取图片并检测角点
# =========================

# 存放三维点和二维图像点
objpoints = []
imgpoints = []

# 支持的图片格式
image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]

image_paths = []
for ext in image_extensions:
    image_paths.extend(IMAGE_DIR.glob(ext))

image_paths = sorted(image_paths)

if len(image_paths) == 0:
    raise FileNotFoundError(f"没有在该文件夹中找到图片：{IMAGE_DIR}")

print(f"共找到 {len(image_paths)} 张图片。")

# 亚像素角点优化的终止条件
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# 用于记录图像大小
image_size = None

success_count = 0
failed_images = []

for idx, img_path in enumerate(image_paths):
    print(f"\n正在处理第 {idx + 1}/{len(image_paths)} 张图片：{img_path.name}")

    # 为了更好支持中文路径，这里使用 np.fromfile + cv2.imdecode 读取图片
    img_array = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        print("读取失败，跳过该图片。")
        failed_images.append(img_path.name)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray.shape[::-1]  # OpenCV 中图像尺寸格式为 (width, height)

    # 检测棋盘格角点
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE,
        None
    )

    if ret:
        success_count += 1
        print("角点检测成功。")

        # 亚像素精度优化
        corners_subpix = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria
        )

        objpoints.append(objp)
        imgpoints.append(corners_subpix)

        # 绘制角点检测结果
        img_corners = img.copy()
        cv2.drawChessboardCorners(
            img_corners,
            CHESSBOARD_SIZE,
            corners_subpix,
            ret
        )

        save_path = OUTPUT_DIR / f"角点检测结果_{success_count:02d}_原图{img_path.stem}.jpg"
        cv2.imencode(".jpg", img_corners)[1].tofile(str(save_path))

    else:
        print("角点检测失败。")
        failed_images.append(img_path.name)


print("\n============================")
print("角点检测统计")
print("============================")
print(f"成功检测图片数量：{success_count}")
print(f"检测失败图片数量：{len(failed_images)}")

if failed_images:
    print("\n检测失败的图片：")
    for name in failed_images:
        print(f"- {name}")

if success_count < 3:
    raise RuntimeError("成功检测角点的图片太少，无法完成可靠标定。建议至少使用 10~15 张有效图片。")


# =========================
# 4. 相机标定
# =========================

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)

print("\n============================")
print("相机标定结果")
print("============================")

print("\n相机内参矩阵 K：")
print(camera_matrix)

print("\n畸变参数 D：")
print(dist_coeffs)

print("\nOpenCV 返回的 RMS 重投影误差：")
print(ret)


# =========================
# 5. 手动计算平均重投影误差
# =========================

total_error = 0
per_image_errors = []

for i in range(len(objpoints)):
    projected_points, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    error = cv2.norm(imgpoints[i], projected_points, cv2.NORM_L2) / len(projected_points)
    per_image_errors.append(error)
    total_error += error

mean_error = total_error / len(objpoints)

print("\n每张图片的平均重投影误差：")
for i, error in enumerate(per_image_errors):
    print(f"第 {i + 1} 张有效图片：{error:.4f} pixel")

print("\n总平均重投影误差：")
print(f"{mean_error:.4f} pixel")


# =========================
# 6. 保存标定结果
# =========================

np.savez(
    OUTPUT_DIR / "camera_calibration_result.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    rvecs=rvecs,
    tvecs=tvecs,
    reprojection_error=mean_error
)

with open(OUTPUT_DIR / "camera_calibration_result.txt", "w", encoding="utf-8") as f:
    f.write("相机标定结果\n")
    f.write("============================\n\n")

    f.write("相机内参矩阵 K：\n")
    f.write(str(camera_matrix))
    f.write("\n\n")

    f.write("畸变参数 D：\n")
    f.write(str(dist_coeffs))
    f.write("\n\n")

    f.write(f"OpenCV RMS 重投影误差：{ret}\n")
    f.write(f"平均重投影误差：{mean_error:.4f} pixel\n\n")

    f.write("每张有效图片的重投影误差：\n")
    for i, error in enumerate(per_image_errors):
        f.write(f"第 {i + 1} 张有效图片：{error:.4f} pixel\n")

    if failed_images:
        f.write("\n角点检测失败图片：\n")
        for name in failed_images:
            f.write(f"{name}\n")


# =========================
# 7. 对一张图片进行去畸变处理
# =========================

# 默认选择第一张成功检测的图片进行去畸变
undistort_img_path = None

for img_path in image_paths:
    img_array = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret_find, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret_find:
        undistort_img_path = img_path
        break

if undistort_img_path is None:
    raise RuntimeError("没有找到可用于去畸变处理的图片。")

img_array = np.fromfile(str(undistort_img_path), dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

h, w = img.shape[:2]

# 获取优化后的新相机矩阵
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    dist_coeffs,
    (w, h),
    alpha=0,
    newImgSize=(w, h)
)

# 去畸变
undistorted = cv2.undistort(
    img,
    camera_matrix,
    dist_coeffs,
    None,
    new_camera_matrix
)

#裁剪有效区域
x, y, roi_w, roi_h = roi
if roi_w > 0 and roi_h > 0:
    undistorted_cropped = undistorted[y:y + roi_h, x:x + roi_w]
else:
    undistorted_cropped = undistorted

# 保存原图和去畸变图
original_save_path = OUTPUT_DIR / "original_image.jpg"
undistorted_save_path = OUTPUT_DIR / "undistorted_image.jpg"
undistorted_cropped_save_path = OUTPUT_DIR / "undistorted_cropped_image.jpg"

cv2.imencode(".jpg", img)[1].tofile(str(original_save_path))
cv2.imencode(".jpg", undistorted)[1].tofile(str(undistorted_save_path))
cv2.imencode(".jpg", undistorted_cropped)[1].tofile(str(undistorted_cropped_save_path))


# =========================
# 8. 生成原图与去畸变图对比图
# =========================

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
undistorted_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(undistorted_rgb)
plt.title("Undistorted Image")
plt.axis("off")

plt.tight_layout()
comparison_save_path = OUTPUT_DIR / "original_vs_undistorted.png"
plt.savefig(comparison_save_path, dpi=300)
plt.close()

print("\n============================")
print("结果文件已保存")
print("============================")
print(f"角点检测图、标定结果、去畸变图片均已保存到：")
print(OUTPUT_DIR)
print(f"\n去畸变对比图：{comparison_save_path}")