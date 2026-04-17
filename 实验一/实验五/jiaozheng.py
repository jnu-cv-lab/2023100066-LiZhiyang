import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_document_corners(image):
    """
    自动检测文档的四个角点
    """
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    # 膨胀和腐蚀，连接边缘
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 找到最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)
    
    # 多边形近似
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    
    # 如果找不到四个角点，让用户手动选择
    if len(approx) != 4:
        print("自动检测失败，请手动选择四个角点...")
        return manual_select_corners(image)
    
    # 按顺序排列角点（左上、右上、右下、左下）
    corners = arrange_corners(approx.reshape(4, 2))
    
    return corners

def manual_select_corners(image):
    """
    手动选择文档的四个角点
    """
    clone = image.copy()
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select corners (4 points in order: TL, TR, BR, BL)", clone)
            if len(points) == 4:
                cv2.destroyAllWindows()
    
    cv2.imshow("Select corners (4 points in order: TL, TR, BR, BL)", clone)
    cv2.setMouseCallback("Select corners (4 points in order: TL, TR, BR, BL)", mouse_callback)
    cv2.waitKey(0)
    
    return np.array(points, dtype=np.float32)

def arrange_corners(points):
    """
    将四个点按顺序排列：左上、右上、右下、左下
    """
    points = points.astype(np.float32)
    
    # 计算中心点
    center = np.mean(points, axis=0)
    
    # 计算每个点相对于中心点的角度
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    
    # 按角度排序
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    
    # 调整顺序：左上、右上、右下、左下
    tl, tr, br, bl = sorted_points[0], sorted_points[1], sorted_points[2], sorted_points[3]
    
    # 确保tl是左上角（x和y最小）
    if tl[0] > tr[0]:
        tl, tr = tr, tl
    if tl[1] > bl[1]:
        tl, bl = bl, tl
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

def correct_perspective(image, corners):
    """
    使用透视变换校正图像
    """
    # 定义目标图像的尺寸
    width = 800
    height = int(width * 1.414)  # 800 * 1.414 ≈ 1131
    
    # 目标图像的四个角点
    dst_corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(corners, dst_corners)   
    # 应用透视变换
    corrected = cv2.warpPerspective(image, matrix, (width, height))   
    return corrected

def show_comparison(original, corrected):
    """
    显示原始图像和校正后的图像对比
    """
    plt.figure(figsize=(15, 8))
    
    # 原始图像
    orig_with_points = original.copy()
    for point in corners:
        cv2.circle(orig_with_points, tuple(point.astype(int)), 10, (0, 255, 0), -1)
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(orig_with_points, cv2.COLOR_BGR2RGB))
    plt.title('Original Image with Detected Corners')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    plt.title('Perspective Corrected Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_results(corrected, output_path='corrected_image.jpg'):
    """
    保存校正后的图像
    """
    cv2.imwrite(output_path, corrected)
    print(f"校正后的图像已保存为: {output_path}")

# 主程序
if __name__ == "__main__":
    # 读取图像
    image_path = '/home/lzy/cv-course/build/实验五/畸变图.jpg'  
    image = cv2.imread(image_path)
    
    if image is None:
        print("错误：无法读取图像，请检查路径")
        print("请将图像文件放在正确的位置，或修改 image_path 变量")
        exit()
    
    print("正在检测文档角点...")
    
    # 自动检测角点
    corners = find_document_corners(image)
    print(f"检测到的角点: {corners}")
    
    # 进行透视校正
    print("正在进行透视校正...")
    corrected = correct_perspective(image, corners)
    
    # 显示结果
    show_comparison(image, corrected)
    
    # 保存结果
    save_results(corrected)
    
