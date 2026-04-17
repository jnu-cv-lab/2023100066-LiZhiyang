import cv2
import numpy as np
import matplotlib.pyplot as plt

# ====================== 读取测试图 ======================
def load_test_image(image_path):

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片，请检查路径：{image_path}")
    return img

# ====================== 相似变换 ======================
def similarity_transform(img):
 
    h, w = img.shape[:2]
    # 以图像中心为旋转中心
    center = (w // 2, h // 2)
    # 生成旋转+缩放矩阵：旋转22度，缩放0.85倍
    M = cv2.getRotationMatrix2D(center, 22, 0.85)
    # 执行变换
    return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

# ====================== 仿射变换 ======================
def affine_transform(img):
 
    h, w = img.shape[:2]
    # 原始图像的3个关键点
    src_pts = np.float32([[50, 50], [w - 50, 50], [50, h - 50]])
    # 变换后的目标点
    dst_pts = np.float32([[30, 120], [w - 80, 60], [100, h - 80]])
    # 计算仿射变换矩阵
    M = cv2.getAffineTransform(src_pts, dst_pts)
    # 执行仿射变换
    return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

# ====================== 透视变换 ======================
def perspective_transform(img):

    h, w = img.shape[:2]
    # 原图四个角点
    src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    # 目标四个角点
    dst_pts = np.float32([
        [100, 80],
        [w - 200, 150],
        [50, h - 50],
        [w - 50, h - 20]
    ])
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # 执行透视变换
    return cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))

# ====================== 保存图片并显示 ======================
def show_and_save(original, sim, aff, pers):
    """
    保存所有结果图片，并使用matplotlib显示对比图
    """
    # 保存结果到文件
    cv2.imwrite("original.png", original)
    cv2.imwrite("similarity.png", sim)
    cv2.imwrite("affine.png", aff)
    cv2.imwrite("perspective.png", pers)

    # 创建大图显示对比
    plt.figure(figsize=(16, 10))
    titles = ["Original", "Similarity", "Affine", "Perspective"]
    images = [original, sim, aff, pers]

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i], fontsize=14)
        plt.axis('off')  # 关闭坐标轴

    plt.tight_layout()
    plt.savefig("result_all.png", dpi=200)  # 保存合成图
    plt.show()

# ====================== 主程序入口 ======================
if __name__ == "__main__":
    # 测试图片路径
    image_path = "/home/lzy/cv-course/build/实验五/测试图.png"  
    
    # 读取原图
    original = load_test_image(image_path)
    # 三种变换
    sim_img = similarity_transform(original)
    aff_img = affine_transform(original)
    pers_img = perspective_transform(original)
    # 显示并保存结果
    show_and_save(original, sim_img, aff_img, pers_img)