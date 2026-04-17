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
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, 22, 0.85)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))

# ====================== 仿射变换 ======================
def affine_transform(img):
    h, w = img.shape[:2]
    # 取图片左上角区域的3个点做仿射变换
    src_pts = np.float32([[50, 50], [w-50, 50], [50, h-50]])
    dst_pts = np.float32([[30, 120], [w-80, 60], [100, h-80]])
    M = cv2.getAffineTransform(src_pts, dst_pts)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))

# ====================== 透视变换 ======================
def perspective_transform(img):
    h, w = img.shape[:2]
    src_pts = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    dst_pts = np.float32([
        [100, 80],
        [w - 200, 150],
        [50, h - 50],
        [w - 50, h - 20]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(255,255,255))

# ====================== 保存图片 ======================
def show_and_save(original, sim, aff, pers):
    cv2.imwrite("original.png", original)
    cv2.imwrite("similarity.png", sim)
    cv2.imwrite("affine.png", aff)
    cv2.imwrite("perspective.png", pers)

    plt.figure(figsize=(16, 10))
    titles = ["Original", "Similarity", "Affine", "Perspective"]
    images = [original, sim, aff, pers]

    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i], fontsize=14)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("result_all.png", dpi=200)
    plt.show()

# ====================== 主程序 ======================
if __name__ == "__main__":
    image_path = "/home/lzy/cv-course/build/测试图.png"  
    
    original = load_test_image(image_path)
    sim_img = similarity_transform(original)
    aff_img = affine_transform(original)
    pers_img = perspective_transform(original)
    show_and_save(original, sim_img, aff_img, pers_img)