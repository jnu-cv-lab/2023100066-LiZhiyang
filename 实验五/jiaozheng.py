import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 强制开启窗口（关键！）
import matplotlib.pyplot as plt

# 开启交互点击
plt.rcParams['figure.figsize'] = [12, 7]

def correct_perspective(image_path):
    # 1. 读取你的畸变图
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. 强制弹出图片！
    print("✅ 正在打开图片...请点击4个角：左上 → 右上 → 右下 → 左下")
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.title("点击4个角，关闭窗口继续", fontsize=14)
    
    # 等待你点击4个点
    points = plt.ginput(4, timeout=-1)  # 永久等待点击
    plt.close()

    # 3. 校正
    w_out, h_out = 600, 800
    dst = np.float32([[0,0], [w_out,0], [w_out,h_out], [0,h_out]])
    src = np.float32(points)
    M = cv2.getPerspectiveTransform(src, dst)
    corrected = cv2.warpPerspective(img, M, (w_out, h_out))

    return img, corrected

# ====================== 运行 ======================
if __name__ == "__main__":
    img_path = "/home/lzy/cv-course/build/畸变图.jpg"
    
    # 校正
    ori, corr = correct_perspective(img_path)
    
    # 保存
    cv2.imwrite("result_corrected.png", cv2.cvtColor(corr, cv2.COLOR_RGB2BGR))

    # 显示最终对比图
    plt.figure()
    plt.subplot(121), plt.imshow(ori), plt.title("畸变图")
    plt.subplot(122), plt.imshow(corr), plt.title("校正完成")
    plt.show()