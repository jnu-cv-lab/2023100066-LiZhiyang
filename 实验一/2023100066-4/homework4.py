import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===================== 实验控制参数 =====================
# 下采样倍数 M
M     = 2
# 高斯滤波标准差 sigma
sigma = 1
# ========================================================

# 生成棋盘格测试图像
def generate_checkerboard(size=256, block_size=8):
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            # 黑白交替形成棋盘格
            if (i//block_size + j//block_size) % 2 == 0:
                img[i,j] = 255
    return img

# 生成Chirp调频测试图像
def generate_chirp(size=256):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)  # 极坐标半径
    img = np.sin(2 * np.pi * (5 * r + 50 * r**2))  # 调频正弦
    img = (img - img.min()) / (img.max() - img.min()) * 255  # 归一化到0-255
    return img.astype(np.uint8)

# 直接下采样
# 仅每隔 M 个像素取一个值，会产生混叠
def downsample(img, M):
    return img[::M, ::M]

# 高斯滤波 + 下采样
# 先做高斯低通滤波，再下采样
def gaussian_downsample(img, M, sigma):
    blurred = cv2.GaussianBlur(img, (5, 5), sigma)  # 高斯滤波去除高频
    return downsample(blurred, M)                   # 再下采样

# 计算图像的FFT频谱
def get_fft_spectrum(img):
    f = np.fft.fft2(img)          # 2D傅里叶变换
    f_shift = np.fft.fftshift(f)  # 将低频移到图像中心
    magnitude = 20 * np.log(np.abs(f_shift) + 1)  # 计算幅度谱并对数拉伸
    return magnitude

# ===================== 绘图函数 =====================
def plot_one_img(img, title_prefix, save_path):
    # 直接下采样
    img_direct = downsample(img, M)
    # 高斯滤波后下采样
    img_gauss  = gaussian_downsample(img, M, sigma)

    # 计算三张图的FFT频谱
    fft_ori    = get_fft_spectrum(img)
    fft_dir    = get_fft_spectrum(img_direct)
    fft_gau    = get_fft_spectrum(img_gauss)

    plt.figure(figsize=(16, 6))

    # 空域图像
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'{title_prefix} Original')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(img_direct, cmap='gray')
    plt.title('Direct Downsample')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(img_gauss, cmap='gray')
    plt.title(f'Gaussian σ={sigma}')
    plt.axis('off')

    # 频域FFT频谱
    plt.subplot(2, 3, 4)
    plt.imshow(fft_ori, cmap='gray')
    plt.title('FFT Original')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(fft_dir, cmap='gray')
    plt.title('FFT Direct')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(fft_gau, cmap='gray')
    plt.title('FFT Gaussian')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)  # 保存图片
    plt.close()

# ===================== 主程序 =====================
# 生成棋盘格与chirp测试图
checker = generate_checkerboard()
chirp   = generate_chirp()

# 分别绘制两张图的实验结果
plot_one_img(checker, "Checkerboard", "result_checker.png")
plot_one_img(chirp,   "Chirp",       "result_chirp.png")

print("两张图已分别保存：")
print("  - result_checker.png")
print("  - result_chirp.png")