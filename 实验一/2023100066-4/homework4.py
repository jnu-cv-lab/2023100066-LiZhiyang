import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===================== 控制变量 =====================
M     = 2
sigma = 1
# ====================================================

# 棋盘格
def generate_checkerboard(size=256, block_size=8):
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if (i//block_size + j//block_size) % 2 == 0:
                img[i,j] = 255
    return img

# Chirp
def generate_chirp(size=256):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    img = np.sin(2 * np.pi * (5 * r + 50 * r**2))
    img = (img - img.min()) / (img.max() - img.min()) * 255
    return img.astype(np.uint8)

# 下采样
def downsample(img, M):
    return img[::M, ::M]

# 高斯下采样
def gaussian_downsample(img, M, sigma):
    blurred = cv2.GaussianBlur(img, (5, 5), sigma)
    return downsample(blurred, M)

# FFT
def get_fft_spectrum(img):
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(f_shift) + 1)
    return magnitude

# ===================== 绘图 =====================
def plot_one_img(img, title_prefix, save_path):
    img_direct = downsample(img, M)
    img_gauss  = gaussian_downsample(img, M, sigma)

    fft_ori    = get_fft_spectrum(img)
    fft_dir    = get_fft_spectrum(img_direct)
    fft_gau    = get_fft_spectrum(img_gauss)

    plt.figure(figsize=(16, 6))

    # 图像部分
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

    # FFT 部分
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
    plt.savefig(save_path, dpi=150)
    plt.close()

# ===================== 分别绘制两张图表 =====================
checker = generate_checkerboard()
chirp   = generate_chirp()

plot_one_img(checker, "Checkerboard", "result_checker.png")
plot_one_img(chirp,   "Chirp",       "result_chirp.png")

print(" 两张图已分别保存：")
print("  - result_checker.png")
print("  - result_chirp.png")