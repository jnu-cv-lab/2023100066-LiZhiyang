import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===================== 全局参数 =====================
M_min = 2       # 细节区最小下采倍数
M_max = 4       # 平坦区最大下采倍数
sigma_coeff = 0.45  # σ经验公式系数
# ======================================================

# 生成棋盘格测试图
def generate_checkerboard(size=256, block_size=8):
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if (i//block_size + j//block_size) % 2 == 0:
                img[i,j] = 255
    return img

# 生成chirp测试图
def generate_chirp(size=256):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    img = np.sin(2 * np.pi * (5 * r + 50 * r**2))
    img = (img - img.min()) / (img.max() - img.min()) * 255
    return img.astype(np.uint8)

# 计算梯度幅值图
def compute_gradient(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
    return grad_mag

# 生成局部M/σ图
def generate_local_M_sigma(grad_mag, M_min, M_max, sigma_coeff):
    local_M = M_max - (M_max - M_min) * grad_mag
    local_M = np.round(local_M).astype(np.int32)
    local_sigma = sigma_coeff * local_M
    return local_M, local_sigma

# 自适应高斯滤波
def adaptive_gaussian_blur(img, local_sigma):
    h, w = img.shape[:2]
    blurred = np.zeros_like(img, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            sigma = local_sigma[i, j]
            ksize = (2*int(4*sigma)+1, 2*int(4*sigma)+1)
            y1, y2 = max(0, i-2), min(h, i+3)
            x1, x2 = max(0, j-2), min(w, j+3)
            patch = img[y1:y2, x1:x2]
            blurred[i, j] = cv2.GaussianBlur(patch, ksize, sigma)[2, 2]
    return blurred.astype(np.uint8)

# 自适应下采样
def adaptive_downsample(img, local_M):
    h, w = img.shape[:2]
    new_h, new_w = h // M_max, w // M_max
    downsampled = np.zeros((new_h, new_w), dtype=np.uint8)
    
    for i in range(new_h):
        for j in range(new_w):
            y_start = i * M_max
            y_end = y_start + M_max
            x_start = j * M_max
            x_end = x_start + M_max
            
            region_M = local_M[y_start:y_end, x_start:x_end].mean()
            M = int(round(region_M))
            M = max(M_min, min(M_max, M))
            
            region = img[y_start:y_end, x_start:x_end]
            downsampled[i, j] = region[::M, ::M].mean()
    return downsampled

# 全局统一下采样
def global_downsample(img, M, sigma):
    ksize = (2*int(4*sigma)+1, 2*int(4*sigma)+1)
    blurred = cv2.GaussianBlur(img, ksize, sigma)
    return blurred[::M, ::M]

# 计算误差图
def compute_error_map(original, downsampled):
    h, w = original.shape[:2]
    upsampled = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_CUBIC)
    error_map = cv2.absdiff(original, upsampled)  # 绝对误差图
    return error_map

# 统一处理函数
def process(img, title_prefix, save_name):
    grad_mag = compute_gradient(img)
    local_M, local_sigma = generate_local_M_sigma(grad_mag, M_min, M_max, sigma_coeff)
    blurred_adaptive = adaptive_gaussian_blur(img, local_sigma)
    down_adaptive = adaptive_downsample(blurred_adaptive, local_M)

    # 全局对照组
    M_global = 4
    sigma_global = sigma_coeff * M_global
    down_global = global_downsample(img, M_global, sigma_global)

    # 计算自适应和全局的误差热力图
    error_adaptive = compute_error_map(img, down_adaptive)
    error_global  = compute_error_map(img, down_global)

    # 绘图
    plt.figure(figsize=(16, 12))

    # 原图、梯度、局部M、局部σ
    plt.subplot(2, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'{title_prefix} Original')
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.imshow(grad_mag, cmap='gray')
    plt.title('Gradient Magnitude')
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(local_M, cmap='gray')
    plt.title('Local M Map')
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.imshow(local_sigma, cmap='gray')
    plt.title('Local σ Map')
    plt.axis('off')

    # 自适应结果、全局结果、自适应误差、全局误差
    plt.subplot(2, 4, 5)
    plt.imshow(down_adaptive, cmap='gray')
    plt.title('Adaptive Downsample')
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.imshow(down_global, cmap='gray')
    plt.title(f'Global Downsample (M={M_global})')
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.imshow(error_adaptive, cmap='hot')
    plt.title('Adaptive Error Map')
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.imshow(error_global, cmap='hot')
    plt.title('Global Error Map')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    plt.close()
    print(f"已保存: {save_name}")

# ===================== 主程序 =====================
if __name__ == "__main__":
    img_checker = generate_checkerboard(256)
    img_chirp = generate_chirp(256)

    process(img_checker, "Checkerboard", "adapt_checker.png")
    process(img_chirp, "Chirp", "adapt_chirp.png")