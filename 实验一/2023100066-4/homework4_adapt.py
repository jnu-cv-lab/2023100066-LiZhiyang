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

# SSIM
def compute_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1*img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

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
    # 梯度大→M小，梯度小→M大
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
            # 取局部邻域做滤波，保证尺寸一致
            y1, y2 = max(0, i-2), min(h, i+3)
            x1, x2 = max(0, j-2), min(w, j+3)
            patch = img[y1:y2, x1:x2]
            blurred[i, j] = cv2.GaussianBlur(patch, ksize, sigma)[2, 2]
    return blurred.astype(np.uint8)

# 自适应下采样
def adaptive_downsample(img, local_M):
    h, w = img.shape[:2]
    # 统一用M_max计算输出尺寸，保证形状一致
    new_h, new_w = h // M_max, w // M_max
    downsampled = np.zeros((new_h, new_w), dtype=np.uint8)
    
    # 遍历输出图的每个像素，从原图对应区域采样
    for i in range(new_h):
        for j in range(new_w):
            # 原图对应区域的坐标
            y_start = i * M_max
            y_end = y_start + M_max
            x_start = j * M_max
            x_end = x_start + M_max
            
            # 取该区域的平均M值
            region_M = local_M[y_start:y_end, x_start:x_end].mean()
            M = int(round(region_M))
            if M < M_min:
                M = M_min
            if M > M_max:
                M = M_max
            
            # 从区域中按M下采样，取中心像素（简单稳定）
            region = img[y_start:y_end, x_start:x_end]
            downsampled[i, j] = region[::M, ::M].mean()
    return downsampled

# 全局统一下采样
def global_downsample(img, M, sigma):
    ksize = (2*int(4*sigma)+1, 2*int(4*sigma)+1)
    blurred = cv2.GaussianBlur(img, ksize, sigma)
    return blurred[::M, ::M]

# 计算误差图和量化指标
def compute_error_and_metrics(original, downsampled, M_global):
    h, w = original.shape[:2]
    # 上采样回原尺寸
    upsampled = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_CUBIC)
    # 误差图
    error_map = cv2.absdiff(original, upsampled)
    # MSE
    mse = np.mean((original - upsampled)**2)
    # PSNR
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    # SSIM
    ssim = compute_ssim(original, upsampled)
    return error_map, mse, psnr, ssim

# 统一处理函数
def process(img, title_prefix, save_name):
    grad_mag = compute_gradient(img)
    local_M, local_sigma = generate_local_M_sigma(grad_mag, M_min, M_max, sigma_coeff)
    blurred_adaptive = adaptive_gaussian_blur(img, local_sigma)
    down_adaptive = adaptive_downsample(blurred_adaptive, local_M)

    # 全局对照组（M=4，σ=1.8）
    M_global = 4
    sigma_global = sigma_coeff * M_global
    down_global = global_downsample(img, M_global, sigma_global)

    # 计算误差和指标
    error_adaptive, mse_adaptive, psnr_adaptive, ssim_adaptive = compute_error_and_metrics(img, down_adaptive, M_global)
    error_global, mse_global, psnr_global, ssim_global = compute_error_and_metrics(img, down_global, M_global)

    # 绘图
    plt.figure(figsize=(16, 12))

    # 原图、梯度图、局部M图、局部σ图
    plt.subplot(3,4,1)
    plt.imshow(img, cmap='gray')
    plt.title(f'{title_prefix} Original')
    plt.axis('off')

    plt.subplot(3,4,2)
    plt.imshow(grad_mag, cmap='gray')
    plt.title('Gradient Magnitude')
    plt.axis('off')

    plt.subplot(3,4,3)
    plt.imshow(local_M, cmap='gray')
    plt.title('Local M Map')
    plt.axis('off')

    plt.subplot(3,4,4)
    plt.imshow(local_sigma, cmap='gray')
    plt.title('Local σ Map')
    plt.axis('off')

    # 自适应下采样、全局下采样、自适应误差图、全局误差图
    plt.subplot(3,4,5)
    plt.imshow(down_adaptive, cmap='gray')
    plt.title('Adaptive Downsample')
    plt.axis('off')

    plt.subplot(3,4,6)
    plt.imshow(down_global, cmap='gray')
    plt.title(f'Global Downsample (M={M_global})')
    plt.axis('off')

    plt.subplot(3,4,7)
    plt.imshow(error_adaptive, cmap='hot')
    plt.title(f'Adaptive Error Map\nPSNR={psnr_adaptive:.2f}dB')
    plt.axis('off')

    plt.subplot(3,4,8)
    plt.imshow(error_global, cmap='hot')
    plt.title(f'Global Error Map\nPSNR={psnr_global:.2f}dB')
    plt.axis('off')

    # 第三行：指标对比
    metrics = ['MSE', 'PSNR (dB)', 'SSIM']
    adaptive_vals = [mse_adaptive, psnr_adaptive, ssim_adaptive]
    global_vals = [mse_global, psnr_global, ssim_global]
    x = np.arange(len(metrics))
    width = 0.35

    plt.subplot(3,4,9)
    plt.bar(x - width/2, adaptive_vals, width, label='Adaptive')
    plt.bar(x + width/2, global_vals, width, label='Global')
    plt.xticks(x, metrics)
    plt.title('Quantitative Metrics Comparison')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    plt.close()
    print(f"已保存: {save_name}")

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 生成两张测试图
    img_checker = generate_checkerboard(256)
    img_chirp = generate_chirp(256)

    # 分别处理，生成两张独立结果图
    process(img_checker, "Checkerboard", "adapt_checker.png")
    process(img_chirp, "Chirp", "adapt_chirp.png")
