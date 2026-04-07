import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

# 输出路径
out_dir = "/home/lzy/cv-course/build/"

# ===================== 工具函数 =====================
# 计算MSE均方误差与PSNR峰值信噪比
def calculate_mse_psnr(original, restored):
    original = original.astype(np.float64)
    restored = restored.astype(np.float64)
    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return mse, psnr

# FFT傅里叶变换分析
def fft_analysis(img):
    img_float = np.float32(img)
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)  # 离散傅里叶变换
    dft_shift = np.fft.fftshift(dft)                         # 频谱中心移到图像中心
    magnitude = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1e-8)  # 对数幅度
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return magnitude_norm

# DCT离散余弦变换分析
def dct_analysis(img):
    img_float = np.float32(img)
    dct = cv2.dct(img_float)                                 # 离散余弦变换
    dct_log = np.log(np.abs(dct) + 1e-8)                     # 对数压缩动态范围
    dct_norm = cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return dct_norm

# 计算DCT左上角1/4区域的能量占比
def calculate_dct_energy_ratio(img):
    img_float = np.float32(img)
    dct = cv2.dct(img_float)
    h, w = dct.shape
    roi_h, roi_w = h // 2, w // 2                            # 取左上角1/4区域
    roi_energy = np.sum(dct[:roi_h, :roi_w] ** 2)
    total_energy = np.sum(dct ** 2)
    return roi_energy / total_energy if total_energy != 0 else 0.0

# ===================== 步骤1. 读入灰度图像 =====================
print("\n===== 正在执行步骤1：读入图像 =====")
img_path = "/home/lzy/cv-course/build/条纹.png"
img_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img_original is None:
    print("找不到 条纹.png")
    exit()

h, w = img_original.shape
scale = 0.5
new_w, new_h = int(w * scale), int(h * scale)  # 下采样目标尺寸

# ===================== 步骤2. 下采样 =====================
print("\n===== 正在执行步骤2：下采样 =====")
# 无预滤波直接下采样
img_down_no_filter = cv2.resize(img_original, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
# 高斯滤波抗混叠后下采样
img_blur = cv2.GaussianBlur(img_original, (5, 5), sigmaX=1.5)
img_down_with_filter = cv2.resize(img_blur, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

# ===================== 步骤3. 采样恢复（3种插值方法） =====================
print("\n===== 正在执行步骤3：图像恢复 =====")
# 最近邻插值
img_up_nn_no_filter = cv2.resize(img_down_no_filter, (w, h), interpolation=cv2.INTER_NEAREST)
img_up_nn_with_filter = cv2.resize(img_down_with_filter, (w, h), interpolation=cv2.INTER_NEAREST)

# 双线性插值
img_up_bilinear_no_filter = cv2.resize(img_down_no_filter, (w, h), interpolation=cv2.INTER_LINEAR)
img_up_bilinear_with_filter = cv2.resize(img_down_with_filter, (w, h), interpolation=cv2.INTER_LINEAR)

# 双三次插值
img_up_bicubic_no_filter = cv2.resize(img_down_no_filter, (w, h), interpolation=cv2.INTER_CUBIC)
img_up_bicubic_with_filter = cv2.resize(img_down_with_filter, (w, h), interpolation=cv2.INTER_CUBIC)

# ===================== 步骤4. 空间域评价（MSE/PSNR） =====================
print("\n===== 正在执行步骤4：空间域比较 =====")
print("\n===== 无预滤波 =====")
mse_nn_no, psnr_nn_no = calculate_mse_psnr(img_original, img_up_nn_no_filter)
mse_bilinear_no, psnr_bilinear_no = calculate_mse_psnr(img_original, img_up_bilinear_no_filter)
mse_bicubic_no, psnr_bicubic_no = calculate_mse_psnr(img_original, img_up_bicubic_no_filter)
print(f"最近邻 MSE={mse_nn_no:.2f} PSNR={psnr_nn_no:.2f}")
print(f"双线性 MSE={mse_bilinear_no:.2f} PSNR={psnr_bilinear_no:.2f}")
print(f"双三次 MSE={mse_bicubic_no:.2f} PSNR={psnr_bicubic_no:.2f}")

print("\n===== 有预滤波 =====")
mse_nn_with, psnr_nn_with = calculate_mse_psnr(img_original, img_up_nn_with_filter)
mse_bilinear_with, psnr_bilinear_with = calculate_mse_psnr(img_original, img_up_bilinear_with_filter)
mse_bicubic_with, psnr_bicubic_with = calculate_mse_psnr(img_original, img_up_bicubic_with_filter)
print(f"最近邻 MSE={mse_nn_with:.2f} PSNR={psnr_nn_with:.2f}")
print(f"双线性 MSE={mse_bilinear_with:.2f} PSNR={psnr_bilinear_with:.2f}")
print(f"双三次 MSE={mse_bicubic_with:.2f} PSNR={psnr_bicubic_with:.2f}")

# 保存下采样对比图
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(img_original, cmap='gray'), plt.title("Original"), plt.axis('off')
plt.subplot(132), plt.imshow(img_down_no_filter, cmap='gray'), plt.title("No filter"), plt.axis('off')
plt.subplot(133), plt.imshow(img_down_with_filter, cmap='gray'), plt.title("Gaussian blur"), plt.axis('off')
plt.tight_layout()
plt.savefig(out_dir + "任务2_下采样对比.png", dpi=150, bbox_inches='tight')
print("\n已保存: 任务2_下采样对比.png")

# 保存无滤波恢复对比图
plt.figure(figsize=(15, 10))
plt.subplot(231), plt.imshow(img_original, cmap='gray'), plt.title("Original"), plt.axis('off')
plt.subplot(232), plt.imshow(img_up_nn_no_filter, cmap='gray'), plt.title("NN"), plt.axis('off')
plt.subplot(233), plt.imshow(img_up_bilinear_no_filter, cmap='gray'), plt.title("Bilinear"), plt.axis('off')
plt.subplot(234), plt.imshow(img_up_bicubic_no_filter, cmap='gray'), plt.title("Bicubic"), plt.axis('off')
plt.subplot(235), plt.imshow(cv2.absdiff(img_original, img_up_nn_no_filter), cmap='gray'), plt.title("NN Diff"), plt.axis('off')
plt.subplot(236), plt.imshow(cv2.absdiff(img_original, img_up_bicubic_no_filter), cmap='gray'), plt.title("Bicubic Diff"), plt.axis('off')
plt.tight_layout()
plt.savefig(out_dir + "任务3_无滤波恢复对比.png", dpi=150, bbox_inches='tight')
print("已保存: 任务3_无滤波恢复对比.png")

# 保存有滤波恢复对比图
plt.figure(figsize=(15, 10))
plt.subplot(231), plt.imshow(img_original, cmap='gray'), plt.title("Original"), plt.axis('off')
plt.subplot(232), plt.imshow(img_up_nn_with_filter, cmap='gray'), plt.title("NN"), plt.axis('off')
plt.subplot(233), plt.imshow(img_up_bilinear_with_filter, cmap='gray'), plt.title("Bilinear"), plt.axis('off')
plt.subplot(234), plt.imshow(img_up_bicubic_with_filter, cmap='gray'), plt.title("Bicubic"), plt.axis('off')
plt.subplot(235), plt.imshow(cv2.absdiff(img_original, img_up_nn_with_filter), cmap='gray'), plt.title("NN Diff"), plt.axis('off')
plt.subplot(236), plt.imshow(cv2.absdiff(img_original, img_up_bicubic_with_filter), cmap='gray'), plt.title("Bicubic Diff"), plt.axis('off')
plt.tight_layout()
plt.savefig(out_dir + "任务3_有滤波恢复对比.png", dpi=150, bbox_inches='tight')
print("已保存: 任务3_有滤波恢复对比.png")

# 单独保存所有单张结果图
cv2.imwrite(out_dir + "任务1_原图.png", img_original)
cv2.imwrite(out_dir + "任务2_下采样无滤波.png", img_down_no_filter)
cv2.imwrite(out_dir + "任务2_下采样有滤波.png", img_down_with_filter)

cv2.imwrite(out_dir + "任务3_恢复_最近邻_无滤波.png", img_up_nn_no_filter)
cv2.imwrite(out_dir + "任务3_恢复_双线性_无滤波.png", img_up_bilinear_no_filter)
cv2.imwrite(out_dir + "任务3_恢复_双三次_无滤波.png", img_up_bicubic_no_filter)

cv2.imwrite(out_dir + "任务3_恢复_最近邻_有滤波.png", img_up_nn_with_filter)
cv2.imwrite(out_dir + "任务3_恢复_双线性_有滤波.png", img_up_bilinear_with_filter)
cv2.imwrite(out_dir + "任务3_恢复_双三次_有滤波.png", img_up_bicubic_with_filter)

# ===================== 步骤5：FFT频域分析 =====================
print("\n===== 正在执行步骤5：傅里叶变换分析 =====")

fft_original = fft_analysis(img_original)
fft_down_no = fft_analysis(img_down_no_filter)
fft_down_with = fft_analysis(img_down_with_filter)

fft_up_bilinear_no = fft_analysis(img_up_bilinear_no_filter)
fft_up_bilinear_with = fft_analysis(img_up_bilinear_with_filter)

# 下采样前后FFT对比
plt.figure(figsize=(18,6))
plt.subplot(131), plt.imshow(fft_original, cmap='gray'), plt.title("FFT Original"), plt.axis('off')
plt.subplot(132), plt.imshow(fft_down_no, cmap='gray'), plt.title("FFT Down No Filter"), plt.axis('off')
plt.subplot(133), plt.imshow(fft_down_with, cmap='gray'), plt.title("FFT Down With Gaussian"), plt.axis('off')
plt.tight_layout()
plt.savefig(out_dir + "任务5_FFT频谱对比.png", dpi=150, bbox_inches='tight')

# 恢复前后FFT对比
plt.figure(figsize=(18,6))
plt.subplot(131), plt.imshow(fft_down_no, cmap='gray'), plt.title("Down No Filter"), plt.axis('off')
plt.subplot(132), plt.imshow(fft_up_bilinear_no, cmap='gray'), plt.title("Restore Bilinear No Filter"), plt.axis('off')
plt.subplot(133), plt.imshow(fft_up_bilinear_with, cmap='gray'), plt.title("Restore Bilinear With Filter"), plt.axis('off')
plt.tight_layout()
plt.savefig(out_dir + "任务5_FFT恢复频谱对比.png", dpi=150, bbox_inches='tight')

# 保存FFT单张图
cv2.imwrite(out_dir + "任务5_FFT_原图.png", fft_original)
cv2.imwrite(out_dir + "任务5_FFT_下采样无滤波.png", fft_down_no)
cv2.imwrite(out_dir + "任务5_FFT_下采样有滤波.png", fft_down_with)
cv2.imwrite(out_dir + "任务5_FFT_恢复无滤波.png", fft_up_bilinear_no)
cv2.imwrite(out_dir + "任务5_FFT_恢复有滤波.png", fft_up_bilinear_with)

# ===================== 步骤6：DCT变换分析与低频能量占比统计 =====================
print("\n===== 正在执行步骤6：DCT变换分析 =====")

dct_original = dct_analysis(img_original)
dct_down_no = dct_analysis(img_down_no_filter)
dct_down_with = dct_analysis(img_down_with_filter)

# DCT频谱对比图
plt.figure(figsize=(18,6))
plt.subplot(131), plt.imshow(dct_original, cmap='gray'), plt.title("DCT Original"), plt.axis('off')
plt.subplot(132), plt.imshow(dct_down_no, cmap='gray'), plt.title("DCT Down No Filter"), plt.axis('off')
plt.subplot(133), plt.imshow(dct_down_with, cmap='gray'), plt.title("DCT Down With Gaussian"), plt.axis('off')
plt.tight_layout()
plt.savefig(out_dir + "任务6_DCT频谱对比.png", dpi=150, bbox_inches='tight')

# 保存DCT单张图
cv2.imwrite(out_dir + "任务6_DCT_原图.png", dct_original)
cv2.imwrite(out_dir + "任务6_DCT_下采样无滤波.png", dct_down_no)
cv2.imwrite(out_dir + "任务6_DCT_下采样有滤波.png", dct_down_with)

# 输出DCT低频能量占比
print("\n===== DCT 左上角能量占比 =====")
energy_original = calculate_dct_energy_ratio(img_original)
energy_down_no = calculate_dct_energy_ratio(img_down_no_filter)
energy_down_with = calculate_dct_energy_ratio(img_down_with_filter)
energy_up_bilinear_no = calculate_dct_energy_ratio(img_up_bilinear_no_filter)
energy_up_bilinear_with = calculate_dct_energy_ratio(img_up_bilinear_with_filter)

print(f"原图: {energy_original:.4f} ({energy_original*100:.2f}%)")
print(f"缩小无滤波: {energy_down_no:.4f} ({energy_down_no*100:.2f}%)")
print(f"缩小有滤波: {energy_down_with:.4f} ({energy_down_with*100:.2f}%)")
print(f"恢复无滤波: {energy_up_bilinear_no:.4f} ({energy_up_bilinear_no*100:.2f}%)")
print(f"恢复有滤波: {energy_up_bilinear_with:.4f} ({energy_up_bilinear_with*100:.2f}%)")