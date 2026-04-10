# 2023100066-3
2023100066自动化李智阳+实验作业4
# 图像下采样与抗混叠实验

## 1. 项目目的
1. 掌握图像下采样中高频信号折叠产生混叠的物理本质，理解抗混叠滤波的必要性
2. 验证经验公式 σ≈0.45M 的正确性，掌握高斯滤波下采样的工程实现方法​
3. 通过傅里叶变换直观分析图像频域特征，验证滤波与下采样对频谱的影响​
4. 实现自适应下采样算法
5. 通过误差热力图直观对比不同下采样方案的优劣

## 2. 运行环境
- Python3
- Linux/Ubuntu
- Matplotlib
- Opencv
- NumPy
  
安装依赖库：
pip install opencv-python numpy matplotlib

## 3. 主要功能
1. 自动生成棋盘格图像和 Chirp 调频测试图，用于观察下采样混叠现象。
2. 实现直接下采样和高斯滤波后下采样两种方式，对比抗混叠效果。
3. 对原图、直接下采样图、高斯下采样图分别计算并显示傅里叶频谱，直观展示混叠与抗混叠差异。
4. 基于图像梯度幅值自动分配局部下采样倍数 M 和高斯滤波 σ：
5. 对比自适应下采样与全局统一下采样的结果，绘制误差热力图，并计算 MSE、PSNR、SSIM 等评价指标。

## 4. 核心代码与说明

### 4.1 下采样与高斯滤波
```python
# 直接下采样
def downsample(img, M):
    return img[::M, ::M]
# 高斯滤波 + 下采样（抗混叠）
def gaussian_downsample(img, M, sigma):
    blurred = cv2.GaussianBlur(img, (5, 5), sigma)
    return downsample(blurred, M)
```

### 4.2 FFT 频谱分析
```python
def get_fft_spectrum(img):
    f = np.fft.fft2(img)          # 傅里叶变换
    f_shift = np.fft.fftshift(f) # 低频移到中心
    magnitude = 20 * np.log(np.abs(f_shift) + 1)
    return magnitude
```

### 4.3 自适应下采样核心
```python
#  计算梯度
def compute_gradient(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    return grad_mag
#  由梯度生成局部 M 和 σ
def generate_local_M_sigma(grad_mag, M_min=2, M_max=4):
    local_M = M_max - (M_max - M_min) * grad_mag
    local_sigma = 0.45 * local_M
    return local_M, local_sigma
```

### 4.4 自适应高斯滤波
```python
def adaptive_gaussian_blur(img, local_sigma):
    h, w = img.shape
    blurred = np.zeros_like(img, dtype=np.float32)
    block = 4
    for i in range(0, h, block):
        for j in range(0, w, block):
            sg = local_sigma[i:i+block, j:j+block].mean()
            ksize = (2*int(4*sg)+1, 2*int(4*sg)+1)
            blurred[i:i+block, j:j+block] = cv2.GaussianBlur(
                img[i:i+block, j:j+block], ksize, sg
            )
    return blurred.astype(np.uint8)
```
### 4.5 自适应下采样
```python
def adaptive_downsample(img, local_M):
    h, w = img.shape
    new_h, new_w = h // 4, w // 4
    out = np.zeros((new_h, new_w), dtype=np.uint8)
    for i in range(new_h):
        for j in range(new_w):
            y = i*4
            x = j*4
            M = int(round(local_M[y:y+4, x:x+4].mean()))
            out[i,j] = img[y:y+4, x:x+4][::M, ::M].mean()
    return out
```
### 4.6 图像质量评价指标
```python
def compute_metrics(original, upsampled):
    mse = np.mean((original - upsampled)**2)
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    ssim = compute_ssim(original, upsampled)
    return mse, psnr, ssim
```

## 5. 核心参数说明
1. 下采样比例：scale = 0.5,宽、高均缩小为原图的 1/2，面积为原图 1/4
2. 高斯滤波：核大小 (5,5)，标准差 sigmaX=1.5（低通滤波，去除高频混叠）
3. 插值方法：​最近邻插值速度最快，边缘锯齿明显​；双线性插值基于 4 邻域加权，边缘平滑；​双三次插值基于 16 邻域三次卷积，细节保留最好
4. 评价指标：​MSE（均方误差）越小表示恢复图与原图差异越小；​PSNR（峰值信噪比）越大表示恢复质量越好；​DCT能量占比，左上角低频区域能量 / 总能量，反映了图像平滑度

## 6. 运行步骤
1. 在ubuntu中打开cv-course/build/目录
2. 在Ubuntu中输入touch homework4.py和touch homework4_adapt.py分别创建两个Python文件
3. 在VScode中编写Python代码
4. 在Ubuntu中输入Linux指令 source /home/lzy/cv-course/.venv-basic/bin/activate激活开发环境
5. 在build目录下输入Linux指令 python3 homework4.py运行脚本
6. 查看输出

## 7. 作者信息
1. 作者：李智阳
2. 日期：2026年4月10日
