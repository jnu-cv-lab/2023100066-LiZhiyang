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
- Linux/wsl
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

### 4.1 下采样
```python
# 方案1：无滤波直接下采样
img_down_no_filter = cv2.resize(img_original, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
# 方案2：先高斯滤波再下采样
img_blur = cv2.GaussianBlur(img_original, (5, 5), sigmaX=1.5)  # 高斯平滑
img_down_with_filter = cv2.resize(img_blur, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
```

### 4.2 图像恢复（三种插值方法）
```python
# 无预滤波的恢复（基于 img_down_no_filter 放大到原尺寸）
img_up_nn_no_filter = cv2.resize(img_down_no_filter, (w, h), interpolation=cv2.INTER_NEAREST)  # 最近邻
img_up_bilinear_no_filter = cv2.resize(img_down_no_filter, (w, h), interpolation=cv2.INTER_LINEAR)  # 双线性
img_up_bicubic_no_filter = cv2.resize(img_down_no_filter, (w, h), interpolation=cv2.INTER_CUBIC)  # 双三次
# 有预滤波的恢复（基于 img_down_with_filter 放大到原尺寸）
img_up_nn_with_filter = cv2.resize(img_down_with_filter, (w, h), interpolation=cv2.INTER_NEAREST)
img_up_bilinear_with_filter = cv2.resize(img_down_with_filter, (w, h), interpolation=cv2.INTER_LINEAR)
img_up_bicubic_with_filter = cv2.resize(img_down_with_filter, (w, h), interpolation=cv2.INTER_CUBIC)
```

### 4.3 计算MSE和PSNR
```python
def calculate_mse_psnr(original, restored):​
    original = original.astype(np.float64)​
    restored = restored.astype(np.float64)​
    mse = np.mean((original - restored) ** 2)​
    psnr = 100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))​
    return mse, psnr
```

### 4.4 FFT 频谱分析（频域可视化）
```python
def fft_analysis(img):​
    img_float = np.float32(img)​
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)  # 离散傅里叶变换​
    dft_shift = np.fft.fftshift(dft)  # 频谱中心化
    magnitude = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1e-8)  # 对数缩放
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # 归一化到 0-255​
    return magnitude_norm
```
### 4.5 DCT 变换分析（能量分布）
```python
def dct_analysis(img):​
    img_float = np.float32(img)​
    dct = cv2.dct(img_float)  # 离散余弦变换​
    dct_log = np.log(np.abs(dct) + 1e-8)  # 对数缩放​
    dct_norm = cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)​
    return dct_norm
```
### 4.6 计算 DCT 低频能量占比
```python
def calculate_dct_energy_ratio(img):​
    img_float = np.float32(img)​
    dct = cv2.dct(img_float)​
    h, w = dct.shape​
    roi_h, roi_w = h // 2, w // 2  # 取左上角 1/4 区域
    roi_energy = np.sum(dct[:roi_h, :roi_w] ** 2)  # 低频能量​
    total_energy = np.sum(dct ** 2)  # 总能量​
    return roi_energy / total_energy if total_energy != 0 else 0.0
```

## 5. 核心参数说明
1. 下采样比例：scale = 0.5,宽、高均缩小为原图的 1/2，面积为原图 1/4
2. 高斯滤波：核大小 (5,5)，标准差 sigmaX=1.5（低通滤波，去除高频混叠）
3. 插值方法：​最近邻插值速度最快，边缘锯齿明显​；双线性插值基于 4 邻域加权，边缘平滑；​双三次插值基于 16 邻域三次卷积，细节保留最好
4. 评价指标：​MSE（均方误差）越小表示恢复图与原图差异越小；​PSNR（峰值信噪比）越大表示恢复质量越好；​DCT能量占比，左上角低频区域能量 / 总能量，反映了图像平滑度

## 6. 运行步骤
1. 将条纹.png 放入 cv-course/build/目录
2. 将核心代码保存为 homework4.py 放入同一目录
3. 在Ubuntu中输入Linux指令 source /home/lzy/cv-course/.venv-basic/bin/activate激活开发环境
4. 在build目录下输入Linux指令 python3 homework4.py运行脚本
5. 查看输出：​控制台：打印各步骤状态、MSE/PSNR 数值、DCT 能量占比​；目录下：生成所有任务对应的结果图。

## 7. 作者信息
1. 作者：李智阳
2. 日期：2026年4月7日
