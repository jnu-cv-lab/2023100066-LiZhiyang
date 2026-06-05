import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ====================== 全局绘图设置======================
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 1. 正弦位置编码 Sinusoidal PE ======================
# 生成绝对位置编码，用于Transformer加法式位置嵌入
def sinusoidal_position_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]       # 位置维度 [seq_len, 1]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))  # 频率分母
    pe[:, 0::2] = np.sin(position * div_term)          # 偶数维度用 sin
    pe[:, 1::2] = np.cos(position * div_term)          # 奇数维度用 cos
    return pe

# 生成并绘制正弦位置编码
seq_len, d_model = 30, 32
pe = sinusoidal_position_encoding(seq_len, d_model)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(pe, cmap='RdBu_r', aspect='auto')
plt.title("Sinusoidal Position Encoding")
plt.xlabel("Dimension")
plt.ylabel("Position")

plt.subplot(1, 2, 2)
plt.plot(pe[:, 0], label="dim 0 (sin)")
plt.plot(pe[:, 1], label="dim 1 (cos)")
plt.title("PE Curves")
plt.xlabel("Position")
plt.legend()
plt.tight_layout()
plt.savefig("1_sinusoidal_pe.png", dpi=150)
plt.show()

# ====================== 2. 二维向量旋转 2D Rotation ======================
# RoPE 的基础几何操作，旋转不改变模长，只改变方向
def rotate_2d(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return x*c - y*s, x*s + y*c

# 演示向量旋转效果
x, y = 1.0, 1.0
theta = np.pi / 4
xr, yr = rotate_2d(x, y, theta)

plt.figure(figsize=(5, 5))
plt.quiver(0, 0, x, y, color='blue', label='Original', scale=4)
plt.quiver(0, 0, xr, yr, color='red', label='Rotated', scale=4)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.grid(alpha=0.3)
plt.title("2D Vector Rotation")
plt.legend()
plt.savefig("2_2d_rotation.png", dpi=150)
plt.show()

# ====================== 3. 高维 RoPE 实现 ======================
# 对高维向量分组旋转，实现相对位置编码
# 将 d_model 维向量两两分组，用复数乘法实现旋转
def rope_forward(x, seq_len, d_model):
    # 计算各维度旋转角度
    angles = 1.0 / (10000 ** (np.arange(0, d_model, 2) / d_model))
    angles = np.outer(np.arange(seq_len), angles)
    
    # 转为复数形式，逐组旋转
    x2 = x.reshape(seq_len, -1, 2)
    x_complex = x2[..., 0] + 1j * x2[..., 1]
    rot = np.exp(1j * angles)          # 复数旋转算子
    x_rot = x_complex * rot            # 执行旋转
    
    # 转回实数向量并返回
    return np.stack([x_rot.real, x_rot.imag], axis=-1).reshape(seq_len, d_model)
# ====================== 5. RoPE 相对位置不变性验证 ======================
# 相同相对距离的 token，注意力分数几乎相同
seq_len, d_model = 20, 16
np.random.seed(42)
x = np.random.randn(seq_len, d_model)
x_rope = rope_forward(x, seq_len, d_model)
score = x_rope @ x_rope.T       # 计算 Q*K^T 注意力分数

plt.figure(figsize=(10, 4.5))
plt.subplot(1, 2, 1)
plt.imshow(score, cmap='coolwarm')
plt.title("RoPE Score Matrix")
plt.xlabel("Position j")
plt.ylabel("Position i")

plt.subplot(1, 2, 2)
# 绘制相同相对位置的得分曲线
for k in [1,2,3]:
    vals = [score[i, i+k] for i in range(seq_len - k)]
    plt.plot(vals, 'o-', label=f"rel pos {k}")
plt.title("Same Relative Position = Similar Score")
plt.legend()
plt.tight_layout()
plt.savefig("3_rope_property.png", dpi=150)
plt.show()