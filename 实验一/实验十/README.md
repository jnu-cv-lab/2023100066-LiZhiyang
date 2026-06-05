# 2023100066-6
2023100066自动化李智阳+实验作业10
# Sinusoidal PE 与 RoPE 原理与验证

## 1. 项目目的
1. 理解绝对位置编码与相对位置编码的区别，掌握Sinusoidal Position Encoding的数学原理。
2. 实现二维向量旋转变换，理解RoPE的几何基础。
3. 实现高维RoPE编码，完成向量位置信息注入。
4. 对比E+pos与RoPE的输入方式与核心差异。
5. 通过数值实验验证RoPE相对位置不变性：注意力得分仅依赖token相对距离。
6. 可视化位置编码分布、向量旋转效果、注意力得分矩阵，完成实验结论分析。

## 2. 运行环境
- 操作系统：Linux / Windows
- Python 3
- 安装依赖库：pip install numpy matplotlib

## 3. 主要功能
1. Sinusoidal PE：生成正余弦绝对位置编码，可视化热力图与维度变化曲线。
2. 二维向量旋转：实现 RoPE 基础旋转操作，可视化旋转前后向量变化。
3. 高维RoPE：基于复数旋转实现高维向量旋转位置编码。
4. 编码方式对比：文字+逻辑对比E+pos与RoPE的核心差异。
5. 相对位置验证：计算RoPE注意力得分矩阵，验证同相对位置得分近似不变。
6. 全流程可视化：输出3张标准实验图，无中文乱码、全英文标题。

## 4. 核心代码与说明

### 4.1 Sinusoidal Position Encoding 实现
```python
# 生成正弦绝对位置编码
def sinusoidal_position_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)  # 偶数维 sin
    pe[:, 1::2] = np.cos(position * div_term)  # 奇数维 cos
    return pe
```

### 4.2 二维向量旋转
```python
# 二维旋转公式：不改变模长，仅改变方向
def rotate_2d(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return x*c - y*s, x*s + y*c
```

### 4.3 高维 RoPE 实现
```python
# 高维向量分组→复数表示→旋转→还原实数
def rope_forward(x, seq_len, d_model):
    angles = 1.0 / (10000 ** (np.arange(0, d_model, 2) / d_model))
    angles = np.outer(np.arange(seq_len), angles)
    
    x2 = x.reshape(seq_len, -1, 2)
    x_complex = x2[..., 0] + 1j * x2[..., 1]
    rot = np.exp(1j * angles)
    x_rot = x_complex * rot
    
    return np.stack([x_rot.real, x_rot.imag], axis=-1).reshape(seq_len, d_model)
```

### 4.4 RoPE 相对位置不变性验证
```python
# 计算 RoPE 编码后 Q*K^T 得分矩阵
x_rope = rope_forward(x, seq_len, d_model)
score = x_rope @ x_rope.T

# 绘制相同相对位置的得分曲线，验证数值近似不变
for k in [1,2,3]:
    vals = [score[i, i+k] for i in range(seq_len - k)]
    plt.plot(vals, 'o-', label=f"rel pos {k}")
```

## 5. 核心参数说明
1. seq_len：序列长度
2. d_model：嵌入向量维度
3. 位置编码基数：10000
4. 随机种子：42
5. 相对位置验证：取k=1,2,3三条曲线展示不变性

## 6. 运行步骤
1. 安装依赖：pip install numpy matplotlib。
2. 分别利用Linux指令touch task10.py创建对应任务的Python文件。
3. 在Ubuntu中激活开发环境，并利用python3 task10.py进行下方运行文件的步骤。
5. 依次查看：正弦PE 图 → 二维旋转图 → RoPE相对位置验证图。
6. 查看控制台输出与保存图片，完成实验结果记录。

## 7. 结果与分析
1. 偶数维sin、奇数维cos，维度越高周期越长，属于绝对位置编码。
2. 二维旋转：RoPE几何基础，旋转不改变向量模长，仅注入位置角度信息。
3. 高维RoPE通过复数旋转实现高效相对位置编码。
4. E+pos vs RoPE：加法编码破坏语义；旋转编码保留原始信息、解耦内容与位置。
5. 相同相对距离token得分近似不变，验证RoPE核心特性。
6. RoPE在语义保留、长文本外推、相对位置建模上优于传统E+pos。

## 8. 作者信息
1. 作者：李智阳
2. 日期：2026年6月5日
