# 2023100066-11
2023100066自动化李智阳+实验作业11

# 基于 MediaPipe Pose 与骨架序列 Transformer 的羽毛球击球动作识别
## 1. 项目目的
1. 理解视频动作识别任务如何转化为人体骨架时间序列分类任务。
2. 掌握使用 MediaPipe Pose 从羽毛球击球视频中提取人体 33 个关键点的方法。
3. 将每帧 33 个关键点的 x、y、z、visibility 展平为 132 维特征。
4. 将不同长度的视频统一重采样为固定长度骨架序列，形成 `[30, 132]` 的输入数据。
5. 搭建轻量级 Skeleton Transformer，利用 Transformer Encoder 学习击球动作的时序特征。
6. 完成数据预处理、模型训练、测试集评估和单视频样本推理。
7. 分析训练曲线、测试准确率和单样本推理结果，理解模型优点与局限。
## 2. 运行环境
- 操作系统：Ubuntu / WSL Ubuntu / Windows
- Python 版本：Python 3
- 开发工具：VS Code
- 主要依赖库：
  ```bash
  pip install opencv-python mediapipe==0.10.13 numpy scikit-learn torch matplotlib pandas
  ```
说明：本实验代码使用 `mp.solutions.pose.Pose`，因此建议使用 `mediapipe==0.10.13`，避免新版 MediaPipe 移除旧版 Solutions API 后出现兼容问题。
## 3. 数据集说明
1. 数据集来源：Kaggle badminton_storke_video 数据集。
2. 数据集包含 6 类羽毛球击球动作视频。
3. 本次实验实际读取并处理视频总数为 836 个。
4. 按照 `test_size = 0.2` 划分训练集与测试集。
5. 划分结果：
   - 训练集：668 个样本
   - 测试集：168 个样本
6. 每个样本的骨架序列形状为：
   ```text
   [30, 132]
   ```
   其中 30 表示每个视频统一重采样后的帧数，132 表示每帧 33 个关键点乘以 4 个特征。
## 4. 主要功能
1. 视频读取：递归遍历数据集目录，读取 `.mp4`、`.avi`、`.mov`、`.mkv` 等视频文件。
2. 标签识别：根据视频所在文件夹名称自动推断动作类别标签。
3. 骨架提取：使用 MediaPipe Pose 提取每帧人体 33 个姿态关键点。
4. 骨架归一化：以左右髋部中心为原点，以肩宽为尺度因子进行归一化。
5. 缺失帧处理：对未检测到人体的帧，采用上一帧有效骨架进行填补。
6. 序列重采样：将不同长度的视频统一重采样为 30 帧。
7. 数据保存：生成 `X_train.npy`、`y_train.npy`、`X_test.npy`、`y_test.npy` 等文件。
8. 模型训练：使用 Skeleton Transformer 完成 6 类击球动作分类。
9. 测试评估：输出测试准确率、混淆矩阵和分类报告。
10. 单视频推理：输入单个视频，输出预测类别和置信度。
11. 训练曲线绘制：根据 `train_history.csv` 绘制 loss 曲线和 accuracy 曲线。
## 5. 文件结构说明
```text
实验11/
├── archive/                         # Kaggle 数据集解压后的文件夹
│   ├── forehand_drive/
│   ├── forehand_lift/
│   ├── forehand_net_shot/
│   ├── forehand_clear/
│   ├── backhand_drive/
│   └── backhand_net_shot/
├── processed/                       # 预处理与训练结果文件夹
│   ├── X_train.npy                  # 训练集骨架序列
│   ├── y_train.npy                  # 训练集标签
│   ├── X_test.npy                   # 测试集骨架序列
│   ├── y_test.npy                   # 测试集标签
│   ├── label_map.json               # 类别编号与类别名称对应关系
│   ├── preprocess_meta.json         # 预处理信息记录
│   ├── train_history.csv            # 训练过程记录
│   └── best_model.pt                # 最佳模型权重
├── 结果曲线/
│   ├── accuracy_curve.png           # 训练集与测试集准确率曲线
│   └── loss_curve.png               # 训练集与测试集损失曲线
├── task11.py                        # 实验主代码
├── plot.py                          # 绘制训练曲线代码
└── README.md                        # 项目说明文件
```

## 6. 核心代码与说明

### 6.1 视频骨架提取
```python
def extract_skeleton_sequence(video_path, target_frames=30, max_source_frames=0):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_indices = build_frame_indices(total_frames, max_source_frames)

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = []
    frame_id = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if selected_indices is not None and frame_id not in selected_indices:
            frame_id += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        vec = pose_landmarks_to_vector(results)
        vec = normalize_skeleton_frame(vec)
        frames.append(vec)

        frame_id += 1

    cap.release()
    pose.close()

    seq = np.stack(frames).astype(np.float32)
    seq = fill_missing_frames(seq)
    seq = resample_sequence(seq, target_frames)
    return seq
```

### 6.2 骨架归一化
```python
def normalize_skeleton_frame(vec, eps=1e-6):
    pts = vec.reshape(33, 4).copy()

    hip_center = (pts[23, :3] + pts[24, :3]) / 2.0
    shoulder_dist = np.linalg.norm(pts[11, :2] - pts[12, :2])
    scale = max(float(shoulder_dist), eps)

    pts[:, :3] = (pts[:, :3] - hip_center) / scale
    return pts.reshape(-1).astype(np.float32)
```

### 6.3 Skeleton Transformer 模型
```python
class SkeletonTransformer(nn.Module):
    def __init__(self, input_dim=132, target_frames=30, num_classes=6,
                 d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, target_frames + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        B, T, D = x.shape
        x = self.input_proj(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :T + 1, :]

        x = self.encoder(x)
        feat = x[:, 0, :]
        logits = self.classifier(feat)
        return logits
```

### 6.4 训练与测试流程
```python
for epoch in range(1, epochs + 1):
    model.train()
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
```

### 6.5 单视频推理
```python
seq = extract_skeleton_sequence(video_path, target_frames=30)
x = torch.from_numpy(seq).unsqueeze(0).to(device)

logits = model(x)
probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

pred_id = int(np.argmax(probs))
confidence = float(probs[pred_id])
print("Predicted class:", label_map[pred_id])
print("Confidence:", confidence)
```

## 7. 核心参数说明
1. `target_frames`：30，每个视频统一重采样后的帧数。
2. `input_dim`：132，每帧骨架特征维度，即 33 个关键点 × 4 个特征。
3. `num_classes`：6，羽毛球击球动作类别数。
4. `d_model`：128，Transformer 的主特征维度。
5. `nhead`：4，多头注意力头数。
6. `num_layers`：2，Transformer Encoder 层数。
7. `dim_feedforward`：256，前馈网络隐藏层维度。
8. `dropout`：0.1，用于缓解过拟合。
9. `batch_size`：16。
10. `epochs`：50。
11. `optimizer`：Adam。
12. `learning_rate`：1e-3。
13. `weight_decay`：1e-4。
14. `test_size`：0.2。
15. `max_source_frames`：60，用于加快预处理速度。

## 8. 运行步骤
### 8.1 进入实验目录
```bash
cd "/home/lzy/cv-course/build/实验11"
```
### 8.2 安装依赖
```bash
pip install opencv-python mediapipe==0.10.13 numpy scikit-learn torch matplotlib pandas
```
### 8.3 预处理数据
```bash
python task11.py preprocess \
  --data-root "./archive" \
  --out-dir "./processed" \
  --target-frames 30 \
  --test-size 0.2 \
  --max-source-frames 60
```
运行完成后会生成：
```text
X_train.npy
y_train.npy
X_test.npy
y_test.npy
label_map.json
preprocess_meta.json
```
### 8.4 训练模型
```bash
python task11.py train \
  --data-dir "./processed" \
  --epochs 50 \
  --batch-size 16 \
  --augment
```
运行完成后会生成：
```text
best_model.pt
train_history.csv
```
### 8.5 单视频推理
```bash
python task11.py infer \
  --video "./archive/forehand_net_shot/071.mp4" \
  --ckpt "./processed/best_model.pt"
```
### 8.6 绘制训练曲线
```bash
python plot.py
```
运行完成后会生成：
```text
结果曲线/accuracy_curve.png
结果曲线/loss_curve.png
```
## 9. 实验结果与分析
### 9.1 预处理结果
程序共处理 836 个视频样本，并按照 8:2 的比例划分训练集与测试集：
```text
X_train: (668, 30, 132)
y_train: (668,)
X_test : (168, 30, 132)
y_test : (168,)
```
说明每个视频样本已经被成功转换为固定长度的骨架时间序列，符合 Skeleton Transformer 的输入要求。
### 9.2 测试集结果
在训练 50 轮并加入轻微骨架噪声增强后，模型在测试集上的结果为：
```text
Accuracy = 0.6012
Macro F1-score = 0.5852
Weighted F1-score = 0.5987
```
说明模型能够学习到一定的击球动作时序特征，但仍存在部分类别混淆。
### 9.3 单视频推理结果
选取视频：
```text
./archive/forehand_net_shot/071.mp4
```
模型输出结果为：
```text
Predicted class: forehand net shot
Confidence: 0.8631
```
该视频真实类别为 `forehand net shot`，模型预测正确，说明完整推理流程可以正常运行。
### 9.4 训练曲线分析
从准确率曲线可以看出，训练集准确率随 epoch 增加持续上升，测试集准确率整体也呈上升趋势，最终达到约 60.12%。从 loss 曲线可以看出，训练集损失持续下降，而测试集损失在训练后期出现波动并有所升高，说明模型存在一定过拟合现象。
产生过拟合的可能原因包括：
1. 数据集规模相对有限。
2. 不同羽毛球击球动作之间骨架变化相似。
3. 部分视频中人体姿态检测可能存在误差。
4. 模型在训练后期对训练集样本记忆增强。
## 10. 实验结论
1. 本实验成功将羽毛球击球视频识别任务转化为人体骨架时间序列分类任务。
2. MediaPipe Pose 能够有效提取人体关键点，大幅减少原始视频像素输入带来的计算量。
3. 通过归一化和重采样处理，不同长度的视频可以统一转换为 `[30, 132]` 的固定输入。
4. Skeleton Transformer 能够学习骨架序列中的时序动作特征，并完成 6 类击球动作分类。
5. 最终模型在测试集上取得 60.12% 的准确率，单视频推理样本预测正确，置信度为 0.8631。
6. 训练曲线表明模型具有一定学习能力，但测试损失后期升高，说明仍存在过拟合问题。
7. 后续可通过更充分的数据增强、早停策略、骨架可视化和 attention 分析进一步提升模型效果。
## 11. 作者信息
1. 作者：李智阳
2. 学号：2023100066
3. 专业：自动化
4. 日期：2026年6月12日

