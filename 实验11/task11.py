#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skeleton Transformer for Badminton Stroke Video Classification

功能：
1. preprocess：读取 Kaggle 羽毛球击球视频，使用 MediaPipe Pose 提取骨架序列，保存 .npy
2. train：训练 Skeleton Transformer，并输出 accuracy / confusion matrix / classification report
3. infer：对单个视频做推理，输出预测类别和置信度

示例：
python badminton_skeleton_transformer.py preprocess --data-root ./badminton_storke_video --out-dir ./processed
python badminton_skeleton_transformer.py train --data-dir ./processed --epochs 30 --batch-size 16
python badminton_skeleton_transformer.py infer --video path/to/demo.mp4 --ckpt ./processed/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# 训练阶段才需要 torch / sklearn；预处理阶段主要需要 cv2 + mediapipe
try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    nn = None
    Dataset = object
    DataLoader = None

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
except ImportError:
    train_test_split = None
    confusion_matrix = None
    classification_report = None
    accuracy_score = None


# Kaggle 数据集与任务书中的 6 类标签
LABEL_MAP: Dict[int, str] = {
    0: "forehand drive",
    1: "forehand lift",
    2: "forehand net shot",
    3: "forehand clear",
    4: "backhand drive",
    5: "backhand net shot",
}
NAME_TO_LABEL: Dict[str, int] = {v: k for k, v in LABEL_MAP.items()}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


# MediaPipe Pose 的关键点索引
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def normalize_name(s: str) -> str:
    """把文件夹名归一化，便于匹配 backhand_drive / backhand drive 等写法。"""
    return s.lower().replace("_", " ").replace("-", " ").strip()


def infer_label_from_path(video_path: Path) -> Optional[int]:
    """
    根据视频路径中的上级文件夹名称推断标签。
    例如：.../forehand_clear/xxx.mp4 或 .../forehand clear/xxx.mp4
    """
    parts = [normalize_name(p.name) for p in video_path.parents]
    for label_id, label_name in LABEL_MAP.items():
        target = normalize_name(label_name)
        for part in parts:
            if target == part or target in part:
                return label_id
    return None


def find_videos(data_root: Path) -> List[Tuple[Path, int]]:
    videos: List[Tuple[Path, int]] = []
    for p in data_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            label = infer_label_from_path(p)
            if label is not None:
                videos.append((p, label))

    videos = sorted(videos, key=lambda x: str(x[0]))
    if not videos:
        raise RuntimeError(
            f"在 {data_root} 下没有找到可识别标签的视频。请确认目录中包含如下类别文件夹："
            f"{list(LABEL_MAP.values())}"
        )
    return videos


def build_frame_indices(total_frames: int, max_source_frames: int = 0) -> Optional[set]:
    """
    如果 max_source_frames > 0，则从原视频中均匀抽取不超过 max_source_frames 帧来提取姿态，
    可显著加快课堂实验速度。若希望严格逐帧处理，请设置 --max-source-frames 0。
    """
    if total_frames <= 0 or max_source_frames <= 0 or total_frames <= max_source_frames:
        return None
    idx = np.linspace(0, total_frames - 1, max_source_frames).astype(int)
    return set(idx.tolist())


def pose_landmarks_to_vector(results) -> np.ndarray:
    """
    MediaPipe Pose 每帧 33 个关键点，每个关键点 x,y,z,visibility，
    展平成 132 维向量。
    """
    if results.pose_landmarks is None:
        return np.zeros((33 * 4,), dtype=np.float32)

    coords: List[float] = []
    for lm in results.pose_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.asarray(coords, dtype=np.float32)


def normalize_skeleton_frame(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    简单骨架归一化：
    1. 以左右髋部中心为原点；
    2. 以肩宽为尺度；
    3. visibility 不做平移缩放。
    """
    pts = vec.reshape(33, 4).copy()
    if float(np.sum(pts[:, 3])) <= eps:
        return vec.astype(np.float32)

    hip_center = (pts[LEFT_HIP, :3] + pts[RIGHT_HIP, :3]) / 2.0
    shoulder_dist = np.linalg.norm(pts[LEFT_SHOULDER, :2] - pts[RIGHT_SHOULDER, :2])
    scale = max(float(shoulder_dist), eps)

    pts[:, :3] = (pts[:, :3] - hip_center) / scale
    return pts.reshape(-1).astype(np.float32)


def fill_missing_frames(seq: np.ndarray) -> np.ndarray:
    """
    对没有检测到人体的帧做简单填补：
    - 如果中间缺失，使用上一帧有效骨架；
    - 如果开头缺失，使用第一帧有效骨架；
    - 如果整段都没有检测到，保持全 0。
    """
    if len(seq) == 0:
        return seq

    valid = np.sum(seq.reshape(len(seq), 33, 4)[:, :, 3], axis=1) > 1e-6
    if not np.any(valid):
        return seq

    first_valid = int(np.argmax(valid))
    seq[:first_valid] = seq[first_valid]

    last = seq[first_valid].copy()
    for i in range(first_valid, len(seq)):
        if valid[i]:
            last = seq[i].copy()
        else:
            seq[i] = last
    return seq


def resample_sequence(seq: np.ndarray, target_frames: int = 30) -> np.ndarray:
    """
    将不同长度的视频骨架序列重采样成固定长度 [target_frames, 132]。
    """
    if len(seq) == 0:
        return np.zeros((target_frames, 132), dtype=np.float32)

    if len(seq) == 1:
        return np.repeat(seq, target_frames, axis=0).astype(np.float32)

    old_x = np.linspace(0.0, 1.0, num=len(seq))
    new_x = np.linspace(0.0, 1.0, num=target_frames)

    out = np.zeros((target_frames, seq.shape[1]), dtype=np.float32)
    for d in range(seq.shape[1]):
        out[:, d] = np.interp(new_x, old_x, seq[:, d])
    return out.astype(np.float32)


def extract_skeleton_sequence(
    video_path: Path,
    target_frames: int = 30,
    max_source_frames: int = 0,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> np.ndarray:
    """
    从单个视频中提取骨架序列，输出 [target_frames, 132]。
    """
    if mp is None:
        raise ImportError("未安装 mediapipe。请先运行：pip install mediapipe")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_indices = build_frame_indices(total_frames, max_source_frames=max_source_frames)

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    frames: List[np.ndarray] = []
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

    if len(frames) == 0:
        seq = np.zeros((1, 132), dtype=np.float32)
    else:
        seq = np.stack(frames).astype(np.float32)

    seq = fill_missing_frames(seq)
    seq = resample_sequence(seq, target_frames=target_frames)
    return seq


def preprocess(args: argparse.Namespace) -> None:
    if train_test_split is None:
        raise ImportError("未安装 scikit-learn。请先运行：pip install scikit-learn")

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(data_root)
    print(f"共找到 {len(videos)} 个视频。")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    failed: List[str] = []

    for i, (video_path, label) in enumerate(videos, start=1):
        print(f"[{i}/{len(videos)}] label={label} {LABEL_MAP[label]} | {video_path}")
        try:
            seq = extract_skeleton_sequence(
                video_path=video_path,
                target_frames=args.target_frames,
                max_source_frames=args.max_source_frames,
                min_detection_confidence=args.min_detection_confidence,
                min_tracking_confidence=args.min_tracking_confidence,
            )
            X_list.append(seq)
            y_list.append(label)
        except Exception as e:
            print(f"  处理失败：{e}")
            failed.append(str(video_path))

    if len(X_list) == 0:
        raise RuntimeError("没有成功处理任何视频，请检查数据路径和视频格式。")

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    # 分层划分；如果某些类别样本过少导致 stratify 失败，则退化为普通随机划分
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=y,
        )
    except ValueError as e:
        print(f"分层划分失败，改用普通随机划分。原因：{e}")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=None,
        )

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)

    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in LABEL_MAP.items()}, f, ensure_ascii=False, indent=2)

    meta = {
        "data_root": str(data_root),
        "target_frames": args.target_frames,
        "input_dim": 132,
        "num_classes": len(LABEL_MAP),
        "num_total": int(len(X)),
        "num_train": int(len(X_train)),
        "num_test": int(len(X_test)),
        "failed_videos": failed,
    }
    with open(out_dir / "preprocess_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n预处理完成：")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")
    print(f"保存目录：{out_dir}")


class SkeletonNpyDataset(Dataset):
    def __init__(self, X_path: Path, y_path: Path, augment: bool = False, noise_std: float = 0.01):
        self.X = np.load(X_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.int64)
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx].copy()
        y = int(self.y[idx])

        # 轻量数据增强：只在训练时加入很小的高斯噪声
        if self.augment and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=x.shape).astype(np.float32)
            # 不扰动 visibility：每 4 维的最后一维
            noise[:, 3::4] = 0.0
            x = x + noise

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class SkeletonTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 132,
        target_frames: int = 30,
        num_classes: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.target_frames = target_frames
        self.num_classes = num_classes
        self.d_model = d_model
        self.use_cls_token = use_cls_token

        self.input_proj = nn.Linear(input_dim, d_model)

        extra_token = 1 if use_cls_token else 0
        self.pos_embed = nn.Parameter(torch.zeros(1, target_frames + extra_token, d_model))

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

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

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: [B, T, 132]
        B, T, D = x.shape
        if T != self.target_frames:
            raise ValueError(f"输入帧数 T={T} 与模型 target_frames={self.target_frames} 不一致。")

        x = self.input_proj(x)  # [B, T, d_model]

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.pos_embed[:, : T + 1, :]
            x = self.encoder(x)
            feat = x[:, 0, :]
        else:
            x = x + self.pos_embed[:, :T, :]
            x = self.encoder(x)
            feat = x.mean(dim=1)

        logits = self.classifier(feat)
        return logits


def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    preds_all: List[int] = []
    labels_all: List[int] = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * X.size(0)
        preds = torch.argmax(logits, dim=1)

        preds_all.extend(preds.detach().cpu().numpy().tolist())
        labels_all.extend(y.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = accuracy_score(labels_all, preds_all) if accuracy_score else float(np.mean(np.array(labels_all) == np.array(preds_all)))
    return avg_loss, float(acc)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    preds_all: List[int] = []
    labels_all: List[int] = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * X.size(0)
        preds = torch.argmax(logits, dim=1)

        preds_all.extend(preds.cpu().numpy().tolist())
        labels_all.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = accuracy_score(labels_all, preds_all) if accuracy_score else float(np.mean(np.array(labels_all) == np.array(preds_all)))
    return avg_loss, float(acc), np.asarray(labels_all), np.asarray(preds_all)


def train(args: argparse.Namespace) -> None:
    if torch is None:
        raise ImportError("未安装 torch。请先安装 PyTorch。")
    if classification_report is None:
        raise ImportError("未安装 scikit-learn。请先运行：pip install scikit-learn")

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    label_map_path = data_dir / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = {int(k): v for k, v in json.load(f).items()}
    else:
        label_map = LABEL_MAP

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"使用设备：{device}")

    train_ds = SkeletonNpyDataset(data_dir / "X_train.npy", data_dir / "y_train.npy", augment=args.augment)
    test_ds = SkeletonNpyDataset(data_dir / "X_test.npy", data_dir / "y_test.npy", augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SkeletonTransformer(
        input_dim=args.input_dim,
        target_frames=args.target_frames,
        num_classes=args.num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        use_cls_token=args.use_cls_token,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(row)

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt = {
                "model_state": model.state_dict(),
                "config": {
                    "input_dim": args.input_dim,
                    "target_frames": args.target_frames,
                    "num_classes": args.num_classes,
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "num_layers": args.num_layers,
                    "dim_feedforward": args.dim_feedforward,
                    "dropout": args.dropout,
                    "use_cls_token": args.use_cls_token,
                },
                "label_map": {str(k): v for k, v in label_map.items()},
                "best_acc": best_acc,
            }
            torch.save(ckpt, data_dir / "best_model.pt")

    # 保存训练曲线 CSV
    csv_path = data_dir / "train_history.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,test_loss,test_acc\n")
        for r in history:
            f.write(f"{r['epoch']},{r['train_loss']},{r['train_acc']},{r['test_loss']},{r['test_acc']}\n")

    # 用最佳模型重新评估并输出报告
    ckpt = torch.load(data_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, final_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    labels = list(range(args.num_classes))
    target_names = [label_map[i] for i in labels]

    print("\n========== Final Test Result ==========")
    print(f"Best test accuracy: {final_acc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=4, zero_division=0))
    print(f"\n最佳模型保存到：{data_dir / 'best_model.pt'}")
    print(f"训练曲线保存到：{csv_path}")


@torch.no_grad()
def infer(args: argparse.Namespace) -> None:
    if torch is None:
        raise ImportError("未安装 torch。请先安装 PyTorch。")

    ckpt_path = Path(args.ckpt)
    video_path = Path(args.video)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    label_map = {int(k): v for k, v in ckpt["label_map"].items()}

    model = SkeletonTransformer(**config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    seq = extract_skeleton_sequence(
        video_path=video_path,
        target_frames=config["target_frames"],
        max_source_frames=args.max_source_frames,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    x = torch.from_numpy(seq).unsqueeze(0).to(device)  # [1, T, 132]
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_id = int(np.argmax(probs))
    confidence = float(probs[pred_id])

    print("\n========== Single Video Inference ==========")
    print(f"Video: {video_path}")
    print(f"Predicted class: {label_map[pred_id]}")
    print(f"Confidence: {confidence:.4f}")
    print("\nClass probabilities:")
    for i in range(config["num_classes"]):
        print(f"  {i} | {label_map[i]:20s}: {probs[i]:.4f}")


def add_preprocess_args(subparsers) -> None:
    p = subparsers.add_parser("preprocess", help="提取 MediaPipe 骨架序列并保存 .npy")
    p.add_argument("--data-root", type=str, required=True, help="Kaggle 数据集解压后的根目录")
    p.add_argument("--out-dir", type=str, default="./processed", help="输出 .npy 的目录")
    p.add_argument("--target-frames", type=int, default=30, help="每个视频统一重采样的帧数")
    p.add_argument("--test-size", type=float, default=0.2, help="测试集比例")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-source-frames", type=int, default=120, help="每个视频最多处理多少原始帧；0 表示处理全部帧")
    p.add_argument("--min-detection-confidence", type=float, default=0.5)
    p.add_argument("--min-tracking-confidence", type=float, default=0.5)
    p.set_defaults(func=preprocess)


def add_train_args(subparsers) -> None:
    p = subparsers.add_parser("train", help="训练 Skeleton Transformer")
    p.add_argument("--data-dir", type=str, default="./processed", help="包含 X_train.npy 等文件的目录")
    p.add_argument("--input-dim", type=int, default=132)
    p.add_argument("--target-frames", type=int, default=30)
    p.add_argument("--num-classes", type=int, default=6)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dim-feedforward", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use-cls-token", action="store_true", default=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--augment", action="store_true", help="训练时加入轻微骨架噪声增强")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="", help="例如 cuda / cpu；留空自动判断")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=train)


def add_infer_args(subparsers) -> None:
    p = subparsers.add_parser("infer", help="对单个视频做推理")
    p.add_argument("--video", type=str, required=True, help="待推理视频路径")
    p.add_argument("--ckpt", type=str, default="./processed/best_model.pt", help="训练得到的模型权重")
    p.add_argument("--max-source-frames", type=int, default=120, help="每个视频最多处理多少原始帧；0 表示处理全部帧")
    p.add_argument("--min-detection-confidence", type=float, default=0.5)
    p.add_argument("--min-tracking-confidence", type=float, default=0.5)
    p.add_argument("--device", type=str, default="", help="例如 cuda / cpu；留空自动判断")
    p.set_defaults(func=infer)


def main() -> None:
    parser = argparse.ArgumentParser(description="MediaPipe Pose + Skeleton Transformer 羽毛球击球动作识别")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_preprocess_args(subparsers)
    add_train_args(subparsers)
    add_infer_args(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
