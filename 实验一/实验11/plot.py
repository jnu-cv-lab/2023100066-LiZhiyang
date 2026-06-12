import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 读取训练历史文件
csv_path = Path("./processed/train_history.csv")
df = pd.read_csv(csv_path)

# 创建输出文件夹
out_dir = Path("./outputs")
out_dir.mkdir(exist_ok=True)

# 绘制 Loss 曲线
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
plt.plot(df["epoch"], df["test_loss"], marker="s", label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "loss_curve.png", dpi=300)
plt.close()

# 绘制 Accuracy 曲线
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_acc"], marker="o", label="Train Accuracy")
plt.plot(df["epoch"], df["test_acc"], marker="s", label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "accuracy_curve.png", dpi=300)
plt.close()

print("训练曲线已保存：")
print(out_dir / "loss_curve.png")
print(out_dir / "accuracy_curve.png")