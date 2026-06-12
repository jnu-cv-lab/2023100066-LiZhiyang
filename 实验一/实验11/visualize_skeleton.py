from pathlib import Path
import argparse
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
def safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("(", "").replace(")", "")

def draw_pose_on_black(image_shape, results):
    """生成黑底骨架图。"""
    h, w = image_shape[:2]
    black = np.zeros((h, w, 3), dtype=np.uint8)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            black,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 255, 255),
                thickness=2,
                circle_radius=2,
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 255, 255),
                thickness=2,
                circle_radius=2,
            ),
        )
    return black

def make_contact_sheet(frames, out_path: Path, title: str):
    """将若干关键帧拼成一张图片。"""
    if not frames:
        return

    n = len(frames)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(4 * cols, 3 * rows))
    for i, frame in enumerate(frames):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(frame_rgb)
        plt.title(f"Frame {i + 1}")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def visualize_video(video_path: Path, out_dir: Path, max_frames: int = 120, sample_count: int = 8):
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在：{video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    name = safe_name(video_path)
    overlay_path = out_dir / f"{name}_pose_overlay.mp4"
    skeleton_path = out_dir / f"{name}_skeleton_only.mp4"
    sheet_path = out_dir / f"{name}_skeleton_keyframes.png"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))
    skeleton_writer = cv2.VideoWriter(str(skeleton_path), fourcc, fps, (width, height))

    # 为了生成拼图，均匀保存若干帧
    process_frames = min(total if total > 0 else max_frames, max_frames)
    sample_indices = set(np.linspace(0, max(0, process_frames - 1), sample_count).astype(int).tolist())
    sampled_frames = []

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_id = 0
    valid_pose_count = 0

    while frame_id < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        overlay = frame_bgr.copy()
        if results.pose_landmarks:
            valid_pose_count += 1
            mp_drawing.draw_landmarks(
                overlay,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=2,
                    circle_radius=2,
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 255),
                    thickness=2,
                    circle_radius=2,
                ),
            )

        skeleton = draw_pose_on_black(frame_bgr.shape, results)

        overlay_writer.write(overlay)
        skeleton_writer.write(skeleton)

        if frame_id in sample_indices:
            # 拼图使用叠加骨架后的画面，更直观
            sampled_frames.append(overlay.copy())

        frame_id += 1

    cap.release()
    pose.close()
    overlay_writer.release()
    skeleton_writer.release()

    make_contact_sheet(sampled_frames, sheet_path, title=f"Skeleton Visualization: {video_path.name}")

    print("骨架可视化完成")
    print(f"输入视频：{video_path}")
    print(f"处理帧数：{frame_id}")
    print(f"检测到人体骨架的帧数：{valid_pose_count}")
    print(f"原视频叠加骨架：{overlay_path}")
    print(f"黑底骨架视频：{skeleton_path}")
    print(f"关键帧拼图：{sheet_path}")


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Pose 骨架可视化")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--out-dir", type=str, default="./结果曲线/skeleton_vis", help="输出目录")
    parser.add_argument("--max-frames", type=int, default=120, help="最多处理多少帧")
    parser.add_argument("--sample-count", type=int, default=8, help="关键帧拼图中展示多少帧")
    args = parser.parse_args()

    visualize_video(
        video_path=Path(args.video),
        out_dir=Path(args.out_dir),
        max_frames=args.max_frames,
        sample_count=args.sample_count,
    )


if __name__ == "__main__":
    main()
