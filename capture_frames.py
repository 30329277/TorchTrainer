import cv2
import os

def capture_frames(video_path, output_dir, interval=10):
    """
    每间隔指定秒数截取视频帧并保存为图片。

    Args:
        video_path (str): 视频文件路径。
        output_dir (str): 保存帧的输出目录。
        interval (int): 截取间隔时间（秒）。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    interval_frames = int(fps * interval)  # 每隔多少帧保存一次
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval_frames == 0:
            # 保存帧到输出目录
            filename = os.path.join(output_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            saved_count += 1

        frame_count += 1

    cap.release()
    print("视频帧捕获完成。")

if __name__ == "__main__":
    video_path = r"D:\PythonProject\data\7.mp4" # 替换为您的视频文件路径
    output_dir = r"D:\PythonProject\objectA model\pictures05"  # 替换为保存帧的目录
    capture_frames(video_path, output_dir, interval=10)
