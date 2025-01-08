import os
import subprocess

def repair_video(input_video_path, output_video_path):
    ffmpeg_path = r"D:\PythonProject\TorchTrainer\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"  # 替换为你的FFmpeg可执行文件的绝对路径

    # FFmpeg 命令，用于修复视频
    command = [
        ffmpeg_path,
        "-i", input_video_path,
        "-c", "copy",
        "-fflags", "+genpts",
        output_video_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"视频已成功修复并保存在 {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"修复视频失败: {e}")

def repair_videos_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.mp4', '.avi', '.mov')):  # Handle various video extensions
            input_video_path = os.path.join(input_folder, filename)
            output_video_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_repaired{os.path.splitext(filename)[1]}")
            repair_video(input_video_path, output_video_path)

if __name__ == "__main__":
    input_path = input("请输入视频文件路径或文件夹路径: ").strip()
    if os.path.isfile(input_path):
        output_video_path = f"{os.path.splitext(input_path)[0]}_repaired{os.path.splitext(input_path)[1]}"
        repair_video(input_path, output_video_path)
    elif os.path.isdir(input_path):
        output_folder = f"{input_path}_repaired"
        repair_videos_in_folder(input_path, output_folder)
    else:
        print("输入的路径无效，请输入有效的视频文件路径或文件夹路径。")
