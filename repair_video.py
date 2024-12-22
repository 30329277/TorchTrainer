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

if __name__ == "__main__":
    input_video_path = r".mp4"
    output_video_path = r"_repaired.mp4"
    repair_video(input_video_path, output_video_path)
