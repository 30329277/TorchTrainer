import os
import subprocess
from tqdm import tqdm  # 导入 tqdm 库

# 输入视频文件夹路径和输出目录
input_video_dir = r"\\192.168.31.1\XiaoMi-usb0\newdownload\2024-2"  # 使用原始字符串，替换为你的 MTS 视频文件夹路径
output_dir = r"D:\PythonProject\posture model04"
os.makedirs(output_dir, exist_ok=True)

# 定义每隔多少秒保存一帧画面
frame_interval = 6 * 60  

# ffmpeg 可执行文件路径
ffmpeg_path = r"ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

def extract_frames(input_path, output_directory, frame_interval):
    # 获取视频文件名（不包括扩展名）
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # 构建输出模板路径 (图片)
    output_image_template = os.path.join(output_directory, f"{base_name}_%03d.jpg")

    # 构建 ffmpeg 命令 (提取帧)
    command_image = [
        ffmpeg_path,
        "-i", input_path,
        "-vf", f"fps=1/{frame_interval}",  # 使用vf过滤器设置帧率
        output_image_template,
        "-loglevel", "error",  # 添加此行以抑制不必要的输出
        "-err_detect", "ignore_err"  # 添加此行以忽略错误
    ]
    
    # 执行命令
    try:
        subprocess.run(command_image, check=True)
        print(f"图片已成功提取并保存在 {output_directory} 文件夹中")

    except subprocess.CalledProcessError as e:
        print(f"处理失败: {e}")

# 获取所有视频文件
video_files = [f for f in os.listdir(input_video_dir) if f.endswith(".mp4")]

# 处理文件夹中的所有视频文件并显示进度条
for file_name in tqdm(video_files, desc="Processing videos"):
    input_video_path = os.path.join(input_video_dir, file_name)
    extract_frames(input_video_path, output_dir, frame_interval)
