import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ffmpeg 可执行文件路径
ffmpeg_path = r"ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

# 支持的输入视频格式
supported_formats = [
    '.avi', '.vob', '.wmv', '.mkv', '.mpeg', '.mpg', 
    '.rmvb', '.3gp', '.mov', '.mts', '.rm', '.m2ts'
]

def convert_to_mp4(input_path, output_path, log_file):
    """Convert a video file to MP4 format."""
    command = [
        ffmpeg_path,
        "-i", input_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]
    try:
        result = subprocess.run(command, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        os.remove(input_path)  # 删除原文件
    except subprocess.CalledProcessError as e:
        with open(log_file, "a") as log:
            log.write(f"Failed to convert {input_path}: {e}\n")
            log.write(f"Error output: {e.stderr.decode()}\n")

def process_folder(folder_path):
    """Process all video files in the folder and its subfolders."""
    tasks = []
    log_file = os.path.join(folder_path, "conversion_errors.log")
    with ThreadPoolExecutor(max_workers=5) as executor:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_formats):
                    input_path = os.path.join(root, file)
                    output_path = os.path.splitext(input_path)[0] + ".mp4"
                    tasks.append((input_path, output_path, executor.submit(convert_to_mp4, input_path, output_path, log_file)))
        
        with tqdm(total=len(tasks), desc="Total Progress") as pbar:
            for input_path, output_path, future in tasks:
                with tqdm(total=1, desc=f"Converting {os.path.basename(input_path)}", leave=False) as video_pbar:
                    try:
                        future.result()
                        video_pbar.update(1)
                    except Exception as e:
                        with open(log_file, "a") as log:
                            log.write(f"Error processing file {input_path}: {e}\n")
                    pbar.update(1)

if __name__ == "__main__":
    folder_path = r"D:\PythonProject\data\videos"  # 替换为你的文件夹路径
    process_folder(folder_path)
