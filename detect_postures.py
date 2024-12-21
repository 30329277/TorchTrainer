import cv2
import os
import torch
import torchvision
import torch.multiprocessing as mp
from torchvision.transforms import functional as F
from tqdm import tqdm
import json
import subprocess
import time
import shlex

def load_models(model_dir):
    """加载所有姿势检测模型."""
    models = {}
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    for filename in os.listdir(model_dir):
        if filename.endswith("_detector_model.pth"):
            model_name = filename[:-len("_detector_model.pth")]
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
            # 使用 weights_only=True 加载模型
            model.load_state_dict(torch.load(os.path.join(model_dir, filename), map_location=device, weights_only=True)) 
            model.eval()
            model.to(device)
            models[model_name] = model
    return models, device

def save_detected_frame(frame, predictions, output_path):
    """保存检测到的帧，并标记方框及得分."""
    for i, box in enumerate(predictions[0]['boxes']):
        score = predictions[0]['scores'][i]
        if score > 0.7:  # 只标记得分大于0.7的检测结果
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite(output_path, frame)

def detect_postures(frame, models, device, label, score_threshold, output_folder, frame_count):
    """检测图像帧中的所有姿势，并保存检测到的帧."""
    input_tensor = F.to_tensor(frame).unsqueeze(0).to(device)
    detected_postures = []
    for model_name, model in models.items():
        with torch.no_grad():
            predictions = model(input_tensor)
        for i, score in enumerate(predictions[0]['scores']):
            if predictions[0]['labels'][i] == label and score > score_threshold:
                detected_postures.append(model_name)
                output_path = os.path.join(output_folder, f"frame_{frame_count}_{model_name}.jpg")
                save_detected_frame(frame.copy(), predictions, output_path)
                break  # Once a posture is detected, move to the next model
    return detected_postures

def format_time(seconds):
    """格式化时间为 时:分:秒."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:06.3f}"

def process_video(video_path, models, device, output_folder, results, label, score_threshold, interval_seconds):
    """处理单个视频."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = int(interval_seconds * fps)  # Interval in seconds
    frame_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_results = []

    with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval_frames == 0:
                detected_postures = detect_postures(frame, models, device, label, score_threshold, output_folder, frame_count)
                timestamp = frame_count / fps
                formatted_timestamp = format_time(timestamp)
                if detected_postures:
                    video_results.append({"video": video_name, "time": formatted_timestamp, "postures": detected_postures})

            frame_count += 1
            pbar.update(1)

    cap.release()
    results[video_name] = video_results

def time_to_seconds(time_str):
    """将时间字符串转换为秒."""
    hours, minutes, seconds = map(float, time_str.split(':'))
    return hours * 3600 + minutes * 60 + seconds

def extract_and_merge_segments(input_folder, output_folder, results, segment_gap_seconds):
    """提取并合并视频片段."""
    # 使用FFmpeg的绝对路径, 请修改为你电脑上FFmpeg的实际路径
    ffmpeg_path = r"D:\PythonProject\TorchTrainer\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"  # 替换为你的FFmpeg可执行文件的绝对路径

    for video_name, video_data in results.items():
        segments = []
        start_time = -1
        for i, data in enumerate(video_data):
            if start_time == -1:
                start_time = time_to_seconds(data['time'])

            if i + 1 == len(video_data) or time_to_seconds(video_data[i+1]['time']) - time_to_seconds(data['time']) > segment_gap_seconds:
                segments.append((start_time, time_to_seconds(data['time'])))
                start_time = -1

        if not segments:
            continue

        input_video = os.path.join(input_folder, f"{video_name}.MP4")
        output_video = os.path.join(output_folder, f"Extracted_{video_name}.MP4")
        temp_files = []

        for i, (start, end) in enumerate(segments):
            # 处理开始和结束时间相同的情况
            if start == end:
                end += 0.1  # Add a small duration (0.1 seconds) to the end time

            temp_file = os.path.join(output_folder, f"temp_{video_name}_{i}.mp4")
            temp_files.append(temp_file)
            cmd = [ffmpeg_path, "-i", input_video, "-ss", str(start), "-to", str(end), "-c", "copy", temp_file]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing FFmpeg: {e}")
                # Handle the error appropriately, e.g., log it, skip the segment, etc.
                continue  # Skip to the next segment

        concat_list_path = os.path.join(output_folder, "concat_list.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for temp_file in temp_files:
                f.write(f"file '{os.path.abspath(temp_file)}'\n")  # 使用绝对路径并引用

        # 使用绝对路径
        concat_list_path = os.path.abspath(concat_list_path)
        cmd = [ffmpeg_path, "-f", "concat", "-safe", "0", "-i", concat_list_path, "-fflags", "+genpts", "-c", "copy", output_video]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing FFmpeg: {e}")

        for temp_file in temp_files:
            os.remove(temp_file)
        os.remove(concat_list_path)

def main():
    # input_folder = "input_video"
    # output_folder = "output_video"
    # input_folder = r"\\192.168.31.1\XiaoMi-usb0\newdownload\2024-2\新建文件夹"
    # output_folder = r"\\192.168.31.1\XiaoMi-usb0\newdownload\2024-2\新建文件夹"
    
    input_folder = r"D:\PythonProject\data\test"
    output_folder = r"D:\PythonProject\data\test"

    result_file = os.path.join(input_folder, "result.json")
    
    label = 1
    score_threshold = 0.7
    interval_seconds = 60*1  # Interval in seconds for processing frames
    segment_gap_seconds = 60*3  # Gap in seconds to consider segments separate

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    models, device = load_models(".")  # Load models from the current directory

    results = {}
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.mp4', '.avi', '.mov')): # Handle various video extensions
            video_path = os.path.join(input_folder, filename)
            process_video(video_path, models, device, output_folder, results, label, score_threshold, interval_seconds)

    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)

    extract_and_merge_segments(input_folder, output_folder, results, segment_gap_seconds)

if __name__ == "__main__":
    mp.freeze_support()
    main()