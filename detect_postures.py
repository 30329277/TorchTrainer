import os
import torch
import torchvision
import torch.multiprocessing as mp
from torchvision.transforms import functional as F
from torchvision.ops import box_iou
from tqdm import tqdm
import json
import subprocess
import time
import av
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

def load_models(model_dir):
    """加载所有姿势检测模型."""
    models = {}
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    for filename in os.listdir(model_dir):
        if filename.endswith("_detector_model.pth"):
            model_name = filename[:-len("_detector_model.pth")]
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
            model.load_state_dict(torch.load(os.path.join(model_dir, filename), map_location=device, weights_only=True)) 
            model.eval()
            model.to(device)
            models[model_name] = model
    return models, device

def save_detected_frame(frame, predictions, output_path):
    """保存检测到的帧，并标记方框及得分."""
    fig, ax = plt.subplots(1)
    ax.imshow(frame)

    for i, box in enumerate(predictions['boxes']):
        score = predictions['scores'][i]
        if score > 0.7:  # 只标记得分大于0.7的检测结果
            x1, y1, x2, y2 = box
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            plt.text(x1, y1, f"{score:.2f}", color='g', fontsize=12, weight='bold')

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def detect_postures(frame, models, device, label, score_threshold):
    """检测图像帧中的所有姿势."""
    input_tensor = F.to_tensor(frame).unsqueeze(0).to(device)
    combined_predictions = {'boxes': [], 'scores': [], 'labels': []}
    for model_name, model in models.items():
        with torch.no_grad():
            predictions = model(input_tensor)[0]
        for i, score in enumerate(predictions['scores']):
            if predictions['labels'][i] == label and score > score_threshold:
                combined_predictions['boxes'].append(predictions['boxes'][i])
                combined_predictions['scores'].append(score)
                combined_predictions['labels'].append(predictions['labels'][i])
    return combined_predictions

def format_time(seconds):
    """格式化时间为 时:分:秒."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:06.3f}"

def process_frame(frame, frame_count, fps, models, device, label, score_threshold):
    """处理单个视频帧."""
    img = frame.to_image()
    img = np.array(img)
    timestamp = frame_count / fps
    predictions = detect_postures(img, models, device, label, score_threshold)
    formatted_timestamp = format_time(timestamp)
    return {"time": formatted_timestamp, "predictions": predictions, "frame": img, "frame_count": frame_count}

def process_video(video_path, models, device, output_folder, results, label, score_threshold, interval_seconds, queue):
    """处理单个视频."""
    container = av.open(video_path)
    fps = float(container.streams.video[0].average_rate)  # 将 fps 转换为浮点数
    total_frames = container.streams.video[0].frames
    interval_frames = int(interval_seconds * fps)  # Interval in seconds
    frame_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_results = []

    with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar, ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for frame in container.decode(video=0):
            if frame_count % interval_frames == 0:
                futures.append(executor.submit(process_frame, frame, frame_count, fps, models, device, label, score_threshold))
            frame_count += 1
            pbar.update(1)

        for future in futures:
            result = future.result()
            if result["predictions"]['boxes']:
                video_results.append({"video": video_name, "time": result["time"], "predictions": result["predictions"]})
                queue.put(result)

    results[video_name] = video_results

def time_to_seconds(time_str):
    """将时间字符串转换为秒."""
    hours, minutes, seconds = map(float, time_str.split(':'))
    return hours * 3600 + minutes * 60 + seconds

def round_time_to_seconds(time_str):
    """将时间字符串取整到秒."""
    hours, minutes, seconds = map(float, time_str.split(':'))
    rounded_seconds = int(seconds)
    return f"{int(hours):02}:{int(minutes):02}:{rounded_seconds:02}"

def extract_and_merge_segments(input_folder, output_folder, results, segment_gap_seconds, interval_seconds):
    """提取并合并视频片段."""
    # 使用FFmpeg的绝对路径, 请修改为你电脑上FFmpeg的实际路径
    ffmpeg_path = r"D:\PythonProject\TorchTrainer\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"  # 替换为你的FFmpeg可执行文件的绝对路径

    for video_name, video_data in results.items():
        segments = []
        processed_times = set()
        for i, data in enumerate(video_data):
            rounded_time = round_time_to_seconds(data['time'])
            current_time = time_to_seconds(rounded_time)

            if rounded_time in processed_times:
                continue

            start_time = max(0, current_time - interval_seconds)
            end_time = current_time + 1

            for j in range(i + 1, len(video_data)):
                next_rounded_time = round_time_to_seconds(video_data[j]['time'])
                next_time = time_to_seconds(next_rounded_time)

                if next_time - current_time > segment_gap_seconds:
                    break

                end_time = next_time + 1
                processed_times.add(next_rounded_time)

            segments.append((start_time, end_time))
            processed_times.add(rounded_time)

        if not segments:
            continue

        input_video = os.path.join(input_folder, f"{video_name}.MP4")
        output_video = os.path.join(output_folder, f"Extracted_{video_name}.MP4")
        temp_files = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i, (start, end) in enumerate(segments):
                temp_file = os.path.join(output_folder, f"temp_{video_name}_{i}.mp4")
                temp_files.append(temp_file)
                cmd = [ffmpeg_path, "-loglevel", "error", "-i", input_video, "-ss", str(start), "-to", str(end), "-c", "copy", temp_file]
                futures.append(executor.submit(subprocess.run, cmd, check=True))

            for future in futures:
                try:
                    future.result()
                except subprocess.CalledProcessError as e:
                    print(f"Error executing FFmpeg: {e}")
                    continue  # Skip to the next segment

        concat_list_path = os.path.join(output_folder, "concat_list.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for temp_file in temp_files:
                f.write(f"file '{os.path.abspath(temp_file)}'\n")  # 使用绝对路径并引用

        # 使用绝对路径
        concat_list_path = os.path.abspath(concat_list_path)
        cmd = [ffmpeg_path, "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", concat_list_path, "-c", "copy", "-fflags", "+genpts", "-movflags", "+faststart", output_video]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing FFmpeg: {e}")

        # 暂时不删除临时文件以便调试
        for temp_file in temp_files:
            os.remove(temp_file)
        os.remove(concat_list_path)

def main():
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
    queue = Queue()

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.mp4', '.avi', '.mov')): # Handle various video extensions
            video_path = os.path.join(input_folder, filename)
            process_video(video_path, models, device, output_folder, results, label, score_threshold, interval_seconds, queue)

    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)

    while not queue.empty():
        result = queue.get()
        frame = result["frame"]
        predictions = result["predictions"]
        frame_count = result["frame_count"]
        formatted_timestamp = result["time"].replace(":", "_")
        output_path = os.path.join(output_folder, f"frame_{frame_count}_{formatted_timestamp}.jpg")
        save_detected_frame(frame, predictions, output_path)

    extract_and_merge_segments(input_folder, output_folder, results, segment_gap_seconds, interval_seconds)

if __name__ == "__main__":
    mp.freeze_support()
    main()