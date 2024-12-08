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

def detect_postures(frame, models, device):
    """检测图像帧中的所有姿势."""
    input_tensor = F.to_tensor(frame).unsqueeze(0).to(device)
    detected_postures = []
    for model_name, model in models.items():
        with torch.no_grad():
            predictions = model(input_tensor)
        for i, score in enumerate(predictions[0]['scores']):
            if predictions[0]['labels'][i] == 1 and score > 0.5:
                detected_postures.append(model_name)
                break  # Once a posture is detected, move to the next model
    return detected_postures

def process_video(video_path, models, device, output_folder, results):
    """处理单个视频."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = int(30 * fps)  # 30 seconds interval
    frame_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_results = []

    with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval_frames == 0:
                detected_postures = detect_postures(frame, models, device)
                timestamp = frame_count / fps
                if detected_postures:
                    video_results.append({"video": video_name, "time": timestamp, "postures": detected_postures})

            frame_count += 1
            pbar.update(1)

    cap.release()
    results[video_name] = video_results


def extract_and_merge_segments(input_folder, output_folder, results):
    """提取并合并视频片段."""
    # 使用FFmpeg的绝对路径,  请修改为你电脑上FFmpeg的实际路径
    ffmpeg_path = r"D:\PythonProject\TorchTrainer\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"  #  替换为你的FFmpeg可执行文件的绝对路径

    for video_name, video_data in results.items():
        segments = []
        start_time = -1
        for i, data in enumerate(video_data):
            if start_time == -1:
                start_time = data['time']

            if i + 1 == len(video_data) or video_data[i+1]['time'] - data['time'] > 180:
                segments.append((start_time, data['time']))
                start_time = -1

        if not segments:
            continue

        input_video = os.path.join(input_folder, f"{video_name}.MP4")
        output_video = os.path.join(output_folder, f"Extracted_{video_name}.MP4")
        temp_files = []

        for i, (start, end) in enumerate(segments):
            #  处理开始和结束时间相同的情况
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

        with open(os.path.join(output_folder, "concat_list.txt"), "w") as f:
            for temp_file in temp_files:
                f.write(f"file '{os.path.abspath(temp_file)}'\n")  # 使用绝对路径

        # 使用绝对路径
        concat_list_path = os.path.abspath(os.path.join(output_folder, "concat_list.txt"))
        cmd = [ffmpeg_path, "-f", "concat", "-safe", "0", "-i", concat_list_path, "-c", "copy", output_video]
        subprocess.run(cmd, check=True)

        for temp_file in temp_files:
            os.remove(temp_file)
        os.remove(os.path.join(output_folder, "concat_list.txt"))


def main():
    input_folder = "input_video"
    output_folder = "output_video"
    result_file = os.path.join(input_folder, "result.json")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    models, device = load_models(".")  # Load models from the current directory

    results = {}
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.mp4', '.avi', '.mov')): # Handle various video extensions
            video_path = os.path.join(input_folder, filename)
            process_video(video_path, models, device, output_folder, results)

    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)

    extract_and_merge_segments(input_folder, output_folder, results)


if __name__ == "__main__":
    mp.freeze_support()
    main()