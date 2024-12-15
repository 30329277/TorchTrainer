import cv2
import os
import torch
import torchvision
import torch.multiprocessing as mp
from torchvision.transforms import functional as F
from tqdm import tqdm

def load_model():
    """
    加载目标检测模型到 GPU 或 CPU。
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
    model.load_state_dict(torch.load("objectA_detector_model.pth", map_location=device))
    model.eval()
    model.to(device)
    return model, device

def detect_objectA(frame, model, device):
    """
    检测图像中的目标。

    Args:
        frame: OpenCV 读取的图像帧 (numpy.ndarray).
        model: 已加载的 PyTorch 模型.
        device: 模型运行的设备 (GPU/CPU).

    Returns:
        bboxes: 检测到的目标边界框列表 (list of lists)，每个边界框格式为 [x1, y1, x2, y2].
        scores: 对应边界框的置信度得分 (list of floats).
    """
    input_tensor = F.to_tensor(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(input_tensor)

    bboxes, scores = [], []
    for i, score in enumerate(predictions[0]['scores']):
        if predictions[0]['labels'][i] == 1 and score > 0.5:  # 1 代表 'objectA' 类别
            box = predictions[0]['boxes'][i].cpu().numpy().astype(int).tolist()
            bboxes.append(box)
            scores.append(score.item())

    return bboxes, scores

def process_frame(frame, frame_count, fps, output_folder, video_name, model, device):
    bboxes, scores = detect_objectA(frame, model, device)

    if bboxes:  # 如果检测到目标
        for box, score in zip(bboxes, scores):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"ObjectA: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            timestamp = frame_count / fps
            # 添加视频名称作为前缀
            image_filename = os.path.join(output_folder, f"{video_name}_frame_{int(timestamp)}s.jpg")
            cv2.imwrite(image_filename, frame)

    return frame

def worker_task(args):
    """
    工作进程任务。
    """
    frame, frame_count, fps, output_folder, video_name, model, device = args
    return process_frame(frame, frame_count, fps, output_folder, video_name, model, device)

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = int(60 * fps)
    frame_count = 0

    output_folder = "output_frames"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取视频名称，无扩展名
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    model, device = load_model()

    num_workers = max(1, mp.cpu_count() - 2)  # 留一些核心避免 CPU 过载
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=num_workers)

    frames_chunk = []
    results = []

    with tqdm(total=total_frames, desc="Processing Frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval_frames == 0:
                frames_chunk.append((frame, frame_count, fps, output_folder, video_name, model, device))

                if len(frames_chunk) >= num_workers:
                    results = pool.map(worker_task, frames_chunk)
                    for frame_result in results:
                        cv2.imshow('Video Analysis', frame_result)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    frames_chunk = []

            frame_count += 1
            pbar.update(1)

        if frames_chunk:
            results = pool.map(worker_task, frames_chunk)
            for frame_result in results:
                cv2.imshow('Video Analysis', frame_result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    pool.close()
    pool.join()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r"D:\PythonProject\data\3.mp4"
    main(video_path)
