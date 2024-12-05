import cv2
import os
import torch
import torchvision
import multiprocessing

# --- detect_human.py 的内容 ---
def detect_objectA(frame):
    """
    检测图像中的目标。

    Args:
        frame: OpenCV 读取的图像帧 (numpy.ndarray).

    Returns:
        bboxes: 检测到的目标边界框列表 (list of lists)，每个边界框格式为 [x1, y1, x2, y2].
        scores:  对应边界框的置信度得分 (list of floats).
    """
    # 加载你训练好的模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2) # num_classes 为 2: 背景和目标
    model.load_state_dict(torch.load("objectA_detector_model.pth"))
    model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    input_tensor = transform(frame).to(device)
    with torch.no_grad():
        predictions = model([input_tensor])

    bboxes = []
    scores = []
    for i in range(len(predictions[0]['boxes'])):
        if predictions[0]['labels'][i] == 1 and predictions[0]['scores'][i] > 0.5: #  1 代表 'objectA' 类别， 0.5 为置信度阈值
            box = predictions[0]['boxes'][i].cpu().numpy().tolist()
            bboxes.append([int(coord) for coord in box])
            scores.append(predictions[0]['scores'][i].item())

    return bboxes, scores


# --- main.py 的内容 ---
def process_frame(frame, frame_count, fps, output_folder):
    bboxes, scores = detect_objectA(frame)  # 直接检测目标

    if bboxes:  # 如果检测到目标
        for box, score in zip(bboxes, scores):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 绘制红色边界框 (目标)
            cv2.putText(frame, f"ObjectA: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            timestamp = frame_count / fps
            image_filename = os.path.join(output_folder, f"frame_{int(timestamp)}s.jpg")
            cv2.imwrite(image_filename, frame)
            print(f"目标检测到！帧已保存为: {image_filename}")

        timestamp = frame_count / fps
        print(f"在 {timestamp:.2f} 秒检测到胸部。")

    return frame

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(30 * fps)  # 每 30 秒间隔的帧数
    frame_count = 0

    # 确保输出文件夹存在
    output_folder = "output_frames"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    num_processes = multiprocessing.cpu_count()  # 获取 CPU 核心数
    pool = multiprocessing.Pool(processes=num_processes)  # 创建进程池

    frames_chunk = []
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval_frames == 0:  # 每 30 秒处理一次
            frames_chunk.append((frame, frame_count, fps, output_folder))

            if len(frames_chunk) >= num_processes: #  累积一定数量的帧后再进行多进程处理
                results = pool.starmap(process_frame, frames_chunk)  # 并行处理帧
                frames_chunk = [] # 清空已处理帧列表

                for frame_result in results:
                    cv2.imshow('Video Analysis', frame_result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        frame_count += 1

    # 处理剩余的帧
    if frames_chunk:
        results = pool.starmap(process_frame, frames_chunk)
        for frame_result in results:
            cv2.imshow('Video Analysis', frame_result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    pool.close()  # 关闭进程池
    pool.join()   # 等待所有进程完成

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r"D:\PythonProject\data\merged_video.mp4"  # 替换为您的视频路径
    main(video_path)
