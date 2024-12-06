# TorchTrainer
code trains an object detection model using Faster R-CNN with a custom dataset. It parses XML annotations, applies transformations, and supports loading pre-trained weights. The model is fine-tuned over multiple epochs with a learning rate scheduler and outputs progress.

# README  

## 项目简介 / Project Overview  
该项目包含三个功能模块：  
1. **视频目标检测**：基于 Faster R-CNN 模型，检测视频帧中的特定目标，并保存包含检测结果的图像。  
2. **目标检测模型训练**：利用自定义数据集（图像及其对应的 XML 标注文件）训练目标检测模型。  
3. **视频帧提取**：从视频中按指定时间间隔提取帧并保存为图片。  

This project includes three modules:  
1. **Object Detection in Videos**: Detect objects in video frames using a Faster R-CNN model and save the annotated frames.  
2. **Object Detection Model Training**: Train an object detection model with a custom dataset (images and XML annotations).  
3. **Video Frame Extraction**: Extract video frames at specified intervals and save them as images.  

---

## 环境依赖 / Environment Requirements  
- Python 3.8+  
- 必需库 / Required Libraries:  
  - `torch`, `torchvision`  
  - `opencv-python`  
  - `tqdm`  

---

## 使用说明 / Usage  

### 1. 视频目标检测 / Object Detection in Videos  
**文件**: `main.py`  
修改 `main()` 中的视频路径 `video_path`，运行脚本即可。  

**File**: `detect_object.py`  
Update the video path (`video_path`) in the `main()` function and run the script.  

### 2. 模型训练 / Model Training  
**文件**: `train_model.py`  
将数据集放在 `data_dir` 路径下，确保包含图像和 XML 标注文件。运行脚本进行模型训练，结果会保存为 `objectA_detector_model.pth`。  

**File**: `train_model.py`  
Place your dataset in the `data_dir` directory, ensuring it contains images and XML annotation files. Run the script to train the model, and the result will be saved as `objectA_detector_model.pth`.  

### 3. 视频帧提取 / Video Frame Extraction  
**文件**: `capture_frames.py`  
修改视频路径 `video_path` 和输出路径 `output_dir`，运行脚本即可。  

**File**: `capture_frames.py`  
Update the video path (`video_path`) and output directory (`output_dir`), then run the script.  

---

## 输出示例 / Example Outputs  
1. **检测后的帧 / Detected Frames**: 帧中绘制了目标的边界框及其置信度分数。  
2. **训练完成的模型 / Trained Model**: 文件名为 `objectA_detector_model.pth`。  
3. **提取的帧 / Extracted Frames**: 按指定间隔保存的图片文件。  

1. **Detected Frames**: Frames with bounding boxes and confidence scores for detected objects.  
2. **Trained Model**: Saved as `objectA_detector_model.pth`.  
3. **Extracted Frames**: Images saved at specified intervals.  

---

## 注意事项 / Notes  
1. 确保视频文件路径和输出目录正确。  
2. 数据集应包含标准 Pascal VOC 格式的 XML 文件。  
3. 训练时建议使用 GPU 提高速度。  

1. Ensure correct paths for video files and output directories.  
2. The dataset should include XML files in standard Pascal VOC format.  
3. Use a GPU for faster training.  

Enjoy coding! 😊  

