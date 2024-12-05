import os
import torch
import torchvision
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import xml.etree.ElementTree as ET
import numpy as np
import torch.multiprocessing as mp
import sys  # 用于退出程序
import tqdm  #  用于进度条


# 将 collate_fn 移到函数外部
def collate_fn(batch):
    return tuple(zip(*batch))

# 数据集类
class ObjectADataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # 获取所有图像和标注文件
        self.image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.annotation_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.xml')]

        # 构建匹配的文件名列表
        self.data = []
        for img_file in self.image_files:
            annotation_file = img_file[:-4] + '.xml'
            if annotation_file in self.annotation_files:
                self.data.append((img_file, annotation_file))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, annotation_name = self.data[idx]
        img_path = os.path.join(self.data_dir, img_name)
        annotation_path = os.path.join(self.data_dir, annotation_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 注意颜色通道转换

        boxes = []
        labels = []
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            label = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # 这里假设所有标注都是 'objectA'，类别标签为 1

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)  # 假设没有 crowd 的标注

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transform:
            img = self.transform(img)

        return img, target  # Return image and target dictionary



# 模型训练函数
def train_model(data_dir, pretrained_model_path="objectA_detector_model.pth", num_classes=2, num_epochs=10, batch_size=2, learning_rate=0.005, momentum=0.9, num_workers=0):
 #  num_workers 默认改为0
    """
    训练目标检测模型。

    Args:
        images_dir:  图像目录路径.
        annotations_dir: 标注文件 (XML) 目录路径.
        pretrained_model_path: 预训练模型路径.  如果文件存在则加载，否则创建新模型.
        num_classes:  类别数量 (包括背景).
        num_epochs: 训练轮数.
        batch_size: 批次大小.
        learning_rate: 学习率.
        momentum: 动量.
        num_workers: 数据加载器使用的进程数.

    Returns:
        训练好的模型.
    """

    # 创建数据集和数据加载器
    dataset = ObjectADataset(data_dir, transform=torchvision.transforms.ToTensor())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)


    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    # 加载预训练模型或创建新模型
    if os.path.exists(pretrained_model_path):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu', weights_only=True)) #  map_location and weights_only added
        print(f"Loaded pretrained model from {pretrained_model_path}")
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)  # 使用预训练的模型初始化
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)  # 替换头部
        print("Initialized model with pretrained weights from torchvision")


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # 仅训练需要梯度的参数
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum)  # 优化后的优化器


    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0  # 用于累积每个 epoch 的损失

        # 使用 tqdm 显示进度条
        with tqdm.tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
            for images, targets in tepoch:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                epoch_loss += losses.item()  # 累积损失

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                tepoch.set_postfix(loss=losses.item()) # 在进度条中显示当前批次的损失

        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss / len(data_loader)}") # 显示平均损失

    return model



if __name__ == '__main__':
    mp.freeze_support()

    data_dir = r"D:\PythonProject\objectA model\pictures05"
    model_path = "objectA_detector_model.pth"
    num_workers = 0  #  建议设置为 0 或较小的值

    trained_model = train_model(data_dir, pretrained_model_path=model_path, num_workers=num_workers)
    torch.save(trained_model.state_dict(), model_path)
    print(f"模型训练完成并保存为 {model_path}")