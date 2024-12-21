import os
import torch
import torchvision
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import xml.etree.ElementTree as ET
import numpy as np
import torch.multiprocessing as mp
import tqdm

# collate_fn
def collate_fn(batch):
    return tuple(zip(*batch))

# Dataset class
class ObjectADataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.annotation_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.xml')]
        self.data = []
        for img_file in self.image_files:
            annotation_file = img_file[:-4] + '.xml'
            if annotation_file in self.annotation_files:
                self.data.append((img_file, annotation_file))

        if len(self.data) == 0:
            raise ValueError("No matching image and annotation files found in the dataset directory.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, annotation_name = self.data[idx]
        img_path = os.path.join(self.data_dir, img_name)
        annotation_path = os.path.join(self.data_dir, annotation_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        try:
            tree = ET.parse(annotation_path)
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse annotation {annotation_path}: {e}")

        root = tree.getroot()
        for obj in root.findall('object'):
            label = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # Modify this to support multiple classes if needed

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        if self.transform:
            img = self.transform(img)

        return img, target

# Training function
def train_model(data_dir, pretrained_model_path="objectA_detector_model.pth", num_classes=2, num_epochs=10, batch_size=8, learning_rate=0.005, momentum=0.9, num_workers=4):
    dataset = ObjectADataset(data_dir, transform=torchvision.transforms.ToTensor())
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )

    # Load or initialize model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    if os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print(f"Loaded weights from {pretrained_model_path}")
    else:
        print("Initialized new model with pretrained weights")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        with tqdm.tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
            for images, targets in tepoch:
                images = list(image.to(device, non_blocking=True) for image in images)
                targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

                try:
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    epoch_loss += losses.item()

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print("Out of memory! Reducing batch size or switching to CPU might help.")
                        raise e

                tepoch.set_postfix(loss=losses.item())

        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss / len(data_loader)}")

    torch.save(model.state_dict(), pretrained_model_path)
    print(f"Model weights saved as {pretrained_model_path}")

    return model

if __name__ == '__main__':
    mp.freeze_support()
    data_dir = r"D:\PythonProject\test"
    model_path = "objectA_detector_model.pth"

    trained_model = train_model(data_dir, pretrained_model_path=model_path)
