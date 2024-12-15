# import cv2

# print("OpenCV 版本:", cv2.__version__)
# print("CUDA 支持设备数量:", cv2.cuda.getCudaEnabledDeviceCount())

# import torch
# print(torch.backends.cudnn.enabled)  # 如果返回 True，说明 cuDNN 已启用

import cv2
import torch

print("OpenCV 版本:", cv2.__version__)
print("CUDA 支持设备数量:", cv2.cuda.getCudaEnabledDeviceCount())
print("cuDNN 版本:", torch.backends.cudnn.version())
print("cuDNN 是否启用:", torch.backends.cudnn.enabled)