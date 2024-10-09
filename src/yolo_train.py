import torch 
from ultralytics import YOLO, checks

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = YOLO('yolov8n.pt').to(device)

model.train(data = 'C:\Python_Projects\FinTwin_project\data\\ft_dataset\dataset.yaml', epochs = 10, batch = 32, imgsz = 640, device = device)