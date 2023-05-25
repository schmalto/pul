from yolov5 import train
from ultralytics import YOLO
import torch
import os


# model = YOLO('yolov8n.pt', device='cuda')
# model.train(data="", epochs="")

train.run(data='/home/tobias/git_ws/pul/data_generation_test/datasets/cars_trucks/cars_trucks.yaml',
           imgsz=320, weights='yolov5m.pt', batch_size=4, epochs=200)

# model = torch.hub.load('ultralytics/yolov5', 'custom',
#                        path='/home/tobias/git_ws/pul/data_generation_test/yolov5/runs/train/exp7/weights/best.pt', force_reload=True)


folder = '/home/tobias/git_ws/pul/data_generation_test/val_images'

for file in os.listdir(folder):
    result = model(folder + '/' + file)
    result.save()

