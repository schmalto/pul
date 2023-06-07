from yolov5 import train
from ultralytics import YOLO
import torch
import os
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import torchvision.transforms as transforms
from torchvision.ops import box_convert


def visualize():
     image = Image.open(
         '/home/tobias/git_ws/pul/data_generation_test/datasets/cars_trucks_patched/images/Screenshot 2022-09-02 182050_0.jpg')
     transform = transforms.Compose([
         transforms.PILToTensor()
     ])
     img_tensor = transform(image)

     boxes = box_convert([0.389062, 0.954688, 0.03125, 0.01875], 'xywh', 'xyxy')

     result = draw_bounding_boxes(img_tensor, boxes, width=5, colors=['red', 'blue'])

def train_model(yaml_file='truck_labeled_patched/truck_labeled_patched.yaml', name_run='truck_labeled_patched'):
     # model = YOLO('yolov8n.pt', device='cuda')

     # model.train(data="", epochs="")
    torch.cuda.empty_cache()
    data_name = '/home/tobias/git_ws/pul/data_generation_test/datasets/' + yaml_file
    train.run(data=data_name,imgsz=320, weights='yolov5l.pt', batch_size=2, epochs=300, project='./runs/', name=name_run)

def analyze():
    torch.cuda.empty_cache()
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                            path='/home/tobias/git_ws/pul/data_generation_test/runs/truck_labeled_many_1080_10802/weights/best.pt', force_reload=True)
    model.eval()
    folder = '/home/tobias/git_ws/pul/data_generation_test/val_images/1080_1080_copy'
    for file in os.listdir(folder):
        if file.startswith('_'):
            continue
        result = model(folder + '/' + file)
        result.save()

if __name__ == "__main__":
    train_model(yaml_file='truck_labeled_many_1080_1080/truck_labeled_many_1080_1080.yaml',
                 name_run='truck_labeled_many_1080_1080')
    train_model(yaml_file='truck_labeled_many_960_960/truck_labeled_many_960_960.yaml', name_run='truck_labeled_many_960_960')
    train_model(yaml_file='truck_labeled_many_640_640/truck_labeled_many_640_640.yaml',
                name_run='truck_labeled_many_640_640')
    train_model(yaml_file='truck_labeled_many_320_320/truck_labeled_many_320_320.yaml',
                name_run='truck_labeled_many_320_320')
    train_model(yaml_file='truck_labeled_many/truck_labeled_many.yaml',
                name_run='truck_labeled_many_original')


    #analyze()


