from yolov5 import train
from ultralytics import YOLO
import torch
import os



def train_model(yaml_file='truck_labeled_patched/truck_labeled_patched.yaml', name_run='truck_labeled_patched'):
     # model = YOLO('yolov8n.pt', device='cuda')

     # model.train(data="", epochs="")
    torch.cuda.empty_cache()
    data_name = base_path + '/datasets/' + yaml_file
    train.run(data=data_name,imgsz=640, weights='yolov5x.pt', batch_size=16, epochs=2000, project='./runs/', name=name_run)



if __name__ == "__main__":
    base_path = os.getcwd()
    train_model(yaml_file='truck_labeled_many_1080_1080/truck_labeled_many_1080_1080.yaml',
                  name_run='truck_labeled_many_1080_1080')
    # train_model(yaml_file='truck_labeled_many_960_960/truck_labeled_many_960_960.yaml', name_run='truck_labeled_many_960_960')
    # train_model(yaml_file='truck_labeled_many_640_640/truck_labeled_many_640_640.yaml',
    #             name_run='truck_labeled_many_640_640')
    # train_model(yaml_file='truck_labeled_many_320_320/truck_labeled_many_320_320.yaml',
    #             name_run='truck_labeled_many_320_320')
    # train_model(yaml_file='truck_labeled_many/truck_labeled_many.yaml',
    #             name_run='truck_labeled_many_original')


