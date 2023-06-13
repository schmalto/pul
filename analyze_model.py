import torch
import os
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import torchvision.transforms as transforms
from torchvision.ops import box_convert
from clean_folders import move_files_to_parent, combine_images
import pandas as pd
from termcolor import colored
from pathlib import Path


def visualize():
     image = Image.open(
         '/home/tobias/git_ws/pul/datasets/cars_trucks_patched/images/Screenshot 2022-09-02 182050_0.jpg')
     transform = transforms.Compose([
         transforms.PILToTensor()
     ])
     img_tensor = transform(image)

     boxes = box_convert([0.389062, 0.954688, 0.03125, 0.01875], 'xywh', 'xyxy')

     result = draw_bounding_boxes(img_tensor, boxes, width=5, colors=['red', 'blue'])

def save_count(df, save_location):
    filename = Path(os.path.join(save_location, 'count.txt'))
    cars = (df.name.values == 'car').sum()
    trucks = (df.name.values == 'truck').sum()
    print(colored('Cars: ' + str(cars), 'green'))
    print(colored('Trucks: ' + str(trucks), 'green'))
    count_up = 0
    while filename.is_file():
        filename = Path(os.path.join(save_location , 'count_'+str(count_up)+'.txt'))
        count_up += 1
    with open(filename, 'w') as f:
        f.write('Cars: ' + str(cars) + '\n')
        f.write('Trucks: ' + str(trucks) + '\n')


def analyze(val_folder='weniger', model_weigths='truck_labeled_many_960_960.pt', base_model='ultralytics/yolov5', save_location='combined_images', original=False):
    print ("[Analyze] Starting analysis.")
    base_path = os.getcwd()
    if not os.path.isdir(os.path.join(base_path , save_location)):
        Path(os.path.join(base_path, save_location)).mkdir(parents=True, exist_ok=True)
    print(os.path.join(base_path, save_location))
    torch.cuda.empty_cache()
    model = torch.hub.load(base_model, 'custom',
                            path= base_path + '/runs/' + model_weigths, force_reload=True)
    model.eval()
    df = pd.DataFrame()
    folder = base_path + '/val_images/' + val_folder
    for file in os.listdir(folder):
        if file.startswith('_'):
            continue
        result = model(folder + '/' + file)
        df = pd.concat([df,result.pandas().xyxy[0]])
        result.save()
    print("[Analyze] Ran interference.")
   
    move_files_to_parent(base_path + '/runs/detect')
    print("[Analyze] Moved all files to parent folder.")
    if not original:
        save_count(df, os.path.join(base_path, save_location))
        print("[Analyze] Counted all class occurence.")
        combine_images(base_path + '/runs/detect', save_location)
        print("[Analyze] Combined all images.")
    else:
        save_count(df, os.path.join(base_path, 'runs/detect'))
        print("[Analyze] Counted all class occurence.")
    print("[Analyze] Done.")


if __name__ == "__main__":
    analyze('__original__', 'truck_labeled_many_original_320.pt', 'ultralytics/yolov5', 'runs/combined_images', original=True)

    