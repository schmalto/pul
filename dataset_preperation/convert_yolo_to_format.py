from pylabel import importer

# dataset = importer.ImportVOC(
#     path="example_set/annotations", path_to_images="example_set/images")

import os
import json

import shutil
import errno

from IPython.core.display import Image
from IPython.display import display



def convert_dataset():

    img_path = "/home/tobias/git_ws/pul/data_generation_test/datasets/example_coco/images"
    annotation_path="/home/tobias/git_ws/pul/data_generation_test/datasets/example_coco/COCO_Tokyo Japan.json"
    test = os.listdir(img_path)

    for item in test:
        if item.endswith(".jpg___fuse.png"):
            os.remove(os.path.join(img_path, item))


    dataset = importer.ImportCoco(annotation_path, path_to_images=img_path)


    # display(dataset.visualize.ShowBoundingBoxes(100))
    # for data in dataset.df:
    #     display(dataset.visualize.ShowBoundingBoxes(data["img_filename"]))
    #dataset.visualize.ShowBoundingBoxes(1)

    dataset.export.ExportToYoloV5()


def convert_dataset_voc():
    path_to_annotations = "/home/tobias/git_ws/pul/datasets/skysat_960_960 copy/labels"
    path_to_images = "/home/tobias/git_ws/pul/datasets/skysat_960_960 copy/images"
    classes = ["truck", "car"]
    export_dir = "/home/tobias/git_ws/pul/datasets/skysat_960_960_voc"
    dataset = importer.ImportYoloV5(path=path_to_annotations, path_to_images=path_to_images, cat_names=classes)
    print(f"Number of images: {dataset.analyze.num_images}")
    print(f"Number of classes: {dataset.analyze.num_classes}")
    print(f"Classes:{dataset.analyze.classes}")
    print(f"Class counts:\n{dataset.analyze.class_counts}")
    dataset.export.ExportToVoc(export_dir)

def convert_dataset_coco():
    import_dir = "/home/tobias/git_ws/pul/datasets/skysat_960_960"
    export_dir = "/home/tobias/git_ws/pul/datasets/skysat_960_960_coco"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    for set in ['train', 'val', 'test']:
        path_to_annotations = import_dir + "/labels/" + set
        path_to_images = import_dir + "/images/" + set
        classes = ["truck", "car"]
        save_path =  export_dir + "/" + set
        dataset = importer.ImportYoloV5(path=path_to_annotations, path_to_images=path_to_images, cat_names=classes)
        dataset.export.ExportToCoco(save_path)
    fix_coco_dataset(export_dir)


def correct_class_number():
    labels_path = "/home/tobias/git_ws/pul/data_generation_test/training/labels"
    correct_by = 1338230
    for file in os.listdir(labels_path):
        with open(os.path.join(labels_path, file), "r") as f:
            lines = f.readlines()
            with open(os.path.join(labels_path, file), "w") as f:
                f.write("")
            for line in lines:
                line = line.split(" ")
                line[0] = str(int(line[0]) - correct_by)
                line = " ".join(line)
                with open(os.path.join(labels_path, file), "a") as f:
                    f.write(line)

def fix_coco_dataset(path_to_dataset_root):
    image_path = ""
    for set in ['train', 'val', 'test']:
        os.rename(os.path.join(path_to_dataset_root, set), os.path.join(path_to_dataset_root, set + ".json"))
    with open(os.path.join(path_to_dataset_root, "train.json"), "r") as f:
        train = json.load(f)
    image_path = train["images"][0]["folder"].split("/")[:-1]
    image_path = "/".join(image_path)
    copy_images_from_yolo_coco(image_path, path_to_dataset_root)
    rename_image_path_json(path_to_dataset_root)
        
def copy_images_from_yolo_coco(image_train_dir, coco_root_dir):
    dest = os.path.join(coco_root_dir, "images")
    try:
        shutil.copytree(image_train_dir, dest)
    except OSError as e:
        print(e)
        if e.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(image_train_dir, dest)
    return 0

def rename_image_path_json(path_to_dataset_root):
    for set in ['train', 'val', 'test']:
        with open(os.path.join(path_to_dataset_root, set + ".json"), "r") as f:
            dataset = json.load(f)
        for image in dataset["images"]:
            image["folder"] = path_to_dataset_root + "/images/" + set
        for annotation in dataset["annotations"]:
            annotation["iscrowd"] = int(annotation["iscrowd"]) 
        with open(os.path.join(path_to_dataset_root, set + ".json"), "w") as f:
            json.dump(dataset, f, indent=4)
    return 0
    


if __name__ == "__main__":
    # convert_dataset()
    # correct_class_number()
    # convert_dataset_voc()
    convert_dataset_coco()
    