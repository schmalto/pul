import os
import pathlib
import random
import shutil

'''
File structure:
    - images
        - 1.jpg
        - 2.jpg
    - labels
        - 1.txt
        - 2.txt
    dataset.yaml
'''

def get_files(folder):
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

def make_val_train_folder(root):
    val_labels = os.path.join(root, "labels/val")
    val_images = os.path.join(root, "images/val")
    train_labels = os.path.join(root, "labels/train")
    train_images = os.path.join(root, "images/train")
    if not os.path.isdir(val_labels):
        os.mkdir(val_labels)
    if not os.path.isdir(val_images):
        os.mkdir(val_images)
    if not os.path.isdir(train_labels):
        os.mkdir(train_labels)
    if not os.path.isdir(train_images):
        os.mkdir(train_images)


def move_files_to_folder(files, folder):
    parent = pathlib.Path(folder).parent.resolve()
    for file in files:
        shutil.move(os.path.join(parent,file), folder)

def get_labels_images(root, split):
    random.seed(42)
    val_labels = []
    images = get_files(os.path.join(root, "images"))
    labels = get_files(os.path.join(root, "labels"))
    test_amount = int(len(labels) * split)


    val_images = random.sample(images, test_amount)
    train_images = [x for x in images if x not in val_images]

    for z in val_images:
        y = z
        replace_str = y.split(".")[-1]
        y = y.replace(replace_str, "txt")
        if y in labels:
            val_labels.append(y)
            #val_images.remove(z)

    train_labels = [x for x in labels if x not in val_labels]
    
    return val_labels, train_labels, val_images, train_images


def main(yaml_file, split=0.2):
    root = pathlib.Path(yaml_file).parent.resolve()
    make_val_train_folder(root)
    val_labels, train_labels, val_images, train_images = get_labels_images(root, split)
    #print(set(val_images).intersection(set(train_images)))
    move_files_to_folder(val_labels, os.path.join(root, "labels/val"))
    move_files_to_folder(train_labels, os.path.join(root, "labels/train"))
    move_files_to_folder(val_images, os.path.join(root, "images/val"))
    move_files_to_folder(train_images, os.path.join(root, "images/train"))
    
    return 0


if __name__ == "__main__":
    main("/home/tobias/git_ws/datasets/skysat_more_data/skysat_more_data.yaml")