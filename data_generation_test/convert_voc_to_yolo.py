from pylabel import importer

# dataset = importer.ImportVOC(
#     path="example_set/annotations", path_to_images="example_set/images")

import os


from IPython.display import Image, display




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


def correct_class_number():
    labels_path = ""
    correct_by = 1338230
    for file in os.listdir(labels_path):
        with open(os.path.join(labels_path, file), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(" ")
                line[0] = str(int(line[0]) - correct_by)
                line = " ".join(line)
                print(line)
                with open(os.path.join(labels_path, file), "w") as f:
                    f.write(line)
