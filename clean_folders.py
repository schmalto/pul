import os
import shutil
from pathlib import Path
from PIL import Image
import ntpath

# move file to its parent directory
def move_file_to_parent(file):
    shutil.move(file, Path(file).parent.parent)

# get all subfolders in the current directory
def get_subfolders(folder):
    return [x[0] for x in os.walk(folder)]

# get all files in the current directory
def get_files(folder):
    return [os.path.join(folder, fn) for fn in next(os.walk(folder))[2]]

def move_files_from_folder_to_folder(source, destination):
    files = get_files(source)
    for file in files:
        shutil.move(file, destination)


def move_files_to_parent(folder):
        for folder in get_subfolders(folder):
            for file in get_files(folder):
                move_file_to_parent(file)
            try:
                os.rmdir(folder)
            except Exception as e:
                pass

def get_image_names(folder):
    folder = "/home/tobias/git_ws/pul/runs/detect"
    img_names = []
    for file in get_files(folder):
        if file.endswith('.jpg'):
            if ntpath.basename(file).split('_')[0] not in img_names:
                img_names.append(ntpath.basename(file).split('_')[0])
    return img_names

def construct_image(folder, img_name, save_location):
    image_arr = [Image.open(x) for x in get_files(folder) if x.find(img_name) != -1]
    rows, cols = get_dimensions(image_arr)
    x_base, y_base = image_arr[0].size
    x_base = int(x_base)
    y_base = int(y_base)
    image = Image.new('RGB', ((cols * x_base), (rows * y_base)))
    x_offset = 0
    y_offset = 0
    for row in range(rows):
        for col in range(cols):
            img = image_arr[row*cols + col]
            image.paste(img, (x_offset, y_offset))
            x_offset += img.size[0]
        y_offset += img.size[1]
        x_offset = 0
    image.save(os.path.join("/home/tobias/git_ws/pul/", save_location,img_name + '.jpg'))
    for img in image_arr:
        os.remove(os.path.join(folder, ntpath.basename(img.filename)))
    #return image

def get_dimensions(images):
    row_arr = []
    col_arr = []
    for image in images:
        row_arr.append(ntpath.basename(image.filename).split('_')[1].split('-')[1])
        col_arr.append(ntpath.basename(image.filename).split('_')[1].split('-')[3].split('.')[0])
    return int(max(row_arr)), int(max(col_arr))

def combine_images(folder, save_location='combined_images'):
    for image_name in get_image_names(folder):
        construct_image(folder, image_name, save_location)



def main():
    combine_images("/home/tobias/git_ws/pul/runs/detect", 'combined_images')
    return 0

if __name__ == '__main__':
    main()