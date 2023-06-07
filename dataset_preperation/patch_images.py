from typing import Union
from pathlib import Path
import numpy as np
from skimage import io
from tqdm import tqdm

from utils_patching import xywh_to_xyxy, xyxy_to_xywh


class Patcher:
    def __init__(
        self, path_to_save: Union[Path, str], base_path: Union[Path, str], valid_or_train: str
    ) -> None:

        self.path_to_save = path_to_save
        self.base_path = base_path
        self.valid_or_train = valid_or_train
        self.create_folders()

    def create_folders(self) -> None:
        self.path_to_save.mkdir(parents=True, exist_ok=True)
        (self.path_to_save / "images"/ self.valid_or_train).mkdir(parents=True, exist_ok=True)
        (self.path_to_save / "labels" / self.valid_or_train).mkdir(parents=True,exist_ok=True)

    def patch_sampler(
        self,
        img: np.ndarray,
        fname: str,
        patch_width: int = 1080,
        patch_height: int = 1080,
    ) -> None:

        # Get image size and stop if it's smaller than patch size
        img_height, img_width, _ = img.shape
        if img_height < patch_height or img_width < patch_width:
            return

        # Get number of horisontal and vertical patches
        horis_ptch_n = int(np.ceil(img_width / patch_width))
        vertic_ptch_n = int(np.ceil(img_height / patch_height))
        y_start = 0

        # Prepare labels
        label_path = (self.base_path / "labels" / self.valid_or_train / fname).with_suffix(".txt")
        with open(label_path) as f:
            lines = f.readlines()

        all_labels = xywh_to_xyxy(lines, *img.shape[:2])
        #####

        # Run and create every crop
        for v in range(vertic_ptch_n):
            x_start = 0

            for h in range(horis_ptch_n):
                idx = v * horis_ptch_n + h

                x_end = x_start + patch_width
                y_end = y_start + patch_height

                # Get the crop
                cropped = img[y_start:y_end, x_start:x_end]

                # Get labels patched
                cur_labels = []
                for label in all_labels:
                    cur_label = label.copy()

                    # Check if label is insde the crop
                    if (
                        label[1] > x_start
                        and label[2] > y_start
                        and label[3] < x_end
                        and label[4] < y_end
                    ):

                        # Change scale from original to crop
                        cur_label[1] -= x_start
                        cur_label[2] -= y_start
                        cur_label[3] -= x_start
                        cur_label[4] -= y_start

                        label_yolo = xyxy_to_xywh(
                            cur_label, patch_width, patch_height)
                        cur_labels.append(label_yolo)

                # Save the label file to the disk
                if len(cur_labels):
                    with open(
                            self.path_to_save / "labels" / self.valid_or_train / f"{fname}_{idx}.txt", "a") as f:
                        f.write("\n".join("{} {} {} {} {}".format(*tup)
                                for tup in cur_labels))
                        f.write("\n")
                #####

                    # Save the crop to disk
                    io.imsave(self.path_to_save / "images" / self.valid_or_train /
                          f"{fname}_{idx}.jpg", cropped)

                # Get horisontal shift for the next crop
                x_start += int(
                    patch_width - (patch_width - img_width % patch_width) /
                    (img_width // patch_width)
                )

            # Get vertical shift for the next crop
            y_start += int(
                patch_height - (patch_height - img_height % patch_height) /
                (img_height // patch_height)
            )


def main():
    '''
    base path structure:

    -> dataset
    -----> images (folder with images)
        -----> train (folder with train images)
        -----> val (folder with val images)
    -----> labels (folder with labels)
        -----> train (folder with train labels)
        -----> val (folder with val labels)
    -----> dataset.yaml (file with dataset info)
    '''

    base_path = Path(
        "/home/tobias/git_ws/pul/data_generation_test/datasets/truck_labeled_many")

    # path were you want to save patched dataset
    path_to_save = Path(
        "/home/tobias/git_ws/pul/data_generation_test/datasets/truck_labeled_many_1080_1080")

    


    for split in ['train', 'val']:
        images_folder_path = base_path / "images" / split
        patcher = Patcher(path_to_save, base_path, split)
        for image_path in tqdm(images_folder_path.glob("*")):
            if image_path.name.startswith("."):
                continue
            image = io.imread(image_path)
            fname = image_path.stem
            print("fname: ", fname)
            patcher.patch_sampler(image, fname)

    '''
        ATTENTION: .yaml file is not patched, so you need to copy it from the original dataset
    '''


if __name__ == "__main__":
    main()
