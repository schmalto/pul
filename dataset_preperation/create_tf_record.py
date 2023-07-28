import sys
assert sys.version_info[0] == 3 and sys.version_info[1] == 9 and sys.version_info[1] >= 1, "Python version must be 3.9 (Because of the modelhub from tensorflow, depending on a bugged version of datasets (see https://github.com/huggingface/datasets/issues/5230), if tf changes the dependency this line can be removed)"
# Creates a .tfrecord file from a COCO dataset.


import tensorflow_datasets as tfds
import tensorflow as tf
import subprocess

def run_create_coco_tf_record(root_dir):
    root_dir = "/home/tobias/git_ws/pul/datasets/skysat_960_960_coco"
    for set in ['train', 'val', 'test']:
        num_shards = 1
        image_dir = root_dir + "/images/" + set
        object_annotations_file = root_dir + "/" + set + ".json"
        output_file_prefix = "/home/tobias/git_ws/pul/datasets/skysat_960_960_tf/" + set + "/"
        caption_annotations_file = None
        panoptic_annotations_file = None
        panoptic_masks_dir = None
        image_info_file = None


        command = [
            "python3.9",
            "tf_vision/models/official/vision/data/create_coco_tf_record.py",]
            
        if panoptic_annotations_file:
            command.append("--panoptic_annotations_file=" + panoptic_annotations_file)
        if panoptic_masks_dir:
            command.append("--panoptic_masks_dir=" + panoptic_masks_dir)
        if output_file_prefix:
            command.append("--output_file_prefix=" + output_file_prefix)
        if num_shards:
            command.append("--num_shards=" + str(num_shards))
        if caption_annotations_file:
            command.append("--caption_annotations_file=" + caption_annotations_file)
        if object_annotations_file:
            command.append("--object_annotations_file=" + object_annotations_file)
        if image_info_file:
            command.append("--image_info_file=" + image_info_file)
        if image_dir:
            command.append("--image_dir=" + image_dir)

        subprocess.run(command)




# Call the function to run the script
if __name__ == "__main__":
    run_create_coco_tf_record("/home/tobias/git_ws/pul/datasets/skysat_960_960_coco")
