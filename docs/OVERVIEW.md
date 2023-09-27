# Overview over the project

## Project structure

config: Contains the configuration files for the whole project. Move to root directory to use them
dataset_preparation: Contains the code for preparing datasets for training and testing
data_generation_test: Contains failed attempts to generate data, will be removed later
datasets: Contains all datasets used for training and testing, this is a git submodule
image_conversion: Contains the code for converting skysat images in tiff format to png format
models: Contains most of the used models. Mostly yolov5 models
regression: Contains the code for training and testing the regression model
runs: Includes the results of the training and testing of the object detection models
tf_vision: Contains code for training and testing object detection models relying on tensorflow !!! NOT WORKING !!!
val_images: Contains images used for validation of the object detection models
yolov5: Contains the code for the training of the yolov5 model. This is a submodule of the original yolov5 repository
misc: Contains some scripts that are not used anymore or don't fit in any other category

## Typical workflow

### Object detection (aka recognizing trucks)

1. Convert the images to png format using the image_conversion script (/image_conversion/convert_tiff_to_png.py)
2. Annotate/Label the images
3. In order to achieve maximal accuracy one should convert the images to appropriate size using dataset_preparation/patch_images.py
4. Create a dataset directory with two subdirectories images and labels
5. Using dataset_preparation/create_val_test_folder.py create the correct folder structure for training and testing
6. Now put a .yaml file in the root directory and specify the paths to the training and testing data, examples are given in the datasets directory
7. Now you can train your own model using the train_yolo.py script in the project root. your model will be saved along some metrics in the runs/ directory
8. For testing your model use the analyze_model.py script in the project root. This will create a .txt file with the results of the testinng and the store the annotated images in the runs/ directory as well. For Validation val_images/ may be used, as it contains images that were not used for training or testing and are available in different resolutions.
9. This data can be used for further analysis. Alternatively, one may want use the pandas dataframe in analyze_model.py to evaluate the data further.

Some recommendations: 
- It is recommended to install the albumentations pip package, as the yolov5 model handles image augmentation directly , when this package is detected. (Also see the script install_gh_albumentations.sh)
- Yolov5 XL is the preferred base model for detection, however the L model provides sufficient accuracy and is much smaller and faster.
- The pip package ultralytics is needed, even though it is not an obvious dependency.
- For more insides into the training process, it is also recommended to use comet.ml, a third party logging service, or to use tensorboard, which is already integrated in the yolov5 structure.

**For the moment the best performing model is google_skysat_combined_extended.pt**

### Regression (predicting the future devolpment of the number of trucks)

Run regression/lstm_own/training.py to train, validate and predict with a lstm model. The model itself is saved in lstm.py
In run the following functions in the script for the desired action:
- train_model(): Trains the model
- evaluate_model(): Validates the model
- predict(): Predicts the future development of the number of trucks

Every step saves its result also to the disk.
