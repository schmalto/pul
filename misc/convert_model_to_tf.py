import torch
import torchvision
import os
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf


def main():
    base_path = os.getcwd()
    model_weigths = 'skysat_google_combined_overfitted.pt'
    save_name = 'skysat_google_combined_overfitted.onnx'
    save_name_tf = 'skysat_google_combined_overfitted.pb'
    save_location = 'runs/models/'

    #Load a pretrained PyTorch model
    model_torch = torch.hub.load('ultralytics/yolov5', 'custom',
                                path= base_path + '/runs/models/' + model_weigths, force_reload=True).to('cpu')

    # Export the PyTorch model to ONNX
    torch.onnx.export(model_torch,               # model being run
                    torch.randn(1, 3, 224, 224).to('cpu'), # dummy input (required)
                    save_name,   # where to save the model (can be a file or file-like object)
                    export_params=True) # store the trained parameter weights inside the model file
    
    model_onnx = onnx.load(save_name)
    onnx.checker.check_model(model_onnx)
    


if __name__ == '__main__':
    main()