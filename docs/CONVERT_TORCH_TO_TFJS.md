run convert_model_to_tf.py
run onnx2tf -i model.onnx -osd 
run tensorflowjs_converter --input_format tf_saved_model --output_format tfjs_graph_model saved_model ./models/tfjs_model 
