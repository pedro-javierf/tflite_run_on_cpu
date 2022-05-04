# Run pyCoral model inference on CPU rather than TPU
# Pedro Javier Fernández - 2022

import time
import numpy as np
#import tensorflow as tf
from PIL import Image
import tflite_runtime.interpreter as tflite
#cambiamos tf.lite por tflite

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="mobilenet_v2_1.0_224_inat_bird_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# NxHxWxC, H:1, W:2
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
img = Image.open("parrot.jpg").resize((width, height))
# add N dim
input_data = np.expand_dims(img, axis=0)
interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.uint8))

# Input data requires preprocessing
#scale = 1
#zero_point = 0
#mean = 128.0
#std = 128.0
#normalized_input = (np.asarray(img) - mean) / (std * scale) + zero_point
#np.clip(normalized_input, 0, 255, out=normalized_input)
#interpreter.set_tensor(input_details[0]['index'], normalized_input.astype(np.uint8))


# Test the model on random input data.
#input_shape = input_details[0]['shape']
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8) #np.float32
#interpreter.set_tensor(input_details[0]['index'], input_data)


# TODO: Cargar labels

# Lanzamos la inferencia en la CPU
start = time.perf_counter()
interpreter.invoke()
inference_time = time.perf_counter() - start
#classes = classify.get_classes(interpreter, args.top_k, args.threshold)
print('%.1fms' % (inference_time * 1000))

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
#output_data = interpreter.get_tensor(output_details[0]['index'])
#print(output_data)

print(interpreter.get_tensor(output_details[0]['index']))
#print(output_details)

# TODO: Sacar output en formato legible (utilizando las labels)
