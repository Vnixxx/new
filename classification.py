import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras import preprocessing

def predict(image):
    model = "best.tflite"
    interpreter = tf.lite.Interpreter(model_path = model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    image = np.array(image.resize((224,224)), dtype=np.float32) 
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probabilities = np.array(output_data[0])
    labels = {0 : "covid", 1 : "others"}
    label_to_probabilities = []
    for i, probability in enumerate(probabilities):
        label_to_probabilities.append([labels[i],float(probability)])
    sorted(label_to_probabilities, key=lambda element: element[1])
    result = { 'covid' : 0 , 'others' : 0 }
    result = f"{label_to_probabilities[np.argmax(probability)][0]} with a { (100 * np.max(probabilities)).round(2)} % confidence." 
    
    return result