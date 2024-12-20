import base64
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

def set_background(image_file):...
    # empty
    



def classify(image, model, class_names):

    # convert image to 224x224
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) -1

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    prediction = model.predict(data)
    # index = np.argmax(prediction)
    index = 0 if prediction[0][0] > 0.99 else 1
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score