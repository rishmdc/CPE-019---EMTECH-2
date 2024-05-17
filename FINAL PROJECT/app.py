#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Dropout

st.header("Image Class Predictor")

def main():
    file_uploaded = st.file_uploader("Choose the file", type=['jpg', 'png', 'jpeg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)

def predict_class(image):
    classifier_model = load_model('/content/drive/MyDrive/Colab Notebooks/Final Project/models/weights-improvement-71-0.88.hdf5')
    shape = (128, 128, 3)
    image = image.resize((128, 128))
    test_image = preprocessing.image.img_to_array(image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = ['cheetah', 'lion']

    predictions = classifier_model.predict(test_image)
    scores = softmax(predictions[0]).numpy()
    image_class = class_names[np.argmax(scores)]
    result = f"The image uploaded is: {image_class}"
    return result

if __name__ == "__main__":
    main()

