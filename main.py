import streamlit as st
from keras.models import load_model
from PIL import Image
from util import classify

# title
st.title('Pneumonia Classifier')

# header
st.header('Please upload a chest X-ray image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./model/pneumonia_keras_model.h5')

#class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()
print(class_names)

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_container_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write(f"Prediction: {class_name}")