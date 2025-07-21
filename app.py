

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image


model = load_model('mask_detector_model.h5')


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


st.set_page_config(page_title="Face Mask Detector")

st.title("😷 Face Mask Detection Web App")
st.write("Upload an image and check if the person is wearing a mask.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def predict_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, 1.1, 4)

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (128, 128))
        face_normalized = face_resized / 255.0
        face_reshaped = np.expand_dims(face_normalized, axis=0)

        result = model.predict(face_reshaped)[0][0]
        label = "Mask" if result < 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert('RGB'))
    output_image = predict_mask(image_np)

    st.image(output_image, channels="BGR", caption="Processed Image")
