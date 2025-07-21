# 😷 Face Mask Detection using CNN and OpenCV

A real-time face mask detection system built with **TensorFlow**, **OpenCV**, and **Streamlit**. This project uses a Convolutional Neural Network (CNN) to detect whether a person is wearing a face mask or not — either from a **webcam** or via **image upload** in a web interface.

---

## 🚀 Features

- Real-time face detection via webcam
- Detects whether face is wearing a **mask** or **no mask**
- Web app using **Streamlit** for easy UI
- Powered by a custom-trained CNN
- Uses OpenCV's Haar Cascade for face localization

---

## 🖼️ Demo

### ✅ Real-Time Webcam Detection
![Webcam Demo]()

### ✅ Streamlit Web App
![Streamlit Screenshot]()

---

## 📁 Folder Structure

face-mask-detector/
│
├── dataset/ # training data (with_mask / without_mask)
├── mask_detector_model.h5 # saved CNN model
├── haarcascade_frontalface_default.xml
├── train_model.py # train the model
├── real_time_mask_detection.py
├── app.py # Streamlit web app run as ## streamlit run app.py
├── requirements.txt
└── README.md

