import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("📸 YOLOv5 Detection using st.camera_input")

# Take a photo from webcam (browser-supported)
img_file = st.camera_input("Take a picture")

if img_file:
    # Load YOLOv5 model (first time will download weights)
    model = YOLO("yolov5s.pt")  # You can also use yolov8s.pt

    # Load image from camera input
    img = Image.open(img_file)
    img_np = np.array(img)

    # Run inference
    results = model(img_np)

    # Plot results
    result_img = results[0].plot()
    st.image(result_img, caption="Detected Objects", use_column_width=True)
