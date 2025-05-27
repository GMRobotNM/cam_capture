import streamlit as st
import torch
from PIL import Image
import numpy as np

st.title("📸 YOLOv5 Detection with Camera Input")

# Camera input
img_file = st.camera_input("Take a picture")

if img_file is not None:
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Convert image to format suitable for model
    img = Image.open(img_file)
    img_np = np.array(img)

    # Inference
    results = model(img_np)
    results.render()  # Draw boxes on image

    st.image(results.ims[0], caption="Detected objects", use_column_width=True)
