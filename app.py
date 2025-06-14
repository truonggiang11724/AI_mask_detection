import streamlit as st
from PIL import Image
import torch
import os
import subprocess
import sys

# --- Clone YOLOv5 nếu chưa có thư mục ---
if not os.path.exists("yolov5"):
    with st.spinner("Đang tải YOLOv5 lần đầu..."):
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
sys.path.append("yolov5")  # Đảm bảo YOLOv5 trong sys.path

# Load mô hình
@st.cache_resource
def load_model():
    model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
    return model

model = load_model()

# Giao diện
st.title("AI kiểm tra đeo khẩu trang đúng cách 😷")
st.write("Tải ảnh lên để mô hình phân tích")

# Upload ảnh
uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã tải lên', use_container_width=True)

    # Dự đoán
    with st.spinner("Đang xử lý..."):
        results = model(image)
        results.save(save_dir="output")

    # Hiển thị ảnh kết quả
    parent_dir = "./"
    output_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d)) and d.startswith("output")]
    output_dirs.sort(key=lambda x: int(x.replace("output", "") if x != "output" else 0))
    latest_dir = os.path.join(parent_dir, output_dirs[-1])

    for filename in os.listdir(latest_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(latest_dir, filename)
            st.image(image_path, caption="Kết quả nhận diện", use_container_width=True)
