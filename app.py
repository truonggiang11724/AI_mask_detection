import streamlit as st
from PIL import Image
import torch
import os

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
    
    import os
    from PIL import Image

    # Đường dẫn đến thư mục cha chứa các thư mục output
    parent_dir = "./"

    # Tìm thư mục mới nhất (giả sử các thư mục này được tạo theo thứ tự tăng dần)
    output_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d)) and d.startswith("output")]
    output_dirs.sort(key=lambda x: int(x.replace("output", "") if x != "output" else 0))
    latest_dir = os.path.join(parent_dir, output_dirs[-1])

    # Tìm ảnh .jpg trong thư mục đó
    for filename in os.listdir(latest_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(latest_dir, filename)
            st.image(image_path, caption="Kết quả nhận diện", use_container_width=True)