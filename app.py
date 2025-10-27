# ------------------------- #
# 🩻 Pneumonia Detection App
# Developed by Nextide Solutions
# ------------------------- #

import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# 🧩 Fix macOS TensorFlow thread & mutex issues
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ------------------------- #
# ⚙️ Page Configuration
# ------------------------- #
st.set_page_config(page_title="🩻 Pneumonia Detection", page_icon="🧠", layout="wide")

# 🌈 Custom Styling
st.markdown("""
<style>
    /* Background */
    .main {
        background: linear-gradient(135deg, #e0f7fa 0%, #e8eaf6 100%);
        color: #2c3e50;
        font-family: 'Inter', sans-serif;
    }

    /* Title Styling */
    h1 {
        font-size: 42px;
        text-align: center;
        font-weight: 800;
        color: #1a237e;
        margin-bottom: 5px;
    }

    h3 {
        text-align: center;
        font-weight: 400;
        color: #3949ab;
        margin-bottom: 30px;
    }

    /* Upload box */
    .upload-box {
        border: 3px dashed #2196f3;
        border-radius: 12px;
        background-color: #ffffff;
        padding: 35px;
        text-align: center;
        transition: 0.3s ease;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    }
    .upload-box:hover {
        # transform: scale(1.02);
        border-color: #1565c0;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #42a5f5, #1e88e5);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 17px;
        padding: 10px 30px;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #2196f3, #1976d2);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e3f2fd 0%, #bbdefb 100%);
        color: #1a237e;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #42a5f5 !important;
    }

    .upload-box:hover {
        background-color: #e3f2fd;
    }

    /* Footer */
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------- #
# 🩸 Sidebar Info
# ------------------------- #
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=80)
st.sidebar.title("🩸 About the App")
st.sidebar.info("""
This app uses a **Deep Learning CNN model** trained on chest X-ray images  
to detect **Pneumonia** with high accuracy.

Upload an image and get an instant AI-powered diagnosis.
""")

# ------------------------- #
# 📦 Load Model
# ------------------------- #
@st.cache_resource
def load_model():
    model_path = "pneumonia_model_fixed.keras"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()
    return model

model = load_model()

# ------------------------- #
# 🩻 App Title
# ------------------------- #
st.markdown("<h1>🩻 Pneumonia Detection</h1>", unsafe_allow_html=True)
st.markdown("<h3>Upload a Chest X-ray image and let AI analyze it.</h3>", unsafe_allow_html=True)

# ------------------------- #
# 📤 Upload Section
# ------------------------- #
st.markdown("""
    <div class="upload-box" onclick="document.getElementById('file_input').click()">
        📤 <b>Upload a Chest X-ray (.jpg / .png)</b>
    </div>
    <script>
        const fileInput = window.parent.document.querySelector('input[type="file"]');
        if (fileInput) {
            fileInput.id = 'file_input';
            fileInput.style.display = 'none';
        }
    </script>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

# ------------------------- #
# 🔄 Image Preprocessing
# ------------------------- #
def preprocess_image(img):
    img = img.resize((64, 64))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = img_array.astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

# ------------------------- #
# 🔍 Prediction
# ------------------------- #
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    img = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(img, caption="🩺 Uploaded Chest X-ray", use_container_width=True)

    with col2:
        st.write("### 🧠 Analyzing image...")
        img_array = preprocess_image(img)

        with st.spinner("Running deep learning model... please wait"):
            time.sleep(1)
            prediction = model.predict(img_array)
            confidence = float(prediction[0][0])

        st.write("### 📊 Model Prediction:")
        progress = int(confidence * 100 if confidence > 0.5 else (1 - confidence) * 100)
        st.progress(progress)
        time.sleep(0.5)

        if confidence > 0.5:
            st.error(f"🦠 **Pneumonia Detected!**\nConfidence: {confidence * 100:.2f}%")
            st.markdown("⚠️ Please consult a medical professional for further evaluation.")
        else:
            st.success(f"✅ **Normal Lungs Detected**\nConfidence: {(1 - confidence) * 100:.2f}%")
            st.markdown("💨 Everything looks good! No signs of pneumonia detected.")

        st.balloons()

# ------------------------- #
# ⚙️ Footer
# ------------------------- #
st.markdown("---")
st.markdown("""
<center>
👨‍⚕️ Developed by <b>Zuraiz Khan</b>  
Powered by TensorFlow & Streamlit
</center>
""", unsafe_allow_html=True)
