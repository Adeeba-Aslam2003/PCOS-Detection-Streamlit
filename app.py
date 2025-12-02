import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
import gdown

st.set_page_config(page_title="PCOS Detection", layout="centered")


# -----------------------------------------
# 1. Download + Load Model from Google Drive
# -----------------------------------------
@st.cache_resource
def load_model():
    model_path = "pcos_model.keras"
    if not os.path.exists(model_path):
        st.warning("Downloading PCOS model from Google Drive… (first time only)")
        drive_url = "https://drive.google.com/uc?id=1d0jvYwVn-2fGBvq8bp5iTII7VbZQikBB"
        gdown.download(drive_url, model_path, quiet=False)
    model = tf.keras.models.load_model(model_path)
    # Force model build (needed for Grad-CAM)
    dummy = tf.zeros((1, 224, 224, 3))
    model.predict(dummy)
    return model

def get_model():
    return load_model()


# -----------------------------------------
# 2. Image Preprocessing
# -----------------------------------------
def preprocess_image(image: Image.Image, img_size=(224, 224)):
    img = image.convert("RGB")
    img = img.resize(img_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, np.array(img)


# -----------------------------------------
# 3. Grad-CAM Heatmap Function
# -----------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv"):
    last_conv = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    h, w, _ = original_img.shape
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(heatmap, alpha, original_img, 1 - alpha, 0)
    return superimposed


# -----------------------------------------
# 4. Simple Login Page
# -----------------------------------------
VALID_USER = "doctor"
VALID_PASS = "12345"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("🔐 PCOS Detection Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == VALID_USER and pwd == VALID_PASS:
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password")

if not st.session_state.logged_in:
    login_page()
    st.stop()


# -----------------------------------------
# 5. Main UI
# -----------------------------------------
st.title("🩺 PCOS Detection from Pelvic Ultrasound")
st.write("**Educational demo — not for clinical use.**")

uploaded_file = st.file_uploader("Upload pelvic ultrasound image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze"):
        with st.spinner("Running model..."):
            model = get_model()
            img_array, display_img = preprocess_image(image)

            # Prediction
            prob = float(model.predict(img_array)[0][0])
            label = "PCOS likely" if prob >= 0.5 else "PCOS unlikely"

            st.subheader("Prediction")
            st.write(f"**Result:** {label}")
            st.write(f"**Probability:** {prob:.2f}")

            # Grad-CAM
            try:
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv")
                display_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
                cam = overlay_heatmap(display_bgr, heatmap)
                cam_rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

                st.subheader("Model Focus Heatmap (Grad-CAM)")
                st.image(cam_rgb, use_column_width=True)
            except Exception as e:
                st.error(f"🔥 Heatmap error: {e}")

st.markdown("---")
st.caption("Built with TensorFlow, Keras, OpenCV, and Streamlit")
