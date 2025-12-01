import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
import gdown

st.set_page_config(page_title="PCOS Detection", layout="centered")


# ---------------- MODEL DOWNLOADING & LOADING ----------------
@st.cache_resource
def load_model():

    model_path = "pcos_model.keras"

    # Download only if missing
    if not os.path.exists(model_path):
        st.warning("Downloading PCOS model from Google Drive… (one-time download)")
        drive_url = "https://drive.google.com/uc?id=1qI08snWzWzq3IEGGKHZlU0lkSnAC1m_O"
        gdown.download(drive_url, model_path, quiet=False)

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Build model so layers have defined shapes
    dummy_input = tf.zeros((1, 224, 224, 3))
    model.predict(dummy_input)

    return model


def get_model():
    return load_model()


# ---------------- PREPROCESSING ----------------
def preprocess_image(image: Image.Image, img_size=(224, 224)):
    img = image.convert("RGB")
    img = img.resize(img_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, np.array(img)


# ---------------- GRAD-CAM HEATMAP ----------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv"):

    try:
        last_conv = model.get_layer(last_conv_layer_name)
    except:
        raise ValueError(f"Layer '{last_conv_layer_name}' not found. Check model.summary().")

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, 0]     # class 0 = PCOS

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    h, w, _ = original_img.shape
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap, alpha, original_img, 1 - alpha, 0)


# ---------------- LOGIN SYSTEM ----------------
VALID_USER = "doctor"
VALID_PASS = "12345"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def login_page():
    st.title("🔐 PCOS Detection Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u == VALID_USER and p == VALID_PASS:
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password")


if not st.session_state.logged_in:
    login_page()
    st.stop()


# ---------------- MAIN UI ----------------
st.title("🩺 PCOS Detection from Pelvic Ultrasound")
st.write("**Educational demo — not for clinical use.**")

uploaded = st.file_uploader("Upload Ultrasound Image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze"):
        with st.spinner("Running model…"):
            model = get_model()
            img_array, disp = preprocess_image(image)

            prob = float(model.predict(img_array)[0][0])
            label = "PCOS likely" if prob >= 0.5 else "PCOS unlikely"

            st.subheader("Prediction")
            st.write(f"**Result:** {label}")
            st.write(f"**Probability (PCOS):** {prob:.2f}")

            # HEATMAP
            try:
                heatmap = make_gradcam_heatmap(img_array, model)
                disp_bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
                output = overlay_heatmap(disp_bgr, heatmap)
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

                st.subheader("Heatmap (Grad-CAM)")
                st.image(output, use_column_width=True)

            except Exception as e:
                st.error(f"🔥 Heatmap error: {e}")


st.markdown("---")
st.caption("Built with ❤️ using TensorFlow, Streamlit & OpenCV")

