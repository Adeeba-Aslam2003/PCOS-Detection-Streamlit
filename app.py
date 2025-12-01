import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
import gdown

st.set_page_config(page_title="PCOS Detection", layout="centered")

# ------------------------------------------------------
#  🔥 1. LOAD MODEL FROM GOOGLE DRIVE (only first time)
# ------------------------------------------------------
@st.cache_resource
def load_model():

    model_path = "pcos_model.keras"   # final model file name

    # If the model does not exist in Streamlit folder → download it
    if not os.path.exists(model_path):

        st.warning("Downloading PCOS model from Google Drive... (only first time)")

        # Your Drive file ID
        drive_url = "https://drive.google.com/uc?id=1qI08snWzWzq3IEGGKHZlU0lkSnAC1m_O"

        # Download using gdown
        gdown.download(drive_url, model_path, quiet=False)

    # Load the model
    model = tf.keras.models.load_model(model_path)

    # 🔥 Force build model graph (fixes the heatmap issue)
    dummy = tf.zeros((1, 224, 224, 3))
    model.predict(dummy)

    return model


def get_model():
    return load_model()

# ------------------------------------------------------
# 2. PREPROCESS IMAGE
# ------------------------------------------------------
def preprocess_image(image: Image.Image, img_size=(224, 224)):
    img = image.convert("RGB")
    img = img.resize(img_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, np.array(img)

# ------------------------------------------------------
# 3. GRAD-CAM HEATMAP
# ------------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv"):

    # Extract the last conv layer
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except:
        raise ValueError(f"❌ Layer '{last_conv_layer_name}' NOT found in model. "
                         f"Check model.summary()")

    # Build a model that maps input → conv output + predictions
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]   # binary classification (0 = no PCOS, 1 = PCOS)

    # Gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    # Weight channels
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

# ------------------------------------------------------
# 4. APPLY HEATMAP ON ORIGINAL IMAGE
# ------------------------------------------------------
def overlay_heatmap(original_img, heatmap, alpha=0.4):
    h, w, _ = original_img.shape
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(heatmap, alpha, original_img, 1 - alpha, 0)
    return superimposed

# ------------------------------------------------------
# 5. SIMPLE LOGIN
# ------------------------------------------------------
VALID_USER = "doctor"
VALID_PASS = "12345"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("🔐 PCOS Detection Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == VALID_USER and password == VALID_PASS:
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password")

if not st.session_state.logged_in:
    login_page()
    st.stop()

# ------------------------------------------------------
# 6. MAIN APP UI
# ------------------------------------------------------
st.title("🩺 PCOS Detection from Pelvic Ultrasound")
st.write("**Educational demo only – not for clinical use.**")

uploaded_file = st.file_uploader(
    "Upload pelvic ultrasound image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze"):
        with st.spinner("Running model... please wait"):

            model = get_model()
            img_array, display_img = preprocess_image(image)

            # prediction
            prob = float(model.predict(img_array)[0][0])
            label = "PCOS likely" if prob >= 0.5 else "PCOS unlikely"

            st.subheader("Prediction")
            st.write(f"**Result:** {label}")
            st.write(f"**Probability (PCOS):** {prob:.2f}")

            # 🔥 Generate heatmap
            try:
                heatmap = make_gradcam_heatmap(
                    img_array,
                    model,
                    last_conv_layer_name="last_conv"
                )

                display_img_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
                superimposed = overlay_heatmap(display_img_bgr, heatmap)
                superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

                st.subheader("Model Focus Heatmap (Grad-CAM)")
                st.image(superimposed_rgb, use_column_width=True)

            except Exception as e:
                st.error(f"❌ Heatmap failed: {e}")

st.markdown("---")
st.caption("Built with Python, TensorFlow, OpenCV, and Streamlit")
