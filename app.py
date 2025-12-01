import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
import gdown

st.set_page_config(page_title="PCOS Detection", layout="centered")


# =========================================================
#                MODEL LOADING (GOOGLE DRIVE)
# =========================================================
@st.cache_resource
def load_model():
    model_path = "pcos_model.h5"

    # If model file does NOT exist → download it from Google Drive
    if not os.path.exists(model_path):
        drive_url = "https://drive.google.com/uc?id=1VHnTaxeJ5eahbm9XgbYLax5cS85e0nJZ"
        st.warning("Downloading PCOS model from Google Drive... (only first time)")
        gdown.download(drive_url, model_path, quiet=False)

    # Load Keras model
    model = tf.keras.models.load_model(model_path)

    # IMPORTANT: Build the model graph for Grad-CAM (do NOT use predict)
    dummy_input = tf.zeros((1, 224, 224, 3))
    _ = model(dummy_input)     # <<< FIXES THE GRAD-CAM ERROR

    return model


def get_model():
    return load_model()



# =========================================================
#                  IMAGE PREPROCESSING
# =========================================================
def preprocess_image(image: Image.Image, img_size=(224, 224)):
    img = image.convert("RGB")
    img = img.resize(img_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, np.array(img)



# =========================================================
#                  GRAD-CAM IMPLEMENTATION
# =========================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv"):

    # Get last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # Model that maps input → activations + predictions
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    # Gradient calculation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)

    # Channel-wise gradient mean
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()



def overlay_heatmap(original_img, heatmap, alpha=0.4):
    h, w, _ = original_img.shape
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    output = cv2.addWeighted(heatmap, alpha, original_img, 1 - alpha, 0)
    return output



# =========================================================
#                        LOGIN PAGE
# =========================================================
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



# =========================================================
#                      MAIN APPLICATION UI
# =========================================================
st.title("🩺 PCOS Detection from Pelvic Ultrasound")
st.write("**Educational demo only – not for clinical use.**")

uploaded_file = st.file_uploader(
    "Upload pelvic ultrasound image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze"):
        with st.spinner("Running model..."):

            model = get_model()
            img_array, display_img = preprocess_image(image)

            # ---- Model Prediction ----
            prob = float(model(img_array)[0][0])
            label = "PCOS likely" if prob >= 0.5 else "PCOS unlikely"

            st.subheader("Prediction")
            st.write(f"**Result:** {label}")
            st.write(f"**Probability (PCOS):** {prob:.2f}")

            # ---- Grad-CAM ----
            try:
                heatmap = make_gradcam_heatmap(
                    img_array,
                    model,
                    last_conv_layer_name="last_conv"
                )

                display_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
                cam_img = overlay_heatmap(display_bgr, heatmap)
                cam_img_rgb = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)

                st.subheader("Model Focus Heatmap (Grad-CAM)")
                st.image(cam_img_rgb, use_column_width=True)

            except Exception as e:
                st.warning(f"Could not generate heatmap: {e}")


st.markdown("---")
st.caption("Built with TensorFlow, OpenCV, Keras, and Streamlit")
