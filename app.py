import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
import gdown

st.set_page_config(page_title="PCOS Detection", layout="centered")

# -------------------------------------------------
# 🔹 DOWNLOAD MODEL FROM GOOGLE DRIVE (only first time)
# -------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "pcos_model.keras"  # NEW .keras model

    # Your latest Drive link:
    drive_url = "https://drive.google.com/uc?id=1d0jvYwVn-2fGBvq8bp5iTII7VbZQikBB"

    if not os.path.exists(model_path):
        st.warning("Downloading PCOS model from Google Drive... (first time only)")
        gdown.download(drive_url, model_path, quiet=False)

    model = tf.keras.models.load_model(model_path)

    # Optional: warm-up call (not strictly needed with the new Grad-CAM code,
    # but harmless)
    dummy_input = tf.zeros((1, 224, 224, 3))
    _ = model.predict(dummy_input)

    return model


def get_model():
    return load_model()


# -------------------------------------------------
# 🔹 PREPROCESSING  (IMPORTANT: no manual /255 now)
# -------------------------------------------------
def preprocess_image(image: Image.Image, img_size=(224, 224)):
    img = image.convert("RGB")
    img = img.resize(img_size)
    # DO NOT divide by 255 here – model already has a Rescaling(1/255) layer
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)  # shape (1, 224, 224, 3)
    return arr, np.array(img)


# -------------------------------------------------
# 🔹 AUTO-DETECT LAST CONV2D LAYER
# -------------------------------------------------
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model!")


# -------------------------------------------------
# 🔹 GRAD-CAM HEATMAP (robust for Sequential models)
# -------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Rebuilds a small functional graph layer-by-layer from your Sequential model,
    so we never see 'sequential has never been called' again.
    """
    # Determine expected input shape (e.g. (224, 224, 3))
    input_shape = model.input_shape[1:]
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs
    last_conv_output = None

    # Re-run all layers on the new Input tensor
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            last_conv_output = x

    if last_conv_output is None:
        raise ValueError(f"Could not find conv layer {last_conv_layer_name}")

    preds = x

    # Model that outputs both conv feature maps and final predictions
    grad_model = tf.keras.Model(inputs=inputs,
                                outputs=[last_conv_output, preds])

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # remove batch dim
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    h, w, _ = original_img.shape
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(heatmap, alpha, original_img, 1 - alpha, 0)
    return superimposed


# -------------------------------------------------
# 🔹 LOGIN SYSTEM
# -------------------------------------------------
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


# -------------------------------------------------
# 🔹 MAIN APP UI
# -------------------------------------------------
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

            # ---------- Prediction ----------
            prob = float(model.predict(img_array)[0][0])
            label = "PCOS likely" if prob >= 0.5 else "PCOS unlikely"

            st.subheader("Prediction")
            st.write(f"**Result:** {label}")
            st.write(f"**Probability (PCOS):** {prob:.2f}")

            # ---------- Grad-CAM ----------
            try:
                last_conv = find_last_conv_layer(model)

                heatmap = make_gradcam_heatmap(
                    img_array,
                    model,
                    last_conv_layer_name=last_conv,
                )

                display_img_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
                superimposed = overlay_heatmap(display_img_bgr, heatmap)
                superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

                st.subheader("Model Focus Heatmap (Grad-CAM)")
                st.image(superimposed_rgb, use_column_width=True)

            except Exception as e:
                st.error(f"🔥 Heatmap error: {e}")

st.markdown("---")
st.caption("Built with ❤️ using TensorFlow, Keras, OpenCV, and Streamlit")
