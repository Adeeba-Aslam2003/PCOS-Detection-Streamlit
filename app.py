import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
import gdown

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="PCOS Detection System", layout="centered")

# -------------------------------------------------
# LOGIN SYSTEM
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
# PROJECT OVERVIEW SECTION (FYP Requirement)
# -------------------------------------------------
st.title("PCOS Detection System 🩺")

st.markdown("""
### 📊 **Project Overview**

This system uses **Deep Learning (Convolutional Neural Networks)** to detect  
**Polycystic Ovary Syndrome (PCOS)** from pelvic ultrasound images.

It performs:
- Image preprocessing  
- CNN-based classification  
- Grad-CAM heatmap explainability  
- Health recommendations  

> ⚠️ **Note:** Academic project — not a medical diagnostic tool.
""")

st.markdown("---")

# -------------------------------------------------
# DATASET & MODEL INFO SECTION
# -------------------------------------------------
st.subheader("📂 Dataset & Model Information")

st.markdown("""
### 📁 **Dataset Details**
- Two classes: **PCOS** and **Normal**
- Ultrasound pelvic images
- Resized to **224 × 224**
- Augmentation used during training:
  - Rotation
  - Zoom
  - Horizontal & vertical flipping
- Normalization applied using **Rescaling(1/255)** inside the model

### 🧠 **CNN Model Architecture**
- Input → Rescaling
- Conv2D + MaxPooling2D (x3)
- Flatten
- Dense layer
- Dropout layer
- Final Sigmoid Output Layer

This structure is ideal for ultrasound classification tasks.
""")

st.markdown("---")


# -------------------------------------------------
# MODEL LOADING (Your original working code)
# -------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "pcos_model.keras"
    drive_url = "https://drive.google.com/uc?id=1d0jvYwVn-2fGBvq8bp5iTII7VbZQikBB"

    if not os.path.exists(model_path):
        st.warning("Downloading PCOS model... (first time only)")
        gdown.download(drive_url, model_path, quiet=False)

    model = tf.keras.models.load_model(model_path)

    # Warm-up call
    dummy = tf.zeros((1, 224, 224, 3))
    model.predict(dummy)

    return model

def get_model():
    return load_model()


# -------------------------------------------------
# IMAGE PREPROCESSING
# -------------------------------------------------
def preprocess_image(image: Image.Image, img_size=(224, 224)):
    img = image.convert("RGB")
    img = img.resize(img_size)

    # IMPORTANT: Do NOT divide by 255 here (model already has Rescaling)
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)

    return arr, np.array(img)


# -------------------------------------------------
# AUTO-DETECT LAST CONV LAYER
# -------------------------------------------------
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found!")


# -------------------------------------------------
# GRAD-CAM IMPLEMENTATION
# -------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_name):

    # rebuild model to avoid "sequential not called" error
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    x = inputs
    last_conv_output = None

    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_name:
            last_conv_output = x

    grad_model = tf.keras.Model(inputs, [last_conv_output, x])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        loss = preds[:, pred_index]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))

    conv_out = conv_out[0]

    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


# -------------------------------------------------
# HEATMAP OVERLAY
# -------------------------------------------------
def overlay_heatmap(original_img, heatmap, alpha=0.4):
    h, w, _ = original_img.shape
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    final = cv2.addWeighted(heatmap, alpha, original_img, 1 - alpha, 0)
    return final


# -------------------------------------------------
# MAIN DETECTION INTERFACE
# -------------------------------------------------
st.title("🩺 PCOS Detection from Pelvic Ultrasound")
st.write("Upload an ultrasound image to begin:")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("Analyze"):
        with st.spinner("Analyzing ultrasound..."):

            model = get_model()
            img_array, img_display = preprocess_image(image)

            # ----- PREDICTION -----
            prob = float(model.predict(img_array)[0][0])
            label = "PCOS likely" if prob >= 0.5 else "PCOS unlikely"

            st.subheader("Prediction Result")
            st.write(f"**Result:** {label}")
            st.write(f"**Probability (PCOS):** {prob:.2f}")

            # ----- GRAD-CAM -----
            try:
                last_conv = find_last_conv_layer(model)
                heatmap = make_gradcam_heatmap(img_array, model, last_conv)

                bgr_img = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
                final = overlay_heatmap(bgr_img, heatmap)
                final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

                st.subheader("Grad-CAM Heatmap")
                st.image(final_rgb)
            except Exception as e:
                st.error(f"Heatmap Error: {e}")


            # -------------------------------------------------
            # DETAILED INTERPRETATION & RECOMMENDATIONS
            # -------------------------------------------------
            st.markdown("---")
            st.subheader("📌 Interpretation & Health Recommendations")

            if label == "PCOS likely":
                st.markdown("### 🩺 Why the Model Detected PCOS")
                st.write("""
- Multiple small follicles  
- String-of-pearls ovarian pattern  
- Enlarged ovary  
- Dense stromal region  
                """)

                st.markdown("### 🌿 What To Do Next")
                st.write("""
**Lifestyle & Exercise:**
- 30–45 mins walking/jogging  
- Yoga (Butterfly, Cobbler, Surya Namaskar)  
- Avoid long sitting hours  

**Diet:**
- Greens, oats, lentils, eggs, paneer  
- Walnuts, almonds, olive oil  
- Avoid sugar, junk food, packaged snacks  

**Medical Advice:**
- Track menstrual cycles  
- Consult gynecologist  
- Hormonal tests if symptoms persist  
                """)

            else:
                st.markdown("### 😊 PCOS Unlikely")
                st.write("""
This image does not show major PCOS indicators.

**Maintain Good Health:**
- Balanced diet  
- 30 minutes exercise daily  
- Avoid stress  
- Sleep 7–8 hours  
- Regular checkups yearly  
                """)

st.markdown("---")
st.caption("Built with TensorFlow, Keras, OpenCV & Streamlit.")
