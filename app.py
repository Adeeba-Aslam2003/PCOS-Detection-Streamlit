import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
import gdown
import io
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(page_title="PCOS Detection System", layout="centered")

# ---------------------------------------------
# SIMPLE LOGIN (Doctor)
# ---------------------------------------------
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

# ---------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "PCOS Detection", "Results & Evaluation", "About"]
)

# ---------------------------------------------
# MODEL LOADING (from Google Drive)
# ---------------------------------------------
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

# ---------------------------------------------
# UTILS – PREPROCESS, GRAD-CAM, PDF
# ---------------------------------------------
def preprocess_image(image: Image.Image, img_size=(224, 224)):
    img = image.convert("RGB")
    img = img.resize(img_size)
    # DO NOT /255 here – model already has Rescaling(1/255)
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    return arr, np.array(img)


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found!")


def make_gradcam_heatmap(img_array, model, last_conv_name):
    # Rebuild functional graph from sequential model
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    x = inputs
    last_conv_output = None

    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_name:
            last_conv_output = x

    if last_conv_output is None:
        raise ValueError(f"Could not find conv layer {last_conv_name}")

    preds = x
    grad_model = tf.keras.Model(inputs, [last_conv_output, preds])

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    h, w, _ = original_img.shape
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    final = cv2.addWeighted(heatmap, alpha, original_img, 1 - alpha, 0)
    return final


def create_pdf_report(patient_name, label, prob, orig_image_np, heatmap_image_np):
    """
    Generates a PDF report in memory and returns bytes.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "PCOS Detection Report")

    # Date & patient
    c.setFont("Helvetica", 11)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    c.drawString(50, height - 80, f"Generated on: {now_str}")
    if patient_name:
        c.drawString(50, height - 100, f"Patient Name: {patient_name}")
    else:
        c.drawString(50, height - 100, "Patient Name: (Not provided)")

    # Prediction
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height - 130, "Prediction")
    c.setFont("Helvetica", 12)
    c.drawString(60, height - 150, f"Result: {label}")
    c.drawString(60, height - 170, f"Estimated Probability (PCOS): {prob:.2f}")

    # Insert original image (small)
    try:
        orig_img = Image.fromarray(orig_image_np.astype("uint8"))
        orig_reader = ImageReader(orig_img)
        c.drawString(50, height - 210, "Uploaded Ultrasound:")
        c.drawImage(orig_reader, 50, height - 410, width=180, height=180, preserveAspectRatio=True, mask='auto')
    except Exception:
        pass

    # Insert heatmap image
    try:
        heat_img = Image.fromarray(heatmap_image_np.astype("uint8"))
        heat_reader = ImageReader(heat_img)
        c.drawString(260, height - 210, "Model Focus (Grad-CAM):")
        c.drawImage(heat_reader, 260, height - 410, width=180, height=180, preserveAspectRatio=True, mask='auto')
    except Exception:
        pass

    # Recommendations (short)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height - 440, "Summary & Note")
    c.setFont("Helvetica", 11)
    text = c.beginText(50, height - 460)
    if "likely" in label:
        text.textLine("• The model detected PCOS-like features in the ultrasound.")
        text.textLine("• This is an AI-based educational tool; not a medical diagnosis.")
        text.textLine("• Clinical consultation with a gynecologist is strongly recommended.")
    else:
        text.textLine("• The model did not detect strong PCOS patterns in this ultrasound.")
        text.textLine("• However, AI results are not a replacement for medical diagnosis.")
        text.textLine("• For any symptoms, consult a qualified doctor.")
    c.drawText(text)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# ---------------------------------------------
# PAGE 1 – OVERVIEW
# ---------------------------------------------
if page == "Overview":
    st.title("PCOS Detection System 🩺")

    st.markdown("""
    ### 📊 Project Overview

    This system uses **Deep Learning (CNN)** to detect 
    **Polycystic Ovary Syndrome (PCOS)** from pelvic ultrasound images.

    - Upload a pelvic ultrasound image  
    - The CNN model predicts **PCOS likely** or **PCOS unlikely**  
    - Grad-CAM highlights where the model is focusing  
    - The system provides explanation and health recommendations  

    > ⚠️ **Note:** This is an academic project and not a medical diagnostic tool.
    """)

    st.markdown("---")
    st.subheader("📂 Dataset & Model Information")

    st.markdown("""
    **Dataset:**
    - Two classes: **PCOS** and **Normal**
    - Ultrasound images resized to **224×224**
    - Images normalized with a **Rescaling(1/255)** layer
    - Data augmentation used during training (rotation, zoom, flips)

    **CNN Architecture (Simplified):**
    - Input → Rescaling(1/255)
    - Conv2D + MaxPooling2D (x3)
    - Flatten
    - Dense + Dropout
    - Sigmoid Output (binary classification: PCOS vs Normal)
    """)

# ---------------------------------------------
# PAGE 2 – PCOS DETECTION (MAIN APP)
# ---------------------------------------------
elif page == "PCOS Detection":
    st.title("🩺 PCOS Detection from Pelvic Ultrasound")
    st.write("Upload an ultrasound image and analyze it using the trained CNN model.")

    patient_name = st.text_input("Patient Name (optional)")

    uploaded_file = st.file_uploader("Upload Ultrasound Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze"):
            with st.spinner("Analyzing ultrasound..."):
                model = get_model()
                img_array, img_display = preprocess_image(image)

                # ----- PREDICTION -----
                preds = model.predict(img_array)
                prob = float(preds[0][0])
                label = "PCOS likely" if prob >= 0.5 else "PCOS unlikely"

                st.subheader("Prediction Result")
                st.write(f"**Result:** {label}")
                st.write(f"**Estimated Probability (PCOS):** {prob:.2f}")

                # ----- GRAD-CAM -----
                try:
                    last_conv = find_last_conv_layer(model)
                    heatmap = make_gradcam_heatmap(img_array, model, last_conv)

                    bgr_img = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
                    final_overlay = overlay_heatmap(bgr_img, heatmap)
                    final_rgb = cv2.cvtColor(final_overlay, cv2.COLOR_BGR2RGB)

                    st.subheader("Grad-CAM Heatmap (Model Focus)")
                    st.image(final_rgb, use_column_width=True)
                except Exception as e:
                    st.error(f"Heatmap Error: {e}")
                    final_rgb = img_display  # fallback

                # ----- INTERPRETATION & RECOMMENDATIONS -----
                st.markdown("---")
                st.subheader("📌 Interpretation & Health Recommendations")

                if label == "PCOS likely":
                    st.markdown("### 🩺 Why the Model Detected PCOS")
                    st.write("""
- Multiple small follicles and cyst-like structures  
- Possible 'string of pearls' appearance along the ovary  
- Increased ovarian volume / dense central region  

The highlighted Grad-CAM regions show areas that influenced the model's decision.
                    """)
                    st.markdown("### 🌿 Suggested Actions")
                    st.write("""
**Lifestyle & Exercise:**
- 30–45 minutes of walking or light jogging daily  
- Yoga and stretching (Surya Namaskar, Butterfly pose, Cobbler pose)  
- Avoid long continuous sitting; add small movement breaks  

**Diet Suggestions:**
- Prefer: vegetables, salads, oats, lentils, eggs, paneer, lean chicken  
- Include: nuts (almonds, walnuts), seeds (chia, flax), olive oil  
- Avoid: sugary drinks, white bread, pastries, deep-fried and junk food  

**Medical Advice:**
- Track menstrual cycles regularly  
- Consult a gynecologist/endocrinologist for proper diagnosis  
- Follow hormone and metabolic tests as suggested by doctor  
                    """)
                else:
                    st.markdown("### 😊 PCOS Unlikely (Based on This Image)")
                    st.write("""
This ultrasound does not show typical PCOS patterns based on the model's learned features.
However, overall health still depends on hormones, lifestyle, and regular checkups.
                    """)
                    st.markdown("### 🌱 General Health & Prevention Tips")
                    st.write("""
**Healthy Diet:**
- Eat fresh fruits, vegetables, whole grains  
- Moderate sugar; avoid highly processed foods  
- Sufficient protein (lentils, paneer, eggs, tofu, lean meats)  

**Activity & Lifestyle:**
- At least 30 minutes exercise daily  
- Maintain healthy body weight  
- Reduce stress (yoga, meditation, breathing exercises)  
- Sleep 7–8 hours daily  

**Preventive Checks:**
- Regular health checkups (blood sugar, hormones)  
- Track menstrual patterns  
- Avoid smoking and excessive alcohol  
                    """)

                # ----- PDF REPORT DOWNLOAD -----
                st.markdown("---")
                st.subheader("📄 Download Report")

                pdf_bytes = create_pdf_report(
                    patient_name=patient_name,
                    label=label,
                    prob=prob,
                    orig_image_np=img_display,
                    heatmap_image_np=final_rgb
                )

                st.download_button(
                    label="📥 Download PCOS Report (PDF)",
                    data=pdf_bytes,
                    file_name="pcos_detection_report.pdf",
                    mime="application/pdf"
                )

# ---------------------------------------------
# PAGE 3 – RESULTS & EVALUATION (GRAPHS + CONFUSION MATRIX)
# ---------------------------------------------
elif page == "Results & Evaluation":
    st.title("📈 Model Results & Evaluation")

    st.markdown("""
Here you can visualize **training performance** and **confusion matrix**  
by uploading CSV files exported from your Colab training notebook.

This is very useful for your **Final Year Project report and viva**.
    """)

    st.markdown("### 📊 Training & Validation Accuracy / Loss")

    hist_file = st.file_uploader(
        "Upload training history CSV (columns: epoch, accuracy, val_accuracy, loss, val_loss)",
        type=["csv"],
        key="hist_csv"
    )

    if hist_file is not None:
        df = pd.read_csv(hist_file)

        # Accuracy graph
        if {"accuracy", "val_accuracy"}.issubset(df.columns):
            fig1, ax1 = plt.subplots()
            ax1.plot(df["epoch"], df["accuracy"], label="Train Accuracy")
            ax1.plot(df["epoch"], df["val_accuracy"], label="Val Accuracy")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Accuracy")
            ax1.set_title("Training vs Validation Accuracy")
            ax1.legend()
            st.pyplot(fig1)
        else:
            st.warning("Columns 'accuracy' and 'val_accuracy' not found in CSV.")

        # Loss graph
        if {"loss", "val_loss"}.issubset(df.columns):
            fig2, ax2 = plt.subplots()
            ax2.plot(df["epoch"], df["loss"], label="Train Loss")
            ax2.plot(df["epoch"], df["val_loss"], label="Val Loss")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.set_title("Training vs Validation Loss")
            ax2.legend()
            st.pyplot(fig2)
        else:
            st.warning("Columns 'loss' and 'val_loss' not found in CSV.")
    else:
        st.info("Upload a training history CSV to see accuracy and loss graphs.")

    st.markdown("---")
    st.markdown("### 🧪 Confusion Matrix")

    cm_file = st.file_uploader(
        "Upload confusion matrix CSV (2x2: rows=Actual [PCOS, Normal], cols=Predicted [PCOS, Normal])",
        type=["csv"],
        key="cm_csv"
    )

    if cm_file is not None:
        cm_df = pd.read_csv(cm_file, header=None)
        st.write("Confusion Matrix (as table):")
        st.dataframe(cm_df)

        # Heatmap
        fig_cm, ax_cm = plt.subplots()
        im = ax_cm.imshow(cm_df.values, cmap="Blues")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Pred PCOS", "Pred Normal"])
        ax_cm.set_yticklabels(["Actual PCOS", "Actual Normal"])
        for i in range(2):
            for j in range(2):
                ax_cm.text(j, i, int(cm_df.values[i, j]), ha="center", va="center", color="black")
        ax_cm.set_title("Confusion Matrix")
        fig_cm.colorbar(im)
        st.pyplot(fig_cm)
    else:
        st.info("Upload a 2x2 confusion matrix CSV to see the plot.")

# ---------------------------------------------
# PAGE 4 – ABOUT
# ---------------------------------------------
elif page == "About":
    st.title("ℹ️ About This Project")

    st.markdown("""
**Title:** Machine Learning Model for Accurate PCOS Detection Using Pelvic Ultrasound Imaging  

**Tech Stack Used:**
- Python  
- TensorFlow / Keras (CNN)  
- OpenCV  
- NumPy, Pandas, Matplotlib  
- Streamlit  

**Key Features:**
- PCOS vs Normal ultrasound classification  
- Grad-CAM based explainability  
- Health and lifestyle recommendations  
- PDF report generation  
- Support for training result visualization (accuracy, loss, confusion matrix)

This project is developed as a **Final Year Project** to demonstrate how  
AI can assist in medical imaging analysis and decision support.
    """)

st.markdown("---")
st.caption("Built with ❤️ using TensorFlow, Keras, OpenCV & Streamlit.")
