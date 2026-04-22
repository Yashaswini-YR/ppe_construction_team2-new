import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="PPE Detection", layout="wide")

# ------------------ TITLE ------------------
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>🦺 PPE Detection System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Upload an image to detect safety equipment</p>",
    unsafe_allow_html=True
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return YOLO("runs/detect/ppe_model4/weights/best.pt")

model = load_model()

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    # -------- LEFT: ORIGINAL IMAGE --------
    with col1:
        st.subheader("📷 Original Image")
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image")
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.stop()

    # -------- SAVE TEMP IMAGE --------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # -------- RUN DETECTION --------
    results = model.predict(source=temp_path, save=False)

    # -------- RIGHT: OUTPUT IMAGE --------
    with col2:
        st.subheader("🧠 Detection Result")
        result_img = results[0].plot()
        st.image(result_img, caption="Detected Output")

    # -------- SHOW LABELS --------
    st.subheader("🔍 Detected Objects")

    labels = []
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            labels.append((label, confidence))

    if labels:
        for label, conf in labels:
            st.write(f"✅ {label} ({conf:.2f})")

            # Alert for missing PPE
            if "no_" in label.lower():
                st.warning(f"⚠️ ALERT: {label} detected!")
    else:
        st.write("❌ No objects detected")

    # -------- CLEANUP --------
    os.remove(temp_path)