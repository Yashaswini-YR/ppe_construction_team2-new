import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# ------------------ CONFIG ------------------
st.set_page_config(page_title="PPE Detection", layout="wide")

# ✅ Only valid classes
VALID_CLASSES = ["Person", "helmet", "vest", "gloves", "goggles", "boots"]

# ------------------ TITLE ------------------
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>🦺 PPE Detection System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Upload an image to detect safety equipment</p>",
    unsafe_allow_html=True
)

# ------------------ MODEL ------------------
@st.cache_resource
def load_model():
    return YOLO("runs/detect/ppe_model4/weights/best.pt")

model = load_model()

# ------------------ UPLOAD ------------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    # -------- ORIGINAL --------
    with col1:
        st.subheader("📷 Original Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image)

    # -------- TEMP SAVE --------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # -------- 🔥 BALANCED DETECTION --------
    results = model.predict(
        source=temp_path,
        conf=0.20,   # ✅ balanced value
        iou=0.40,
        save=False
    )

    # -------- OUTPUT IMAGE --------
    with col2:
        st.subheader("🧠 Detection Result")
        result_img = results[0].plot()
        st.image(result_img)

    # -------- EXTRACT FILTERED LABELS --------
    st.subheader("🔍 Detected Objects")

    labels = []

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])

            # ✅ FILTER unwanted detections
            if label in VALID_CLASSES:
                labels.append(label)
                st.write(f"✅ {label} ({conf:.2f})")

    if not labels:
        st.write("❌ No valid objects detected")

    # -------- SAFETY LOGIC --------
    st.subheader("🚨 Safety Alerts")

    persons = labels.count("Person")
    helmets = labels.count("helmet")
    vests = labels.count("vest")
    gloves = labels.count("gloves")
    goggles = labels.count("goggles")
    boots = labels.count("boots")

    violations = []

    if persons > helmets:
        violations.append("Missing Helmet")

    if persons > vests:
        violations.append("Missing Vest")

    if persons > gloves:
        violations.append("Missing Gloves")

    if persons > goggles:
        violations.append("Missing Goggles")

    if persons > boots:
        violations.append("Missing Boots")

    # -------- ALERT DISPLAY --------
    if violations:
        for v in violations:
            st.warning(f"⚠️ {v}")
    else:
        st.success("✅ All safety equipment detected")

    # -------- CLEANUP --------
    os.remove(temp_path)