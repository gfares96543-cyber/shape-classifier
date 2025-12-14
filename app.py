import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

st.set_page_config(page_title="Shape Classifier", layout="centered")

st.title("🔍 Shape Classification")
st.write("Upload an image to predict its class")

# ---- Load model ----
MODEL_PATH = "final_model.keras"

@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found: final_model.keras")
    st.stop()

model = load_my_model()

# ---- Class names ----
class_names = [
    'circle_irregular', 'circle_medium', 'circle_perfect',
    'square_irregular', 'square_medium', 'square_perfect',
    'triangle_irregular', 'triangle_medium', 'triangle_perfect'
]

# ---- Upload image ----
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # ---- Preprocess image ----
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # ---- Predict ----
    pred = model.predict(img_array)[0]

    # Apply softmax if needed
    if not np.isclose(pred.sum(), 1.0):
        exp = np.exp(pred - np.max(pred))
        probs = exp / exp.sum()
    else:
        probs = pred

    class_idx = int(np.argmax(probs))
    class_name = class_names[class_idx]
    confidence = probs[class_idx] * 100

    st.success(f"✅ **Prediction:** {class_name}")
    st.info(f"📊 **Confidence:** {confidence:.2f}%")

    st.subheader("All class probabilities")
    for i, p in enumerate(probs):
        st.write(f"{class_names[i]} → {p*100:.2f}%")
