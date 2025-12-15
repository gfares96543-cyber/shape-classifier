import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os

st.set_page_config(page_title="Shape Classifier", layout="centered")
st.title("✏️ Draw & Classify Shape")

# ---------------- MODEL ----------------
MODEL_PATH = "final_model.keras"

@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model not found!")
    st.stop()

model = load_my_model()

class_names = [
    'circle_irregular', 'circle_medium', 'circle_perfect',
    'square_irregular', 'square_medium', 'square_perfect',
    'triangle_irregular', 'triangle_medium', 'triangle_perfect'
]

# ---------------- DRAW CANVAS ----------------
st.subheader("Draw a shape")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=300,
    height=300,
    drawing_mode="freedraw",
    key="canvas",
)

# ---------------- PREDICT ----------------
if st.button("🔍 Classify Drawing"):
    if canvas_result.image_data is None:
        st.warning("Please draw something first")
    else:
        # Convert canvas to image
        img = Image.fromarray(canvas_result.image_data.astype("uint8"))
        img = img.convert("RGB")
        img = img.resize((224, 224))

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        pred = model.predict(img_array)[0]

        # Softmax if needed
        if not np.isclose(pred.sum(), 1.0):
            exp = np.exp(pred - np.max(pred))
            probs = exp / exp.sum()
        else:
            probs = pred

        class_idx = int(np.argmax(probs))
        class_name = class_names[class_idx]
        confidence = probs[class_idx] * 100

        st.success(f"✅ Prediction: **{class_name}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        st.subheader("All probabilities")
        for i, p in enumerate(probs):
            st.write(f"{class_names[i]} → {p*100:.2f}%")
