import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# =========================
# Load Trained Model
# =========================
MODEL_PATH = "best_model.h5"
if not os.path.exists(MODEL_PATH):
    st.error("❌ Trained model not found! Please run app.py first to generate best_model.h5")
    st.stop()

model = load_model(MODEL_PATH)

# =========================
# Class Labels (must match training)
# =========================
# Replace with your actual class names from train/
class_labels = sorted(os.listdir("train"))  

# =========================
# Streamlit UI
# =========================
st.title("🍄 Mushroom Classifier")
st.write("Upload an image of a mushroom to predict its class.")

# Upload image
uploaded_file = st.file_uploader("Choose a mushroom image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    st.success(f"✅ Predicted Class: **{class_labels[predicted_class]}**")
    st.info(f"🔹 Confidence: {confidence*100:.2f}%")
