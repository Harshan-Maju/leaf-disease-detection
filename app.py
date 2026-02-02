import streamlit as st
import os
import json
import joblib
from PIL import Image
import numpy as np
import zipfile
import gdown

# ------------------------------
# --- CONFIGURATION / PATHS ---
# ------------------------------
os.makedirs("models", exist_ok=True)

# Models (joblib)
MODEL_PATH = "models/final_leaf_disease_model.joblib"
PCA_PATH = "models/pca.joblib"  # optional if your pipeline needs PCA
PIPELINE_PATH = "models/production_pipeline.joblib"  # optional

# Metadata
METADATA_PATH = "models/metadata.json"
METADATA_URL = "https://drive.google.com/uc?export=download&id=17QTd2spFzpcwX6o256Xegqjksvl5zI6t"

# ------------------------------
# --- DOWNLOAD METADATA IF MISSING ---
# ------------------------------
def download_file(url, path, description):
    if not os.path.exists(path):
        with st.spinner(f"📥 Downloading {description}..."):
            gdown.download(url, path, quiet=False)

download_file(METADATA_URL, METADATA_PATH, "metadata")

# ------------------------------
# --- LOAD METADATA ---
# ------------------------------
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

if "class_to_idx" not in metadata:
    st.error("❌ metadata.json missing 'class_to_idx'")
    st.stop()

class_names = {v: k for k, v in metadata["class_to_idx"].items()}

# ------------------------------
# --- LOAD MODEL ---
# ------------------------------
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file {MODEL_PATH} not found. Please upload it.")
    st.stop()

model = joblib.load(MODEL_PATH)
st.success("✅ Model loaded successfully!")

# Optional: load PCA or pipeline if used
# pca = joblib.load(PCA_PATH) if os.path.exists(PCA_PATH) else None
# pipeline = joblib.load(PIPELINE_PATH) if os.path.exists(PIPELINE_PATH) else None

# ------------------------------
# --- IMAGE PREPROCESSING ---
# ------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Converts a PIL image into the format your joblib model expects.
    For example, flatten and normalize.
    Adjust this function based on your model's training pipeline.
    """
    image = image.resize((224, 224))  # or your trained size
    arr = np.array(image).astype(np.float32)
    arr = arr.flatten()  # flatten image
    arr = arr.reshape(1, -1)
    return arr

# ------------------------------
# --- PREDICTION FUNCTION ---
# ------------------------------
def predict_image(image: Image.Image):
    X = preprocess_image(image)
    pred_idx = model.predict(X)[0]
    # Map prediction index to class name
    return class_names.get(pred_idx, f"Class {pred_idx}")

# ------------------------------
# --- STREAMLIT UI ---
# ------------------------------
st.set_page_config(page_title="Leaf Disease Detection", layout="wide")
st.title("🌿 Leaf Disease Detection (joblib version)")

st.markdown("""
Upload single leaf images or a ZIP of multiple images to predict leaf diseases.
""")

# --- SINGLE IMAGE UPLOADER ---
st.header("Single Image Prediction")
single_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"], key="single")

if single_file is not None:
    try:
        image = Image.open(single_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        prediction = predict_image(image)
        st.success(f"Prediction: **{prediction}**")
    except Exception as e:
        st.error(f"Error processing image: {e}")

# --- BATCH IMAGE UPLOADER ---
st.header("Batch Prediction (ZIP)")
batch_file = st.file_uploader("Upload a ZIP of leaf images", type=["zip"], key="batch")

if batch_file is not None:
    try:
        with zipfile.ZipFile(batch_file) as z:
            image_files = [f for f in z.namelist() if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if not image_files:
                st.warning("No valid image files found in ZIP.")
            else:
                st.info(f"Found {len(image_files)} images. Processing...")
                results = []
                for file_name in image_files:
                    with z.open(file_name) as f:
                        img = Image.open(f).convert("RGB")
                        pred = predict_image(img)
                        results.append((file_name, pred))

                st.subheader("Batch Predictions")
                for fname, pred in results:
                    st.write(f"**{fname}** → {pred}")

    except Exception as e:
        st.error(f"Error processing ZIP: {e}")

# ------------------------------
# --- FOOTER ---
# ------------------------------
st.markdown("---")
st.markdown("|Leaf Disease Detector|")
