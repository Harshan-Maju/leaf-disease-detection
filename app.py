import streamlit as st
import gdown
import os
import json
import torch
from torchvision import transforms
from PIL import Image
import zipfile

# ------------------------------
# --- CONFIGURATION / PATHS ---
# ------------------------------
os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/leaf_model.pth"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1vdGvyehvrAWtvVTWM9G0Dq-sQh13Sp2k"

METADATA_PATH = "models/metadata.json"
METADATA_URL = "https://drive.google.com/uc?export=download&id=17QTd2spFzpcwX6o256Xegqjksvl5zI6t"

# ------------------------------
# --- DOWNLOAD FUNCTION ---
# ------------------------------
def download_file(url, path, description):
    if not os.path.exists(path):
        try:
            with st.spinner(f"📥 Downloading {description}..."):
                gdown.download(url, path, quiet=False)
        except Exception as e:
            st.error(f"Failed to download {description}. Please download manually.\n{e}")
            st.stop()

download_file(MODEL_URL, MODEL_PATH, "model")
download_file(METADATA_URL, METADATA_PATH, "metadata")

# ------------------------------
# --- LOAD METADATA & MODEL ---
# ------------------------------
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

if "class_to_idx" not in metadata:
    st.error("❌ metadata.json missing 'class_to_idx'")
    st.stop()

class_names = {v: k for k, v in metadata["class_to_idx"].items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

# ------------------------------
# --- IMAGE TRANSFORMS ---
# ------------------------------
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------------------
# --- PREDICTION FUNCTION ---
# ------------------------------
def predict_image(image):
    img = image_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, pred_idx = torch.max(outputs, 1)
    return class_names[int(pred_idx)]

# ------------------------------
# --- STREAMLIT UI ---
# ------------------------------
st.set_page_config(page_title="Leaf Disease Detection", layout="wide")
st.title("🌿 Leaf Disease Detection")

st.markdown("""
This app allows you to detect leaf diseases using a pre-trained model.
- Upload single leaf images for prediction
- Upload ZIP files for batch predictions
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

                # Display results
                st.subheader("Batch Predictions")
                for fname, pred in results:
                    st.write(f"**{fname}** → {pred}")

    except Exception as e:
        st.error(f"Error processing ZIP: {e}")

# ------------------------------
# --- FOOTER ---
# ------------------------------
st.markdown("---")
st.markdown("Leaf Disease Detector V1.1")
