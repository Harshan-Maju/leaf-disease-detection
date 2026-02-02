import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import joblib
import numpy as np
from pathlib import Path
import pandas as pd

# ----------------------------
# Paths and Config
# ----------------------------
FEATURE_DIR = Path(r"C:\Users\harsh\Documents\LeafDiseaseProject\feature_dataset_merged")
BINARY_MODEL = FEATURE_DIR / "binary_model_rf.pkl"   # Random Forest (binary)
MULTI_MODEL = FEATURE_DIR / "multi_model_rf.pkl"     # Random Forest (multi-class)
MULTI_MAPPING = FEATURE_DIR / "multi_mapping.pkl"

DEFAULT_UNCERTAINTY = 0.55
SHOW_UNCERTAIN = True

# ----------------------------
# Helpers
# ----------------------------
def clamp_0_1(x):
    x = float(x)
    return max(0.0, min(1.0, x))

# ----------------------------
# Cached loaders
# ----------------------------
@st.cache_resource
def load_models():
    bin_clf = joblib.load(BINARY_MODEL)
    multi_clf = joblib.load(MULTI_MODEL)
    label_mapping = joblib.load(MULTI_MAPPING)
    idx_to_label = {v: k for k, v in label_mapping.items()}
    return bin_clf, multi_clf, idx_to_label

@st.cache_resource
def load_feature_extractor():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    model.to(device)
    return model, device

# ----------------------------
# Transforms (must match training)
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----------------------------
# Utility functions
# ----------------------------
def extract_plant_name(class_name: str) -> str:
    if '_' in class_name:
        return class_name.split('_')[0].capitalize()
    return class_name.capitalize()

def extract_disease_name(class_name: str) -> str:
    if '_healthy' in class_name.lower():
        return "Healthy"
    if '_' in class_name:
        parts = class_name.split('_')[1:]
        return ' '.join(parts).replace('_', ' ').title()
    return class_name.title()

def extract_features_single(image: Image.Image, model, device) -> np.ndarray:
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            feats = model(x)
    return feats.squeeze(-1).squeeze(-1).cpu().numpy()

def predict_image(image, bin_clf, multi_clf, idx_to_label, feature_model, device, threshold, show_uncertain=True):
    feats = extract_features_single(image, feature_model, device)

    # Binary prediction
    bin_proba = bin_clf.predict_proba(feats)[0]  # e.g. [prob_healthy, prob_diseased] depending on how trained
    bin_pred = bin_clf.predict(feats)[0]

    # Multi-class prediction
    multi_proba = multi_clf.predict_proba(feats)[0]
    multi_pred = multi_clf.predict(feats)[0]
    multi_conf = float(np.max(multi_proba))

    # Class names
    full_class_name = idx_to_label[multi_pred]
    plant_name = extract_plant_name(full_class_name)

    # Decide output
    if show_uncertain and multi_conf < threshold:
        status = "Uncertain"
        disease_name = "N/A"
        confidence = multi_conf
        plant_out = "N/A"
    else:
        if bin_pred == 0:
            status = "Healthy"
            disease_name = "No disease detected"
            # For Healthy, use binary-probability-of-healthy as main confidence
            confidence = float(bin_proba[0])
            plant_out = plant_name
        else:
            status = "Diseased"
            disease_name = extract_disease_name(full_class_name)
            # For Diseased, use the multi-class confidence (max of multi_proba)
            confidence = multi_conf
            plant_out = plant_name

    return status, plant_out, disease_name, confidence

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Leaf Disease Detection", page_icon="🍃", layout="wide")
st.title("🍃 Leaf Disease Detection (Random Forest)")

with st.spinner("Loading models and feature extractor..."):
    bin_clf, multi_clf, idx_to_label = load_models()
    feature_model, device = load_feature_extractor()

if torch.cuda.is_available():
    st.success(f"✅ GPU enabled: {torch.cuda.get_device_name(0)}")
else:
    st.info("ℹ️ Running on CPU")

# Sidebar
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Uncertainty Threshold", 0.0, 1.0, DEFAULT_UNCERTAINTY, 0.05)
show_uncertain = st.sidebar.checkbox("Enable Uncertain status", value=SHOW_UNCERTAIN)
st.sidebar.caption(f"Current threshold: {threshold:.2f}")
st.sidebar.markdown("---")
st.sidebar.info("This app uses ResNet50 feature extraction and calibrated Random Forest classifiers.")

# Tabs
tab1, tab2 = st.tabs(["📷 Single Image Upload", "📁 Batch Upload"])

# ----------------------------
# Single Image Upload
# ----------------------------
with tab1:
    st.subheader("Single Image")
    single_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="single")
    if single_file:
        image = Image.open(single_file).convert('RGB')
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption=single_file.name, use_container_width=True)

        with col2:
            with st.spinner("Analyzing image..."):
                status, plant, disease, conf = predict_image(
                    image, bin_clf, multi_clf, idx_to_label, feature_model, device,
                    threshold=threshold, show_uncertain=show_uncertain
                )

            st.subheader("Analysis Results")
            if status == "Uncertain":
                st.warning(f"⚠️ Status: {status}")
            elif status == "Healthy":
                st.success(f"✅ Status: {status}")
            else:
                st.error(f"🦠 Status: {status}")

            st.markdown(f"• Plant: {plant}")
            st.markdown(f"• Disease: {disease}")
            st.markdown(f"• Confidence: {conf:.2%}")

            # progress expects 0..1 so clamp just in case
            st.progress(clamp_0_1(conf))

# ----------------------------
# Batch Upload
# ----------------------------
with tab2:
    st.subheader("Batch Images")
    batch_files = st.file_uploader("Choose multiple images", type=["jpg", "jpeg", "png"],
                                   accept_multiple_files=True, key="batch")
    if batch_files:
        results = []
        cols = st.columns(3)

        for i, f in enumerate(batch_files):
            img = Image.open(f).convert('RGB')
            with st.spinner(f"Analyzing {f.name}..."):
                status, plant, disease, conf = predict_image(
                    img, bin_clf, multi_clf, idx_to_label, feature_model, device,
                    threshold=threshold, show_uncertain=show_uncertain
                )

            with cols[i % 3]:
                st.image(img, caption=f.name, use_container_width=True)
                if status == "Uncertain":
                    st.warning(f"⚠️ {status}")
                elif status == "Healthy":
                    st.success(f"✅ {status}")
                else:
                    st.error(f"🦠 {status}")

                st.caption(f"Plant: {plant}")
                st.caption(f"Disease: {disease}")
                st.caption(f"Confidence: {conf:.2%}")

            results.append({
                "Filename": f.name,
                "Status": status,
                "Plant": plant,
                "Disease": disease,
                "Confidence": f"{conf:.2%}"
            })

        st.markdown("---")
        st.subheader("Results Summary")
        df_out = pd.DataFrame(results)
        st.dataframe(df_out, use_container_width=True)
        st.download_button(
            label="📥 Download Results CSV",
            data=df_out.to_csv(index=False).encode('utf-8'),
            file_name="leaf_disease_results_rf.csv",
            mime="text/csv"
        )
    else:
        st.info("Upload multiple images to start batch analysis")
