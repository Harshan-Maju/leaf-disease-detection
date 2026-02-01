import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet50_Weights
from joblib import load
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
import time
from datetime import datetime
import json
import os

st.set_page_config(
    page_title="Leaf Disease Detection",
    page_icon="🌿",
    layout="wide"
)

@st.cache_resource
def load_disease_model():
    model_path = "models/final_leaf_disease_model.joblib"
    metadata_options = ["data/metadata.json", "results/metadata.json", "models/metadata.json"]
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at: `{model_path}`")
        st.stop()

    try:
        rf_model = load(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()

        class_names = None
        for metadata_path in metadata_options:
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    class_names = {v: k for k, v in metadata["class_to_idx"].items()}
                    break
                except Exception:
                    continue

        if class_names is None:
            class_names = {i: f"Disease_Class_{i}" for i in range(38)}

        return rf_model, feature_extractor, class_names, device
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model, extractor, classes, device = load_disease_model()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_disease(image: Image.Image, threshold: float = 0.25):
    try:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feats = extractor(img_tensor)
            feats = feats.view(feats.size(0), -1)
            feats_np = feats.cpu().numpy()

        proba = model.predict_proba(feats_np)[0]
        pred_class = int(proba.argmax())
        confidence = float(proba[pred_class])

        top5_indices = np.argsort(proba)[-5:][::-1]
        entropy = -np.sum(proba * np.log(proba + 1e-10))
        uncertainty = "Low" if entropy < 1.5 else "Medium" if entropy < 2.5 else "High"

        disease_name = classes[pred_class].replace("_", " ").title()
        
        return {
            "disease": disease_name,
            "confidence": confidence,
            "passes_threshold": confidence >= threshold,
            "top5_classes": [classes[i].replace("_", " ").title() for i in top5_indices],
            "top5_scores": proba[top5_indices].tolist(),
            "entropy": float(entropy),
            "uncertainty": uncertainty,
            "all_proba": proba.tolist()
        }
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None

with st.sidebar:
    st.title("⚙️ Settings")
    mode = st.radio("Select Mode", ["Single Image", "Batch Processing"])
    
    st.divider()
    st.subheader("🎯 Confidence Threshold")
    threshold_pct = st.slider("Minimum Confidence (%)", 10, 80, 25, 5)
    threshold = threshold_pct / 100.0
    st.info(f"**Current threshold:** {threshold_pct}%")
    
    st.divider()
    st.subheader("📊 System Info")
    st.info(f"""
    **Model:** Random Forest  
    **Feature Extractor:** ResNet50  
    **Device:** {device.upper()}  
    **Classes:** {len(classes)}  
    **Status:** ✅ Ready
    """)
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Multiclass", "93%")
    with col2:
        st.metric("Binary", "97%")
    st.metric("Test Images", "15,480")

st.title("🌿 Leaf Disease Detection")
c1, c2, c3 = st.columns(3)
with c1:
    st.success("✅ Models loaded")
with c2:
    st.info("🔍 Single" if mode == "Single Image" else "📦 Batch")
with c3:
    st.success(f"🎯 {threshold_pct}%")
st.divider()

if mode == "Single Image":
    st.header("Single Image Analysis")
    
    uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file:
        st.success(f"✅ {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("📷 Original Image")
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)
            st.info(f"**{image.size[0]}×{image.size[1]}** px | {image.format or 'Unknown'}")
        
        with col_right:
            st.subheader("🔬 Results")
            if st.button("🔍 Analyze", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    start_time = time.time()
                    result = predict_disease(image, threshold)
                    elapsed = time.time() - start_time

                if result and result["disease"]:
                    confidence_pct = result["confidence"] * 100
                    
                    if result["passes_threshold"]:
                        st.success("✅ Prediction")
                        st.markdown(f"# **{result['disease']}**")
                        
                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("Confidence", f"{confidence_pct:.1f}%")
                        with m2:
                            st.metric("Time", f"{elapsed:.2f}s")
                        
                        st.progress(result["confidence"])
                        
                        if confidence_pct >= 50:
                            st.success(f"🎯 **HIGH** ({confidence_pct:.1f}%)")
                        elif confidence_pct >= 30:
                            st.info(f"✓ **MEDIUM** ({confidence_pct:.1f}%)")
                        elif confidence_pct >= 25:
                            st.warning(f"ℹ️ **LOW** ({confidence_pct:.1f}%) - {result['uncertainty']}")
                        
                        st.subheader("📊 Top 5")
                        st.caption(f"Uncertainty: {result['uncertainty']} (Entropy: {result['entropy']:.2f})")
                        
                        top5_cols = st.columns(5)
                        colors = ["🟢", "🟡", "🟠", "🔵", "🟣"]
                        for i, (name, score) in enumerate(zip(result["top5_classes"], result["top5_scores"])):
                            with top5_cols[i]:
                                st.metric(f"{colors[i]} #{i+1}", f"{score*100:.1f}%", name[:20])
                    else:
                        st.warning("⚠️ Below Threshold")
                        st.info(f"{result['disease']} ({confidence_pct:.1f}%)")
                        
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append({
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "File": uploaded_file.name,
                    "Disease": result["disease"] if result else "Error",
                    "Confidence": f"{confidence_pct:.1f}%" if result else "N/A",
                    "Uncertainty": result["uncertainty"] if result else "N/A",
                    "Threshold": f"{threshold_pct}%",
                    "Passed": result["passes_threshold"] if result else False
                })

elif mode == "Batch Processing":
    st.header("Batch Processing")
    uploaded_files = st.file_uploader("Select images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            process_btn = st.button("🚀 Process All", type="primary", use_container_width=True)
        with c2:
            st.metric("Files", len(uploaded_files))
        with c3:
            st.metric("Threshold", f"{threshold_pct}%")
        
        if process_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []

            for idx, file in enumerate(uploaded_files):
                status_text.info(f"Processing {idx+1}/{len(uploaded_files)}: {file.name}")
                
                try:
                    img = Image.open(file).convert("RGB")
                    result = predict_disease(img, threshold)
                    
                    status = "✅ Pass" if result and result["passes_threshold"] else "⚠️ Low"
                    results.append({
                        "Filename": file.name,
                        "Disease": result["disease"] if result else "Error",
                        "Confidence": f"{result['confidence']*100:.1f}%" if result else "N/A",
                        "Uncertainty": result["uncertainty"] if result else "N/A",
                        "Threshold": f"{threshold_pct}%",
                        "Status": status
                    })
                except:
                    results.append({
                        "Filename": file.name, "Disease": "Error", "Confidence": "N/A",
                        "Uncertainty": "N/A", "Threshold": f"{threshold_pct}%", "Status": "❌ Failed"
                    })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))

            st.success("✅ Complete")
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, height=400)
            
            total, passed = len(results), sum(1 for r in results if "Pass" in r["Status"])
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total", total)
            with col2: st.metric("Passed", passed)
            with col3: st.metric("Rate", f"{passed/total*100:.0f}%" if total else "0%")
            
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV", csv, f"results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv", use_container_width=True, type="primary"
            )

if "history" in st.session_state and st.session_state.history:
    with st.expander("📜 History"):
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total", len(st.session_state.history))
        with col2: st.metric("Passed", sum(1 for h in st.session_state.history if h["Passed"]))
        with col3: st.metric("Avg Conf", f"{df_hist['Confidence'].str.rstrip('%').astype(float).mean():.0f}%")
        
        if st.button("🗑️ Clear"):
            st.session_state.history = []
            st.rerun()

st.divider()
col1, col2, col3 = st.columns(3)
with col1: st.info("ResNet50 + Random Forest")
with col2: st.success("93% Multiclass | 97% Binary")
with col3: st.info(f"Threshold: {threshold_pct}%")
