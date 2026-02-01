# 🌿 Leaf Disease Detection

AI-powered plant disease classifier (**93% accuracy**) using ResNet50 + Random Forest.

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen?style=for-the-badge&logo=streamlit)](https://leaf-disease-detection-harshannaju.streamlit.app/)
[![GitHub Stars](https://img.shields.io/github/stars/Harshan-Maju/leaf-disease-detection?style=social)](https://github.com/Harshan-Maju/leaf-disease-detection)
[![License](https://img.shields.io/github/license/Harshan-Maju/leaf-disease-detection)](LICENSE)

## 🚀 Features
- Single Image & Batch Processing (50+ images)
- Top-5 Predictions with confidence scores
- Uncertainty Analysis (Low/Medium/High via entropy)
- Image Enhancement (contrast + sharpness)
- Confidence Threshold (10-80% adjustable)
- CSV Export & Prediction History

## 📊 Performance Metrics
| Metric | Score |
|--------|-------|
| Multiclass Accuracy | 93.2% |
| Binary Accuracy | 97.1% |
| Classes | 38 |
| Test Images | 15,480 |

## 🖥 Quick Start
```bash
git clone https://github.com/Harshan-Maju/leaf-disease-detection.git
cd leaf-disease-detection
pip install -r requirements.txt
streamlit run app.py
