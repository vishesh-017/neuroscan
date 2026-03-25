# ============================================================================
# BRAIN TUMOR DETECTION SYSTEM - STREAMLIT UI
# ============================================================================
# Simple web interface for MRI analysis using trained models
# Run with: streamlit run app.py
# ============================================================================

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import json
from pathlib import Path

from torchvision import models, transforms

import sys
sys.path.append('..')
sys.path.append('../utils')

try:
    from utils.gradcam import generate_gradcam, overlay_heatmap
except:
    from gradcam import generate_gradcam, overlay_heatmap

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set page config
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# LOAD MODELS (Cached for performance)
# ============================================================================

@st.cache_resource
def load_multiclass_model():
    """Load the multi-class MobileNetV2 model"""
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 4)
    model.load_state_dict(torch.load('../models/multiclass_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_binary_model():
    """Load the binary classification model"""
    
    class ImprovedCNN(nn.Module):
        def __init__(self):
            super(ImprovedCNN, self).__init__()
            
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 8 * 8, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    model = ImprovedCNN()
    model.load_state_dict(torch.load('../models/binary_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

@st.cache_data
def load_results():
    """Load saved model results"""
    with open('../models/multiclass_results.json', 'r') as f:
        mc_results = json.load(f)
    with open('../models/binary_results.json', 'r') as f:
        bin_results = json.load(f)
    return mc_results, bin_results

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_image_quality(image):
    """
    Check if image quality is sufficient
    Returns: (is_good, brightness, message)
    """
    # Convert PIL to numpy
    img_np = np.array(image)
    
    # Convert to grayscale
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    brightness = np.mean(gray)
    
    if brightness < 30:
        return False, brightness, "⚠️ Image too dark"
    elif brightness > 225:
        return False, brightness, "⚠️ Image overexposed"
    else:
        return True, brightness, "✅ Good quality"

def predict_multiclass(image, model):
    """
    Predict using multi-class model
    Returns: (class_name, confidence, all_probs)
    """
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
    
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    return class_names[pred.item()], confidence.item(), probs.cpu().numpy()[0], img_tensor

def predict_binary(image, model):
    """
    Predict using binary model
    Returns: (class_name, confidence)
    """
    # Transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
    
    class_names = ['no', 'yes']
    
    return class_names[pred.item()], confidence.item()

def get_confidence_color(confidence):
    """Return color based on confidence level"""
    if confidence >= 0.8:
        return "green"
    elif confidence >= 0.6:
        return "orange"
    else:
        return "red"

def get_status_message(confidence):
    """Return status message based on confidence"""
    if confidence >= 0.8:
        return "✅ Confident Prediction"
    elif confidence >= 0.6:
        return "⚠️ Moderate Confidence - Review Recommended"
    else:
        return "🚨 Low Confidence - Manual Review Required"

# ============================================================================
# MAIN UI
# ============================================================================

def main():
    # Load models
    model_mc = load_multiclass_model()
    model_bin = load_binary_model()
    mc_results, bin_results = load_results()
    
    # ========== HEADER ==========
    st.title("🧠 Brain Tumor Detection System")
    st.markdown("---")
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "Select Model",
            ["Multi-Class Classification", "Binary Detection"],
            help="Multi-Class: Identifies tumor type (Glioma, Meningioma, Pituitary, No Tumor)\nBinary: Detects presence of tumor (Yes/No)"
        )
        
        st.markdown("---")
        
        # Model info
        st.subheader("📊 Model Performance")
        if model_choice == "Multi-Class Classification":
            st.metric("Test Accuracy", f"{mc_results['test_accuracy']:.2%}")
            st.caption("Classes: Glioma, Meningioma, Pituitary, No Tumor")
        else:
            st.metric("Test Accuracy", f"{bin_results['test_accuracy']:.2%}")
            st.caption("Classes: Tumor / No Tumor")
        
        st.markdown("---")
        
        # About
        st.subheader("ℹ️ About")
        st.caption("""
        This system uses deep learning to analyze MRI scans for brain tumors.
        
        **Features:**
        - Image quality check
        - Confidence scoring
        - Explainable AI (Grad-CAM)
        - Uncertainty detection
        """)
    
    # ========== MAIN CONTENT ==========
    
    # File uploader
    st.subheader("📤 Upload MRI Image")
    uploaded_file = st.file_uploader(
        "Choose an MRI scan image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a brain MRI scan in JPG or PNG format"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖼️ Original Image")
            st.image(image, use_container_width=True)
        
        # Quality check
        st.markdown("---")
        st.subheader("🔍 Image Quality Check")
        
        is_good_quality, brightness, quality_msg = check_image_quality(image)
        
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.metric("Brightness", f"{brightness:.1f}")
        with col_q2:
            if is_good_quality:
                st.success(quality_msg)
            else:
                st.warning(quality_msg)
        
        if not is_good_quality:
            st.warning("⚠️ Image quality may affect prediction accuracy. Proceed with caution.")
        
        # Analyze button
        st.markdown("---")
        
        if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
            
            with st.spinner("Analyzing MRI scan..."):
                
                # Make prediction based on model choice
                if model_choice == "Multi-Class Classification":
                    pred_class, confidence, all_probs, img_tensor = predict_multiclass(image, model_mc)
                    
                    # Results
                    st.markdown("---")
                    st.subheader("📋 Analysis Results")
                    
                    # Prediction
                    col_r1, col_r2, col_r3 = st.columns(3)
                    
                    with col_r1:
                        st.metric("Prediction", pred_class.upper())
                    
                    with col_r2:
                        confidence_color = get_confidence_color(confidence)
                        st.metric("Confidence", f"{confidence:.1%}")
                        st.markdown(f":{confidence_color}[{'●' * int(confidence * 10)}]")
                    
                    with col_r3:
                        status_msg = get_status_message(confidence)
                        if confidence >= 0.8:
                            st.success(status_msg)
                        elif confidence >= 0.6:
                            st.warning(status_msg)
                        else:
                            st.error(status_msg)
                    
                    # Class probabilities
                    st.markdown("---")
                    st.subheader("📊 Class Probabilities")
                    
                    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
                    prob_data = {
                        'Class': class_names,
                        'Probability': [f"{p:.1%}" for p in all_probs]
                    }
                    st.dataframe(prob_data, use_container_width=True, hide_index=True)
                    
                    # Grad-CAM
                    st.markdown("---")
                    st.subheader("🔥 Explainable AI - Grad-CAM")
                    st.caption("Heatmap shows where the model focuses attention")
                    
                    try:
                        heatmap = generate_gradcam(model_mc, img_tensor, model_mc.features[-1])
                        
                        # Prepare image for overlay
                        img_display = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                        img_display = np.clip(img_display, 0, 1)
                        
                        overlay = overlay_heatmap(img_display, heatmap, alpha=0.4)
                        
                        with col2:
                            st.subheader("🔥 Grad-CAM Overlay")
                            st.image(overlay, use_container_width=True)
                            st.caption("Red areas indicate model focus")
                        
                    except Exception as e:
                        st.error(f"Grad-CAM generation failed: {e}")
                
                else:  # Binary Detection
                    pred_class, confidence = predict_binary(image, model_bin)
                    
                    # Results
                    st.markdown("---")
                    st.subheader("📋 Analysis Results")
                    
                    col_r1, col_r2, col_r3 = st.columns(3)
                    
                    with col_r1:
                        if pred_class == 'yes':
                            st.metric("Detection", "TUMOR DETECTED", delta="Positive")
                        else:
                            st.metric("Detection", "NO TUMOR", delta="Negative")
                    
                    with col_r2:
                        confidence_color = get_confidence_color(confidence)
                        st.metric("Confidence", f"{confidence:.1%}")
                        st.markdown(f":{confidence_color}[{'●' * int(confidence * 10)}]")
                    
                    with col_r3:
                        status_msg = get_status_message(confidence)
                        if confidence >= 0.8:
                            st.success(status_msg)
                        elif confidence >= 0.6:
                            st.warning(status_msg)
                        else:
                            st.error(status_msg)
                
                # Recommendation
                st.markdown("---")
                st.subheader("💡 Recommendation")
                
                if confidence >= 0.8 and is_good_quality:
                    st.success("✅ High confidence prediction with good image quality. Results are reliable.")
                elif confidence >= 0.6:
                    st.warning("⚠️ Moderate confidence. Consider additional review or better quality image.")
                else:
                    st.error("🚨 Low confidence prediction. Manual expert review strongly recommended.")
    
    else:
        # Instructions when no file uploaded
        st.info("👆 Please upload an MRI image to begin analysis")
        
        st.markdown("---")
        st.subheader("📖 How to Use")
        st.markdown("""
        1. Select model type from sidebar
        2. Upload MRI image (JPG/PNG format)
        3. Review image quality check
        4. Click 'Analyze Image' button
        5. Review results and confidence score
        6. Check Grad-CAM heatmap (multi-class only)
        """)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()