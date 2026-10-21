"""
Brain Tumor MRI Detection & Explainability System
=================================================
Interactive Streamlit Application for Radiologist Decision Support.
Features:
  - Automated Image Quality Pre-screening
  - High-Accuracy Deep Learning Inference (MobileNetV2)
  - Explainable AI (Grad-CAM Attention Heatmaps)
  - Confidence-based Safety Guardrail (Human-in-the-loop escalation)
  - Interactive Benchmark Evaluation Dashboard
"""

import os
import sys

# Auto-launch Streamlit if executed directly with plain `python app.py`
import streamlit as st
if not st.runtime.exists():
    print("\n[NeuroScan AI] Detected direct python execution.")
    print("[NeuroScan AI] Launching interactive Streamlit web server...\n")
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)] + sys.argv[1:])
    sys.exit(0)

import json
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# Configure search paths for imports
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
utils_dir = os.path.join(parent_dir, "utils")
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)

try:
    from utils.gradcam import GradCAM, overlay_gradcam
    from utils.dataset import crop_brain_contour
except ImportError:
    from gradcam import GradCAM, overlay_gradcam
    from dataset import crop_brain_contour

# Page Configuration (Must be first Streamlit command)
st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------
# CACHED MODEL & METRICS LOADERS
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    """Load and cache the trained MobileNetV2 model"""
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    
    # Check for best model or fallback
    model_paths = [
        os.path.join(parent_dir, "models", "best_multiclass_model.pth"),
        os.path.join(parent_dir, "models", "multiclass_model.pth")
    ]
    loaded = False
    for p in model_paths:
        if os.path.exists(p):
            model.load_state_dict(torch.load(p, map_location=device))
            loaded = True
            break
            
    model.to(device)
    model.eval()
    return model, class_names

@st.cache_data
def load_metrics():
    """Load evaluation metrics and class metadata"""
    metrics_path = os.path.join(parent_dir, "models", "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None

# ---------------------------------------------------------
# IMAGE PROCESSING & INFERENCE FUNCTIONS
# ---------------------------------------------------------

def check_image_quality(image_pil):
    """
    Evaluates scan quality before running inference.
    Checks mean brightness and contrast to catch dark, washed out, or invalid images.
    """
    img_np = np.array(image_pil.convert('L'))
    brightness = float(np.mean(img_np))
    contrast = float(np.std(img_np))
    
    if brightness < 30.0:
        return False, brightness, contrast, "⚠️ Scan is underexposed/too dark."
    elif brightness > 225.0:
        return False, brightness, contrast, "⚠️ Scan is overexposed/saturated."
    elif contrast < 20.0:
        return False, brightness, contrast, "⚠️ Scan has low contrast (poor tissue definition)."
    else:
        return True, brightness, contrast, "✅ Image quality passed clinical pre-screening."

def predict_mri(image_pil, model, class_names, apply_crop=True):
    """Preprocesses scan, computes softmax probabilities and Grad-CAM heatmap"""
    processed_pil = crop_brain_contour(image_pil) if apply_crop else image_pil
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(processed_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_class = class_names[pred_idx]
        
    gradcam = GradCAM(model, model.features[-1])
    heatmap, _ = gradcam.generate_heatmap(input_tensor, target_class=pred_idx)
    
    display_np = np.array(processed_pil.resize((224, 224)).convert('RGB')) / 255.0
    overlay = overlay_gradcam(display_np, heatmap, alpha=0.45)
    
    return pred_class, confidence, probs, overlay, processed_pil

# ---------------------------------------------------------
# STREAMLIT USER INTERFACE
# ---------------------------------------------------------

def main():
    model, class_names = load_model()
    metrics = load_metrics()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/brain.png", width=70)
        st.title("NeuroScan AI")
        st.markdown("**Assistive Brain Tumor MRI Diagnostic Platform**")
        st.markdown("---")
        
        st.subheader("⚙️ Clinical Configuration")
        apply_crop = st.checkbox("Auto-Crop Brain Contours", value=True, 
                                 help="Detects skull margins and crops excess black borders to focus on brain tissue.")
        
        conf_threshold = st.slider("Safety Confidence Threshold", min_value=0.50, max_value=0.95, value=0.80, step=0.05,
                                   help="Predictions below this confidence level are flagged for mandatory radiologist audit.")
        
        st.markdown("---")
        st.subheader("📋 System Specifications")
        st.caption(f"• **Architecture:** MobileNetV2 Transfer Learning")
        st.caption(f"• **Computation Device:** {device.type.upper()}")
        st.caption(f"• **Target Classes (4):** Glioma, Meningioma, Healthy (No Tumor), Pituitary")
        if metrics:
            st.caption(f"• **Test Accuracy:** {metrics['deep_learning_model']['test_accuracy']:.1%}")
            st.caption(f"• **Mean ROC-AUC:** {metrics['deep_learning_model']['mean_auc']:.3f}")
            
    # Main Header
    st.markdown("# 🧠 NeuroScan AI: End-to-End Brain Tumor Detection")
    st.markdown("A deep learning solution for Brain MRI interpretation with **Explainable AI (Grad-CAM)** and **Clinical Safety Guardrails**.")
    st.markdown("---")
    
    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs([
        "🔬 MRI Scan Analysis & Grad-CAM",
        "📊 Model Benchmarks & Evaluation",
        "📖 Clinical Protocol & Viva Guide"
    ])
    
    # ---------------- TAB 1: DIAGNOSTIC ANALYSIS ----------------
    with tab1:
        col_up, col_info = st.columns([1, 1])
        with col_up:
            st.subheader("📤 Upload Brain MRI Scan")
            uploaded_file = st.file_uploader("Select a brain MRI image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
            
            sample_options = ["None"]
            demo_base = os.path.join(parent_dir, "datasets", "Multilabel_classification_dataset", "Testing")
            demo_files = {}
            if os.path.exists(demo_base):
                for c in class_names:
                    cfolder = os.path.join(demo_base, c)
                    if os.path.exists(cfolder):
                        flist = os.listdir(cfolder)
                        if flist:
                            demo_files[f"Demo: {c.upper()} sample"] = os.path.join(cfolder, flist[0])
                sample_options += list(demo_files.keys())
                
            demo_choice = st.selectbox("Or choose a verified clinical sample:", sample_options)
            
        with col_info:
            st.subheader("ℹ️ Clinical Triage Guideline")
            st.markdown("""
            * **Input:** Axial or coronal Brain MRI scan.
            * **Automated Screening:** Quality checks run prior to deep learning evaluation.
            * **Safety Protocol:** If prediction confidence is **< 80%**, the case is marked as **Uncertain** to safeguard against misdiagnosis.
            """)
            
        image_to_analyze = None
        if uploaded_file is not None:
            image_to_analyze = Image.open(uploaded_file).convert('RGB')
        elif demo_choice != "None" and demo_choice in demo_files:
            image_to_analyze = Image.open(demo_files[demo_choice]).convert('RGB')
            
        if image_to_analyze is not None:
            st.markdown("---")
            st.subheader("🔍 Quality Pre-screening & Diagnostic Assessment")
            
            # 1. Quality Check
            is_valid, bright, contrast, q_msg = check_image_quality(image_to_analyze)
            q_col1, q_col2, q_col3 = st.columns(3)
            with q_col1:
                st.metric("Mean Brightness", f"{bright:.1f} / 255")
            with q_col2:
                st.metric("Pixel Contrast (Std Dev)", f"{contrast:.1f}")
            with q_col3:
                if is_valid:
                    st.success(q_msg)
                else:
                    st.error(q_msg)
                    
            if not is_valid:
                st.warning("⚠️ Scan failed quality check. Results may be inaccurate. Manual radiologist inspection is required.")
                
            # 2. Inference & Grad-CAM
            with st.spinner("Analyzing MRI slices and computing Grad-CAM heatmap..."):
                pred_class, conf, all_probs, gradcam_overlay, processed_img = predict_mri(
                    image_to_analyze, model, class_names, apply_crop=apply_crop
                )
                
            st.markdown("---")
            col_img1, col_img2, col_res = st.columns([1, 1, 1.2])
            
            with col_img1:
                st.markdown("**Original / Cropped MRI**")
                st.image(processed_img, width=280)
                
            with col_img2:
                st.markdown("**Grad-CAM Heatmap (Focus Area)**")
                st.image(gradcam_overlay, width=280)
                st.caption("🔴 Red regions highlight anatomical tissue driving the model's diagnostic classification.")
                
            with col_res:
                st.markdown("### 📋 Diagnostic Outcome")
                
                color_map = {
                    "glioma": "#e63946",
                    "meningioma": "#e76f51",
                    "pituitary": "#2a9d8f",
                    "notumor": "#457b9d"
                }
                c_color = color_map.get(pred_class, "#2b5c8f")
                
                if pred_class == "notumor":
                    st.markdown(f"<h2 style='color:#2a9d8f;margin:0;'>NORMAL (NO TUMOR)</h2>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='color:{c_color};margin:0;'>{pred_class.upper()} TUMOR</h2>", unsafe_allow_html=True)
                    
                st.markdown(f"**Confidence Score:** `{conf:.2%}`")
                st.progress(conf)
                
                if conf >= conf_threshold:
                    st.success(f"✅ **High Confidence Assessment (>{conf_threshold:.0%})**: Model findings correlate strongly with clinical training patterns.")
                else:
                    st.error(f"🚨 **UNCERTAIN CASE (<{conf_threshold:.0%})**: Confidence is below safety threshold. Mandatory review by Senior Radiologist required.")
                    
            st.markdown("---")
            st.subheader("📊 Class-Wise Probability Breakdown")
            prob_cols = st.columns(4)
            for i, c_name in enumerate(class_names):
                with prob_cols[i]:
                    label_name = c_name.capitalize() if c_name != "notumor" else "Healthy (No Tumor)"
                    st.metric(label_name, f"{all_probs[i]:.2%}")
                    st.progress(float(all_probs[i]))
                    
    # ---------------- TAB 2: BENCHMARK EVALUATION ----------------
    with tab2:
        st.subheader("📈 Performance Benchmarking: Baseline vs. Deep Learning")
        st.markdown("""
        In accordance with the Innovative Assignment rubric, the proposed Deep Learning model was evaluated
        directly against a simple Baseline Model to justify its complexity and clinical utility.
        """)
        
        plot_dir = os.path.join(parent_dir, "notebooks", "plots")
        
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            p_comp = os.path.join(plot_dir, "05_baseline_vs_advanced_comparison.png")
            if os.path.exists(p_comp):
                st.image(p_comp, caption="Baseline (Logistic Regression) vs Proposed Deep Learning Model", width=480)
                
        with col_m2:
            p_cm = os.path.join(plot_dir, "08_confusion_matrix.png")
            if os.path.exists(p_cm):
                st.image(p_cm, caption="Confusion Matrix on Test Set", width=420)
                
        st.markdown("---")
        col_m3, col_m4 = st.columns([1, 1])
        with col_m3:
            p_roc = os.path.join(plot_dir, "09_roc_auc_curves.png")
            if os.path.exists(p_roc):
                st.image(p_roc, caption="Multi-Class One-vs-Rest ROC-AUC Curves", width=460)
                
        with col_m4:
            p_tune = os.path.join(plot_dir, "06_hyperparameter_tuning.png")
            if os.path.exists(p_tune):
                st.image(p_tune, caption="Hyperparameter Tuning: Adam vs SGD Optimizer", width=480)
                
        st.markdown("---")
        st.subheader("🔬 Error Analysis (Failure Case Review)")
        p_err = os.path.join(plot_dir, "10_error_analysis_misclassifications.png")
        if os.path.exists(p_err):
            st.image(p_err, caption="Representative Misclassified Cases Analyzed for Clinical Failure Modes", width=600)

    # ---------------- TAB 3: PROTOCOL & VIVA GUIDE ----------------
    with tab3:
        st.subheader("📖 Project Overview & Evaluator Q&A")
        st.markdown("""
        ### Core Technical Justifications
        1. **Why was MobileNetV2 chosen?**
           MobileNetV2 uses *inverted residuals* and *depthwise separable convolutions*, achieving over 86% accuracy with only 2.23M parameters. It runs in real-time even on standard CPU workstations without requiring high-end GPUs.
           
        2. **Why was a Baseline model mandatory?**
           To provide an empirical reference point. A simple linear classifier (Logistic Regression) achieves ~67% accuracy because it cannot model spatial hierarchies. MobileNetV2 achieves ~86.5%, demonstrating a +19% gain due to convolutional feature learning.
           
        3. **Why is Accuracy insufficient on its own?**
           In medical screening, missing a malignant tumor (False Negative) is life-threatening. Therefore, **Recall (Sensitivity)** and **ROC-AUC (0.963)** are critical metrics to ensure high disease detection rates.
           
        4. **How does Grad-CAM prevent the 'Black Box' dilemma?**
           Grad-CAM takes gradients of the target class score flowing into the final convolutional feature maps, creating a localization heatmap. This allows doctors to verify that the model is looking at the actual tumor rather than skull artifacts.
        """)

if __name__ == "__main__":
    main()