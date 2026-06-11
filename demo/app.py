import os
# Ensure legacy Keras environment variable is set
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

from models import load_flower_model, predict_image, labels
from attacks import fgsm_attack, pgd_attack

# Page configuration
st.set_page_config(
    page_title="Adversarial Training & Robustness Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caching model loading to optimize performance
@st.cache_resource
def load_all_models():
    model_non_robust = load_flower_model('my_flower_model_6e.keras')
    model_fgsm = load_flower_model('robust_flower_model_FGSM_100e.keras')
    model_pgd = load_flower_model('robust_flower_model_PGD_200e.keras')
    return model_non_robust, model_fgsm, model_pgd

try:
    model_non_robust, model_fgsm, model_pgd = load_all_models()
except Exception as e:
    st.error(f"Error loading models. Please verify that all three model files exist in the 'demo/' directory. Details: {e}")
    st.stop()

# Helper for image preprocessing
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    preprocessed = np.expand_dims(img_array, axis=0)
    return preprocessed, img_resized

# Helper to generate custom CSS styling for prediction cards
def make_prediction_card(model_name, label, confidence, color="#1f77b4"):
    return f"""
    <div style="border-left: 5px solid {color}; padding: 12px; border-radius: 4px; background-color: #f8f9fa; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <h5 style="margin: 0; color: #333; font-size: 0.95rem;">{model_name}</h5>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
            <span style="font-size: 1.1rem; font-weight: 600; color: #2b2b2b;">{label.upper()}</span>
            <span style="font-size: 1.1rem; font-weight: 600; color: {color};">{confidence*100:.2f}%</span>
        </div>
    </div>
    """

# Sidebar Layout
st.sidebar.title("🛠️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Flower Image", type=["png", "jpg", "jpeg"])

# Initialize session state variables
if 'orig_classified' not in st.session_state:
    st.session_state.orig_classified = False
if 'orig_results' not in st.session_state:
    st.session_state.orig_results = None
if 'adv_generated' not in st.session_state:
    st.session_state.adv_generated = False
if 'adv_image' not in st.session_state:
    st.session_state.adv_image = None
if 'noise_map' not in st.session_state:
    st.session_state.noise_map = None
if 'adv_results' not in st.session_state:
    st.session_state.adv_results = None

# Reset state if a new file is uploaded
if uploaded_file is not None:
    if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.orig_classified = False
        st.session_state.orig_results = None
        st.session_state.adv_generated = False
        st.session_state.adv_image = None
        st.session_state.noise_map = None
        st.session_state.adv_results = None

st.sidebar.subheader("⚔️ Attack Configuration")
attack_option = st.sidebar.selectbox(
    "Choose Attack Method",
    ["FGSM", "PGD"]
)
attack_model_option = st.sidebar.selectbox(
    "Model for Attack Generation",
    ["Non-Robust Model", "FGSM-Robust Model", "PGD-Robust Model"]
)

# Configure parameters based on attack option
if attack_option == "FGSM":
    epsilon = st.sidebar.slider(
        "Epsilon (perturbation limit)",
        min_value=0.001,
        max_value=0.100,
        value=0.010,
        step=0.001,
        format="%.3f"
    )
elif attack_option == "PGD":
    epsilon = st.sidebar.slider(
        "Epsilon (perturbation limit)",
        min_value=0.001,
        max_value=0.100,
        value=0.010,
        step=0.001,
        format="%.3f"
    )
    max_iter = st.sidebar.slider(
        "Max Iterations (steps)",
        min_value=5,
        max_value=50,
        value=10,
        step=1
    )
    alpha = 2.5 * epsilon / max_iter
    st.sidebar.caption(f"Calculated Step Size (Alpha): **{alpha:.5f}**")

# Main Area Layout
st.title("🛡️ Adversarial Training Research Dashboard")
st.markdown("This interface evaluates flower classification model vulnerability and defense efficiency against standard adversarial attacks.")

col_left, col_right = st.columns(2)

if uploaded_file is not None:
    # Load and preprocess uploaded image
    preprocessed_img, display_img = preprocess_image(uploaded_file)
    
    # Left Column: Original Image & Classification results
    with col_left:
        st.subheader("📸 Original Image")
        st.image(display_img, use_container_width=True)
        
        classify_button = st.button("🔍 Classify Original Image", use_container_width=True)
        
        if classify_button or st.session_state.orig_classified:
            if not st.session_state.orig_classified:
                with st.spinner("Classifying..."):
                    lbl_non, conf_non, _ = predict_image(model_non_robust, preprocessed_img)
                    lbl_fgsm, conf_fgsm, _ = predict_image(model_fgsm, preprocessed_img)
                    lbl_pgd, conf_pgd, _ = predict_image(model_pgd, preprocessed_img)
                    
                    st.session_state.orig_results = {
                        "Non-Robust Model": (lbl_non, conf_non, "#6c757d"),
                        "FGSM-Robust Model": (lbl_fgsm, conf_fgsm, "#fd7e14"),
                        "PGD-Robust Model": (lbl_pgd, conf_pgd, "#20c997")
                    }
                    st.session_state.orig_classified = True
            
            st.markdown("### Original Model Predictions")
            for model_name, (lbl, conf, color) in st.session_state.orig_results.items():
                st.markdown(make_prediction_card(model_name, lbl, conf, color), unsafe_allow_html=True)

    # Right Column: Perturbation, Adversarial Image & results
    with col_right:
        st.subheader("😈 Adversarial Vulnerability Analysis")
        
        generate_button = st.sidebar.button("⚡ Generate Adversarial Image", use_container_width=True)
        
        if generate_button or st.session_state.adv_generated:
            if generate_button:
                with st.spinner("Generating adversarial perturbation..."):
                    # Classify if not done to get predicted index
                    if not st.session_state.orig_classified:
                        lbl_non, conf_non, _ = predict_image(model_non_robust, preprocessed_img)
                        lbl_fgsm, conf_fgsm, _ = predict_image(model_fgsm, preprocessed_img)
                        lbl_pgd, conf_pgd, _ = predict_image(model_pgd, preprocessed_img)
                        st.session_state.orig_results = {
                            "Non-Robust Model": (lbl_non, conf_non, "#6c757d"),
                            "FGSM-Robust Model": (lbl_fgsm, conf_fgsm, "#fd7e14"),
                            "PGD-Robust Model": (lbl_pgd, conf_pgd, "#20c997")
                        }
                        st.session_state.orig_classified = True
                    
                    # Map selected model option to the model instance
                    if attack_model_option == "Non-Robust Model":
                        target_model = model_non_robust
                    elif attack_model_option == "FGSM-Robust Model":
                        target_model = model_fgsm
                    else:
                        target_model = model_pgd

                    # Target index is the prediction of the selected model to be attacked
                    orig_pred_label = st.session_state.orig_results[attack_model_option][0]
                    label_idx = labels.index(orig_pred_label)
                    
                    # Attack execution using the selected model
                    if attack_option == "FGSM":
                        adv_arr = fgsm_attack(target_model, preprocessed_img, label_idx, epsilon)
                    elif attack_option == "PGD":
                        adv_arr = pgd_attack(target_model, preprocessed_img, label_idx, epsilon, max_iter, alpha)
                        
                    # Calculate noise map
                    noise_map = adv_arr - preprocessed_img
                    
                    # Run predictions on perturbed image
                    lbl_non_adv, conf_non_adv, _ = predict_image(model_non_robust, adv_arr)
                    lbl_fgsm_adv, conf_fgsm_adv, _ = predict_image(model_fgsm, adv_arr)
                    lbl_pgd_adv, conf_pgd_adv, _ = predict_image(model_pgd, adv_arr)
                    
                    st.session_state.adv_image = adv_arr
                    st.session_state.noise_map = noise_map
                    st.session_state.adv_results = {
                        "Non-Robust Model": (lbl_non_adv, conf_non_adv, "#6c757d"),
                        "FGSM-Robust Model": (lbl_fgsm_adv, conf_fgsm_adv, "#fd7e14"),
                        "PGD-Robust Model": (lbl_pgd_adv, conf_pgd_adv, "#20c997")
                    }
                    st.session_state.adv_generated = True
            
            # Display adversarial image and noise map side-by-side
            sub_col1, sub_col2 = st.columns(2)
            
            # Convert adv image array to PIL
            adv_pil = Image.fromarray((st.session_state.adv_image[0] * 255).astype(np.uint8))
            
            # Shift and scale noise map to [0, 1] with 0.5 as neutral gray
            noise_raw = st.session_state.noise_map[0]
            min_val = np.min(noise_raw)
            max_val = np.max(noise_raw)
            abs_max = max(abs(min_val), abs(max_val))
            if abs_max > 0:
                # Scale to [-0.5, 0.5] and shift to [0, 1]
                noise_display = (noise_raw / (2.0 * abs_max)) + 0.5
            else:
                noise_display = np.zeros_like(noise_raw) + 0.5
            noise_pil = Image.fromarray((noise_display * 255).astype(np.uint8))
            
            with sub_col1:
                st.markdown("**Adversarial Image**")
                st.image(adv_pil, use_container_width=True)
            with sub_col2:
                st.markdown("**Noise Map (Perturbation)**")
                st.image(noise_pil, use_container_width=True)
                
            st.markdown("### Model Predictions after Attack")
            for model_name, (lbl, conf, color) in st.session_state.adv_results.items():
                st.markdown(make_prediction_card(model_name, lbl, conf, color), unsafe_allow_html=True)
        else:
            st.info("👈 Please set attack parameters in the sidebar and click 'Generate Adversarial Image' to view adversarial effects.")
else:
    st.info("👈 Please upload an image file in the sidebar control panel to start the demo.")
