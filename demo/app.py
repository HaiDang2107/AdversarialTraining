import os
import io
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

# Helper to determine preprocess function for a given model type
def get_preprocess_fn(arch, model_type):
    if arch == "MobileNetV2":
        if model_type == "Non-Robust Model":
            # Clean MobileNetV2 expects [-1, 1] input range
            return lambda x: x * 2.0 - 1.0
        else:
            # Robust MobileNetV2 expects [0, 1] input range
            return None
    else:  # ResNet50
        if model_type == "Non-Robust Model":
            # Clean ResNet50 expects BGR and ImageNet mean subtraction
            def resnet_preprocess(x):
                x_scaled = x * 255.0
                r = x_scaled[..., 0]
                g = x_scaled[..., 1]
                b = x_scaled[..., 2]
                b_pre = b - 103.939
                g_pre = g - 116.779
                r_pre = r - 123.68
                if isinstance(x, tf.Tensor):
                    return tf.stack([b_pre, g_pre, r_pre], axis=-1)
                else:
                    return np.stack([b_pre, g_pre, r_pre], axis=-1)
            return resnet_preprocess
        else:
            # Robust ResNet50 models have embedded lambda preprocessing layer, so they expect [0, 1]
            return None

# Helper to get preprocessed model input
def get_model_input(img, arch, model_type):
    fn = get_preprocess_fn(arch, model_type)
    if fn is not None:
        return fn(img)
    return img


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

architecture = st.sidebar.selectbox(
    "Select Model Architecture",
    ["MobileNetV2", "ResNet50"]
)

# Caching model loading to optimize performance
@st.cache_resource
def load_all_models(arch):
    if arch == "MobileNetV2":
        model_non_robust = load_flower_model('my_flower_model_mobilenetv2.keras')
        model_fgsm = load_flower_model('robust_flower_model_mobilenetv2_FGSM_100e.keras')
        model_pgd = load_flower_model('robust_flower_model_mobilenetv2_PGD_200e.keras')
    else:  # ResNet50
        model_non_robust = load_flower_model('my_flower_model_resnet50.keras')
        model_fgsm = load_flower_model('robust_flower_model_resnet50_FGSM_150e.keras')
        model_pgd = load_flower_model('robust_flower_model_resnet50_PGD_200e.keras')
    return model_non_robust, model_fgsm, model_pgd

try:
    model_non_robust, model_fgsm, model_pgd = load_all_models(architecture)
except Exception as e:
    st.error(f"Error loading {architecture} models. Please verify that all three model files exist in the 'demo/' directory. Details: {e}")
    st.stop()

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

# Reset state if a new file is uploaded or architecture changes
if uploaded_file is not None:
    if ('current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name or
        'current_arch' not in st.session_state or st.session_state.current_arch != architecture):
        st.session_state.current_file = uploaded_file.name
        st.session_state.current_arch = architecture
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
                    lbl_non, conf_non, _ = predict_image(model_non_robust, get_model_input(preprocessed_img, architecture, "Non-Robust Model"))
                    lbl_fgsm, conf_fgsm, _ = predict_image(model_fgsm, get_model_input(preprocessed_img, architecture, "FGSM-Robust Model"))
                    lbl_pgd, conf_pgd, _ = predict_image(model_pgd, get_model_input(preprocessed_img, architecture, "PGD-Robust Model"))
                    
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
                        lbl_non, conf_non, _ = predict_image(model_non_robust, get_model_input(preprocessed_img, architecture, "Non-Robust Model"))
                        lbl_fgsm, conf_fgsm, _ = predict_image(model_fgsm, get_model_input(preprocessed_img, architecture, "FGSM-Robust Model"))
                        lbl_pgd, conf_pgd, _ = predict_image(model_pgd, get_model_input(preprocessed_img, architecture, "PGD-Robust Model"))
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
                    target_preprocess_fn = get_preprocess_fn(architecture, attack_model_option)
                    if attack_option == "FGSM":
                        adv_arr = fgsm_attack(target_model, preprocessed_img, label_idx, epsilon, target_preprocess_fn)
                    elif attack_option == "PGD":
                        adv_arr = pgd_attack(target_model, preprocessed_img, label_idx, epsilon, max_iter, alpha, target_preprocess_fn)
                        
                    # Calculate noise map
                    noise_map = adv_arr - preprocessed_img
                    
                    # Run predictions on perturbed image
                    lbl_non_adv, conf_non_adv, _ = predict_image(model_non_robust, get_model_input(adv_arr, architecture, "Non-Robust Model"))
                    lbl_fgsm_adv, conf_fgsm_adv, _ = predict_image(model_fgsm, get_model_input(adv_arr, architecture, "FGSM-Robust Model"))
                    lbl_pgd_adv, conf_pgd_adv, _ = predict_image(model_pgd, get_model_input(adv_arr, architecture, "PGD-Robust Model"))
                    
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
                
                # Convert PIL to bytes for download
                buf = io.BytesIO()
                adv_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="💾 Download Adversarial Image",
                    data=byte_im,
                    file_name=f"adversarial_{attack_option}_{epsilon:.3f}.png",
                    mime="image/png",
                    use_container_width=True
                )
            with sub_col2:
                st.markdown("**Noise Map (Perturbation)**")
                st.image(noise_pil, use_container_width=True)
                
                # Convert noise PIL to bytes for download
                buf_noise = io.BytesIO()
                noise_pil.save(buf_noise, format="PNG")
                byte_noise = buf_noise.getvalue()
                
                st.download_button(
                    label="💾 Download Noise Map",
                    data=byte_noise,
                    file_name=f"noise_{attack_option}_{epsilon:.3f}.png",
                    mime="image/png",
                    use_container_width=True
                )
                
            st.markdown("### Model Predictions after Attack")
            for model_name, (lbl, conf, color) in st.session_state.adv_results.items():
                st.markdown(make_prediction_card(model_name, lbl, conf, color), unsafe_allow_html=True)
        else:
            st.info("👈 Please set attack parameters in the sidebar and click 'Generate Adversarial Image' to view adversarial effects.")
else:
    st.info("👈 Please upload an image file in the sidebar control panel to start the demo.")
