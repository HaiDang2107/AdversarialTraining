#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
explain_model.py

Explain flower classification models using SHAP (SHapley Additive exPlanations).
This script loads the three trained MobileNetV2 models (non-robust, FGSM-robust, PGD-robust)
and explains their predictions on a test image using the SHAP partition explainer for images.

Conda environment: human_detection
"""

import os
import sys
import argparse

# Set non-interactive backend for matplotlib BEFORE importing plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure TensorFlow legacy Keras environment variable is set
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import shap
from PIL import Image

# Add the script's directory to python path to resolve local imports (models.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from models import load_flower_model, labels
except ImportError as e:
    print(f"Error: Could not import models.py. Make sure you are running in the correct directory. Details: {e}")
    sys.exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Explain 3 MobileNetV2 models using SHAP.")
    
    # Default image path from the flower test set
    default_img = os.path.join(
        current_dir, "flower_photos_test", "daisy", "10555749515_13a12a026e.jpg"
    )
    
    parser.add_argument(
        "--image_path",
        type=str,
        default=default_img,
        help="Path to the input image for analysis."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(current_dir, "shap_explanations"),
        help="Directory where the generated SHAP explanation plots will be saved."
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=2000,
        help="Number of evaluations/perturbations for SHAP (higher values, e.g. 1500-2000, make the heatmap resolution finer but computation slower)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size used during SHAP evaluation."
    )
    return parser.parse_args()


def preprocess_image(image_path):
    """
    Loads and preprocesses an image to match the model's expected input shape and scale.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image file not found at: {image_path}")
        
    print(f"Loading and preprocessing image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    preprocessed_batch = np.expand_dims(img_array, axis=0)
    return preprocessed_batch, img_resized


def explain_model_with_shap(model, model_name, img_batch, max_evals, batch_size, output_dir):
    """
    Runs prediction, computes SHAP partition explanation, and saves the plot.
    """
    print(f"\n" + "=" * 50)
    print(f"Analyzing Model: {model_name}")
    print("=" * 50)
    
    # 1. Run prediction
    preds = model.predict(img_batch, verbose=0)
    top_class_idx = np.argmax(preds[0])
    predicted_label = labels[top_class_idx]
    confidence = preds[0][top_class_idx]
    print(f"Prediction: {predicted_label.upper()} with confidence {confidence * 100:.2f}%")
    
    # 2. Setup SHAP image explainer (Partition explainer)
    # Using 'inpaint_telea' masker to simulate occlusion
    masker = shap.maskers.Image("inpaint_telea", (224, 224, 3))
    explainer = shap.Explainer(model, masker, output_names=labels)
    
    print(f"Computing SHAP values for target class '{predicted_label}'...")
    print(f"Evaluations (max_evals): {max_evals}, Batch size: {batch_size}")
    
    # Compute shap values specifically for the top predicted class
    shap_values = explainer(
        img_batch, 
        max_evals=max_evals, 
        batch_size=batch_size, 
        outputs=[top_class_idx]
    )
    
    # 3. Plot and save explanation
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    clean_model_name = os.path.splitext(model_name)[0]
    output_plot_path = os.path.join(output_dir, f"shap_{clean_model_name}.png")
    
    print("Generating and saving SHAP plot...")
    plt.figure(figsize=(10, 5))
    shap.image_plot(shap_values, show=False)
    
    # Add title using matplotlib after shap.image_plot finishes
    plt.suptitle(
        f"SHAP Explanation: {clean_model_name}\nPrediction: {predicted_label.upper()} ({confidence * 100:.2f}%)", 
        fontsize=14, 
        weight='bold', 
        y=0.95
    )
    
    plt.savefig(output_plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"SHAP explanation successfully saved to: {output_plot_path}")


def main():
    args = parse_arguments()
    
    # List of the 3 MobileNetV2 models to load and explain
    models_to_explain = [
        'my_flower_model_6e.keras',
        'robust_flower_model_FGSM_100e.keras',
        'robust_flower_model_PGD_200e.keras'
    ]
    
    try:
        # Load and preprocess image
        img_batch, _ = preprocess_image(args.image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
        
    for model_file in models_to_explain:
        try:
            print(f"\nLoading model: {model_file}...")
            model = load_flower_model(model_file)
            explain_model_with_shap(
                model=model,
                model_name=model_file,
                img_batch=img_batch,
                max_evals=args.max_evals,
                batch_size=args.batch_size,
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"Failed to explain model {model_file}. Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"SHAP analysis completed! All plots saved in: {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
