import os
import keras
import tensorflow as tf
import tensorflow_hub as hub

# Set environment variable to match notebook context
os.environ["TF_USE_LEGACY_KERAS"] = "1"

demo_dir = "/home/haidang/Desktop/AdversarialTraining/demo"
models = [
    "my_flower_model_mobilenetv2.keras",
    "my_flower_model_resnet50.keras",
    "robust_flower_model_mobilenetv2_FGSM_100e.keras",
    "robust_flower_model_mobilenetv2_PGD_200e.keras",
    "robust_flower_model_resnet50_FGSM_150e.keras",
    "robust_flower_model_resnet50_PGD_200e.keras"
]

for model_name in models:
    model_path = os.path.join(demo_dir, model_name)
    if not os.path.exists(model_path):
        print(f"Skipping {model_name} (not found)")
        continue
    try:
        print(f"\n--- Testing {model_name} ---")
        m = keras.models.load_model(
            model_path,
            custom_objects={'KerasLayer': hub.KerasLayer},
            safe_mode=False
        )
        print(f"Success loading {model_name}!")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
