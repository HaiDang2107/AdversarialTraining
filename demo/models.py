import os
# Set environment variable to match notebook context BEFORE other imports
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

# Define labels based on the flower dataset classes in alphabetical order
labels = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def load_flower_model(model_name):
    """
    Loads a specific flower classification model from the demo directory.
    model_name can be one of:
      - 'my_flower_model_6e.keras'
      - 'robust_flower_model_FGSM_100e.keras'
      - 'robust_flower_model_PGD_200e.keras'
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
        
    # hub.KerasLayer is required in custom_objects because the models use pre-trained hub weights
    model = keras.models.load_model(
        model_path, 
        custom_objects={'KerasLayer': hub.KerasLayer}
    )
    return model

def predict_image(model, preprocessed_image):
    """
    Runs model inference on the preprocessed image.
    Returns:
      - predicted_label: string
      - confidence: float (between 0 and 1)
      - probabilities: list of floats for all classes
    """
    predictions = model.predict(preprocessed_image, verbose=0)
    idx = np.argmax(predictions[0])
    confidence = predictions[0][idx]
    
    return labels[idx], float(confidence), predictions[0].tolist()
