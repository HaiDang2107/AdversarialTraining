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

def load_keras_model_safely(model_path, custom_objects=None):
    if custom_objects is None:
        custom_objects = {}
    custom_objects['KerasLayer'] = hub.KerasLayer
    
    try:
        import zipfile
        import json
        import tempfile
        
        has_lambda_preprocessing = False
        if zipfile.is_zipfile(model_path):
            with zipfile.ZipFile(model_path, 'r') as zf:
                if 'config.json' in zf.namelist():
                    config_data = zf.read('config.json').decode('utf-8')
                    config = json.loads(config_data)
                    if 'config' in config and 'layers' in config['config']:
                        for layer in config['config']['layers']:
                            if layer.get('class_name') == 'Lambda' and layer.get('config', {}).get('name') == 'resnet50_preprocessing':
                                has_lambda_preprocessing = True
                                break
                                
        if has_lambda_preprocessing:
            print("Phát hiện mô hình chứa lớp Lambda tiền xử lý. Đang tự động chuyển đổi để tương thích với phiên bản Python hiện tại...")
            temp_fd, temp_path = tempfile.mkstemp(suffix='.keras')
            os.close(temp_fd)
            
            try:
                class PreprocessLayer(keras.layers.Layer):
                    def __init__(self, **kwargs):
                        kwargs.pop('function', None)
                        kwargs.pop('function_type', None)
                        kwargs.pop('module', None)
                        kwargs.pop('output_shape', None)
                        kwargs.pop('output_shape_type', None)
                        kwargs.pop('output_shape_module', None)
                        kwargs.pop('arguments', None)
                        kwargs.pop('batch_input_shape', None)
                        super().__init__(**kwargs)
                    def call(self, inputs):
                        return tf.keras.applications.resnet50.preprocess_input(inputs * 255.0)
                
                custom_objects['PreprocessLayer'] = PreprocessLayer
                
                with zipfile.ZipFile(model_path, 'r') as src_zip:
                    with zipfile.ZipFile(temp_path, 'w') as dst_zip:
                        for item in src_zip.infolist():
                            if item.filename == 'config.json':
                                config_data = src_zip.read(item.filename).decode('utf-8')
                                config = json.loads(config_data)
                                layers = config['config']['layers']
                                for layer in layers:
                                    if layer.get('class_name') == 'Lambda' and layer.get('config', {}).get('name') == 'resnet50_preprocessing':
                                        layer['module'] = None
                                        layer['class_name'] = 'PreprocessLayer'
                                        layer['registered_name'] = 'PreprocessLayer'
                                        clean_config = {
                                            'name': layer['config']['name'],
                                            'trainable': layer['config']['trainable'],
                                            'dtype': layer['config']['dtype']
                                        }
                                        layer['config'] = clean_config
                                dst_zip.writestr(item.filename, json.dumps(config, indent=2))
                            else:
                                dst_zip.writestr(item, src_zip.read(item.filename))
                                
                model = keras.models.load_model(temp_path, custom_objects=custom_objects, safe_mode=False)
                return model
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    except Exception as e:
        print(f"Lưu ý: Không thể tự động chuyển đổi lớp Lambda ({e}). Thử tải trực tiếp...")
        
    return keras.models.load_model(model_path, custom_objects=custom_objects, safe_mode=False)

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
        
    return load_keras_model_safely(model_path)

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
