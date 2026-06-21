import zipfile
import json
import os
import tf_keras as keras
import tensorflow as tf
import tensorflow_hub as hub

# Set legacy Keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"

original_model_path = "/home/haidang/Desktop/AdversarialTraining/demo/robust_flower_model_resnet50_PGD_200e.keras"
temp_model_path = "/home/haidang/Desktop/AdversarialTraining/brain/5630263c-8138-4053-8b20-759079a7fc5e/scratch/temp_model_tf_keras.keras"

# 1. Define custom PreprocessLayer
class PreprocessLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        # Remove any Lambda config args to avoid unrecognized kwargs error
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

# 2. Modify config.json
with zipfile.ZipFile(original_model_path, 'r') as src_zip:
    with zipfile.ZipFile(temp_model_path, 'w') as dst_zip:
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
                        # Clean config to only have basic Layer parameters
                        clean_config = {
                            'name': layer['config']['name'],
                            'trainable': layer['config']['trainable'],
                            'dtype': layer['config']['dtype']
                        }
                        layer['config'] = clean_config
                
                dst_zip.writestr(item.filename, json.dumps(config, indent=2))
            else:
                dst_zip.writestr(item, src_zip.read(item.filename))

# 3. Load with tf_keras
try:
    print("Loading modified model with tf_keras...")
    model = keras.models.load_model(
        temp_model_path,
        custom_objects={
            'KerasLayer': hub.KerasLayer,
            'PreprocessLayer': PreprocessLayer
        },
        safe_mode=False
    )
    print("Successfully loaded modified model with tf_keras!")
    model.summary()
except Exception as e:
    print(f"Failed to load: {e}")
    import traceback
    traceback.print_exc()
