import keras
import tensorflow as tf
import os
import tensorflow_hub as hub

model_path = "/home/haidang/Desktop/AdversarialTraining/demo/robust_flower_model_resnet50_PGD_200e.keras"

class SafeLambda(keras.layers.Lambda):
    @classmethod
    def from_config(cls, config, custom_objects=None):
        name = config.get('name')
        print(f"Deserializing Lambda layer with name: {name}")
        
        if name == 'resnet50_preprocessing':
            func = lambda x: tf.keras.applications.resnet50.preprocess_input(x * 255.0)
        else:
            func = lambda x: x
            
        init_kwargs = {
            'function': func,
            'name': name,
            'trainable': config.get('trainable', True),
            'dtype': config.get('dtype', 'float32'),
            'arguments': config.get('arguments', {})
        }
        
        return cls(**init_kwargs)

try:
    print("Trying to load model with SafeLambda...")
    model = keras.models.load_model(
        model_path,
        custom_objects={
            'KerasLayer': hub.KerasLayer,
            'Lambda': SafeLambda
        },
        safe_mode=False
    )
    print("Successfully loaded model!")
    model.summary()
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
