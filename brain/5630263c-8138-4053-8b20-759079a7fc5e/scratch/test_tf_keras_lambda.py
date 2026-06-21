import os
import tf_keras as keras
import tensorflow as tf
import tensorflow_hub as hub

# Set legacy keras flag
os.environ["TF_USE_LEGACY_KERAS"] = "1"

model_path = "/home/haidang/Desktop/AdversarialTraining/demo/robust_flower_model_resnet50_PGD_200e.keras"

class SafeLambda(keras.layers.Lambda):
    @classmethod
    def from_config(cls, config, custom_objects=None):
        name = config.get('name')
        print(f"tf_keras deserializing Lambda layer: {name}")
        if name == 'resnet50_preprocessing':
            # In tf_keras / Keras 2, we can just assign the Python function to config['function']
            # or directly define the layer.
            # Let's inspect config first.
            print("Config keys:", config.keys())
            # We can replace the function with our custom preprocessing function
            config['function'] = lambda x: tf.keras.applications.resnet50.preprocess_input(x * 255.0)
            # We also need to remove or replace python bytecode to prevent it from failing
            config['function_type'] = 'lambda'
        
        # Let's call the base class deserialization
        try:
            return super(SafeLambda, cls).from_config(config, custom_objects)
        except Exception as e:
            print(f"Error in super.from_config: {e}")
            # If it fails, we can construct the Lambda layer manually
            func = lambda x: tf.keras.applications.resnet50.preprocess_input(x * 255.0)
            return cls(func, name=name)

try:
    print("Loading with tf_keras and SafeLambda...")
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
    print(f"Failed loading: {e}")
    import traceback
    traceback.print_exc()
