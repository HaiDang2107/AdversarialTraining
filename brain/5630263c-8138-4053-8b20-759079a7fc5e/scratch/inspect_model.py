import zipfile
import json
import os

model_path = "/home/haidang/Desktop/AdversarialTraining/demo/robust_flower_model_resnet50_PGD_200e.keras"

if not os.path.exists(model_path):
    print("Model file not found!")
    exit(1)

with zipfile.ZipFile(model_path, 'r') as zip_ref:
    print("Files in zip:")
    for file in zip_ref.namelist():
        print(f"  {file}")
    
    if 'config.json' in zip_ref.namelist():
        config_data = zip_ref.read('config.json').decode('utf-8')
        config = json.loads(config_data)
        print("\nModel Config keys:", config.keys())
        # Print the layers to see how Lambda is configured
        if 'config' in config and 'layers' in config['config']:
            for i, layer in enumerate(config['config']['layers']):
                print(f"Layer {i}: {layer.get('class_name')} - {layer.get('config', {}).get('name')}")
                if layer.get('class_name') == 'Lambda':
                    # Print Lambda config details, avoiding printing binary fields
                    print("  Lambda config keys:", layer['config'].keys())
                    print("  Lambda function details:", {k: v for k, v in layer['config'].items() if k != 'function'})
