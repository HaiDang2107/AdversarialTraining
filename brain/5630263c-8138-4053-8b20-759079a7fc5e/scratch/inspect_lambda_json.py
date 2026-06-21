import zipfile
import json
import os

model_path = "/home/haidang/Desktop/AdversarialTraining/demo/robust_flower_model_resnet50_PGD_200e.keras"

with zipfile.ZipFile(model_path, 'r') as zip_ref:
    config_data = zip_ref.read('config.json').decode('utf-8')
    config = json.loads(config_data)
    for layer in config['config']['layers']:
        if layer.get('class_name') == 'Lambda':
            print(json.dumps(layer, indent=2))
