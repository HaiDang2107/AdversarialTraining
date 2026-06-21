#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate_origin_model.py

Đánh giá mô hình gốc (sạch) trên tập dữ liệu kiểm thử.
Hỗ trợ cả mô hình ResNet50 và MobileNetV2 bằng cách tự động phát hiện tiền xử lý (preprocessing) phù hợp.
In ra báo cáo đánh giá với tên nhãn chính xác (daisy, dandelion, roses, sunflowers, tulips).
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# Cấu hình tăng trưởng bộ nhớ cho GPU nếu có để tránh lỗi OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Không thể cấu hình GPU Memory Growth: {e}")

# Thiết lập biến môi trường tương thích với legacy Keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"

try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

import tensorflow_hub as hub

# Thêm thư mục chứa script vào python path để giải quyết các import cục bộ (models.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def load_model_by_name_or_path(model_input):
    """
    Tải mô hình phân loại hoa bằng tên hoặc đường dẫn đầy đủ.
    """
    # Import the safe loader from models.py
    try:
        from models import load_keras_model_safely
    except ImportError:
        load_keras_model_safely = None

    # 1. Thử xem có phải đường dẫn file tồn tại trực tiếp không
    if os.path.exists(model_input):
        print(f"Đang tải mô hình từ đường dẫn trực tiếp: {model_input}...")
        if load_keras_model_safely:
            return load_keras_model_safely(model_input)
        return keras.models.load_model(
            model_input,
            custom_objects={'KerasLayer': hub.KerasLayer},
            safe_mode=False
        )
    
    # 2. Thử tải thông qua models.py
    try:
        from models import load_flower_model
        print(f"Đang tải mô hình bằng load_flower_model: {model_input}...")
        return load_flower_model(model_input)
    except Exception:
        pass
        
    # 3. Thử tìm trong thư mục chứa file script
    model_path = os.path.join(current_dir, model_input)
    if os.path.exists(model_path):
        print(f"Đang tải mô hình từ thư mục của script: {model_path}...")
        if load_keras_model_safely:
            return load_keras_model_safely(model_path)
        return keras.models.load_model(
            model_path,
            custom_objects={'KerasLayer': hub.KerasLayer},
            safe_mode=False
        )
        
    raise FileNotFoundError(f"Không tìm thấy file mô hình '{model_input}' trong thư mục hiện tại hoặc thư mục demo/.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình gốc (sạch) trên tập dữ liệu kiểm thử.")
    
    parser.add_argument(
        "--model",
        type=str,
        default="my_flower_model_resnet50.keras",
        help="Tên mô hình hoặc đường dẫn đầy đủ tới file mô hình Keras."
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=os.path.join(current_dir, "flower_photos_test"),
        help="Đường dẫn đến thư mục ảnh kiểm thử."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Kích thước batch khi thực hiện dự báo."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Kiểm tra thư mục dữ liệu kiểm thử
    if not os.path.exists(args.test_dir):
        print(f"Lỗi: Không tìm thấy thư mục kiểm thử tại '{args.test_dir}'")
        sys.exit(1)
        
    # Tải mô hình
    try:
        model = load_model_by_name_or_path(args.model)
    except Exception as e:
        print(f"Lỗi tải mô hình: {e}")
        sys.exit(1)
        
    # Tự động xác định hàm tiền xử lý (preprocessing)
    model_name_lower = os.path.basename(args.model).lower()
    if "resnet" in model_name_lower:
        print("Mô hình ResNet50 được phát hiện. Sử dụng hàm tiền xử lý resnet50.preprocess_input...")
        from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
        test_datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=resnet_preprocess
        )
    else:
        print("Mô hình không phải ResNet50. Sử dụng tỉ lệ tiền xử lý 1/255...")
        test_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0/255.0
        )
        
    # Tạo generator cho dữ liệu kiểm thử
    print(f"Đang đọc dữ liệu từ: {args.test_dir}")
    try:
        test_generator = test_datagen.flow_from_directory(
            args.test_dir,
            target_size=(224, 224),
            batch_size=args.batch_size,
            shuffle=False
        )
    except Exception as e:
        print(f"Lỗi đọc tập dữ liệu: {e}")
        sys.exit(1)
        
    # Chạy dự báo
    print("Đang tiến hành dự báo trên tập kiểm thử...")
    predictions = model.predict(test_generator, verbose=1)
    
    # Lấy các nhãn dự đoán và nhãn thực tế
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_names = list(test_generator.class_indices.keys())
    
    # In ra báo cáo phân loại theo đúng định dạng được yêu cầu
    print("\nCLEAN DATA CLASSIFICATION REPORT:")
    report = classification_report(true_classes, predicted_classes, target_names=class_names, digits=2)
    print(report)

if __name__ == "__main__":
    main()
