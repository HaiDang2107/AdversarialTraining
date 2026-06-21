#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate_mobilenetv2_under_attack.py

Đánh giá mô hình phân loại hoa dưới cuộc tấn công FGSM hoặc PGD.
Hỗ trợ cả MobileNetV2 và ResNet50 thông qua tự động phát hiện tiền xử lý tương ứng.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf

# Enable memory growth for GPUs to avoid allocating all memory at once and prevent OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Không thể cấu hình GPU Memory Growth: {e}")

# Check if DISPLAY is in os.environ, otherwise use Agg backend for matplotlib
import matplotlib
if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set environment variable to match legacy Keras context
os.environ["TF_USE_LEGACY_KERAS"] = "1"

try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

import tensorflow_hub as hub

# Add the script's directory to python path to resolve local imports (models.py)
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

def evaluate_robustness(model, dataset, class_names, attack_type="PGD", epsilon=0.01, num_visualize=3, pgd_steps=10, pgd_alpha=0.0025, preprocess_fn=None):
    """
    Hàm đánh giá độ bền vững của mô hình và trực quan hóa kết quả dưới cuộc tấn công FGSM hoặc PGD.
    
    Args:
        model: Mô hình Keras cần đánh giá.
        dataset: Tập dữ liệu (VD: test_ds).
        class_names: Danh sách tên các nhãn (VD: ['daisy', 'dandelion', ...]).
        attack_type: Loại tấn công, 'FGSM' hoặc 'PGD'.
        epsilon: Biên độ nhiễu.
        num_visualize: Số lượng ví dụ muốn hiển thị ra biểu đồ.
        pgd_steps: Số bước lặp của PGD (chỉ dùng cho PGD).
        pgd_alpha: Step size của PGD (chỉ dùng cho PGD).
        preprocess_fn: Hàm tiền xử lý động (chỉ dùng cho ResNet50 gốc).
    """
    clean_acc = tf.keras.metrics.CategoricalAccuracy()
    adv_acc = tf.keras.metrics.CategoricalAccuracy()
    
    total_correct_clean = 0
    successful_attacks = 0
    
    sum_conf_adv_correct = 0.0
    count_adv_correct = 0
    
    # Biến lưu trữ dữ liệu để vẽ biểu đồ
    viz_clean_imgs, viz_adv_imgs, viz_noises = [], [], []
    viz_clean_preds, viz_adv_preds, viz_true_labels = [], [], []

    print(f"Bắt đầu đánh giá mô hình với cuộc tấn công {attack_type.upper()}...")
    
    for x, y in dataset:
        # Nhớ để training=False để Dropout/BatchNorm không hoạt động sai
        x_clean = preprocess_fn(x) if preprocess_fn is not None else x
        clean_preds = model(x_clean, training=False) 
        clean_acc.update_state(y, clean_preds)
        
        if attack_type.upper() == "FGSM":
            with tf.GradientTape() as tape:
                tape.watch(x)
                x_prep = preprocess_fn(x) if preprocess_fn is not None else x
                preds = model(x_prep, training=False)
                loss = tf.keras.losses.CategoricalCrossentropy()(y, preds)
            
            grad = tape.gradient(loss, x)
            signed_grad = tf.sign(grad)
            x_adv = x + epsilon * signed_grad
            x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
        else:  # PGD
            # Khởi tạo ngẫu nhiên (Random Initialization) quanh điểm gốc x
            noise = tf.random.uniform(tf.shape(x), minval=-epsilon, maxval=epsilon)
            x_adv = x + noise
            x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
            
            for _ in range(pgd_steps):
                with tf.GradientTape() as tape:
                    tape.watch(x_adv)
                    x_adv_prep = preprocess_fn(x_adv) if preprocess_fn is not None else x_adv
                    preds = model(x_adv_prep, training=False)
                    loss = tf.keras.losses.CategoricalCrossentropy()(y, preds)
                
                grad = tape.gradient(loss, x_adv)
                signed_grad = tf.sign(grad)
                
                # Tiến 1 bước nhỏ pgd_alpha
                x_adv = x_adv + pgd_alpha * signed_grad
                # Phép chiếu (Projection): Kẹp vào vùng epsilon của ảnh gốc x
                x_adv = tf.clip_by_value(x_adv, x - epsilon, x + epsilon)
                # Kẹp vào không gian màu hợp lệ [0, 1]
                x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
        
        x_adv_final = preprocess_fn(x_adv) if preprocess_fn is not None else x_adv
        adv_preds = model(x_adv_final, training=False)
        adv_acc.update_state(y, adv_preds)
        
        # --- TÍNH TOÁN ATTACK SUCCESS RATE (ASR) ---
        true_labels = tf.argmax(y, axis=1)
        clean_labels = tf.argmax(clean_preds, axis=1)
        adv_labels = tf.argmax(adv_preds, axis=1)
        
        clean_correct = tf.equal(clean_labels, true_labels)
        adv_incorrect = tf.not_equal(adv_labels, true_labels)
        adv_correct = tf.equal(adv_labels, true_labels)
        
        total_correct_clean += tf.reduce_sum(tf.cast(clean_correct, tf.int32)).numpy()
        successful_attacks += tf.reduce_sum(tf.cast(tf.logical_and(clean_correct, adv_incorrect), tf.int32)).numpy()
        
        # Calculate confidence for correct adversarial predictions
        correct_confidences = tf.boolean_mask(tf.reduce_max(adv_preds, axis=1), adv_correct)
        sum_conf_adv_correct += tf.reduce_sum(correct_confidences).numpy()
        count_adv_correct += tf.shape(correct_confidences)[0].numpy()
        
        # --- LƯU LẠI VÍ DỤ ĐỂ VẼ BIỂU ĐỒ ---
        if len(viz_clean_imgs) < num_visualize:
            for i in range(len(x)):
                if len(viz_clean_imgs) >= num_visualize:
                    break
                # Chỉ hiển thị những ảnh mà Tấn công thành công (Lúc đầu đoán đúng, sau bị lừa đoán sai)
                if clean_correct[i] and adv_incorrect[i]:
                    viz_clean_imgs.append(x[i].numpy())
                    viz_adv_imgs.append(x_adv[i].numpy())
                    # viz_noises lưu lại hướng nhiễu (signed_grad)
                    if attack_type.upper() == "FGSM":
                        viz_noises.append(signed_grad[i].numpy())
                    else:
                        # Với PGD, nhiễu thực tế là (x_adv - x) được phóng đại để dễ quan sát
                        diff = x_adv[i].numpy() - x[i].numpy()
                        # Chuẩn hóa về khoảng [-1, 1] để đồng bộ với signed_grad
                        max_diff = np.max(np.abs(diff))
                        viz_noises.append(diff / max_diff if max_diff > 0 else diff)
                    
                    viz_clean_preds.append((clean_labels[i].numpy(), np.max(clean_preds[i].numpy())))
                    viz_adv_preds.append((adv_labels[i].numpy(), np.max(adv_preds[i].numpy())))
                    viz_true_labels.append(true_labels[i].numpy())

    # --- IN KẾT QUẢ TỔNG QUAN ---
    c_acc = clean_acc.result().numpy() * 100
    a_acc = adv_acc.result().numpy() * 100
    asr = (successful_attacks / total_correct_clean * 100) if total_correct_clean > 0 else 0.0
    avg_conf_adv_correct = (sum_conf_adv_correct / count_adv_correct * 100) if count_adv_correct > 0 else 0.0

    print("\n" + "="*50)
    print(f" BÁO CÁO ĐÁNH GIÁ ĐỘ BỀN VỮNG ({attack_type.upper()} ATTACK REPORT)")
    print("="*50)
    print(f" Phương thức tấn công      : {attack_type.upper()}")
    print(f" Biên độ nhiễu (Epsilon)   : {epsilon}")
    if attack_type.upper() == "PGD":
        print(f" Số bước lặp (Steps)       : {pgd_steps}")
        print(f" Bước nhảy (Alpha)         : {pgd_alpha}")
    print(f" 1. Standard Accuracy      : {c_acc:.2f}%  (Độ chính xác trên ảnh sạch)")
    print(f" 2. Robust Accuracy        : {a_acc:.2f}%  (Độ chính xác trên ảnh nhiễu)")
    print(f" 3. Attack Success Rate    : {asr:.2f}%  (Tỷ lệ đánh lừa mô hình thành công)")
    print(f" 4. Avg Conf on Correct Adv: {avg_conf_adv_correct:.2f}%  (Độ tự tin TB trên ảnh nhiễu đoán đúng)")
    print("="*50)

    # --- VẼ BIỂU ĐỒ (DIAGRAMS) ---
    if len(viz_clean_imgs) > 0:
        print(f"\n Hiển thị {len(viz_clean_imgs)} ví dụ bị tấn công thành công:")
        fig = plt.figure(figsize=(12, 4 * len(viz_clean_imgs)))
        
        for i in range(len(viz_clean_imgs)):
            true_name = class_names[viz_true_labels[i]]
            
            clean_name = class_names[viz_clean_preds[i][0]]
            clean_conf = viz_clean_preds[i][1] * 100
            
            adv_name = class_names[viz_adv_preds[i][0]]
            adv_conf = viz_adv_preds[i][1] * 100
            
            # Cột 1: Ảnh gốc
            ax1 = fig.add_subplot(len(viz_clean_imgs), 3, i*3 + 1)
            ax1.imshow(viz_clean_imgs[i])
            ax1.set_title(f"Ảnh Gốc (Thực: {true_name})\nDự đoán: {clean_name} ({clean_conf:.1f}%)", color='green')
            ax1.axis('off')
            
            # Cột 2: Ma trận nhiễu
            ax2 = fig.add_subplot(len(viz_clean_imgs), 3, i*3 + 2)
            # Normalize nhiễu (-1 đến 1) về khoảng (0 đến 1) để có thể hiển thị bằng màu sắc
            noise_visual = (viz_noises[i] + 1.0) / 2.0 
            ax2.imshow(noise_visual)
            ax2.set_title(f"Bản đồ nhiễu {attack_type.upper()}\n(Đã phóng đại cường độ)")
            ax2.axis('off')
            
            # Cột 3: Ảnh đối kháng
            ax3 = fig.add_subplot(len(viz_clean_imgs), 3, i*3 + 3)
            ax3.imshow(viz_adv_imgs[i])
            ax3.set_title(f"Ảnh Đối Kháng (Eps={epsilon})\nDự đoán sai: {adv_name} ({adv_conf:.1f}%)", color='red')
            ax3.axis('off')
            
        plt.tight_layout()
        output_plot_path = os.path.join(current_dir, f"robustness_eval_{attack_type.lower()}.png")
        plt.savefig(output_plot_path, bbox_inches='tight', dpi=150)
        print(f"\n Biểu đồ kết quả đã được lưu tại: {output_plot_path}")
        
        try:
            plt.show()
        except Exception:
            print(" Không thể hiển thị biểu đồ trực tiếp trên màn hình (có thể do kết nối SSH hoặc thiếu GUI).")
    else:
        print("\n Tuyệt vời! Không tìm thấy ví dụ nào bị tấn công thành công trong giới hạn quét.")

def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình phân loại hoa dưới cuộc tấn công FGSM hoặc PGD.")
    
    parser.add_argument(
        "--attack", "-a",
        type=str,
        default="PGD",
        choices=["FGSM", "PGD", "fgsm", "pgd"],
        help="Thuật toán tấn công để đánh giá: FGSM hoặc PGD. (Mặc định: PGD)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="my_flower_model_6e.keras",
        help="Tên file mô hình hoặc đường dẫn file .keras."
    )
    
    parser.add_argument(
        "--test_dir", "-t",
        type=str,
        default=os.path.join(current_dir, "flower_photos_test"),
        help="Đường dẫn đến thư mục tập test. (Mặc định: demo/flower_photos_test)"
    )
    
    parser.add_argument(
        "--epsilon", "-e",
        type=float,
        default=0.01,
        help="Biên độ nhiễu epsilon. (Mặc định: 0.01)"
    )
    
    parser.add_argument(
        "--num_visualize", "-n",
        type=int,
        default=3,
        help="Số lượng ví dụ trực quan hóa. (Mặc định: 3)"
    )
    
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=16,
        help="Batch size dùng khi tải dữ liệu. (Mặc định: 16)"
    )
    
    parser.add_argument(
        "--pgd_steps",
        type=int,
        default=10,
        help="Số bước lặp tối ưu của PGD. (Mặc định: 10)"
    )
    
    parser.add_argument(
        "--pgd_alpha",
        type=float,
        default=0.0025,
        help="Kích thước bước nhảy của PGD. (Mặc định: 0.0025)"
    )
    
    args = parser.parse_args()
    
    # 1. Tải mô hình
    try:
        model = load_model_by_name_or_path(args.model)
    except Exception as e:
        print(f"Lỗi tải mô hình: {e}")
        sys.exit(1)
        
    # 2. Tải tập dữ liệu test
    if not os.path.exists(args.test_dir):
        print(f"Lỗi: Thư mục tập test '{args.test_dir}' không tồn tại.")
        sys.exit(1)
        
    print(f"Đang tải tập dữ liệu test từ: {args.test_dir}")
    try:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            args.test_dir,
            image_size=(224, 224),
            batch_size=args.batch_size,
            label_mode='categorical',
            shuffle=False,
            interpolation='nearest'
        )
    except Exception as e:
        print(f"Lỗi tải tập dữ liệu: {e}")
        sys.exit(1)
        
    class_names = test_ds.class_names
    
    # Chuẩn hóa giá trị pixel ảnh về khoảng [0, 1] trước khi áp dụng tiền xử lý
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Thiết lập hàm tiền xử lý (preprocess_fn) nếu là mô hình ResNet50 hoặc MobileNetV2 gốc/sạch
    preprocess_fn = None
    model_name_lower = os.path.basename(args.model).lower()
    if "resnet50" in model_name_lower and "robust" not in model_name_lower:
        print("Mô hình ResNet50 sạch được phát hiện. Áp dụng tiền xử lý ResNet50 (RGB -> BGR và trừ trung bình ImageNet) trong quá trình tấn công...")
        def resnet_preprocess(x):
            # Quy đổi ảnh từ [0, 1] sang [0, 255]
            x_scaled = x * 255.0
            # Tách các kênh màu RGB
            r = x_scaled[..., 0]
            g = x_scaled[..., 1]
            b = x_scaled[..., 2]
            # Trừ trung bình tập ImageNet (BGR): [103.939, 116.779, 123.68]
            b_pre = b - 103.939
            g_pre = g - 116.779
            r_pre = r - 123.68
            # Gộp lại thành ảnh BGR
            return tf.stack([b_pre, g_pre, r_pre], axis=-1)
        preprocess_fn = resnet_preprocess
    elif "my_flower_model" in model_name_lower and "resnet" not in model_name_lower:
        print("Mô hình MobileNetV2 sạch được phát hiện. Áp dụng tiền xử lý MobileNetV2 (scale về [-1, 1]) trong quá trình tấn công...")
        preprocess_fn = lambda x: x * 2.0 - 1.0
        
    # 3. Chạy đánh giá độ bền vững
    evaluate_robustness(
        model=model,
        dataset=test_ds,
        class_names=class_names,
        attack_type=args.attack,
        epsilon=args.epsilon,
        num_visualize=args.num_visualize,
        pgd_steps=args.pgd_steps,
        pgd_alpha=args.pgd_alpha,
        preprocess_fn=preprocess_fn
    )

if __name__ == "__main__":
    main()