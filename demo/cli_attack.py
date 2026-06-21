import os
import argparse
import numpy as np
from PIL import Image

# Đảm bảo sử dụng Legacy Keras tương thích với dự án của bạn
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Import các hàm từ file có sẵn của bạn
from models import load_flower_model, predict_image, labels
from attacks import fgsm_attack, pgd_attack

# Định nghĩa hàm tiền xử lý tương tự như trong app.py để đảm bảo tính chính xác
def get_preprocess_fn(arch, model_type):
    if arch == "MobileNetV2":
        if model_type == "Non-Robust Model":
            return lambda x: x * 2.0 - 1.0
        else:
            return None
    else:  # ResNet50
        if model_type == "Non-Robust Model":
            import tensorflow as tf
            def resnet_preprocess(x):
                x_scaled = x * 255.0
                r = x_scaled[..., 0]
                g = x_scaled[..., 1]
                b = x_scaled[..., 2]
                b_pre = b - 103.939
                g_pre = g - 116.779
                r_pre = r - 123.68
                if isinstance(x, tf.Tensor):
                    return tf.stack([b_pre, g_pre, r_pre], axis=-1)
                else:
                    return np.stack([b_pre, g_pre, r_pre], axis=-1)
            return resnet_preprocess
        else:
            return None

def get_model_input(img, arch, model_type):
    fn = get_preprocess_fn(arch, model_type)
    if fn is not None:
        return fn(img)
    return img

def preprocess_image_cli(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    preprocessed = np.expand_dims(img_array, axis=0)
    return preprocessed

def main():
    parser = argparse.ArgumentParser(description="CLI Tool đánh giá Adversarial Attacks trên Flower Models")
    
    # Định nghĩa các tham số nhập từ dòng lệnh
    parser.add_argument('--image', type=str, required=True, help="Đường dẫn tới file ảnh hoa (JPG/PNG)")
    parser.add_argument('--arch', type=str, choices=["MobileNetV2", "ResNet50"], default="MobileNetV2", help="Kiến trúc mô hình")
    parser.add_argument('--attack_model_file', type=str, required=True, help="Tên file mô hình dùng để tạo nhiễu (VD: my_flower_model_mobilenetv2.keras)")
    parser.add_argument('--attack_model_type', type=str, choices=["Non-Robust Model", "FGSM-Robust Model", "PGD-Robust Model"], required=True, help="Loại mô hình tạo nhiễu")
    parser.add_argument('--predict_model_file', type=str, required=True, help="Tên file mô hình dùng để dự đoán (VD: robust_flower_model_mobilenetv2_PGD_200e.keras)")
    parser.add_argument('--predict_model_type', type=str, choices=["Non-Robust Model", "FGSM-Robust Model", "PGD-Robust Model"], required=True, help="Loại mô hình dự đoán")
    parser.add_argument('--method', type=str, choices=["FGSM", "PGD"], required=True, help="Phương thức tấn công (FGSM hoặc PGD)")
    parser.add_argument('--epsilon', type=float, default=0.01, help="Biên độ nhiễu Epsilon (mặc định: 0.01)")
    
    args = parser.parse_args()

    # 1. Tải hai mô hình dựa trên tham số truyền vào
    print("\n⏳ Đang tải các mô hình...")
    try:
        model_attack = load_flower_model(args.attack_model_file)
        model_predict = load_flower_model(args.predict_model_file)
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {e}")
        return

    # 2. Tiền xử lý ảnh gốc
    if not os.path.exists(args.image):
        print(f"❌ Không tìm thấy file ảnh tại đường dẫn: {args.image}")
        return
    orig_img = preprocess_image_cli(args.image)

    # 3. Dự đoán trên ảnh gốc bằng mô hình tấn công để lấy nhãn mục tiêu (giống logic app.py)
    input_for_attack_model_orig = get_model_input(orig_img, args.arch, args.attack_model_type)
    lbl_orig, conf_orig, _ = predict_image(model_attack, input_for_attack_model_orig)
    label_idx = labels.index(lbl_orig)
    
    print("-" * 50)
    print(f"📸 Ảnh gốc (Mô hình tấn công nhận diện): {lbl_orig.upper()} ({conf_orig*100:.2f}%)")
    print("-" * 50)

    # 4. Thực hiện sinh ảnh adversarial dựa trên phương thức được chọn
    print(f"⚡ Đang tiến hành tạo ảnh nhiễu bằng phương thức {args.method} (Epsilon={args.epsilon})...")
    target_preprocess_fn = get_preprocess_fn(args.arch, args.attack_model_type)
    
    if args.method == "FGSM":
        adv_img = fgsm_attack(model_attack, orig_img, label_idx, args.epsilon, target_preprocess_fn)
    elif args.method == "PGD":
        # Thiết lập mặc định max_iter=10 và alpha tự động tính toán giống app.py
        max_iter = 10
        alpha = 2.5 * args.epsilon / max_iter
        adv_img = pgd_attack(model_attack, orig_img, label_idx, args.epsilon, max_iter, alpha, target_preprocess_fn)

    # 5. Đưa ảnh nhiễu vào mô hình dự đoán (Predict Model) và in kết quả
    input_for_predict_model_adv = get_model_input(adv_img, args.arch, args.predict_model_type)
    lbl_adv, conf_adv, _ = predict_image(model_predict, input_for_predict_model_adv)

    print("\n" + "=" * 50)
    print("🎯 KẾT QUẢ DỰ ĐOÁN SAU KHI TẤN CÔNG:")
    print(f"▪️ Mô hình dự đoán: {args.predict_model_file}")
    print(f"▪️ Nhãn dự đoán mới: {lbl_adv.upper()}")
    print(f"▪️ Độ tự tin (Confidence): {conf_adv * 100:.2f}%")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()