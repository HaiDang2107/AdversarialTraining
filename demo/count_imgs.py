import os

def count_images(target_dir):
    if not os.path.exists(target_dir):
        print(f"Thư mục '{target_dir}' không tồn tại.")
        return

    print(f"Đang đếm số lượng ảnh trong thư mục: {target_dir}")
    print("=" * 50)
    
    total_images = 0
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
    
    # Duyệt qua các thư mục con (các lớp học/categories)
    subdirs = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])
    
    if not subdirs:
        # Nếu không có thư mục con, đếm trực tiếp trong thư mục chỉ định
        files = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f)) and f.lower().endswith(image_extensions)]
        print(f"Số lượng ảnh trực tiếp: {len(files)}")
        return
        
    for subdir in subdirs:
        subdir_path = os.path.join(target_dir, subdir)
        files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f)) and f.lower().endswith(image_extensions)]
        count = len(files)
        total_images += count
        print(f" - Lớp '{subdir}': {count} ảnh")
        
    print("=" * 50)
    print(f"Tổng cộng: {total_images} ảnh")

if __name__ == "__main__":
    # Đường dẫn tương đối từ vị trí script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_folder = os.path.join(script_dir, "flower_photos")
    count_images(target_folder)
