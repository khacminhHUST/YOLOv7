import os

# Đường dẫn đến 2 file cần sửa
detect_path = "detect.py"
experimental_path = os.path.join("models", "experimental.py")

# ====== Sửa detect.py (loại bỏ đối số weights_only nếu có) ======
if os.path.exists(detect_path):
    with open(detect_path, "r") as f:
        lines = f.readlines()

    with open(detect_path, "w") as f:
        for line in lines:
            if "attempt_load" in line and "weights_only=" in line:
                # Xóa phần weights_only=False
                line = line.replace(", weights_only=False", "")
            f.write(line)
    print("✅ detect.py đã được sửa.")

# ====== Sửa experimental.py (thêm weights_only=False thủ công nếu chưa có) ======
if os.path.exists(experimental_path):
    with open(experimental_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if "def attempt_load" in line and "weights_only" not in line:
            # Thêm weights_only=False vào định nghĩa hàm
            line = line.rstrip("\n").rstrip(")") + ", weights_only=False):\n"
        elif "torch.load(" in line and "weights_only=" not in line:
            # Thêm đối số weights_only nếu thiếu
            if "#" not in line:  # tránh sửa dòng đã comment
                line = line.replace("torch.load(", "torch.load(").rstrip(")\n") + ", weights_only=weights_only)\n"
        new_lines.append(line)

    with open(experimental_path, "w") as f:
        f.writelines(new_lines)
    print("✅ models/experimental.py đã được sửa.")
