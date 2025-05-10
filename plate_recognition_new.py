import cv2
import torch
import numpy as np
import os
from pathlib import Path
import easyocr
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from models.experimental import attempt_load

# Cấu hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = "best_new.pt"
img_size = 640

# Load model
model = attempt_load(weights_path, map_location=device)
model.eval().to(device)

# Hàm phát hiện biển số
def detect_plate(image_path):
    img0 = cv2.imread(image_path)
    assert img0 is not None, f"Không tìm thấy ảnh: {image_path}"

    # Preprocess ảnh giống YOLOv7
    img = letterbox(img0, new_shape=img_size, stride=32, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # Dự đoán
    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    if pred[0] is None or len(pred[0]) == 0:
        print("❌ Không phát hiện vật thể nào.")
        return None, None

    # Scale box về ảnh gốc
    det = pred[0]
    det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()

    x1, y1, x2, y2, conf, cls = det[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # Vẽ hộp
    img_result = img0.copy()
    cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    save_path = f"results_detect/{Path(image_path).name}"
    os.makedirs("results_detect", exist_ok=True)
    cv2.imwrite(save_path, img_result)
    print(f"✅ Ảnh đã lưu tại: {save_path}")

    return image_path, x1, y1, x2 - x1, y2 - y1

# Cắt vùng biển số
def crop_plate(image_path, x, y, w, h):
    img = cv2.imread(image_path)
    crop_img = img[y:y + h, x:x + w]
    save_path = f"results_crop/{Path(image_path).name}"
    os.makedirs("results_crop", exist_ok=True)
    cv2.imwrite(save_path, crop_img)
    print(f"📸 Biển số đã cắt và lưu tại: {save_path}")
    return save_path

# Đọc ký tự từ ảnh
def run_ocr(image_path):
    reader = easyocr.Reader(['en'])  # Có thể thêm ['en', 'vi'] nếu cần
    result = reader.readtext(image_path)
    if result:
        plate = ' '.join([res[1] for res in result])
        print(f"🔤 Kết quả OCR: {plate}")
    else:
        print("❌ Không đọc được ký tự.")
        plate = ""
    return plate

# Hàm chính
def main():
    image_path = "bien_3.jpg"  # Ảnh input
    try:
        path, x, y, w, h = detect_plate(image_path)
        if path:
            crop_path = crop_plate(path, x, y, w, h)
            run_ocr(crop_path)
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()
