from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import uuid
import cv2
from plate_recognition_new import detect_plate, crop_plate, run_ocr
from db_vehicle import init_db, save_vehicle, get_plate_by_id
import os

app = FastAPI()
init_db()

# Hàm bật webcam, nhận diện biển số, trả text
import time
import os


def capture_plate_text(timeout=5):
    cam = cv2.VideoCapture(0)
    start_time = time.time()
    plate_text = None

    while time.time() - start_time < timeout:
        ret, frame = cam.read()
        if not ret:
            continue

        temp_path = "webcam_capture.jpg"
        cv2.imwrite(temp_path, frame)

        try:
            path, x, y, w, h = detect_plate(temp_path)
            if path:
                crop_path = crop_plate(path, x, y, w, h)
                plate_text = run_ocr(crop_path)
                if plate_text:
                    break
        except:
            continue
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    cam.release()
    return plate_text

# Xe vào: tạo ID, lưu biển số
@app.post("/vehicle/in")
def vehicle_in():
    plate_text = capture_plate_text()
    if not plate_text:
        return JSONResponse({"status": "ERROR", "reason": "Không nhận diện được biển số"}, status_code=400)

    id = uuid.uuid4().hex[:8]  # ID ngắn gọn
    save_vehicle(id, plate_text)
    return {"id": id, "plate_text": plate_text, "status": "SAVED"}

# Xe ra: nhập ID, check khớp biển số
@app.post("/vehicle/out")
def vehicle_out(id: str = Query(..., min_length=3)):
    expected_plate = get_plate_by_id(id)
    if not expected_plate:
        return JSONResponse({
            "status": "ERROR",
            "reason": "ID not found"
        }, status_code=404)

    current_plate = capture_plate_text()
    if not current_plate:
        return JSONResponse({
            "status": "ERROR",
            "reason": "Không đọc được biển số"
        }, status_code=400)

    # So sánh sau khi hiển thị cả 2
    if current_plate.replace(" ", "") == expected_plate.replace(" ", ""):
        return {
            "status": "SUCCESS",
            "expected_plate": expected_plate,
            "current_plate": current_plate
        }
    else:
        return {
            "status": "ERROR",
            "reason": "Mismatch",
            "expected_plate": expected_plate,
            "current_plate": current_plate
        }


# (Tuỳ chọn) Xem biển số từ ID
@app.get("/vehicle/{id}")
def get_vehicle(id: str):
    plate = get_plate_by_id(id)
    if not plate:
        return JSONResponse({"status": "ERROR", "reason": "ID not found"}, status_code=404)
    return {"plate_text": plate}
