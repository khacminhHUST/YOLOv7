import torch
import cv2 
import easyocr

cam = cv2.VideoCapture(0)

### load model
from models.experimental import attempt_load
model = attempt_load("best_new.pt", map_location="cpu")
#model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model = "C:\\Users\\Admin\\Desktop\\My project\\yolov7.pt", force_reload=True)
#model = torch.hub.load('WongKinYiu/yolov7', 'yolov7', pretrained=True)

def load_model(imagepath = "bien_3.jpg"):

    img = cv2.imread(imagepath)
    # Inference
    results = model(imagepath)
    results.pandas().xyxy[0]    
    image_draw = img.copy()

    print("===> DETECTION RESULTS:")
    print(results.pandas().xyxy[0])  # In bảng kết quả để debug


    for i in range(len(results.pandas().xyxy[0])):
        x_min, y_min, x_max, y_max, conf, clas = results.xyxy[0][i].numpy()
        width = x_max - x_min
        height = y_max - y_min


        if True:     # nguyên bản là if clas == 0
            image_draw = cv2.rectangle(image_draw, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            path_detect = f'results_detect/{imagepath.split("/")[-1]}'
            #cv2.imwrite(path_detect, image_draw)
    cv2.imshow("Detection Result", image_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return imagepath, int(x_min), int(y_min), int(width), int(height)

### crop object
def crop(imagepath, x, y, w, h): 
    image = cv2.imread(imagepath)
    crop_img = image[y:y+h, x:x+w]
    path_crop = f"results_crop/{imagepath.split('/')[-1]}"
    cv2.imwrite(path_crop, crop_img)
    
    return path_crop

### extract value 
def OCR(path): 
    IMAGE_PATH = path
    reader = easyocr.Reader(['en'])
    result = reader.readtext(IMAGE_PATH)
    plate = ' '.join(detect[1] for detect in result)
    print("EXTRACT: ", plate)
    return

def main(): 
    try:
        path, x, y, w, h = load_model()
        croppath = crop(path, x, y, w, h)
        #OCR(croppath)

    except: 
        print("No detected plate")
        pass

if __name__ == '__main__': 
    main()