import cv2
from ultralytics import YOLO
import pytesseract

# Pytesseract'ı kurduğun dosya yolu ile değiştir
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

class ocr():
    def __init__(self, detection_model) -> None:
        self.model = YOLO(detection_model)
        
    def get_image_position(self, image) -> list:
        model = self.model
        return_list = []
        results = model.track(image, persist= True)
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            obj_id = box.id[0].cpu().numpy().astype(int) if box.id is not None else -1
            return_list.append((obj_id, x1, y1, x2, y2))
        return return_list

    def _xyxy_to_xywh(self, position: list)-> list:
        x1, y1, x2, y2 = position
        return [x1, y1, x2 - x1, y2 - y1]

    def get_receipt_info(self, image) -> list:
        receip_possitions = self.get_image_position(image)
        extracted_datas = []
        for position in receip_possitions:
            pos = position[1:]
            pos = self._xyxy_to_xywh(pos)
            x, y, w, h = pos
            print(x, y, w, h)

            roi = image[y:y+h, x:x+w]
            text = pytesseract.image_to_string(roi, lang= "tur")
            extracted_datas.append(text)
        return extracted_datas