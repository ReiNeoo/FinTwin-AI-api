from flask import Flask, jsonify, request
import numpy as np
import cv2

from OCR import ocr

class fetch_receipt_data(ocr):
    def __init__(self, detection_model, app) -> None:
        super().__init__(detection_model)
        self.app = app

        self._setup_routes()
        
    def _setup_routes(self):
        self.app.add_url_rule(
            "/api/data",
            "fetch_data",
            self.fetch_data,
            methods=["POST"],
        )
    
    def fetch_data(self):
        img_file = request.files['image']
        image_file = img_file.read()
        image_name=img_file.filename
        if '.jpg' in image_name or '.jpeg' in image_name:    
            image_file_bytes = np.fromstring(image_file, np.uint8)
            image = cv2.imdecode(image_file_bytes, cv2.IMREAD_COLOR)
            extracted_texts = self.get_receipt_info(image)
            return jsonify({"response" : extracted_texts})
        
        else:
            return jsonify({"error" : "select an imaege file"})