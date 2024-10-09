from consumer_classifier_api import consumer_classifier_api
from OCR_api import fetch_receipt_data
from flask import Flask

app = Flask(__name__)
model_path = "C:\Python_Projects\FinTwin_project\models\k_means_model.pkl"

data_field_types = {
            "EducationCode" : int,
            "IndustryCode" : int,
            "StatusCode" : int,
            "HomeownershipStatus" : int,
            "TypeOfResidence" : int,
            "NumberOfDependents" : int,
            "LAT" : float,
            "LONG" : float,
        }

api = consumer_classifier_api(app, model_path, data_field_types)

detection_model = "C:\Python_Projects\FinTwin_project\models\\train7\weights\\best.pt"

api = fetch_receipt_data(detection_model, app)

if __name__ == "__main__":
    app.run(debug= True)