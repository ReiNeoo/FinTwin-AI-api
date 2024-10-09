from flask import Flask, jsonify, request
import pickle
import numpy as np


class consumer_classifier_api:
    def __init__(self, app, model_path, data_field_types) -> None:
        self.app = app
        self.model_path = model_path
        self.data_field_types = data_field_types

        self.model = self._get_model(self.model_path)
        self._setup_routes()

    def _get_model(self, model_path):
        with open(model_path, "rb") as file:
            loaded_model = pickle.load(file)
        return loaded_model

    def _setup_routes(self):
        self.app.add_url_rule(
            "/api/data",
            "predict_purchasing_power_class",
            self.predict_purchasing_power_class,
            methods=["POST"],
        )

    def predict_purchasing_power_class(self):
        request_data = request.json

        for field, _ in self.data_field_types.items():
            if field not in request_data:
                return jsonify({"error": f"missing {field} field in data"}), 400
        
        check_same = lambda data_field, data : self.data_field_types.get(data_field) == type(data)
        for data_field, data in request_data.items():
            if not check_same(data_field, data):
                return jsonify({"error" : f"{data_field} is in wrong data type. must be {self.data_field_types.get(data_field)} type"}), 400
                        
        request_data_values = np.array(list(request_data.values())).reshape(1,-1)    
        predicted_class = self.model.predict(request_data_values)
        predicted_class = int(predicted_class[0])

        return jsonify({"PredictedClass" : predicted_class}), 201