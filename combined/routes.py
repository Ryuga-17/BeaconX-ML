import logging
import torch
from flask import Blueprint, request, jsonify

from config import config
from models.model_loader import model_loader
from utils.data_processing import DataProcessor
from utils.validators import InputValidator

logger = logging.getLogger(__name__)

severity_labels = {0: "Mild", 1: "Moderate", 2: "Severe", 3: "Catastrophic"}

predict_earthquake_bp = Blueprint('earthquake_predict', __name__)

@predict_earthquake_bp.route("/predict", methods=["POST"])
def predict_earthquake():
    try:
        data = request.get_json()
        validation_errors = InputValidator.validate_earthquake_input(data)
        if validation_errors:
            return jsonify({"error": f"Validation failed: {'; '.join(validation_errors)}"}), 400

        features = DataProcessor.preprocess_earthquake_data(data)
        scaler = model_loader.get_model('earthquake_scaler')
        input_scaled = scaler.transform(features)
        input_tensor = torch.FloatTensor(input_scaled)

        with torch.no_grad():
            autoencoder = model_loader.get_model('autoencoder')
            encoded_features = autoencoder.encoder(input_tensor).numpy()
            reconstructed_data = autoencoder(input_tensor)
        reconstruction_error = torch.mean((input_tensor - reconstructed_data) ** 2).item()
        threshold = config.parameters.reconstruction_threshold
        _anomaly = reconstruction_error > threshold
        cluster_model = model_loader.get_model('knn_model')
        cluster_label = cluster_model.predict(encoded_features)[0]
        severity = severity_labels.get(cluster_label, "Unknown")

        return jsonify({"severity": severity})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

speed_bp = Blueprint('speed', __name__)

@speed_bp.route("/predict-speed", methods=["POST"])
def predict_speed():
    try:
        data = request.get_json()
        validation_errors = InputValidator.validate_cyclone_input(data)
        if validation_errors:
            return jsonify({"error": f"Validation failed: {'; '.join(validation_errors)}"}), 400

        X = DataProcessor.preprocess_cyclone_data(data, task='speed_dir')
        speed_model = model_loader.get_model('speed_model')
        dir_model = model_loader.get_model('dir_model')
        speed_pred = speed_model.predict(X)
        dir_pred = dir_model.predict(X)
        return jsonify({
            "predicted_speed": speed_pred.tolist(),
            "predicted_direction": dir_pred.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

severity_bp = Blueprint('severity', __name__)

@severity_bp.route("/classify-severity", methods=["POST"])
def classify_severity():
    try:
        data = request.get_json()
        validation_errors = InputValidator.validate_severity_input(data)
        if validation_errors:
            return jsonify({"error": f"Validation failed: {'; '.join(validation_errors)}"}), 400

        X = DataProcessor.preprocess_severity_data(data)
        scaler_severity = model_loader.get_model('severity_scaler')
        if not hasattr(scaler_severity, 'mean_'):
            raise ValueError("Severity scaler is not fitted.")

        X_scaled = scaler_severity.transform(X)
        encoder = model_loader.get_model('severity_encoder')
        expected_features = encoder.input_shape[1]
        if X_scaled.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {X_scaled.shape[1]}")

        latent_features = encoder.predict(X_scaled)
        kmeans = model_loader.get_model('severity_kmeans')
        cluster_labels = kmeans.predict(latent_features)
        severity = severity_labels.get(cluster_labels[0], "Unknown")
        return jsonify({"severity": severity})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

