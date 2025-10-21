"""
Cyclone prediction routes for BeaconX-ML API.
"""
import logging
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify
from typing import Dict, Any

from models.model_loader import model_loader
from utils.data_processing import DataProcessor
from utils.validators import InputValidator
from schemas import CyclonePredictionRequest, CyclonePredictionResponse

logger = logging.getLogger(__name__)

# Create blueprint
cyclone_bp = Blueprint('cyclone', __name__)

@cyclone_bp.route('/predict-path', methods=['POST'])
def predict_cyclone_path():
    """
    Predict cyclone path using LSTM model.
    
    Expected JSON payload:
    {
        "ISO_TIME": "2024-01-01T00:00:00Z",
        "LAT": 25.0,
        "LON": 80.0,
        "STORM_SPEED": 50.0,
        "STORM_DIR": 180.0
    }
    
    Returns:
        JSON response with predicted latitude and longitude
    """
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate input
        validation_errors = InputValidator.validate_cyclone_input(data)
        if validation_errors:
            return jsonify({
                "success": False,
                "error": f"Validation failed: {'; '.join(validation_errors)}"
            }), 400
        
        # Preprocess data
        X = DataProcessor.preprocess_cyclone_data(data, task='path')
        
        # Get models
        lstm_model = model_loader.get_model('lstm_model')
        scaler_X = model_loader.get_model('scaler_X')
        scaler_y = model_loader.get_model('scaler_y')
        
        # Scale features
        X_scaled = scaler_X.transform(X)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        # Make prediction
        y_pred_scaled = lstm_model.predict(X_reshaped)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        # Prepare response
        response_data = {
            "Predicted_LAT": round(float(y_pred[0][0]), 4),
            "Predicted_LON": round(float(y_pred[0][1]), 4)
        }
        
        logger.info(f"Cyclone path prediction successful: {response_data}")
        return jsonify({
            "success": True,
            "data": response_data
        }), 200
        
    except ValueError as ve:
        logger.error(f"Validation error in cyclone prediction: {ve}")
        return jsonify({
            "success": False,
            "error": str(ve)
        }), 400
        
    except Exception as e:
        logger.error(f"Error in cyclone path prediction: {e}")
        return jsonify({
            "success": False,
            "error": f"Internal Server Error: {str(e)}"
        }), 500
