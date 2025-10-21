"""
Model loading utilities for BeaconX-ML.
Handles loading and initialization of all ML models.
"""
import logging
import joblib
import torch
from typing import Dict, Any, Optional
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

from .autoencoder import Autoencoder
from config import config

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Centralized model loading and management.
    
    This class handles loading all ML models used in the application,
    providing a single point of access and error handling.
    """
    
    def __init__(self):
        """Initialize the model loader."""
        self.models: Dict[str, Any] = {}
        self._load_all_models()
    
    def _load_all_models(self) -> None:
        """Load all required models."""
        try:
            self._load_earthquake_models()
            self._load_cyclone_models()
            self._load_severity_models()
            logger.info("All models loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _load_earthquake_models(self) -> None:
        """Load earthquake prediction models."""
        try:
            # Load scaler
            self.models['earthquake_scaler'] = joblib.load(config.model.scaler_path)
            
            # Load autoencoder
            autoencoder = Autoencoder(
                input_dim=config.parameters.autoencoder_input_dim,
                hidden_dim=config.parameters.autoencoder_hidden_dim,
                latent_dim=config.parameters.autoencoder_latent_dim
            )
            autoencoder.load_state_dict(torch.load(config.model.autoencoder_path))
            autoencoder.eval()
            self.models['autoencoder'] = autoencoder
            
            # Load KNN model for clustering
            self.models['knn_model'] = joblib.load(config.model.knn_model_path)
            
            logger.info("Earthquake models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading earthquake models: {e}")
            raise
    
    def _load_cyclone_models(self) -> None:
        """Load cyclone prediction models."""
        try:
            # Load LSTM model
            custom_objects = {"mse": MeanSquaredError()}
            lstm_model = load_model(config.model.lstm_model_path, custom_objects=custom_objects)
            lstm_model.compile(optimizer='adam', loss='mse')
            self.models['lstm_model'] = lstm_model
            
            # Load XGBoost models
            self.models['speed_model'] = joblib.load(config.model.speed_model_path)
            self.models['dir_model'] = joblib.load(config.model.dir_model_path)
            
            # Load scalers
            self.models['scaler_X'] = joblib.load(config.model.scaler_x_path)
            self.models['scaler_y'] = joblib.load(config.model.scaler_y_path)
            
            logger.info("Cyclone models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading cyclone models: {e}")
            raise
    
    def _load_severity_models(self) -> None:
        """Load severity classification models."""
        try:
            # Load severity models
            self.models['severity_encoder'] = load_model(config.model.encoder_path)
            self.models['severity_scaler'] = joblib.load(config.model.severity_scaler_path)
            self.models['severity_kmeans'] = joblib.load(config.model.kmeans_path)
            
            logger.info("Severity models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading severity models: {e}")
            raise
    
    def get_model(self, model_name: str) -> Any:
        """
        Get a loaded model by name.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            The requested model
            
        Raises:
            KeyError: If model name is not found
        """
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        return self.models[model_name]
    
    def get_available_models(self) -> list:
        """
        Get list of available model names.
        
        Returns:
            List of available model names
        """
        return list(self.models.keys())


# Global model loader instance
model_loader = ModelLoader()
