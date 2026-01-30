"""
Model loader that keeps everything in one place.
"""
import logging
import joblib
import torch
from typing import Dict, Any
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

from .autoencoder import Autoencoder
from config import config

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads and stores all ML models in one spot.
    """
    
    def __init__(self):
        """Create the loader and pull in all models."""
        self.models: Dict[str, Any] = {}
        self._load_all_models()
    
    def _load_all_models(self) -> None:
        """Load every model we need at startup."""
        try:
            self._load_earthquake_models()
            self._load_cyclone_models()
            self._load_severity_models()
            logger.info("All models loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _load_earthquake_models(self) -> None:
        """Load models used for earthquake severity."""
        try:
            # Scaler for earthquake inputs
            self.models['earthquake_scaler'] = joblib.load(config.model.scaler_path)
            
            # Autoencoder for anomaly scoring
            autoencoder = Autoencoder(
                input_dim=config.parameters.autoencoder_input_dim,
                hidden_dim=config.parameters.autoencoder_hidden_dim,
                latent_dim=config.parameters.autoencoder_latent_dim
            )
            autoencoder.load_state_dict(torch.load(config.model.autoencoder_path))
            autoencoder.eval()
            self.models['autoencoder'] = autoencoder
            
            # KNN clustering model
            self.models['knn_model'] = joblib.load(config.model.knn_model_path)
            
            logger.info("Earthquake models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading earthquake models: {e}")
            raise
    
    def _load_cyclone_models(self) -> None:
        """Load models used for cyclone predictions."""
        try:
            # LSTM path model
            custom_objects = {"mse": MeanSquaredError()}
            lstm_model = load_model(config.model.lstm_model_path, custom_objects=custom_objects)
            lstm_model.compile(optimizer='adam', loss='mse')
            self.models['lstm_model'] = lstm_model
            
            # XGBoost speed/direction models
            self.models['speed_model'] = joblib.load(config.model.speed_model_path)
            self.models['dir_model'] = joblib.load(config.model.dir_model_path)
            
            # Feature scalers
            self.models['scaler_X'] = joblib.load(config.model.scaler_x_path)
            self.models['scaler_y'] = joblib.load(config.model.scaler_y_path)
            
            logger.info("Cyclone models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading cyclone models: {e}")
            raise
    
    def _load_severity_models(self) -> None:
        """Load models for cyclone severity clustering."""
        try:
            # Encoder + scaler + clustering model
            self.models['severity_encoder'] = load_model(config.model.encoder_path)
            self.models['severity_scaler'] = joblib.load(config.model.severity_scaler_path)
            self.models['severity_kmeans'] = joblib.load(config.model.kmeans_path)
            
            logger.info("Severity models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading severity models: {e}")
            raise
    
    def get_model(self, model_name: str) -> Any:
        """
        Fetch a model by name (or raise if missing).
        """
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        return self.models[model_name]
    
    def get_available_models(self) -> list:
        """
        List available model keys.
        """
        return list(self.models.keys())


# Global model loader used across the app
model_loader = ModelLoader()
