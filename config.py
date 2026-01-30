"""
Simple config helpers for the BeaconX-ML app.
"""
import os
from typing import Dict, Any
from dataclasses import dataclass


BASE_DIR = os.path.dirname(__file__)


@dataclass
class ModelConfig:
    """Paths and filenames for ML models."""
    autoencoder_path: str = os.path.join(BASE_DIR, "combined/autoencoder2.pth")
    lstm_model_path: str = os.path.join(BASE_DIR, "combined/cyclone_lstm_model.h5")
    scaler_path: str = os.path.join(BASE_DIR, "combined/scaler.pkl")
    speed_model_path: str = os.path.join(BASE_DIR, "combined/speed_model.pkl")
    dir_model_path: str = os.path.join(BASE_DIR, "combined/dir_model.pkl")
    scaler_x_path: str = os.path.join(BASE_DIR, "combined/scaler_X.pkl")
    scaler_y_path: str = os.path.join(BASE_DIR, "combined/scaler_y.pkl")
    encoder_path: str = os.path.join(BASE_DIR, "combined/severity_encoder.h5")
    severity_scaler_path: str = os.path.join(BASE_DIR, "combined/severity_scaler.pkl")
    kmeans_path: str = os.path.join(BASE_DIR, "combined/severity_kmeans.pkl")
    knn_model_path: str = os.path.join(BASE_DIR, "combined/knn_model.pkl")


@dataclass
class APIConfig:
    """API settings like host, port, and logging."""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    log_level: str = "INFO"


@dataclass
class ModelParameters:
    """Knobs for model sizes and thresholds."""
    autoencoder_input_dim: int = 4
    autoencoder_hidden_dim: int = 16
    autoencoder_latent_dim: int = 64
    reconstruction_threshold: float = 0.01


class Config:
    """Main config object pulled together in one place."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.api = APIConfig()
        self.parameters = ModelParameters()
        
        # Let env vars override defaults when provided
        self._load_from_env()
    
    def _load_from_env(self):
        """Read any overrides from environment variables."""
        self.api.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.api.host = os.getenv("HOST", self.api.host)
        self.api.port = int(os.getenv("PORT", self.api.port))
        self.api.log_level = os.getenv("LOG_LEVEL", self.api.log_level)


# Global config instance used across the app
config = Config()
