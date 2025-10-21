"""
Configuration management for BeaconX-ML application.
"""
import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    autoencoder_path: str = "combined/autoencoder2.pth"
    lstm_model_path: str = "combined/cyclone_lstm_model.h5"
    scaler_path: str = "combined/scaler.pkl"
    speed_model_path: str = "combined/speed_model.pkl"
    dir_model_path: str = "combined/dir_model.pkl"
    scaler_x_path: str = "combined/scaler_X.pkl"
    scaler_y_path: str = "combined/scaler_y.pkl"
    encoder_path: str = "combined/severity_encoder.h5"
    severity_scaler_path: str = "combined/severity_scaler.pkl"
    kmeans_path: str = "combined/severity_kmeans.pkl"
    knn_model_path: str = "combined/knn_model.pkl"


@dataclass
class APIConfig:
    """Configuration for API settings."""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    log_level: str = "INFO"


@dataclass
class ModelParameters:
    """Model architecture parameters."""
    autoencoder_input_dim: int = 4
    autoencoder_hidden_dim: int = 16
    autoencoder_latent_dim: int = 64
    reconstruction_threshold: float = 0.01


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.api = APIConfig()
        self.parameters = ModelParameters()
        
        # Override with environment variables if present
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        self.api.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.api.host = os.getenv("HOST", self.api.host)
        self.api.port = int(os.getenv("PORT", self.api.port))
        self.api.log_level = os.getenv("LOG_LEVEL", self.api.log_level)


# Global configuration instance
config = Config()
