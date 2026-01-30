"""
BeaconX-ML: Unified Disaster Prediction API

A comprehensive machine learning API for predicting and classifying
earthquakes and cyclones using state-of-the-art ML models.

Author: Your Name
Version: 1.0.0
"""
import logging
from datetime import datetime, timezone
from flask import Flask, jsonify
from flask_cors import CORS

# Blueprints that define the API routes
from cyclone.routes import cyclone_bp
from combined.routes import speed_bp, severity_bp, predict_earthquake_bp

# App configuration (host, port, log level, etc.)
from config import config

# Basic logging so we can see what's happening
logging.basicConfig(
    level=getattr(logging, config.api.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app() -> Flask:
    """
    Build and configure the Flask app.
    
    Returns:
        A ready-to-run Flask application
    """
    app = Flask(__name__)
    CORS(app)
    
    # Wire up route groups with their URL prefixes
    app.register_blueprint(speed_bp, url_prefix='/api/v1/combined')
    app.register_blueprint(severity_bp, url_prefix='/api/v1/combined')
    app.register_blueprint(predict_earthquake_bp, url_prefix='/api/v1/combined')
    app.register_blueprint(cyclone_bp, url_prefix='/api/v1/cyclone')
    
    @app.route('/')
    def home():
        """Friendly landing page with API info."""
        return jsonify({
            "message": "BeaconX-ML: Unified Disaster Prediction API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "earthquake": "/api/v1/combined/predict",
                "cyclone_path": "/api/v1/cyclone/predict-path",
                "cyclone_speed": "/api/v1/combined/predict-speed",
                "severity": "/api/v1/combined/classify-severity"
            }
        })
    
    @app.route('/health')
    def health_check():
        """Simple health check for uptime monitors."""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    return app

# Create the app instance once
app = create_app()

if __name__ == '__main__':
    logger.info(f"Starting BeaconX-ML API on {config.api.host}:{config.api.port}")
    app.run(
        host=config.api.host,
        port=config.api.port,
        debug=config.api.debug
    )
