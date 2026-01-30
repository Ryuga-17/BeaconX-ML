"""
Lightweight API tests for BeaconX-ML.
"""
import pytest
import json
from app import create_app


@pytest.fixture
def app():
    """Spin up a test app instance."""
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Get a test client for requests."""
    return app.test_client()


class TestEarthquakeAPI:
    """Earthquake prediction endpoint tests."""
    
    def test_earthquake_prediction_valid_data(self, client):
        """Happy path: valid earthquake input."""
        data = {
            "magnitude": 5.5,
            "depth": 10.0,
            "latitude": 25.0,
            "longitude": 80.0
        }
        
        response = client.post('/api/v1/combined/predict', 
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'severity' in result
    
    def test_earthquake_prediction_invalid_magnitude(self, client):
        """Reject invalid magnitude."""
        data = {
            "magnitude": 15.0,  # Out of allowed range
            "depth": 10.0,
            "latitude": 25.0,
            "longitude": 80.0
        }
        
        response = client.post('/api/v1/combined/predict',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 400
    
    def test_earthquake_prediction_missing_fields(self, client):
        """Reject missing required fields."""
        data = {
            "magnitude": 5.5,
            "depth": 10.0
            # Latitude and longitude intentionally omitted
        }
        
        response = client.post('/api/v1/combined/predict',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 400


class TestCycloneAPI:
    """Cyclone prediction endpoint tests."""
    
    def test_cyclone_path_prediction_valid_data(self, client):
        """Happy path: valid cyclone input."""
        data = {
            "ISO_TIME": "2024-01-01T00:00:00Z",
            "LAT": 25.0,
            "LON": 80.0,
            "STORM_SPEED": 50.0,
            "STORM_DIR": 180.0
        }
        
        response = client.post('/api/v1/cyclone/predict-path',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'success' in result
        assert result['success'] is True
    
    def test_cyclone_path_prediction_invalid_coordinates(self, client):
        """Reject invalid latitude values."""
        data = {
            "ISO_TIME": "2024-01-01T00:00:00Z",
            "LAT": 95.0,  # Out of allowed range
            "LON": 80.0,
            "STORM_SPEED": 50.0,
            "STORM_DIR": 180.0
        }
        
        response = client.post('/api/v1/cyclone/predict-path',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 400


class TestHealthEndpoints:
    """Health and info endpoint tests."""
    
    def test_root_endpoint(self, client):
        """Root endpoint returns API info."""
        response = client.get('/')
        assert response.status_code == 200
        
        result = json.loads(response.data)
        assert 'message' in result
        assert 'version' in result
        assert 'endpoints' in result
    
    def test_health_endpoint(self, client):
        """Health endpoint returns healthy status."""
        response = client.get('/health')
        assert response.status_code == 200
        
        result = json.loads(response.data)
        assert result['status'] == 'healthy'

