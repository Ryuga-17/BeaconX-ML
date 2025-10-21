"""
API endpoint tests for BeaconX-ML.
"""
import pytest
import json
from app import create_app


@pytest.fixture
def app():
    """Create test app instance."""
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestEarthquakeAPI:
    """Test earthquake prediction endpoints."""
    
    def test_earthquake_prediction_valid_data(self, client):
        """Test earthquake prediction with valid data."""
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
        """Test earthquake prediction with invalid magnitude."""
        data = {
            "magnitude": 15.0,  # Invalid magnitude
            "depth": 10.0,
            "latitude": 25.0,
            "longitude": 80.0
        }
        
        response = client.post('/api/v1/combined/predict',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 400
    
    def test_earthquake_prediction_missing_fields(self, client):
        """Test earthquake prediction with missing fields."""
        data = {
            "magnitude": 5.5,
            "depth": 10.0
            # Missing latitude and longitude
        }
        
        response = client.post('/api/v1/combined/predict',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 400


class TestCycloneAPI:
    """Test cyclone prediction endpoints."""
    
    def test_cyclone_path_prediction_valid_data(self, client):
        """Test cyclone path prediction with valid data."""
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
        """Test cyclone path prediction with invalid coordinates."""
        data = {
            "ISO_TIME": "2024-01-01T00:00:00Z",
            "LAT": 95.0,  # Invalid latitude
            "LON": 80.0,
            "STORM_SPEED": 50.0,
            "STORM_DIR": 180.0
        }
        
        response = client.post('/api/v1/cyclone/predict-path',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 400


class TestHealthEndpoints:
    """Test health and info endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get('/')
        assert response.status_code == 200
        
        result = json.loads(response.data)
        assert 'message' in result
        assert 'version' in result
        assert 'endpoints' in result
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        
        result = json.loads(response.data)
        assert result['status'] == 'healthy'

