"""
Input validation utilities for BeaconX-ML.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class InputValidator:
    """
    Input validation utilities for API endpoints.
    """
    
    @staticmethod
    def validate_earthquake_input(data: Dict[str, Any]) -> List[str]:
        """
        Validate earthquake prediction input.
        
        Args:
            data: Input data dictionary
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        required_fields = ['magnitude', 'depth', 'latitude', 'longitude']
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return errors
        
        # Validate magnitude
        if not isinstance(data['magnitude'], (int, float)):
            errors.append("Magnitude must be a number")
        elif not (0 <= data['magnitude'] <= 10):
            errors.append("Magnitude must be between 0 and 10")
        
        # Validate depth
        if not isinstance(data['depth'], (int, float)):
            errors.append("Depth must be a number")
        elif not (0 <= data['depth'] <= 700):
            errors.append("Depth must be between 0 and 700 kilometers")
        
        # Validate coordinates
        if not isinstance(data['latitude'], (int, float)):
            errors.append("Latitude must be a number")
        elif not (-90 <= data['latitude'] <= 90):
            errors.append("Latitude must be between -90 and 90")
        
        if not isinstance(data['longitude'], (int, float)):
            errors.append("Longitude must be a number")
        elif not (-180 <= data['longitude'] <= 180):
            errors.append("Longitude must be between -180 and 180")
        
        return errors
    
    @staticmethod
    def validate_cyclone_input(data: Dict[str, Any]) -> List[str]:
        """
        Validate cyclone prediction input.
        
        Args:
            data: Input data dictionary
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        required_fields = ['ISO_TIME', 'LAT', 'LON', 'STORM_SPEED', 'STORM_DIR']
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return errors
        
        # Validate coordinates
        if not isinstance(data['LAT'], (int, float)):
            errors.append("LAT must be a number")
        elif not (-90 <= data['LAT'] <= 90):
            errors.append("LAT must be between -90 and 90")
        
        if not isinstance(data['LON'], (int, float)):
            errors.append("LON must be a number")
        elif not (-180 <= data['LON'] <= 180):
            errors.append("LON must be between -180 and 180")
        
        # Validate storm speed
        if not isinstance(data['STORM_SPEED'], (int, float)):
            errors.append("STORM_SPEED must be a number")
        elif not (0 <= data['STORM_SPEED'] <= 300):
            errors.append("STORM_SPEED must be between 0 and 300 km/h")
        
        # Validate storm direction
        if not isinstance(data['STORM_DIR'], (int, float)):
            errors.append("STORM_DIR must be a number")
        elif not (0 <= data['STORM_DIR'] <= 360):
            errors.append("STORM_DIR must be between 0 and 360 degrees")
        
        # Validate timestamp
        try:
            if isinstance(data['ISO_TIME'], str):
                datetime.fromisoformat(data['ISO_TIME'].replace('Z', '+00:00'))
        except ValueError:
            errors.append("ISO_TIME must be a valid ISO timestamp")
        
        return errors
    
    @staticmethod
    def validate_severity_input(data: Dict[str, Any]) -> List[str]:
        """
        Validate severity classification input.
        
        Args:
            data: Input data dictionary
            
        Returns:
            List of validation errors (empty if valid)
        """
        # Use same validation as cyclone input
        return InputValidator.validate_cyclone_input(data)
