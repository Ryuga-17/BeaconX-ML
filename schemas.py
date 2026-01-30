"""
Pydantic schemas that keep our API inputs and outputs tidy.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Union
from datetime import datetime


class EarthquakePredictionRequest(BaseModel):
    """
    Shape and rules for an earthquake prediction request.
    """
    magnitude: float = Field(..., ge=0.0, le=10.0, description="Earthquake magnitude (0-10)")
    depth: float = Field(..., ge=0.0, le=700.0, description="Earthquake depth in kilometers (0-700)")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude (-90 to 90)")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude (-180 to 180)")
    
    @validator('magnitude')
    def validate_magnitude(cls, v):
        if v < 0 or v > 10:
            raise ValueError('Magnitude must be between 0 and 10')
        return v
    
    @validator('depth')
    def validate_depth(cls, v):
        if v < 0 or v > 700:
            raise ValueError('Depth must be between 0 and 700 kilometers')
        return v


class CyclonePredictionRequest(BaseModel):
    """
    Shape and rules for a cyclone path request.
    """
    ISO_TIME: Union[str, datetime] = Field(..., description="ISO timestamp")
    LAT: float = Field(..., ge=-90.0, le=90.0, description="Latitude (-90 to 90)")
    LON: float = Field(..., ge=-180.0, le=180.0, description="Longitude (-180 to 180)")
    STORM_SPEED: float = Field(..., ge=0.0, le=300.0, description="Storm speed in km/h (0-300)")
    STORM_DIR: float = Field(..., ge=0.0, le=360.0, description="Storm direction in degrees (0-360)")
    
    @validator('ISO_TIME')
    def validate_iso_time(cls, v):
        if isinstance(v, str):
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError('Invalid ISO timestamp format')
        return v


class SeverityClassificationRequest(BaseModel):
    """
    Shape and rules for cyclone severity classification.
    """
    ISO_TIME: Union[str, datetime] = Field(..., description="ISO timestamp")
    LAT: float = Field(..., ge=-90.0, le=90.0, description="Latitude (-90 to 90)")
    LON: float = Field(..., ge=-180.0, le=180.0, description="Longitude (-180 to 180)")
    STORM_SPEED: float = Field(..., ge=0.0, le=300.0, description="Storm speed in km/h (0-300)")
    STORM_DIR: float = Field(..., ge=0.0, le=360.0, description="Storm direction in degrees (0-360)")


class PredictionResponse(BaseModel):
    """
    Standard response wrapper for all predictions.
    """
    success: bool = Field(..., description="Whether the prediction was successful")
    data: Optional[dict] = Field(None, description="Prediction results")
    error: Optional[str] = Field(None, description="Error message if prediction failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class EarthquakePredictionResponse(PredictionResponse):
    """
    Response body for earthquake predictions.
    """
    data: Optional[dict] = Field(None, description="Contains 'severity' field with prediction result")


class CyclonePredictionResponse(PredictionResponse):
    """
    Response body for cyclone path predictions.
    """
    data: Optional[dict] = Field(None, description="Contains 'Predicted_LAT' and 'Predicted_LON' fields")


class SpeedPredictionResponse(PredictionResponse):
    """
    Response body for speed/direction predictions.
    """
    data: Optional[dict] = Field(None, description="Contains 'predicted_speed' and 'predicted_direction' fields")


class SeverityResponse(PredictionResponse):
    """
    Response body for severity classifications.
    """
    data: Optional[dict] = Field(None, description="Contains 'severity' field with classification result")
