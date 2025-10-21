"""
Data processing utilities for BeaconX-ML.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Centralized data processing utilities for all ML models.
    """
    
    @staticmethod
    def preprocess_earthquake_data(data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess earthquake data for prediction.
        
        Args:
            data: Dictionary containing earthquake features
            
        Returns:
            Preprocessed numpy array ready for model input
        """
        try:
            # Extract features in the correct order
            features = np.array([[
                data['magnitude'],
                data['depth'], 
                data['latitude'],
                data['longitude']
            ]])
            
            logger.debug(f"Preprocessed earthquake data: {features.shape}")
            return features
            
        except KeyError as e:
            logger.error(f"Missing required field: {e}")
            raise ValueError(f"Missing required field: {e}")
        except Exception as e:
            logger.error(f"Error preprocessing earthquake data: {e}")
            raise
    
    @staticmethod
    def preprocess_cyclone_data(data: Dict[str, Any], task: str = 'path') -> np.ndarray:
        """
        Preprocess cyclone data for prediction.
        
        Args:
            data: Dictionary containing cyclone features
            task: Type of prediction ('path' or 'speed_dir')
            
        Returns:
            Preprocessed numpy array ready for model input
        """
        try:
            df = pd.DataFrame([data])
            
            # Convert ISO_TIME to datetime
            df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
            df['HOUR'] = df['ISO_TIME'].dt.hour
            df['MONTH'] = df['ISO_TIME'].dt.month
            
            # Circular encoding for wind direction
            df['dir_sin'] = np.sin(np.deg2rad(df['STORM_DIR']))
            df['dir_cos'] = np.cos(np.deg2rad(df['STORM_DIR']))
            
            # Interaction terms
            df['lat_lon_interaction'] = df['LAT'] * df['LON']
            df['speed_lat_interaction'] = df['STORM_SPEED'] * df['LAT']
            df['speed_lon_interaction'] = df['STORM_SPEED'] * df['LON']
            
            if task == 'path':
                features = [
                    'LAT', 'LON', 'STORM_SPEED', 'HOUR', 'MONTH',
                    'lat_lon_interaction', 'speed_lat_interaction', 'speed_lon_interaction',
                    'dir_sin', 'dir_cos'
                ]
            elif task == 'speed_dir':
                # Add lag features for speed/direction prediction
                df['STORM_SPEED_LAG1'] = df['STORM_SPEED']
                df['LAT_LAG'] = df['LAT']
                df['LON_LAG'] = df['LON']
                df['SPEED_MA3'] = df['STORM_SPEED']
                
                features = [
                    'LAT', 'LON', 'STORM_SPEED', 'HOUR', 'MONTH', 'dir_sin', 'dir_cos',
                    'STORM_SPEED_LAG1', 'LAT_LAG', 'LON_LAG', 'SPEED_MA3',
                    'lat_lon_interaction', 'speed_lat_interaction'
                ]
            else:
                raise ValueError(f"Invalid task type: {task}. Choose 'path' or 'speed_dir'")
            
            X = df[features].values
            logger.debug(f"Preprocessed cyclone data for {task}: {X.shape}")
            return X
            
        except KeyError as e:
            logger.error(f"Missing required field: {e}")
            raise ValueError(f"Missing required field: {e}")
        except Exception as e:
            logger.error(f"Error preprocessing cyclone data: {e}")
            raise
    
    @staticmethod
    def preprocess_severity_data(data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess data for severity classification.
        
        Args:
            data: Dictionary containing cyclone features
            
        Returns:
            Preprocessed numpy array ready for model input
        """
        try:
            df = pd.DataFrame([data])
            df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
            df['HOUR'] = df['ISO_TIME'].dt.hour.fillna(0)
            df['MONTH'] = df['ISO_TIME'].dt.month.fillna(0)
            
            # Circular encoding for wind direction
            df['dir_sin'] = np.sin(np.deg2rad(df['STORM_DIR']))
            df['dir_cos'] = np.cos(np.deg2rad(df['STORM_DIR']))
            
            features = ['LAT', 'LON', 'STORM_SPEED', 'HOUR', 'MONTH', 'dir_sin', 'dir_cos']
            X = df[features].values
            
            logger.debug(f"Preprocessed severity data: {X.shape}")
            return X
            
        except KeyError as e:
            logger.error(f"Missing required field: {e}")
            raise ValueError(f"Missing required field: {e}")
        except Exception as e:
            logger.error(f"Error preprocessing severity data: {e}")
            raise
    
    @staticmethod
    def validate_coordinates(lat: float, lon: float) -> bool:
        """
        Validate coordinate ranges.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            True if coordinates are valid
        """
        return (-90 <= lat <= 90) and (-180 <= lon <= 180)
    
    @staticmethod
    def validate_magnitude(magnitude: float) -> bool:
        """
        Validate earthquake magnitude.
        
        Args:
            magnitude: Earthquake magnitude
            
        Returns:
            True if magnitude is valid
        """
        return 0 <= magnitude <= 10
    
    @staticmethod
    def validate_storm_speed(speed: float) -> bool:
        """
        Validate storm speed.
        
        Args:
            speed: Storm speed in km/h
            
        Returns:
            True if speed is valid
        """
        return 0 <= speed <= 300
