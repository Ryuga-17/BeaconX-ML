"""
Data prep helpers for BeaconX-ML.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Small helpers that turn raw inputs into model-ready arrays.
    """
    
    @staticmethod
    def preprocess_earthquake_data(data: Dict[str, Any]) -> np.ndarray:
        """
        Turn earthquake request data into a model-ready array.

        Args:
            data: Incoming request fields

        Returns:
            Numpy array in the expected feature order
        """
        try:
            # Keep the feature order consistent with training
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
        Build features for cyclone predictions.

        Args:
            data: Incoming request fields
            task: 'path' for LSTM, 'speed_dir' for XGBoost

        Returns:
            Numpy array ready for the selected model
        """
        try:
            df = pd.DataFrame([data])
            
            # Parse time and derive time features
            df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
            df['HOUR'] = df['ISO_TIME'].dt.hour
            df['MONTH'] = df['ISO_TIME'].dt.month
            
            # Encode wind direction as sine/cosine
            df['dir_sin'] = np.sin(np.deg2rad(df['STORM_DIR']))
            df['dir_cos'] = np.cos(np.deg2rad(df['STORM_DIR']))
            
            # A few interaction features that help the models
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
                # Lightweight lag features for speed/direction
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
        Build features for cyclone severity classification.

        Args:
            data: Incoming request fields

        Returns:
            Numpy array ready for the encoder + clustering steps
        """
        try:
            df = pd.DataFrame([data])
            df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
            df['HOUR'] = df['ISO_TIME'].dt.hour.fillna(0)
            df['MONTH'] = df['ISO_TIME'].dt.month.fillna(0)
            
            # Encode wind direction as sine/cosine
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
        Quick latitude/longitude bounds check.
        """
        return (-90 <= lat <= 90) and (-180 <= lon <= 180)
    
    @staticmethod
    def validate_magnitude(magnitude: float) -> bool:
        """
        Simple magnitude bounds check.
        """
        return 0 <= magnitude <= 10
    
    @staticmethod
    def validate_storm_speed(speed: float) -> bool:
        """
        Simple storm speed bounds check.
        """
        return 0 <= speed <= 300
