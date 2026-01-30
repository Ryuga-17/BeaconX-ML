# 🌊 BeaconX-ML: Disaster Prediction with Machine Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.1.2-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.20.0-orange.svg)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

I built this ML system to help predict and classify natural disasters like earthquakes and cyclones. It's part of a larger disaster response project called BeaconX, where I'm using machine learning to provide early warnings and help communities prepare for natural disasters.

## What This System Does

I've combined several machine learning approaches to tackle different aspects of disaster prediction:

### **🌍 Earthquake Analysis**
- **Severity Prediction**: I use an autoencoder (a type of neural network) to detect unusual earthquake patterns and predict how severe they might be
- **Anomaly Detection**: The system can spot earthquakes that don't follow normal patterns, which often indicates more dangerous events

### **🌪️ Cyclone Forecasting** 
- **Path Prediction**: I trained an LSTM (Long Short-Term Memory) network to predict where cyclones will go next
- **Speed & Direction**: XGBoost models help predict how fast and in which direction the storm will move
- **Severity Classification**: K-means clustering groups storms by their potential impact level

## Key Features

### **The ML Models I Built**
- **PyTorch Autoencoders**: For detecting unusual earthquake patterns
- **TensorFlow LSTM**: For predicting cyclone paths over time
- **XGBoost**: For predicting storm speed and direction
- **K-means Clustering**: For grouping disasters by severity

### **Data Processing**
- **Feature Engineering**: I extract meaningful features from raw earthquake and cyclone data
- **Data Validation**: Everything gets checked before hitting the models
- **Preprocessing**: Data gets cleaned and normalized for better predictions
- **Real-time Processing**: The system can handle live data streams

### **Production Features**
- **REST API**: Clean endpoints for easy integration
- **Input Validation**: Pydantic schemas ensure data quality
- **Error Handling**: Proper logging and error messages
- **Health Checks**: Monitor if the system is running properly

### **Code Quality**
- **Testing**: Full test suite to catch bugs
- **Code Standards**: Automated formatting and linting
- **Security**: Regular security scans
- **CI/CD**: Automated testing and deployment

## How It Works

### **The Data Flow**
```
Raw Data → Feature Engineering → ML Models → Predictions
    ↓              ↓                ↓           ↓
Earthquake    Extract features   Autoencoder   Severity
Cyclone       Clean & scale      LSTM         Path
Data          Validate input     XGBoost      Speed/Direction
```

### **Project Structure**
```
BeaconX-ML/
├── models/               # My ML model definitions
│   ├── autoencoder.py    # PyTorch autoencoder for earthquakes
│   └── model_loader.py   # Handles loading all models
├── utils/                # Helper functions
│   ├── data_processing.py # Feature engineering
│   └── validators.py     # Input validation
├── combined/             # Earthquake prediction API
├── cyclone/              # Cyclone prediction API
├── tests/                # Test suite
├── config.py             # Configuration settings
├── schemas.py            # Data validation schemas
└── app.py               # Main Flask application
```

## Getting Started

### What You Need
- Python 3.9 or higher
- pip (comes with Python)

### Quick Setup

1. **Get the code**
   ```bash
   git clone https://github.com/yourusername/BeaconX-ML.git
   cd BeaconX-ML
   ```

2. **Set up a virtual environment** (keeps dependencies clean)
   ```bash
   python -m venv beaconx_env
   source beaconx_env/bin/activate  # On Windows: beaconx_env\Scripts\activate
   ```

3. **Install everything**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the API**
   ```bash
   python app.py
   ```

The API will be running at `http://localhost:5000`

## API Endpoints

The API runs on `http://localhost:5000/api/v1`

### 🌍 Earthquake Prediction
**POST** `/combined/predict`

Send earthquake data and get severity prediction:

```json
{
  "magnitude": 5.5,
  "depth": 10.0,
  "latitude": 25.0,
  "longitude": 80.0
}
```

You'll get back something like:
```json
{
  "success": true,
  "data": {
    "severity": "Moderate"
  }
}
```

### 🌪️ Cyclone Path Prediction
**POST** `/cyclone/predict-path`

Predict where a cyclone will go next:

```json
{
  "ISO_TIME": "2024-01-01T00:00:00Z",
  "LAT": 25.0,
  "LON": 80.0,
  "STORM_SPEED": 50.0,
  "STORM_DIR": 180.0
}
```

Response:
```json
{
  "success": true,
  "data": {
    "Predicted_LAT": 25.1234,
    "Predicted_LON": 80.5678
  }
}
```

### ⚡ Cyclone Speed/Direction
**POST** `/combined/predict-speed`

Predict how fast and which way the storm will move:

```json
{
  "ISO_TIME": "2024-01-01T00:00:00Z",
  "LAT": 25.0,
  "LON": 80.0,
  "STORM_SPEED": 50.0,
  "STORM_DIR": 180.0
}
```

Response:
```json
{
  "success": true,
  "data": {
    "predicted_speed": [52.3],
    "predicted_direction": [185.7]
  }
}
```

### 🚨 Severity Classification
**POST** `/combined/classify-severity`

Classify how severe a disaster might be:

```json
{
  "ISO_TIME": "2024-01-01T00:00:00Z",
  "LAT": 25.0,
  "LON": 80.0,
  "STORM_SPEED": 50.0,
  "STORM_DIR": 180.0
}
```

Response:
```json
{
  "success": true,
  "data": {
    "severity": "Severe"
  }
}
```

### Health Check
**GET** `/health`

Check if the API is running:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Testing

I've included a full test suite to make sure everything works:

```bash
# Install test dependencies
pip install pytest pytest-flask

# Run all tests
pytest tests/ -v

# Check test coverage
pytest tests/ --cov=. --cov-report=html
```

## Configuration

You can customize the API with environment variables:

```bash
# API settings
export HOST=0.0.0.0
export PORT=5000
export DEBUG=false
export LOG_LEVEL=INFO
```

## The ML Models I Built

### **🌍 Earthquake Analysis**

#### **Autoencoder for Anomaly Detection**
I built a PyTorch autoencoder that learns normal earthquake patterns and flags unusual ones:
- **What it does**: Detects earthquakes that don't follow normal patterns
- **Input**: Magnitude, depth, latitude, longitude
- **Output**: Reconstruction error (higher = more anomalous)
- **Performance**: Reconstruction error (MSE) with anomaly thresholding

#### **K-means Clustering for Severity**
I use K-means to group earthquakes by severity:
- **Purpose**: Classify earthquakes as Mild, Moderate, Severe, or Catastrophic
- **How it works**: Groups similar earthquake patterns together
- **Features**: Uses the encoded features from the autoencoder
- **Evaluation**: Silhouette score + Davies–Bouldin index (unsupervised)

### **🌪️ Cyclone Prediction**

#### **LSTM for Path Prediction**
I trained a TensorFlow LSTM to predict where cyclones will go:
- **What it does**: Predicts latitude and longitude 6 hours ahead
- **Input**: Current location, speed, direction, time features
- **Performance**: MAE/RMSE + R² on the test set (see notebook output)
- **Why LSTM**: Great for time-series data like storm paths

#### **XGBoost for Speed & Direction**
I use XGBoost (gradient boosting) for predicting storm dynamics:
- **Speed Model**: Predicts how fast the storm will move
- **Direction Model**: Predicts which direction it will go
- **Features**: 13 engineered features including lag variables
- **Performance**: MAE < 1.0 km/h for speed, R² > 0.93 for direction

### **Feature Engineering**

I spent a lot of time engineering features to make the models work better:

#### **Earthquake Features**
```python
# Basic earthquake data
magnitude: float          # How strong (0-10)
depth: float             # How deep underground (0-700 km)
latitude: float          # Where on Earth (-90 to 90)
longitude: float         # Where on Earth (-180 to 180)

# Features I added
distance_to_city: float   # How close to major cities
distance_to_fault: float # How close to fault lines
```

#### **Cyclone Features**
```python
# Basic storm data
ISO_TIME: datetime       # When it happened
LAT: float              # Current position
LON: float              # Current position  
STORM_SPEED: float      # How fast (km/h)
STORM_DIR: float        # Which direction (degrees)

# Features I engineered
hour: int               # Time of day (affects storms)
month: int              # Season (affects storms)
dir_sin: float          # Direction as sine (circular data)
dir_cos: float          # Direction as cosine (circular data)
lat_lon_interaction: float # Position interactions
speed_lat_interaction: float # Speed × position
speed_lon_interaction: float # Speed × position
```

### **How Well Do They Work?**

| Model | What It Does | Performance |
|-------|-------------|-------------|
| **Earthquake Autoencoder** | Detects unusual earthquakes | Reconstruction error (MSE) + threshold |
| **Cyclone LSTM** | Predicts storm path | MAE/RMSE + R² (test set) |
| **Cyclone XGBoost Speed** | Predicts storm speed | MAE < 1.0 km/h |
| **Cyclone XGBoost Direction** | Predicts storm direction | R² > 0.93 |
| **Severity Classification (K-means)** | Groups by severity | Silhouette + Davies–Bouldin |

## Deployment

### Docker (Easy Way)
```bash
# Build the image
docker build -t beaconx-ml .

# Run it
docker run -p 5000:5000 beaconx-ml
```

### Production Server
```bash
# Install production server
pip install gunicorn

# Run with multiple workers
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## How I Built This

### **The Process**
1. **Data Collection**: I used USGS earthquake data and NOAA cyclone data
2. **Data Cleaning**: Lots of preprocessing to make the data usable
3. **Feature Engineering**: Created meaningful features from raw data
4. **Model Training**: Trained each model with cross-validation
5. **Evaluation**: Tested performance and fine-tuned parameters
6. **Deployment**: Built the API and made it production-ready

### **What I Learned**
- **Autoencoders**: Great for finding unusual patterns in data
- **LSTM Networks**: Perfect for time-series prediction like storm paths
- **XGBoost**: Excellent for tabular data and ensemble methods
- **Feature Engineering**: Often more important than the model choice
- **Circular Encoding**: How to handle directional data properly

### **Future Improvements**
- **Transformer Models**: Could be better for sequence prediction
- **Graph Neural Networks**: For modeling spatial relationships
- **Explainable AI**: Making the models more interpretable
- **Real-time Learning**: Updating models with new data

## Contributing

I'd love help improving this project! Check out the [Contributing Guidelines](CONTRIBUTING.md) for how to get involved.

### **Areas I'd Love Help With**
- **Better Models**: New architectures, hyperparameter tuning
- **Feature Engineering**: Creative new features, data augmentation
- **Performance**: Making it faster and more efficient
- **New Disaster Types**: Adding other natural disasters

## Data Sources

### **Earthquake Data**
- **USGS**: United States Geological Survey earthquake catalog
- **Citation**: USGS Earthquake Hazards Program

### **Cyclone Data**
- **NOAA**: National Oceanic and Atmospheric Administration
- **IBTrACS**: International Best Track Archive for Climate Stewardship
- **Citation**: Knapp, K.R., et al. (2010). The International Best Track Archive for Climate Stewardship (IBTrACS)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to:
- **USGS** for the earthquake data
- **NOAA** for the cyclone data
- **PyTorch Team** for the deep learning framework
- **TensorFlow Team** for the neural network tools
- **XGBoost Community** for the gradient boosting library
- **Open Source Community** for all the amazing tools



---

**Built with ❤️ for disaster preparedness**

*This system combines machine learning, real-time processing, and careful engineering to help communities prepare for natural disasters.*
