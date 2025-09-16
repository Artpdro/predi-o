from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import uuid
from datetime import datetime
import pickle
import joblib
from geopy.distance import geodesic
import asyncio
import math

# XGBoost imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, using mock predictions")

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Global model variable
model = None
model_features = None

# Define Models
class RouteRequest(BaseModel):
    start_address: str
    end_address: str
    
class Coordinates(BaseModel):
    lat: float
    lng: float

class RoutePoint(BaseModel):
    coordinates: Coordinates
    risk_score: float
    accident_count: int
    safety_level: str  # "low", "medium", "high"

class RouteAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_address: str
    end_address: str
    start_coords: Coordinates
    end_coords: Coordinates
    route_points: List[RoutePoint]
    overall_risk_score: float
    safety_recommendation: str
    total_distance_km: float
    estimated_accidents: int
    risk_factors: List[str]
    alternative_routes_available: bool
    created_at: datetime = Field(default_factory=datetime.utcnow)

class AccidentData(BaseModel):
    lat: float
    lng: float
    severity: int  # 0-4 scale
    date: str
    cause: str
    type: str

# Mock geocoding function (in real app, use Google Maps API or similar)
def mock_geocode(address: str) -> Tuple[float, float]:
    """Mock geocoding - returns coordinates for Brazilian cities"""
    mock_coords = {
        "são paulo": (-23.5505, -46.6333),
        "rio de janeiro": (-22.9068, -43.1729),
        "brasília": (-15.7942, -47.8822),
        "salvador": (-12.9714, -38.5014),
        "fortaleza": (-3.7172, -38.5433),
        "belo horizonte": (-19.9191, -43.9386),
        "manaus": (-3.1190, -60.0217),
        "curitiba": (-25.4284, -49.2733),
        "recife": (-8.0476, -34.8770),
        "goiânia": (-16.6799, -49.2554)
    }
    
    # Simple address matching
    address_lower = address.lower()
    for city, coords in mock_coords.items():
        if city in address_lower:
            return coords
    
    # Default to São Paulo if no match
    return (-23.5505, -46.6333)

async def load_accident_data():
    """Load and process accident data for the model"""
    try:
        # Load the accident data
        df = pd.read_csv('/app/acidentes2025_todas_causas_tipos.csv', 
                        sep=';', encoding='latin-1', nrows=50000)  # Limit for demo
        
        # Clean and process data
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Convert latitude/longitude to float
        df['latitude'] = pd.to_numeric(df['latitude'].astype(str).str.replace(',', '.'), errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'].astype(str).str.replace(',', '.'), errors='coerce')
        
        # Remove invalid coordinates
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]
        df = df[(df['latitude'].between(-35, 5)) & (df['longitude'].between(-75, -30))]
        
        # Create severity score
        df['severity_score'] = (df['mortos'] * 4 + 
                               df['feridos_graves'] * 2 + 
                               df['feridos_leves'] * 1)
        
        return df
        
    except Exception as e:
        print(f"Error loading accident data: {e}")
        return None

def train_route_safety_model(df):
    """Train XGBoost model for route safety prediction"""
    if not XGBOOST_AVAILABLE or df is None:
        print("Using mock model - XGBoost not available or no data")
        return None, None
        
    try:
        # Feature engineering
        df['hour'] = pd.to_datetime(df['horario'], format='%H:%M', errors='coerce').dt.hour
        df['day_of_week'] = df['dia_semana'].map({
            'SEGUNDA-FEIRA': 1, 'TERÇA-FEIRA': 2, 'QUARTA-FEIRA': 3,
            'QUINTA-FEIRA': 4, 'SEXTA-FEIRA': 5, 'SÁBADO': 6, 'DOMINGO': 7
        })
        
        # Weather condition encoding
        weather_map = {'Céu Claro': 0, 'Chuva': 1, 'Nublado': 2, 'Sol': 0, 'Garoa/Chuvisco': 1}
        df['weather_encoded'] = df['condicao_metereologica'].map(weather_map).fillna(0)
        
        # Road type encoding
        road_map = {'Dupla': 0, 'Simples': 1, 'Múltipla': 0}
        df['road_type_encoded'] = df['tipo_pista'].map(road_map).fillna(1)
        
        # Prepare features
        features = ['latitude', 'longitude', 'hour', 'day_of_week', 
                   'weather_encoded', 'road_type_encoded', 'br']
        
        # Remove rows with missing features
        model_df = df[features + ['severity_score']].dropna()
        
        if len(model_df) < 100:
            print("Not enough data for training")
            return None, None
            
        X = model_df[features]
        y = model_df['severity_score']
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X, y)
        print(f"Model trained successfully with {len(model_df)} samples")
        
        return model, features
        
    except Exception as e:
        print(f"Error training model: {e}")
        return None, None

def predict_route_risk(start_coords: Tuple[float, float], 
                      end_coords: Tuple[float, float],
                      model, features) -> RouteAnalysis:
    """Predict risk for a route between two points"""
    
    # Generate route points (simplified - in real app use routing API)
    route_points = []
    num_points = 10
    
    start_lat, start_lng = start_coords
    end_lat, end_lng = end_coords
    
    total_distance = geodesic(start_coords, end_coords).kilometers
    
    risk_scores = []
    accident_counts = []
    
    for i in range(num_points):
        # Interpolate coordinates along the route
        ratio = i / (num_points - 1)
        lat = start_lat + (end_lat - start_lat) * ratio
        lng = start_lng + (end_lng - start_lng) * ratio
        
        # Predict risk for this point
        if model is not None and features is not None:
            try:
                # Create feature vector for prediction
                feature_vector = [
                    lat, lng,  # latitude, longitude
                    12,  # hour (noon as default)
                    3,   # day_of_week (Wednesday as default)
                    0,   # weather_encoded (clear sky)
                    1,   # road_type_encoded (simple road)
                    101  # br (highway number)
                ]
                
                prediction = model.predict([feature_vector])[0]
                risk_score = min(max(prediction, 0), 10)  # Normalize to 0-10
                
            except Exception as e:
                print(f"Prediction error: {e}")
                risk_score = np.random.uniform(2, 7)  # Fallback
        else:
            # Mock prediction based on coordinates
            risk_score = np.random.uniform(2, 7)
        
        # Simulate accident count
        accident_count = int(risk_score * 2 + np.random.uniform(0, 3))
        
        # Determine safety level
        if risk_score < 3:
            safety_level = "low"
        elif risk_score < 6:
            safety_level = "medium"
        else:
            safety_level = "high"
            
        route_points.append(RoutePoint(
            coordinates=Coordinates(lat=lat, lng=lng),
            risk_score=risk_score,
            accident_count=accident_count,
            safety_level=safety_level
        ))
        
        risk_scores.append(risk_score)
        accident_counts.append(accident_count)
    
    # Calculate overall metrics
    overall_risk = np.mean(risk_scores)
    estimated_accidents = sum(accident_counts)
    
    # Generate safety recommendation
    if overall_risk < 3:
        safety_recommendation = "This route has low accident risk. Safe to travel with normal precautions."
        risk_factors = ["Generally safe route", "Low accident density"]
    elif overall_risk < 6:
        safety_recommendation = "This route has moderate accident risk. Drive carefully and consider avoiding peak hours."
        risk_factors = ["Moderate accident density", "Weather-dependent risk", "Traffic congestion possible"]
    else:
        safety_recommendation = "This route has high accident risk. Consider alternative routes or travel during safer hours."
        risk_factors = ["High accident density", "Dangerous road conditions", "Frequent severe accidents", "Poor visibility conditions"]
    
    return RouteAnalysis(
        start_address="",  # Will be filled by the API
        end_address="",   # Will be filled by the API
        start_coords=Coordinates(lat=start_lat, lng=start_lng),
        end_coords=Coordinates(lat=end_lat, lng=end_lng),
        route_points=route_points,
        overall_risk_score=overall_risk,
        safety_recommendation=safety_recommendation,
        total_distance_km=total_distance,
        estimated_accidents=estimated_accidents,
        risk_factors=risk_factors,
        alternative_routes_available=overall_risk > 5
    )

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global model, model_features
    print("Loading accident data and training model...")
    
    # Run in thread to avoid blocking
    def train_model():
        df = asyncio.run(load_accident_data())
        return train_route_safety_model(df)
    
    try:
        model, model_features = await asyncio.get_event_loop().run_in_executor(
            None, train_model
        )
        print("Model initialization complete")
    except Exception as e:
        print(f"Model initialization failed: {e}")
        model, model_features = None, None

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Route Safety Prediction API", "model_ready": model is not None}

@api_router.post("/analyze-route", response_model=RouteAnalysis)
async def analyze_route(request: RouteRequest):
    """Analyze route safety between two addresses"""
    try:
        # Geocode addresses
        start_coords = mock_geocode(request.start_address)
        end_coords = mock_geocode(request.end_address)
        
        # Predict route risk
        analysis = predict_route_risk(start_coords, end_coords, model, model_features)
        
        # Fill in addresses
        analysis.start_address = request.start_address
        analysis.end_address = request.end_address
        
        # Store in database
        await db.route_analyses.insert_one(analysis.dict())
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route analysis failed: {str(e)}")

@api_router.get("/accident-data")
async def get_accident_heatmap():
    """Get accident data for heatmap visualization"""
    try:
        # Load sample accident data
        df = pd.read_csv('/app/acidentes2025_todas_causas_tipos.csv', 
                        sep=';', encoding='latin-1', nrows=1000)
        
        # Clean coordinates
        df['latitude'] = pd.to_numeric(df['latitude'].astype(str).str.replace(',', '.'), errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'].astype(str).str.replace(',', '.'), errors='coerce')
        
        # Filter valid coordinates
        df = df[(df['latitude'].notna()) & (df['longitude'].notna())]
        df = df[(df['latitude'].between(-35, 5)) & (df['longitude'].between(-75, -30))]
        
        # Create severity score
        df['severity'] = (df['mortos'].fillna(0) * 4 + df['feridos_graves'].fillna(0) * 2 + df['feridos_leves'].fillna(0) * 1)
        
        accident_data = []
        for _, row in df.head(500).iterrows():  # Limit for performance
            if pd.notna(row['latitude']) and pd.notna(row['longitude']) and pd.notna(row['severity']):
                accident_data.append({
                    "lat": float(row['latitude']),
                    "lng": float(row['longitude']),
                    "severity": int(row['severity']),
                    "date": str(row['data_inversa']),
                    "cause": str(row['causa_principal'])[:50] if pd.notna(row['causa_principal']) else "Unknown",
                    "type": str(row['tipo_acidente'])[:50] if pd.notna(row['tipo_acidente']) else "Unknown"
                })
            
        return {"accidents": accident_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load accident data: {str(e)}")

@api_router.get("/route-history", response_model=List[RouteAnalysis])
async def get_route_history():
    """Get previous route analyses"""
    try:
        analyses = await db.route_analyses.find().sort("created_at", -1).limit(10).to_list(10)
        return [RouteAnalysis(**analysis) for analysis in analyses]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch route history: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()