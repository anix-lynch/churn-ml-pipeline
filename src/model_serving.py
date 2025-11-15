#!/usr/bin/env python3
"""
Churn ML Pipeline Model Serving - FastAPI service for churn predictions

🎯 PURPOSE: Serve trained churn prediction model via REST API with input validation and health monitoring
📊 FEATURES: FastAPI endpoints, Pydantic validation, prediction logging, model health checks, comprehensive error handling
🏗️ ARCHITECTURE: Production-ready API service with Docker compatibility and offline-only operations
⚡ STATUS: Deployable FastAPI application with automatic model loading and input preprocessing
"""
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from .config import Config
from .utils.logging_utils import setup_logger
from .utils.io_utils import load_json


# Pydantic models for API requests/responses
class PredictionRequest(BaseModel):
    """Request model for churn prediction"""

    # Customer demographics
    country_encoded: int = Field(..., ge=0, description="Encoded country (0-10)")
    segment_encoded: int = Field(..., ge=0, description="Encoded segment (0-3)")
    age: float = Field(..., ge=18, le=80, description="Customer age")
    gender_encoded: int = Field(..., ge=0, description="Encoded gender (0-2)")

    # Transaction features
    transaction_count: float = Field(..., ge=0, description="Number of transactions")
    total_amount: float = Field(..., ge=0, description="Total transaction amount")
    avg_amount: float = Field(..., ge=0, description="Average transaction amount")
    std_amount: float = Field(..., ge=0, description="Transaction amount standard deviation")

    # Session features
    session_count: float = Field(..., ge=0, description="Number of sessions")
    total_pages_viewed: float = Field(..., ge=0, description="Total pages viewed")
    avg_pages_per_session: float = Field(..., ge=0, description="Average pages per session")
    preferred_device_encoded: int = Field(..., ge=0, description="Encoded preferred device")

    # Support features
    ticket_count: float = Field(..., ge=0, description="Number of support tickets")
    avg_satisfaction: float = Field(..., ge=0, le=5, description="Average satisfaction score")
    common_ticket_category_encoded: Optional[int] = Field(None, ge=0, description="Encoded common ticket category")

    # RFM features
    recency_days: float = Field(..., ge=0, description="Days since last activity")
    recency_days_score: int = Field(..., ge=1, le=5, description="Recency score (1-5)")
    frequency_transactions_score: int = Field(..., ge=1, le=5, description="Frequency score (1-5)")
    monetary_total_score: int = Field(..., ge=1, le=5, description="Monetary score (1-5)")
    rfm_score: int = Field(..., ge=3, le=15, description="Combined RFM score")

    # Behavioral features
    session_engagement_rate: float = Field(..., ge=0, le=1, description="Session engagement rate")
    session_frequency_per_day: float = Field(..., ge=0, description="Sessions per day")
    support_engagement_rate: float = Field(..., ge=0, le=1, description="Support engagement rate")
    account_age_days: float = Field(..., ge=0, description="Account age in days")
    activity_span_days: float = Field(..., ge=0, description="Activity span in days")
    transaction_intensity: float = Field(..., ge=0, description="Transactions per day")
    session_intensity: float = Field(..., ge=0, description="Sessions per day")

    # Channel preferences
    preferred_channel_encoded: Optional[int] = Field(None, ge=0, description="Encoded preferred channel")

    class Config:
        schema_extra = {
            "example": {
                "country_encoded": 1,
                "segment_encoded": 2,
                "age": 35.0,
                "gender_encoded": 0,
                "transaction_count": 15.0,
                "total_amount": 1250.50,
                "avg_amount": 83.37,
                "std_amount": 45.20,
                "session_count": 45.0,
                "total_pages_viewed": 180.0,
                "avg_pages_per_session": 4.0,
                "preferred_device_encoded": 1,
                "ticket_count": 2.0,
                "avg_satisfaction": 4.0,
                "common_ticket_category_encoded": 1,
                "recency_days": 15.0,
                "recency_days_score": 5,
                "frequency_transactions_score": 4,
                "monetary_total_score": 3,
                "rfm_score": 12,
                "session_engagement_rate": 0.4,
                "session_frequency_per_day": 0.8,
                "support_engagement_rate": 0.02,
                "account_age_days": 365.0,
                "activity_span_days": 300.0,
                "transaction_intensity": 0.041,
                "session_intensity": 0.123,
                "preferred_channel_encoded": 2
            }
        }


class PredictionResponse(BaseModel):
    """Response model for churn prediction"""
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn (0-1)")
    churn_prediction: int = Field(..., ge=0, le=1, description="Binary churn prediction (0=no churn, 1=churn)")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence score")
    prediction_timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version identifier")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    feature_count: Optional[int] = Field(None, description="Number of model features")
    last_prediction: Optional[str] = Field(None, description="Last prediction timestamp")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ModelServer:
    """Churn prediction model server"""

    def __init__(self, config_path: str):
        """
        Initialize model server

        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.logger = setup_logger('model_serving')
        self.model = None
        self.feature_columns = None
        self.metrics = None
        self.start_time = datetime.now()
        self.last_prediction = None

        # Load model and metadata
        self._load_model()

    def _load_model(self):
        """Load trained model and feature metadata"""
        try:
            model_path = self.config.get('model_save_path')
            metrics_path = self.config.get('metrics_save_path')

            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

            # Load metrics
            self.metrics = load_json(metrics_path)

            # Extract feature information from metrics
            self.feature_columns = [f"feature_{i}" for i in range(self.metrics.get('n_features', 0))]

            self.logger.info(f"Loaded model with {self.metrics.get('n_features', 0)} features")
            self.logger.info(f"Model type: {self.metrics.get('model_type', 'unknown')}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make churn prediction

        Args:
            features: Feature dictionary

        Returns:
            Prediction results
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([features])

            # Make prediction
            prediction_proba = self.model.predict_proba(df)[:, 1]
            prediction_binary = (prediction_proba >= 0.5).astype(int)

            # Calculate confidence (distance from 0.5 threshold)
            confidence = np.abs(prediction_proba - 0.5) * 2

            # Update last prediction timestamp
            self.last_prediction = datetime.now().isoformat()

            result = {
                'churn_probability': float(prediction_proba[0]),
                'churn_prediction': int(prediction_binary[0]),
                'confidence_score': float(confidence[0]),
                'prediction_timestamp': self.last_prediction,
                'model_version': f"{self.metrics.get('model_type', 'unknown')}_{self.metrics.get('training_timestamp', 'unknown')[:10]}"
            }

            self.logger.info(".3f")

            return result

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )

    def get_health(self) -> Dict[str, Any]:
        """Get service health status"""
        uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            'status': 'healthy' if self.model is not None else 'unhealthy',
            'model_loaded': self.model is not None,
            'feature_count': self.metrics.get('n_features') if self.metrics else None,
            'last_prediction': self.last_prediction,
            'uptime_seconds': uptime
        }


# Global server instance
server = None

# FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="Machine learning API for predicting customer churn",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize model server on startup"""
    global server
    config_path = "configs/training.yaml"  # Default config path
    server = ModelServer(config_path)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if server is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized"
        )
    return server.get_health()


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(request: PredictionRequest):
    """Predict customer churn probability"""
    if server is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model server not available"
        )

    # Convert request to dict and predict
    features = request.dict()
    return server.predict(features)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Health check",
            "POST /predict": "Churn prediction",
            "GET /docs": "API documentation"
        }
    }


if __name__ == "__main__":
    # Run server directly (for development)
    import argparse

    parser = argparse.ArgumentParser(description="Churn Prediction API Server")
    parser.add_argument("--config", default="configs/training.yaml", help="Configuration file path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    args = parser.parse_args()

    # Override global server config
    global server
    server = ModelServer(args.config)

    uvicorn.run(app, host=args.host, port=args.port)
