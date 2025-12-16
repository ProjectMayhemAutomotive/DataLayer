import os
import sys
import pickle
import json
from typing import Dict, Any
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import logging
import asyncio

# Load environment variables
load_dotenv()

# Importing ModelTrainer from models.v1
from models.v1.ModelTrainer import ModelTrainer

CURRENT_VERSION = 'v1'
MODEL_PATH = f'D:\\EY Hackathon\\Data Layer\\models\\{CURRENT_VERSION}\\bin\\{CURRENT_VERSION}.pkl'
ENV_TOKEN = os.getenv('MASTER_AGENT_TOKEN', '')
CLASSIFICATION_TARGETS = [
    'engine_failure_imminent',
    'brake_issue_imminent',
    'battery_issue_imminent'
]
REGRESSION_TARGETS = [
    'failure_year',
    'failure_month',
    'failure_day'
]
ALL_TARGETS = CLASSIFICATION_TARGETS + REGRESSION_TARGETS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_model():    
    global model_trainer
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_trainer = pickle.load(f)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model_trainer = None

mcp = FastMCP("malfunction-predictor", host="localhost", port=8000)

@mcp.tool()
def predict_engine_failure(vehicle_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict if engine failure is imminent based on vehicle data.
    
    Args:
        vehicle_data: Dictionary containing vehicle sensor and performance metrics
    
    Returns:
        Dictionary with prediction result and confidence
    """
    if model_trainer is None:
        return {'error': 'Model not initialized', 'status': 'error'}
    
    try:
        prediction = model_trainer.infer(new_data=vehicle_data, target='engine_failure_imminent')
        return {
            'target': 'engine_failure_imminent',
            'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            'status': 'success'
        }
    except Exception as e:
        logger.error(f"Error predicting engine failure: {str(e)}")
        return {'error': str(e), 'status': 'error'}

@mcp.tool()
def predict_brake_issue(vehicle_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict if brake issue is imminent based on vehicle data.
    
    Args:
        vehicle_data: Dictionary containing vehicle sensor and performance metrics
    
    Returns:
        Dictionary with prediction result and confidence
    """
    if model_trainer is None:
        return {'error': 'Model not initialized', 'status': 'error'}
    
    try:
        prediction = model_trainer.infer(new_data=vehicle_data, target='brake_issue_imminent')
        return {
            'target': 'brake_issue_imminent',
            'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            'status': 'success'
        }
    except Exception as e:
        logger.error(f"Error predicting brake issue: {str(e)}")
        return {'error': str(e), 'status': 'error'}

@mcp.tool()
def predict_battery_issue(vehicle_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict if battery issue is imminent based on vehicle data.
    
    Args:
        vehicle_data: Dictionary containing vehicle sensor and performance metrics
    
    Returns:
        Dictionary with prediction result and confidence
    """
    if model_trainer is None:
        return {'error': 'Model not initialized', 'status': 'error'}
    
    try:
        prediction = model_trainer.infer(new_data=vehicle_data, target='battery_issue_imminent')
        return {
            'target': 'battery_issue_imminent',
            'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            'status': 'success'
        }
    except Exception as e:
        logger.error(f"Error predicting battery issue: {str(e)}")
        return {'error': str(e), 'status': 'error'}

@mcp.tool()
def predict_failure_timeline(vehicle_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict the timeline of vehicle failure (year, month, day).
    
    Args:
        vehicle_data: Dictionary containing vehicle sensor and performance metrics
    
    Returns:
        Dictionary with failure timeline predictions (year, month, day)
    """
    if model_trainer is None:
        return {'error': 'Model not initialized', 'status': 'error'}
    
    try:
        results = {}
        for target in REGRESSION_TARGETS:
            prediction = model_trainer.infer(new_data=vehicle_data, target=target)
            results[target] = prediction.tolist() if hasattr(prediction, 'tolist') else prediction
        
        return {
            'target': 'failure_timeline',
            'predictions': results,
            'status': 'success'
        }
    except Exception as e:
        logger.error(f"Error predicting failure timeline: {str(e)}")
        return {'error': str(e), 'status': 'error'}

@mcp.tool()
def predict_all_malfunctions(vehicle_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run complete malfunction prediction analysis on vehicle data.
    Returns predictions for all classification and regression targets.
    
    Args:
        vehicle_data: Dictionary containing vehicle sensor and performance metrics
    
    Returns:
        Dictionary with all predictions (engine, brake, battery, and timeline)
    """
    if model_trainer is None:
        return {'error': 'Model not initialized', 'status': 'error'}
    
    results = {}
    for target in ALL_TARGETS:
        try:
            prediction = model_trainer.infer(new_data=vehicle_data, target=target)
            results[target] = {
                'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                'status': 'success'
            }
        except Exception as e:
            results[target] = {'error': str(e), 'status': 'error'}
            logger.error(f"Error inferring {target}: {str(e)}")
    
    return {
        'version': CURRENT_VERSION,
        'results': results,
        'overall_status': 'success' if all(r['status'] == 'success' for r in results.values()) else 'partial'
    }

if __name__ == "__main__":
    # Initialize model before creating MCP server
    initialize_model()
    mcp.run(transport="streamable-http")