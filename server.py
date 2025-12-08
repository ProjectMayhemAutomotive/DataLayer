import os
import pickle
import json
from typing import Dict, Any
from dotenv import load_dotenv
from fastmcp import FastMCP
import logging

# Load environment variables
load_dotenv()

# ============ CONSTANTS ============
CURRENT_VERSION = 'v1'
MODEL_PATH = f'models/{CURRENT_VERSION}/bin/dev.pkl'
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

# ============ LOGGER ============
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ GLOBAL MODEL INSTANCE ============
model_trainer = None

def initialize_model():
    """Initialize model trainer from pickle file - called once at startup"""
    global model_trainer
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model_trainer = pickle.load(f)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}")
            model_trainer = None
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model_trainer = None

# ============ FASTMCP APP ============
mcp = FastMCP("malfunction-predictor")

# Initialize model on startup
@mcp.on_startup
async def startup():
    """Initialize model when server starts"""
    initialize_model()
    logger.info(f"MCP Server started - Model version: {CURRENT_VERSION}")

# ============ MCP TOOLS ============

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

# ============ MAIN ============
if __name__ == "__main__":
    initialize_model()
    mcp.run()