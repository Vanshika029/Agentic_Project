import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = BASE_DIR / "models"
MODEL_V1_DIR = MODELS_DIR / "v1"

# Ensure runtime directories exist
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Model Artifacts
MODEL_PATH = MODEL_V1_DIR / "attrition_pipeline.joblib"
METADATA_PATH = MODEL_V1_DIR / "metadata.json"
FEATURE_NAMES_PATH = MODEL_V1_DIR / "feature_names.joblib"

# Data Artifacts
MASTER_INTELLIGENCE_CSV = PROCESSED_DATA_DIR / "employee_intelligence_master.csv"
ORG_SKILL_GAPS_CSV = PROCESSED_DATA_DIR / "organization_skill_gaps.csv"
EMPLOYEE_SKILLS_CSV = PROCESSED_DATA_DIR / "employee_skills_inventory.csv"
PREDICTION_LOGS_CSV = PREDICTIONS_DIR / "prediction_logs.csv"

# API & Server Config
API_HOST = "0.0.0.0"
API_PORT = 8000
API_V1_STR = "/api/v1"
PROJECT_NAME = "Enterprise HR AI — Workforce Intelligence Platform"
