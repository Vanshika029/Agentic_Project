import json
import joblib
from pathlib import Path
from app.utils.config import MODEL_PATH, METADATA_PATH, FEATURE_NAMES_PATH
from app.utils.logger import logger

class ModelLoader:
    _instance = None
    _pipeline = None
    _metadata = None
    _feature_names = None

    @classmethod
    def get_pipeline(cls):
        if cls._pipeline is None:
            if not MODEL_PATH.exists():
                logger.error(f"Model file not found at {MODEL_PATH}")
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            logger.info(f"Loading ML pipeline from {MODEL_PATH}")
            cls._pipeline = joblib.load(MODEL_PATH)
        return cls._pipeline

    @classmethod
    def get_metadata(cls):
        if cls._metadata is None:
            if METADATA_PATH.exists():
                with open(METADATA_PATH, "r") as f:
                    cls._metadata = json.load(f)
            else:
                cls._metadata = {"version": "v1.0", "algorithm": "XGBoost"}
        return cls._metadata

    @classmethod
    def get_feature_names(cls):
        if cls._feature_names is None:
            if FEATURE_NAMES_PATH.exists():
                cls._feature_names = joblib.load(FEATURE_NAMES_PATH)
            else:
                cls._feature_names = []
        return cls._feature_names
