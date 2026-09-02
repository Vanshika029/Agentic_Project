import pandas as pd
import numpy as np
import shap
from datetime import datetime
from app.ml.model_loader import ModelLoader
from app.utils.logger import logger
from app.utils.config import PREDICTION_LOGS_CSV

class AttritionPredictor:
    def __init__(self):
        self.pipeline = ModelLoader.get_pipeline()
        self.metadata = ModelLoader.get_metadata()
        self.feature_names = ModelLoader.get_feature_names()
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        try:
            clf = self.pipeline.named_steps.get('classifier')
            if clf is not None:
                self.explainer = shap.TreeExplainer(clf)
        except Exception as e:
            logger.warning(f"Could not initialize TreeExplainer: {e}")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies consistent business feature engineering."""
        df = df.copy()
        
        # Domain features
        df['IncomePerYearAtCompany'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1.0)
        df['PromotionLagRatio'] = df['YearsSinceLastPromotion'] / (df['YearsInCurrentRole'] + 1.0)
        df['TotalSatisfactionScore'] = (
            df['JobSatisfaction'] + 
            df['EnvironmentSatisfaction'] + 
            df['RelationshipSatisfaction'] + 
            df['WorkLifeBalance']
        )
        df['ExperienceRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1.0)
        
        return df

    def predict_single(self, input_dict: dict) -> dict:
        emp_id = input_dict.get('EmployeeID', 9999)
        logger.info(f"Prediction request received for EmployeeID {emp_id}")

        df = pd.DataFrame([input_dict])
        df_feat = self.engineer_features(df)
        
        # Ensure all expected raw/engineered columns exist
        expected_features = self.metadata.get("features", [])
        for col in expected_features:
            if col not in df_feat.columns:
                df_feat[col] = np.nan

        # Drop identifiers if present
        cols_to_drop = [c for c in ['EmployeeID', 'EmployeeNumber', 'Attrition'] if c in df_feat.columns]
        X = df_feat.drop(columns=cols_to_drop)
        
        if expected_features:
            # Reorder columns to match pipeline training order
            valid_cols = [c for c in expected_features if c in X.columns]
            X = X[valid_cols]

        # Inference
        probs = self.pipeline.predict_proba(X)[0, 1]
        prob_rounded = round(float(probs), 4)

        # Risk Classification
        if prob_rounded >= 0.60:
            risk_level = "HIGH"
        elif prob_rounded >= 0.30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Local SHAP drivers
        top_drivers = self._get_local_drivers(X)

        # Log prediction to audit log
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_prediction(emp_id, prob_rounded, risk_level, timestamp_str)

        logger.info(f"Prediction completed: Emp {emp_id} | Prob: {prob_rounded} | Risk: {risk_level}")

        return {
            "EmployeeID": emp_id,
            "AttritionProbability": prob_rounded,
            "RiskLevel": risk_level,
            "TopRiskDrivers": top_drivers,
            "ModelVersion": self.metadata.get("version", "v1.0"),
            "Timestamp": timestamp_str
        }

    def _get_local_drivers(self, X: pd.DataFrame) -> list:
        if self.explainer is None:
            return ["OverTime", "TotalSatisfactionScore", "MonthlyIncome"]
        try:
            prep = self.pipeline.named_steps.get('preprocessor')
            X_trans = prep.transform(X)
            shap_vals = self.explainer.shap_values(X_trans)
            row_shaps = shap_vals[0]
            top_indices = np.argsort(row_shaps)[-3:][::-1]
            
            drivers = []
            for idx in top_indices:
                if row_shaps[idx] > 0 and idx < len(self.feature_names):
                    feat_name = self.feature_names[idx].split('__')[-1]
                    drivers.append(feat_name)
            
            if not drivers and len(top_indices) > 0 and top_indices[0] < len(self.feature_names):
                drivers.append(self.feature_names[top_indices[0]].split('__')[-1])
            
            return drivers if drivers else ["OverTime", "JobSatisfaction", "MonthlyIncome"]
        except Exception as e:
            logger.warning(f"Error computing local SHAP: {e}")
            return ["OverTime", "JobSatisfaction", "MonthlyIncome"]

    def _log_prediction(self, emp_id: int, prob: float, risk: str, ts: str):
        try:
            log_row = pd.DataFrame([{
                "Timestamp": ts,
                "EmployeeID": emp_id,
                "ModelVersion": self.metadata.get("version", "v1.0"),
                "Probability": prob,
                "RiskLevel": risk
            }])
            hdr = not PREDICTION_LOGS_CSV.exists()
            log_row.to_csv(PREDICTION_LOGS_CSV, mode="a", header=hdr, index=False)
        except Exception as e:
            logger.error(f"Failed to write prediction log: {e}")
