import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from app.utils.logger import logger
from app.utils.config import PROCESSED_DATA_DIR, PREDICTIONS_DIR

class DriftDetector:
    def __init__(self):
        self.baseline_path = PROCESSED_DATA_DIR / "employee_attrition_processed.csv"
        self.prod_log_path = PREDICTIONS_DIR / "prediction_logs.csv"

    def calculate_numerical_drift(self, baseline_col: pd.Series, prod_col: pd.Series) -> dict:
        """Computes Kolmogorov-Smirnov test for distribution drift."""
        stat, p_value = stats.ks_2samp(baseline_col.dropna(), prod_col.dropna())
        drift_detected = p_value < 0.05
        return {
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": bool(drift_detected)
        }

    def check_prediction_drift(self) -> dict:
        """Evaluates drift on live predictions vs baseline expectation."""
        if not self.prod_log_path.exists():
            return {"status": "No production logs found to evaluate drift."}
        
        prod_logs = pd.read_csv(self.prod_log_path)
        if len(prod_logs) < 10:
            return {"status": f"Insufficient sample size ({len(prod_logs)} records). Need >= 10 records."}
        
        baseline_df = pd.read_csv(self.baseline_path)
        base_rate = (baseline_df['Attrition'] == 'Yes').mean()
        prod_avg_prob = prod_logs['Probability'].mean()

        report = {
            "total_production_predictions": len(prod_logs),
            "baseline_attrition_rate": round(float(base_rate), 4),
            "production_average_probability": round(float(prod_avg_prob), 4),
            "drift_magnitude": round(float(abs(prod_avg_prob - base_rate)), 4),
            "alert": "INVESTIGATION REQUIRED" if abs(prod_avg_prob - base_rate) > 0.15 else "NORMAL"
        }
        logger.info(f"Drift Analysis completed: {report}")
        return report

if __name__ == "__main__":
    detector = DriftDetector()
    res = detector.check_prediction_drift()
    print("=== Data Drift Monitoring Report ===")
    print(res)
