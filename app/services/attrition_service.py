import pandas as pd
from app.ml.predictor import AttritionPredictor
from app.utils.config import MASTER_INTELLIGENCE_CSV

class AttritionService:
    def __init__(self):
        self.predictor = AttritionPredictor()

    def predict(self, employee_data: dict) -> dict:
        return self.predictor.predict_single(employee_data)

    def get_department_risk_summary(self) -> list:
        if not MASTER_INTELLIGENCE_CSV.exists():
            return []
        df = pd.read_csv(MASTER_INTELLIGENCE_CSV)
        dept_risk = df.groupby(['Department', 'RiskLevel']).size().unstack(fill_value=0).reset_index()
        for col in ['HIGH', 'MEDIUM', 'LOW']:
            if col not in dept_risk.columns:
                dept_risk[col] = 0
        
        dept_risk['Total'] = dept_risk['HIGH'] + dept_risk['MEDIUM'] + dept_risk['LOW']
        dept_risk['HighRiskPct'] = (dept_risk['HIGH'] / dept_risk['Total'] * 100).round(1)
        return dept_risk.to_dict(orient='records')
