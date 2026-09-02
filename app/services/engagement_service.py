import pandas as pd
from app.utils.config import MASTER_INTELLIGENCE_CSV

class EngagementService:
    def get_summary(self) -> dict:
        if not MASTER_INTELLIGENCE_CSV.exists():
            return {
                "total_employees": 0,
                "high_risk_employees": 0,
                "average_engagement": 0.0,
                "average_attrition_prob": 0.0
            }
        df = pd.read_csv(MASTER_INTELLIGENCE_CSV)
        total = len(df)
        high_risk = int((df['RiskLevel'] == 'HIGH').sum())
        avg_eng = round(float(df['EngagementScore'].mean()), 1)
        avg_prob = round(float(df['AttritionProbability'].mean() * 100), 1)

        return {
            "total_employees": total,
            "high_risk_employees": high_risk,
            "average_engagement": avg_eng,
            "average_attrition_probability_pct": avg_prob
        }

    def get_by_department(self) -> list:
        if not MASTER_INTELLIGENCE_CSV.exists():
            return []
        df = pd.read_csv(MASTER_INTELLIGENCE_CSV)
        dept_eng = df.groupby('Department').agg(
            AverageEngagement=('EngagementScore', 'mean'),
            MedianEngagement=('EngagementScore', 'median'),
            Headcount=('EmployeeID', 'count')
        ).round(1).reset_index()
        return dept_eng.to_dict(orient='records')
