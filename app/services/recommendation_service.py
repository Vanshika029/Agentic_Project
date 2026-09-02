import pandas as pd
from app.utils.config import MASTER_INTELLIGENCE_CSV

class RecommendationService:
    def get_all_recommendations(self, limit: int = 100, department: str = None) -> list:
        if not MASTER_INTELLIGENCE_CSV.exists():
            return []
        df = pd.read_csv(MASTER_INTELLIGENCE_CSV)
        if department:
            df = df[df['Department'].str.lower() == department.lower()]
        
        subset = df[['EmployeeID', 'Department', 'JobRole', 'RiskLevel', 'SkillGaps', 'UpskillingRecommendation']].head(limit)
        return subset.to_dict(orient='records')

    def get_employee_record(self, employee_id: int) -> dict:
        if not MASTER_INTELLIGENCE_CSV.exists():
            return None
        df = pd.read_csv(MASTER_INTELLIGENCE_CSV)
        row = df[df['EmployeeID'] == employee_id]
        if row.empty:
            return None
        rec = row.iloc[0].to_dict()
        return {
            "EmployeeID": int(rec["EmployeeID"]),
            "Department": rec["Department"],
            "JobRole": rec["JobRole"],
            "Age": int(rec["Age"]),
            "MonthlyIncome": float(rec["MonthlyIncome"]),
            "AttritionProbability": float(rec["AttritionProbability"]),
            "RiskLevel": rec["RiskLevel"],
            "EngagementScore": int(rec["EngagementScore"]),
            "PerformanceRating": int(rec["PerformanceRating"]),
            "WorkLifeBalanceScore": int(rec["WorkLifeBalanceScore"]),
            "SkillGaps": [s.strip() for s in str(rec.get("SkillGaps", "")).split(",") if s.strip()],
            "GapCount": int(rec.get("GapCount", 0)),
            "UpskillingRecommendation": rec.get("UpskillingRecommendation", ""),
            "TopRiskDrivers": [s.strip() for s in str(rec.get("TopRiskDrivers", "")).split(",") if s.strip()]
        }
