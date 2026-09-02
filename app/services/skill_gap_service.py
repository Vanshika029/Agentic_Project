import pandas as pd
from app.utils.config import ORG_SKILL_GAPS_CSV, MASTER_INTELLIGENCE_CSV

class SkillGapService:
    def get_organization_gaps(self, min_count: int = 0) -> list:
        if not ORG_SKILL_GAPS_CSV.exists():
            return []
        df = pd.read_csv(ORG_SKILL_GAPS_CSV)
        if min_count > 0:
            df = df[df['MissingEmployeeCount'] >= min_count]
        return df.to_dict(orient='records')

    def get_employee_gap(self, employee_id: int) -> dict:
        if not MASTER_INTELLIGENCE_CSV.exists():
            return {"error": "Intelligence dataset unavailable"}
        df = pd.read_csv(MASTER_INTELLIGENCE_CSV)
        row = df[df['EmployeeID'] == employee_id]
        if row.empty:
            return {"error": f"Employee {employee_id} not found"}
        record = row.iloc[0].to_dict()
        return {
            "EmployeeID": int(record["EmployeeID"]),
            "JobRole": record["JobRole"],
            "Department": record["Department"],
            "SkillGaps": [s.strip() for s in str(record.get("SkillGaps", "")).split(",") if s.strip()],
            "GapCount": int(record.get("GapCount", 0)),
            "UpskillingRecommendation": record.get("UpskillingRecommendation", "")
        }
