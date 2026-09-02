from fastapi import APIRouter, HTTPException, Query, status
from typing import Optional, List
from app.services.engagement_service import EngagementService
from app.services.attrition_service import AttritionService
from app.services.skill_gap_service import SkillGapService
from app.services.recommendation_service import RecommendationService

router = APIRouter(tags=["Executive Dashboard & Employees"])
eng_service = EngagementService()
attrition_service = AttritionService()
skill_service = SkillGapService()
rec_service = RecommendationService()

@router.get("/dashboard/summary")
def get_dashboard_summary():
    """Returns high-level workforce KPIs (total headcount, at-risk count, average engagement)."""
    return eng_service.get_summary()

@router.get("/dashboard/attrition-by-department")
def get_attrition_by_department():
    """Returns departmental breakdown of attrition risk levels."""
    return attrition_service.get_department_risk_summary()

@router.get("/dashboard/skill-gaps")
def get_skill_gaps(min_count: int = Query(default=0, ge=0)):
    """Returns organization-wide critical skill gaps with severity ratings."""
    return skill_service.get_organization_gaps(min_count=min_count)

@router.get("/dashboard/recommendations")
def get_recommendations(limit: int = Query(default=50, ge=1, le=500), department: Optional[str] = None):
    """Returns tailored upskilling recommendations for employees."""
    return rec_service.get_all_recommendations(limit=limit, department=department)

@router.get("/employees/{employee_id}")
def get_employee_detail(employee_id: int):
    """Returns full 360-degree workforce intelligence record for a single employee."""
    record = rec_service.get_employee_record(employee_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Employee with ID {employee_id} was not found."
        )
    return record
