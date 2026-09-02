from fastapi import APIRouter, HTTPException, status
from app.services.skill_gap_service import SkillGapService

router = APIRouter(prefix="/skills", tags=["Skill Intelligence"])
skill_service = SkillGapService()

@router.get("/employee/{employee_id}")
def get_employee_skill_gaps(employee_id: int):
    """Retrieves specific skill gaps and upskilling pathways for an individual employee."""
    gap_info = skill_service.get_employee_gap(employee_id)
    if "error" in gap_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=gap_info["error"]
        )
    return gap_info

@router.get("/organization")
def get_organization_skill_gaps():
    """Retrieves all enterprise-level skill gaps."""
    return skill_service.get_organization_gaps()
